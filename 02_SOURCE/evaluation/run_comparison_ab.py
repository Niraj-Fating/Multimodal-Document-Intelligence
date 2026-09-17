"""
02_SOURCE/evaluation/run_comparison_ab.py
Comparative Evaluation Harness: Approach A (1D Text Baseline) vs. Approach B (Multimodal VLM)

Features:
1. Loads benchmark queries from 03_DATA_EVALUATION/test_cases/ground_truth.json.
2. Evaluates both Approach A and Approach B sequentially.
3. Computes Exact Match (EM) and Tolerance Match (±2% or GT tolerance).
4. Detects and flags visual hallucinations (fabricated numbers / ungrounded claims).
5. Categorizes specific failure modes (column misalignments, chart parsing loss, adversarial hallucinations).
6. Exports full results to 04_EVIDENCE/baseline/results_ab.csv.
7. Displays formatted comparative summary tables.
"""

import os
import re
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tabulate import tabulate

# Add source directory to Python path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]  # NIRAJ_FATING_AI_ML_LLM
SRC_DIR = CURRENT_DIR.parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_text import query_baseline_text
from vlm_direct import query_vlm_direct


def extract_numerical_value(val_str: Any) -> Optional[float]:
    """
    Extracts the primary floating-point numerical value from an answer string.
    Removes currency symbols, commas, percent signs, and conversational text.
    """
    if val_str is None:
        return None
    if isinstance(val_str, (int, float)):
        return float(val_str)
        
    s = str(val_str).strip()
    if s.upper() in ["NOT_FOUND", "NOT_PRESENT", "N/A", "NONE"]:
        return None
        
    # Remove commas in numbers (e.g. 1,320.0 -> 1320.0)
    cleaned = re.sub(r",(?=\d)", "", s)
    
    # Extract floating point numbers (including signs and decimals)
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", cleaned)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


def normalize_string(val: Any) -> str:
    """
    Normalizes text strings for categorical comparisons.
    """
    if val is None:
        return ""
    s = str(val).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def evaluate_response(
    pred_raw: Any,
    gt_val: Any,
    tolerance_pct: float,
    is_adversarial: bool
) -> Tuple[bool, bool, bool, str]:
    """
    Evaluates model prediction against ground truth.
    Returns:
      (exact_match, tolerance_match, is_hallucination, failure_mode)
    """
    pred_str = str(pred_raw).strip()
    pred_normalized = normalize_string(pred_str)
    
    # Check if ground truth is NOT_PRESENT (adversarial trap)
    if is_adversarial or str(gt_val).upper() == "NOT_PRESENT":
        is_not_found = any(term in pred_normalized for term in ["not found", "not present", "na", "absent", "none", "cannot be determined"])
        if is_not_found:
            return True, True, False, "CORRECT_REJECTION"
        else:
            # Model fabricated a number or fact on an adversarial query
            return False, False, True, "ADVERSARIAL_HALLUCINATION"

    # Check if model returned NOT_FOUND when information was actually present
    if any(term in pred_normalized for term in ["not found", "not present", "na", "absent", "cannot be determined"]):
        return False, False, False, "FALSE_NEGATIVE_NOT_FOUND"

    # Check if ground truth is strictly numeric
    gt_is_numeric = isinstance(gt_val, (int, float)) or (isinstance(gt_val, str) and bool(re.match(r"^\s*[-+]?(?:\d*\.\d+|\d+)\s*$", str(gt_val))))

    if gt_is_numeric:
        gt_num = extract_numerical_value(gt_val)
        pred_num = extract_numerical_value(pred_str)

        if pred_num is None:
            return False, False, False, "NUMERICAL_PARSE_ERROR"

        # Tolerance threshold: max of query tolerance or 2.0%
        effective_tol_frac = max(0.02, tolerance_pct / 100.0)
        abs_diff = abs(pred_num - gt_num)
        
        # Exact match (tolerance < 1e-4 or exact float)
        exact_match = (abs_diff < 1e-4) or (round(pred_num, 2) == round(gt_num, 2))
        
        # Relative error
        denom = abs(gt_num) if abs(gt_num) > 1e-7 else 1.0
        rel_error = abs_diff / denom
        tolerance_match = exact_match or (rel_error <= effective_tol_frac)
        
        if tolerance_match:
            return exact_match, True, False, "CORRECT"
        else:
            # Check if this is a severe deviation (fabricated / wrong column)
            if rel_error > 0.15:
                return False, False, False, "COLUMN_OR_SERIES_MISMATCH"
            else:
                return False, False, False, "NUMERICAL_TOLERANCE_EXCEEDED"
                
    else:
        # Categorical / string ground truth (e.g. "Net Margin", "Q4-24", "EMEA", "bucket 2", "increases")
        gt_normalized = normalize_string(gt_val)
        exact_match = (gt_normalized == pred_normalized) or (gt_normalized in pred_normalized)
        return exact_match, exact_match, False, "CORRECT" if exact_match else "CATEGORICAL_MISMATCH"


def run_benchmark(
    limit: Optional[int] = 15,
    force_mock: bool = False,
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the comparative benchmark across test queries.
    """
    gt_path = REPO_ROOT / "03_DATA_EVALUATION" / "test_cases" / "ground_truth.json"
    if not gt_path.is_file():
        # Try alternate path
        gt_path = Path.cwd() / "03_DATA_EVALUATION" / "test_cases" / "ground_truth.json"
        
    if not gt_path.is_file():
        raise FileNotFoundError(f"Could not find ground_truth.json at: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth_cases = json.load(f)

    if limit is not None and limit > 0:
        test_cases = ground_truth_cases[:limit]
    else:
        test_cases = ground_truth_cases

    print("=" * 80)
    print(f"  RUNNING COMPARATIVE BENCHMARK: APPROACH A vs. APPROACH B")
    print(f"  Total Test Cases to Evaluate: {len(test_cases)}")
    print(f"  Model Under Test: gemini-2.5-flash (Google GenAI)")
    print("=" * 80)

    records = []

    for idx, tc in enumerate(test_cases, 1):
        q_id = tc["id"]
        doc = tc["document"]
        question = tc["question"]
        q_type = tc["type"]
        gt_val = tc["ground_truth_value"]
        tol_pct = tc.get("tolerance_pct", 0.0)
        is_adv = tc.get("is_adversarial", False)

        print(f"\n[{idx:02d}/{len(test_cases):02d}] Evaluating {q_id} ({q_type}) on {doc}...")
        print(f"     Q: {question}")
        print(f"    GT: {gt_val} (Tol: ±{tol_pct}%, Adversarial: {is_adv})")

        # Approach A: 1D Text Parser Baseline
        res_a = query_baseline_text(doc, question, force_mock=force_mock)
        ans_a = res_a.get("answer", "NOT_FOUND")
        em_a, tol_a, hall_a, mode_a = evaluate_response(ans_a, gt_val, tol_pct, is_adv)

        # Approach B: Direct Zero-Shot Multimodal VLM
        res_b = query_vlm_direct(doc, question, force_mock=force_mock)
        ans_b = res_b.get("answer", "NOT_FOUND")
        spatial_b = res_b.get("spatial_location", "N/A")
        em_b, tol_b, hall_b, mode_b = evaluate_response(ans_b, gt_val, tol_pct, is_adv)

        print(f"    [Approach A] Ans: {str(ans_a):<15} | EM: {str(em_a):<5} | Tol: {str(tol_a):<5} | Latency: {res_a['latency_sec']:.2f}s | Mode: {mode_a}")
        print(f"    [Approach B] Ans: {str(ans_b):<15} | EM: {str(em_b):<5} | Tol: {str(tol_b):<5} | Latency: {res_b['latency_sec']:.2f}s | Mode: {mode_b}")
        if spatial_b != "N/A":
            print(f"                 Spatial: {spatial_b}")

        records.append({
            "query_id": q_id,
            "document": doc,
            "query_type": q_type,
            "question": question,
            "ground_truth": gt_val,
            "tolerance_pct": tol_pct,
            "is_adversarial": is_adv,
            # Approach A Metrics
            "approach_a_answer": ans_a,
            "approach_a_em": em_a,
            "approach_a_tol": tol_a,
            "approach_a_hallucination": hall_a,
            "approach_a_latency_sec": res_a["latency_sec"],
            "approach_a_input_tokens": res_a.get("input_tokens", 0),
            "approach_a_output_tokens": res_a.get("output_tokens", 0),
            "approach_a_failure_mode": mode_a,
            "approach_a_reasoning": res_a.get("reasoning", ""),
            # Approach B Metrics
            "approach_b_answer": ans_b,
            "approach_b_spatial_location": spatial_b,
            "approach_b_em": em_b,
            "approach_b_tol": tol_b,
            "approach_b_hallucination": hall_b,
            "approach_b_latency_sec": res_b["latency_sec"],
            "approach_b_input_tokens": res_b.get("input_tokens", 0),
            "approach_b_output_tokens": res_b.get("output_tokens", 0),
            "approach_b_failure_mode": mode_b,
            "approach_b_reasoning": res_b.get("reasoning", ""),
        })

    df = pd.DataFrame(records)

    # Save to CSV
    if output_path is None:
        out_dir = REPO_ROOT / "04_EVIDENCE" / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        final_csv_path = out_dir / "results_ab.csv"
    else:
        final_csv_path = Path(output_path)
        final_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(final_csv_path, index=False, encoding="utf-8")
    print(f"\n[OK] Intermediate benchmark results successfully saved to: {final_csv_path}")

    # Compute summary statistics
    n_total = len(df)
    summary_stats = {
        "total_queries": n_total,
        "approach_a_em_acc": round(df["approach_a_em"].mean() * 100, 2),
        "approach_a_tol_acc": round(df["approach_a_tol"].mean() * 100, 2),
        "approach_a_hallucinations": int(df["approach_a_hallucination"].sum()),
        "approach_a_avg_latency": round(df["approach_a_latency_sec"].mean(), 3),
        "approach_b_em_acc": round(df["approach_b_em"].mean() * 100, 2),
        "approach_b_tol_acc": round(df["approach_b_tol"].mean() * 100, 2),
        "approach_b_hallucinations": int(df["approach_b_hallucination"].sum()),
        "approach_b_avg_latency": round(df["approach_b_latency_sec"].mean(), 3),
    }

    # Print summary tables
    print_benchmark_tables(df, summary_stats)

    return df, summary_stats


def print_benchmark_tables(df: pd.DataFrame, stats: Dict[str, Any]):
    """
    Renders clean comparative tables using tabulate.
    """
    print("\n" + "=" * 80)
    print("                    EXECUTIVE BENCHMARK SUMMARY")
    print("=" * 80)
    
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    exec_table = [
        ["Metric", "Approach A (1D Text Parser)", "Approach B (Multimodal VLM)", "Diff (Advantage)"],
        ["Exact Match (EM) Accuracy", f"{stats['approach_a_em_acc']}%", f"{stats['approach_b_em_acc']}%", f"+{stats['approach_b_em_acc'] - stats['approach_a_em_acc']:.1f}%"],
        ["Tolerance (±2%) Accuracy", f"{stats['approach_a_tol_acc']}%", f"{stats['approach_b_tol_acc']}%", f"+{stats['approach_b_tol_acc'] - stats['approach_a_tol_acc']:.1f}%"],
        ["Visual Hallucinations Flagged", f"{stats['approach_a_hallucinations']} queries", f"{stats['approach_b_hallucinations']} queries", f"{stats['approach_b_hallucinations'] - stats['approach_a_hallucinations']:+d}"],
        ["Average Latency (s)", f"{stats['approach_a_avg_latency']}s", f"{stats['approach_b_avg_latency']}s", f"+{stats['approach_b_avg_latency'] - stats['approach_a_avg_latency']:.3f}s"],
    ]
    print(tabulate(exec_table, headers="firstrow", tablefmt="grid"))

    # Breakdown by Query Type
    print("\n" + "=" * 80)
    print("                PERFORMANCE BREAKDOWN BY QUERY TYPE")
    print("=" * 80)
    
    type_breakdown = []
    for q_type, group in df.groupby("query_type"):
        a_em = group["approach_a_em"].mean() * 100
        a_tol = group["approach_a_tol"].mean() * 100
        b_em = group["approach_b_em"].mean() * 100
        b_tol = group["approach_b_tol"].mean() * 100
        type_breakdown.append([
            q_type,
            len(group),
            f"{a_em:.1f}%",
            f"{a_tol:.1f}%",
            f"{b_em:.1f}%",
            f"{b_tol:.1f}%"
        ])
    print(tabulate(
        type_breakdown,
        headers=["Query Type", "Count", "A (EM)", "A (Tol)", "B (EM)", "B (Tol)"],
        tablefmt="grid"
    ))

    # Query Level Comparison Table
    print("\n" + "=" * 80)
    print("                    CASE-BY-CASE COMPARATIVE TABLE")
    print("=" * 80)
    case_table = []
    for _, row in df.iterrows():
        case_table.append([
            row["query_id"],
            row["document"][:12] + "...",
            row["query_type"],
            str(row["ground_truth"])[:10],
            str(row["approach_a_answer"])[:10],
            "PASS" if row["approach_a_tol"] else "FAIL",
            str(row["approach_b_answer"])[:10],
            "PASS" if row["approach_b_tol"] else "FAIL",
            "YES" if row["approach_b_hallucination"] else "NO"
        ])
    print(tabulate(
        case_table,
        headers=["ID", "Doc", "Type", "GT", "Ans A", "Res A", "Ans B", "Res B", "B Halluc?"],
        tablefmt="grid"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparative Evaluation Harness: Approach A vs. Approach B")
    parser.add_argument("--limit", type=int, default=15, help="Number of queries to evaluate (default: 15)")
    parser.add_argument("--all", action="store_true", help="Evaluate all 30 queries")
    parser.add_argument("--mock", action="store_true", help="Force simulation mode")
    parser.add_argument("--output", type=str, default=None, help="Custom CSV output path")
    args = parser.parse_args()

    num_queries = None if args.all else args.limit
    run_benchmark(limit=num_queries, force_mock=args.mock, output_path=args.output)
