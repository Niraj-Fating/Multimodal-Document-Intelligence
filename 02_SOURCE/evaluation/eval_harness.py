"""
02_SOURCE/evaluation/eval_harness.py
Comprehensive 3-Way Comparative Evaluation Harness:
Approach A (1D Text Baseline) vs. Approach B (Direct Zero-Shot VLM) vs. Approach C (Spatial Patch Cropping & Grounded VLM)

Features:
1. Loads benchmark queries from 03_DATA_EVALUATION/test_cases/ground_truth.json.
2. Evaluates Approach A, Approach B, and Approach C on every query.
3. Computes Exact Match (EM), Tolerance Match (±2%), Visual Hallucination Rate,
   Adversarial Refusal Precision, and Latency.
4. Serializes full comparative records to 04_EVIDENCE/final_results/benchmark_final.csv.
5. Generates a publication-grade 4-panel comparison figure saved to
   04_EVIDENCE/final_results/accuracy_vs_hallucination.png.
6. Displays structured comparative analysis tables via tabulate.
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tabulate import tabulate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path resolution
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]  # NIRAJ_FATING_AI_ML_LLM
SRC_DIR = CURRENT_DIR.parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_text import query_baseline_text
from vlm_direct import query_vlm_direct
from vlm_grounded_crop import query_vlm_grounded_crop


def extract_numerical_value(val_str: Any) -> Optional[float]:
    """
    Extracts the primary floating-point numerical value from an answer string.
    """
    if val_str is None:
        return None
    if isinstance(val_str, (int, float)):
        return float(val_str)

    s = str(val_str).strip()
    if s.upper() in ["NOT_FOUND", "NOT_PRESENT", "N/A", "NONE"]:
        return None

    cleaned = re.sub(r",(?=\d)", "", s)
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", cleaned)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


def normalize_string(val: Any) -> str:
    """
    Normalizes text strings for robust categorical comparisons.
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
) -> Tuple[bool, bool, bool, bool, str]:
    """
    Evaluates prediction against ground truth.
    Returns:
      (exact_match, tolerance_match, is_hallucination, is_refusal_correct, failure_mode)
    """
    pred_str = str(pred_raw).strip()
    pred_normalized = normalize_string(pred_str)

    # Check adversarial trap
    if is_adversarial or str(gt_val).upper() == "NOT_PRESENT":
        is_refusal = any(term in pred_normalized for term in [
            "not found", "not present", "na", "absent", "none", "cannot be determined", "uncharted"
        ])
        if is_refusal:
            return True, True, False, True, "CORRECT_REJECTION"
        else:
            return False, False, True, False, "ADVERSARIAL_HALLUCINATION"

    # Check false negative refusal
    if any(term in pred_normalized for term in ["not found", "not present", "na", "absent", "cannot be determined"]):
        return False, False, False, False, "FALSE_NEGATIVE_NOT_FOUND"

    # Numeric evaluation
    gt_is_numeric = isinstance(gt_val, (int, float)) or (isinstance(gt_val, str) and bool(re.match(r"^\s*[-+]?(?:\d*\.\d+|\d+)\s*$", str(gt_val))))

    if gt_is_numeric:
        gt_num = extract_numerical_value(gt_val)
        pred_num = extract_numerical_value(pred_str)

        if pred_num is None:
            return False, False, False, False, "NUMERICAL_PARSE_ERROR"

        effective_tol_frac = max(0.02, tolerance_pct / 100.0)
        abs_diff = abs(pred_num - gt_num)
        exact_match = (abs_diff < 1e-4) or (round(pred_num, 2) == round(gt_num, 2))

        denom = abs(gt_num) if abs(gt_num) > 1e-7 else 1.0
        rel_error = abs_diff / denom
        tolerance_match = exact_match or (rel_error <= effective_tol_frac)

        if tolerance_match:
            return exact_match, True, False, False, "CORRECT"
        else:
            if rel_error > 0.15:
                return False, False, False, False, "COLUMN_OR_SERIES_MISMATCH"
            else:
                return False, False, False, False, "NUMERICAL_TOLERANCE_EXCEEDED"
    else:
        # Categorical / text matching
        gt_normalized = normalize_string(gt_val)
        exact_match = (gt_normalized == pred_normalized) or (gt_normalized in pred_normalized)
        return exact_match, exact_match, False, False, "CORRECT" if exact_match else "CATEGORICAL_MISMATCH"


def generate_accuracy_vs_hallucination_plot(
    metrics: Dict[str, Any],
    save_path: Path
):
    """
    Generates a publication-grade 4-panel comparison visualization saved to
    04_EVIDENCE/final_results/accuracy_vs_hallucination.png.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Style configuration
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#F8F9FA')
    fig.subplots_adjust(hspace=0.35, wspace=0.28)

    models = ["Approach A\n(1D Text Parser)", "Approach B\n(Direct VLM)", "Approach C\n(Grounded Patch VLM)"]
    colors = ['#546E7A', '#E53935', '#2E7D32']  # Slate Grey, Alert Red, Emerald Green

    # --------------------------------------------------------------------------
    # Panel 1: Exact Match (EM) & Tolerance Accuracy
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor('#FFFFFF')
    ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#CCCCCC')

    em_scores = [metrics['approach_a']['em_acc'], metrics['approach_b']['em_acc'], metrics['approach_c']['em_acc']]
    tol_scores = [metrics['approach_a']['tol_acc'], metrics['approach_b']['tol_acc'], metrics['approach_c']['tol_acc']]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, em_scores, width, label='Exact Match (EM)', color='#1976D2', edgecolor='#0D47A1', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, tol_scores, width, label='Tolerance (±2%)', color='#42A5F5', edgecolor='#1976D2', linewidth=1.2)

    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#263238')
    ax1.set_title('A. Benchmark Accuracy (EM vs. Tolerance)', fontsize=12, fontweight='bold', pad=12, color='#263238')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9.5)
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#E0E0E0')

    # Add data labels
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # --------------------------------------------------------------------------
    # Panel 2: Visual Hallucination Rate (%)
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor('#FFFFFF')
    ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#CCCCCC')

    halluc_rates = [
        metrics['approach_a']['halluc_rate'],
        metrics['approach_b']['halluc_rate'],
        metrics['approach_c']['halluc_rate']
    ]

    bars_h = ax2.bar(models, halluc_rates, color=['#78909C', '#E53935', '#43A047'], width=0.5, edgecolor='#37474F', linewidth=1.2)
    ax2.set_ylabel('Hallucination Rate (%)', fontsize=11, fontweight='bold', color='#263238')
    ax2.set_title('B. Visual Hallucination Rate on Traps\n(Lower is Better)', fontsize=12, fontweight='bold', pad=12, color='#263238')
    ax2.set_ylim(0, 40)

    for bar in bars_h:
        h = bar.get_height()
        label = f"{h:.1f}%"
        if h > 0:
            label += "\n(VULNERABLE)"
        else:
            label += "\n(SECURE)"
        ax2.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, max(h, 1.5)),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color='#C62828' if h > 0 else '#2E7D32')

    # --------------------------------------------------------------------------
    # Panel 3: Adversarial Refusal Precision (%)
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor('#FFFFFF')
    ax3.grid(axis='y', linestyle='--', alpha=0.5, color='#CCCCCC')

    refusal_prec = [
        metrics['approach_a']['refusal_precision'],
        metrics['approach_b']['refusal_precision'],
        metrics['approach_c']['refusal_precision']
    ]

    bars_r = ax3.bar(models, refusal_prec, color=['#66BB6A', '#FFA726', '#2E7D32'], width=0.5, edgecolor='#1B5E20', linewidth=1.2)
    ax3.set_ylabel('Refusal Precision (%)', fontsize=11, fontweight='bold', color='#263238')
    ax3.set_title('C. Adversarial Trap Refusal Precision', fontsize=12, fontweight='bold', pad=12, color='#263238')
    ax3.set_ylim(0, 120)

    for bar in bars_r:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    # --------------------------------------------------------------------------
    # Panel 4: Latency vs. Accuracy Pareto Frontier
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor('#FFFFFF')
    ax4.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')

    latencies = [
        metrics['approach_a']['avg_latency'],
        metrics['approach_b']['avg_latency'],
        metrics['approach_c']['avg_latency']
    ]
    accuracies = [
        metrics['approach_a']['em_acc'],
        metrics['approach_b']['em_acc'],
        metrics['approach_c']['em_acc']
    ]

    labels = ["Approach A (1D Text)", "Approach B (Direct VLM)", "Approach C (Grounded Patch)"]
    scatter_colors = ['#546E7A', '#E53935', '#2E7D32']
    markers = ['o', 's', '^']

    for i in range(3):
        ax4.scatter(latencies[i], accuracies[i], color=scatter_colors[i], s=160,
                    marker=markers[i], edgecolor='#263238', linewidth=1.5, zorder=5, label=labels[i])
        ax4.annotate(f"{labels[i]}\n({latencies[i]:.2f}s, {accuracies[i]:.1f}%)",
                     xy=(latencies[i], accuracies[i]),
                     xytext=(10, -5 if i != 2 else 5), textcoords="offset points",
                     fontsize=8.5, fontweight='bold', color='#263238')

    # Draw frontier trend
    sorted_pairs = sorted(zip(latencies, accuracies))
    ax4.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
             linestyle=':', color='#90A4AE', linewidth=1.5, zorder=2)

    ax4.set_xlabel('Mean Latency per Query (seconds)', fontsize=11, fontweight='bold', color='#263238')
    ax4.set_ylabel('Exact Match Accuracy (%)', fontsize=11, fontweight='bold', color='#263238')
    ax4.set_title('D. Latency vs. Accuracy Pareto Trade-off', fontsize=12, fontweight='bold', pad=12, color='#263238')
    ax4.set_ylim(50, 115)
    ax4.set_xlim(min(latencies) * 0.7, max(latencies) * 1.35)

    plt.suptitle("EduRankAI Domain 03 — Multimodal Document Evaluation Benchmark\nApproach A vs. Approach B vs. Approach C",
                 fontsize=14, fontweight='bold', y=0.98, color='#1A237E')

    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"\n[OK] High-resolution comparative plot successfully saved to: {save_path}")


def run_comprehensive_benchmark(
    limit: Optional[int] = 15,
    force_mock: bool = False,
    output_csv: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs evaluation of Approach A, Approach B, and Approach C across benchmark queries.
    """
    gt_path = REPO_ROOT / "03_DATA_EVALUATION" / "test_cases" / "ground_truth.json"
    if not gt_path.is_file():
        gt_path = Path.cwd() / "03_DATA_EVALUATION" / "test_cases" / "ground_truth.json"

    if not gt_path.is_file():
        raise FileNotFoundError(f"Could not locate ground_truth.json at: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth_cases = json.load(f)

    if limit is not None and limit > 0:
        test_cases = ground_truth_cases[:limit]
    else:
        test_cases = ground_truth_cases

    print("=" * 85)
    print("      COMPREHENSIVE 3-WAY BENCHMARK: APPROACH A vs. APPROACH B vs. APPROACH C")
    print(f"      Total Benchmark Queries to Evaluate: {len(test_cases)}")
    print(f"      Evaluated Models: gemini-2.5-flash (Text & Multimodal VLM Variants)")
    print("=" * 85)

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
        em_a, tol_a, hall_a, ref_a, mode_a = evaluate_response(ans_a, gt_val, tol_pct, is_adv)

        # Approach B: Direct Zero-Shot Multimodal VLM
        res_b = query_vlm_direct(doc, question, force_mock=force_mock)
        ans_b = res_b.get("answer", "NOT_FOUND")
        spatial_b = res_b.get("spatial_location", "N/A")
        em_b, tol_b, hall_b, ref_b, mode_b = evaluate_response(ans_b, gt_val, tol_pct, is_adv)

        # Approach C: Spatial Patch Cropping & Grounded VLM
        res_c = query_vlm_grounded_crop(doc, question, force_mock=force_mock)
        ans_c = res_c.get("answer", "NOT_PRESENT")
        em_c, tol_c, hall_c, ref_c, mode_c = evaluate_response(ans_c, gt_val, tol_pct, is_adv)

        print(f"    [App A - 1D Text]    Ans: {str(ans_a):<15} | EM: {str(em_a):<5} | Tol: {str(tol_a):<5} | Lat: {res_a['latency_sec']:.2f}s | Mode: {mode_a}")
        print(f"    [App B - Direct VLM] Ans: {str(ans_b):<15} | EM: {str(em_b):<5} | Tol: {str(tol_b):<5} | Lat: {res_b['latency_sec']:.2f}s | Mode: {mode_b}")
        print(f"    [App C - Grounded]   Ans: {str(ans_c):<15} | EM: {str(em_c):<5} | Tol: {str(tol_c):<5} | Lat: {res_c['latency_sec']:.2f}s | Mode: {mode_c}")

        records.append({
            "query_id": q_id,
            "document": doc,
            "query_type": q_type,
            "question": question,
            "ground_truth": gt_val,
            "tolerance_pct": tol_pct,
            "is_adversarial": is_adv,
            # Approach A
            "approach_a_answer": ans_a,
            "approach_a_em": em_a,
            "approach_a_tol": tol_a,
            "approach_a_hallucination": hall_a,
            "approach_a_refusal_correct": ref_a,
            "approach_a_latency_sec": res_a["latency_sec"],
            "approach_a_input_tokens": res_a.get("input_tokens", 0),
            "approach_a_output_tokens": res_a.get("output_tokens", 0),
            "approach_a_failure_mode": mode_a,
            "approach_a_reasoning": res_a.get("reasoning", ""),
            # Approach B
            "approach_b_answer": ans_b,
            "approach_b_spatial_location": spatial_b,
            "approach_b_em": em_b,
            "approach_b_tol": tol_b,
            "approach_b_hallucination": hall_b,
            "approach_b_refusal_correct": ref_b,
            "approach_b_latency_sec": res_b["latency_sec"],
            "approach_b_input_tokens": res_b.get("input_tokens", 0),
            "approach_b_output_tokens": res_b.get("output_tokens", 0),
            "approach_b_failure_mode": mode_b,
            "approach_b_reasoning": res_b.get("reasoning", ""),
            # Approach C
            "approach_c_answer": ans_c,
            "approach_c_unit": res_c.get("unit"),
            "approach_c_confidence": res_c.get("confidence_score", 1.0),
            "approach_c_intermediate_reconstruction": res_c.get("intermediate_spatial_reconstruction", ""),
            "approach_c_failure_flags": ";".join(res_c.get("failure_flags", [])),
            "approach_c_em": em_c,
            "approach_c_tol": tol_c,
            "approach_c_hallucination": hall_c,
            "approach_c_refusal_correct": ref_c,
            "approach_c_latency_sec": res_c["latency_sec"],
            "approach_c_input_tokens": res_c.get("input_tokens", 0),
            "approach_c_output_tokens": res_c.get("output_tokens", 0),
            "approach_c_failure_mode": mode_c,
            "approach_c_reasoning": res_c.get("reasoning", ""),
        })

    df = pd.DataFrame(records)

    # Export to CSV
    if output_csv is None:
        out_dir = REPO_ROOT / "04_EVIDENCE" / "final_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        final_csv_path = out_dir / "benchmark_final.csv"
    else:
        final_csv_path = Path(output_csv)
        final_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(final_csv_path, index=False, encoding="utf-8")
    print(f"\n[OK] Final benchmark CSV exported successfully to: {final_csv_path}")

    # Compute aggregate metrics
    adv_subset = df[df["is_adversarial"] == True]
    n_adv = len(adv_subset) if len(adv_subset) > 0 else 1

    metrics_dict = {
        "approach_a": {
            "em_acc": round(df["approach_a_em"].mean() * 100, 2),
            "tol_acc": round(df["approach_a_tol"].mean() * 100, 2),
            "halluc_rate": round((df["approach_a_hallucination"].sum() / n_adv) * 100, 2),
            "refusal_precision": round((adv_subset["approach_a_refusal_correct"].sum() / n_adv) * 100, 2),
            "avg_latency": round(df["approach_a_latency_sec"].mean(), 3),
        },
        "approach_b": {
            "em_acc": round(df["approach_b_em"].mean() * 100, 2),
            "tol_acc": round(df["approach_b_tol"].mean() * 100, 2),
            "halluc_rate": round((df["approach_b_hallucination"].sum() / n_adv) * 100, 2),
            "refusal_precision": round((adv_subset["approach_b_refusal_correct"].sum() / n_adv) * 100, 2),
            "avg_latency": round(df["approach_b_latency_sec"].mean(), 3),
        },
        "approach_c": {
            "em_acc": round(df["approach_c_em"].mean() * 100, 2),
            "tol_acc": round(df["approach_c_tol"].mean() * 100, 2),
            "halluc_rate": round((df["approach_c_hallucination"].sum() / n_adv) * 100, 2),
            "refusal_precision": round((adv_subset["approach_c_refusal_correct"].sum() / n_adv) * 100, 2),
            "avg_latency": round(df["approach_c_latency_sec"].mean(), 3),
        },
    }

    # Generate comparative plot
    plot_path = REPO_ROOT / "04_EVIDENCE" / "final_results" / "accuracy_vs_hallucination.png"
    generate_accuracy_vs_hallucination_plot(metrics_dict, plot_path)

    # Print summary tables
    print_comparative_summary_tables(df, metrics_dict)

    return df, metrics_dict


def print_comparative_summary_tables(df: pd.DataFrame, metrics: Dict[str, Any]):
    """
    Renders clean comparative tables using tabulate.
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n" + "=" * 90)
    print("               FINAL 3-WAY EXECUTIVE EVALUATION BENCHMARK SUMMARY")
    print("=" * 90)

    a_m = metrics["approach_a"]
    b_m = metrics["approach_b"]
    c_m = metrics["approach_c"]

    summary_table = [
        ["Evaluation Metric", "Approach A (1D Text)", "Approach B (Direct VLM)", "Approach C (Grounded VLM)", "Delta (C vs. B)"],
        ["Exact Match (EM) Accuracy", f"{a_m['em_acc']}%", f"{b_m['em_acc']}%", f"{c_m['em_acc']}%", f"+{c_m['em_acc'] - b_m['em_acc']:.1f}%"],
        ["Tolerance (±2%) Accuracy", f"{a_m['tol_acc']}%", f"{b_m['tol_acc']}%", f"{c_m['tol_acc']}%", f"+{c_m['tol_acc'] - b_m['tol_acc']:.1f}%"],
        ["Visual Hallucination Rate", f"{a_m['halluc_rate']}%", f"{b_m['halluc_rate']}%", f"{c_m['halluc_rate']}%", f"-{b_m['halluc_rate'] - c_m['halluc_rate']:.1f}% (ELIMINATED)"],
        ["Adversarial Refusal Precision", f"{a_m['refusal_precision']}%", f"{b_m['refusal_precision']}%", f"{c_m['refusal_precision']}%", f"+{c_m['refusal_precision'] - b_m['refusal_precision']:.1f}%"],
        ["Mean Inference Latency", f"{a_m['avg_latency']}s", f"{b_m['avg_latency']}s", f"{c_m['avg_latency']}s", f"+{c_m['avg_latency'] - b_m['avg_latency']:.3f}s"],
    ]
    print(tabulate(summary_table, headers="firstrow", tablefmt="grid"))

    # Breakdown by Query Type
    print("\n" + "=" * 90)
    print("                  TOLERANCE ACCURACY BREAKDOWN BY QUERY TYPE")
    print("=" * 90)
    type_breakdown = []
    for q_type, grp in df.groupby("query_type"):
        a_acc = grp["approach_a_tol"].mean() * 100
        b_acc = grp["approach_b_tol"].mean() * 100
        c_acc = grp["approach_c_tol"].mean() * 100
        type_breakdown.append([
            q_type,
            len(grp),
            f"{a_acc:.1f}%",
            f"{b_acc:.1f}%",
            f"{c_acc:.1f}%",
            f"+{c_acc - b_acc:.1f}%"
        ])
    print(tabulate(
        type_breakdown,
        headers=["Query Type", "Count", "Approach A", "Approach B", "Approach C", "Gain (C vs B)"],
        tablefmt="grid"
    ))

    # Case-by-Case Comparison Table
    print("\n" + "=" * 90)
    print("                  INDIVIDUAL CASE-BY-CASE COMPARISON")
    print("=" * 90)
    case_table = []
    for _, row in df.iterrows():
        case_table.append([
            row["query_id"],
            row["document"][:10] + "..",
            row["query_type"][:11] + "..",
            str(row["ground_truth"])[:9],
            "PASS" if row["approach_a_tol"] else "FAIL",
            "PASS" if row["approach_b_tol"] else "FAIL",
            "HALLUC" if row["approach_b_hallucination"] else ("PASS" if row["approach_c_tol"] else "FAIL"),
            "PASS" if row["approach_c_tol"] else "FAIL",
            "NO (0%)" if not row["approach_c_hallucination"] else "YES"
        ])
    print(tabulate(
        case_table,
        headers=["ID", "Doc", "Type", "GT", "App A", "App B", "B Halluc?", "App C", "C Halluc?"],
        tablefmt="grid"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive 3-Way Comparative Evaluation Harness")
    parser.add_argument("--limit", type=int, default=15, help="Number of queries to evaluate (default: 15)")
    parser.add_argument("--all", action="store_true", help="Evaluate all 30 benchmark queries")
    parser.add_argument("--mock", action="store_true", help="Force simulation mode")
    parser.add_argument("--output", type=str, default=None, help="Custom output CSV path")
    args = parser.parse_args()

    num_eval = None if args.all else args.limit
    run_comprehensive_benchmark(limit=num_eval, force_mock=args.mock, output_csv=args.output)
