"""
05_DEMO/walkthrough.py
Interactive CLI Demonstration: Approach A vs. Approach B vs. Approach C

Allows reviewers and engineers to evaluate any synthetic document page with
arbitrary or preset questions, executing all three paradigms side by side
with detailed intermediate spatial transcriptions and confidence calibration flags.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from tabulate import tabulate

# Add source directory to Python path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_DIR = REPO_ROOT / "02_SOURCE" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_text import query_baseline_text
from vlm_direct import query_vlm_direct
from vlm_grounded_crop import query_vlm_grounded_crop

PRESET_QUERIES = {
    "trap": {
        "doc": "page_3_quarterly_margins.png",
        "question": "What was the Free Cash Flow margin reported in Q2 2024?",
        "description": "Adversarial trap: tests hallucination resistance when a metric is unchartable."
    },
    "table": {
        "doc": "page_1_income_statement.png",
        "question": "How much did Nova Corp spend on Research and development (R&D) in FY2023?",
        "description": "Multi-column 2D table alignment: tests resistance to 1D column swaps."
    },
    "chart": {
        "doc": "page_3_quarterly_margins.png",
        "question": "What was the Gross Margin percentage recorded in Q3 2024?",
        "description": "Direct visual chart reading: tests visual bar height and data label extraction."
    },
    "trend": {
        "doc": "page_3_quarterly_margins.png",
        "question": "Which of the three charted margin metrics maintains the lowest percentage value in every observed quarter?",
        "description": "Comparative chart trend perception: tests multi-quarter visual reasoning."
    },
    "fineprint": {
        "doc": "page_5_fine_print_table.png",
        "question": "According to footnote [***], what systemic score bucket is the G-SIB surcharge based upon?",
        "description": "Micro-typography resolution: tests fine footnote qualifier discrimination at 300 DPI."
    },
    "scientific": {
        "doc": "page_6_dense_scientific_chart.png",
        "question": "What is the model inference throughput measured at concurrent batch size 16?",
        "description": "Dense dual-axis chart parsing: tests high-density curve reading."
    }
}


def run_demo(document: str, question: str, force_mock: bool = False):
    """
    Executes all three approaches on the target document and question.
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 85)
    print("      EduRankAI Domain 03 — Interactive Multimodal Document Intelligence Demo")
    print("=" * 85)
    print(f"Target Document : {document}")
    print(f"User Question   : \"{question}\"")
    print(f"Execution Mode  : {'Offline Calibrated Simulation' if force_mock or not os.environ.get('GEMINI_API_KEY') else 'Live Gemini API'}")
    print("-" * 85)

    print("\n[1/3] Executing Approach A (1D Sequential Text Parser Baseline)...")
    res_a = query_baseline_text(document, question, force_mock=force_mock)

    print("[2/3] Executing Approach B (Direct Zero-Shot Multimodal VLM)...")
    res_b = query_vlm_direct(document, question, force_mock=force_mock)

    print("[3/3] Executing Approach C (Spatial Patch Cropping & Grounded Structured VLM)...")
    res_c = query_vlm_grounded_crop(document, question, force_mock=force_mock)

    # Display Comparative Summary
    print("\n" + "=" * 85)
    print("                        COMPARATIVE EXTRACTION SUMMARY")
    print("=" * 85)

    comp_table = [
        ["Attribute", "Approach A (1D Text)", "Approach B (Direct VLM)", "Approach C (Grounded Patch)"],
        ["Extracted Answer", str(res_a.get("answer", "N/A")), str(res_b.get("answer", "N/A")), str(res_c.get("answer", "N/A"))],
        ["Unit / Scale", "N/A", "N/A", str(res_c.get("unit", "None"))],
        ["Spatial Grounding", "None (Ungrounded 1D)", str(res_b.get("spatial_location", "N/A"))[:35] + "...", f"Native Crop BBox: {res_c.get('cropped_bbox', [])}"],
        ["Confidence Score", "N/A", "Uncalibrated", f"{res_c.get('confidence_score', 1.0):.2f}"],
        ["Failure / Trap Flags", "None", "Vulnerable to Trap", "; ".join(res_c.get("failure_flags", [])) or "None (Clean)"],
        ["Latency (seconds)", f"{res_a.get('latency_sec', 0.0):.3f}s", f"{res_b.get('latency_sec', 0.0):.3f}s", f"{res_c.get('latency_sec', 0.0):.3f}s"],
    ]
    print(tabulate(comp_table, headers="firstrow", tablefmt="grid"))

    # Display Stage 1 Intermediate Markdown Reconstruction
    print("\n" + "=" * 85)
    print("     APPROACH C: STAGE 1 INTERMEDIATE SPATIAL RECONSTRUCTION (HIGH-RES CROP)")
    print("=" * 85)
    recon = res_c.get("intermediate_spatial_reconstruction", "N/A")
    print(recon)

    # Detailed Reasoning
    print("\n" + "=" * 85)
    print("                           STEP-BY-STEP REASONING")
    print("=" * 85)
    print(f"[Approach A Reasoning]:\n  {res_a.get('reasoning', 'N/A')}\n")
    print(f"[Approach B Reasoning]:\n  {res_b.get('reasoning', 'N/A')}\n")
    print(f"[Approach C Reasoning]:\n  {res_c.get('reasoning', 'N/A')}")
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive Multimodal Document QA CLI Demo")
    parser.add_argument("--document", "-d", type=str, default=None, help="Document filename (e.g. page_3_quarterly_margins.png)")
    parser.add_argument("--question", "-q", type=str, default=None, help="Question string")
    parser.add_argument("--preset", "-p", choices=list(PRESET_QUERIES.keys()), default="trap",
                        help=f"Select preset benchmark scenario: {list(PRESET_QUERIES.keys())} (default: trap)")
    parser.add_argument("--mock", action="store_true", help="Force offline deterministic simulation")
    args = parser.parse_args()

    if args.document is not None and args.question is not None:
        doc = args.document
        q = args.question
    elif args.preset in PRESET_QUERIES:
        preset_info = PRESET_QUERIES[args.preset]
        doc = preset_info["doc"]
        q = preset_info["question"]
        print(f"\n[INFO] Loaded Preset '{args.preset}': {preset_info['description']}")
    else:
        doc = "page_3_quarterly_margins.png"
        q = "What was the Free Cash Flow margin reported in Q2 2024?"

    run_demo(doc, q, force_mock=args.mock)


if __name__ == "__main__":
    main()
