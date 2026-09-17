"""
02_SOURCE/evaluation/check_repo_integrity.py
Repository Integrity & Submission Readiness Audit Script

Validates the complete file system, schema compliance, sample documents,
source code, evidence artifacts, demo scripts, and declarations for the
EduRankAI Domain 03 Final Submission Package.
"""

import os
import sys
import json
from pathlib import Path
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[2]  # NIRAJ_FATING_AI_ML_LLM


def run_integrity_audit():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 85)
    print("      EduRankAI Domain 03 — Final Repository Integrity & Readiness Audit")
    print(f"      Target Repository Root: {REPO_ROOT}")
    print("=" * 85)

    checks = []

    # 1. Directory Structure
    required_dirs = [
        "01_FINAL_REPORT",
        "02_SOURCE/src",
        "02_SOURCE/evaluation",
        "03_DATA_EVALUATION/data_dictionary",
        "03_DATA_EVALUATION/sample_pages",
        "03_DATA_EVALUATION/test_cases",
        "04_EVIDENCE/baseline",
        "04_EVIDENCE/failures",
        "04_EVIDENCE/final_results",
        "05_DEMO",
        "06_DECLARATION"
    ]
    for d in required_dirs:
        dir_path = REPO_ROOT / d
        exists = dir_path.is_dir()
        checks.append(("Directory", d, "EXISTS" if exists else "MISSING", exists))

    # 2. Key Files Verification
    required_files = [
        ("Config", "requirements.txt"),
        ("Config", "pyproject.toml"),
        ("Source", "02_SOURCE/src/baseline_text.py"),
        ("Source", "02_SOURCE/src/vlm_direct.py"),
        ("Source", "02_SOURCE/src/vlm_grounded_crop.py"),
        ("Source", "02_SOURCE/src/generate_test_suite.py"),
        ("Evaluation", "02_SOURCE/evaluation/eval_harness.py"),
        ("Data", "03_DATA_EVALUATION/data_dictionary/data_dictionary.json"),
        ("Data", "03_DATA_EVALUATION/test_cases/ground_truth.json"),
        ("Evidence", "04_EVIDENCE/baseline/results_ab.csv"),
        ("Evidence", "04_EVIDENCE/failures/failure_case_analysis.md"),
        ("Evidence", "04_EVIDENCE/final_results/benchmark_final.csv"),
        ("Evidence", "04_EVIDENCE/final_results/accuracy_vs_hallucination.png"),
        ("Demo", "05_DEMO/walkthrough.py"),
        ("Demo", "05_DEMO/README.md"),
        ("Declaration", "06_DECLARATION/AI_EXTERNAL_TOOLS.md"),
        ("Report", "01_FINAL_REPORT/FINAL_REPORT.md"),
    ]

    for category, rel_path in required_files:
        f_path = REPO_ROOT / rel_path
        if f_path.is_file():
            size = f_path.stat().st_size
            status = f"PRESENT ({size:,} B)"
            ok = size > 0
        else:
            status = "MISSING"
            ok = False
        checks.append((category, rel_path, status, ok))

    # 3. Sample Pages Check (6 PNGs + 6 PDFs = 12 files)
    sample_pages_dir = REPO_ROOT / "03_DATA_EVALUATION" / "sample_pages"
    for i in range(1, 7):
        # Check PNG
        png_cand = list(sample_pages_dir.glob(f"page_{i}_*.png"))
        pdf_cand = list(sample_pages_dir.glob(f"page_{i}_*.pdf"))
        
        has_png = len(png_cand) > 0 and png_cand[0].stat().st_size > 1000
        has_pdf = len(pdf_cand) > 0 and pdf_cand[0].stat().st_size > 1000
        
        checks.append(("Sample Document", f"Page {i} (300 DPI PNG)", f"OK ({png_cand[0].name})" if has_png else "MISSING", has_png))
        checks.append(("Sample Document", f"Page {i} (Vector PDF)", f"OK ({pdf_cand[0].name})" if has_pdf else "MISSING", has_pdf))

    # 4. Benchmark JSON Schema Audit
    gt_path = REPO_ROOT / "03_DATA_EVALUATION" / "test_cases" / "ground_truth.json"
    if gt_path.is_file():
        with open(gt_path, "r", encoding="utf-8") as f:
            cases = json.load(f)
        case_count_ok = len(cases) == 30
        adv_count = sum(1 for c in cases if c.get("is_adversarial"))
        checks.append(("Data Integrity", "ground_truth.json query count", f"{len(cases)} queries (Required: 30)", case_count_ok))
        checks.append(("Data Integrity", "ground_truth.json adversarial traps", f"{adv_count} traps (Expected: 6)", adv_count == 6))

    # Render Table
    table_rows = []
    all_passed = True
    for cat, item, stat, ok in checks:
        if not ok:
            all_passed = False
        table_rows.append([cat, item, stat, "PASS" if ok else "FAIL"])

    print("\n" + tabulate(table_rows, headers=["Category", "Artifact / Component", "Details", "Audit Status"], tablefmt="grid"))

    print("\n" + "=" * 85)
    if all_passed:
        print("   >>> FINAL SUBMISSION PACKAGE VERIFICATION: 100% PASS (READY FOR EVALUATION) <<<")
    else:
        print("   >>> WARNING: SOME CHECKS FAILED! INSPECT TABLE ABOVE <<<")
    print("=" * 85 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_integrity_audit()
    sys.exit(0 if success else 1)
