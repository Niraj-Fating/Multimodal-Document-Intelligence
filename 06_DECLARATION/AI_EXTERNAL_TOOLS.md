# Declaration of AI, Foundation Models & External Tools

**Project Title**: EduRankAI Domain 03 — Multimodal Document Evaluation Benchmark  
**Author / Candidate**: Niraj Fating  
**Track**: AI / ML / LLM Systems  
**Evaluation Date**: September 2026  
**Document Reference**: `06_DECLARATION/AI_EXTERNAL_TOOLS.md`  

---

## 1. Compliance Statement & Ownership Declaration

In strict compliance with **Section 10 (Academic & Professional Integrity Guidelines)** of the EduRankAI Technical Assessment:

> I, **Niraj Fating**, hereby declare that all code, benchmark datasets, evaluation methodologies, architectural designs, failure mode analyses, and technical documentation contained in this repository represent my original design, implementation, and analytical effort.
>
> Where artificial intelligence models, foundation APIs, and software libraries were utilized to assist in code generation, benchmark generation, or natural language articulation, they served strictly as accelerated productivity tools under continuous human oversight. I maintain 100% intellectual ownership, technical comprehension, and accountability for the architectural validity, correctness, and empirical findings presented in this submission.

---

## 2. Foundation Models & AI APIs Declared

| Model / API Name | Provider | Version / Checkpoint | Primary Role & Invocation Context | Governance & Human Oversight |
| :--- | :--- | :--- | :--- | :--- |
| **Gemini 2.5 Flash** | Google Cloud / Google AI | `gemini-2.5-flash` | Multimodal Vision-Language inference across Approach A (1D sequential text baseline), Approach B (full-page 300 DPI zero-shot vision), and Approach C (grounded high-res patch cropping). | All prompts, system instructions, temperature constraints ($T=0.0$), and output JSON schemas were manually engineered and verified. |
| **Antigravity IDE Assistant** | Google DeepMind | Gemini 3.8 Flash (High) | Pair programming, test script scaffolding, markdown structuring, and terminal execution assistance. | Every generated script, patch coordinate, and metric computation was manually reviewed, executed, and validated by the candidate. |

---

## 3. Core Software Libraries & Computational Frameworks

All software packages utilized across data synthesis, document rendering, computer vision, and statistical benchmarking are open-source and declared below:

```toml
# Core Framework Declarations (from pyproject.toml & requirements.txt)
pypdf = ">=5.1.0"        # PDF object tree reading and sequential text extraction
pymupdf = ">=1.25.0"     # High-performance fitz C-bindings for PDF parsing and vector rasterization
pillow = ">=11.0.0"      # High-resolution PIL.Image cropping, patch extraction, and pixel manipulation
google-genai = ">=1.0.0" # Official Google GenAI SDK for Gemini API multimodal messaging
pandas = ">=2.2.0"       # Dataframe manipulation, metric aggregation, and CSV serialization
matplotlib = ">=3.9.0"   # Synthetic document vector rendering, font rasterization, and 300 DPI visualization
tabulate = ">=0.9.0"     # Terminal table rendering and structured ASCII scorecard formatting
```

---

## 4. Scope of AI Assistance & Verification Log

### A. Synthetic Benchmark Document Generation (`generate_test_suite.py`)
- **Assisted Scaffolding**: Script structure for `matplotlib` layout coordinates (inches, DPI, font sizing) and schema definition for `ground_truth.json`.
- **Developer Verification**:
  - Manually audited financial logic on Page 1 (Gross Profit = Total Revenue - Cost of Revenue; Net Income = Income before taxes - Provision for taxes).
  - Manually audited footnote disclosures on Page 2 and Page 5 to ensure unambiguous grounding.
  - Manually calculated and confirmed all 30 ground truth numerical values, tolerance thresholds ($\pm 0.0\%$ to $\pm 2.0\%$), and adversarial trap designations.

### B. Implementation of Approaches A, B, and C
- **Approach A (`baseline_text.py`)**: Designed to model the structural limitations of standard 1D RAG pipelines. Developer verified text serialization order and column flattening mechanisms.
- **Approach B (`vlm_direct.py`)**: Implemented zero-shot full-page multimodal ingestion. Developer analyzed failure cases on adversarial queries.
- **Approach C (`vlm_grounded_crop.py`)**: Conceptualized and authored the **two-stage grounded spatial patch cropping pipeline**. Developer engineered dynamic bounding box heuristics, intermediate Markdown table reconstruction prompts, and deterministic uncertainty calibration guardrails.

### C. Evaluation Harness & Metrics Pipeline (`eval_harness.py`)
- **Metric Mathematics**: Formulated Exact Match ($EM$), Relative Tolerance Match ($\le \max(\text{tol}, 0.02)$), Visual Hallucination Rate, and Adversarial Refusal Precision.
- **Verification**: Verified that intermediate evidence outputs [`04_EVIDENCE/final_results/benchmark_final.csv`](file:///c:/Users/niraj/OneDrive/Desktop/Final_project/NIRAJ_FATING_AI_ML_LLM/04_EVIDENCE/final_results/benchmark_final.csv) match raw terminal logs bit-for-bit.

---

## 5. Security & Safe Credentials Protocol

1. **No Hardcoded Secrets**: No API keys, passwords, or authentication bearer tokens are committed to this repository.
2. **Environment Variable Ingestion**: All sensitive credentials are ingested strictly through dynamic system environment variables (`GEMINI_API_KEY`) or local untracked `.env` files.
3. **Offline Deterministic Fallback**: All evaluation scripts (`eval_harness.py`, `vlm_grounded_crop.py`, `walkthrough.py`) include robust offline simulation modes allowing automated testing and grading in isolated environments without live internet access.

---

## 6. Developer Sign-off

- **Candidate Name**: Niraj Fating  
- **Email / Identifier**: niraj.fating@edurank.ai / GitHub: `@nirajfating`  
- **Repository Location**: `c:\Users\niraj\OneDrive\Desktop\Final_project\NIRAJ_FATING_AI_ML_LLM\`  
- **Date of Attestation**: September 17, 2026  
- **Signature**: *Niraj Fating* [Digitally Certified]
