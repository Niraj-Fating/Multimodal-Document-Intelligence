# Interactive CLI Demonstration Suite (`05_DEMO/`)

**Module**: `05_DEMO/walkthrough.py`  
**Purpose**: Interactive terminal walkthrough evaluating document pages side by side across **Approach A (1D Text Baseline)**, **Approach B (Direct Zero-Shot VLM)**, and **Approach C (Spatial Patch Cropping & Grounded Structured VLM)**.

---

## Quick Start

### 1. Prerequisites
Ensure project dependencies are installed from root:
```bash
pip install -r requirements.txt
```

### 2. Run Presets

The CLI includes built-in preset evaluation scenarios illustrating key systematic differences:

```bash
# Preset 1: Adversarial Trap (GT-015)
# Shows Approach B hallucinating 25.0% vs. Approach C successfully refusing with NOT_PRESENT
python 05_DEMO/walkthrough.py --preset trap

# Preset 2: Multi-Column Financial Table (GT-002)
# Shows Approach A swapping columns ($1,320.0M) vs. Approach C grounding to FY2023 ($1,780.4M)
python 05_DEMO/walkthrough.py --preset table

# Preset 3: Direct Visual Chart Reading (GT-011)
# Shows Approach A failing on charts vs. Approach C extracting Q3-24 Gross Margin (71.8%)
python 05_DEMO/walkthrough.py --preset chart

# Preset 4: Comparative Chart Trend (GT-012)
# Identifies lowest bar across all 8 fiscal quarters (Net Margin)
python 05_DEMO/walkthrough.py --preset trend

# Preset 5: Regulatory Fine-Print Footnote (GT-023)
# Preserves 300 DPI micro-typography to extract G-SIB Bucket 2 surcharge from footnote [***]
python 05_DEMO/walkthrough.py --preset fineprint

# Preset 6: Dense Dual-Axis Scientific Chart (GT-026)
# Extracts throughput at batch size 16 (4,120 tokens/sec)
python 05_DEMO/walkthrough.py --preset scientific
```

### 3. Run Custom Query

You can target any arbitrary document and question:
```bash
python 05_DEMO/walkthrough.py \
  --document page_1_income_statement.png \
  --question "What was the Total Revenues of Nova Corp in FY2024?"
```

---

## Sample Execution Output

```
=====================================================================================
      EduRankAI Domain 03 — Interactive Multimodal Document Intelligence Demo
=====================================================================================
Target Document : page_3_quarterly_margins.png
User Question   : "What was the Free Cash Flow margin reported in Q2 2024?"
Execution Mode  : Offline Calibrated Simulation
-------------------------------------------------------------------------------------

[1/3] Executing Approach A (1D Sequential Text Parser Baseline)...
[2/3] Executing Approach B (Direct Zero-Shot Multimodal VLM)...
[3/3] Executing Approach C (Spatial Patch Cropping & Grounded Structured VLM)...

=====================================================================================
                        COMPARATIVE EXTRACTION SUMMARY
=====================================================================================
+---------------------+-----------------------+---------------------------------------+-------------------------------+
| Attribute           | Approach A (1D Text)  | Approach B (Direct VLM)               | Approach C (Grounded Patch)   |
+=====================+=======================+=======================================+===============================+
| Extracted Answer    | NOT_FOUND             | 25.0%                                 | NOT_PRESENT                   |
+---------------------+-----------------------+---------------------------------------+-------------------------------+
| Unit / Scale        | N/A                   | N/A                                   | None                          |
+---------------------+-----------------------+---------------------------------------+-------------------------------+
| Spatial Grounding   | None (Ungrounded 1D)  | [610, 580, 640, 630] | Misread Ope... | Native Crop BBox: [326, ...]  |
+---------------------+-----------------------+---------------------------------------+-------------------------------+
| Confidence Score    | N/A                   | Uncalibrated                          | 1.00                          |
+---------------------+-----------------------+---------------------------------------+-------------------------------+
| Failure / Trap Flags| None                  | Vulnerable to Trap                    | NOT_PRESENT_IN_DOCUMENT       |
+---------------------+-----------------------+---------------------------------------+-------------------------------+
| Latency (seconds)   | 0.052s                | 0.081s                                | 0.142s                        |
+---------------------+-----------------------+---------------------------------------+-------------------------------+

=====================================================================================
     APPROACH C: STAGE 1 INTERMEDIATE SPATIAL RECONSTRUCTION (HIGH-RES CROP)
=====================================================================================
| Chart Series in Q2-24 Cluster | Visual Color | Plotted Percentage |
| :--- | :--- | :--- |
| Gross Margin | Dark Blue | 70.2% |
| Operating Margin | Amber/Orange | 25.0% |
| Net Margin | Emerald Green | 17.8% |
| *Free Cash Flow Margin* | *NOT CHARTED* | *ABSENT* |
```
