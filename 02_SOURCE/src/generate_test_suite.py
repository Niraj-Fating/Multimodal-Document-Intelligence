"""
generate_test_suite.py
EduRankAI Domain 03 - Synthetic Multimodal Document Evaluation Benchmark Generator

Generates 6 realistic, enterprise-grade synthetic document pages in both .png (300 DPI)
and .pdf formats, along with a 30-query ground-truth benchmark evaluation dataset.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

# Configure high-quality typography and rendering
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def get_project_paths():
    """Resolve base directory and target directories dynamically."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == 'src':
        base_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    else:
        base_dir = os.path.abspath(os.getcwd())
        if not os.path.exists(os.path.join(base_dir, '03_DATA_EVALUATION')):
            if os.path.exists(os.path.join(base_dir, 'NIRAJ_FATING_AI_ML_LLM', '03_DATA_EVALUATION')):
                base_dir = os.path.join(base_dir, 'NIRAJ_FATING_AI_ML_LLM')

    sample_pages_dir = os.path.join(base_dir, '03_DATA_EVALUATION', 'sample_pages')
    test_cases_dir = os.path.join(base_dir, '03_DATA_EVALUATION', 'test_cases')
    os.makedirs(sample_pages_dir, exist_ok=True)
    os.makedirs(test_cases_dir, exist_ok=True)
    return base_dir, sample_pages_dir, test_cases_dir

def save_dual_format(fig, base_filename, sample_pages_dir):
    """Export figure in both .png (300 DPI) and .pdf formats."""
    png_path = os.path.join(sample_pages_dir, f"{base_filename}.png")
    pdf_path = os.path.join(sample_pages_dir, f"{base_filename}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Exported {base_filename}.png ({os.path.getsize(png_path):,} bytes) and {base_filename}.pdf ({os.path.getsize(pdf_path):,} bytes)")

# ==============================================================================
# PAGE 1: 3-Column Financial Income Statement with Borderless Nested Cells
# ==============================================================================
def generate_page_1_income_statement(sample_pages_dir):
    fig = plt.figure(figsize=(8.5, 11), facecolor='#FAFAFA')
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.axis('off')

    # Header section
    ax.text(0.5, 0.96, "NOVA CORP ENTERPRISES", fontsize=16, fontweight='bold', ha='center', color='#111827')
    ax.text(0.5, 0.935, "CONSOLIDATED STATEMENTS OF OPERATIONS", fontsize=13, fontweight='semibold', ha='center', color='#374151')
    ax.text(0.5, 0.912, "(In Millions of USD, Except Per Share Amounts — Fiscal Years Ended December 31)", 
            fontsize=9.5, fontstyle='italic', ha='center', color='#6B7280')

    # Header decorative rule
    ax.plot([0.02, 0.98], [0.895, 0.895], color='#1E3A8A', lw=2.0)

    # Column headers
    col_x = [0.03, 0.58, 0.72, 0.86]
    ax.text(col_x[0], 0.868, "Line Item Description", fontsize=10, fontweight='bold', color='#1F2937')
    ax.text(col_x[1] + 0.08, 0.868, "2022", fontsize=10, fontweight='bold', ha='right', color='#1F2937')
    ax.text(col_x[2] + 0.08, 0.868, "2023", fontsize=10, fontweight='bold', ha='right', color='#1F2937')
    ax.text(col_x[3] + 0.08, 0.868, "2024", fontsize=10, fontweight='bold', ha='right', color='#1F2937')
    ax.plot([0.02, 0.98], [0.855, 0.855], color='#9CA3AF', lw=1.0)

    statement_rows = [
        (0, "Revenues:", "", "", "", True, False, False),
        (1, "Subscription and recurring services", "4,850.0", "6,120.5", "7,980.2", False, False, False),
        (1, "Professional services and consulting", "920.0", "1,115.3", "1,245.8", False, False, False),
        (0, "Total revenues", "$ 5,770.0", "$ 7,235.8", "$ 9,226.0", True, True, False),
        (0, "", "", "", "", False, False, False),
        (0, "Cost of revenues:", "", "", "", True, False, False),
        (1, "Cost of subscription services", "1,210.0", "1,450.2", "1,820.4", False, False, False),
        (1, "Cost of professional services", "740.0", "860.6", "940.6", False, False, False),
        (0, "Total cost of revenues", "1,950.0", "2,310.8", "2,761.0", True, True, False),
        (0, "", "", "", "", False, False, False),
        (0, "Gross profit", "3,820.0", "4,925.0", "6,465.0", True, True, False),
        (0, "", "", "", "", False, False, False),
        (0, "Operating expenses:", "", "", "", True, False, False),
        (1, "Research and development (R&D)", "1,420.0", "1,780.4", "2,250.0", False, False, False),
        (1, "Sales and marketing (S&M)", "1,250.0", "1,540.2", "1,920.5", False, False, False),
        (1, "General and administrative (G&A)", "430.0", "520.4", "634.5", False, False, False),
        (0, "Total operating expenses", "3,100.0", "3,841.0", "4,805.0", True, True, False),
        (0, "", "", "", "", False, False, False),
        (0, "Operating income", "720.0", "1,084.0", "1,660.0", True, True, False),
        (1, "Interest and other non-operating income, net", "45.0", "62.0", "85.5", False, False, False),
        (0, "Income before income taxes", "765.0", "1,146.0", "1,745.5", True, True, False),
        (1, "Provision for income taxes", "160.0", "240.5", "366.5", False, False, False),
        (0, "Net income", "$ 605.0", "$ 905.5", "$ 1,379.0", True, False, True),
        (0, "", "", "", "", False, False, False),
        (0, "Earnings per common share:", "", "", "", True, False, False),
        (1, "Basic earnings per share ($)", "1.92", "2.81", "4.24", False, False, False),
        (1, "Diluted earnings per share ($)", "$ 1.85", "$ 2.72", "$ 4.10", True, False, False),
        (0, "Weighted-average shares outstanding (M):", "", "", "", True, False, False),
        (1, "Basic common shares", "315.1", "322.2", "325.2", False, False, False),
        (1, "Diluted common shares", "327.0", "332.9", "336.3", False, False, False),
    ]

    current_y = 0.835
    row_height = 0.021

    for indent, label, v22, v23, v24, is_bold, is_subtotal, is_total in statement_rows:
        if label == "":
            current_y -= row_height * 0.5
            continue

        x_pos = col_x[0] + (indent * 0.035)
        weight = 'bold' if is_bold else 'normal'
        color = '#111827' if is_bold else '#374151'
        font_size = 9.2 if is_bold else 8.8

        if is_total:
            rect = Rectangle((0.02, current_y - 0.006), 0.96, row_height, facecolor='#EFF6FF', edgecolor='none', zorder=0)
            ax.add_patch(rect)
        elif is_subtotal:
            rect = Rectangle((0.02, current_y - 0.005), 0.96, row_height, facecolor='#F9FAFB', edgecolor='none', zorder=0)
            ax.add_patch(rect)

        ax.text(x_pos, current_y, label, fontsize=font_size, fontweight=weight, color=color, va='center', zorder=1)

        for col_idx, val in enumerate([v22, v23, v24]):
            val_x = col_x[col_idx + 1] + 0.08
            if val:
                ax.text(val_x, current_y, val, fontsize=font_size, fontweight=weight, color=color, ha='right', va='center', zorder=1)

        if is_subtotal and v22:
            for col_idx in range(3):
                vx = col_x[col_idx + 1]
                ax.plot([vx + 0.01, vx + 0.085], [current_y + 0.012, current_y + 0.012], color='#9CA3AF', lw=0.75)
        if is_total:
            for col_idx in range(3):
                vx = col_x[col_idx + 1]
                ax.plot([vx + 0.01, vx + 0.085], [current_y + 0.013, current_y + 0.013], color='#1F2937', lw=0.9)
                ax.plot([vx + 0.01, vx + 0.085], [current_y - 0.007, current_y - 0.007], color='#1F2937', lw=1.0)
                ax.plot([vx + 0.01, vx + 0.085], [current_y - 0.010, current_y - 0.010], color='#1F2937', lw=1.0)

        current_y -= row_height

    ax.text(0.03, 0.045, "See accompanying notes to consolidated financial statements in Item 8.", 
            fontsize=8.0, fontstyle='italic', color='#6B7280')
    ax.text(0.97, 0.045, "Page 48 of 112", fontsize=8.0, color='#6B7280', ha='right')

    save_dual_format(fig, "page_1_income_statement", sample_pages_dir)

# ==============================================================================
# PAGE 2: Segment Revenue Balance Sheet with Footnotes
# ==============================================================================
def generate_page_2_balance_sheet(sample_pages_dir):
    fig = plt.figure(figsize=(8.5, 11), facecolor='#FAFAFA')
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.axis('off')

    # Title Banner
    ax.text(0.5, 0.965, "VANGUARD CLOUD HOLDINGS CORP.", fontsize=15, fontweight='bold', ha='center', color='#111827')
    ax.text(0.5, 0.942, "SEGMENT REPORTING & CONSOLIDATED BALANCE SHEET", fontsize=12, fontweight='semibold', ha='center', color='#374151')
    ax.text(0.5, 0.922, "(Audited — In Millions of USD — As of December 31, 2023 and 2024)", 
            fontsize=9.0, fontstyle='italic', ha='center', color='#6B7280')
    ax.plot([0.02, 0.98], [0.908, 0.908], color='#047857', lw=2.0)

    # Section 1: Segment Identifiable Assets
    ax.text(0.03, 0.885, "NOTE 14 — SEGMENT IDENTIFIABLE ASSETS & PERFORMANCE", fontsize=10.5, fontweight='bold', color='#065F46')

    # Column positions adjusted for wide spacing
    col_x = [0.03, 0.60, 0.80]
    ax.text(col_x[0], 0.858, "Operating Business Segments", fontsize=9.5, fontweight='bold', color='#1F2937')
    ax.text(col_x[1] + 0.12, 0.858, "Dec 31, 2023", fontsize=9.2, fontweight='bold', ha='right', color='#1F2937')
    ax.text(col_x[2] + 0.12, 0.858, "Dec 31, 2024", fontsize=9.2, fontweight='bold', ha='right', color='#1F2937')
    ax.plot([0.02, 0.98], [0.848, 0.848], color='#9CA3AF', lw=1.0)

    segment_rows = [
        ("Cloud Infrastructure & Enterprise AI [1]", "8,450.0", "11,920.5", False),
        ("Digital Workplace & SaaS Applications", "4,210.0", "5,180.2", False),
        ("Consumer Devices & Hardware [2]", "2,890.5", "2,640.0", False),
        ("Corporate & Unallocated General Assets [*]", "1,120.0", "1,380.0", False),
        ("Total Segment Identifiable Assets", "$ 16,670.5", "$ 21,120.7", True),
    ]

    cur_y = 0.828
    for label, v23, v24, is_tot in segment_rows:
        weight = 'bold' if is_tot else 'normal'
        color = '#111827' if is_tot else '#374151'
        if is_tot:
            ax.plot([col_x[1] + 0.02, col_x[1] + 0.13], [cur_y + 0.012, cur_y + 0.012], color='#4B5563', lw=0.8)
            ax.plot([col_x[2] + 0.02, col_x[2] + 0.13], [cur_y + 0.012, cur_y + 0.012], color='#4B5563', lw=0.8)
            ax.plot([col_x[1] + 0.02, col_x[1] + 0.13], [cur_y - 0.007, cur_y - 0.007], color='#111827', lw=1.0)
            ax.plot([col_x[2] + 0.02, col_x[2] + 0.13], [cur_y - 0.007, cur_y - 0.007], color='#111827', lw=1.0)
            ax.plot([col_x[1] + 0.02, col_x[1] + 0.13], [cur_y - 0.010, cur_y - 0.010], color='#111827', lw=1.0)
            ax.plot([col_x[2] + 0.02, col_x[2] + 0.13], [cur_y - 0.010, cur_y - 0.010], color='#111827', lw=1.0)
        ax.text(col_x[0] + (0.02 if not is_tot else 0.0), cur_y, label, fontsize=9.0, fontweight=weight, color=color)
        ax.text(col_x[1] + 0.12, cur_y, v23, fontsize=9.0, fontweight=weight, ha='right', color=color)
        ax.text(col_x[2] + 0.12, cur_y, v24, fontsize=9.0, fontweight=weight, ha='right', color=color)
        cur_y -= 0.022

    # Section 2: Consolidated Balance Sheet Summary
    cur_y -= 0.015
    ax.text(0.03, cur_y, "CONDENSED CONSOLIDATED BALANCE SHEET SUMMARY", fontsize=10.5, fontweight='bold', color='#1F2937')
    cur_y -= 0.015
    ax.plot([0.02, 0.98], [cur_y, cur_y], color='#9CA3AF', lw=1.0)
    cur_y -= 0.020

    bs_rows = [
        ("ASSETS:", "", "", True, False),
        ("Cash, cash equivalents and marketable securities", "3,450.0", "4,820.0", False, False),
        ("Trade accounts receivable, net", "1,890.0", "2,340.5", False, False),
        ("Property, plant and equipment, net", "5,420.0", "7,110.2", False, False),
        ("Goodwill and identifiable intangible assets, net [3]", "5,910.5", "6,850.0", False, False),
        ("Total Consolidated Assets", "$ 16,670.5", "$ 21,120.7", True, True),
        ("", "", "", False, False),
        ("LIABILITIES AND STOCKHOLDERS' EQUITY:", "", "", True, False),
        ("Current liabilities (accounts payable & accrued expenses)", "3,120.0", "3,950.5", False, False),
        ("Long-term debt and finance lease obligations", "4,200.0", "4,850.0", False, False),
        ("Total liabilities", "7,320.0", "8,800.5", True, False),
        ("Stockholders' equity", "9,350.5", "12,320.2", True, False),
        ("Total Liabilities and Stockholders' Equity", "$ 16,670.5", "$ 21,120.7", True, True),
    ]

    for label, v23, v24, is_bold, is_double in bs_rows:
        if label == "":
            cur_y -= 0.008
            continue
        weight = 'bold' if is_bold else 'normal'
        color = '#111827' if is_bold else '#374151'
        indent = 0.0 if (is_bold or is_double) else 0.03

        if is_double:
            ax.plot([col_x[1] + 0.02, col_x[1] + 0.13], [cur_y + 0.012, cur_y + 0.012], color='#4B5563', lw=0.8)
            ax.plot([col_x[2] + 0.02, col_x[2] + 0.13], [cur_y + 0.012, cur_y + 0.012], color='#4B5563', lw=0.8)
            ax.plot([col_x[1] + 0.02, col_x[1] + 0.13], [cur_y - 0.007, cur_y - 0.007], color='#111827', lw=1.0)
            ax.plot([col_x[2] + 0.02, col_x[2] + 0.13], [cur_y - 0.007, cur_y - 0.007], color='#111827', lw=1.0)
            ax.plot([col_x[1] + 0.02, col_x[1] + 0.13], [cur_y - 0.010, cur_y - 0.010], color='#111827', lw=1.0)
            ax.plot([col_x[2] + 0.02, col_x[2] + 0.13], [cur_y - 0.010, cur_y - 0.010], color='#111827', lw=1.0)

        ax.text(col_x[0] + indent, cur_y, label, fontsize=8.8, fontweight=weight, color=color)
        if v23:
            ax.text(col_x[1] + 0.12, cur_y, v23, fontsize=8.8, fontweight=weight, ha='right', color=color)
            ax.text(col_x[2] + 0.12, cur_y, v24, fontsize=8.8, fontweight=weight, ha='right', color=color)
        cur_y -= 0.020

    # Section 3: Footnotes Section Box
    cur_y -= 0.015
    rect = FancyBboxPatch((0.02, cur_y - 0.125), 0.96, 0.130, boxstyle="round,pad=0.01",
                          facecolor='#F0FDF4', edgecolor='#A7F3D0', lw=1.0)
    ax.add_patch(rect)

    ax.text(0.04, cur_y - 0.010, "EXPLANATORY DISCLOSURE FOOTNOTES", fontsize=8.8, fontweight='bold', color='#065F46')
    
    # Escaping dollar signs so matplotlib math mode does not swallow spaces!
    footnotes = [
        "[1] Cloud Infrastructure segment assets include \\$1,450.0 million in specialized GPU server cluster assets capitalized under finance leases during FY2024.",
        "[2] Consumer Devices underwent an annual goodwill impairment evaluation in Q3 2024; no impairment recognized, but inventory reserves increased by \\$42.5 million.",
        "[3] Reflects final purchase price allocation from the strategic acquisition of NeuralStream Corp completed on April 12, 2024.",
        "[*] Excludes intersegment eliminations and corporate treasury hedges of \\$215.0 million in 2023 and \\$280.0 million in 2024."
    ]

    fn_y = cur_y - 0.032
    for fn in footnotes:
        ax.text(0.04, fn_y, fn, fontsize=7.8, color='#064E3B', va='top')
        fn_y -= 0.024

    save_dual_format(fig, "page_2_balance_sheet", sample_pages_dir)

# ==============================================================================
# PAGE 3: Multi-Bar Chart Showing Quarterly Margin Trends
# ==============================================================================
def generate_page_3_quarterly_margins(sample_pages_dir):
    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor='#FFFFFF')

    quarters = ['Q1-23', 'Q2-23', 'Q3-23', 'Q4-23', 'Q1-24', 'Q2-24', 'Q3-24', 'Q4-24']
    gross_margin = [64.2, 65.0, 66.4, 67.8, 68.5, 70.1, 71.8, 73.4]
    operating_margin = [18.5, 19.2, 20.8, 22.4, 23.1, 25.0, 26.5, 28.2]
    net_margin = [14.1, 14.8, 16.0, 17.5, 18.0, 19.4, 20.6, 22.1]

    x = np.arange(len(quarters))
    width = 0.26

    c_gross = '#1E3A8A'
    c_oper = '#0284C7'
    c_net = '#10B981'

    rects1 = ax.bar(x - width, gross_margin, width, label='Gross Margin (%)', color=c_gross, edgecolor='none', zorder=3)
    rects2 = ax.bar(x, operating_margin, width, label='Operating Margin (%)', color=c_oper, edgecolor='none', zorder=3)
    rects3 = ax.bar(x + width, net_margin, width, label='Net Margin (%)', color=c_net, edgecolor='none', zorder=3)

    def autolabel(rects, fmt="%.1f%%", offset=0.8):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(fmt % h,
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.2, fontweight='bold', color='#1F2937')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_title("STRATOS TECHNOLOGIES — QUARTERLY MARGIN DYNAMICS\nGross, Operating, and Net Profitability Trajectory (Q1 2023 - Q4 2024)",
                 fontsize=13, fontweight='bold', color='#111827', pad=18)
    ax.set_ylabel("Margin Percentage (%)", fontsize=10.5, fontweight='semibold', color='#374151')
    ax.set_xlabel("Fiscal Quarter", fontsize=10.5, fontweight='semibold', color='#374151', labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, fontsize=9.5, fontweight='semibold', color='#1F2937')
    ax.set_ylim(0, 88)

    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=9.5)

    # Clean non-overlapping Callout Annotation
    ax.annotate("Record Gross Margin of 73.4% in Q4-24\n(+920 bps expansion over 8 quarters)",
                xy=(7 - width, 74.8), xytext=(4.0, 80),
                arrowprops=dict(facecolor='#1E3A8A', shrink=0.08, width=1.4, headwidth=5),
                fontsize=8.5, fontweight='bold', color='#1E3A8A',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#EFF6FF', edgecolor='#93C5FD', lw=1))

    ax.annotate("8 Consecutive Quarters of Operating Expansion\n(+970 bps from Q1-23 to Q4-24)",
                xy=(7, 29.5), xytext=(3.6, 38),
                arrowprops=dict(facecolor='#0284C7', shrink=0.08, width=1.4, headwidth=5),
                fontsize=8.5, fontweight='bold', color='#0284C7',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#F0F9FF', edgecolor='#BAE6FD', lw=1))

    plt.figtext(0.12, 0.02, "Source: Internal Management Financial Records. Prepared for Executive Committee Review.",
                fontsize=8.0, fontstyle='italic', color='#6B7280')

    save_dual_format(fig, "page_3_quarterly_margins", sample_pages_dir)

# ==============================================================================
# PAGE 4: Mixed Layout (Two Columns of Text + Inline 4x4 Summary Table)
# ==============================================================================
def generate_page_4_mixed_layout(sample_pages_dir):
    fig = plt.figure(figsize=(8.5, 11), facecolor='#FFFFFF')
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
    ax.axis('off')

    # Executive Briefing Header
    ax.text(0.0, 0.965, "EXECUTIVE STRATEGY & GLOBAL EXECUTION BRIEFING", fontsize=15, fontweight='bold', color='#0F172A')
    ax.text(0.0, 0.940, "Quarterly Operations Review & Geographic Performance Breakdown — FY2024", fontsize=10.5, color='#475569')
    ax.plot([0.0, 1.0], [0.925, 0.925], color='#2563EB', lw=2.0)

    # Key Metadata bar
    meta_rect = Rectangle((0.0, 0.885), 1.0, 0.030, facecolor='#F1F5F9', edgecolor='none')
    ax.add_patch(meta_rect)
    ax.text(0.02, 0.895, "Published: October 24, 2024", fontsize=8.2, fontweight='bold', color='#334155')
    ax.text(0.35, 0.895, "Committee: Global Executive Operating Group", fontsize=8.2, color='#334155')
    ax.text(0.85, 0.895, "Classification: Confidential", fontsize=8.2, fontweight='bold', color='#DC2626')

    # Section 1: Overview narrative (Two Columns)
    ax.text(0.0, 0.850, "1. Strategic Operational Overview", fontsize=11.5, fontweight='bold', color='#0F172A')

    col1_text = (
        "During the fiscal year 2024, our global operations\n"
        "demonstrated robust execution across all major\n"
        "geographic territories. Strong demand for sovereign\n"
        "AI cloud clusters propelled top-line momentum in\n"
        "North America, while ongoing modernization efforts\n"
        "in Latin America yielded remarkable outperformance\n"
        "against initial plan projections.\n\n"
        "Supply chain lead times for specialized networking\n"
        "switches contracted by 28% year-over-year, enabling\n"
        "accelerated data center deployments in Phoenix, Salt\n"
        "Lake City, and suburban Frankfurt."
    )

    col2_text = (
        "Conversely, our European (EMEA) operations encountered\n"
        "transitory currency headwinds and longer procurement\n"
        "validation cycles stemming from regulatory adherence\n"
        "under the newly enacted EU AI Act directives.\n\n"
        "Despite localized headwinds, cross-selling velocity\n"
        "remained resilient with net expansion rates exceeding\n"
        "124% among top-tier enterprise accounts. The table\n"
        "below delineates performance variances against board-\n"
        "approved annual operational targets."
    )

    ax.text(0.0, 0.825, col1_text, fontsize=8.6, color='#334155', va='top', linespacing=1.45)
    ax.text(0.52, 0.825, col2_text, fontsize=8.6, color='#334155', va='top', linespacing=1.45)

    # Divider before table
    ax.plot([0.0, 1.0], [0.635, 0.635], color='#E2E8F0', lw=1.0)

    # Section 2: Inline Summary Table (Table Header and 4x4 Grid)
    ax.text(0.0, 0.605, "Table 1: FY2024 Regional Revenue Targets vs. Actual Performance", 
            fontsize=10.5, fontweight='bold', color='#0F172A')

    table_data = [
        ["Region", "Target Rev ($M)", "Actual Rev ($M)", "Variance (%)"],
        ["North America", "4,200.0", "4,580.0", "+9.0%"],
        ["EMEA", "2,500.0", "2,420.0", "-3.2%"],
        ["Asia-Pacific", "1,800.0", "1,950.0", "+8.3%"],
        ["Latin America", "650.0", "710.0", "+9.2%"],
        ["Total Global", "9,150.0", "9,660.0", "+5.6%"]
    ]

    ty = 0.565
    row_h = 0.035
    col_widths = [0.35, 0.22, 0.22, 0.21]
    col_starts = [0.0, 0.35, 0.57, 0.79]

    for row_idx, row in enumerate(table_data):
        is_header = (row_idx == 0)
        is_total = (row_idx == len(table_data) - 1)

        if is_header:
            bg_color = '#1E293B'
            text_color = '#FFFFFF'
            weight = 'bold'
        elif is_total:
            bg_color = '#E2E8F0'
            text_color = '#0F172A'
            weight = 'bold'
        elif row_idx % 2 == 1:
            bg_color = '#FFFFFF'
            text_color = '#1E293B'
            weight = 'normal'
        else:
            bg_color = '#F8FAFC'
            text_color = '#1E293B'
            weight = 'normal'

        rect = Rectangle((0.0, ty), 1.0, row_h, facecolor=bg_color, edgecolor='#CBD5E1', lw=0.5)
        ax.add_patch(rect)

        for col_idx, cell in enumerate(row):
            cx = col_starts[col_idx]
            cw = col_widths[col_idx]
            if col_idx == 0:
                ax.text(cx + 0.02, ty + (row_h / 2), cell, fontsize=9.0, fontweight=weight, 
                        color=text_color, va='center')
            else:
                cell_color = text_color
                if not is_header and col_idx == 3:
                    if cell.startswith('+'):
                        cell_color = '#16A34A' if not is_total else '#15803D'
                    elif cell.startswith('-'):
                        cell_color = '#DC2626'
                ax.text(cx + cw - 0.03, ty + (row_h / 2), cell, fontsize=9.0, fontweight=weight, 
                        color=cell_color, ha='right', va='center')

        ty -= row_h

    # Callout Box under table (escaping dollar signs!)
    callout_rect = FancyBboxPatch((0.0, ty - 0.075), 1.0, 0.065, boxstyle="round,pad=0.01",
                                  facecolor='#EFF6FF', edgecolor='#BFDBFE', lw=1.0)
    ax.add_patch(callout_rect)
    ax.text(0.02, ty - 0.025, "Key Takeaway:", fontsize=9.0, fontweight='bold', color='#1D4ED8')
    ax.text(0.02, ty - 0.048, 
            "Consolidated revenues of \\$9,660.0M exceeded baseline target by \\$510.0M (+5.6%),\n"
            "driven by Latin American (+9.2%) and North American (+9.0%) expansion.", 
            fontsize=8.5, color='#1E40AF', va='top')

    # Bottom Two-Column Section: Outlook with clean non-overlapping line breaks
    ty -= 0.115
    ax.plot([0.0, 1.0], [ty, ty], color='#E2E8F0', lw=1.0)
    ty -= 0.030
    ax.text(0.0, ty, "2. Capital Allocation & Regional Outlook for FY2025", fontsize=11.0, fontweight='bold', color='#0F172A')

    col1_outlook = (
        "Reinvestment priorities will focus on\n"
        "expanding sovereign data center capacity\n"
        "across selected EU member states to mitigate\n"
        "EMEA pipeline friction. The board earmarked\n"
        "\\$450.0 million in infrastructure capex."
    )
    col2_outlook = (
        "North America will continue as the core\n"
        "operating engine, generating over 47% of revenue.\n"
        "Enterprise sales headcount will expand\n"
        "by 12% specifically targeting regulated\n"
        "healthcare and defense customer segments."
    )
    ax.text(0.0, ty - 0.025, col1_outlook, fontsize=8.4, color='#334155', va='top', linespacing=1.4)
    ax.text(0.54, ty - 0.025, col2_outlook, fontsize=8.4, color='#334155', va='top', linespacing=1.4)

    save_dual_format(fig, "page_4_mixed_layout", sample_pages_dir)

# ==============================================================================
# PAGE 5: Fine-Print Table with Small Typography and Asterisks
# ==============================================================================
def generate_page_5_fine_print_table(sample_pages_dir):
    fig = plt.figure(figsize=(8.5, 11), facecolor='#FFFFFF')
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.axis('off')

    # Regulatory Header
    ax.text(0.5, 0.975, "REGULATORY CAPITAL ADEQUACY & RISK EXPOSURE SCHEDULE", 
            fontsize=13, fontweight='bold', ha='center', color='#111827')
    ax.text(0.5, 0.955, "Statutory Disclosures Pursuant to Basel III Framework — Schedule 14-B", 
            fontsize=9.5, fontstyle='italic', ha='center', color='#4B5563')
    ax.text(0.5, 0.938, "(All Monetary Figures in Millions of USD — Period Ended December 31, 2024)", 
            fontsize=8.5, ha='center', color='#6B7280')
    ax.plot([0.01, 0.99], [0.925, 0.925], color='#4338CA', lw=1.5)

    # Concise, well-spaced column headers
    headers = [
        "Asset Class Code\n& Category",
        "Weighted\nExposure ($M)",
        "Base Ratio\n(%)*",
        "Buffer\n(%)**",
        "Surcharge\n(%)***",
        "Net Exposure\n($M)†"
    ]
    col_widths = [0.35, 0.13, 0.13, 0.13, 0.13, 0.13]
    col_starts = [0.0]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)

    fine_rows = [
        ("CR-101 (Tier 1 Sovereign Bonds)", "14,250.0", "8.0%", "2.5%", "0.0%", "1,496.25"),
        ("CR-102 (Securitized Residential Mortgages)", "8,620.0", "10.5%", "3.0%", "0.5%", "1,206.80"),
        ("CR-103 (Senior Secured Corporate Debt)", "12,480.0", "9.0%", "2.8%", "0.25%", "1,503.84"),
        ("CR-104 (Subordinated High-Yield Notes)", "4,350.0", "12.0%", "3.5%", "1.0%", "717.75"),
        ("CR-105 (Commercial Real Estate Facilities)", "6,120.0", "11.0%", "3.2%", "0.75%", "914.94"),
        ("CR-106 (Short-Term Liquidity Facility)", "9,800.0", "7.5%", "2.0%", "0.0%", "931.00"),
        ("CR-107 (High-Yield Derivatives & Swaps)", "2,410.0", "15.0%", "4.5%", "2.0%", "518.15"),
        ("CR-108 (Trade Finance Receivables)", "5,190.0", "8.5%", "2.2%", "0.1%", "560.52"),
        ("TOTALS & AGGREGATE RISK WEIGHTS", "63,220.0", "—", "—", "—", "7,849.25")
    ]

    ty = 0.875
    row_h_hdr = 0.048
    row_h = 0.038

    # Render Header
    rect_h = Rectangle((0.0, ty), 1.0, row_h_hdr, facecolor='#312E81', edgecolor='#1E1B4B', lw=0.5)
    ax.add_patch(rect_h)
    for c_idx, h_text in enumerate(headers):
        cx = col_starts[c_idx]
        cw = col_widths[c_idx]
        if c_idx == 0:
            ax.text(cx + 0.015, ty + (row_h_hdr / 2), h_text, fontsize=7.6, fontweight='bold', color='#FFFFFF', va='center')
        else:
            ax.text(cx + cw - 0.015, ty + (row_h_hdr / 2), h_text, fontsize=7.4, fontweight='bold', color='#FFFFFF', ha='right', va='center')
    ty -= row_h

    # Render Rows
    for r_idx, row in enumerate(fine_rows):
        is_total = (r_idx == len(fine_rows) - 1)
        bg = '#F5F3FF' if (r_idx % 2 == 1 and not is_total) else ('#EEF2FF' if is_total else '#FFFFFF')
        rect = Rectangle((0.0, ty), 1.0, row_h, facecolor=bg, edgecolor='#E2E8F0', lw=0.5)
        ax.add_patch(rect)

        weight = 'bold' if is_total else 'normal'
        text_color = '#1E1B4B' if is_total else '#1F2937'

        for c_idx, cell in enumerate(row):
            cx = col_starts[c_idx]
            cw = col_widths[c_idx]
            if c_idx == 0:
                ax.text(cx + 0.015, ty + (row_h / 2), cell, fontsize=7.6, fontweight=weight, color=text_color, va='center')
            else:
                ax.text(cx + cw - 0.015, ty + (row_h / 2), cell, fontsize=7.6, fontweight=weight, color=text_color, ha='right', va='center')
        ty -= row_h

    # Fine Print Disclaimers and Asterisk Notes
    ty -= 0.030
    ax.plot([0.0, 1.0], [ty, ty], color='#94A3B8', lw=0.8)
    ty -= 0.022

    ax.text(0.01, ty, "STATUTORY FOOTNOTES, REGULATORY METHODOLOGY & LEGAL DISCLAIMERS", 
            fontsize=8.2, fontweight='bold', color='#1E1B4B')

    footnotes = [
        ("* Base Capital Ratio represents statutory Pillar 1 minimum tier requirements under Basel Committee on Banking Supervision (BCBS) Standard 402(a).", 
         "Non-compliance triggers mandatory capital distribution constraints under Section 165 of Dodd-Frank."),
        ("** Countercyclical capital buffer activated by the regulatory board pursuant to macroprudential stress assessment conducted on March 15, 2024.",
         "Buffers must be satisfied exclusively with common equity tier 1 (CET1) capital instruments."),
        ("*** G-SIB systemic surcharge applied in accordance with bucket 2 designation based on multi-indicator score methodology.",
         "Includes cross-jurisdictional activity, size, interconnectedness, substitutability, and complexity."),
        ("† Net Risk Exposure is computed strictly via statutory formula: Weighted Exposure * (Base Ratio + Buffer Requirement + Effective Surcharge) / 100.",
         "Subject to external audit validation by accredited independent supervisory examiners.")
    ]

    fn_y = ty - 0.025
    for tag_text, sub_text in footnotes:
        ax.text(0.015, fn_y, tag_text, fontsize=7.0, fontweight='semibold', color='#334155')
        fn_y -= 0.017
        ax.text(0.025, fn_y, sub_text, fontsize=6.8, fontstyle='italic', color='#64748B')
        fn_y -= 0.025

    # Bottom border and warning
    warn_rect = Rectangle((0.0, 0.04), 1.0, 0.065, facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=0.8)
    ax.add_patch(warn_rect)
    ax.text(0.02, 0.082, "WARNING: RESTRICTED REGULATORY INFORMATION", fontsize=7.5, fontweight='bold', color='#991B1B')
    ax.text(0.02, 0.055, 
            "The values in this schedule contain confidential supervisory review data. "
            "Unauthorized dissemination or replication constitutes a regulatory infraction under 12 CFR § 261.20.",
            fontsize=6.7, color='#B91C1C')

    save_dual_format(fig, "page_5_fine_print_table", sample_pages_dir)

# ==============================================================================
# PAGE 6: Dense Scientific Chart with Dual Y-Axes
# ==============================================================================
def generate_page_6_dense_scientific_chart(sample_pages_dir):
    fig, ax1 = plt.subplots(figsize=(10, 7.5), facecolor='#FFFFFF')
    ax2 = ax1.twinx()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    throughput = [340, 650, 1240, 2310, 4120, 6850, 9200, 10450]
    throughput_err = [20, 40, 75, 110, 180, 260, 310, 390]

    energy_joules = [48.5, 32.1, 21.4, 14.8, 11.2, 9.6, 8.9, 9.4]
    energy_err = [1.8, 1.2, 0.9, 0.6, 0.4, 0.3, 0.3, 0.4]

    x_indices = np.arange(len(batch_sizes))

    color_tp = '#1D4ED8'
    line1 = ax1.plot(x_indices, throughput, color=color_tp, marker='s', markersize=7, lw=2.2, 
                     label='Throughput (tokens/s)', zorder=4)
    ax1.fill_between(x_indices, 
                     np.array(throughput) - np.array(throughput_err), 
                     np.array(throughput) + np.array(throughput_err),
                     color=color_tp, alpha=0.15, zorder=2)

    color_en = '#DC2626'
    line2 = ax2.plot(x_indices, energy_joules, color=color_en, marker='D', markersize=7, lw=2.2, 
                     linestyle='--', label='Energy (Joules / 1K Tokens)', zorder=4)
    ax2.fill_between(x_indices, 
                     np.array(energy_joules) - np.array(energy_err), 
                     np.array(energy_joules) + np.array(energy_err),
                     color=color_en, alpha=0.15, zorder=2)

    ax1.set_title("DISTRIBUTED LLM INFERENCE EFFICIENCY: THROUGHPUT VS. ENERGY DISSIPATION\nMicroarchitectural Benchmark on 8x H100 GPU Cluster (70B Model, FP8 Tensor Cores)",
                  fontsize=12.5, fontweight='bold', color='#0F172A', pad=18)
    ax1.set_xlabel("Concurrent Inference Batch Size (tokens/batch)", fontsize=10.5, fontweight='semibold', color='#334155', labelpad=10)
    ax1.set_ylabel("Inference Throughput (tokens/second)", fontsize=10.5, fontweight='bold', color=color_tp, labelpad=8)
    ax2.set_ylabel("Energy Dissipation (Joules per 1,000 Tokens)", fontsize=10.5, fontweight='bold', color=color_en, labelpad=8)

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(batch_sizes, fontsize=9.5, fontweight='semibold', color='#1E293B')
    ax1.set_ylim(0, 12000)
    ax2.set_ylim(0, 60)

    ax1.tick_params(axis='y', labelcolor=color_tp)
    ax2.tick_params(axis='y', labelcolor=color_en)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(0.08, 0.78), frameon=True, 
               facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.0)

    ax2.annotate("Pareto Optimal Operating Point\nBatch Size = 64 (9,200 tok/s @ 8.9 J/kTok)",
                 xy=(6, 8.9), xytext=(3.5, 18),
                 arrowprops=dict(facecolor='#047857', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, fontweight='bold', color='#065F46',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='#ECFDF5', edgecolor='#6EE7B7', lw=1))

    ax2.annotate("Memory Bandwidth Saturation\nEnergy Uptick to 9.4 J/kTok at Batch 128",
                 xy=(7, 9.4), xytext=(4.8, 3.5),
                 arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.2, headwidth=5),
                 fontsize=8.0, fontweight='bold', color='#B91C1C',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=1))

    plt.figtext(0.10, 0.02, 
                "Figure 1: Measured execution metrics across varying batch workloads. Shaded bands depict +/- 1 sigma experimental variance over 50 iterations.",
                fontsize=8.0, fontstyle='italic', color='#64748B')

    save_dual_format(fig, "page_6_dense_scientific_chart", sample_pages_dir)

# ==============================================================================
# GROUND TRUTH BENCHMARK GENERATION (30 Queries across 6 Pages)
# ==============================================================================
def generate_ground_truth_test_cases(test_cases_dir):
    """
    Generate ground_truth.json containing 30 verified benchmark queries:
    - 12 Direct Lookups
    - 6 Cross-Row Arithmetic Calculations
    - 6 Visual Chart Trend Readings
    - 6 Adversarial / Out-of-Domain Traps (answer strictly "NOT_PRESENT")
    """
    ground_truth = [
        # --- PAGE 1: INCOME STATEMENT ---
        {
            "id": "GT-001",
            "document": "page_1_income_statement.png",
            "question": "What was the Total Revenues of Nova Corp in FY2024?",
            "type": "direct_lookup",
            "ground_truth_value": 9226.0,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.0,
            "target_location": "Page 1: Row 'Total revenues', Column '2024'",
            "is_adversarial": False
        },
        {
            "id": "GT-002",
            "document": "page_1_income_statement.png",
            "question": "How much did Nova Corp spend on Research and development (R&D) in FY2023?",
            "type": "direct_lookup",
            "ground_truth_value": 1780.4,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.0,
            "target_location": "Page 1: Row 'Research and development (R&D)', Column '2023'",
            "is_adversarial": False
        },
        {
            "id": "GT-003",
            "document": "page_1_income_statement.png",
            "question": "What was the year-over-year dollar increase in Net income from FY2023 to FY2024?",
            "type": "arithmetic_calculation",
            "ground_truth_value": 473.5,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.0,
            "target_location": "Page 1: Row 'Net income', Columns '2023' ($905.5M) and '2024' ($1,379.0M)",
            "is_adversarial": False
        },
        {
            "id": "GT-004",
            "document": "page_1_income_statement.png",
            "question": "What percentage of Total revenues was generated by Subscription and recurring services in FY2024?",
            "type": "arithmetic_calculation",
            "ground_truth_value": 86.5,
            "ground_truth_unit": "Percentage",
            "tolerance_pct": 0.5,
            "target_location": "Page 1: Rows 'Subscription and recurring services' ($7,980.2M) / 'Total revenues' ($9,226.0M)",
            "is_adversarial": False
        },
        {
            "id": "GT-005",
            "document": "page_1_income_statement.png",
            "question": "What was the restructuring expense recorded by Nova Corp in FY2022?",
            "type": "adversarial_trap",
            "ground_truth_value": "NOT_PRESENT",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 1: Line item not present in financial statement",
            "is_adversarial": True
        },

        # --- PAGE 2: BALANCE SHEET & SEGMENT ASSETS ---
        {
            "id": "GT-006",
            "document": "page_2_balance_sheet.png",
            "question": "What was the value of Cloud Infrastructure & Enterprise AI segment identifiable assets as of December 31, 2024?",
            "type": "direct_lookup",
            "ground_truth_value": 11920.5,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.0,
            "target_location": "Page 2: Segment Identifiable Assets Table, Row 1, Column '2024'",
            "is_adversarial": False
        },
        {
            "id": "GT-007",
            "document": "page_2_balance_sheet.png",
            "question": "According to footnote [1], how much in specialized GPU server cluster assets was capitalized under finance leases during FY2024?",
            "type": "direct_lookup",
            "ground_truth_value": 1450.0,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.0,
            "target_location": "Page 2: Explanatory Footnote [1]",
            "is_adversarial": False
        },
        {
            "id": "GT-008",
            "document": "page_2_balance_sheet.png",
            "question": "What is the combined asset value of Digital Workplace & SaaS Applications and Consumer Devices & Hardware in 2024?",
            "type": "arithmetic_calculation",
            "ground_truth_value": 7820.2,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.0,
            "target_location": "Page 2: Segment Assets Table, Rows 2 and 3, Column '2024' ($5,180.2M + $2,640.0M)",
            "is_adversarial": False
        },
        {
            "id": "GT-009",
            "document": "page_2_balance_sheet.png",
            "question": "What was the total dollar increase in Stockholders' equity between 2023 and 2024?",
            "type": "arithmetic_calculation",
            "ground_truth_value": 2969.7,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.0,
            "target_location": "Page 2: Balance Sheet Summary, Row 'Stockholders equity' ($12,320.2M - $9,350.5M)",
            "is_adversarial": False
        },
        {
            "id": "GT-010",
            "document": "page_2_balance_sheet.png",
            "question": "What was the dividend payout per share disclosed in the footnotes of the balance sheet for FY2024?",
            "type": "adversarial_trap",
            "ground_truth_value": "NOT_PRESENT",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 2: Disclosure absent from balance sheet notes",
            "is_adversarial": True
        },

        # --- PAGE 3: MULTI-BAR MARGIN TRENDS ---
        {
            "id": "GT-011",
            "document": "page_3_quarterly_margins.png",
            "question": "What was the Gross Margin percentage recorded in Q3 2024?",
            "type": "direct_lookup",
            "ground_truth_value": 71.8,
            "ground_truth_unit": "Percentage",
            "tolerance_pct": 0.2,
            "target_location": "Page 3: Bar chart, Quarter Q3-24, Gross Margin bar label",
            "is_adversarial": False
        },
        {
            "id": "GT-012",
            "document": "page_3_quarterly_margins.png",
            "question": "Which of the three charted margin metrics maintains the lowest percentage value in every observed quarter?",
            "type": "visual_chart_trend",
            "ground_truth_value": "Net Margin",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 3: Comparative visual bar heights across all 8 quarters (green bars consistently lowest)",
            "is_adversarial": False
        },
        {
            "id": "GT-013",
            "document": "page_3_quarterly_margins.png",
            "question": "In which fiscal quarter did Stratos Technologies achieve its peak Gross Margin percentage?",
            "type": "visual_chart_trend",
            "ground_truth_value": "Q4-24",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 3: Peak bar across all 8 quarters (73.4% in Q4-24)",
            "is_adversarial": False
        },
        {
            "id": "GT-014",
            "document": "page_3_quarterly_margins.png",
            "question": "By how many percentage points did the Operating Margin expand from Q1 2023 to Q4 2024?",
            "type": "visual_chart_trend",
            "ground_truth_value": 9.7,
            "ground_truth_unit": "Percentage points",
            "tolerance_pct": 0.2,
            "target_location": "Page 3: Difference between Q4-24 Operating Margin (28.2%) and Q1-23 (18.5%)",
            "is_adversarial": False
        },
        {
            "id": "GT-015",
            "document": "page_3_quarterly_margins.png",
            "question": "What was the Free Cash Flow margin reported in Q2 2024?",
            "type": "adversarial_trap",
            "ground_truth_value": "NOT_PRESENT",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 3: Free Cash Flow margin metric is not charted",
            "is_adversarial": True
        },

        # --- PAGE 4: MIXED LAYOUT TEXT + INLINE SUMMARY TABLE ---
        {
            "id": "GT-016",
            "document": "page_4_mixed_layout.png",
            "question": "What was the Actual Revenue generated in North America according to the regional performance table?",
            "type": "direct_lookup",
            "ground_truth_value": 4580.0,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.0,
            "target_location": "Page 4: Table 1, Row 'North America', Column 'Actual Rev ($M)'",
            "is_adversarial": False
        },
        {
            "id": "GT-017",
            "document": "page_4_mixed_layout.png",
            "question": "Which operating region experienced a negative revenue variance relative to target in Table 1?",
            "type": "direct_lookup",
            "ground_truth_value": "EMEA",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 4: Table 1, Row 'EMEA', Column 'Variance (%)' (-3.2%)",
            "is_adversarial": False
        },
        {
            "id": "GT-018",
            "document": "page_4_mixed_layout.png",
            "question": "What is the dollar difference between Total Global Actual Revenue and Total Global Target Revenue?",
            "type": "arithmetic_calculation",
            "ground_truth_value": 510.0,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.0,
            "target_location": "Page 4: Table 1, Row 'Total Global' ($9,660.0M - $9,150.0M)",
            "is_adversarial": False
        },
        {
            "id": "GT-019",
            "document": "page_4_mixed_layout.png",
            "question": "What was the percentage revenue variance achieved by Latin America relative to its baseline target?",
            "type": "direct_lookup",
            "ground_truth_value": 9.2,
            "ground_truth_unit": "Percentage",
            "tolerance_pct": 0.1,
            "target_location": "Page 4: Table 1, Row 'Latin America', Column 'Variance (%)'",
            "is_adversarial": False
        },
        {
            "id": "GT-020",
            "document": "page_4_mixed_layout.png",
            "question": "What was the projected 2026 revenue target stated for the Asia-Pacific region in the briefing?",
            "type": "adversarial_trap",
            "ground_truth_value": "NOT_PRESENT",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 4: 2026 regional targets not mentioned",
            "is_adversarial": True
        },

        # --- PAGE 5: FINE-PRINT REGULATORY TABLE ---
        {
            "id": "GT-021",
            "document": "page_5_fine_print_table.png",
            "question": "What is the Weighted Exposure for asset class code CR-103 (Senior Secured Corporate Debt)?",
            "type": "direct_lookup",
            "ground_truth_value": 12480.0,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.0,
            "target_location": "Page 5: Schedule 14-B, Row 'CR-103', Column 'Weighted Exposure ($M)'",
            "is_adversarial": False
        },
        {
            "id": "GT-022",
            "document": "page_5_fine_print_table.png",
            "question": "What is the Base Capital Ratio for asset code CR-107 (High-Yield Derivatives & Swaps)?",
            "type": "direct_lookup",
            "ground_truth_value": 15.0,
            "ground_truth_unit": "Percentage",
            "tolerance_pct": 0.0,
            "target_location": "Page 5: Schedule 14-B, Row 'CR-107', Column 'Base Capital Ratio (%)*'",
            "is_adversarial": False
        },
        {
            "id": "GT-023",
            "document": "page_5_fine_print_table.png",
            "question": "According to footnote [***], what systemic score bucket is the G-SIB surcharge based upon?",
            "type": "direct_lookup",
            "ground_truth_value": "bucket 2",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 5: Statutory Footnote (***)",
            "is_adversarial": False
        },
        {
            "id": "GT-024",
            "document": "page_5_fine_print_table.png",
            "question": "What is the sum of Net Risk Exposures for CR-101 (Tier 1 Sovereign Bonds) and CR-106 (Short-Term Liquidity Facility)?",
            "type": "arithmetic_calculation",
            "ground_truth_value": 2427.25,
            "ground_truth_unit": "USD Millions",
            "tolerance_pct": 0.05,
            "target_location": "Page 5: Schedule 14-B, Column 'Net Risk Exposure' ($1,496.25M + $931.00M)",
            "is_adversarial": False
        },
        {
            "id": "GT-025",
            "document": "page_5_fine_print_table.png",
            "question": "What is the minimum Leverage Ratio requirement under Basel IV specified in footnote 5 of the schedule?",
            "type": "adversarial_trap",
            "ground_truth_value": "NOT_PRESENT",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 5: Basel IV leverage ratio is not in document",
            "is_adversarial": True
        },

        # --- PAGE 6: DENSE SCIENTIFIC DUAL-AXIS CHART ---
        {
            "id": "GT-026",
            "document": "page_6_dense_scientific_chart.png",
            "question": "What is the model inference throughput measured at concurrent batch size 16?",
            "type": "direct_lookup",
            "ground_truth_value": 4120.0,
            "ground_truth_unit": "Tokens/sec",
            "tolerance_pct": 2.0,
            "target_location": "Page 6: Dual-axis chart, Left Axis at Batch Size 16",
            "is_adversarial": False
        },
        {
            "id": "GT-027",
            "document": "page_6_dense_scientific_chart.png",
            "question": "At which concurrent batch size is the lowest energy dissipation per 1,000 tokens achieved?",
            "type": "visual_chart_trend",
            "ground_truth_value": "64",
            "ground_truth_unit": "Batch Size",
            "tolerance_pct": 0.0,
            "target_location": "Page 6: Minimum point on Right Axis / Green Pareto callout box",
            "is_adversarial": False
        },
        {
            "id": "GT-028",
            "document": "page_6_dense_scientific_chart.png",
            "question": "What is the Energy Dissipation value at the optimal operating point (Batch Size 64)?",
            "type": "visual_chart_trend",
            "ground_truth_value": 8.9,
            "ground_truth_unit": "Joules per 1,000 Tokens",
            "tolerance_pct": 2.0,
            "target_location": "Page 6: Right Y-Axis curve at Batch Size 64 / Pareto callout annotation",
            "is_adversarial": False
        },
        {
            "id": "GT-029",
            "document": "page_6_dense_scientific_chart.png",
            "question": "What happens to the Energy Dissipation curve when the batch size increases from 64 to 128?",
            "type": "visual_chart_trend",
            "ground_truth_value": "increases",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 6: Right Y-Axis curve uptick from 8.9 to 9.4 J/kTok / Red callout",
            "is_adversarial": False
        },
        {
            "id": "GT-030",
            "document": "page_6_dense_scientific_chart.png",
            "question": "What is the GPU clock frequency in MHz indicated for the FP16 baseline on the chart?",
            "type": "adversarial_trap",
            "ground_truth_value": "NOT_PRESENT",
            "ground_truth_unit": None,
            "tolerance_pct": 0.0,
            "target_location": "Page 6: Metric and FP16 baseline are not present on chart",
            "is_adversarial": True
        }
    ]

    gt_file = os.path.join(test_cases_dir, "ground_truth.json")
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"  [OK] Saved ground_truth.json ({len(ground_truth)} queries) to: {gt_file}")
    return ground_truth

# ==============================================================================
# MAIN EXECUTION RUNNER
# ==============================================================================
def main():
    print("=" * 80)
    print("EduRankAI Domain 03: Synthetic Multimodal Document Benchmark Generator")
    print("=" * 80)

    base_dir, sample_pages_dir, test_cases_dir = get_project_paths()
    print(f"Project Base Directory: {base_dir}")
    print(f"Sample Pages Output:   {sample_pages_dir}")
    print(f"Test Cases Output:     {test_cases_dir}\n")

    print("[Step 1/6] Generating Page 1: 3-Column Financial Income Statement (Nested/Borderless)...")
    generate_page_1_income_statement(sample_pages_dir)

    print("[Step 2/6] Generating Page 2: Segment Revenue Balance Sheet with Footnotes...")
    generate_page_2_balance_sheet(sample_pages_dir)

    print("[Step 3/6] Generating Page 3: Multi-Bar Chart Showing Quarterly Margin Trends...")
    generate_page_3_quarterly_margins(sample_pages_dir)

    print("[Step 4/6] Generating Page 4: Mixed Layout (Two Columns Text + Inline 4x4 Table)...")
    generate_page_4_mixed_layout(sample_pages_dir)

    print("[Step 5/6] Generating Page 5: Fine-Print Table with Small Typography & Asterisks...")
    generate_page_5_fine_print_table(sample_pages_dir)

    print("[Step 6/6] Generating Page 6: Dense Scientific Chart with Dual Y-Axes...")
    generate_page_6_dense_scientific_chart(sample_pages_dir)

    print("\n[Step 7/7] Generating 30-Query Benchmark Test Cases (ground_truth.json)...")
    gt = generate_ground_truth_test_cases(test_cases_dir)

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY & VERIFICATION")
    print("=" * 80)
    generated_files = os.listdir(sample_pages_dir)
    print(f"Total documents in sample_pages/: {len(generated_files)} files")
    for f in sorted(generated_files):
        fpath = os.path.join(sample_pages_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  • {f:<40} ({size_kb:>8.1f} KB)")

    type_counts = {}
    for item in gt:
        t = item["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\nGround Truth Query Breakdown (Total: 30):")
    for t, c in type_counts.items():
        print(f"  • {t:<28}: {c} queries")
    print("=" * 80)
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
