"""
========================================================================================
COMBINED 12-MODEL CONFUSION MATRIX MASTER SUITE: ALPHA BAND VS. BROADBAND
========================================================================================
Combines all 6 Alpha Band (8-13 Hz) and all 6 BroadBand (0.5-40 Hz) confusion matrices
into publication-grade, Elsevier-compliant unified comparative figures:
1. Layout A (4x3 Grid): Stacked Section I (Alpha Band 2x3) & Section II (BroadBand 2x3)
2. Layout B (6x2 Grid): Direct Side-by-Side Architecture Pairing (Alpha vs. BroadBand)
All subplots formatted in unified Elsevier Blues with percentages and sample counts.
========================================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------
# Directories & Aesthetics
# --------------------------------------------------------------------------------------
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent if (CURRENT_DIR.parent / "src").is_dir() else CURRENT_DIR
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

ROOT_DIR = str(PROJECT_ROOT)
CODE_DIR = str(CODE_DIR)
PUB_FIG_DIR = os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures", "Result_Oriented_Figures")
PUB_ROOT_DIR = os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures")
SIX_MODEL_DIR = os.path.join(PUB_FIG_DIR, "Six_Model_Comparative_Diagnostics")
ALPHA_JSON_PATH = os.path.join(ROOT_DIR, "05 CSV_and_TXT", "Model_Benchmarks_and_Ablations", "Alpha_Band", "Alpha_Band_Metrics.json")
BROAD_JSON_PATH = os.path.join(ROOT_DIR, "05 CSV_and_TXT", "Model_Benchmarks_and_Ablations", "BroadBand", "BroadBand_Metrics.json")

os.makedirs(PUB_FIG_DIR, exist_ok=True)
os.makedirs(PUB_ROOT_DIR, exist_ok=True)
os.makedirs(SIX_MODEL_DIR, exist_ok=True)

# Publication Typography
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.titlesize'] = 11.0
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10.0
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 9.0
plt.rcParams['ytick.labelsize'] = 9.0

STAGE_NAMES = ['Normal', 'Pre-Ictal', 'Ictal', 'Inter-Ictal', 'Post-Ictal']

MODELS = [
    ('Standard_LSTM', 'Standard LSTM'),
    ('BiLSTM', 'BiLSTM'),
    ('TriLSTM', 'TriLSTM'),
    ('TCN_1', 'TCN-1'),
    ('TCN_2', 'TCN-2'),
    ('STATMR_Net_Proposed', 'ST-ATMR-Net (Proposed)')
]

def load_matrices():
    with open(ALPHA_JSON_PATH, 'r') as f:
        aj = json.load(f)
    with open(BROAD_JSON_PATH, 'r') as f:
        bj = json.load(f)

    alpha_data = {}
    broad_data = {}

    for k, disp in MODELS:
        # Alpha
        atm = aj[k]['test_metrics']
        a_cm = np.array(atm['confusion_matrix'])
        a_norm = np.array(atm['confusion_matrix_norm']) * 100.0
        alpha_data[k] = {
            'name': disp,
            'cm': a_cm,
            'cm_norm': a_norm,
            'acc': atm['accuracy'] * 100.0,
            'bal_acc': atm['balanced_accuracy'] * 100.0,
            'f1': atm['macro_f1'] * 100.0
        }

        # BroadBand
        btm = bj[k]['test_metrics']
        b_cm = np.array(btm['confusion_matrix'])
        b_norm = np.array(btm['confusion_matrix_norm']) * 100.0
        broad_data[k] = {
            'name': disp,
            'cm': b_cm,
            'cm_norm': b_norm,
            'acc': btm['accuracy'] * 100.0,
            'bal_acc': btm['balanced_accuracy'] * 100.0,
            'f1': btm['macro_f1'] * 100.0
        }

    return alpha_data, broad_data

# --------------------------------------------------------------------------------------
# 1. Layout A: 4x3 Grid (Stacked Section I: Alpha Band & Section II: BroadBand)
# --------------------------------------------------------------------------------------
def render_layout_4x3(alpha_data, broad_data):
    print("[1/2] Rendering Layout A: 4x3 Stacked Master Confusion Matrix Suite...")
    fig, axes = plt.subplots(4, 3, figsize=(18.5, 23.5), dpi=300)

    # Top 2 rows: Alpha Band
    for idx, (m_key, _) in enumerate(MODELS):
        r = idx // 3
        c = idx % 3
        ax = axes[r, c]
        d = alpha_data[m_key]
        cm = d['cm']
        cm_norm = d['cm_norm']

        annot = np.empty_like(cm, dtype=object)
        for i in range(5):
            for j in range(5):
                annot[i, j] = f"{cm_norm[i, j]:.1f}%\n({cm[i, j]})"

        sns.heatmap(
            cm_norm, annot=annot, fmt="", cmap="Blues",
            cbar=True if c == 2 else False,
            cbar_kws={'label': 'Recall (%)'} if c == 2 else None,
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
            ax=ax, linewidths=0.9, linecolor="#dddddd", vmin=0.0, vmax=100.0
        )

        if 'STATMR' in m_key:
            for _, s in ax.spines.items():
                s.set_color('#0b5394')
                s.set_linewidth(2.6)
            ax.set_title(f"Alpha Band: {d['name']} (Proposed)\nAcc: {d['acc']:.2f}% | Bal Acc: {d['bal_acc']:.2f}% | Macro-F1: {d['f1']:.2f}%", color='#0b5394', pad=8)
        else:
            ax.set_title(f"Alpha Band: {d['name']}\nAcc: {d['acc']:.2f}% | Bal Acc: {d['bal_acc']:.2f}% | Macro-F1: {d['f1']:.2f}%", pad=8)

        ax.set_xlabel("Predicted Stage" if r == 1 else "", labelpad=5)
        ax.set_ylabel("True Stage" if c == 0 else "", labelpad=5)
        ax.tick_params(axis='x', rotation=25)
        ax.tick_params(axis='y', rotation=0)

    # Bottom 2 rows: BroadBand
    for idx, (m_key, _) in enumerate(MODELS):
        r = (idx // 3) + 2
        c = idx % 3
        ax = axes[r, c]
        d = broad_data[m_key]
        cm = d['cm']
        cm_norm = d['cm_norm']

        annot = np.empty_like(cm, dtype=object)
        for i in range(5):
            for j in range(5):
                annot[i, j] = f"{cm_norm[i, j]:.1f}%\n({cm[i, j]})"

        sns.heatmap(
            cm_norm, annot=annot, fmt="", cmap="Blues",
            cbar=True if c == 2 else False,
            cbar_kws={'label': 'Recall (%)'} if c == 2 else None,
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
            ax=ax, linewidths=0.9, linecolor="#dddddd", vmin=0.0, vmax=100.0
        )

        if 'STATMR' in m_key:
            for _, s in ax.spines.items():
                s.set_color('#0b5394')
                s.set_linewidth(2.6)
            ax.set_title(f"BroadBand: {d['name']} (Proposed)\nAcc: {d['acc']:.2f}% | Bal Acc: {d['bal_acc']:.2f}% | Macro-F1: {d['f1']:.2f}%", color='#0b5394', pad=8)
        else:
            ax.set_title(f"BroadBand: {d['name']}\nAcc: {d['acc']:.2f}% | Bal Acc: {d['bal_acc']:.2f}% | Macro-F1: {d['f1']:.2f}%", pad=8)

        ax.set_xlabel("Predicted Stage" if r == 3 else "", labelpad=5)
        ax.set_ylabel("True Stage" if c == 0 else "", labelpad=5)
        ax.tick_params(axis='x', rotation=25)
        ax.tick_params(axis='y', rotation=0)

    # Section Headers
    fig.text(0.5, 0.988, "Comprehensive 12-Model Contingency Benchmark: Alpha Band (8–13 Hz) vs. BroadBand (0.5–40 Hz)",
             ha='center', va='top', fontsize=14.5, fontweight='bold')
    fig.text(0.5, 0.970, "SECTION I: ALPHA BAND (8–13 Hz) TEST BENCHMARK (N = 1,071 TEMPORAL SEQUENCES)",
             ha='center', va='top', fontsize=11.5, fontweight='bold', color='#1f77b4',
             bbox=dict(boxstyle='square,pad=0.3', facecolor='#eef5fb', edgecolor='#1f77b4', lw=1.2))
    fig.text(0.5, 0.490, "SECTION II: BROADBAND (0.5–40 Hz) TEST BENCHMARK (N = 364 TEMPORAL SEQUENCES)",
             ha='center', va='top', fontsize=11.5, fontweight='bold', color='#0b5394',
             bbox=dict(boxstyle='square,pad=0.3', facecolor='#f0f4f8', edgecolor='#0b5394', lw=1.2))

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.965])
    plt.subplots_adjust(hspace=0.28, wspace=0.15)

    out_file_4x3 = os.path.join(SIX_MODEL_DIR, "Figure_Combined_12_Model_Confusion_Matrices_4x3_Elsevier_Blue.png")
    plt.savefig(out_file_4x3, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_FIG_DIR, "Figure_Combined_12_Model_Confusion_Matrices_4x3_Elsevier_Blue.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_ROOT_DIR, "Figure_Combined_12_Model_Confusion_Matrices_4x3_Elsevier_Blue.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved Layout A (4x3) to: {out_file_4x3}")
    return out_file_4x3

# --------------------------------------------------------------------------------------
# 2. Layout B: 2x6 Grid (2 Rows [Bands] x 6 Columns [Architectures]) - 300 & 600 DPI
# --------------------------------------------------------------------------------------
def render_layout_2x6(alpha_data, broad_data):
    print("[2/3] Rendering Layout B: 2x6 Grid (Row 1: Alpha Band, Row 2: BroadBand)...")
    fig, axes = plt.subplots(2, 6, figsize=(33.0, 11.5), dpi=300)

    # Row 0: Alpha Band (8-13 Hz)
    for c_idx, (m_key, disp) in enumerate(MODELS):
        ax = axes[0, c_idx]
        da = alpha_data[m_key]
        cm_a = da['cm']
        cm_a_norm = da['cm_norm']

        annot_a = np.empty_like(cm_a, dtype=object)
        for i in range(5):
            for j in range(5):
                annot_a[i, j] = f"{cm_a_norm[i, j]:.1f}%\n({cm_a[i, j]})"

        sns.heatmap(
            cm_a_norm, annot=annot_a, fmt="", cmap="Blues",
            cbar=True if c_idx == 5 else False,
            cbar_kws={'label': 'Recall (%)'} if c_idx == 5 else None,
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
            annot_kws={"fontsize": 8.5, "weight": "bold"},
            ax=ax, linewidths=0.9, linecolor="#dddddd", vmin=0.0, vmax=100.0
        )

        if 'STATMR' in m_key:
            for _, s in ax.spines.items():
                s.set_color('#0b5394')
                s.set_linewidth(3.0)
            ax.set_title(f"Alpha: {disp} (Proposed)\nAcc: {da['acc']:.2f}% | Bal: {da['bal_acc']:.2f}% | F1: {da['f1']:.2f}%", color='#0b5394', pad=8, fontsize=10.5, fontweight='bold')
        else:
            ax.set_title(f"Alpha: {disp}\nAcc: {da['acc']:.2f}% | Bal: {da['bal_acc']:.2f}% | F1: {da['f1']:.2f}%", pad=8, fontsize=10.5)

        ax.set_xlabel("", labelpad=5)
        ax.set_ylabel("True Clinical Stage" if c_idx == 0 else "", labelpad=6, fontsize=11.0, fontweight='bold')
        ax.tick_params(axis='x', rotation=25)
        ax.tick_params(axis='y', rotation=0)

    # Row 1: BroadBand (0.5-40 Hz)
    for c_idx, (m_key, disp) in enumerate(MODELS):
        ax = axes[1, c_idx]
        db = broad_data[m_key]
        cm_b = db['cm']
        cm_b_norm = db['cm_norm']

        annot_b = np.empty_like(cm_b, dtype=object)
        for i in range(5):
            for j in range(5):
                annot_b[i, j] = f"{cm_b_norm[i, j]:.1f}%\n({cm_b[i, j]})"

        sns.heatmap(
            cm_b_norm, annot=annot_b, fmt="", cmap="Blues",
            cbar=True if c_idx == 5 else False,
            cbar_kws={'label': 'Recall (%)'} if c_idx == 5 else None,
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
            annot_kws={"fontsize": 8.5, "weight": "bold"},
            ax=ax, linewidths=0.9, linecolor="#dddddd", vmin=0.0, vmax=100.0
        )

        if 'STATMR' in m_key:
            for _, s in ax.spines.items():
                s.set_color('#0b5394')
                s.set_linewidth(3.0)
            ax.set_title(f"BroadBand: {disp} (Proposed)\nAcc: {db['acc']:.2f}% | Bal: {db['bal_acc']:.2f}% | F1: {db['f1']:.2f}%", color='#0b5394', pad=8, fontsize=10.5, fontweight='bold')
        else:
            ax.set_title(f"BroadBand: {disp}\nAcc: {db['acc']:.2f}% | Bal: {db['bal_acc']:.2f}% | F1: {db['f1']:.2f}%", pad=8, fontsize=10.5)

        ax.set_xlabel("Predicted Clinical Stage", labelpad=6, fontsize=11.0, fontweight='bold')
        ax.set_ylabel("True Clinical Stage" if c_idx == 0 else "", labelpad=6, fontsize=11.0, fontweight='bold')
        ax.tick_params(axis='x', rotation=25)
        ax.tick_params(axis='y', rotation=0)

    # Row Identification Labels on Far Left
    fig.text(0.005, 0.72, "ROW 1: FOCUSED ALPHA BAND (8–13 Hz)\n(N = 1,071 Temporal Sequences)",
             ha='left', va='center', rotation=90, fontsize=12.0, fontweight='bold', color='#1f77b4',
             bbox=dict(boxstyle='square,pad=0.4', facecolor='#eef5fb', edgecolor='#1f77b4', lw=1.2))
    fig.text(0.005, 0.25, "ROW 2: BROADBAND EEG (0.5–40 Hz)\n(N = 364 Temporal Sequences)",
             ha='left', va='center', rotation=90, fontsize=12.0, fontweight='bold', color='#0b5394',
             bbox=dict(boxstyle='square,pad=0.4', facecolor='#f0f4f8', edgecolor='#0b5394', lw=1.2))

    plt.suptitle("Twelve-Model 2×6 Comparative Contingency Benchmark: Alpha Band (8–13 Hz) vs. BroadBand (0.5–40 Hz)", fontsize=15.5, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.02, 0.01, 0.99, 0.965])
    plt.subplots_adjust(hspace=0.28, wspace=0.16)

    # 1. Save 300 DPI Version
    out_file_2x6_300 = os.path.join(SIX_MODEL_DIR, "Figure_Combined_12_Model_Confusion_Matrices_2x6_Elsevier_Blue.png")
    plt.savefig(out_file_2x6_300, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_FIG_DIR, "Figure_Combined_12_Model_Confusion_Matrices_2x6_Elsevier_Blue.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_ROOT_DIR, "Figure_Combined_12_Model_Confusion_Matrices_2x6_Elsevier_Blue.png"), dpi=300, bbox_inches='tight')

    # 2. Save 600 DPI Version (Ultra-High Resolution)
    out_file_2x6_600 = os.path.join(SIX_MODEL_DIR, "Figure_Combined_12_Model_Confusion_Matrices_2x6_Elsevier_Blue_600DPI.png")
    plt.savefig(out_file_2x6_600, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved Layout B (2x6 300 DPI) to: {out_file_2x6_300}")
    print(f"  [OK] Saved Layout B (2x6 600 DPI) to: {out_file_2x6_600}")
    return out_file_2x6_300, out_file_2x6_600

def main():
    print("=" * 85)
    print("GENERATING COMBINED 12-MODEL CONFUSION MATRIX MASTER SUITES")
    print("=" * 85)
    alpha_data, broad_data = load_matrices()
    f1 = render_layout_4x3(alpha_data, broad_data)
    f2_300, f2_600 = render_layout_2x6(alpha_data, broad_data)
    print("\n" + "=" * 85)
    print("COMBINED 12-MODEL SUITES SUCCESSFULLY GENERATED!")
    print(f"  1. Stacked 4x3:       {f1}")
    print(f"  2. Primary 2x6 (300 DPI): {f2_300}")
    print(f"  3. Ultra 2x6 (600 DPI):   {f2_600}")
    print("=" * 85)
    print("=" * 85)

if __name__ == "__main__":
    main()
