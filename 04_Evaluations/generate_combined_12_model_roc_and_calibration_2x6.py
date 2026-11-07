"""
========================================================================================
COMBINED 12-MODEL 2x6 SUITE: MULTICLASS ROC & PROBABILITY CALIBRATION
========================================================================================
Generates publication-grade, Elsevier-compliant 2x6 comparative figures:
1. 2x6 Multiclass One-vs-Rest (OvR) ROC Curves (Row 1: Alpha Band, Row 2: BroadBand)
   - 300 DPI & 600 DPI Ultra-HD versions
2. 2x6 Probability Calibration (Reliability Diagrams) with Sample-Weighted ECE & Brier Score
   - 300 DPI & 600 DPI Ultra-HD versions
3. 2x6 Precision-Recall (PR) Curves (Row 1: Alpha Band, Row 2: BroadBand)
   - 300 DPI & 600 DPI Ultra-HD versions
All subplots formatted with Elsevier-standard colorblind-safe palettes and thick navy borders
highlighting the proposed ST-ATMR-Net architecture.
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
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

from src.dataset import build_dataloaders
from src.models import StandardLSTM, BiLSTM, TriLSTM, TCN1, TCN2, STATMRNet

# --------------------------------------------------------------------------------------
# Directories & Elsevier Typography Configuration
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

os.makedirs(PUB_FIG_DIR, exist_ok=True)
os.makedirs(PUB_ROOT_DIR, exist_ok=True)
os.makedirs(SIX_MODEL_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.titlesize'] = 11.0
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10.0
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 9.0
plt.rcParams['ytick.labelsize'] = 9.0
plt.rcParams['legend.fontsize'] = 8.0
plt.rcParams['figure.titlesize'] = 15.5
plt.rcParams['figure.titleweight'] = 'bold'

STAGE_NAMES = ['Normal', 'Pre-Ictal', 'Ictal', 'Inter-Ictal', 'Post-Ictal']

ELSEVIER_STAGE_COLORS = {
    'Normal': '#1f77b4',       # Elsevier Classic Deep Blue
    'Pre-Ictal': '#ff7f0e',    # High-Visibility Warning Amber
    'Ictal': '#d62728',        # Acute Paroxysm Crimson Red
    'Inter-Ictal': '#9467bd',  # Clustered Transition Purple
    'Post-Ictal': '#2ca02c'    # Recovery Emerald Green
}

MODEL_METADATA = [
    ('Standard_LSTM', 'Standard LSTM', lambda: StandardLSTM(input_dim=506, hidden_dim=128, dropout=0.3)),
    ('BiLSTM', 'BiLSTM', lambda: BiLSTM(input_dim=506, hidden_dim=128, dropout=0.3)),
    ('TriLSTM', 'TriLSTM', lambda: TriLSTM(input_dim=506, hidden_dim=128, dropout=0.3)),
    ('TCN_1', 'TCN-1', lambda: TCN1(input_dim=506, num_channels=[128, 128], dropout=0.3)),
    ('TCN_2', 'TCN-2', lambda: TCN2(input_dim=506, num_channels=[128, 128, 128], dropout=0.3)),
    ('STATMR_Net', 'ST-ATMR-Net (Proposed)', lambda: STATMRNet(num_channels=23, feats_per_ch=22, embed_dim=48, hidden_dim=128, dropout=0.25))
]

# --------------------------------------------------------------------------------------
# 1. Batch Inference Engine Across Both Bands
# --------------------------------------------------------------------------------------
def evaluate_band(band_title, band_folder, prefix, ckpt_dir):
    print(f"\nEvaluating 6 models for {band_title}...")
    _, _, test_loader, _, _, info = build_dataloaders(
        band_folder_name=band_folder,
        window_size=15,
        batch_size=64,
        per_recording_norm=True
    )
    print(f"  Test cohort: {info['num_test_sequences']} sequences.")

    results = {}
    for m_key, disp_name, factory in MODEL_METADATA:
        fname = f"{prefix}{m_key}.pt" if m_key != 'STATMR_Net' else f"{prefix}STATMR_Net_Seed101.pt"
        fpath = os.path.join(ckpt_dir, fname)
        if not os.path.exists(fpath):
            alt_path = os.path.join(CODE_DIR, "checkpoints", "BroadBand" if "Broad" in band_title else "Alpha_Band", fname)
            if os.path.exists(alt_path):
                fpath = alt_path

        ckpt = torch.load(fpath, map_location="cpu")
        model = factory()
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        all_probs, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x_context']
                targets = batch['label']
                if 'STATMR' in m_key:
                    d1 = batch['x_transition']
                    d2 = batch['x_second_order']
                    logits = model(x, x_transition=d1, x_second_order=d2)
                else:
                    logits = model(x)
                probs = F.softmax(logits, dim=-1).numpy()
                all_probs.append(probs)
                all_targets.append(targets.numpy())

        probs = np.concatenate(all_probs, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        results[m_key] = {
            'disp_name': disp_name,
            'probs': probs,
            'targets': targets
        }
        print(f"    [OK] {disp_name:<24s} evaluated.")

    return results

# --------------------------------------------------------------------------------------
# 2. Render 2x6 Multiclass ROC Curves Suite (300 & 600 DPI)
# --------------------------------------------------------------------------------------
def render_2x6_roc_suite(alpha_res, broad_res):
    print("\n[Rendering] 2x6 Multiclass ROC Curves Suite (Row 1: Alpha, Row 2: BroadBand)...")
    fig, axes = plt.subplots(2, 6, figsize=(33.0, 11.5), dpi=300)

    for row_idx, (b_title, b_res, prefix_label) in enumerate([("Alpha Band", alpha_res, "Alpha"), ("BroadBand", broad_res, "BroadBand")]):
        for col_idx, (m_key, disp, _) in enumerate(MODEL_METADATA):
            ax = axes[row_idx, col_idx]
            data = b_res[m_key]
            probs = data['probs']
            targets = data['targets']

            roc_dict = {}
            for i, name in enumerate(STAGE_NAMES):
                y_bin = (targets == i).astype(int)
                pos = np.sum(y_bin)
                if pos > 0:
                    fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_dict[name] = float(roc_auc)
                    ax.plot(
                        fpr, tpr,
                        color=ELSEVIER_STAGE_COLORS[name],
                        lw=2.1,
                        label=f"{name:11s} (AUC = {roc_auc:.3f})"
                    )
                else:
                    ax.plot([], [], color=ELSEVIER_STAGE_COLORS[name], linestyle=':', lw=1.5, label=f"{name:11s} (AUC = N/A, Supp=0)")

            active_classes = [i for i in range(5) if np.sum(targets == i) > 0]
            y_true_active = np.zeros((len(targets), len(active_classes)))
            for idx_c, c in enumerate(active_classes):
                y_true_active[:, idx_c] = (targets == c).astype(int)
            probs_active = probs[:, active_classes]

            fpr_micro, tpr_micro, _ = roc_curve(y_true_active.ravel(), probs_active.ravel())
            auc_micro = auc(fpr_micro, tpr_micro)
            macro_auc = np.mean(list(roc_dict.values()))

            ax.plot(fpr_micro, tpr_micro, color='#000000', linestyle='--', lw=2.0, label=f"Micro-Avg (AUC = {auc_micro:.3f})")
            ax.plot([0, 1], [0, 1], color='#7f7f7f', linestyle=':', lw=1.5, label="Chance (AUC = 0.500)")

            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.04])

            if 'STATMR' in m_key:
                for _, s in ax.spines.items():
                    s.set_color('#0b5394')
                    s.set_linewidth(3.0)
                ax.set_title(f"{prefix_label}: {disp} (Proposed)\nMacro-AUC: {macro_auc:.3f} | Micro-AUC: {auc_micro:.3f}", color='#0b5394', pad=8, fontsize=10.5, fontweight='bold')
            else:
                ax.set_title(f"{prefix_label}: {disp}\nMacro-AUC: {macro_auc:.3f} | Micro-AUC: {auc_micro:.3f}", pad=8, fontsize=10.5)

            ax.set_xlabel("False Positive Rate (1 - Specificity)" if row_idx == 1 else "", labelpad=6, fontsize=11.0, fontweight='bold')
            ax.set_ylabel("True Positive Rate (Sensitivity)" if col_idx == 0 else "", labelpad=6, fontsize=11.0, fontweight='bold')
            ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor='#ffffff', edgecolor='#cccccc', fontsize=7.8)
            ax.grid(True, linestyle=":", alpha=0.5)

    # Row Identification Labels on Far Left
    fig.text(0.005, 0.72, "ROW 1: FOCUSED ALPHA BAND (8–13 Hz)\n(N = 1,071 Temporal Sequences)",
             ha='left', va='center', rotation=90, fontsize=12.0, fontweight='bold', color='#1f77b4',
             bbox=dict(boxstyle='square,pad=0.4', facecolor='#eef5fb', edgecolor='#1f77b4', lw=1.2))
    fig.text(0.005, 0.25, "ROW 2: BROADBAND EEG (0.5–40 Hz)\n(N = 364 Temporal Sequences)",
             ha='left', va='center', rotation=90, fontsize=12.0, fontweight='bold', color='#0b5394',
             bbox=dict(boxstyle='square,pad=0.4', facecolor='#f0f4f8', edgecolor='#0b5394', lw=1.2))

    plt.suptitle("Twelve-Model 2×6 Comparative Multiclass One-vs-Rest (OvR) ROC Benchmark: Alpha Band vs. BroadBand", fontsize=15.5, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.02, 0.01, 0.99, 0.965])
    plt.subplots_adjust(hspace=0.28, wspace=0.16)

    out_file_300 = os.path.join(SIX_MODEL_DIR, "Figure_Combined_12_Model_ROC_Curves_2x6_Elsevier_Colors.png")
    out_file_600 = os.path.join(SIX_MODEL_DIR, "Figure_Combined_12_Model_ROC_Curves_2x6_Elsevier_Colors_600DPI.png")

    plt.savefig(out_file_300, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_FIG_DIR, "Figure_Combined_12_Model_ROC_Curves_2x6_Elsevier_Colors.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_ROOT_DIR, "Figure_Combined_12_Model_ROC_Curves_2x6_Elsevier_Colors.png"), dpi=300, bbox_inches='tight')

    plt.savefig(out_file_600, dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_FIG_DIR, "Figure_Combined_12_Model_ROC_Curves_2x6_Elsevier_Colors_600DPI.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_ROOT_DIR, "Figure_Combined_12_Model_ROC_Curves_2x6_Elsevier_Colors_600DPI.png"), dpi=600, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved 2x6 ROC Suite (300 DPI): {out_file_300}")
    print(f"  [OK] Saved 2x6 ROC Suite (600 DPI): {out_file_600}")
    return out_file_300, out_file_600

# --------------------------------------------------------------------------------------
# 3. Render 2x6 Probability Calibration (Reliability Diagrams) Suite (300 & 600 DPI)
# --------------------------------------------------------------------------------------
def render_2x6_calibration_suite(alpha_res, broad_res):
    print("\n[Rendering] 2x6 Probability Calibration Diagrams Suite (Row 1: Alpha, Row 2: BroadBand)...")
    fig, axes = plt.subplots(2, 6, figsize=(33.0, 11.5), dpi=300)
    bins_arr = np.linspace(0, 1, 9)

    for row_idx, (b_title, b_res, prefix_label) in enumerate([("Alpha Band", alpha_res, "Alpha"), ("BroadBand", broad_res, "BroadBand")]):
        for col_idx, (m_key, disp, _) in enumerate(MODEL_METADATA):
            ax = axes[row_idx, col_idx]
            data = b_res[m_key]
            probs = data['probs']
            targets = data['targets']
            N_total = len(targets)

            ax.plot([0, 1], [0, 1], color='#000000', linestyle='--', lw=1.6, label="Perfect Calibration (y = x)")

            brier_list = []
            ece_weighted_list = []

            for i, name in enumerate(STAGE_NAMES):
                y_bin = (targets == i).astype(int)
                pos = np.sum(y_bin)
                brier = brier_score_loss(y_bin, probs[:, i])
                brier_list.append(brier)

                if pos > 0:
                    prob_true, prob_pred = calibration_curve(y_bin, probs[:, i], n_bins=8, strategy='uniform')
                    
                    # Sample-weighted Expected Calibration Error (Guo et al., ICML 2017)
                    w_ece = 0.0
                    for b_idx in range(8):
                        low, high = bins_arr[b_idx], bins_arr[b_idx+1]
                        b_mask = (probs[:, i] >= low) & (probs[:, i] < high) if b_idx < 7 else (probs[:, i] >= low) & (probs[:, i] <= high)
                        n_bin = np.sum(b_mask)
                        if n_bin > 0:
                            acc_b = np.mean(y_bin[b_mask])
                            conf_b = np.mean(probs[:, i][b_mask])
                            w_ece += (n_bin / N_total) * abs(acc_b - conf_b)
                    ece_weighted_list.append(w_ece)

                    ax.plot(
                        prob_pred, prob_true,
                        marker='s', markersize=5.0, lw=1.9,
                        color=ELSEVIER_STAGE_COLORS[name],
                        label=f"{name:11s} (Brier={brier:.3f}, ECE={w_ece:.3f})"
                    )
                else:
                    ax.plot([], [], color=ELSEVIER_STAGE_COLORS[name], linestyle=':', lw=1.5, label=f"{name:11s} (Brier={brier:.3f}, Supp=0)")

            mean_brier = np.mean(brier_list)
            mean_ece = np.mean(ece_weighted_list) if len(ece_weighted_list) > 0 else 0.0

            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])

            if 'STATMR' in m_key:
                for _, s in ax.spines.items():
                    s.set_color('#0b5394')
                    s.set_linewidth(3.0)
                ax.set_title(f"{prefix_label}: {disp} (Proposed)\nMean Brier: {mean_brier:.3f} | Mean Weighted ECE: {mean_ece:.3f}", color='#0b5394', pad=8, fontsize=10.5, fontweight='bold')
            else:
                ax.set_title(f"{prefix_label}: {disp}\nMean Brier: {mean_brier:.3f} | Mean Weighted ECE: {mean_ece:.3f}", pad=8, fontsize=10.5)

            ax.set_xlabel("Predicted Posterior Probability (Confidence)" if row_idx == 1 else "", labelpad=6, fontsize=11.0, fontweight='bold')
            ax.set_ylabel("Empirical True Fraction (Accuracy)" if col_idx == 0 else "", labelpad=6, fontsize=11.0, fontweight='bold')
            ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor='#ffffff', edgecolor='#cccccc', fontsize=7.6)
            ax.grid(True, linestyle=":", alpha=0.5)

    # Row Identification Labels on Far Left
    fig.text(0.005, 0.72, "ROW 1: FOCUSED ALPHA BAND (8–13 Hz)\n(N = 1,071 Temporal Sequences)",
             ha='left', va='center', rotation=90, fontsize=12.0, fontweight='bold', color='#1f77b4',
             bbox=dict(boxstyle='square,pad=0.4', facecolor='#eef5fb', edgecolor='#1f77b4', lw=1.2))
    fig.text(0.005, 0.25, "ROW 2: BROADBAND EEG (0.5–40 Hz)\n(N = 364 Temporal Sequences)",
             ha='left', va='center', rotation=90, fontsize=12.0, fontweight='bold', color='#0b5394',
             bbox=dict(boxstyle='square,pad=0.4', facecolor='#f0f4f8', edgecolor='#0b5394', lw=1.2))

    plt.suptitle("Twelve-Model 2×6 Comparative Probability Calibration (Reliability Diagrams): Alpha Band vs. BroadBand", fontsize=15.5, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.02, 0.01, 0.99, 0.965])
    plt.subplots_adjust(hspace=0.28, wspace=0.16)

    out_file_300 = os.path.join(SIX_MODEL_DIR, "Figure_Combined_12_Model_Probability_Calibration_2x6_Elsevier.png")
    out_file_600 = os.path.join(SIX_MODEL_DIR, "Figure_Combined_12_Model_Probability_Calibration_2x6_Elsevier_600DPI.png")

    plt.savefig(out_file_300, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_FIG_DIR, "Figure_Combined_12_Model_Probability_Calibration_2x6_Elsevier.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_ROOT_DIR, "Figure_Combined_12_Model_Probability_Calibration_2x6_Elsevier.png"), dpi=300, bbox_inches='tight')

    plt.savefig(out_file_600, dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_FIG_DIR, "Figure_Combined_12_Model_Probability_Calibration_2x6_Elsevier_600DPI.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_ROOT_DIR, "Figure_Combined_12_Model_Probability_Calibration_2x6_Elsevier_600DPI.png"), dpi=600, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved 2x6 Calibration Suite (300 DPI): {out_file_300}")
    print(f"  [OK] Saved 2x6 Calibration Suite (600 DPI): {out_file_600}")
    return out_file_300, out_file_600

# --------------------------------------------------------------------------------------
# 4. Render 2x6 Precision-Recall (PR) Curves Suite (300 & 600 DPI)
# --------------------------------------------------------------------------------------
def render_2x6_pr_suite(alpha_res, broad_res):
    print("\n[Rendering] 2x6 Precision-Recall (PR) Curves Suite (Row 1: Alpha, Row 2: BroadBand)...")
    fig, axes = plt.subplots(2, 6, figsize=(33.0, 11.5), dpi=300)

    for row_idx, (b_title, b_res, prefix_label) in enumerate([("Alpha Band", alpha_res, "Alpha"), ("BroadBand", broad_res, "BroadBand")]):
        for col_idx, (m_key, disp, _) in enumerate(MODEL_METADATA):
            ax = axes[row_idx, col_idx]
            data = b_res[m_key]
            probs = data['probs']
            targets = data['targets']

            pr_dict = {}
            for i, name in enumerate(STAGE_NAMES):
                y_bin = (targets == i).astype(int)
                pos = np.sum(y_bin)
                if pos > 0:
                    prec, rec, _ = precision_recall_curve(y_bin, probs[:, i])
                    ap = average_precision_score(y_bin, probs[:, i])
                    pr_dict[name] = float(ap)
                    ax.plot(
                        rec, prec,
                        color=ELSEVIER_STAGE_COLORS[name],
                        lw=2.1,
                        label=f"{name:11s} (AP = {ap:.3f})"
                    )
                else:
                    ax.plot([], [], color=ELSEVIER_STAGE_COLORS[name], linestyle=':', lw=1.5, label=f"{name:11s} (AP = N/A, Supp=0)")

            active_classes = [i for i in range(5) if np.sum(targets == i) > 0]
            y_true_active = np.zeros((len(targets), len(active_classes)))
            for idx_c, c in enumerate(active_classes):
                y_true_active[:, idx_c] = (targets == c).astype(int)
            probs_active = probs[:, active_classes]

            prec_micro, rec_micro, _ = precision_recall_curve(y_true_active.ravel(), probs_active.ravel())
            ap_micro = average_precision_score(y_true_active, probs_active, average="micro")
            macro_ap = np.mean(list(pr_dict.values()))

            ax.plot(rec_micro, prec_micro, color='#000000', linestyle='--', lw=2.0, label=f"Micro-Avg (AP = {ap_micro:.3f})")

            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.04])

            if 'STATMR' in m_key:
                for _, s in ax.spines.items():
                    s.set_color('#0b5394')
                    s.set_linewidth(3.0)
                ax.set_title(f"{prefix_label}: {disp} (Proposed)\nMacro-AP: {macro_ap:.3f} | Micro-AP: {ap_micro:.3f}", color='#0b5394', pad=8, fontsize=10.5, fontweight='bold')
            else:
                ax.set_title(f"{prefix_label}: {disp}\nMacro-AP: {macro_ap:.3f} | Micro-AP: {ap_micro:.3f}", pad=8, fontsize=10.5)

            ax.set_xlabel("Recall (Detection Sensitivity)" if row_idx == 1 else "", labelpad=6, fontsize=11.0, fontweight='bold')
            ax.set_ylabel("Precision (Positive Predictive Value)" if col_idx == 0 else "", labelpad=6, fontsize=11.0, fontweight='bold')
            ax.legend(loc="upper right" if row_idx == 0 else "lower left", frameon=True, framealpha=0.92, facecolor='#ffffff', edgecolor='#cccccc', fontsize=7.8)
            ax.grid(True, linestyle=":", alpha=0.5)

    fig.text(0.005, 0.72, "ROW 1: FOCUSED ALPHA BAND (8–13 Hz)\n(N = 1,071 Temporal Sequences)",
             ha='left', va='center', rotation=90, fontsize=12.0, fontweight='bold', color='#1f77b4',
             bbox=dict(boxstyle='square,pad=0.4', facecolor='#eef5fb', edgecolor='#1f77b4', lw=1.2))
    fig.text(0.005, 0.25, "ROW 2: BROADBAND EEG (0.5–40 Hz)\n(N = 364 Temporal Sequences)",
             ha='left', va='center', rotation=90, fontsize=12.0, fontweight='bold', color='#0b5394',
             bbox=dict(boxstyle='square,pad=0.4', facecolor='#f0f4f8', edgecolor='#0b5394', lw=1.2))

    plt.suptitle("Twelve-Model 2×6 Comparative Precision-Recall (PR) Benchmark: Alpha Band vs. BroadBand", fontsize=15.5, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.02, 0.01, 0.99, 0.965])
    plt.subplots_adjust(hspace=0.28, wspace=0.16)

    out_file_300 = os.path.join(SIX_MODEL_DIR, "Figure_Combined_12_Model_Precision_Recall_2x6_Elsevier.png")
    out_file_600 = os.path.join(SIX_MODEL_DIR, "Figure_Combined_12_Model_Precision_Recall_2x6_Elsevier_600DPI.png")

    plt.savefig(out_file_300, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_FIG_DIR, "Figure_Combined_12_Model_Precision_Recall_2x6_Elsevier.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_ROOT_DIR, "Figure_Combined_12_Model_Precision_Recall_2x6_Elsevier.png"), dpi=300, bbox_inches='tight')

    plt.savefig(out_file_600, dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_FIG_DIR, "Figure_Combined_12_Model_Precision_Recall_2x6_Elsevier_600DPI.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_ROOT_DIR, "Figure_Combined_12_Model_Precision_Recall_2x6_Elsevier_600DPI.png"), dpi=600, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved 2x6 PR Suite (300 DPI): {out_file_300}")
    print(f"  [OK] Saved 2x6 PR Suite (600 DPI): {out_file_600}")
    return out_file_300, out_file_600

def main():
    print("=" * 85)
    print("STARTING COMBINED 12-MODEL 2x6 ROC & CALIBRATION SUITE ENGINE")
    print("=" * 85)

    alpha_ckpt_dir = os.path.join(CODE_DIR, "checkpoints", "Alpha_Band")
    alpha_res = evaluate_band("Alpha Band (8-13 Hz)", "Alpha Band (8-13hz)", "Alpha_", alpha_ckpt_dir)

    broad_ckpt_dir = os.path.join(CODE_DIR, "checkpoints", "BroadBand")
    broad_res = evaluate_band("BroadBand (0.5-40 Hz)", "All Band (0.5-40hz)", "BroadBand_", broad_ckpt_dir)

    # 1. Render 2x6 ROC
    render_2x6_roc_suite(alpha_res, broad_res)

    # 2. Render 2x6 Calibration
    render_2x6_calibration_suite(alpha_res, broad_res)

    # 3. Render 2x6 PR
    render_2x6_pr_suite(alpha_res, broad_res)

    print("\n" + "=" * 85)
    print("ALL 12-MODEL 2x6 SUITES (ROC, CALIBRATION, PR) SUCCESSFULLY GENERATED!")
    print("=" * 85)

if __name__ == "__main__":
    main()
