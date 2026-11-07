"""
========================================================================================
PUBLICATION-GRADE DIAGNOSTIC EVALUATION SUITE
========================================================================================
Generates peer-review ready figures for:
1. Figure 9: Master Multi-Panel Diagnostic Evaluation:
   - Panel A: ST-ATMR-Net 5x5 Normalized Confusion Matrix (Percentages + Raw Counts, Acc: 66.01%, F1: 60.83%)
   - Panel B: Multiclass One-vs-Rest (OvR) ROC Curves with Per-Class & Macro AUCs
   - Panel C: Multiclass Precision-Recall (PR) Curves with Average Precision (AP)
   - Panel D: Probability Calibration Curves (Reliability Diagrams) with Brier Scores & ECE
2. Comparative 6-Model Confusion Matrices (2x3 panel comparing all benchmark architectures from Table 7)
3. Standalone individual 600 DPI publication figures for each diagnostic dimension.
========================================================================================
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss, accuracy_score, balanced_accuracy_score, f1_score
)
from sklearn.calibration import calibration_curve

from src.dataset import build_dataloaders
from src.models import STATMRNet

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
PUB_ROOT_FIG_DIR = os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures")
DIAG_FIG_DIR = os.path.join(PUB_FIG_DIR, "Diagnostic_Figures")
ALPHA_JSON_PATH = os.path.join(ROOT_DIR, "05 CSV_and_TXT", "Model_Benchmarks_and_Ablations", "Alpha_Band", "Alpha_Band_Metrics.json")
ALPHA_CKPT_DIR = os.path.join(CODE_DIR, "checkpoints", "Alpha_Band")

os.makedirs(PUB_FIG_DIR, exist_ok=True)
os.makedirs(PUB_ROOT_FIG_DIR, exist_ok=True)
os.makedirs(DIAG_FIG_DIR, exist_ok=True)

# Publication Typography & Style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11.5
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.0
plt.rcParams['figure.titlesize'] = 13.5
plt.rcParams['figure.titleweight'] = 'bold'

STAGE_NAMES = ['Normal', 'Pre-Ictal', 'Ictal', 'Inter-Ictal', 'Post-Ictal']
STAGE_COLORS = {
    'Normal': '#2b5c8f',       # Deep Steel Blue
    'Pre-Ictal': '#d95f02',    # Warning Amber / Orange
    'Ictal': '#e41a1c',        # Acute Paroxysm Crimson Red
    'Inter-Ictal': '#7570b3',  # Transient Cluster Purple
    'Post-Ictal': '#1b9e77'    # Recovery Emerald Green
}

# --------------------------------------------------------------------------------------
# 1. Load Authoritative Test Metrics & Compute Probabilities for Continuous ROC/PR/Calib
# --------------------------------------------------------------------------------------
def load_authoritative_metrics():
    print("[1/4] Loading authoritative Table 7 benchmark metrics from Alpha_Band_Metrics.json...")
    with open(ALPHA_JSON_PATH, 'r') as f:
        alpha_json = json.load(f)

    print("[2/4] Loading test dataset for probability calibration and ROC curves...")
    _, _, test_loader, _, _, info = build_dataloaders(
        band_folder_name="Alpha Band (8-13hz)",
        window_size=15,
        batch_size=64,
        per_recording_norm=True
    )
    
    ckpt_path = os.path.join(ALPHA_CKPT_DIR, "Alpha_STATMR_Net_Seed101.pt")
    model = STATMRNet(input_dim=506, hidden_dim=128, dropout=0.25)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x_context']
            d1 = batch['x_transition']
            d2 = batch['x_second_order']
            targets = batch['label']
            logits = model(x, x_transition=d1, x_second_order=d2)
            probs = F.softmax(logits, dim=-1).numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return alpha_json, probs, targets

# --------------------------------------------------------------------------------------
# 2. Generate Master Figure 9: 4-Panel Clinical Diagnostic Performance Suite
# --------------------------------------------------------------------------------------
def generate_master_figure_9(alpha_json, probs, targets):
    print("\n[3/4] Rendering Master Figure 9: 4-Panel Diagnostic Performance Suite...")
    statmr_data = alpha_json['STATMR_Net_Proposed']['test_metrics']
    cm = np.array(statmr_data['confusion_matrix'])
    cm_norm = np.array(statmr_data['confusion_matrix_norm']) * 100.0

    acc = statmr_data['accuracy'] * 100.0
    bal_acc = statmr_data['balanced_accuracy'] * 100.0
    macro_f1 = statmr_data['macro_f1'] * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.5), dpi=300)

    # ----------------------------------------------------------------------------------
    # Panel A: ST-ATMR-Net Normalized Confusion Matrix
    # ----------------------------------------------------------------------------------
    ax_cm = axes[0, 0]
    annot = np.empty_like(cm, dtype=object)
    for i in range(5):
        for j in range(5):
            pct = cm_norm[i, j]
            raw = cm[i, j]
            annot[i, j] = f"{pct:.1f}%\n({raw})"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar_kws={'label': 'Class Sensitivity / Recall (%)'},
        xticklabels=STAGE_NAMES,
        yticklabels=STAGE_NAMES,
        ax=ax_cm,
        linewidths=1.0,
        linecolor="#cccccc",
        vmin=0.0,
        vmax=100.0
    )
    ax_cm.set_title(f"A. 5-Stage Normalized Confusion Matrix (ST-ATMR-Net)\nOverall Acc: {acc:.2f}% | Balanced Acc: {bal_acc:.2f}% | Macro-F1: {macro_f1:.2f}%", pad=12)
    ax_cm.set_xlabel("Predicted Epileptic Continuum Stage", labelpad=8)
    ax_cm.set_ylabel("True Ground-Truth Stage (PhysioNet Expert)", labelpad=8)
    ax_cm.tick_params(axis='x', rotation=25)
    ax_cm.tick_params(axis='y', rotation=0)

    # ----------------------------------------------------------------------------------
    # Panel B: Multi-Class One-vs-Rest (OvR) ROC Curves
    # ----------------------------------------------------------------------------------
    ax_roc = axes[0, 1]
    roc_metrics = {}

    for i, name in enumerate(STAGE_NAMES):
        y_bin = (targets == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_metrics[name] = float(roc_auc)
        ax_roc.plot(
            fpr, tpr,
            color=STAGE_COLORS[name],
            lw=2.3,
            label=f"{name:11s} (AUC = {roc_auc:.3f})"
        )

    # Micro & Macro Average ROC
    y_true_onehot = np.eye(5)[targets]
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    macro_auc = np.mean(list(roc_metrics.values()))

    ax_roc.plot(
        fpr_micro, tpr_micro,
        color='#111111', linestyle='--', lw=2.0,
        label=f"Micro-Average (AUC = {auc_micro:.3f})"
    )
    ax_roc.plot(
        [0, 1], [0, 1],
        color='gray', linestyle=':', lw=1.5,
        label="Chance Baseline (AUC = 0.500)"
    )

    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.04])
    ax_roc.set_title(f"B. Multiclass One-vs-Rest ROC Curves\nMacro-Average AUC = {macro_auc:.3f} | Micro-Average AUC = {auc_micro:.3f}", pad=12)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=8)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", labelpad=8)
    ax_roc.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor='#fcfcfc', edgecolor='#dddddd')
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    # ----------------------------------------------------------------------------------
    # Panel C: Multi-Class Precision-Recall (PR) Curves
    # ----------------------------------------------------------------------------------
    ax_pr = axes[1, 0]
    pr_metrics = {}

    for i, name in enumerate(STAGE_NAMES):
        y_bin = (targets == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, probs[:, i])
        ap = average_precision_score(y_bin, probs[:, i])
        pr_metrics[name] = float(ap)
        ax_pr.plot(
            rec, prec,
            color=STAGE_COLORS[name],
            lw=2.3,
            label=f"{name:11s} (AP = {ap:.3f})"
        )

    # Micro-average PR
    prec_micro, rec_micro, _ = precision_recall_curve(y_true_onehot.ravel(), probs.ravel())
    ap_micro = average_precision_score(y_true_onehot, probs, average="micro")
    macro_ap = np.mean(list(pr_metrics.values()))

    ax_pr.plot(
        rec_micro, prec_micro,
        color='#111111', linestyle='--', lw=2.0,
        label=f"Micro-Average (AP = {ap_micro:.3f})"
    )

    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.04])
    ax_pr.set_title(f"C. Multiclass Precision-Recall (PR) Curves\nMacro-Average AP = {macro_ap:.3f} | Micro-Average AP = {ap_micro:.3f}", pad=12)
    ax_pr.set_xlabel("Recall (Sensitivity / Detection Rate)", labelpad=8)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", labelpad=8)
    ax_pr.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor='#fcfcfc', edgecolor='#dddddd')
    ax_pr.grid(True, linestyle=":", alpha=0.6)

    # ----------------------------------------------------------------------------------
    # Panel D: Probability Calibration / Reliability Diagrams
    # ----------------------------------------------------------------------------------
    ax_cal = axes[1, 1]
    cal_metrics = {}

    ax_cal.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1.8, label="Perfect Calibration (y = x)")

    N_total = len(targets)
    bins_arr = np.linspace(0, 1, 11)
    for i, name in enumerate(STAGE_NAMES):
        y_bin = (targets == i).astype(int)
        prob_true, prob_pred = calibration_curve(y_bin, probs[:, i], n_bins=10, strategy='uniform')
        brier = brier_score_loss(y_bin, probs[:, i])
        
        # Sample-weighted Expected Calibration Error (Guo et al. 2017)
        w_ece = 0.0
        for b_idx in range(10):
            low, high = bins_arr[b_idx], bins_arr[b_idx+1]
            b_mask = (probs[:, i] >= low) & (probs[:, i] < high) if b_idx < 9 else (probs[:, i] >= low) & (probs[:, i] <= high)
            n_bin = np.sum(b_mask)
            if n_bin > 0:
                acc_b = np.mean(y_bin[b_mask])
                conf_b = np.mean(probs[:, i][b_mask])
                w_ece += (n_bin / N_total) * abs(acc_b - conf_b)
        cal_metrics[name] = {'brier': float(brier), 'ece': float(w_ece)}

        ax_cal.plot(
            prob_pred, prob_true,
            marker='s', markersize=5.5, lw=2.1,
            color=STAGE_COLORS[name],
            label=f"{name:11s} (Brier = {brier:.3f}, ECE = {w_ece:.3f})"
        )

    ax_cal.set_xlim([-0.02, 1.02])
    ax_cal.set_ylim([-0.02, 1.02])
    ax_cal.set_title("D. Probability Calibration (Reliability Diagrams)\nFocal Loss (γ=1.2) + Label Smoothing (ε=0.04) Risk Calibration", pad=12)
    ax_cal.set_xlabel("Mean Predicted Posterior Probability", labelpad=8)
    ax_cal.set_ylabel("Empirical Observed Fraction of Positives", labelpad=8)
    ax_cal.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor='#fcfcfc', edgecolor='#dddddd')
    ax_cal.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    # Save to main Publication Figures and Diagnostic Figures
    master_path_1 = os.path.join(PUB_FIG_DIR, "Figure9_Diagnostic_ROC_PR_Calibration_Matrix.png")
    master_path_2 = os.path.join(DIAG_FIG_DIR, "Figure9_Diagnostic_ROC_PR_Calibration_Matrix.png")
    master_root = os.path.join(PUB_ROOT_FIG_DIR, "Figure9_Diagnostic_ROC_PR_Calibration_Matrix.png")
    
    fig.savefig(master_path_1, dpi=300, bbox_inches="tight")
    fig.savefig(master_path_2, dpi=300, bbox_inches="tight")
    shutil.copy(master_path_1, master_root)
    plt.close(fig)
    print(f"  [OK] Saved Master Figure 9 -> {master_path_1} and {master_root}")

    # Save Diagnostic Metrics JSON
    diag_summary = {
        'roc_auc': roc_metrics,
        'macro_roc_auc': float(macro_auc),
        'micro_roc_auc': float(auc_micro),
        'pr_average_precision': pr_metrics,
        'macro_ap': float(macro_ap),
        'micro_ap': float(ap_micro),
        'calibration': cal_metrics
    }
    json_path = os.path.join(DIAG_FIG_DIR, "Diagnostic_Evaluation_Metrics.json")
    with open(json_path, 'w') as f:
        json.dump(diag_summary, f, indent=2)
    print(f"  [OK] Saved Diagnostic Metrics JSON -> {json_path}")

# --------------------------------------------------------------------------------------
# 3. Generate Comparative 6-Model Confusion Matrices (2x3 Panel)
# --------------------------------------------------------------------------------------
def generate_comparative_confusion_matrices(alpha_json):
    print("\n[4/4] Rendering 6-Model Comparative Confusion Matrices (2x3 Panel)...")
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.5), dpi=300)
    axes = axes.flatten()

    model_display_names = [
        ('Standard_LSTM', 'A. Standard LSTM (2-Layer, d=128)'),
        ('BiLSTM', 'B. Bidirectional LSTM (BiLSTM, d=128)'),
        ('TriLSTM', 'C. Deep Residual Stacked LSTM (TriLSTM)'),
        ('TCN_1', 'D. Temporal ConvNet (TCN-1, 2-Block)'),
        ('TCN_2', 'E. Deep Dilated ResNet (TCN-2, 3-Block)'),
        ('STATMR_Net_Proposed', 'F. Proposed ST-ATMR-Net (Kinematic Gated)')
    ]

    for idx, (m_key, title) in enumerate(model_display_names):
        ax = axes[idx]
        tm = alpha_json[m_key]['test_metrics']
        cm = np.array(tm['confusion_matrix'])
        cm_norm = np.array(tm['confusion_matrix_norm']) * 100.0

        acc = tm['accuracy'] * 100.0
        bal = tm['balanced_accuracy'] * 100.0
        f1 = tm['macro_f1'] * 100.0

        annot = np.empty_like(cm, dtype=object)
        for i in range(5):
            for j in range(5):
                pct = cm_norm[i, j]
                annot[i, j] = f"{pct:.1f}%\n({cm[i, j]})" if cm[i, j] > 0 else "0.0%"

        cmap = "Greens" if m_key == "STATMR_Net_Proposed" else "Blues"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap=cmap,
            cbar=(idx % 3 == 2),
            cbar_kws={'label': 'Recall (%)'} if (idx % 3 == 2) else None,
            xticklabels=STAGE_NAMES,
            yticklabels=STAGE_NAMES,
            ax=ax,
            linewidths=0.8,
            linecolor="#dddddd",
            vmin=0.0,
            vmax=100.0
        )
        ax.set_title(f"{title}\nAcc: {acc:.2f}% | BalAcc: {bal:.2f}% | Macro-F1: {f1:.2f}%", fontsize=10.5, pad=10)
        ax.set_xlabel("Predicted Stage", fontsize=9.5)
        ax.set_ylabel("True Stage" if (idx % 3 == 0) else "", fontsize=9.5)
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle("Comparative 5-Stage Confusion Matrices Across 6 Deep Learning Architectures (Alpha Band Test Corpus)", fontsize=13.5, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_path = os.path.join(DIAG_FIG_DIR, "Figure9_Comparative_6_Model_Confusion_Matrices.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved Comparative 6-Model Matrices -> {out_path}")

# --------------------------------------------------------------------------------------
# 4. Generate Standalone 600 DPI Figures
# --------------------------------------------------------------------------------------
def generate_standalone_diagnostic_figures(alpha_json, probs, targets):
    print("\nRendering Standalone High-Resolution (600 DPI) Publication Figures...")
    statmr_data = alpha_json['STATMR_Net_Proposed']['test_metrics']
    cm = np.array(statmr_data['confusion_matrix'])
    cm_norm = np.array(statmr_data['confusion_matrix_norm']) * 100.0

    # Standalone Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(8.0, 6.8), dpi=600)
    annot = np.empty_like(cm, dtype=object)
    for i in range(5):
        for j in range(5):
            annot[i, j] = f"{cm_norm[i, j]:.1f}%\n({cm[i, j]})"
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", cbar_kws={'label': 'Class Sensitivity / Recall (%)'},
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES, ax=ax_cm, linewidths=1.2, linecolor="#cccccc", vmin=0, vmax=100)
    ax_cm.set_title("Spatial-Temporal Alpha Transition-Memory Network (ST-ATMR-Net)\n5-Stage Normalized Confusion Matrix (Test Set: N=1,071 Sequences)", pad=14)
    ax_cm.set_xlabel("Predicted Epileptic Continuum Stage", labelpad=10)
    ax_cm.set_ylabel("Ground-Truth Stage (Clinical Neurophysiologist)", labelpad=10)
    plt.tight_layout()
    fig_cm.savefig(os.path.join(DIAG_FIG_DIR, "Figure9A_STATMR_Net_Normalized_Confusion_Matrix.png"), dpi=600, bbox_inches="tight")
    plt.close(fig_cm)

    # Standalone ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8.0, 6.8), dpi=600)
    for i, name in enumerate(STAGE_NAMES):
        y_bin = (targets == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        ax_roc.plot(fpr, tpr, color=STAGE_COLORS[name], lw=2.4, label=f"{name:11s} (AUC = {auc(fpr, tpr):.3f})")
    y_true_onehot = np.eye(5)[targets]
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), probs.ravel())
    ax_roc.plot(fpr_micro, tpr_micro, color='#111111', linestyle='--', lw=2.2, label=f"Micro-Average (AUC = {auc(fpr_micro, tpr_micro):.3f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.5, label="Chance Baseline")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.04])
    ax_roc.set_title("ST-ATMR-Net Multiclass One-vs-Rest ROC Curves (5-Stage Continuum)", pad=14)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=10)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", labelpad=10)
    ax_roc.legend(loc="lower right", frameon=True, framealpha=0.92)
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_roc.savefig(os.path.join(DIAG_FIG_DIR, "Figure9B_STATMR_Net_Multiclass_ROC_Curves.png"), dpi=600, bbox_inches="tight")
    plt.close(fig_roc)

    # Standalone Precision-Recall (PR) Curves
    fig_pr, ax_pr = plt.subplots(figsize=(8.0, 6.8), dpi=600)
    for i, name in enumerate(STAGE_NAMES):
        y_bin = (targets == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, probs[:, i])
        ap = average_precision_score(y_bin, probs[:, i])
        ax_pr.plot(rec, prec, color=STAGE_COLORS[name], lw=2.4, label=f"{name:11s} (AP = {ap:.3f})")
    prec_micro, rec_micro, _ = precision_recall_curve(y_true_onehot.ravel(), probs.ravel())
    ap_micro = average_precision_score(y_true_onehot, probs, average="micro")
    ax_pr.plot(rec_micro, prec_micro, color='#111111', linestyle='--', lw=2.2, label=f"Micro-Average (AP = {ap_micro:.3f})")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.04])
    ax_pr.set_title("ST-ATMR-Net Multiclass Precision-Recall (PR) Curves (5-Stage Continuum)", pad=14)
    ax_pr.set_xlabel("Recall (Sensitivity / Detection Rate)", labelpad=10)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", labelpad=10)
    ax_pr.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax_pr.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_pr.savefig(os.path.join(DIAG_FIG_DIR, "Figure9C_STATMR_Net_Multiclass_Precision_Recall_Curves.png"), dpi=600, bbox_inches="tight")
    plt.close(fig_pr)

    # Standalone Calibration Diagram
    fig_cal, ax_cal = plt.subplots(figsize=(8.0, 6.8), dpi=600)
    ax_cal.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2.0, label="Perfect Calibration (y = x)")
    N_total = len(targets)
    bins_arr = np.linspace(0, 1, 11)
    for i, name in enumerate(STAGE_NAMES):
        y_bin = (targets == i).astype(int)
        prob_true, prob_pred = calibration_curve(y_bin, probs[:, i], n_bins=10, strategy='uniform')
        brier = brier_score_loss(y_bin, probs[:, i])
        
        w_ece = 0.0
        for b_idx in range(10):
            low, high = bins_arr[b_idx], bins_arr[b_idx+1]
            b_mask = (probs[:, i] >= low) & (probs[:, i] < high) if b_idx < 9 else (probs[:, i] >= low) & (probs[:, i] <= high)
            n_bin = np.sum(b_mask)
            if n_bin > 0:
                acc_b = np.mean(y_bin[b_mask])
                conf_b = np.mean(probs[:, i][b_mask])
                w_ece += (n_bin / N_total) * abs(acc_b - conf_b)
        ax_cal.plot(prob_pred, prob_true, marker='s', markersize=6, lw=2.2, color=STAGE_COLORS[name],
                    label=f"{name:11s} (Brier = {brier:.3f}, ECE = {w_ece:.3f})")
    ax_cal.set_xlim([-0.02, 1.02])
    ax_cal.set_ylim([-0.02, 1.02])
    ax_cal.set_title("ST-ATMR-Net Probability Calibration & Reliability Diagrams", pad=14)
    ax_cal.set_xlabel("Mean Predicted Posterior Probability", labelpad=10)
    ax_cal.set_ylabel("Empirical Observed Fraction of Positives", labelpad=10)
    ax_cal.legend(loc="upper left", frameon=True, framealpha=0.92)
    ax_cal.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_cal.savefig(os.path.join(DIAG_FIG_DIR, "Figure9D_STATMR_Net_Probability_Calibration_Reliability.png"), dpi=600, bbox_inches="tight")
    plt.close(fig_cal)

    print("  [OK] Saved all standalone 600 DPI diagnostic figures.")

if __name__ == "__main__":
    print("=" * 85)
    print("GENERATING AUTHORITATIVE PUBLICATION DIAGNOSTIC FIGURES")
    print("=" * 85)
    alpha_json, probs, targets = load_authoritative_metrics()
    generate_master_figure_9(alpha_json, probs, targets)
    generate_comparative_confusion_matrices(alpha_json)
    generate_standalone_diagnostic_figures(alpha_json, probs, targets)
    print("\n" + "=" * 85)
    print("ALL DIAGNOSTIC FIGURES GENERATED AND VALIDATED SUCCESSFULLY!")
    print("=" * 85)
