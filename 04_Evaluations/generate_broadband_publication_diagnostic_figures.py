"""
========================================================================================
PUBLICATION-GRADE DIAGNOSTIC EVALUATION SUITE: BROADBAND (0.5-40 Hz)
========================================================================================
Generates publication-ready figures for BroadBand (0.5-40 Hz) benchmark:
1. Figure S2 / BroadBand Diagnostic Master (4-Panel):
   - Panel A: BroadBand ST-ATMR-Net Normalized Confusion Matrix
   - Panel B: Multiclass One-vs-Rest (OvR) ROC Curves with Per-Class & Macro AUCs
   - Panel C: Multiclass Precision-Recall (PR) Curves with Average Precision (AP)
   - Panel D: Probability Calibration Curves (Reliability Diagrams) with Brier Scores & ECE
2. BroadBand Comparative 6-Model Confusion Matrices (2x3 panel comparing all 6 architectures)
3. Standalone individual 600 DPI publication figures for each diagnostic dimension
4. BroadBand Diagnostic Evaluation Metrics JSON summary
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
from src.models import StandardLSTM, BiLSTM, TriLSTM, TCN1, TCN2, STATMRNet

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
BROAD_DIAG_DIR = os.path.join(PUB_FIG_DIR, "BroadBand_Diagnostics")
BROAD_EXP_FIG_DIR = os.path.join(ROOT_DIR, "04 Figures", "BroadBand_Benchmark")
BROAD_JSON_PATH = os.path.join(ROOT_DIR, "05 CSV_and_TXT", "Model_Benchmarks_and_Ablations", "BroadBand", "BroadBand_Metrics.json")
BROAD_CKPT_DIR = os.path.join(CODE_DIR, "checkpoints", "BroadBand")

os.makedirs(PUB_FIG_DIR, exist_ok=True)
os.makedirs(PUB_ROOT_FIG_DIR, exist_ok=True)
os.makedirs(BROAD_DIAG_DIR, exist_ok=True)
os.makedirs(BROAD_EXP_FIG_DIR, exist_ok=True)

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
# 1. Load Data, Checkpoints & Compute Continuous Predictions
# --------------------------------------------------------------------------------------
def load_broadband_eval():
    print("[1/4] Loading BroadBand (0.5-40 Hz) test dataset...")
    _, _, test_loader, _, _, info = build_dataloaders(
        band_folder_name="All Band (0.5-40hz)",
        window_size=15,
        batch_size=64,
        per_recording_norm=True
    )
    
    ckpt_path = os.path.join(BROAD_CKPT_DIR, "BroadBand_STATMR_Net_Seed101.pt")
    model = STATMRNet(num_channels=23, feats_per_ch=22, embed_dim=48, hidden_dim=128, dropout=0.25)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x_context']
            d1 = batch['x_transition']
            d2 = batch['x_second_order']
            targets = batch['label']
            logits = model(x, x_transition=d1, x_second_order=d2)
            probs = F.softmax(logits, dim=-1).numpy()
            preds = np.argmax(probs, axis=-1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Load baseline json for comparative models
    with open(BROAD_JSON_PATH, 'r') as f:
        broad_json = json.load(f)

    return broad_json, probs, preds, targets

# --------------------------------------------------------------------------------------
# 2. Generate BroadBand Master 4-Panel Diagnostic Figure
# --------------------------------------------------------------------------------------
def generate_broadband_master_figure(probs, preds, targets):
    print("\n[2/4] Rendering BroadBand Master 4-Panel Diagnostic Figure...")
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2, 3, 4])
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = (cm / row_sums) * 100.0

    acc = accuracy_score(targets, preds) * 100.0
    bal_acc = balanced_accuracy_score(targets, preds) * 100.0
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0) * 100.0

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
    ax_cm.set_title(f"A. BroadBand (0.5-40 Hz) 5-Stage Confusion Matrix (ST-ATMR-Net)\nOverall Acc: {acc:.2f}% | Balanced Acc: {bal_acc:.2f}% | Macro-F1: {macro_f1:.2f}%", pad=12)
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
        pos = np.sum(y_bin)
        if pos > 0:
            fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
            roc_auc = auc(fpr, tpr)
            roc_metrics[name] = float(roc_auc)
            ax_roc.plot(
                fpr, tpr,
                color=STAGE_COLORS[name],
                lw=2.3,
                label=f"{name:11s} (AUC = {roc_auc:.3f})"
            )
        else:
            ax_roc.plot(
                [], [],
                color=STAGE_COLORS[name],
                linestyle=':',
                lw=1.5,
                label=f"{name:11s} (AUC = N/A, Supp=0)"
            )

    # Micro & Macro Average ROC
    active_classes = [i for i in range(5) if np.sum(targets == i) > 0]
    y_true_active = np.zeros((len(targets), len(active_classes)))
    for idx_c, c in enumerate(active_classes):
        y_true_active[:, idx_c] = (targets == c).astype(int)
    probs_active = probs[:, active_classes]

    fpr_micro, tpr_micro, _ = roc_curve(y_true_active.ravel(), probs_active.ravel())
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
    ax_roc.set_title(f"B. BroadBand One-vs-Rest ROC Curves\nMacro-Average AUC = {macro_auc:.3f} | Micro-Average AUC = {auc_micro:.3f}", pad=12)
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
        pos = np.sum(y_bin)
        if pos > 0:
            prec, rec, _ = precision_recall_curve(y_bin, probs[:, i])
            ap = average_precision_score(y_bin, probs[:, i])
            pr_metrics[name] = float(ap)
            ax_pr.plot(
                rec, prec,
                color=STAGE_COLORS[name],
                lw=2.3,
                label=f"{name:11s} (AP = {ap:.3f})"
            )
        else:
            ax_pr.plot(
                [], [],
                color=STAGE_COLORS[name],
                linestyle=':',
                lw=1.5,
                label=f"{name:11s} (AP = N/A, Supp=0)"
            )

    prec_micro, rec_micro, _ = precision_recall_curve(y_true_active.ravel(), probs_active.ravel())
    ap_micro = average_precision_score(y_true_active, probs_active, average="micro")
    macro_ap = np.mean(list(pr_metrics.values()))

    ax_pr.plot(
        rec_micro, prec_micro,
        color='#111111', linestyle='--', lw=2.0,
        label=f"Micro-Average (AP = {ap_micro:.3f})"
    )

    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.04])
    ax_pr.set_title(f"C. BroadBand Precision-Recall (PR) Curves\nMacro-Average AP = {macro_ap:.3f} | Micro-Average AP = {ap_micro:.3f}", pad=12)
    ax_pr.set_xlabel("Recall (Sensitivity / Detection Rate)", labelpad=8)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", labelpad=8)
    ax_pr.legend(loc="lower left", frameon=True, framealpha=0.92, facecolor='#fcfcfc', edgecolor='#dddddd')
    ax_pr.grid(True, linestyle=":", alpha=0.6)

    # ----------------------------------------------------------------------------------
    # Panel D: Probability Calibration / Reliability Diagrams
    # ----------------------------------------------------------------------------------
    ax_cal = axes[1, 1]
    cal_metrics = {}

    ax_cal.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1.8, label="Perfect Calibration (y = x)")

    for i, name in enumerate(STAGE_NAMES):
        y_bin = (targets == i).astype(int)
        pos = np.sum(y_bin)
        brier = brier_score_loss(y_bin, probs[:, i])
        if pos > 0:
            prob_true, prob_pred = calibration_curve(y_bin, probs[:, i], n_bins=8, strategy='uniform')
            ece = np.mean(np.abs(prob_true - prob_pred))
            cal_metrics[name] = {'brier': float(brier), 'ece': float(ece)}

            ax_cal.plot(
                prob_pred, prob_true,
                marker='s', markersize=5.5, lw=2.1,
                color=STAGE_COLORS[name],
                label=f"{name:11s} (Brier = {brier:.3f}, ECE = {ece:.3f})"
            )
        else:
            cal_metrics[name] = {'brier': float(brier), 'ece': None}
            ax_cal.plot(
                [], [],
                color=STAGE_COLORS[name],
                linestyle=':',
                lw=1.5,
                label=f"{name:11s} (Brier = {brier:.3f}, Supp=0)"
            )

    ax_cal.set_xlim([-0.02, 1.02])
    ax_cal.set_ylim([-0.02, 1.02])
    ax_cal.set_title("D. BroadBand Probability Calibration & Reliability Diagrams\nObserved Frequency vs. Predicted Posterior Confidence", pad=12)
    ax_cal.set_xlabel("Mean Predicted Posterior Probability (Bin Confidence)", labelpad=8)
    ax_cal.set_ylabel("Empirical True Fraction in Class (Reliability)", labelpad=8)
    ax_cal.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor='#fcfcfc', edgecolor='#dddddd')
    ax_cal.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Clinical Diagnostic Evaluation Suite: BroadBand EEG Continuum (0.5–40 Hz)", fontsize=14, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])

    out_path_sub = os.path.join(PUB_FIG_DIR, "Figure_BroadBand_Diagnostic_ROC_PR_Calibration_Matrix.png")
    out_path_root = os.path.join(PUB_ROOT_FIG_DIR, "Figure_BroadBand_Diagnostic_ROC_PR_Calibration_Matrix.png")
    out_path_exp = os.path.join(BROAD_EXP_FIG_DIR, "Figure_BroadBand_Diagnostic_ROC_PR_Calibration_Matrix.png")
    
    plt.savefig(out_path_sub, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_root, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_exp, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved Master BroadBand Figure to:\n       -> {out_path_sub}\n       -> {out_path_root}")

    # Export metrics json
    summary_metrics = {
        'band': 'BroadBand (0.5-40 Hz)',
        'overall': {
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'macro_f1': float(macro_f1),
            'macro_auc': float(macro_auc),
            'micro_auc': float(auc_micro),
            'macro_ap': float(macro_ap),
            'micro_ap': float(ap_micro)
        },
        'confusion_matrix_raw': cm.tolist(),
        'confusion_matrix_norm_pct': cm_norm.tolist(),
        'roc_auc_per_class': roc_metrics,
        'precision_recall_ap_per_class': pr_metrics,
        'calibration_metrics_per_class': cal_metrics
    }

    metrics_json_path = os.path.join(BROAD_DIAG_DIR, "BroadBand_Diagnostic_Evaluation_Metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"  [OK] Exported metrics JSON to: {metrics_json_path}")

    return summary_metrics

# --------------------------------------------------------------------------------------
# 3. Generate BroadBand Comparative 6-Model Confusion Matrices (2x3 Grid)
# --------------------------------------------------------------------------------------
def generate_broadband_comparative_matrices(broad_json, targets, preds_statmr):
    print("\n[3/4] Generating BroadBand Comparative 6-Model Confusion Matrices (2x3 Grid)...")
    model_order = [
        ('Standard_LSTM', 'Standard LSTM', 'Blues'),
        ('BiLSTM', 'BiLSTM', 'Oranges'),
        ('TriLSTM', 'TriLSTM', 'Greens'),
        ('TCN_1', 'TCN-1', 'Purples'),
        ('TCN_2', 'TCN-2', 'YlOrBr'),
        ('STATMR_Net_Proposed', 'ST-ATMR-Net (Proposed)', 'crest')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19.0, 12.0), dpi=300)
    axes = axes.flatten()

    for idx, (m_key, title_label, cmap_name) in enumerate(model_order):
        ax = axes[idx]
        if m_key == 'STATMR_Net_Proposed':
            cm = confusion_matrix(targets, preds_statmr, labels=[0, 1, 2, 3, 4])
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = (cm / row_sums) * 100.0
            acc = accuracy_score(targets, preds_statmr) * 100.0
            mf1 = f1_score(targets, preds_statmr, average='macro', zero_division=0) * 100.0
        else:
            m_data = broad_json[m_key]['test_metrics']
            cm = np.array(m_data['confusion_matrix'])
            cm_norm = np.array(m_data['confusion_matrix_norm']) * 100.0
            acc = m_data['accuracy'] * 100.0
            mf1 = m_data['macro_f1'] * 100.0

        annot = np.empty_like(cm, dtype=object)
        for i in range(5):
            for j in range(5):
                annot[i, j] = f"{cm_norm[i, j]:.1f}%\n({cm[i, j]})"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap=cmap_name,
            cbar=True if idx in [2, 5] else False,
            cbar_kws={'label': 'Recall (%)'} if idx in [2, 5] else None,
            xticklabels=STAGE_NAMES,
            yticklabels=STAGE_NAMES,
            ax=ax,
            linewidths=0.8,
            linecolor="#dddddd",
            vmin=0.0,
            vmax=100.0
        )
        ax.set_title(f"{title_label}\nAcc: {acc:.2f}% | Macro-F1: {mf1:.2f}%", pad=9, fontsize=11, fontweight='bold')
        ax.set_xlabel("Predicted Stage" if idx >= 3 else "", labelpad=6)
        ax.set_ylabel("True Stage" if idx % 3 == 0 else "", labelpad=6)
        ax.tick_params(axis='x', rotation=25)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle("BroadBand (0.5–40 Hz) Test Set Confusion Matrix Comparison Across 6 Architectures", fontsize=13.5, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])

    comp_out_path = os.path.join(BROAD_DIAG_DIR, "Figure_BroadBand_Comparative_6_Model_Confusion_Matrices.png")
    plt.savefig(comp_out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved BroadBand Comparative Confusion Matrices to:\n       -> {comp_out_path}")

# --------------------------------------------------------------------------------------
# 4. Generate Standalone High-Resolution (600 DPI) Publication Figures
# --------------------------------------------------------------------------------------
def generate_broadband_standalone_600dpi(probs, preds, targets):
    print("\n[4/4] Rendering Standalone 600 DPI Publication Figures for BroadBand...")
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2, 3, 4])
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = (cm / row_sums) * 100.0

    acc = accuracy_score(targets, preds) * 100.0
    bal_acc = balanced_accuracy_score(targets, preds) * 100.0
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0) * 100.0

    # 1. Confusion Matrix (600 DPI)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=600)
    annot = np.empty_like(cm, dtype=object)
    for i in range(5):
        for j in range(5):
            annot[i, j] = f"{cm_norm[i, j]:.1f}%\n({cm[i, j]})"
    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues",
        cbar_kws={'label': 'Class Sensitivity / Recall (%)'},
        xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
        ax=ax, linewidths=1.0, linecolor="#cccccc", vmin=0.0, vmax=100.0
    )
    ax.set_title(f"BroadBand 5-Stage Confusion Matrix (ST-ATMR-Net)\nOverall Acc: {acc:.2f}% | Balanced Acc: {bal_acc:.2f}% | Macro-F1: {macro_f1:.2f}%", pad=12)
    ax.set_xlabel("Predicted Stage", labelpad=8)
    ax.set_ylabel("True Stage", labelpad=8)
    ax.tick_params(axis='x', rotation=25)
    plt.tight_layout()
    p1 = os.path.join(BROAD_DIAG_DIR, "Figure_BroadBand_STATMR_Net_Normalized_Confusion_Matrix.png")
    plt.savefig(p1, dpi=600, bbox_inches='tight')
    plt.close()

    # 2. ROC Curves (600 DPI)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=600)
    roc_metrics = {}
    for i, name in enumerate(STAGE_NAMES):
        y_bin = (targets == i).astype(int)
        pos = np.sum(y_bin)
        if pos > 0:
            fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
            roc_auc = auc(fpr, tpr)
            roc_metrics[name] = float(roc_auc)
            ax.plot(fpr, tpr, color=STAGE_COLORS[name], lw=2.3, label=f"{name:11s} (AUC = {roc_auc:.3f})")

    active_classes = [i for i in range(5) if np.sum(targets == i) > 0]
    y_true_active = np.zeros((len(targets), len(active_classes)))
    for idx_c, c in enumerate(active_classes):
        y_true_active[:, idx_c] = (targets == c).astype(int)
    probs_active = probs[:, active_classes]

    fpr_micro, tpr_micro, _ = roc_curve(y_true_active.ravel(), probs_active.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    macro_auc = np.mean(list(roc_metrics.values()))

    ax.plot(fpr_micro, tpr_micro, color='#111111', linestyle='--', lw=2.0, label=f"Micro-Average (AUC = {auc_micro:.3f})")
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label="Chance Baseline (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.04])
    ax.set_title(f"BroadBand Multiclass One-vs-Rest ROC Curves\nMacro-Average AUC = {macro_auc:.3f} | Micro-Average AUC = {auc_micro:.3f}", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity)", labelpad=8)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor='#fcfcfc', edgecolor='#dddddd')
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p2 = os.path.join(BROAD_DIAG_DIR, "Figure_BroadBand_STATMR_Net_Multiclass_ROC_Curves.png")
    plt.savefig(p2, dpi=600, bbox_inches='tight')
    plt.close()

    # 3. Precision-Recall Curves (600 DPI)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=600)
    pr_metrics = {}
    for i, name in enumerate(STAGE_NAMES):
        y_bin = (targets == i).astype(int)
        pos = np.sum(y_bin)
        if pos > 0:
            prec, rec, _ = precision_recall_curve(y_bin, probs[:, i])
            ap = average_precision_score(y_bin, probs[:, i])
            pr_metrics[name] = float(ap)
            ax.plot(rec, prec, color=STAGE_COLORS[name], lw=2.3, label=f"{name:11s} (AP = {ap:.3f})")

    prec_micro, rec_micro, _ = precision_recall_curve(y_true_active.ravel(), probs_active.ravel())
    ap_micro = average_precision_score(y_true_active, probs_active, average="micro")
    macro_ap = np.mean(list(pr_metrics.values()))

    ax.plot(rec_micro, prec_micro, color='#111111', linestyle='--', lw=2.0, label=f"Micro-Average (AP = {ap_micro:.3f})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.04])
    ax.set_title(f"BroadBand Multiclass Precision-Recall (PR) Curves\nMacro-Average AP = {macro_ap:.3f} | Micro-Average AP = {ap_micro:.3f}", pad=12)
    ax.set_xlabel("Recall (Sensitivity)", labelpad=8)
    ax.set_ylabel("Precision (Positive Predictive Value)", labelpad=8)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, facecolor='#fcfcfc', edgecolor='#dddddd')
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p3 = os.path.join(BROAD_DIAG_DIR, "Figure_BroadBand_STATMR_Net_Multiclass_Precision_Recall_Curves.png")
    plt.savefig(p3, dpi=600, bbox_inches='tight')
    plt.close()

    # 4. Calibration Curve (600 DPI)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=600)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1.8, label="Perfect Calibration (y = x)")
    for i, name in enumerate(STAGE_NAMES):
        y_bin = (targets == i).astype(int)
        pos = np.sum(y_bin)
        brier = brier_score_loss(y_bin, probs[:, i])
        if pos > 0:
            prob_true, prob_pred = calibration_curve(y_bin, probs[:, i], n_bins=8, strategy='uniform')
            ece = np.mean(np.abs(prob_true - prob_pred))
            ax.plot(prob_pred, prob_true, marker='s', markersize=5.5, lw=2.1, color=STAGE_COLORS[name], label=f"{name:11s} (Brier={brier:.3f}, ECE={ece:.3f})")
        else:
            ax.plot([], [], color=STAGE_COLORS[name], linestyle=':', lw=1.5, label=f"{name:11s} (Brier={brier:.3f}, Supp=0)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_title("BroadBand Probability Calibration (Reliability Diagram)\nPosterior Risk Confidence Calibration", pad=12)
    ax.set_xlabel("Mean Predicted Posterior Probability (Bin Confidence)", labelpad=8)
    ax.set_ylabel("Empirical True Fraction in Class", labelpad=8)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor='#fcfcfc', edgecolor='#dddddd')
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p4 = os.path.join(BROAD_DIAG_DIR, "Figure_BroadBand_STATMR_Net_Probability_Calibration_Reliability.png")
    plt.savefig(p4, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved 600 DPI figures:\n       -> {p1}\n       -> {p2}\n       -> {p3}\n       -> {p4}")

def main():
    print("=" * 85)
    print("EXECUTING BROADBAND (0.5-40 Hz) PUBLICATION DIAGNOSTIC SUITE")
    print("=" * 85)
    broad_json, probs, preds, targets = load_broadband_eval()
    summary_metrics = generate_broadband_master_figure(probs, preds, targets)
    generate_broadband_comparative_matrices(broad_json, targets, preds)
    generate_broadband_standalone_600dpi(probs, preds, targets)
    print("\n" + "=" * 85)
    print("ALL BROADBAND PUBLICATION DIAGNOSTIC FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 85)

if __name__ == "__main__":
    main()
