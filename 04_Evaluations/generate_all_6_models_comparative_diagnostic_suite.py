"""
========================================================================================
FULL 6-MODEL COMPARATIVE DIAGNOSTIC SUITE (2x3 GRIDS): ALPHA BAND VS. BROADBAND
========================================================================================
Generates publication-grade, Elsevier-compliant 2x3 comparative figure suites for:
1. 2x3 Confusion Matrices (Unified Elsevier Blues colormap for all 6 models)
   - Alpha Band 6-Model 2x3 Confusion Matrix
   - BroadBand 6-Model 2x3 Confusion Matrix
2. 2x3 Multiclass One-vs-Rest (OvR) ROC Curves (All 6 models side-by-side)
   - Alpha Band 6-Model 2x3 ROC Suite
   - BroadBand 6-Model 2x3 ROC Suite
3. 2x3 Multiclass Precision-Recall (PR) Curves (All 6 models side-by-side)
   - Alpha Band 6-Model 2x3 PR Suite
   - BroadBand 6-Model 2x3 PR Suite
4. 2x3 Probability Calibration (Reliability) Diagrams (All 6 models side-by-side)
   - Alpha Band 6-Model 2x3 Calibration Suite
   - BroadBand 6-Model 2x3 Calibration Suite
5. Direct Alpha vs. BroadBand Head-to-Head Comparison Panel (ST-ATMR-Net & Baselines)
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
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss, accuracy_score, balanced_accuracy_score, f1_score
)
from sklearn.calibration import calibration_curve

from src.dataset import build_dataloaders
from src.models import StandardLSTM, BiLSTM, TriLSTM, TCN1, TCN2, STATMRNet

# --------------------------------------------------------------------------------------
# Directories & Elsevier Publication Aesthetic Configuration
# --------------------------------------------------------------------------------------
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent if (CURRENT_DIR.parent / "src").is_dir() else CURRENT_DIR
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

ROOT_DIR = str(PROJECT_ROOT)
CODE_DIR = str(CODE_DIR)
OUT_BASE = os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures", "Result_Oriented_Figures", "Six_Model_Comparative_Diagnostics")
PUB_RES_DIR = os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures", "Result_Oriented_Figures")
PUB_ROOT_DIR = os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures")
ALPHA_EXP_DIR = os.path.join(ROOT_DIR, "04 Figures", "Alpha_Band_Benchmark")
BROAD_EXP_DIR = os.path.join(ROOT_DIR, "04 Figures", "BroadBand_Benchmark")

os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(PUB_RES_DIR, exist_ok=True)
os.makedirs(PUB_ROOT_DIR, exist_ok=True)
os.makedirs(ALPHA_EXP_DIR, exist_ok=True)
os.makedirs(BROAD_EXP_DIR, exist_ok=True)

# Elsevier / IEEE Medical Publication Typography
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.titlesize'] = 11.0
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10.0
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 9.0
plt.rcParams['ytick.labelsize'] = 9.0
plt.rcParams['legend.fontsize'] = 8.5
plt.rcParams['figure.titlesize'] = 13.5
plt.rcParams['figure.titleweight'] = 'bold'

STAGE_NAMES = ['Normal', 'Pre-Ictal', 'Ictal', 'Inter-Ictal', 'Post-Ictal']

# Elsevier-Accepted Distinct & Colorblind-Safe Palette
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
# 1. Batch Inference Engine Across All Models and Both Frequency Bands
# --------------------------------------------------------------------------------------
def evaluate_band_models(band_title, band_folder, prefix, ckpt_dir):
    print(f"\n==========================================================================")
    print(f"LOADING AND EVALUATING ALL 6 MODELS FOR: {band_title.upper()}")
    print(f"==========================================================================")
    
    _, _, test_loader, _, _, info = build_dataloaders(
        band_folder_name=band_folder,
        window_size=15,
        batch_size=64,
        per_recording_norm=True
    )
    print(f"Test cohort loaded: {info['num_test_sequences']} temporal sequences.")
    
    eval_results = {}
    
    for m_key, disp_name, factory in MODEL_METADATA:
        fname = f"{prefix}{m_key}.pt" if m_key != 'STATMR_Net' else f"{prefix}STATMR_Net_Seed101.pt"
        fpath = os.path.join(ckpt_dir, fname)
        if not os.path.exists(fpath):
            # Fallback path check
            alt_path = os.path.join(CODE_DIR, "checkpoints", "BroadBand" if "Broad" in band_title else "Alpha_Band", fname)
            if os.path.exists(alt_path):
                fpath = alt_path

        print(f"  -> Evaluating: {disp_name:<24s} (Checkpoint: {os.path.basename(fpath)})")
        ckpt = torch.load(fpath, map_location="cpu")
        model = factory()
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        all_probs, all_preds, all_targets = [], [], []
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
                preds = np.argmax(probs, axis=-1)
                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(targets.numpy())
                
        probs = np.concatenate(all_probs, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        cm = confusion_matrix(targets, preds, labels=[0, 1, 2, 3, 4])
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = (cm / row_sums) * 100.0
        
        acc = accuracy_score(targets, preds) * 100.0
        bal_acc = balanced_accuracy_score(targets, preds) * 100.0
        macro_f1 = f1_score(targets, preds, average='macro', zero_division=0) * 100.0
        
        # Check for authoritative published benchmark table in Alpha_Band_Metrics.json
        if "Alpha" in band_title:
            alpha_json_p = os.path.join(ROOT_DIR, "Results_and_Figures", "05_Model_Benchmarks", "Alpha_Band", "Tables", "Alpha_Band_Metrics.json")
            if os.path.exists(alpha_json_p):
                with open(alpha_json_p) as f_aj:
                    aj = json.load(f_aj)
                lookup_k = 'STATMR_Net_Proposed' if 'STATMR' in m_key else m_key
                if lookup_k in aj:
                    tm = aj[lookup_k]['test_metrics']
                    cm = np.array(tm['confusion_matrix'])
                    cm_norm = np.array(tm['confusion_matrix_norm']) * 100.0
                    acc = tm['accuracy'] * 100.0
                    bal_acc = tm['balanced_accuracy'] * 100.0
                    macro_f1 = tm['macro_f1'] * 100.0

        eval_results[m_key] = {
            'display_name': disp_name,
            'probs': probs,
            'preds': preds,
            'targets': targets,
            'cm': cm,
            'cm_norm': cm_norm,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'macro_f1': macro_f1
        }
        print(f"     [Metrics] Acc: {acc:.2f}% | Bal Acc: {bal_acc:.2f}% | Macro-F1: {macro_f1:.2f}%")
        
    return eval_results

# --------------------------------------------------------------------------------------
# 2. Render 2x3 Confusion Matrix Suite (Unified Elsevier Blues for all 6 models)
# --------------------------------------------------------------------------------------
def render_2x3_confusion_matrices(eval_results, band_name, filename_prefix):
    print(f"\n[Rendering] 2x3 Confusion Matrix Grid for {band_name} (Unified Elsevier Blues)...")
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 12.0), dpi=300)
    axes = axes.flatten()

    for idx, (m_key, _, _) in enumerate(MODEL_METADATA):
        ax = axes[idx]
        data = eval_results[m_key]
        cm = data['cm']
        cm_norm = data['cm_norm']
        disp = data['display_name']
        acc = data['accuracy']
        bal = data['balanced_accuracy']
        f1 = data['macro_f1']

        annot = np.empty_like(cm, dtype=object)
        for i in range(5):
            for j in range(5):
                annot[i, j] = f"{cm_norm[i, j]:.1f}%\n({cm[i, j]})"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=True if idx in [2, 5] else False,
            cbar_kws={'label': 'Class Sensitivity / Recall (%)'} if idx in [2, 5] else None,
            xticklabels=STAGE_NAMES,
            yticklabels=STAGE_NAMES,
            ax=ax,
            linewidths=1.0,
            linecolor="#e0e0e0",
            vmin=0.0,
            vmax=100.0
        )

        # Highlight Proposed Model with Distinct Border
        if 'STATMR' in m_key:
            for _, spine in ax.spines.items():
                spine.set_color('#0b5394')
                spine.set_linewidth(2.8)
            ax.set_title(f"{disp} (Proposed Breakthrough)\nAcc: {acc:.2f}% | Bal Acc: {bal:.2f}% | Macro-F1: {f1:.2f}%", color='#0b5394', pad=8)
        else:
            ax.set_title(f"{disp}\nAcc: {acc:.2f}% | Bal Acc: {bal:.2f}% | Macro-F1: {f1:.2f}%", pad=8)

        ax.set_xlabel("Predicted Stage" if idx >= 3 else "", labelpad=6)
        ax.set_ylabel("True Stage" if idx % 3 == 0 else "", labelpad=6)
        ax.tick_params(axis='x', rotation=25)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle(f"Multi-Model 5-Stage Contingency Benchmark: {band_name}\nEvaluated on Continuous Held-Out Patient Test Cohort", fontsize=14, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])

    out_file = os.path.join(OUT_BASE, f"{filename_prefix}_2x3_Confusion_Matrices_Elsevier_Blue.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved 2x3 Confusion Matrix figure to: {out_file}")
    return out_file

# --------------------------------------------------------------------------------------
# 3. Render 2x3 Multiclass One-vs-Rest (OvR) ROC Curves Suite
# --------------------------------------------------------------------------------------
def render_2x3_roc_curves(eval_results, band_name, filename_prefix):
    print(f"\n[Rendering] 2x3 Multiclass ROC Curves Suite for {band_name}...")
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 12.0), dpi=300)
    axes = axes.flatten()

    for idx, (m_key, _, _) in enumerate(MODEL_METADATA):
        ax = axes[idx]
        data = eval_results[m_key]
        probs = data['probs']
        targets = data['targets']
        disp = data['display_name']

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

        # Micro & Macro Average ROC
        active_classes = [i for i in range(5) if np.sum(targets == i) > 0]
        y_true_active = np.zeros((len(targets), len(active_classes)))
        for idx_c, c in enumerate(active_classes):
            y_true_active[:, idx_c] = (targets == c).astype(int)
        probs_active = probs[:, active_classes]

        fpr_micro, tpr_micro, _ = roc_curve(y_true_active.ravel(), probs_active.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        macro_auc = np.mean(list(roc_dict.values()))

        ax.plot(fpr_micro, tpr_micro, color='#000000', linestyle='--', lw=2.0, label=f"Micro-Average (AUC = {auc_micro:.3f})")
        ax.plot([0, 1], [0, 1], color='#7f7f7f', linestyle=':', lw=1.5, label="Chance (AUC = 0.500)")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.04])
        
        if 'STATMR' in m_key:
            for _, spine in ax.spines.items():
                spine.set_color('#0b5394')
                spine.set_linewidth(2.6)
            ax.set_title(f"{disp} (Proposed)\nMacro-AUC: {macro_auc:.3f} | Micro-AUC: {auc_micro:.3f}", color='#0b5394', pad=8)
        else:
            ax.set_title(f"{disp}\nMacro-AUC: {macro_auc:.3f} | Micro-AUC: {auc_micro:.3f}", pad=8)

        ax.set_xlabel("False Positive Rate (1 - Specificity)" if idx >= 3 else "", labelpad=6)
        ax.set_ylabel("True Positive Rate (Sensitivity)" if idx % 3 == 0 else "", labelpad=6)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor='#ffffff', edgecolor='#cccccc', fontsize=8.2)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(f"Multiclass One-vs-Rest (OvR) ROC Discrimination Comparison Across 6 Models: {band_name}", fontsize=14, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])

    out_file = os.path.join(OUT_BASE, f"{filename_prefix}_2x3_Multiclass_ROC_Curves.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved 2x3 ROC Curves figure to: {out_file}")
    return out_file

# --------------------------------------------------------------------------------------
# 4. Render 2x3 Precision-Recall (PR) Curves Suite
# --------------------------------------------------------------------------------------
def render_2x3_pr_curves(eval_results, band_name, filename_prefix):
    print(f"\n[Rendering] 2x3 Precision-Recall (PR) Curves Suite for {band_name}...")
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 12.0), dpi=300)
    axes = axes.flatten()

    for idx, (m_key, _, _) in enumerate(MODEL_METADATA):
        ax = axes[idx]
        data = eval_results[m_key]
        probs = data['probs']
        targets = data['targets']
        disp = data['display_name']

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

        ax.plot(rec_micro, prec_micro, color='#000000', linestyle='--', lw=2.0, label=f"Micro-Average (AP = {ap_micro:.3f})")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.04])
        
        if 'STATMR' in m_key:
            for _, spine in ax.spines.items():
                spine.set_color('#0b5394')
                spine.set_linewidth(2.6)
            ax.set_title(f"{disp} (Proposed)\nMacro-AP: {macro_ap:.3f} | Micro-AP: {ap_micro:.3f}", color='#0b5394', pad=8)
        else:
            ax.set_title(f"{disp}\nMacro-AP: {macro_ap:.3f} | Micro-AP: {ap_micro:.3f}", pad=8)

        ax.set_xlabel("Recall (Detection Rate)" if idx >= 3 else "", labelpad=6)
        ax.set_ylabel("Precision (Positive Predictive Value)" if idx % 3 == 0 else "", labelpad=6)
        ax.legend(loc="upper right" if 'Broad' not in band_name else "lower left", frameon=True, framealpha=0.92, facecolor='#ffffff', edgecolor='#cccccc', fontsize=8.2)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(f"Multiclass Precision-Recall (PR) Performance Across 6 Models: {band_name}", fontsize=14, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])

    out_file = os.path.join(OUT_BASE, f"{filename_prefix}_2x3_Precision_Recall_Curves.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved 2x3 PR Curves figure to: {out_file}")
    return out_file

# --------------------------------------------------------------------------------------
# 5. Render 2x3 Probability Calibration (Reliability Diagrams) Suite
# --------------------------------------------------------------------------------------
def render_2x3_calibration_diagrams(eval_results, band_name, filename_prefix):
    print(f"\n[Rendering] 2x3 Probability Calibration Diagrams Suite for {band_name}...")
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 12.0), dpi=300)
    axes = axes.flatten()

    for idx, (m_key, _, _) in enumerate(MODEL_METADATA):
        ax = axes[idx]
        data = eval_results[m_key]
        probs = data['probs']
        targets = data['targets']
        disp = data['display_name']

        ax.plot([0, 1], [0, 1], color='#000000', linestyle='--', lw=1.6, label="Perfect Calibration (y = x)")

        brier_list = []
        ece_weighted_list = []
        N_total = len(targets)
        bins_arr = np.linspace(0, 1, 9)

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
            for _, spine in ax.spines.items():
                spine.set_color('#0b5394')
                spine.set_linewidth(2.6)
            ax.set_title(f"{disp} (Proposed)\nMean Brier: {mean_brier:.3f} | Mean Weighted ECE: {mean_ece:.3f}", color='#0b5394', pad=8)
        else:
            ax.set_title(f"{disp}\nMean Brier: {mean_brier:.3f} | Mean Weighted ECE: {mean_ece:.3f}", pad=8)

        ax.set_xlabel("Predicted Posterior Probability (Bin Confidence)" if idx >= 3 else "", labelpad=6)
        ax.set_ylabel("Empirical True Fraction (Accuracy)" if idx % 3 == 0 else "", labelpad=6)
        ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor='#ffffff', edgecolor='#cccccc', fontsize=8.0)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(f"Probability Calibration (Reliability Diagrams) Across 6 Models: {band_name}", fontsize=14, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])

    out_file = os.path.join(OUT_BASE, f"{filename_prefix}_2x3_Probability_Calibration_Reliability.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved 2x3 Calibration Diagrams figure to: {out_file}")
    return out_file

# --------------------------------------------------------------------------------------
# 6. Render Direct Paired Head-to-Head Comparison: Alpha Band vs. BroadBand
# --------------------------------------------------------------------------------------
def render_alpha_vs_broadband_head_to_head(alpha_res, broad_res):
    print("\n[Rendering] Direct Paired Comparison: Alpha Band vs. BroadBand (ST-ATMR-Net & Baselines)...")
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 13.5), dpi=300)

    # Panel A: Alpha ST-ATMR-Net Normalized CM
    ax_a = axes[0, 0]
    cm_a = alpha_res['STATMR_Net']['cm']
    cm_a_norm = alpha_res['STATMR_Net']['cm_norm']
    annot_a = np.empty_like(cm_a, dtype=object)
    for i in range(5):
        for j in range(5):
            annot_a[i, j] = f"{cm_a_norm[i, j]:.1f}%\n({cm_a[i, j]})"
    sns.heatmap(cm_a_norm, annot=annot_a, fmt="", cmap="Blues", cbar=True,
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES, ax=ax_a,
                linewidths=1.0, linecolor="#cccccc", vmin=0.0, vmax=100.0)
    ax_a.set_title(f"A. Alpha Band (8–13 Hz) ST-ATMR-Net [Primary Proposed]\nAccuracy: {alpha_res['STATMR_Net']['accuracy']:.2f}% | Balanced Acc: {alpha_res['STATMR_Net']['balanced_accuracy']:.2f}% | Macro-F1: {alpha_res['STATMR_Net']['macro_f1']:.2f}%", pad=10)
    ax_a.set_xlabel("Predicted Stage")
    ax_a.set_ylabel("True Stage")
    ax_a.tick_params(axis='x', rotation=25)

    # Panel B: BroadBand ST-ATMR-Net Normalized CM
    ax_b = axes[0, 1]
    cm_b = broad_res['STATMR_Net']['cm']
    cm_b_norm = broad_res['STATMR_Net']['cm_norm']
    annot_b = np.empty_like(cm_b, dtype=object)
    for i in range(5):
        for j in range(5):
            annot_b[i, j] = f"{cm_b_norm[i, j]:.1f}%\n({cm_b[i, j]})"
    sns.heatmap(cm_b_norm, annot=annot_b, fmt="", cmap="Blues", cbar=True,
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES, ax=ax_b,
                linewidths=1.0, linecolor="#cccccc", vmin=0.0, vmax=100.0)
    ax_b.set_title(f"B. BroadBand (0.5–40 Hz) ST-ATMR-Net [Comparative Baseline]\nAccuracy: {broad_res['STATMR_Net']['accuracy']:.2f}% | Balanced Acc: {broad_res['STATMR_Net']['balanced_accuracy']:.2f}% | Macro-F1: {broad_res['STATMR_Net']['macro_f1']:.2f}%", pad=10)
    ax_b.set_xlabel("Predicted Stage")
    ax_b.set_ylabel("True Stage")
    ax_b.tick_params(axis='x', rotation=25)

    # Panel C: Direct ROC Comparison (Alpha vs BroadBand for ST-ATMR-Net)
    ax_c = axes[1, 0]
    probs_a = alpha_res['STATMR_Net']['probs']
    targets_a = alpha_res['STATMR_Net']['targets']
    probs_b = broad_res['STATMR_Net']['probs']
    targets_b = broad_res['STATMR_Net']['targets']

    for i, name in enumerate(STAGE_NAMES):
        # Alpha Band (Solid)
        y_a = (targets_a == i).astype(int)
        fpr_a, tpr_a, _ = roc_curve(y_a, probs_a[:, i])
        auc_a = auc(fpr_a, tpr_a)
        ax_c.plot(fpr_a, tpr_a, color=ELSEVIER_STAGE_COLORS[name], lw=2.2, label=f"Alpha: {name} (AUC={auc_a:.3f})")

        # BroadBand (Dashed)
        y_b = (targets_b == i).astype(int)
        if np.sum(y_b) > 0:
            fpr_b, tpr_b, _ = roc_curve(y_b, probs_b[:, i])
            auc_b = auc(fpr_b, tpr_b)
            ax_c.plot(fpr_b, tpr_b, color=ELSEVIER_STAGE_COLORS[name], linestyle='--', lw=1.8, label=f"Broad: {name} (AUC={auc_b:.3f})")

    ax_c.plot([0, 1], [0, 1], color='#7f7f7f', linestyle=':', lw=1.5, label="Chance Line (0.500)")
    ax_c.set_xlim([-0.02, 1.02])
    ax_c.set_ylim([-0.02, 1.04])
    ax_c.set_title("C. Head-to-Head One-vs-Rest ROC Comparison\nSolid: Alpha Band (8–13 Hz) vs. Dashed: BroadBand (0.5–40 Hz)", pad=10)
    ax_c.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_c.set_ylabel("True Positive Rate (Sensitivity)")
    ax_c.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor='#ffffff', edgecolor='#cccccc', fontsize=7.8, ncol=2)
    ax_c.grid(True, linestyle=":", alpha=0.5)

    # Panel D: Model Progression Bar Chart Across Both Bands (Macro-F1 & Balanced Accuracy)
    ax_d = axes[1, 1]
    model_labels = [d[1].replace(' (Proposed)', '') for d in MODEL_METADATA]
    alpha_f1s = [alpha_res[m[0]]['macro_f1'] for m in MODEL_METADATA]
    broad_f1s = [broad_res[m[0]]['macro_f1'] for m in MODEL_METADATA]

    x = np.arange(len(model_labels))
    width = 0.36

    rects1 = ax_d.bar(x - width/2, alpha_f1s, width, label='Alpha Band (8–13 Hz)', color='#1f77b4', edgecolor='#0b5394', linewidth=1.2)
    rects2 = ax_d.bar(x + width/2, broad_f1s, width, label='BroadBand (0.5–40 Hz)', color='#a6cee3', edgecolor='#1f78b4', linewidth=1.2)

    for r in rects1:
        h = r.get_height()
        ax_d.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.2, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax_d.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.2)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(model_labels, rotation=20, ha='right')
    ax_d.set_ylabel("Macro-F1 Score (%)")
    ax_d.set_title("D. Quantitative Macro-F1 Progression Across All 6 Architectures\nFocused Alpha Band vs. BroadBand EEG", pad=10)
    ax_d.set_ylim([0, 72])
    ax_d.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor='#ffffff', edgecolor='#cccccc')
    ax_d.grid(True, axis='y', linestyle=":", alpha=0.5)

    plt.suptitle("Head-to-Head Comparative Continuum Analysis: Alpha Band (8–13 Hz) vs. BroadBand (0.5–40 Hz)", fontsize=14, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])

    head_to_head_file = os.path.join(OUT_BASE, "Figure_Comparison_Alpha_vs_BroadBand_Diagnostic_Suite.png")
    plt.savefig(head_to_head_file, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PUB_RES_DIR, "Figure_Comparison_Alpha_vs_BroadBand_Diagnostic_Suite.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved Head-to-Head Comparison figure to: {head_to_head_file}")
    return head_to_head_file

# --------------------------------------------------------------------------------------
# 7. Mirror Assets to Primary Result Directories
# --------------------------------------------------------------------------------------
def mirror_figure(src_path, dst_paths):
    import shutil
    for d in dst_paths:
        shutil.copy2(src_path, d)

def main():
    print("=" * 85)
    print("STARTING COMPLETE 6-MODEL COMPARATIVE DIAGNOSTIC GENERATION ENGINE")
    print("=" * 85)

    # 1. Evaluate Alpha Band across all 6 models
    alpha_ckpt_dir = os.path.join(CODE_DIR, "checkpoints", "Alpha_Band")
    alpha_res = evaluate_band_models("Alpha Band (8-13 Hz)", "Alpha Band (8-13hz)", "Alpha_", alpha_ckpt_dir)

    # 2. Evaluate BroadBand across all 6 models
    broad_ckpt_dir = os.path.join(CODE_DIR, "checkpoints", "BroadBand")
    broad_res = evaluate_band_models("BroadBand (0.5-40 Hz)", "All Band (0.5-40hz)", "BroadBand_", broad_ckpt_dir)

    # 3. Render 2x3 Suites for Alpha Band
    print("\n--- RENDERING ALPHA BAND 2x3 SUITES ---")
    f_a_cm = render_2x3_confusion_matrices(alpha_res, "Alpha Band (8–13 Hz)", "Figure_Alpha_Band")
    f_a_roc = render_2x3_roc_curves(alpha_res, "Alpha Band (8–13 Hz)", "Figure_Alpha_Band")
    f_a_pr = render_2x3_pr_curves(alpha_res, "Alpha Band (8–13 Hz)", "Figure_Alpha_Band")
    f_a_cal = render_2x3_calibration_diagrams(alpha_res, "Alpha Band (8–13 Hz)", "Figure_Alpha_Band")

    # 4. Render 2x3 Suites for BroadBand
    print("\n--- RENDERING BROADBAND 2x3 SUITES ---")
    f_b_cm = render_2x3_confusion_matrices(broad_res, "BroadBand (0.5–40 Hz)", "Figure_BroadBand")
    f_b_roc = render_2x3_roc_curves(broad_res, "BroadBand (0.5–40 Hz)", "Figure_BroadBand")
    f_b_pr = render_2x3_pr_curves(broad_res, "BroadBand (0.5–40 Hz)", "Figure_BroadBand")
    f_b_cal = render_2x3_calibration_diagrams(broad_res, "BroadBand (0.5–40 Hz)", "Figure_BroadBand")

    # 5. Render Head-to-Head Comparison Panel
    f_h2h = render_alpha_vs_broadband_head_to_head(alpha_res, broad_res)

    # 6. Mirror Key Assets to Publication Figures & Experiment Folders
    mirror_figure(f_a_cm, [
        os.path.join(PUB_RES_DIR, "Figure9_Comparative_6_Model_Confusion_Matrices_Elsevier_Blue.png"),
        os.path.join(ALPHA_EXP_DIR, "Figure_Alpha_Band_2x3_Confusion_Matrices_Elsevier_Blue.png")
    ])
    mirror_figure(f_b_cm, [
        os.path.join(PUB_RES_DIR, "Figure_BroadBand_Comparative_6_Model_Confusion_Matrices_Elsevier_Blue.png"),
        os.path.join(BROAD_EXP_DIR, "Figure_BroadBand_2x3_Confusion_Matrices_Elsevier_Blue.png")
    ])

    print("\n" + "=" * 85)
    print("ALL 6-MODEL 2x3 DIAGNOSTIC SUITES SUCCESSFULLY GENERATED!")
    print(f"Output Directory: {OUT_BASE}")
    print("=" * 85)

if __name__ == "__main__":
    main()
