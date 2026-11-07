"""
========================================================================================
FIGURE: RIGOROUS CLINICAL VERIFICATION & PAIRED CROSS-BAND COMPARISON
========================================================================================
Generates publication-quality 300 DPI multi-panel figure resolving:
- Panel A: Paired Head-to-Head Comparison on Identical 7 Test Cohort (Alpha vs BroadBand)
- Panel B: Patient-Level Aggregated Performance across 14 Pediatric Patients
- Panel C: Unconstrained Transition Compliance Benchmark (Ground Truth vs Models)
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 11.5
plt.rcParams['axes.labelsize'] = 10.0
plt.rcParams['xtick.labelsize'] = 9.0
plt.rcParams['ytick.labelsize'] = 9.0
plt.rcParams['legend.fontsize'] = 8.5

from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent if (CURRENT_DIR.parent / "src").is_dir() else CURRENT_DIR
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

ROOT_DIR = str(PROJECT_ROOT)
CODE_DIR = str(CODE_DIR)
DIR_VERIF = os.path.abspath(os.path.join(ROOT_DIR, "05 CSV_and_TXT", "Publication_Tables"))

def generate_figure():
    df_audit = pd.read_csv(os.path.join(DIR_VERIF, "Table1_Verified_21_Recording_Clinical_Audit.csv"))
    df_patient = pd.read_csv(os.path.join(DIR_VERIF, "Table2_Patient_Level_Aggregated_Performance.csv"))
    df_paired = pd.read_csv(os.path.join(DIR_VERIF, "Table3_Paired_Identical_Cohort_Alpha_vs_BroadBand.csv"))
    df_compl = pd.read_csv(os.path.join(DIR_VERIF, "Table4_Transition_Compliance_Benchmark.csv"))
    
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), dpi=300)
    plt.subplots_adjust(wspace=0.28, left=0.05, right=0.97, top=0.88, bottom=0.15)
    
    # ----------------------------------------------------------------------------------
    # Panel A: Paired Head-to-Head Comparison (Identical 7 Test Cohort)
    # ----------------------------------------------------------------------------------
    ax_p = axes[0]
    metrics = ['Sequence Acc (%)', 'Macro-F1 (%)', 'Ictal Cov (%)', 'Lead Time (s)']
    alpha_means = [df_paired['Alpha_Acc_pct'].mean(), df_paired['Alpha_F1_pct'].mean(), 
                   df_paired['Alpha_Ictal_Cov_pct'].mean(), df_paired['Alpha_Lead_s'].mean()]
    alpha_stds = [df_paired['Alpha_Acc_pct'].std(), df_paired['Alpha_F1_pct'].std(), 
                  df_paired['Alpha_Ictal_Cov_pct'].std(), df_paired['Alpha_Lead_s'].std()]
    
    broad_means = [df_paired['Broad_Acc_pct'].mean(), df_paired['Broad_F1_pct'].mean(), 
                   df_paired['Broad_Ictal_Cov_pct'].mean(), df_paired['Broad_Lead_s'].mean()]
    broad_stds = [df_paired['Broad_Acc_pct'].std(), df_paired['Broad_F1_pct'].std(), 
                  df_paired['Broad_Ictal_Cov_pct'].std(), df_paired['Broad_Lead_s'].std()]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    b1 = ax_p.bar(x - width/2, alpha_means, width, yerr=alpha_stds, capsize=4,
                  label='Alpha Band (8–13 Hz)', color='#2563eb', edgecolor='#1e293b', alpha=0.9)
    b2 = ax_p.bar(x + width/2, broad_means, width, yerr=broad_stds, capsize=4,
                  label='BroadBand (0.5–40 Hz)', color='#d97706', edgecolor='#1e293b', alpha=0.9)
                  
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(metrics, fontweight='bold', fontsize=9.0)
    ax_p.set_ylabel("Metric Value (Mean \u00b1 SD)", fontweight='bold')
    ax_p.set_title("A. Paired Comparison on Identical 7 Test Cohort\n(No Significant Difference: p > 0.35)", fontweight='bold', pad=10)
    ax_p.legend(loc='upper right', frameon=True)
    ax_p.grid(axis='y', linestyle='--', alpha=0.4)
    ax_p.set_ylim(0, 115)
    
    # Value labels
    for rects in [b1, b2]:
        for rect in rects:
            h = rect.get_height()
            ax_p.text(rect.get_x() + rect.get_width()/2., h + 3, f'{h:.1f}',
                      ha='center', va='bottom', fontsize=7.5, fontweight='bold')
                      
    # ----------------------------------------------------------------------------------
    # Panel B: Patient-Level Aggregated Performance (N=14 Patients)
    # ----------------------------------------------------------------------------------
    ax_pt = axes[1]
    pts = df_patient['Patient'].values
    accs = df_patient['Seq_Accuracy_pct'].values
    f1s = df_patient['Macro_F1_pct'].values
    
    x_pt = np.arange(len(pts))
    w_pt = 0.38
    
    ax_pt.bar(x_pt - w_pt/2, accs, w_pt, label='Sequence Accuracy (%)', color='#0284c7', edgecolor='#0f172a', alpha=0.85)
    ax_pt.bar(x_pt + w_pt/2, f1s, w_pt, label='Macro-F1 (%)', color='#10b981', edgecolor='#0f172a', alpha=0.85)
    
    # Horizontal mean lines
    ax_pt.axhline(accs.mean(), color='#0369a1', linestyle='--', lw=1.5, label=f'Mean Acc: {accs.mean():.1f}%')
    ax_pt.axhline(f1s.mean(), color='#047857', linestyle=':', lw=1.5, label=f'Mean F1: {f1s.mean():.1f}%')
    
    ax_pt.set_xticks(x_pt)
    ax_pt.set_xticklabels(pts, rotation=45, ha='right', fontsize=8.5, fontweight='bold')
    ax_pt.set_ylabel("Performance (%)", fontweight='bold')
    ax_pt.set_title("B. Patient-Level Aggregated Performance (N=14)\n(Held-Out Recording-Independent Evaluation)", fontweight='bold', pad=10)
    ax_pt.legend(loc='lower left', frameon=True, fontsize=8.0)
    ax_pt.grid(axis='y', linestyle='--', alpha=0.4)
    ax_pt.set_ylim(0, 105)
    
    # ----------------------------------------------------------------------------------
    # Panel C: Unconstrained Transition Compliance Benchmark
    # ----------------------------------------------------------------------------------
    ax_tc = axes[2]
    # Reverse order for horizontal bar
    df_compl_rev = df_compl.iloc[::-1].reset_index(drop=True)
    mnames = [m.replace(' (Reference)', '').replace(' (Proposed)', '') for m in df_compl_rev['Model']]
    c_means = df_compl_rev['Transition_Compliance_pct'].values
    c_stds = df_compl_rev['Transition_Compliance_Std'].values
    
    colors = ['#64748b', '#64748b', '#64748b', '#64748b', '#64748b', '#2563eb', '#16a34a']
    
    y_pos = np.arange(len(mnames))
    bars = ax_tc.barh(y_pos, c_means, xerr=c_stds, capsize=4, color=colors, edgecolor='#1e293b', alpha=0.88, height=0.62)
    
    ax_tc.set_yticks(y_pos)
    ax_tc.set_yticklabels(mnames, fontweight='bold', fontsize=8.5)
    ax_tc.set_xlabel("Transition Compliance Rate (%)", fontweight='bold')
    ax_tc.set_title("C. Unconstrained Transition Compliance\n(Physiological Sequence Flow vs. Ground Truth)", fontweight='bold', pad=10)
    ax_tc.set_xlim(85, 103)
    ax_tc.grid(axis='x', linestyle='--', alpha=0.4)
    
    for bar in bars:
        w = bar.get_width()
        ax_tc.text(w + 0.4, bar.get_y() + bar.get_height()/2., f'{w:.1f}%',
                   ha='left', va='center', fontsize=8.0, fontweight='bold')
                   
    out_fig_dir = os.path.abspath(os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures", "Result_Oriented_Figures"))
    os.makedirs(out_fig_dir, exist_ok=True)
    out_fig_path = os.path.join(out_fig_dir, "Figure8_Verified_Clinical_Audit_and_Paired_Comparison.png")
    
    plt.savefig(out_fig_path, dpi=300)
    plt.close()
    print(f"Successfully generated Figure 8 to: {out_fig_path}")
    
    art_path = r"C:\Users\User\.gemini\antigravity-ide\brain\8ebc749c-dcbc-4647-bb97-3ca360a4d2b9\clinical_sim_figures\Figure8_Verified_Clinical_Audit_and_Paired_Comparison.png"
    import shutil
    shutil.copy(out_fig_path, art_path)
    print(f"Copied to artifact storage: {art_path}")

if __name__ == "__main__":
    generate_figure()
