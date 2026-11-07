"""
========================================================================================
FIGURE 5: 10-20 ELECTRODE CONFIGURATION AND 4-STAGE ALPHA-POWER TOPOGRAPHIC MAPS
========================================================================================
Translates machine learning features into intuitive clinical neurology data:
- Panel (A): 10-20 International Scalp Electrode System Configuration & Bipolar Chains
- Panel (B): 4-Stage Horizontal Continuum of Scalp Alpha-Band (8-13 Hz) Power Topography:
    1. Interictal: High, healthy posterior resting alpha power (warm red over O1/O2)
    2. Preictal: Thalamocortical alpha desynchronization (cool blues/greens, fading power)
    3. Ictal: Complete suppression of organized alpha rhythm, global paroxysmal surge
    4. Postictal: Postictal amplitude depression and slow recovery
100% Data-Driven: Computed directly from patient EEG signals (chb12_08.edf) via Welch PSD.
========================================================================================
"""

import os
import sys
import shutil
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent if (CURRENT_DIR.parent / "src").is_dir() else CURRENT_DIR
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

ROOT_DIR = str(PROJECT_ROOT)
CODE_DIR = str(CODE_DIR)
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent if (CURRENT_DIR.parent / "src").is_dir() else CURRENT_DIR
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

ROOT_DIR = str(PROJECT_ROOT)
CODE_DIR = str(CODE_DIR)
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arc
from scipy.signal import welch

# Publication typography
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 11.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 9.0
plt.rcParams['ytick.labelsize'] = 9.0
plt.rcParams['legend.fontsize'] = 9.0

def compute_real_alpha_psd():
    edf_path = r"E:\Seizure Detection\Dataset\Training\Seizure\chb12_08.edf"
    csv_path = os.path.abspath(os.path.join(ROOT_DIR, "03 Dataset", "Alpha Band (8-13hz)", "Preprocessed Data", "testing", "chb12_08.csv"))
    
    print("[1/3] Loading continuous EDF and calculating Welch Alpha Band PSD for all 10-20 electrodes...")
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    fs = int(raw.info['sfreq'])
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, -1].values
    
    chs_1020 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    mapping = {
        'Fp1': ['FP1-F7', 'FP1-F3'], 'Fp2': ['FP2-F8', 'FP2-F4'],
        'F7': ['FP1-F7', 'F7-T7'], 'F3': ['FP1-F3', 'F3-C3'], 'Fz': ['FZ-CZ'],
        'F4': ['FP2-F4', 'F4-C4'], 'F8': ['FP2-F8', 'F8-T8'],
        'T7': ['F7-T7', 'T7-P7'], 'C3': ['F3-C3', 'C3-P3'], 'Cz': ['FZ-CZ', 'CZ-PZ'],
        'C4': ['F4-C4', 'C4-P4'], 'T8': ['F8-T8', 'T8-P8'],
        'P7': ['T7-P7', 'P7-O1'], 'P3': ['C3-P3', 'P3-O1'], 'Pz': ['CZ-PZ'],
        'P4': ['C4-P4', 'P4-O2'], 'P8': ['T8-P8', 'P8-O2'],
        'O1': ['P7-O1', 'P3-O1'], 'O2': ['P8-O2', 'P4-O2']
    }
    
    stages = {'Interictal': 3, 'Preictal': 1, 'Ictal': 2, 'Postictal': 4}
    stage_psd = {s: [] for s in stages}
    
    for sname, scode in stages.items():
        eps = np.where(labels == scode)[0]
        powers_per_elec = []
        for elec in chs_1020:
            b_chs = mapping[elec]
            elec_powers = []
            for b_ch in b_chs:
                m = [i for i, c in enumerate(raw.ch_names) if b_ch in c]
                if not m:
                    continue
                ch_idx = m[0]
                for ep in eps:
                    t0, t1 = int(ep * 6.0 * fs), int((ep + 1) * 6.0 * fs)
                    sig = raw[ch_idx, t0:t1][0].flatten() * 1e6  # uV
                    f, pxx = welch(sig, fs=fs, nperseg=fs * 2)
                    alpha_mask = (f >= 8.0) & (f <= 13.0)
                    elec_powers.append(np.trapezoid(pxx[alpha_mask], f[alpha_mask]))
            powers_per_elec.append(np.mean(elec_powers) if elec_powers else 0.0)
        stage_psd[sname] = np.array(powers_per_elec)
        print(f"  {sname}: Mean={np.mean(stage_psd[sname]):.1f} uV^2 | O1={stage_psd[sname][chs_1020.index('O1')]:.1f} | O2={stage_psd[sname][chs_1020.index('O2')]:.1f}")
        
    return chs_1020, stage_psd

def draw_1020_layout(ax):
    """Draws a clean, publication-grade 10-20 electrode layout with Double Banana chains."""
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Head outline (circle)
    head_radius = 1.0
    circle = Circle((0, 0), head_radius, edgecolor='#1e293b', facecolor='#f8fafc', lw=2.0)
    ax.add_patch(circle)
    
    # Nose (triangle at top)
    nose = Polygon([[ -0.15, 0.98 ], [ 0.0, 1.18 ], [ 0.15, 0.98 ]],
                   edgecolor='#1e293b', facecolor='#f8fafc', lw=2.0, joinstyle='round')
    ax.add_patch(nose)
    
    # Left and Right Ears (arcs)
    ear_l = Arc((-1.02, 0), 0.22, 0.45, angle=0, theta1=90, theta2=270, edgecolor='#1e293b', lw=2.0)
    ear_r = Arc((1.02, 0), 0.22, 0.45, angle=0, theta1=270, theta2=90, edgecolor='#1e293b', lw=2.0)
    ax.add_patch(ear_l)
    ax.add_patch(ear_r)
    
    # Anatomical labels
    ax.text(0, 1.25, "Nasion (Anterior)", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#475569')
    ax.text(0, -1.14, "Inion (Posterior)", ha='center', va='top', fontsize=8.5, fontweight='bold', color='#475569')
    ax.text(-1.22, 0, "LPA\n(Left)", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#475569')
    ax.text(1.22, 0, "RPA\n(Right)", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#475569')
    
    # 10-20 Standard 2D Coordinates
    coords = {
        'Fp1': (-0.28, 0.78), 'Fp2': (0.28, 0.78),
        'F7': (-0.72, 0.52), 'F3': (-0.36, 0.46), 'Fz': (0.0, 0.46), 'F4': (0.36, 0.46), 'F8': (0.72, 0.52),
        'T7': (-0.82, 0.0), 'C3': (-0.42, 0.0), 'Cz': (0.0, 0.0), 'C4': (0.42, 0.0), 'T8': (0.82, 0.0),
        'P7': (-0.72, -0.52), 'P3': (-0.36, -0.46), 'Pz': (0.0, -0.46), 'P4': (0.36, -0.46), 'P8': (0.72, -0.52),
        'O1': (-0.28, -0.78), 'O2': (0.28, -0.78)
    }
    
    # Bipolar chains (Double Banana clinical montage)
    chains = [
        # Left Temporal
        (['Fp1', 'F7', 'T7', 'P7', 'O1'], '#2563eb'),
        # Left Parasagittal
        (['Fp1', 'F3', 'C3', 'P3', 'O1'], '#0284c7'),
        # Midline
        (['Fz', 'Cz', 'Pz'], '#475569'),
        # Right Parasagittal
        (['Fp2', 'F4', 'C4', 'P4', 'O2'], '#059669'),
        # Right Temporal
        (['Fp2', 'F8', 'T8', 'P8', 'O2'], '#16a34a')
    ]
    
    for ch_nodes, color in chains:
        xs = [coords[n][0] for n in ch_nodes]
        ys = [coords[n][1] for n in ch_nodes]
        ax.plot(xs, ys, color=color, linewidth=1.5, linestyle='-', alpha=0.75, zorder=2)
        
    # Draw electrode nodes
    for name, (x, y) in coords.items():
        is_occipital = name in ['O1', 'O2']
        fc = '#fee2e2' if is_occipital else '#ffffff'
        ec = '#b91c1c' if is_occipital else '#1e293b'
        lw = 2.0 if is_occipital else 1.2
        
        node = Circle((x, y), 0.088, facecolor=fc, edgecolor=ec, lw=lw, zorder=4)
        ax.add_patch(node)
        ax.text(x, y, name, ha='center', va='center', fontsize=7.8, fontweight='bold',
                color='#991b1b' if is_occipital else '#0f172a', zorder=5)
        
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.30, 1.35)

def generate_figure5():
    chs_1020, stage_psd = compute_real_alpha_psd()
    
    print("[2/3] Setting up MNE montage and topomap parameters...")
    info = mne.create_info(chs_1020, sfreq=256, ch_types='eeg')
    info.set_montage('standard_1020')
    
    # 1 row, 2 main sections: Left (Panel A), Right (Panel B with 4 topomaps + colorbar)
    fig = plt.figure(figsize=(16.5, 6.8), dpi=300)
    gs_main = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.4], wspace=0.20,
                                left=0.08, right=0.95, top=0.88, bottom=0.08)
    
    # Panel A: 10-20 Layout
    ax_layout = fig.add_subplot(gs_main[0, 0])
    draw_1020_layout(ax_layout)
    
    # Panel B: 4 Topomaps + Colorbar
    gs_topo = gs_main[0, 1].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.08], wspace=0.12)
    
    stage_keys = [
        ('Interictal', '1. Interictal Baseline\n(Resting Alpha Rhythm)'),
        ('Preictal', '2. Pre-Ictal Warning\n(Thalamocortical ERD)'),
        ('Ictal', '3. Ictal Seizure\n(Paroxysmal Disruption)'),
        ('Postictal', '4. Post-Ictal Phase\n(Amplitude Depression)')
    ]
    
    all_vals = np.concatenate([stage_psd[k[0]] for k in stage_keys])
    vmin = np.percentile(all_vals, 2)
    vmax = np.percentile(all_vals, 98)
    
    print(f"  Uniform Topomap Range: {vmin:.1f} to {vmax:.1f} uV^2/Hz")
    
    last_im = None
    for idx, (skey, stitle) in enumerate(stage_keys):
        ax = fig.add_subplot(gs_topo[0, idx])
        data = stage_psd[skey]
        im, _ = mne.viz.plot_topomap(
            data, info, axes=ax, show=False,
            cmap='turbo', vlim=(vmin, vmax),
            contours=6, sphere=(0, 0, 0, 0.095)
        )
        last_im = im
        ax.set_title(stitle, fontweight='bold', fontsize=9.2, pad=10)
        
    # Shared Colorbar
    cbar_ax = fig.add_subplot(gs_topo[0, 4])
    cbar = plt.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Scalp Alpha-Band PSD (\u03bcV\u00b2 / Hz)", fontweight='bold', fontsize=9.5, labelpad=8)
    cbar.ax.tick_params(labelsize=8.5)
    
    fig.text(0.20, 0.95, "A. 10–20 International Scalp Electrode Placement", 
             fontweight='bold', fontsize=12.0, ha='center')
    fig.text(0.65, 0.95, "B. 4-Stage Scalp Alpha-Band (8–13 Hz) Power Topographic Continuum", 
             fontweight='bold', fontsize=12.0, ha='center')
    
    out_dir = os.path.abspath(os.path.join(ROOT_DIR, "04 Figures", "Publication_Figures", "Result_Oriented_Figures"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Figure5_Electrode_Layout_and_Alpha_Topography_Continuum.png")
    
    print("[3/3] Saving high-resolution publication figure (300 DPI)...")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Successfully saved Figure 5 to:\n  {out_path}")
    
    # Copy to artifact storage
    art_path = r"C:\Users\User\.gemini\antigravity-ide\brain\8ebc749c-dcbc-4647-bb97-3ca360a4d2b9\clinical_sim_figures\Figure5_Electrode_Layout_and_Alpha_Topography_Continuum.png"
    if os.path.exists(os.path.dirname(art_path)):
        shutil.copy(out_path, art_path)

if __name__ == "__main__":
    generate_figure5()
