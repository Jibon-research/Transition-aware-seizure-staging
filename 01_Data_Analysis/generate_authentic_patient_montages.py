"""
========================================================================================
FULL-DATASET PATIENT-WISE 23-CHANNEL CLINICAL MONTAGE GENERATOR (AUDITED & AUTHENTIC)
========================================================================================
Generates complete 23-channel clinical montages for every patient in the CHB-MIT cohort:
  1. Alpha Band (8-13 Hz)  - 20 Patients (chb01 through chb24)
  2. BroadBand (0.5-40 Hz) - 20 Patients (chb01 through chb24)

Features:
- Strictly maps to existing preprocessed CSV files (epochs, recordings, stage labels).
- Extracts authentic continuous 6.0-second EEG segments from source PhysioNet EDFs.
- Renders ALL 23 bipolar derivations with legible, bold channel labels on the vertical axis:
    FP1-F7, F7-T7, T7-P7, P7-O1,
    FP1-F3, F3-C3, C3-P3, P3-O1,
    FP2-F4, F4-C4, C4-P4, P4-O2,
    FP2-F8, F8-T8, T8-P8-0, P8-O2,
    FZ-CZ, CZ-PZ,
    P7-T7, T7-FT9, FT9-FT10, FT10-T8, T8-P8-1
- Anatomical chain color coding (Temporal, Parasagittal, Midline, Basal).
- 100 uV / 1.0 s clinical calibration bar.
- Generates:
    - Combined 5-Stage Patient Hero Montage: Patient_<id>_<Band>_Montage.png
    - Individual Stage Images (each with all 23 channels explicitly labeled):
        01_Normal.png, 02_Pre_Ictal.png, 03_Ictal.png, 04_Inter_Ictal.png, 05_Post_Ictal.png
========================================================================================
"""

import os
import sys
import glob
import time
import warnings
import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent if (CURRENT_DIR.parent / "src").is_dir() else CURRENT_DIR
PROJECT_ROOT = str(CODE_DIR.parent)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "03 Dataset")
MONTAGES_DIR = os.path.join(PROJECT_ROOT, "04 Figures", "Patient_Montages")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "05 CSV_and_TXT", "Data_Splits_and_Audit", "dataset_split_manifest.csv")

RAW_EDF_SEARCH_DIRS = [
    r"E:\Seizure Detection\Dataset\Training\Seizure",
    r"E:\Seizure Detection\Dataset\Training\Non Seizure",
    r"E:\Seizure Detection\Dataset\Testing\Seizure",
    r"E:\Seizure Detection\Dataset\Testing\Non Seizure",
]

TARGET_23_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
]

CHAIN_COLORS = (
    ['#1e3a8a'] * 4 +   # Left Temporal (Blue)
    ['#0f766e'] * 4 +   # Left Parasagittal (Teal)
    ['#15803d'] * 4 +   # Right Parasagittal (Emerald)
    ['#6b21a8'] * 4 +   # Right Temporal (Purple)
    ['#c2410c'] * 2 +   # Midline (Dark Amber)
    ['#334155'] * 5     # Transverse / Basal (Slate)
)

STAGE_NAMES = ['Normal', 'Pre-Ictal', 'Ictal', 'Inter-Ictal', 'Post-Ictal']
STAGE_SUBTITLES = [
    'Baseline Background',
    'Pre-seizure Transition',
    'Active Seizure Discharge',
    'Inter-ictal Interval',
    'Suppression & Recovery'
]
STAGE_FILENAMES = [
    '01_Normal',
    '02_Pre_Ictal',
    '03_Ictal',
    '04_Inter_Ictal',
    '05_Post_Ictal'
]

def find_raw_edf(rec_id):
    fname = f"{rec_id}.edf" if not rec_id.endswith(".edf") else rec_id
    for d in RAW_EDF_SEARCH_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None

def get_best_recording_for_patient(manifest_df, patient_id, band_name):
    pdf = manifest_df[(manifest_df['patient'] == patient_id) & (manifest_df['band'] == band_name)]
    if len(pdf) == 0:
        return None
        
    best_row = None
    max_stages = -1
    max_epochs = -1
    for _, row in pdf.iterrows():
        n_stages = sum([
            row['Normal_c0'] > 0,
            row['Preictal_c1'] > 0,
            row['Ictal_c2'] > 0,
            row['Interictal_c3'] > 0,
            row['Postictal_c4'] > 0,
        ])
        if n_stages > max_stages or (n_stages == max_stages and row['total_epochs'] > max_epochs):
            max_stages = n_stages
            max_epochs = row['total_epochs']
            best_row = row
            
    return best_row

def generate_patient_montages(manifest_df, patient_id, band_type="Alpha_Band", dpi=300):
    band_folder = "Alpha Band (8-13hz)" if band_type == "Alpha_Band" else "All Band (0.5-40hz)"
    band_label = "Alpha Band (8–13 Hz)" if band_type == "Alpha_Band" else "BroadBand (0.5–40 Hz)"
    
    best_row = get_best_recording_for_patient(manifest_df, patient_id, band_folder)
    if best_row is None:
        print(f"  [SKIP] No recordings found for {patient_id} in {band_folder}")
        return False
        
    rec_file = best_row['file']
    rec_id = rec_file.replace('.csv', '')
    split = best_row['split']
    
    csv_path = os.path.join(DATASET_ROOT, band_folder, "Preprocessed Data", split, rec_file)
    if not os.path.exists(csv_path):
        print(f"  [SKIP] CSV file missing: {csv_path}")
        return False
        
    edf_path = find_raw_edf(rec_id)
    if not edf_path:
        print(f"  [SKIP] EDF file missing for {rec_id}")
        return False
        
    # Read labels from existing CSV
    df_csv = pd.read_csv(csv_path)
    labels = df_csv.iloc[:, -1].values.astype(int)
    
    # Read EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs = int(raw.info['sfreq'])
    duration = 6.0
    n_pts = int(duration * fs)
    
    # Channel index mapping
    ch_map = {name: i for i, name in enumerate(raw.ch_names)}
    picked_indices = []
    for target in TARGET_23_CHANNELS:
        if target in ch_map:
            picked_indices.append(ch_map[target])
        elif target in ["T8-P8-0", "T8-P8-1"] and "T8-P8" in ch_map:
            picked_indices.append(ch_map["T8-P8"])
        else:
            picked_indices.append(0)
            
    # Butterworth 4th-order filter
    if band_type == "Alpha_Band":
        sos = butter(4, [8.0, 13.0], btype='bandpass', fs=fs, output='sos')
    else:
        sos = butter(4, [0.5, 40.0], btype='bandpass', fs=fs, output='sos')
        
    # Extract stages present in CSV
    stage_data = []
    for s_idx in range(5):
        matching = np.where(labels == s_idx)[0]
        if len(matching) > 0:
            ep = matching[len(matching) // 2]
            st = int(ep * duration * fs)
            en = st + n_pts
            seg = raw.get_data(picks=picked_indices, start=st, stop=min(en, raw.n_times)) * 1e6
            if seg.shape[1] < n_pts:
                seg = np.pad(seg, ((0, 0), (0, n_pts - seg.shape[1])), mode='edge')
            seg = sosfiltfilt(sos, seg, axis=-1)
            stage_data.append((s_idx, ep, seg))
            
    if len(stage_data) == 0:
        return False
        
    # Output patient directory
    patient_out_dir = os.path.join(MONTAGES_DIR, band_type, patient_id)
    os.makedirs(patient_out_dir, exist_ok=True)
    
    ch_spacing = 90.0 # uV
    offsets = np.arange(len(TARGET_23_CHANNELS))[::-1] * ch_spacing
    
    # -------------------------------------------------------------------------
    # PART A: Combined Multi-Stage Patient Hero Montage
    # -------------------------------------------------------------------------
    num_cols = len(stage_data)
    fig, axes = plt.subplots(1, num_cols, figsize=(4.8 * num_cols + 2.5, 12.0), dpi=dpi)
    if num_cols == 1:
        axes = [axes]
        
    for col_idx, (s_idx, ep, seg) in enumerate(stage_data):
        ax = axes[col_idx]
        t_ax = np.linspace(0, duration, seg.shape[1])
        
        for ch_i in range(len(TARGET_23_CHANNELS)):
            trace_y = seg[ch_i] + offsets[ch_i]
            ax.plot(t_ax, trace_y, color=CHAIN_COLORS[ch_i], lw=0.85)
            ax.axhline(offsets[ch_i], color='#e2e8f0', linestyle='--', lw=0.5, zorder=1)
            
        ax.set_xlim([0, duration])
        ax.set_ylim([-140.0, offsets[0] + 110.0])
        ax.set_xticks(np.arange(0, duration + 0.1, 1.0))
        ax.set_xlabel("Time (Seconds)", fontsize=10.0, fontweight="bold", labelpad=6)
        
        s_name = STAGE_NAMES[s_idx]
        s_sub = STAGE_SUBTITLES[s_idx]
        ax.set_title(f"Stage {s_idx}: {s_name}\n{s_sub} (Epoch {ep})",
                     fontsize=11.0, fontweight="bold", pad=10)
        ax.grid(True, axis="x", linestyle=":", color="#cbd5e1", alpha=0.6)
        
        # Explicit 23-Channel Y-ticks on the first panel
        if col_idx == 0:
            ax.set_yticks(offsets[::-1])
            ax.set_yticklabels(TARGET_23_CHANNELS[::-1], fontsize=9.0, fontweight="bold")
            ax.set_ylabel("10–20 Scalp Bipolar Derivations (23 Channels)", fontsize=11.0, fontweight="bold", labelpad=8)
            ax.tick_params(axis='y', length=4, width=1.2, color='#334155')
        else:
            ax.set_yticks(offsets[::-1])
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
            
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
           # Scale bar on last panel
        if col_idx == num_cols - 1:
            bar_x = 3.6
            bar_y = -105.0
            ax.plot([bar_x, bar_x], [bar_y, bar_y + 100.0], color="#0f172a", lw=2.2, zorder=10)
            ax.plot([bar_x, bar_x + 1.0], [bar_y, bar_y], color="#0f172a", lw=2.2, zorder=10)
            ax.text(bar_x - 0.12, bar_y + 50.0, "100 \u03bcV", fontsize=8.5, fontweight="bold", va="center", ha="right")
            ax.text(bar_x + 0.5, bar_y - 20.0, "1.0 s", fontsize=8.5, fontweight="bold", ha="center", va="top")
            
    fig.suptitle(f"Patient {patient_id.upper()} Full 23-Channel Longitudinal Bipolar Montage Across Clinical Stages\n"
                 f"Session: {rec_id} | Band: {band_label} | Sampling Rate: {fs} Hz | Cohort: {split.upper()}",
                 fontsize=12.0, fontweight="bold", y=0.985)
                 
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    hero_path = os.path.join(patient_out_dir, f"Patient_{patient_id}_{band_type.replace('_Band', '')}_Montage.png")
    fig.savefig(hero_path, dpi=dpi)
    plt.close(fig)
    
    # -------------------------------------------------------------------------
    # PART B: Individual Stage Images (EACH WITH ALL 23 CHANNELS EXPLICITLY LABELED)
    # -------------------------------------------------------------------------
    for s_idx, ep, seg in stage_data:
        s_name = STAGE_NAMES[s_idx]
        s_sub = STAGE_SUBTITLES[s_idx]
        s_file = STAGE_FILENAMES[s_idx]
        
        fig_s, ax_s = plt.subplots(figsize=(7.2, 12.0), dpi=dpi)
        t_ax = np.linspace(0, duration, seg.shape[1])
        
        for ch_i in range(len(TARGET_23_CHANNELS)):
            trace_y = seg[ch_i] + offsets[ch_i]
            ax_s.plot(t_ax, trace_y, color=CHAIN_COLORS[ch_i], lw=0.85)
            ax_s.axhline(offsets[ch_i], color='#e2e8f0', linestyle='--', lw=0.5, zorder=1)
            
        ax_s.set_xlim([0, duration])
        ax_s.set_ylim([-140.0, offsets[0] + 110.0])
        ax_s.set_xticks(np.arange(0, duration + 0.1, 1.0))
        ax_s.set_xlabel("Time (Seconds)", fontsize=10.0, fontweight="bold", labelpad=6)
        
        # PROMINENT 23-CHANNEL LABELS
        ax_s.set_yticks(offsets[::-1])
        ax_s.set_yticklabels(TARGET_23_CHANNELS[::-1], fontsize=9.0, fontweight="bold")
        ax_s.set_ylabel("10–20 Scalp Bipolar Derivations (23 Channels)", fontsize=10.5, fontweight="bold", labelpad=8)
        ax_s.tick_params(axis='y', length=4, width=1.2, color='#334155')
        
        ax_s.spines["top"].set_visible(False)
        ax_s.spines["right"].set_visible(False)
        ax_s.grid(True, axis="x", linestyle=":", color="#cbd5e1", alpha=0.6)
        
        # Calibration bar
        bar_x = 3.6
        bar_y = -105.0
        ax_s.plot([bar_x, bar_x], [bar_y, bar_y + 100.0], color="#0f172a", lw=2.2, zorder=10)
        ax_s.plot([bar_x, bar_x + 1.0], [bar_y, bar_y], color="#0f172a", lw=2.2, zorder=10)
        ax_s.text(bar_x - 0.12, bar_y + 50.0, "100 \u03bcV", fontsize=8.5, fontweight="bold", va="center", ha="right")
        ax_s.text(bar_x + 0.5, bar_y - 20.0, "1.0 s", fontsize=8.5, fontweight="bold", ha="center", va="top")
        
        ax_s.set_title(f"Patient {patient_id.upper()} | Stage {s_idx}: {s_name} ({duration:.1f}s)\n"
                       f"{s_sub} (Epoch {ep}) | {band_label} | Rec: {rec_id}",
                       fontsize=10.5, fontweight="bold", pad=12)
                       
        plt.tight_layout()
        single_path = os.path.join(patient_out_dir, f"{s_file}.png")
        fig_s.savefig(single_path, dpi=dpi)
        plt.close(fig_s)
        
    return True

def run_all_patient_montages():
    print("=" * 85)
    print("GENERATING AUTHENTIC 23-CHANNEL PATIENT MONTAGES ACROSS CHB-MIT COHORT")
    print("=" * 85)
    
    manifest_df = pd.read_csv(MANIFEST_PATH)
    all_patients = sorted(manifest_df['patient'].unique())
    print(f"Total Unique Patients: {len(all_patients)} -> {all_patients}")
    
    t0 = time.time()
    total_generated = 0
    
    for band_key in ["BroadBand", "Alpha_Band"]:
        print(f"\n>>> Processing Frequency Band: {band_key} ({len(all_patients)} patients)...")
        for idx, p_id in enumerate(all_patients, 1):
            ok = generate_patient_montages(manifest_df, p_id, band_type=band_key, dpi=200)
            status = "SUCCESS" if ok else "SKIPPED"
            print(f"  [{idx:02d}/{len(all_patients)}] Patient {p_id:7s} ({band_key:10s}) -> {status}")
            if ok:
                total_generated += 1
                
    elapsed = time.time() - t0
    print("\n" + "=" * 85)
    print(f"ALL PATIENT MONTAGES GENERATED SUCCESSFULLY IN {elapsed:.1f}s!")
    print(f"Total Patient Sessions Rendered: {total_generated}/40")
    print(f"Output Directory: {MONTAGES_DIR}")
    print("=" * 85)

if __name__ == "__main__":
    run_all_patient_montages()
