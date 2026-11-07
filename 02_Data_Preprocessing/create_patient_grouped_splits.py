"""
========================================================================================
PATIENT-GROUPED DATASET PARTITIONING & AUDIT TOOL
========================================================================================
Implements strict patient-level grouping:
  - Consolidates patient identities (chb17a and chb17b -> chb17).
  - Ensures 0% patient ID overlap across Train, Validation, and Testing splits.
  - Analyzes the class distribution and highlights the physiological bottleneck of
    Inter-Ictal (Class 3) clustered seizures (present only in chb12 and chb24).
  - Exports:
      1. patient_grouped_split_manifest.csv
      2. patient_grouped_split_audit_report.md
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent if (CURRENT_DIR.parent / "src").is_dir() else CURRENT_DIR
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
DATA_SPLITS_DIR = PROJECT_ROOT / "05 CSV_and_TXT" / "Data_Splits_and_Audit"
MANIFEST_PATH = str(DATA_SPLITS_DIR / "dataset_split_manifest.csv")

def run_patient_grouping():
    print("=" * 80)
    print("ANALYZING AND GENERATING STRICT PATIENT-GROUPED SPLITS")
    print("=" * 80)

    df = pd.read_csv(MANIFEST_PATH)

    # 1. Normalize Patient IDs (unify chb17a and chb17b)
    df['patient_norm'] = df['patient'].apply(lambda p: 'chb17' if p in ['chb17a', 'chb17b'] else p)

    print("\n[Step 1] Unique normalized patients across CHB-MIT subset:")
    all_patients = sorted(df['patient_norm'].unique())
    print(f"Total Unique Patients: {len(all_patients)} -> {all_patients}")

    # Inspect patient-level distributions for Alpha Band
    df_alpha = df[df['band'] == 'Alpha Band (8-13hz)'].copy()
    patient_stats = df_alpha.groupby('patient_norm').agg({
        'file': 'nunique',
        'total_epochs': 'sum',
        'Normal_c0': 'sum',
        'Preictal_c1': 'sum',
        'Ictal_c2': 'sum',
        'Interictal_c3': 'sum',
        'Postictal_c4': 'sum'
    }).reset_index().rename(columns={'file': 'num_recordings'})

    print("\n[Step 2] Patient-level Epoch Distribution (Alpha Band):")
    print(patient_stats.to_string(index=False))

    # Physiological bottleneck check for Inter-Ictal (Class 3)
    patients_with_c3 = patient_stats[patient_stats['Interictal_c3'] > 0]['patient_norm'].tolist()
    print(f"\n[CRITICAL PHYSIOLOGICAL FINDING] Patients with Inter-Ictal (Class 3) epochs: {patients_with_c3}")
    print("Notice: ONLY chb12 and chb24 have clustered seizures with inter-ictal intervals <= 50 epochs.")
    print("Therefore, in any strict patient-grouped split:")
    print("  - If chb12 and chb24 are placed in Training, Test will have ZERO Class 3 epochs (identical to BroadBand).")
    print("  - If one is placed in Test (e.g., chb12) and one in Train (chb24), Test has Class 3 from exactly 1 patient.")

    # 3. Formulate an Optimal Patient-Grouped Partition
    # Target: ~70% Train, ~15% Val, ~15% Test
    # Test Patients (N=3): chb01, chb03, chb12 (provides 1,027 epochs, includes Class 3 via chb12)
    # Val Patients (N=3): chb05, chb14, chb20 (provides 815 epochs)
    # Train Patients (N=13): chb02, chb04, chb07, chb08, chb10, chb11, chb13, chb17, chb18, chb19, chb21, chb22, chb24 (provides 4,675 epochs, includes Class 3 via chb24)

    test_patients = ['chb01', 'chb03', 'chb12']
    val_patients = ['chb05', 'chb14', 'chb20']
    train_patients = [p for p in all_patients if p not in test_patients and p not in val_patients]

    print("\n[Step 3] Designed Patient-Grouped Partition:")
    print(f"  Training Patients ({len(train_patients)}): {train_patients}")
    print(f"  Validation Patients ({len(val_patients)}): {val_patients}")
    print(f"  Testing Patients ({len(test_patients)}): {test_patients}")

    # Verify 0% patient overlap
    set_tr, set_val, set_ts = set(train_patients), set(val_patients), set(test_patients)
    assert len(set_tr & set_val) == 0, "Train-Val overlap detected!"
    assert len(set_tr & set_ts) == 0, "Train-Test overlap detected!"
    assert len(set_val & set_ts) == 0, "Val-Test overlap detected!"
    print("  [VERIFIED] Exact 0% patient identity overlap across all partitions.")

    # Assign new splits
    df['patient_split'] = 'training'
    df.loc[df['patient_norm'].isin(val_patients), 'patient_split'] = 'validation'
    df.loc[df['patient_norm'].isin(test_patients), 'patient_split'] = 'testing'

    # Export Patient-Grouped Manifest
    out_manifest = str(DATA_SPLITS_DIR / "patient_grouped_split_manifest.csv")
    df.to_csv(out_manifest, index=False)
    print(f"  Saved Patient-Grouped Manifest to: {out_manifest}")

    # Summary table per band
    report_lines = [
        "# Patient-Grouped Split Audit & Feasibility Report",
        "",
        "## 1. Motivation & Leakage Resolution",
        "The primary audit identified that while recording filenames are disjoint across splits, patient IDs appear in multiple splits (13 shared patients between Train and Test in Alpha Band, 8 in All Band).",
        "This tool provides a **strict patient-disjoint partition** where 100% of recordings from any individual patient reside in exactly one split.",
        "",
        "## 2. Patient Partition Assignment",
        "- **Training Cohort (13 Patients)**: `chb02`, `chb04`, `chb07`, `chb08`, `chb10`, `chb11`, `chb13`, `chb17` (chb17a + chb17b), `chb18`, `chb19`, `chb21`, `chb22`, `chb24`",
        "- **Validation Cohort (3 Patients)**: `chb05`, `chb14`, `chb20`",
        "- **Testing Cohort (3 Patients)**: `chb01`, `chb03`, `chb12`",
        "",
        "## 3. Class Breakdown under Patient-Grouped Partition",
        ""
    ]

    for band in ['Alpha Band (8-13hz)', 'All Band (0.5-40hz)']:
        b_df = df[df['band'] == band]
        report_lines.append(f"### {band}")
        report_lines.append("| Split | Patients | Recordings | Total Epochs | Normal (c0) | Preictal (c1) | Ictal (c2) | Interictal (c3) | Postictal (c4) | Split Ratio |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        tot = len(b_df)
        tot_epochs = b_df['total_epochs'].sum()

        for sp in ['training', 'validation', 'testing']:
            sdf = b_df[b_df['patient_split'] == sp]
            n_p = sdf['patient_norm'].nunique()
            n_rec = len(sdf)
            ep = sdf['total_epochs'].sum()
            c0 = sdf['Normal_c0'].sum()
            c1 = sdf['Preictal_c1'].sum()
            c2 = sdf['Ictal_c2'].sum()
            c3 = sdf['Interictal_c3'].sum()
            c4 = sdf['Postictal_c4'].sum()
            ratio = (ep / tot_epochs) * 100.0 if tot_epochs > 0 else 0.0
            report_lines.append(f"| **{sp.capitalize()}** | {n_p} | {n_rec} | {ep:,} | {c0} ({c0/ep*100:.1f}%) | {c1} ({c1/ep*100:.1f}%) | {c2} ({c2/ep*100:.1f}%) | {c3} ({c3/ep*100:.1f}%) | {c4} ({c4/ep*100:.1f}%) | **{ratio:.1f}%** |")

        report_lines.append("")

    report_lines.extend([
        "## 4. Key Physiological & Methodological Findings",
        "1. **Inter-Ictal (Class 3) Sparsity**: In the CHB-MIT subset, clustered seizures exhibiting inter-ictal intervals (<= 50 epochs / 300 s) occur exclusively in patients `chb12` and `chb24`. In a patient-grouped split, if `chb12` is assigned to testing and `chb24` to training, the test set has Class 3 epochs from only one patient (257 epochs). If `chb12` were placed in training, the test set would have **zero** Class 3 epochs.",
        "2. **Dual-Evaluation Protocol Recommendation**: To provide complete scientific rigor for publication in *Journal of Clinical Neurophysiology* (JCN):",
        "   - **Primary Protocol**: Continuous Intra-Subject Temporal Staging across separate recording sessions (recording-held-out), as currently reported, evaluating temporal continuity, Markovian compliance, and advance forecasting.",
        "   - **Secondary Protocol**: Patient-Independent Cross-Subject Generalization on the zero-overlap cohort above, reporting domain-shift degradation honestly.",
        ""
    ])

    out_md = str(DATA_SPLITS_DIR / "patient_grouped_split_audit_report.md")
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"  [OK] Saved Patient-Grouped Audit Report to: {out_md}")

if __name__ == "__main__":
    run_patient_grouping()
