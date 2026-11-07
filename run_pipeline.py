#!/usr/bin/env python
"""
========================================================================================
MASTER REPRODUCIBILITY PIPELINE & EXECUTION HARNESS
========================================================================================
Central coordinator for running all stages of the research pipeline:
  1. Data Analysis (01_Data_Analysis)
  2. Data Preprocessing (02_Data_Preprocessing)
  3. Model Training & Baselines (03_Model_Training_Baselines)
  4. Evaluations & Publication Figures (04_Evaluations)
  5. Explainable AI Analysis (05_XAI_Analysis)
  6. Interactive Notebooks (06_Notebooks)

Usage:
  python run_pipeline.py --stage verify        # Fast checkpoint & integrity verification
  python run_pipeline.py --stage analysis      # Reproduces all data analysis & methodology figures (Figs 1, 2, 5)
  python run_pipeline.py --stage preprocessing # Reproduces patient-grouped partition & audit
  python run_pipeline.py --stage evaluations   # Reproduces all benchmark tables, diagnostic suites & Figs 6, 7, 8
  python run_pipeline.py --stage xai           # Reproduces multi-stage model XAI saliency topomaps
  python run_pipeline.py --stage all           # Runs full evaluation & visualization pipeline
========================================================================================
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

STAGES = {
    'verify': {
        'desc': 'Validate checkpoint integrity and model forward inference',
        'commands': [
            (CODE_DIR / '04_Evaluations' / 'verify_and_load_checkpoints.py', 'PyTorch 18-Checkpoint Validation'),
        ]
    },
    'analysis': {
        'desc': 'Data Analysis: Generate clinical timeline (Fig 1), extraction pipeline (Fig 2), alpha topography (Fig 5), montages',
        'commands': [
            (CODE_DIR / '01_Data_Analysis' / 'generate_trustworthy_figure1_physionet_offset.py', 'Figure 1: Clinical Evolution Timeline'),
            (CODE_DIR / '01_Data_Analysis' / 'generate_publication_figure2_broadband_to_alpha.py', 'Figure 2: BroadBand to Alpha Decomposition'),
            (CODE_DIR / '01_Data_Analysis' / 'generate_clean_publication_figure5.py', 'Figure 5: 10-20 Scalp Electrode Layout & Alpha Topography'),
        ]
    },
    'preprocessing': {
        'desc': 'Data Preprocessing: Execute strict patient-grouped partition and cross-patient leakage audit',
        'commands': [
            (CODE_DIR / '02_Data_Preprocessing' / 'create_patient_grouped_splits.py', 'Patient-Grouped Splits & Leakage Audit'),
        ]
    },
    'evaluations': {
        'desc': 'Evaluations: Generate Figures 6 & 7, Figure 8, 2x6 ROC & Calibration, and 12-Model Confusion Matrices',
        'commands': [
            (CODE_DIR / '04_Evaluations' / 'verify_and_load_checkpoints.py', 'Checkpoint Validation'),
            (CODE_DIR / '04_Evaluations' / 'generate_audited_figures6_and_7.py', 'Figures 6 & 7: Multi-Seed Boxplots & Ablation Drop'),
            (CODE_DIR / '04_Evaluations' / 'generate_verification_and_paired_comparison_figure.py', 'Figure 8: Paired Comparison & Compliance'),
            (CODE_DIR / '04_Evaluations' / 'generate_combined_12_model_confusion_matrices.py', 'Combined 12-Model Confusion Matrices (2x6)'),
            (CODE_DIR / '04_Evaluations' / 'generate_combined_radar_comparison.py', 'Combined 1x2 Clinical Radar Benchmark'),
        ]
    },
    'xai': {
        'desc': 'XAI Analysis: Generate authentic 5-stage scalp saliency topomaps & class probabilities',
        'commands': [
            (CODE_DIR / '05_XAI_Analysis' / 'generate_authentic_patient_5stage_signals_and_model_xai.py', '5-Stage Authentic Signal & Model XAI Suite'),
        ]
    }
}

def run_script(script_path, label):
    print("\n" + "=" * 80)
    print(f"RUNNING: {label}")
    print(f"Script:  {script_path.relative_to(CODE_DIR)}")
    print("=" * 80)
    
    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, cwd=str(script_path.parent))
    if result.returncode != 0:
        print(f"[ERROR] Script failed with exit code {result.returncode}: {script_path.name}")
        return False
    print(f"[SUCCESS] Completed: {label}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline for Clinical EEG Research Project")
    parser.add_argument('--stage', choices=['verify', 'analysis', 'preprocessing', 'evaluations', 'xai', 'all'],
                        default='verify', help="Pipeline stage to execute (default: verify)")
    parser.add_argument('--list', action='store_true', help="List all available stages and scripts")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Pipeline Stages:")
        for stage, data in STAGES.items():
            print(f"\n  --stage {stage}: {data['desc']}")
            for p, name in data['commands']:
                print(f"      • {name} ({p.relative_to(CODE_DIR)})")
        print("\n  --stage all: Runs analysis, preprocessing, evaluations, and xai sequentially.\n")
        return

    stages_to_run = ['analysis', 'preprocessing', 'evaluations', 'xai'] if args.stage == 'all' else [args.stage]
    
    total = sum(len(STAGES[s]['commands']) for s in stages_to_run)
    success = 0
    
    print("\n" + "#" * 80)
    print(f"# EXECUTING PIPELINE STAGE: {args.stage.upper()}")
    print("#" * 80)
    
    for stage_name in stages_to_run:
        stage = STAGES[stage_name]
        print(f"\n>>> Starting Stage: {stage_name.upper()} - {stage['desc']}")
        for script_path, label in stage['commands']:
            if run_script(script_path, label):
                success += 1
            else:
                print(f"\n[ABORT] Pipeline failed on stage {stage_name} ({script_path.name}).")
                sys.exit(1)
                
    print("\n" + "#" * 80)
    print(f"# PIPELINE EXECUTION COMPLETED: {success}/{total} tasks succeeded!")
    print("#" * 80 + "\n")

if __name__ == '__main__':
    main()

