"""
========================================================================================
FULL-POWER BENCHMARK EXECUTION: ALPHA BAND (8-13 Hz) [v3 UPGRADED]
========================================================================================
Trains & rigorously evaluates all clinical deep learning architectures using:
  1. Per-recording baseline normalization (eliminating cross-patient impedance drift)
  2. Receptive temporal horizon T=15 (90.0s context)
  3. Upgraded Proposed Architecture: ST-ATMR-Net (Kinematic Spatial-Temporal Network)
  4. Dual-Context and Regularized Baselines (Standard LSTM, BiLSTM, TriLSTM, TCN1, TCN2)
  5. Warmup + Cosine Annealing learning rate schedule & Feature Gaussian Jittering
========================================================================================
"""

import os
import sys
import json
import torch
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent if (CURRENT_DIR.parent / "src").is_dir() else CURRENT_DIR
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders
from src.models import StandardLSTM, BiLSTM, TriLSTM, TCN1, TCN2, STATMRNet
from src.engine import fit_model
from src.metrics import plot_confusion_matrix, compute_all_metrics, measure_inference_latency
from src.utils import plot_learning_curves, plot_per_class_radar, format_benchmark_dataframe

def set_seed(seed=42):
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)

def run_alpha_band_experiments():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print(f"FULL-POWER BENCHMARK: ALPHA BAND (8-13 Hz) ON DEVICE: {device.upper()}")
    print("=" * 85)
    
    from pathlib import Path
    code_dir = CODE_DIR
    project_root = PROJECT_ROOT
    ckpt_dir = str(code_dir / "checkpoints" / "Alpha_Band")
    fig_dir = str(project_root / "04 Figures" / "Alpha_Band_Benchmark")
    tab_dir = str(project_root / "05 CSV_and_TXT" / "Model_Benchmarks_and_Ablations" / "Alpha_Band")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    
    # 1. Build Dataloaders with Per-Recording Normalization & T=15
    print("\n[Step 1/3] Building Alpha Band DataLoaders (Window T=15, 90.0s context, per_recording_norm=True)...")
    train_loader, val_loader, test_loader, smooth_weights, scaler, info = build_dataloaders(
        band_folder_name="Alpha Band (8-13hz)",
        window_size=15,
        batch_size=64,
        per_recording_norm=True
    )
    print(f"  Train: {info['num_train_sequences']:,} sequences | Val: {info['num_val_sequences']:,} | Test: {info['num_test_sequences']:,}")
    
    # Balanced weights
    weights = torch.tensor([1.0, 1.2, 1.0, 2.0, 1.1])
    
    models_dict = {
        'Standard_LSTM': StandardLSTM(input_dim=506, hidden_dim=128, dropout=0.3),
        'BiLSTM': BiLSTM(input_dim=506, hidden_dim=128, dropout=0.3),
        'TriLSTM': TriLSTM(input_dim=506, hidden_dim=128, dropout=0.3),
        'TCN_1': TCN1(input_dim=506, num_channels=[128, 128], dropout=0.3),
        'TCN_2': TCN2(input_dim=506, num_channels=[128, 128, 128], dropout=0.3),
        'STATMR_Net_Proposed': STATMRNet(input_dim=506, hidden_dim=128, dropout=0.25)
    }
    
    all_results = {}
    results_list = []
    
    # 2. Train & Evaluate Models with Full Power
    for idx, (m_name, model) in enumerate(models_dict.items(), 1):
        print(f"\n({idx}/{len(models_dict)}) Training: {m_name}...")
        p = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"  -> Model Parameters: {p:,}")
        
        res = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_weights=weights,
            model_name=f"Alpha_{m_name}",
            num_epochs=100,
            lr=1e-3,
            weight_decay=4e-3,
            gamma=1.2,
            label_smoothing=0.04,
            feature_noise=0.015,
            warmup_epochs=5,
            patience=15,
            checkpoint_dir=ckpt_dir,
            device=device,
            verbose=True
        )
        all_results[m_name] = res
        results_list.append(res)
        
        # Save Learning Curves
        lc_path = os.path.join(fig_dir, f"Learning_Curves_{m_name}.png")
        plot_learning_curves(res['history'], title=f"Alpha Band - {m_name}", save_path=lc_path)
        
        # Save Confusion Matrix
        cm_path = os.path.join(fig_dir, f"Confusion_Matrix_{m_name}.png")
        tm = res['test_metrics']
        plot_confusion_matrix(
            cm_norm=np.array(tm['confusion_matrix_norm']),
            cm_raw=np.array(tm['confusion_matrix']),
            title=f"Alpha Band - {m_name} (Macro-F1: {tm['macro_f1']*100:.2f}%)",
            save_path=cm_path
        )
        print(f"  -> Best Epoch: {res['best_epoch']}/{res['stopped_epoch']} | Val Bal Acc: {res['val_metrics']['balanced_accuracy']*100:.2f}% | Test Acc: {tm['accuracy']*100:.2f}% | Bal Acc: {tm['balanced_accuracy']*100:.2f}% | Macro-F1: {tm['macro_f1']*100:.2f}% | W-F1: {tm['weighted_f1']*100:.2f}%")

    # 3. Save Master Summary Table and Radar Chart
    print("\n[Step 3/3] Exporting Summary Tables & Comparative Visualizations for 6 Core Architectures...")
    df_summary = format_benchmark_dataframe(results_list, "Alpha Band (8-13 Hz)")
    csv_summary_path = os.path.join(tab_dir, "Alpha_Band_Benchmark_Summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"  [OK] Saved summary table to: {csv_summary_path}")
    
    radar_path = os.path.join(fig_dir, "Alpha_Band_Per_Class_Radar_Comparison.png")
    plot_per_class_radar(results_list, title="Alpha Band (8-13 Hz) - 5-Stage Clinical Recall", save_path=radar_path)
    print(f"  [OK] Saved radar comparison chart to: {radar_path}")
    
    json_path = os.path.join(tab_dir, "Alpha_Band_Metrics.json")
    with open(json_path, 'w') as f:
        json.dump({k: {m: v[m] for m in ['params', 'latency_ms_per_seq', 'test_metrics']} for k, v in all_results.items()}, f, indent=2)
    print(f"  [OK] Saved raw metrics JSON to: {json_path}")
    print("\n" + "=" * 85)
    print("ALPHA BAND BENCHMARK EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 85)
    return all_results

if __name__ == "__main__":
    run_alpha_band_experiments()
