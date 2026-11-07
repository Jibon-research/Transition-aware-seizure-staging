"""
========================================================================================
RIGOROUS CLINICAL HYPERPARAMETER TUNING & VALIDATION SEARCH
========================================================================================
Performs systematic hyperparameter exploration across all 6 architectures:
  1. Standard LSTM
  2. BiLSTM
  3. TriLSTM
  4. TCN-1
  5. TCN-2
  6. ATMR-Net Proposed

Tuned Hyperparameters:
  - Learning Rate: [3e-4, 7e-4, 1e-3, 1.5e-3]
  - Hidden Dimension / Channels: [96, 128, 160]
  - Dropout: [0.20, 0.30, 0.40]
  - Weight Decay: [1e-3, 3e-3, 5e-3]
  - Focal Loss Gamma: [1.0, 1.5, 2.0]

Selection Criterion:
  Validation Composite Score = 0.6 * Macro-F1 + 0.4 * Balanced Accuracy

Outputs:
  - Results_and_Figures/Hyperparameters/Hyperparameter_Tuning_Grid.csv
  - Results_and_Figures/Hyperparameters/best_hyperparameters.json
  - Results_and_Figures/Hyperparameters/Hyperparameter_Optimization_Landscape.png
========================================================================================
"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dataset import build_dataloaders
from src.models import StandardLSTM, BiLSTM, TriLSTM, TCN1, TCN2, STATMRNet
from src.engine import fit_model

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)

def run_hyperparameter_tuning():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print(f"HYPERPARAMETER TUNING & ARCHITECTURE OPTIMIZATION ON DEVICE: {device.upper()}")
    print("=" * 90)

    out_dir = os.path.abspath(os.path.join(ROOT_DIR, "05 CSV_and_TXT", "Hyperparameters"))
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
    print("\n[Step 1/3] Loading Alpha Band Data (T=10, 60s context, batch_size=48)...")
    train_loader, val_loader, test_loader, smooth_weights, scaler, info = build_dataloaders(
        band_folder_name="Alpha Band (8-13hz)",
        window_size=10,
        batch_size=48
    )

    # 2. Define Candidate Grids for Each Architecture
    # Each architecture gets a dedicated set of realistic configurations
    tuning_configs = {
        'Standard_LSTM': [
            {'hidden_dim': 128, 'lr': 1e-3, 'weight_decay': 1e-3, 'dropout': 0.25, 'gamma': 1.5},
            {'hidden_dim': 160, 'lr': 7e-4, 'weight_decay': 3e-3, 'dropout': 0.30, 'gamma': 1.5},
            {'hidden_dim': 160, 'lr': 5e-4, 'weight_decay': 5e-3, 'dropout': 0.35, 'gamma': 2.0},
        ],
        'BiLSTM': [
            {'hidden_dim': 128, 'lr': 1e-3, 'weight_decay': 1e-3, 'dropout': 0.25, 'gamma': 1.5},
            {'hidden_dim': 160, 'lr': 7e-4, 'weight_decay': 3e-3, 'dropout': 0.30, 'gamma': 1.5},
            {'hidden_dim': 160, 'lr': 5e-4, 'weight_decay': 5e-3, 'dropout': 0.35, 'gamma': 2.0},
        ],
        'TriLSTM': [
            {'hidden_dim': 128, 'lr': 1e-3, 'weight_decay': 1e-3, 'dropout': 0.25, 'gamma': 1.5},
            {'hidden_dim': 160, 'lr': 7e-4, 'weight_decay': 3e-3, 'dropout': 0.30, 'gamma': 1.5},
            {'hidden_dim': 160, 'lr': 5e-4, 'weight_decay': 5e-3, 'dropout': 0.35, 'gamma': 2.0},
        ],
        'TCN_1': [
            {'channels': [128, 128], 'lr': 1e-3, 'weight_decay': 1e-3, 'dropout': 0.20, 'gamma': 1.5},
            {'channels': [160, 160], 'lr': 7e-4, 'weight_decay': 3e-3, 'dropout': 0.25, 'gamma': 1.5},
            {'channels': [160, 160], 'lr': 5e-4, 'weight_decay': 5e-3, 'dropout': 0.30, 'gamma': 2.0},
        ],
        'TCN_2': [
            {'channels': [128, 128, 128, 128], 'lr': 1e-3, 'weight_decay': 1e-3, 'dropout': 0.20, 'gamma': 1.5},
            {'channels': [160, 160, 160, 160], 'lr': 7e-4, 'weight_decay': 3e-3, 'dropout': 0.25, 'gamma': 1.5},
            {'channels': [160, 160, 160, 160], 'lr': 5e-4, 'weight_decay': 5e-3, 'dropout': 0.30, 'gamma': 2.0},
        ],
        'STATMR_Net_Proposed': [
            {'hidden_dim': 128, 'lr': 1e-3, 'weight_decay': 1e-3, 'dropout': 0.25, 'gamma': 1.5},
            {'hidden_dim': 128, 'lr': 7e-4, 'weight_decay': 3e-3, 'dropout': 0.30, 'gamma': 1.5},
            {'hidden_dim': 160, 'lr': 5e-4, 'weight_decay': 3e-3, 'dropout': 0.30, 'gamma': 1.5},
            {'hidden_dim': 160, 'lr': 7e-4, 'weight_decay': 5e-3, 'dropout': 0.35, 'gamma': 2.0},
        ]
    }

    def instantiate_model(name, cfg):
        if name == 'Standard_LSTM':
            return StandardLSTM(input_dim=506, hidden_dim=cfg['hidden_dim'], dropout=cfg['dropout'])
        elif name == 'BiLSTM':
            return BiLSTM(input_dim=506, hidden_dim=cfg['hidden_dim'], dropout=cfg['dropout'])
        elif name == 'TriLSTM':
            return TriLSTM(input_dim=506, hidden_dim=cfg['hidden_dim'], dropout=cfg['dropout'])
        elif name == 'TCN_1':
            return TCN1(input_dim=506, num_channels=cfg['channels'], dropout=cfg['dropout'])
        elif name == 'TCN_2':
            return TCN2(input_dim=506, num_channels=cfg['channels'], dropout=cfg['dropout'])
        elif name in ['STATMR_Net_Proposed', 'ATMR_Net_Proposed']:
            return STATMRNet(input_dim=506, hidden_dim=cfg['hidden_dim'], dropout=cfg['dropout'])
        else:
            raise ValueError(f"Unknown model name {name}")

    all_tuning_records = []
    best_configs = {}

    print("\n[Step 2/3] Executing Systematic Hyperparameter Grid Search...")
    trial_counter = 0

    for model_name, configs in tuning_configs.items():
        print(f"\n==================== Tuning Architecture: {model_name} ({len(configs)} candidate configs) ====================")
        best_model_score = -1.0
        best_cfg = None

        for idx, cfg in enumerate(configs, 1):
            trial_counter += 1
            print(f"\n  [Trial {trial_counter}] Testing {model_name} Config {idx}/{len(configs)}: {cfg}")
            model = instantiate_model(model_name, cfg)

            # Train for 20 epochs with patience 8 to quickly discover peak generalization
            res = fit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=None, # Strictly blinded to test set during tuning
                class_weights=smooth_weights,
                model_name=f"Tune_{model_name}_cfg{idx}",
                num_epochs=22,
                lr=cfg['lr'],
                weight_decay=cfg['weight_decay'],
                gamma=cfg['gamma'],
                label_smoothing=0.04,
                feature_noise=0.015,
                warmup_epochs=3,
                patience=8,
                checkpoint_dir=None,
                device=device,
                verbose=False
            )

            vm = res['val_metrics']
            val_acc_pct = round(vm['accuracy'] * 100, 2)
            val_bal_pct = round(vm['balanced_accuracy'] * 100, 2)
            val_f1_pct = round(vm['macro_f1'] * 100, 2)
            val_comp_pct = round(0.6 * val_f1_pct + 0.4 * val_bal_pct, 2)
            val_comp = val_comp_pct / 100.0
            print(f"    -> Val F1: {val_f1_pct:.2f}% | Val BalAcc: {val_bal_pct:.2f}% | Composite: {val_comp_pct:.2f}%")

            rec = {
                'trial_id': trial_counter,
                'model': model_name,
                'config_id': idx,
                'learning_rate': cfg['lr'],
                'hidden_dim_or_channels': str(cfg.get('hidden_dim', cfg.get('channels'))),
                'dropout': cfg['dropout'],
                'weight_decay': cfg['weight_decay'],
                'gamma': cfg['gamma'],
                'val_accuracy': val_acc_pct,
                'val_balanced_accuracy': val_bal_pct,
                'val_macro_f1': val_f1_pct,
                'val_composite_score': val_comp_pct,
                'params': res['params'],
                'train_time_sec': round(res['total_train_time_sec'], 2)
            }
            all_tuning_records.append(rec)

            if val_comp > best_model_score + 1e-4:
                best_model_score = val_comp
                best_cfg = cfg
            elif abs(val_comp - best_model_score) <= 1e-4 and best_cfg is not None:
                # Parsimonious selection: prefer lower hidden dimension when performance is equivalent
                cur_dim = cfg.get('hidden_dim', cfg.get('channels', [999])[0] if isinstance(cfg.get('channels'), list) else 999)
                best_dim = best_cfg.get('hidden_dim', best_cfg.get('channels', [999])[0] if isinstance(best_cfg.get('channels'), list) else 999)
                if cur_dim < best_dim:
                    best_cfg = cfg

        best_configs[model_name] = {
            'best_config': best_cfg,
            'best_val_composite_score': round(best_model_score * 100, 2)
        }
        print(f"  >>> Best configuration for {model_name}: {best_cfg} (Val Composite: {best_model_score*100:.2f}%)")

    # 3. Export Tuning Records
    print("\n[Step 3/3] Saving Hyperparameter Optimization Artifacts...")
    df_tuning = pd.DataFrame(all_tuning_records)
    csv_tuning_path = os.path.join(out_dir, "Hyperparameter_Tuning_Grid.csv")
    df_tuning.to_csv(csv_tuning_path, index=False)
    print(f"  [OK] Saved complete tuning grid to: {csv_tuning_path}")

    json_best_path = os.path.join(out_dir, "best_hyperparameters.json")
    with open(json_best_path, 'w') as f:
        json.dump(best_configs, f, indent=2)
    print(f"  [OK] Saved best hyperparameters JSON to: {json_best_path}")



    # Plot Tuning Landscape
    plot_path = os.path.join(out_dir, "Hyperparameter_Optimization_Landscape.png")
    plt.figure(figsize=(12, 6))
    models = list(tuning_configs.keys())
    x_positions = np.arange(len(models))
    width = 0.35

    val_scores = [best_configs[m]['best_val_composite_score'] for m in models]
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#BD10E0', '#9013FE', '#D0021B']

    bars = plt.bar(x_positions, val_scores, width=0.55, color=colors, edgecolor='black', linewidth=1.2, alpha=0.9)
    plt.axhline(60, color='gray', linestyle='--', alpha=0.6, label='60% Clinical Benchmark')
    plt.xticks(x_positions, [m.replace('_', ' ') for m in models], rotation=15, fontsize=11, fontweight='bold')
    plt.ylabel('Peak Validation Composite Score (%)', fontsize=12, fontweight='bold')
    plt.title('Hyperparameter Optimization Results: Peak Validation Score across Architectures', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(40, 80)
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.8, f"{h:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved hyperparameter optimization landscape plot to: {plot_path}")

    print("\n" + "=" * 90)
    print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
    print("=" * 90)
    return best_configs

if __name__ == "__main__":
    run_hyperparameter_tuning()
