import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STAGE_NAMES = ['Normal', 'Pre-Ictal', 'Ictal', 'Inter-Ictal', 'Post-Ictal']
STAGE_COLORS = ['#2b5c8f', '#d97724', '#c0392b', '#8e44ad', '#27ae60']

def plot_learning_curves(history, title="Training Dynamics", save_path=None, dpi=300):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=dpi)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', lw=1.8)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', color='#d62728', lw=1.8, linestyle='--')
    axes[0].set_title(f"{title} - Loss Convergence", fontsize=11.0, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=9.5, fontweight='bold')
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=9.5, fontweight='bold')
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()
    
    # Macro-F1 plot
    axes[1].plot(epochs, [f * 100 for f in history['train_macro_f1']], label='Train Macro-F1', color='#1f77b4', lw=1.8)
    axes[1].plot(epochs, [f * 100 for f in history['val_macro_f1']], label='Val Macro-F1', color='#2ca02c', lw=1.8, linestyle='--')
    axes[1].set_title(f"{title} - Validation Macro-F1", fontsize=11.0, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=9.5, fontweight='bold')
    axes[1].set_ylabel("Macro-F1 Score (%)", fontsize=9.5, fontweight='bold')
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_per_class_radar(results_dict, metric_key='f1', title="Per-Class Performance Radar", save_path=None, dpi=300):
    """
    Renders radar chart comparing per-class performance across different architectures.
    """
    labels = STAGE_NAMES
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=dpi)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    if isinstance(results_dict, list):
        items = [(r.get('model_name', f"Model_{i}"), r) for i, r in enumerate(results_dict)]
    else:
        items = list(results_dict.items())
        
    for idx, (m_name, res) in enumerate(items):
        clean_name = str(m_name).replace('Alpha_', '').replace('BroadBand_', '')
        vals = [res['test_metrics']['per_class'][s][metric_key] * 100 for s in labels]
        vals += vals[:1]
        c = colors[idx % len(colors)]
        ax.plot(angles, vals, label=clean_name, color=c, lw=2.0)
        ax.fill(angles, vals, color=c, alpha=0.08)

        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10.0, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=12.0, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=8.5)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def format_benchmark_dataframe(results_list, band_name):
    """Formats list of model results into publication-ready DataFrame."""
    rows = []
    for r in results_list:
        tm = r['test_metrics']
        pc = tm['per_class']
        rows.append({
            'Model': r['model_name'],
            'Band': band_name,
            'Accuracy (%)': f"{tm['accuracy']*100:.2f}",
            'Balanced Acc (%)': f"{tm['balanced_accuracy']*100:.2f}",
            'Macro-F1 (%)': f"{tm['macro_f1']*100:.2f}",
            'Weighted-F1 (%)': f"{tm['weighted_f1']*100:.2f}",
            'Normal F1 (%)': f"{pc['Normal']['f1']*100:.2f}",
            'Pre-Ictal F1 (%)': f"{pc['Pre-Ictal']['f1']*100:.2f}",
            'Ictal F1 (%)': f"{pc['Ictal']['f1']*100:.2f}",
            'Inter-Ictal F1 (%)': f"{pc['Inter-Ictal']['f1']*100:.2f}",
            'Post-Ictal F1 (%)': f"{pc['Post-Ictal']['f1']*100:.2f}",
            'Params': f"{r['params']:,}",
            'Latency (ms)': f"{r['latency_ms_per_seq']:.3f}"
        })
    return pd.DataFrame(rows)
