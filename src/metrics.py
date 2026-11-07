import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix
)

STAGE_NAMES = ['Normal', 'Pre-Ictal', 'Ictal', 'Inter-Ictal', 'Post-Ictal']

def compute_all_metrics(y_true, y_pred):
    """
    Computes all standard and clinical neurophysiology metrics for 5-stage classification.
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))
        
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_norm': cm_norm.tolist(),
        'per_class': {}
    }
    
    for i, name in enumerate(STAGE_NAMES):
        metrics['per_class'][name] = {
            'precision': float(prec_per_class[i]),
            'recall': float(rec_per_class[i]),
            'f1': float(f1_per_class[i]),
            'support': int((y_true == i).sum())
        }
        
    return metrics

def plot_confusion_matrix(cm_norm, cm_raw, title="Confusion Matrix", save_path=None, dpi=300):
    """
    Renders publication-grade 5x5 confusion matrix with normalized percentages and raw counts.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=dpi)
    
    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(5):
        for j in range(5):
            pct = cm_norm[i, j] * 100.0
            raw = cm_raw[i, j]
            annot[i, j] = f"{pct:.1f}%\n({raw})"
            
    sns.heatmap(
        cm_norm * 100.0,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar_kws={'label': 'Recall Percentage (%)'},
        xticklabels=STAGE_NAMES,
        yticklabels=STAGE_NAMES,
        ax=ax,
        linewidths=0.8,
        linecolor="#dddddd",
        vmin=0.0,
        vmax=100.0
    )
    
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Stage", fontsize=10.0, fontweight="bold")
    ax.set_ylabel("True Ground-Truth Stage", fontsize=10.0, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9.0)
    plt.yticks(rotation=0, fontsize=9.0)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def measure_inference_latency(model, sample_batch, device, num_runs=50):
    """
    Measures median inference latency per sequence window (ms).
    """
    model.eval()
    x = sample_batch['x_context'].to(device)
    x_d1 = sample_batch['x_transition'].to(device)
    x_d2 = sample_batch['x_second_order'].to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, x_transition=x_d1, x_second_order=x_d2)
            
    # Benchmark
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(x, x_transition=x_d1, x_second_order=x_d2)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms
            
    batch_size = x.size(0)
    median_batch_ms = float(np.median(timings))
    latency_per_sample_ms = median_batch_ms / batch_size
    return latency_per_sample_ms
