import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

# Dynamically locate project root and 03 Dataset folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "03 Dataset")

STAGE_NAMES = ['Normal', 'Pre-Ictal', 'Ictal', 'Inter-Ictal', 'Post-Ictal']

def compute_temporal_derivatives(X):
    """
    Computes first-order and second-order temporal differences strictly within a single continuous recording.
    X: shape (N, D) where N is number of epochs, D is feature dimension (506).
    Returns:
      delta_1: shape (N, D) where delta_1[t] = X[t] - X[t-1] (delta_1[0] = 0)
      delta_2: shape (N, D) where delta_2[t] = delta_1[t] - delta_1[t-1] (delta_2[0:2] = 0)
    """
    N, D = X.shape
    delta_1 = np.zeros_like(X)
    if N > 1:
        delta_1[1:] = X[1:] - X[:-1]
        
    delta_2 = np.zeros_like(X)
    if N > 2:
        delta_2[2:] = delta_1[2:] - delta_1[1:-1]
        
    return delta_1, delta_2

def slice_recording_into_sequences(X, delta_1, delta_2, y, window_size=10, stride=1):
    """
    Slices a continuous recording into sliding sequences of length `window_size`.
    The target label corresponds to the current epoch at the end of the window: y[i + window_size - 1].
    """
    N = len(y)
    if N < window_size:
        return [], [], [], []
        
    seq_x = []
    seq_d1 = []
    seq_d2 = []
    seq_y = []
    
    for i in range(0, N - window_size + 1, stride):
        seq_x.append(X[i : i + window_size])
        seq_d1.append(delta_1[i : i + window_size])
        seq_d2.append(delta_2[i : i + window_size])
        seq_y.append(int(y[i + window_size - 1]))
        
    return seq_x, seq_d1, seq_d2, seq_y

class EEGSequenceDataset(Dataset):
    """
    PyTorch Dataset providing:
      - x_context: (T, 506)
      - x_transition: (T, 506)
      - x_second_order: (T, 506)
      - y: int label (0..4)
    """
    def __init__(self, sequences_x, sequences_d1, sequences_d2, labels):
        self.x = torch.tensor(np.array(sequences_x), dtype=torch.float32)
        self.d1 = torch.tensor(np.array(sequences_d1), dtype=torch.float32)
        self.d2 = torch.tensor(np.array(sequences_d2), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return {
            'x_context': self.x[idx],
            'x_transition': self.d1[idx],
            'x_second_order': self.d2[idx],
            'label': self.y[idx]
        }

def load_partition_recordings(split_dir):
    """Loads all CSV files in a split directory into a list of (rec_id, X, y)."""
    recordings = []
    files = sorted([f for f in os.listdir(split_dir) if f.endswith('.csv')])
    for f in files:
        rec_id = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(split_dir, f))
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(int)
        recordings.append((rec_id, X, y))
    return recordings

def build_dataloaders(band_folder_name, window_size=15, batch_size=64, stride=1, scaler_type='standard', per_recording_norm=True, norm_mode=None, baseline_epochs=10):
    """
    Builds training, validation, and testing dataloaders from split folders.
    
    Normalization modes supported via `norm_mode` (or legacy `per_recording_norm`):
      1. 'training_fitted' (or per_recording_norm=False):
         Fits a StandardScaler or RobustScaler strictly on the training recordings.
         Validation and test recordings are standardized using the frozen training parameters.
         Zero future or test set leakage; 100% causal and deployment-ready.
      2. 'causal_baseline':
         Standardizes each recording using the mean and standard deviation of its initial
         baseline epochs (default: first 10 epochs = 60.0s). Represents clinical admission
         calibration without using any future recording epochs.
      3. 'session' (or per_recording_norm=True):
         Standardizes each recording using its full-session mean and standard deviation.
         Transductive session-level normalization across the recording duration.
    """
    if norm_mode is None:
        norm_mode = 'session' if per_recording_norm else 'training_fitted'
        
    band_dir = os.path.join(DATASET_ROOT, band_folder_name, "Preprocessed Data")
    train_dir = os.path.join(band_dir, "training")
    val_dir = os.path.join(band_dir, "validation")
    test_dir = os.path.join(band_dir, "testing")
    
    train_recs = load_partition_recordings(train_dir)
    val_recs = load_partition_recordings(val_dir)
    test_recs = load_partition_recordings(test_dir)
    
    scaler = None
    if norm_mode == 'training_fitted':
        # Fit scaler strictly on training recordings
        all_train_features = np.vstack([rec[1] for rec in train_recs])
        if scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(all_train_features)
    
    def process_partition(recs):
        all_x, all_d1, all_d2, all_y = [], [], [], []
        for rec_id, X_raw, y_raw in recs:
            if norm_mode == 'training_fitted':
                X_scaled = scaler.transform(X_raw)
            elif norm_mode == 'causal_baseline':
                # Calibrate strictly from the initial baseline epochs (first K epochs)
                k = min(len(X_raw), max(baseline_epochs, 1))
                mu = np.mean(X_raw[:k], axis=0, keepdims=True)
                sig = np.std(X_raw[:k], axis=0, keepdims=True) + 1e-6
                X_scaled = (X_raw - mu) / sig
            else: # 'session' (per-recording full session)
                mu = np.mean(X_raw, axis=0, keepdims=True)
                sig = np.std(X_raw, axis=0, keepdims=True) + 1e-6
                X_scaled = (X_raw - mu) / sig
                
            # Compute temporal derivatives strictly within recording
            d1, d2 = compute_temporal_derivatives(X_scaled)
            # Slice into sliding sequences
            sx, sd1, sd2, sy = slice_recording_into_sequences(X_scaled, d1, d2, y_raw, window_size=window_size, stride=stride)
            all_x.extend(sx)
            all_d1.extend(sd1)
            all_d2.extend(sd2)
            all_y.extend(sy)
        return all_x, all_d1, all_d2, all_y

    train_x, train_d1, train_d2, train_y = process_partition(train_recs)
    val_x, val_d1, val_d2, val_y = process_partition(val_recs)
    test_x, test_d1, test_d2, test_y = process_partition(test_recs)
    
    train_dataset = EEGSequenceDataset(train_x, train_d1, train_d2, train_y)
    val_dataset = EEGSequenceDataset(val_x, val_d1, val_d2, val_y)
    test_dataset = EEGSequenceDataset(test_x, test_d1, test_d2, test_y)
    
    # Compute inverse and smooth class frequencies on training labels
    class_counts = np.bincount(train_y, minlength=5)
    total_samples = len(train_y)
    raw_weights = total_samples / (5.0 * np.maximum(class_counts, 1.0))
    class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float32)
    
    # Smooth square-root weighting normalized to mean 1.0
    smooth_weights = (total_samples / (5.0 * np.maximum(class_counts, 1.0))) ** 0.5
    smooth_weights = smooth_weights / np.mean(smooth_weights)
    smooth_weights_tensor = torch.tensor(smooth_weights, dtype=torch.float32)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    info = {
        'band': band_folder_name,
        'window_size': window_size,
        'num_train_sequences': len(train_dataset),
        'num_val_sequences': len(val_dataset),
        'num_test_sequences': len(test_dataset),
        'train_class_counts': class_counts.tolist(),
        'class_weights': raw_weights.tolist(),
        'smooth_weights': smooth_weights.tolist()
    }
    
    return train_loader, val_loader, test_loader, smooth_weights_tensor, scaler, info


if __name__ == "__main__":
    print("Testing data loader for Alpha Band...")
    tr_l, val_l, ts_l, cw, scaler, info = build_dataloaders("Alpha Band (8-13hz)", window_size=10, batch_size=64)
    print("Alpha Band Info:", info)
    batch = next(iter(tr_l))
    print("Batch keys:", batch.keys())
    print("Context shape:", batch['x_context'].shape)
    print("Transition shape:", batch['x_transition'].shape)
    print("Second-order shape:", batch['x_second_order'].shape)
    print("Label shape:", batch['label'].shape)
    print("[OK] Dataset module verified successfully.")
