"""
========================================================================================
CONTINUOUS CLINICAL TEMPORAL SIMULATION MODULE (AUDITED & CLINICALLY GROUNDED)
========================================================================================
Provides continuous sliding-window streaming evaluation of trained EEG models across
continuous multi-channel recordings, adhering to clinical neurophysiology audit standards:
  1. Multi-Seizure Accounting: Detects and evaluates every distinct seizure bout per file.
  2. Strict Pre-Ictal Warning: Evaluates forecasting strictly via pre-ictal class (1),
     rejecting premature ictal false alarms (2) as valid warnings.
  3. Causal & Audited False Alarm Accounting: Removes arbitrary multipliers. Implements
     explicit 2-epoch persistence filtering and alarm-merging refractory periods (60s).
  4. Nominal Timing Disclaimer: Reports metrics as nominal window times (stride * 6.0s),
     explicitly noting reliance on preprocessed row sequence continuity.
========================================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.dataset import compute_temporal_derivatives, slice_recording_into_sequences
from src.metrics import compute_all_metrics

def load_model_from_checkpoint(model_cls, ckpt_path, **kwargs):
    """
    Loads model weights from checkpoint into eval mode.
    Handles both state dict wrappers and raw checkpoints.
    """
    model = model_cls(**kwargs)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

LEGITIMATE_TRANSITIONS = {
    (0, 0), (0, 1), (0, 3),
    (1, 1), (1, 2), (1, 0),
    (2, 2), (2, 4),
    (4, 4), (4, 0), (4, 3),
    (3, 3), (3, 1), (3, 0)
}

def simulate_recording(model, csv_path, window_size=15, is_statmr=True, norm_mode='session', scaler=None, baseline_epochs=10):
    """
    Executes continuous sliding-window temporal simulation across an entire recording.
    Returns timestamps, true stages, predicted stages, predicted probabilities.
    
    Normalization modes:
      - 'session': Transductive full-session mean & std.
      - 'causal_baseline': Strictly calibrates from the first `baseline_epochs` (e.g. 10 epochs = 60s) of recording.
      - 'training_fitted': Uses pre-fitted `scaler` from training set (strictly inductive).
    """
    df = pd.read_csv(csv_path)
    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values.astype(int)
    N = len(y_raw)
    
    # 1. Normalization
    if norm_mode == 'training_fitted' and scaler is not None:
        X_scaled = scaler.transform(X_raw)
    elif norm_mode == 'causal_baseline':
        k = min(N, max(baseline_epochs, 1))
        mu = np.mean(X_raw[:k], axis=0, keepdims=True)
        sig = np.std(X_raw[:k], axis=0, keepdims=True) + 1e-6
        X_scaled = (X_raw - mu) / sig
    else: # 'session' (default)
        mu = np.mean(X_raw, axis=0, keepdims=True)
        sig = np.std(X_raw, axis=0, keepdims=True) + 1e-6
        X_scaled = (X_raw - mu) / sig
    
    # 2. Compute temporal kinematics
    d1, d2 = compute_temporal_derivatives(X_scaled)
    
    # 3. Slide continuous window
    sx, sd1, sd2, sy = slice_recording_into_sequences(X_scaled, d1, d2, y_raw, window_size=window_size, stride=1)
    if len(sy) == 0:
        return None
        
    tx = torch.tensor(np.array(sx), dtype=torch.float32)
    td1 = torch.tensor(np.array(sd1), dtype=torch.float32)
    td2 = torch.tensor(np.array(sd2), dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        if is_statmr:
            logits = model(tx, td1, td2)
        else:
            logits = model(tx)
        probs = F.softmax(logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=-1)
        
    true_stages = np.array(sy)
    # Nominal elapsed time based on sequence row index (6.0s per epoch)
    timestamps = np.array([(i + window_size - 1) * 6.0 for i in range(len(sy))])
    
    return {
        'timestamps': timestamps,
        'true_stages': true_stages,
        'pred_stages': preds,
        'probs': probs,
        'total_duration_sec': timestamps[-1] - timestamps[0] + 6.0,
        'num_epochs': len(sy)
    }

def extract_seizure_bouts(y_true, timestamps):
    """
    Identifies all distinct continuous seizure events (class 2) in a recording.
    Returns a list of dicts: [{'start_idx', 'end_idx', 'onset_sec', 'end_sec', 'duration_sec'}]
    """
    ictal_indices = np.where(y_true == 2)[0]
    if len(ictal_indices) == 0:
        return []
        
    # Split into contiguous bouts (separated by > 1 non-ictal epoch)
    bouts = []
    current_bout = [ictal_indices[0]]
    
    for idx in ictal_indices[1:]:
        if idx == current_bout[-1] + 1:
            current_bout.append(idx)
        else:
            bouts.append(current_bout)
            current_bout = [idx]
    bouts.append(current_bout)
    
    seizures = []
    for b in bouts:
        start_idx = b[0]
        end_idx = b[-1]
        onset_sec = timestamps[start_idx]
        end_sec = timestamps[end_idx] + 6.0
        seizures.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'onset_sec': onset_sec,
            'end_sec': end_sec,
            'duration_sec': end_sec - onset_sec
        })
    return seizures

def analyze_clinical_events(sim_res, preictal_horizon_sec=300.0):
    """
    Audited event-level clinical evaluation:
    1. Evaluates EVERY distinct seizure bout separately.
    2. Strictly requires pre-ictal prediction (class 1) for early warning;
       premature ictal predictions (class 2) do not count as pre-ictal warning.
    3. Computes false positive rates with explicit 2-epoch persistence filtering and 60s refractory merging.
    """
    t = sim_res['timestamps']
    y_true = sim_res['true_stages']
    y_pred = sim_res['pred_stages']
    
    seizures = extract_seizure_bouts(y_true, t)
    seizure_metrics = []
    
    for sz in seizures:
        t_onset = sz['onset_sec']
        t_end = sz['end_sec']
        sz_dur = sz['duration_sec']
        
        # 1. Strict Pre-Ictal Warning (Lead Time)
        # Look strictly for class 1 within the specified pre-ictal horizon before onset
        pre_window_mask = (t >= t_onset - preictal_horizon_sec) & (t < t_onset)
        pre_alerts = np.where(pre_window_mask & (y_pred == 1))[0]
        
        if len(pre_alerts) > 0:
            earliest_alert_t = t[pre_alerts[0]]
            lead_time = max(0.0, t_onset - earliest_alert_t)
            warned = True
        else:
            lead_time = 0.0
            warned = False
            
        # 2. Ictal Onset Detection Latency
        ictal_preds = np.where((t >= t_onset) & (t <= t_end) & (y_pred == 2))[0]
        if len(ictal_preds) > 0:
            first_ictal_pred_t = t[ictal_preds[0]]
            onset_delay = max(0.0, first_ictal_pred_t - t_onset)
            detected = True
        else:
            onset_delay = sz_dur
            detected = False
            
        seizure_metrics.append({
            'onset_sec': t_onset,
            'end_sec': t_end,
            'duration_sec': sz_dur,
            'warned': warned,
            'lead_time_sec': lead_time,
            'detected': detected,
            'onset_delay_sec': onset_delay
        })
        
    # 3. Physiological Markovian Transition Compliance
    transitions = list(zip(y_pred[:-1], y_pred[1:]))
    if len(transitions) > 0:
        legit_count = sum(1 for tr in transitions if tr in LEGITIMATE_TRANSITIONS)
        compliance_pct = (legit_count / len(transitions)) * 100.0
    else:
        compliance_pct = 100.0
        
    # 4. Rigorous False Alarm Accounting (with explicit 2-epoch persistence)
    # Only evaluate during ground-truth Normal baseline epochs (class 0)
    normal_mask = (y_true == 0)
    normal_indices = np.where(normal_mask)[0]
    total_normal_hours = (len(normal_indices) * 6.0) / 3600.0 if len(normal_indices) > 0 else 1.0
    
    # Raw unadjusted false alarms (any single epoch predicted as Ictal during Normal)
    raw_fa_epochs = np.sum((y_pred[normal_indices] == 2)) if len(normal_indices) > 0 else 0
    raw_far_per_hour = raw_fa_epochs / max(total_normal_hours, 1e-4)
    
    # Explicit 2-epoch persistence alarm filter:
    # Alarm is raised only when >= 2 consecutive epochs are predicted as Ictal
    persistent_alarms = 0
    refractory_until = -1.0
    
    for i in range(len(y_pred) - 1):
        if y_true[i] == 0 and y_true[i+1] == 0:
            if y_pred[i] == 2 and y_pred[i+1] == 2:
                alarm_t = t[i]
                if alarm_t > refractory_until:
                    persistent_alarms += 1
                    refractory_until = alarm_t + 60.0 # 60-second refractory lockout
                    
    persistent_far_per_hour = persistent_alarms / max(total_normal_hours, 1e-4)
    
    return {
        'num_seizures': len(seizures),
        'seizure_metrics': seizure_metrics,
        'transition_compliance_pct': compliance_pct,
        'raw_far_per_hour': raw_far_per_hour,
        'persistent_far_per_hour': persistent_far_per_hour,
        'total_normal_hours': total_normal_hours
    }

def evaluate_clinical_cohort(model, test_csv_dir, window_size=15, is_statmr=True, norm_mode='session', scaler=None, baseline_epochs=10):
    """
    Evaluates an entire patient recording cohort with event-level multi-seizure accounting.
    """
    test_files = sorted(glob.glob(os.path.join(test_csv_dir, "*.csv")))
    all_seizure_events = []
    all_compliances = []
    all_raw_fars = []
    all_persistent_fars = []
    total_cohort_normal_hours = 0.0
    
    for fpath in test_files:
        sim = simulate_recording(model, fpath, window_size=window_size, is_statmr=is_statmr,
                                 norm_mode=norm_mode, scaler=scaler, baseline_epochs=baseline_epochs)
        if sim is None:
            continue
        ev = analyze_clinical_events(sim)
        all_seizure_events.extend(ev['seizure_metrics'])
        all_compliances.append(ev['transition_compliance_pct'])
        all_raw_fars.append(ev['raw_far_per_hour'])
        all_persistent_fars.append(ev['persistent_far_per_hour'])
        total_cohort_normal_hours += ev['total_normal_hours']
        
    num_total_seizures = len(all_seizure_events)
    if num_total_seizures > 0:
        warned_count = sum(1 for s in all_seizure_events if s['warned'])
        detected_count = sum(1 for s in all_seizure_events if s['detected'])
        
        preictal_sensitivity = (warned_count / num_total_seizures) * 100.0
        ictal_detection_sensitivity = (detected_count / num_total_seizures) * 100.0
        
        positive_leads = [s['lead_time_sec'] for s in all_seizure_events if s['warned']]
        mean_lead = np.mean(positive_leads) if positive_leads else 0.0
        mean_delay = np.mean([s['onset_delay_sec'] for s in all_seizure_events])
    else:
        preictal_sensitivity = 0.0
        ictal_detection_sensitivity = 0.0
        mean_lead = 0.0
        mean_delay = 0.0
        
    mean_compliance = np.mean(all_compliances) if all_compliances else 100.0
    # Mean across recordings, unadjusted and with explicit 2-epoch persistence
    mean_raw_fpr = np.mean(all_raw_fars) if all_raw_fars else 0.0
    mean_persistent_fpr = np.mean(all_persistent_fars) if all_persistent_fars else 0.0
    
    return {
        'num_test_recordings': len(test_files),
        'num_total_seizures': num_total_seizures,
        'preictal_sensitivity': preictal_sensitivity,
        'ictal_detection_sensitivity': ictal_detection_sensitivity,
        'mean_lead_time_sec': mean_lead,
        'mean_detection_delay_sec': mean_delay,
        'raw_fpr_per_hour': mean_raw_fpr,
        'fpr_per_hour': mean_persistent_fpr, # Explicit 2-epoch persistent FPR
        'transition_compliance': mean_compliance,
        'total_cohort_normal_hours': total_cohort_normal_hours
    }
