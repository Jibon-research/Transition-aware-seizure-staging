"""
========================================================================================
FULL-POWER CLINICAL TRAINING & EVALUATION ENGINE (v2 UPGRADED)
========================================================================================
Features:
- Multi-class Focal Loss with smooth class weights & label smoothing
- Warmup + Cosine Annealing Learning Rate Scheduler
- Feature Gaussian Noise Augmentation during training to prevent tabular overfitting
- Validation Checkpointing monitoring composite (0.6 * Macro-F1 + 0.4 * Balanced Acc)
- Gradient Norm Clipping (1.0) for numerical stability
- Inference latency profiling & metric extraction
========================================================================================
"""

import os
import copy
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from src.metrics import compute_all_metrics, measure_inference_latency
from src.losses import FocalLoss

class WarmupCosineScheduler:
    """Linear warmup for warmup_epochs followed by Cosine Annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def train_epoch(model, loader, criterion, optimizer, device, feature_noise=0.015):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in loader:
        x = batch['x_context'].to(device)
        x_d1 = batch['x_transition'].to(device)
        x_d2 = batch['x_second_order'].to(device)
        targets = batch['label'].to(device)
        
        # Apply stochastic feature jittering to prevent tabular/sequence overfitting
        if feature_noise > 0.0:
            x = x + torch.randn_like(x) * feature_noise
            x_d1 = x_d1 + torch.randn_like(x_d1) * feature_noise
            x_d2 = x_d2 + torch.randn_like(x_d2) * feature_noise
        
        optimizer.zero_grad()
        logits = model(x, x_transition=x_d1, x_second_order=x_d2)
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(targets)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_all_metrics(all_targets, all_preds)
    return avg_loss, metrics

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch['x_context'].to(device)
            x_d1 = batch['x_transition'].to(device)
            x_d2 = batch['x_second_order'].to(device)
            targets = batch['label'].to(device)
            
            logits = model(x, x_transition=x_d1, x_second_order=x_d2)
            loss = criterion(logits, targets)
            
            total_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_all_metrics(all_targets, all_preds)
    return avg_loss, metrics

def fit_model(model, train_loader, val_loader, test_loader, class_weights,
              model_name="Model", num_epochs=40, lr=1e-3, weight_decay=3e-3,
              gamma=1.5, label_smoothing=0.04, feature_noise=0.015,
              warmup_epochs=3, patience=12,
              checkpoint_dir=None, device="cpu", verbose=True):
    """
    Standardized training harness with:
    - Linear Warmup + Cosine Annealing learning rate schedule
    - Multi-Class Focal Loss with smooth class weights
    - Early stopping monitoring validation composite score:
      Score = 0.6 * Macro-F1 + 0.4 * Balanced Accuracy.
    - Saves best checkpoint and evaluates on independent test set.
    """
    device = torch.device(device)
    model = model.to(device)
    
    # Advanced Focal Loss with smooth class weights
    criterion = FocalLoss(weights=class_weights.to(device), gamma=gamma, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        base_lr=lr,
        min_lr=1e-5
    )
    
    best_val_score = -1.0
    best_model_weights = None
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_macro_f1': [],
        'val_macro_f1': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'val_balanced_accuracy': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_metrics = train_epoch(model, train_loader, criterion, optimizer, device, feature_noise=feature_noise)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        current_lr = scheduler.step()
        
        val_f1 = val_metrics['macro_f1']
        val_bal = val_metrics['balanced_accuracy']
        composite_score = 0.6 * val_f1 + 0.4 * val_bal
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_macro_f1'].append(tr_metrics['macro_f1'])
        history['val_macro_f1'].append(val_f1)
        history['train_accuracy'].append(tr_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_balanced_accuracy'].append(val_bal)
        
        if composite_score > best_val_score:
            best_val_score = composite_score
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
            improved_flag = "*"
        else:
            epochs_no_improve += 1
            improved_flag = " "
            
        if verbose and (epoch % 5 == 0 or epoch == 1 or improved_flag == "*"):
            print(f"  Epoch {epoch:02d}/{num_epochs:02d} (lr: {current_lr:.1e}) | Train Loss: {tr_loss:.4f} | "
                  f"Val F1: {val_f1*100:.1f}%, Bal Acc: {val_bal*100:.1f}%, Acc: {val_metrics['accuracy']*100:.1f}% {improved_flag}")
                  
        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch} (best Val Composite: {best_val_score*100:.2f}% at epoch {best_epoch})")
            stopped_epoch = epoch
            break
    else:
        stopped_epoch = num_epochs
            
    total_train_time = time.time() - start_time
    
    # Load best checkpoint
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    
    # Final evaluation on validation and optional test sets
    final_val_loss, final_val_metrics = evaluate(model, val_loader, criterion, device)
    if test_loader is not None:
        final_test_loss, final_test_metrics = evaluate(model, test_loader, criterion, device)
        sample_batch = next(iter(test_loader))
    else:
        final_test_loss, final_test_metrics = None, None
        sample_batch = next(iter(val_loader))
        
    latency_ms = measure_inference_latency(model, sample_batch, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    result = {
        'model_name': model_name,
        'params': param_count,
        'latency_ms_per_seq': latency_ms,
        'total_train_time_sec': total_train_time,
        'best_epoch': best_epoch if 'best_epoch' in locals() else num_epochs,
        'stopped_epoch': stopped_epoch,
        'best_val_score': best_val_score,
        'history': history,
        'val_metrics': final_val_metrics,
        'test_metrics': final_test_metrics
    }
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
        torch.save({
            'model_name': model_name,
            'model_state_dict': model.state_dict(),
            'best_epoch': result['best_epoch'],
            'stopped_epoch': result['stopped_epoch'],
            'best_val_score': result['best_val_score'],
            'result': result
        }, ckpt_path)
        if verbose:
            print(f"  [Checkpoint Saved] -> {ckpt_path}")
        
    return result
