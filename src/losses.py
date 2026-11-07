import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_smooth_class_weights(class_counts, total_samples, power=0.5):
    """
    Computes sublinear smooth class weights to prevent gradient explosion on rare classes.
    w_c = (N / (num_classes * N_c)) ** power
    Normalized so that mean(w) = 1.0
    """
    counts = np.maximum(np.array(class_counts, dtype=np.float32), 1.0)
    num_classes = len(counts)
    raw_weights = (total_samples / (num_classes * counts)) ** power
    normalized_weights = raw_weights / np.mean(raw_weights)
    return torch.tensor(normalized_weights, dtype=torch.float32)

class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss with Label Smoothing & Class Balancing.
    Down-weights easy well-classified background examples and focuses gradient on hard transition boundaries.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, weights=None, gamma=1.5, label_smoothing=0.04):
        super(FocalLoss, self).__init__()
        self.weights = weights
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        device = logits.device
        
        # Apply label smoothing to target one-hot distribution
        with torch.no_grad():
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
            if self.smoothing > 0:
                one_hot = (1.0 - self.smoothing) * one_hot + self.smoothing / num_classes
                
        probs = F.softmax(logits, dim=-1)
        pt = torch.sum(probs * one_hot, dim=-1)
        log_pt = F.log_softmax(logits, dim=-1)
        ce_loss = -torch.sum(one_hot * log_pt, dim=-1)
        
        # Modulating factor
        focal_weight = torch.pow(1.0 - pt.clamp(0.0, 1.0), self.gamma)
        loss = focal_weight * ce_loss
        
        if self.weights is not None:
            w = self.weights.to(device)[targets]
            loss = loss * w
            
        return loss.mean()
