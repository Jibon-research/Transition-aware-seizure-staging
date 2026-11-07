"""
========================================================================================
PROPOSED NOVEL ARCHITECTURE:
SPATIAL-TEMPORAL ALPHA TRANSITION-MEMORY RECURRENT NETWORK (ST-ATMR-NET)
========================================================================================
Journal Target: Journal of Clinical Neurophysiology (JCN)

Core Methodological & Clinical Innovations:
1. Triple Kinematic Pathways:
   - Context Pathway: Instantaneous feature context x_t (506 dims)
   - Velocity Pathway: Spectral rate-of-change \\Delta x_t (506 dims)
   - Acceleration Pathway: Paroxysmal transition acceleration \\Delta^2 x_t (506 dims)
2. Squeeze-and-Excitation (SE) Channel Attention:
   - Preserves focal electrode topography across all 23 scalp channels without dilution.
3. Multi-Scale Causal Dilated Residual Filtering (dilation 1, 2, 4):
   - Receptive horizon across sequence without future data leakage.
4. Dynamic Transition Gating:
   - \\Gamma_t = \\sigma(W_g [h_c, h_v, h_a]) adaptively gates kinematic flows.
5. Transition-Coupled Recurrent Memory (GRU + Residual Identity Skip):
   - Seamlessly tracks progressive 5-stage states over 90s context.
6. Dual-Context Terminal + Attention Sequence Pooling:
   - Fuses instantaneous terminal state with sequence-wide attention context.
========================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class SqueezeExcitation(nn.Module):
    """Channel attention block for temporal convolutional features."""
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 16)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 16), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.fc(x).view(b, c, 1)
        return x * w

class DilatedSEBlock(nn.Module):
    """Multi-scale causal dilated residual block with Squeeze-and-Excitation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(DilatedSEBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        
        self.se = SqueezeExcitation(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.drop1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.drop2(out)
        out = self.se(out)
        return out + res

class STATMRNet(nn.Module):
    """
    PROPOSED NOVEL ARCHITECTURE:
    Spatial-Temporal Alpha Transition-Memory Recurrent Network (ST-ATMR-Net).
    """
    def __init__(self, input_dim=506, hidden_dim=128, num_classes=5, dropout=0.25, **kwargs):
        super(STATMRNet, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Level 1: Dedicated Dilated SE Encoders for Context, Velocity, Acceleration
        self.enc_context = nn.Sequential(
            DilatedSEBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout * 0.5),
            DilatedSEBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout * 0.5)
        )
        self.enc_velocity = nn.Sequential(
            DilatedSEBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout * 0.5),
            DilatedSEBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout * 0.5)
        )
        self.enc_accel = nn.Sequential(
            DilatedSEBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout * 0.5),
            DilatedSEBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout * 0.5)
        )
        
        # Level 2: Dynamic Transition Gating
        self.gate_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )
        self.proj_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Level 3: Dilated Causal Stage Filter + Recurrent Transition Memory
        self.stage_filter = DilatedSEBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ln_gru = nn.LayerNorm(hidden_dim)
        
        # Level 4: Dual-Context Attention Sequence Pooling
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )
        
        # Level 5: Regularized Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, x_transition=None, x_second_order=None):
        B, T, D = x.shape
        if x_transition is None:
            x_transition = torch.zeros_like(x)
        if x_second_order is None:
            x_second_order = torch.zeros_like(x)
            
        x_c_t = x.transpose(1, 2)
        x_v_t = x_transition.transpose(1, 2)
        x_a_t = x_second_order.transpose(1, 2)
        
        # 1. Dilated SE Filtering for each kinematic pathway
        h_c = self.enc_context(x_c_t).transpose(1, 2) # (B, T, hidden_dim)
        h_v = self.enc_velocity(x_v_t).transpose(1, 2) # (B, T, hidden_dim)
        h_a = self.enc_accel(x_a_t).transpose(1, 2) # (B, T, hidden_dim)
        
        # 2. Dynamic Transition Gating
        h_concat = torch.cat([h_c, h_v, h_a], dim=-1) # (B, T, hidden_dim * 3)
        gate = self.gate_fusion(h_concat)
        fused = self.proj_fusion(h_concat)
        h_kinematic = gate * fused + (1.0 - gate) * h_c # Residual gating
        
        # 3. Stage Dilated Filtering & Recurrent Memory
        h_filtered = self.stage_filter(h_kinematic.transpose(1, 2)).transpose(1, 2)
        h_seq, _ = self.gru(h_filtered + h_kinematic)
        h_seq = self.ln_gru(h_seq)
        
        # 4. Dual-Context Pooling
        terminal_state = h_seq[:, -1, :] # (B, hidden_dim)
        scores = self.attn(h_seq) # (B, T, 1)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(weights * h_seq, dim=1) # (B, hidden_dim)
        
        combined = torch.cat([terminal_state, pooled], dim=-1) # (B, hidden_dim * 2)
        logits = self.classifier(combined)
        return logits

# Alias for backward compatibility
STATMR_HybridNet = STATMRNet

if __name__ == "__main__":
    dummy_x = torch.randn(4, 15, 506)
    dummy_d1 = torch.randn(4, 15, 506)
    dummy_d2 = torch.randn(4, 15, 506)
    model = STATMRNet()
    out = model(dummy_x, dummy_d1, dummy_d2)
    p = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"ST-ATMR-Net initialized successfully. Output shape: {out.shape} | Parameters: {p:,}")
