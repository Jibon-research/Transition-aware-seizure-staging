"""
========================================================================================
FULL-POWER CLINICAL DEEP LEARNING BASELINES (v2 UPGRADED)
========================================================================================
1. StandardLSTM: 2-Layer LSTM with LayerNorm, Attention Pooling & Target State Skip
2. BiLSTM: 2-Layer Bidirectional LSTM with Multi-Head Attention & Target State Skip
3. TriLSTM: 3-Layer Deep Residual Stacked LSTM with LayerNorm on each level
4. TCN1: Temporal Convolutional Network with Squeeze-and-Excitation Channel Attention
5. TCN2: Deep Multi-Scale Dilated Residual Network with SE blocks & Dual Temporal Attention
========================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """Learned additive attention pooling over the sequence window."""
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, h_seq):
        # h_seq: (batch, seq_len, hidden_dim)
        scores = self.attn(h_seq)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(weights * h_seq, dim=1)
        return pooled

class StandardLSTM(nn.Module):
    """Baseline 1: Standard LSTM with LayerNorm, Temporal Attention & Target State Skip."""
    def __init__(self, input_dim=506, hidden_dim=160, num_layers=2, num_classes=5, dropout=0.25):
        super(StandardLSTM, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.6)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = TemporalAttention(hidden_dim)
        # Dual-context: concat([last_state, pooled]) -> (hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, **kwargs):
        h = self.input_proj(x)
        out, _ = self.lstm(h)
        pooled = self.attn(out)
        target_state = out[:, -1, :]
        combined = torch.cat([target_state, pooled], dim=-1)
        logits = self.classifier(combined)
        return logits

class BiLSTM(nn.Module):
    """Baseline 2: Bidirectional LSTM with Multi-Head Attention & Target State Skip."""
    def __init__(self, input_dim=506, hidden_dim=160, num_layers=2, num_classes=5, dropout=0.25):
        super(BiLSTM, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.6)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = TemporalAttention(hidden_dim * 2)
        # Dual-context: concat([last_state, pooled]) -> (hidden_dim * 4)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
    def forward(self, x, **kwargs):
        h = self.input_proj(x)
        out, _ = self.lstm(h)
        pooled = self.attn(out)
        target_state = out[:, -1, :]
        combined = torch.cat([target_state, pooled], dim=-1)
        logits = self.classifier(combined)
        return logits

class TriLSTM(nn.Module):
    """Baseline 3: 3-layer Deep Stacked Residual LSTM with attention pooling & target skip."""
    def __init__(self, input_dim=506, hidden_dim=160, num_classes=5, dropout=0.25):
        super(TriLSTM, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.6)
        )
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout * 0.8)
        
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(dropout)
        
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.drop3 = nn.Dropout(dropout * 1.2)
        
        self.attn = TemporalAttention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, **kwargs):
        h0 = self.input_proj(x)
        out1, _ = self.lstm1(h0)
        out1 = self.drop1(self.ln1(out1 + h0))
        
        out2, _ = self.lstm2(out1)
        out2 = self.drop2(self.ln2(out2 + out1))
        
        out3, _ = self.lstm3(out2)
        out3 = self.drop3(self.ln3(out3 + out2))
        
        pooled = self.attn(out3)
        target_state = out3[:, -1, :]
        combined = torch.cat([target_state, pooled], dim=-1)
        logits = self.classifier(combined)
        return logits

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

class TemporalBlockSE(nn.Module):
    """Causal dilated convolutional block with Squeeze-and-Excitation."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlockSE, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.se = SqueezeExcitation(n_outputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.se(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN1(nn.Module):
    """Baseline 4: TCN-1 with dilated causal convolutions & SE channel attention."""
    def __init__(self, input_dim=506, num_channels=[128, 128], kernel_size=3, num_classes=5, dropout=0.2):
        super(TCN1, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlockSE(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(num_channels[-1]),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1], 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, **kwargs):
        x_trans = x.transpose(1, 2)
        y = self.network(x_trans)
        logits = self.classifier(y)
        return logits

class TCN2(nn.Module):
    """Baseline 5: Deep Multi-Scale TCN-2 with SE blocks and dual temporal attention."""
    def __init__(self, input_dim=506, num_channels=[160, 160, 160, 160], kernel_size=3, num_classes=5, dropout=0.25):
        super(TCN2, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlockSE(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.attn = TemporalAttention(num_channels[-1])
        # Dual-context: concat([last_temporal_step, pooled]) -> (channels * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_channels[-1] * 2),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] * 2, num_channels[-1]),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(num_channels[-1], num_classes)
        )

    def forward(self, x, **kwargs):
        x_trans = x.transpose(1, 2)
        y = self.network(x_trans)
        y_t = y.transpose(1, 2)  # (batch, seq_len, channels)
        pooled = self.attn(y_t)
        last_step = y_t[:, -1, :]
        combined = torch.cat([last_step, pooled], dim=-1)
        logits = self.classifier(combined)
        return logits

if __name__ == "__main__":
    dummy_x = torch.randn(8, 10, 506)
    models = {
        'StandardLSTM': StandardLSTM(),
        'BiLSTM': BiLSTM(),
        'TriLSTM': TriLSTM(),
        'TCN1': TCN1(),
        'TCN2': TCN2()
    }
    print("Testing upgraded Baselines v2...")
    for name, m in models.items():
        out = m(dummy_x)
        p = sum(param.numel() for param in m.parameters() if param.requires_grad)
        print(f"  [OK] {name:15s} -> Output: {list(out.shape)} | Params: {p:,}")
    print("[OK] All baselines upgraded successfully.")
