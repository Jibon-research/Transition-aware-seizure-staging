"""
========================================================================================
PROPOSED ARCHITECTURE: ALPHA TRANSITION-MEMORY RECURRENT NETWORK (ATMR-NET v3)
========================================================================================
Key Innovations for Temporal 5-Stage Seizure Classification:
1. Dual/Triple Gated Pathway Encoders:
   - Instantaneous clinical EEG context x_t in R^506
   - First-order temporal velocity \Delta x_t = x_t - x_{t-1} in R^506
   - Optional second-order acceleration \Delta^2 x_t = \Delta x_t - \Delta x_{t-1} in R^506
2. Dynamic Transition-Gated Recurrent Cell (ATMRCell v3):
   - Computes dynamic transition vector \Gamma_t = \sigma(W_\gamma [e_c, e_d, h_{t-1}] + b_\gamma)
   - Modulates nonlinear velocity representation v_t = \tanh(W_v e_d + b_v)
   - Dynamic blend: u_t = (1 - \Gamma_t) \odot e_c + \Gamma_t \odot v_t
   - Memory Gate Transition Modulation:
     * Forget gate: f_t = \sigma(W_{hf} [u_t, h_{t-1}] - W_{\gamma f} \Gamma_t + b_f)
       (flushes obsolete baseline state when abrupt seizure transition occurs)
     * Input gate:  i_t = \sigma(W_{hi} [u_t, h_{t-1}] + W_{\gamma i} \Gamma_t + b_i)
       (accelerates absorption of transition dynamics)
3. Bidirectional Temporal Processing:
   - Forward unrolling models progressive seizure build-up
   - Backward unrolling models post-ictal suppression and neuro-stabilization
4. Dual-Context Temporal Pooling:
   - Target-epoch terminal state h_T preserves exact instantaneous staging context
   - 4-head attention pooling captures critical antecedent morphological transients
   - Concatenated representation: z = LayerNorm([h_T, AttentionPool(H)])
5. Deep Residual Hierarchical Classifier Head
========================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ATMRCell(nn.Module):
    """
    Alpha Transition-Memory Recurrent Cell (ATMRCell v3).
    Couples dynamic transition gating directly into recurrent state forgetting and ingestion.
    """
    def __init__(self, input_dim=506, hidden_dim=128, pathway_mode='dual', dropout=0.2):
        super(ATMRCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pathway_mode = pathway_mode  # 'dual', 'context_only', 'transition_only', 'triple'

        # Context Pathway Encoder
        self.enc_context = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Transition Pathway Encoder (First-Order Difference)
        if self.pathway_mode in ['dual', 'transition_only', 'triple']:
            self.enc_transition = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.6),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            # Dynamic Transition Gate conditioned on [context, velocity, h_{t-1}]
            self.gate_trans = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Sigmoid()
            )
            self.proj_v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            # Transition-coupled memory modulation weights
            self.w_trans_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.w_trans_i = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Second-Order Acceleration Pathway (for Ablation study)
        if self.pathway_mode == 'triple':
            self.enc_second = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.6),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.gate_second = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Sigmoid()
            )
            self.proj_a = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            self.w_second_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.w_second_i = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Standard Recurrent Gate Projections
        self.w_ih = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.w_hh = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.ln_c = nn.LayerNorm(hidden_dim)

    def forward(self, x_c, x_d1=None, x_d2=None, prev_state=None):
        batch_size = x_c.size(0)
        if prev_state is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=x_c.device)
            c_prev = torch.zeros(batch_size, self.hidden_dim, device=x_c.device)
        else:
            h_prev, c_prev = prev_state

        # 1. Compute Pathway Representations & Transition Gating
        if self.pathway_mode == 'context_only':
            u_t = self.enc_context(x_c)
            gates = self.w_ih(u_t) + self.w_hh(h_prev)
            f_gate, i_gate, c_cand, o_gate = gates.chunk(4, dim=-1)
            f_t = torch.sigmoid(f_gate)
            i_t = torch.sigmoid(i_gate)

        elif self.pathway_mode == 'transition_only':
            u_t = self.enc_transition(x_d1 if x_d1 is not None else x_c)
            gates = self.w_ih(u_t) + self.w_hh(h_prev)
            f_gate, i_gate, c_cand, o_gate = gates.chunk(4, dim=-1)
            f_t = torch.sigmoid(f_gate)
            i_t = torch.sigmoid(i_gate)

        elif self.pathway_mode == 'dual':
            e_c = self.enc_context(x_c)
            e_d1 = self.enc_transition(x_d1 if x_d1 is not None else x_c)
            gamma = self.gate_trans(torch.cat([e_c, e_d1, h_prev], dim=-1))
            v_t = self.proj_v(e_d1)
            # Dynamic blend between context and velocity
            u_t = (1.0 - gamma) * e_c + gamma * v_t

            gates = self.w_ih(u_t) + self.w_hh(h_prev)
            f_gate, i_gate, c_cand, o_gate = gates.chunk(4, dim=-1)
            # Modulate forget & input gates based on transition magnitude
            f_t = torch.sigmoid(f_gate - self.w_trans_f(gamma))
            i_t = torch.sigmoid(i_gate + self.w_trans_i(gamma))

        elif self.pathway_mode == 'triple':
            e_c = self.enc_context(x_c)
            e_d1 = self.enc_transition(x_d1 if x_d1 is not None else x_c)
            e_d2 = self.enc_second(x_d2 if x_d2 is not None else x_c)
            gamma_1 = self.gate_trans(torch.cat([e_c, e_d1, h_prev], dim=-1))
            gamma_2 = self.gate_second(torch.cat([e_c, e_d1, e_d2, h_prev], dim=-1))
            v_t = self.proj_v(e_d1)
            a_t = self.proj_a(e_d2)

            u_t = (1.0 - 0.5 * (gamma_1 + gamma_2)) * e_c + gamma_1 * v_t + gamma_2 * a_t

            gates = self.w_ih(u_t) + self.w_hh(h_prev)
            f_gate, i_gate, c_cand, o_gate = gates.chunk(4, dim=-1)
            f_t = torch.sigmoid(f_gate - self.w_trans_f(gamma_1) - self.w_second_f(gamma_2))
            i_t = torch.sigmoid(i_gate + self.w_trans_i(gamma_1) + self.w_second_i(gamma_2))
        else:
            raise ValueError(f"Unknown pathway mode: {self.pathway_mode}")

        c_cand = torch.tanh(c_cand)
        o_t = torch.sigmoid(o_gate)

        # 2. Recurrent Cell Memory Updates
        c_next = f_t * c_prev + i_t * c_cand
        h_next = o_t * torch.tanh(self.ln_c(c_next))

        return h_next, c_next

class ATMRNet(nn.Module):
    """
    Alpha Transition-Memory Recurrent Network (ATMR-Net v3).
    Bidirectional transition unrolling with target-epoch residual skip and multi-head temporal pooling.
    """
    def __init__(self, input_dim=506, hidden_dim=128, num_classes=5,
                 pathway_mode='dual', dropout=0.25, num_heads=4):
        super(ATMRNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pathway_mode = pathway_mode

        # Forward & Backward Transition Cells
        self.fwd_cell = ATMRCell(input_dim=input_dim, hidden_dim=hidden_dim, pathway_mode=pathway_mode, dropout=dropout)
        self.bwd_cell = ATMRCell(input_dim=input_dim, hidden_dim=hidden_dim, pathway_mode=pathway_mode, dropout=dropout)

        # Multi-Head Attention Temporal Pooling over sequence window
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout * 0.8
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim * 2))

        # Hierarchical Residual Classification Head
        # Inputs: concat([h_target, h_attn_pool]) of dimension (hidden_dim * 4)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, x_transition=None, x_second_order=None):
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Forward unroll
        h_f = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_f = torch.zeros(batch_size, self.hidden_dim, device=device)
        fwd_states = []
        for t in range(seq_len):
            x_c_t = x[:, t, :]
            x_d1_t = x_transition[:, t, :] if x_transition is not None else None
            x_d2_t = x_second_order[:, t, :] if x_second_order is not None else None
            h_f, c_f = self.fwd_cell(x_c_t, x_d1_t, x_d2_t, prev_state=(h_f, c_f))
            fwd_states.append(h_f.unsqueeze(1))
        fwd_seq = torch.cat(fwd_states, dim=1)

        # Backward unroll
        h_b = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_b = torch.zeros(batch_size, self.hidden_dim, device=device)
        bwd_states = []
        for t in reversed(range(seq_len)):
            x_c_t = x[:, t, :]
            x_d1_t = x_transition[:, t, :] if x_transition is not None else None
            x_d2_t = x_second_order[:, t, :] if x_second_order is not None else None
            h_b, c_b = self.bwd_cell(x_c_t, x_d1_t, x_d2_t, prev_state=(h_b, c_b))
            bwd_states.insert(0, h_b.unsqueeze(1))
        bwd_seq = torch.cat(bwd_states, dim=1)

        # Bidirectional sequence representation: (batch, seq_len, 2 * hidden_dim)
        bi_seq = torch.cat([fwd_seq, bwd_seq], dim=-1)

        # Target epoch terminal state (the staged epoch at index -1)
        target_state = bi_seq[:, -1, :]  # (batch, 2 * hidden_dim)

        # Multi-Head Attention Temporal Pooling over full 60s context
        q = self.query.expand(batch_size, -1, -1)
        attn_out, _ = self.mha(q, bi_seq, bi_seq)
        pooled = attn_out.squeeze(1)  # (batch, 2 * hidden_dim)

        # Combine target state with attention pooled historical context
        combined = torch.cat([target_state, pooled], dim=-1)  # (batch, 4 * hidden_dim)

        logits = self.classifier(combined)
        return logits

class ATMRNetSecondOrder(ATMRNet):
    """Convenience alias for ATMR-Net with triple (second-order) acceleration pathway."""
    def __init__(self, input_dim=506, hidden_dim=128, num_classes=5, dropout=0.25):
        super(ATMRNetSecondOrder, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            pathway_mode='triple',
            dropout=dropout
        )

if __name__ == "__main__":
    dummy_x = torch.randn(8, 10, 506)
    dummy_d1 = torch.randn(8, 10, 506)
    dummy_d2 = torch.randn(8, 10, 506)

    modes = ['dual', 'context_only', 'transition_only', 'triple']
    print("Testing upgraded ATMR-Net v3 variants...")
    for mode in modes:
        model = ATMRNet(input_dim=506, hidden_dim=128, pathway_mode=mode)
        out = model(dummy_x, dummy_d1, dummy_d2)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [OK] ATMR-Net ({mode:15s}) -> Output: {list(out.shape)} | Params: {params:,}")
    print("[OK] Upgraded ATMR-Net v3 architecture verified successfully.")
