"""
exp_new_ideas.py — Test 3 untried Gram memory ideas (offline-safe)

Uses a simple character-level tokenizer to avoid tiktoken network dependency.
Tests: iterated_gram, learned_transition, separate_rw vs standard + online_mem baselines.

Usage:
  uv run python exp_new_ideas.py
"""

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import sys
from pathlib import Path
from itertools import combinations

# ── Config ────────────────────────────────────────────────────────────────────

class Config:
    d_model = 128
    n_heads = 4
    n_layers = 2
    dropout = 0.1
    seq_len = 128
    batch_size = 64
    n_epochs = 10
    lr = 3e-4
    grad_clip = 1.0
    warmup_steps = 50
    patience = 3
    min_delta = 1.0

# ── Plücker primitives ───────────────────────────────────────────────────────

_PAIRS4 = list(combinations(range(4), 2))  # 6 pairs for P³

_J6 = torch.tensor([
    [0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
    [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],
], dtype=torch.float32)

def exterior(p1, p2, pairs=None):
    if pairs is None:
        pairs = _PAIRS4
    parts = [p1[...,i]*p2[...,j] - p1[...,j]*p2[...,i] for i,j in pairs]
    L = torch.stack(parts, dim=-1)
    return L / L.norm(dim=-1, keepdim=True).clamp(min=1e-12)

# ── Attention variants ────────────────────────────────────────────────────────

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class OnlineMemoryAttention(nn.Module):
    """Baseline online Gram memory (scalar gate)."""
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.mem_value = nn.Linear(d_model, d_model)
        self.mem_gate = nn.Linear(d_model, n_heads)
        self.mem_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J', _J6)
        self.decay_logits = nn.Parameter(torch.full((n_heads,), 4.6))

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        std_attn = std_attn.masked_fill(mask, float('-inf'))
        std_attn = self.drop(F.softmax(std_attn, dim=-1))
        seq_out = (std_attn @ v).transpose(1, 2).reshape(B, T, D)

        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)

        J = self.J
        J_write = torch.einsum('bthi,ij->bthj', write_lines, J)
        read_h = read_lines.permute(0, 2, 1, 3)
        Jwrite_h = J_write.permute(0, 2, 1, 3)
        incidence = read_h @ Jwrite_h.transpose(-1, -2)
        incidence_sq = incidence ** 2
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=0).bool()
        incidence_sq = incidence_sq.masked_fill(causal, 0.0)

        decays = torch.sigmoid(self.decay_logits)
        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        diffs = positions.unsqueeze(1) - positions.unsqueeze(0)
        weights = decays.reshape(H, 1, 1) ** diffs.unsqueeze(0)
        weights = weights * (diffs > 0).float().unsqueeze(0)
        incidence_sq = incidence_sq * weights.unsqueeze(0)

        mem_score = incidence_sq.sum(dim=-1)
        mem_val = self.mem_value(x)
        gate = torch.sigmoid(self.mem_gate(x))
        scale = self.mem_scale.reshape(1, H, 1)
        mem_score_t = mem_score.permute(0, 2, 1)
        gated = torch.sigmoid(mem_score_t * scale.permute(0, 2, 1)) * gate
        gated = gated.mean(dim=-1, keepdim=True)

        return self.out(seq_out + gated * mem_val)


class IteratedGramAttention(nn.Module):
    """Iterated Gram: interpolate between rd·M·rd and rd·M²·rd (2-hop matching)."""
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        pd = 6  # plucker dim for G(2,4)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.mem_value = nn.Linear(d_model, d_model)
        self.mem_gate = nn.Linear(d_model, n_heads)
        self.mem_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.iter_mix = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5 init
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J', _J6)
        self.decay_logits = nn.Parameter(torch.full((n_heads,), 4.6))

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        pd = 6

        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        std_attn = std_attn.masked_fill(mask, float('-inf'))
        std_attn = self.drop(F.softmax(std_attn, dim=-1))
        seq_out = (std_attn @ v).transpose(1, 2).reshape(B, T, D)

        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)

        J = self.J
        Jw = torch.einsum('bthi,ij->bthj', write_lines, J).permute(0, 2, 1, 3)  # (B,H,T,6)
        rd = read_lines.permute(0, 2, 1, 3)  # (B,H,T,6)

        # Build causal Grams via cumsum with learned decay
        decays = torch.sigmoid(self.decay_logits)
        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        outer = Jw.unsqueeze(-1) * Jw.unsqueeze(-2)  # (B,H,T,6,6)
        decay_inv = (1.0 / decays.clamp(min=1e-6)).unsqueeze(1) ** positions.unsqueeze(0)
        outer_scaled = outer * decay_inv.reshape(1, H, T, 1, 1)
        M_cumsum = torch.cumsum(outer_scaled, dim=2)
        M_cumsum = torch.cat([
            torch.zeros(B, H, 1, pd, pd, device=x.device, dtype=x.dtype),
            M_cumsum[:, :, :-1]
        ], dim=2)
        decay_fwd = decays.unsqueeze(1) ** positions.unsqueeze(0)
        M_all = M_cumsum * decay_fwd.reshape(1, H, T, 1, 1)  # (B,H,T,6,6)

        # M score: rd · M · rd^T
        rd_M = torch.einsum('bhti,bhtij->bhtj', rd, M_all)
        score_M = (rd_M * rd).sum(dim=-1)  # (B,H,T)

        # M² score: ||rd · M||²
        score_M2 = (rd_M * rd_M).sum(dim=-1)  # (B,H,T)

        alpha = torch.sigmoid(self.iter_mix)
        mem_score = (1 - alpha) * score_M + alpha * score_M2

        mem_val = self.mem_value(x)
        gate = torch.sigmoid(self.mem_gate(x))
        scale = self.mem_scale.reshape(1, H, 1)
        mem_score_t = mem_score.permute(0, 2, 1)
        gated = torch.sigmoid(mem_score_t * scale.permute(0, 2, 1)) * gate
        gated = gated.mean(dim=-1, keepdim=True)

        return self.out(seq_out + gated * mem_val)


class LearnedTransitionAttention(nn.Module):
    """Learned state transition: M_t = A · M_{t-1} · A^T + w_t ⊗ w_t."""
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        pd = 6

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.mem_value = nn.Linear(d_model, d_model)
        self.mem_gate = nn.Linear(d_model, n_heads)
        self.mem_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J', _J6)

        # Per-head transition matrix, init as 0.99*I (near scalar decay)
        A_init = 0.99 * torch.eye(pd).unsqueeze(0).expand(n_heads, -1, -1).clone()
        self.A = nn.Parameter(A_init)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        pd = 6

        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        std_attn = std_attn.masked_fill(mask, float('-inf'))
        std_attn = self.drop(F.softmax(std_attn, dim=-1))
        seq_out = (std_attn @ v).transpose(1, 2).reshape(B, T, D)

        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)

        J = self.J
        Jw = torch.einsum('bthi,ij->bthj', write_lines, J).permute(0, 2, 1, 3)
        rd = read_lines.permute(0, 2, 1, 3)

        # Sequential scan: M_t = A · M_{t-1} · A^T + Jw_t ⊗ Jw_t
        A = self.A
        A_T = A.transpose(-1, -2)
        M = torch.zeros(B, H, pd, pd, device=x.device, dtype=x.dtype)
        scores = []
        for t in range(T):
            rd_t = rd[:, :, t, :]
            score_t = torch.einsum('bhp,bhpq,bhq->bh', rd_t, M, rd_t)
            scores.append(score_t)
            Jw_t = Jw[:, :, t, :]
            M = torch.einsum('hpq,bhqr,hrs->bhps', A, M, A_T) + \
                Jw_t.unsqueeze(-1) * Jw_t.unsqueeze(-2)

        mem_score = torch.stack(scores, dim=-1)

        mem_val = self.mem_value(x)
        gate = torch.sigmoid(self.mem_gate(x))
        scale = self.mem_scale.reshape(1, H, 1)
        mem_score_t = mem_score.permute(0, 2, 1)
        gated = torch.sigmoid(mem_score_t * scale.permute(0, 2, 1)) * gate
        gated = gated.mean(dim=-1, keepdim=True)

        return self.out(seq_out + gated * mem_val)


class SeparateRWGramAttention(nn.Module):
    """Separate read/write Gram: forward + backward association."""
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        pd = 6

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.mem_value = nn.Linear(d_model, d_model)
        self.mem_gate = nn.Linear(d_model, n_heads)
        self.mem_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.rw_mix = nn.Parameter(torch.tensor(0.0))
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J', _J6)
        self.decay_logits = nn.Parameter(torch.full((n_heads,), 4.6))

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        pd = 6

        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        std_attn = std_attn.masked_fill(mask, float('-inf'))
        std_attn = self.drop(F.softmax(std_attn, dim=-1))
        seq_out = (std_attn @ v).transpose(1, 2).reshape(B, T, D)

        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)

        J = self.J
        Jw = torch.einsum('bthi,ij->bthj', write_lines, J).permute(0, 2, 1, 3)
        Jr = torch.einsum('bthi,ij->bthj', read_lines, J).permute(0, 2, 1, 3)
        rd = read_lines.permute(0, 2, 1, 3)

        decays = torch.sigmoid(self.decay_logits)
        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        decay_inv = (1.0 / decays.clamp(min=1e-6)).unsqueeze(1) ** positions.unsqueeze(0)
        decay_fwd = decays.unsqueeze(1) ** positions.unsqueeze(0)

        # Write Gram: M_write
        outer_w = Jw.unsqueeze(-1) * Jw.unsqueeze(-2)
        ow_s = outer_w * decay_inv.reshape(1, H, T, 1, 1)
        Mw = torch.cumsum(ow_s, dim=2)
        Mw = torch.cat([torch.zeros(B, H, 1, pd, pd, device=x.device, dtype=x.dtype), Mw[:, :, :-1]], dim=2)
        Mw = Mw * decay_fwd.reshape(1, H, T, 1, 1)

        # Read Gram: M_read
        outer_r = Jr.unsqueeze(-1) * Jr.unsqueeze(-2)
        or_s = outer_r * decay_inv.reshape(1, H, T, 1, 1)
        Mr = torch.cumsum(or_s, dim=2)
        Mr = torch.cat([torch.zeros(B, H, 1, pd, pd, device=x.device, dtype=x.dtype), Mr[:, :, :-1]], dim=2)
        Mr = Mr * decay_fwd.reshape(1, H, T, 1, 1)

        # Forward: rd · M_write · rd^T
        rd_Mw = torch.einsum('bhti,bhtij->bhtj', rd, Mw)
        score_w = (rd_Mw * rd).sum(dim=-1)

        # Backward: Jw · M_read · Jw^T
        Jw_Mr = torch.einsum('bhti,bhtij->bhtj', Jw, Mr)
        score_r = (Jw_Mr * Jw).sum(dim=-1)

        alpha = torch.sigmoid(self.rw_mix)
        mem_score = (1 - alpha) * score_w + alpha * score_r

        mem_val = self.mem_value(x)
        gate = torch.sigmoid(self.mem_gate(x))
        scale = self.mem_scale.reshape(1, H, 1)
        mem_score_t = mem_score.permute(0, 2, 1)
        gated = torch.sigmoid(mem_score_t * scale.permute(0, 2, 1)) * gate
        gated = gated.mean(dim=-1, keepdim=True)

        return self.out(seq_out + gated * mem_val)


# ── Model ─────────────────────────────────────────────────────────────────────

ATTN_CLASSES = {
    "standard": StandardAttention,
    "online_mem": OnlineMemoryAttention,
    "iterated_gram": IteratedGramAttention,
    "learned_transition": LearnedTransitionAttention,
    "separate_rw": SeparateRWGramAttention,
}

class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_type, dropout=0.1):
        super().__init__()
        self.attn = ATTN_CLASSES[attn_type](d_model, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class LM(nn.Module):
    def __init__(self, vocab_size, cfg, attn_type):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, attn_type, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ── Data generation ───────────────────────────────────────────────────────────

def make_data(n_tokens, vocab_size, seq_len, batch_size):
    """Generate synthetic token data with bigram structure (not pure random)."""
    torch.manual_seed(42)
    # Create a bigram transition matrix with some structure
    transition = torch.randn(vocab_size, vocab_size).softmax(dim=-1)
    # Generate tokens following bigram distribution
    tokens = torch.zeros(n_tokens, dtype=torch.long)
    tokens[0] = torch.randint(0, vocab_size, (1,))
    for i in range(1, n_tokens):
        tokens[i] = torch.multinomial(transition[tokens[i-1]], 1)

    n_seqs = (n_tokens - 1) // seq_len // batch_size * batch_size
    data = tokens[:n_seqs * seq_len + 1]
    x = data[:-1].reshape(n_seqs, seq_len)
    y = data[1:].reshape(n_seqs, seq_len)
    perm = torch.randperm(n_seqs)
    return [(x[perm[i:i+batch_size]], y[perm[i:i+batch_size]])
            for i in range(0, n_seqs, batch_size)
            if i + batch_size <= n_seqs]

# ── Training ──────────────────────────────────────────────────────────────────

def get_lr(step, cfg, total_steps):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))

def train(attn_type, cfg, device, vocab_size, train_batches, val_batches):
    model = LM(vocab_size, cfg, attn_type).to(device)
    n_params = model.count_params()
    print(f"  {attn_type}: {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    total_steps = len(train_batches) * cfg.n_epochs

    losses, ppls = [], []
    best_ppl = float('inf')
    patience_counter = 0
    global_step = 0

    for ep in range(cfg.n_epochs):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        for xb, yb in train_batches:
            lr = get_lr(global_step, cfg, total_steps)
            for pg in opt.param_groups:
                pg['lr'] = lr
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb).view(-1, vocab_size), yb.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); ep_loss += loss.item()
            global_step += 1
        avg = ep_loss / len(train_batches); losses.append(avg)

        model.eval()
        with torch.no_grad():
            vl = sum(F.cross_entropy(model(x.to(device)).view(-1, vocab_size),
                     y.to(device).view(-1)).item() for x, y in val_batches) / len(val_batches)
        ppl = math.exp(min(vl, 20)); ppls.append(ppl)

        elapsed = time.time() - t0
        improved = "★" if ppl < best_ppl - cfg.min_delta else ""
        print(f"    ep {ep+1:2d}: loss={avg:.3f}  ppl={ppl:.1f}  ({elapsed:.1f}s) {improved}")

        if ppl < best_ppl - cfg.min_delta:
            best_ppl = ppl
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"    Early stopping (no improvement for {cfg.patience} epochs)")
                break

    return best_ppl, n_params, ppls

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()
    vocab_size = 256  # character-level
    n_train = 500_000
    n_val = 50_000

    print(f"\nGenerating synthetic bigram data (vocab={vocab_size}, train={n_train:,}, val={n_val:,})...")
    train_batches = make_data(n_train, vocab_size, cfg.seq_len, cfg.batch_size)
    val_batches = make_data(n_val, vocab_size, cfg.seq_len, cfg.batch_size)
    print(f"  {len(train_batches)} train batches, {len(val_batches)} val batches")

    variants = ["standard", "online_mem", "iterated_gram", "learned_transition", "separate_rw"]
    results = {}

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"  Training: {variant}")
        print(f"{'='*60}")
        best_ppl, n_params, ppls = train(variant, cfg, device, vocab_size,
                                          train_batches, val_batches)
        results[variant] = {"ppl": best_ppl, "params": n_params}

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY (2-layer, d=128, synthetic bigram data)")
    print(f"{'='*60}")
    std_ppl = results["standard"]["ppl"]
    for variant in variants:
        r = results[variant]
        delta = ((r["ppl"] - std_ppl) / std_ppl) * 100
        marker = "←baseline" if variant == "standard" else f"{delta:+.1f}%"
        print(f"  {variant:25s}  PPL {r['ppl']:7.1f}  params {r['params']:,}  {marker}")

    # Print learned parameter values for new variants
    print(f"\n  Learned parameters:")
    for variant in ["iterated_gram", "learned_transition", "separate_rw"]:
        if variant == "iterated_gram":
            pass  # iter_mix printed during training isn't easily accessible
        print(f"    {variant}: see training logs above")

if __name__ == "__main__":
    main()
