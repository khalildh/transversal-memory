"""
exp_fast.py — Fast iteration harness for Plücker attention experiments

Speed optimizations:
  - Cached tokenized data (load once, reuse)
  - No baseline re-training (known: PPL ~206-209)
  - batch_size=128 (sweet spot for MPS throughput)
  - 7 epochs default (convergence plateau)
  - Early stopping (patience=2, min_delta=1.0 PPL)
  - Init from standard checkpoint (fine-tune 3 epochs)
  - Cosine LR schedule with warmup

Usage:
  python exp_fast.py dual_path              # dual-pathway incidence attention
  python exp_fast.py online_mem             # current best (scalar gate)
  python exp_fast.py dual_path --from-scratch  # train from scratch (no checkpoint init)
  python exp_fast.py dual_path --fast       # 2-layer screening model (~1 min)
  python exp_fast.py dual_path --baseline   # also train standard baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import pyarrow.parquet as pq
import time
import math
import sys
import argparse
from pathlib import Path
from itertools import combinations

# ── Config ───────────────────────────────────────────────────────────────────

class Config:
    d_model = 192
    n_heads = 6
    n_layers = 4
    dropout = 0.1
    seq_len = 128
    batch_size = 128
    n_epochs = 7
    lr = 3e-4
    grad_clip = 1.0
    warmup_steps = 50
    patience = 2         # early stopping patience (epochs)
    min_delta = 1.0      # min PPL improvement to count
    data_dir = Path("data/wikitext")
    cache_dir = Path("data/cache")

# ── Plücker primitives ──────────────────────────────────────────────────────

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

def make_plucker_pairs(point_dim):
    """Return pairs and dimensions for arbitrary Grassmannian G(2, point_dim)."""
    pairs = list(combinations(range(point_dim), 2))
    plucker_dim = len(pairs)  # C(point_dim, 2)
    return pairs, plucker_dim

# ── Attention variants ──────────────────────────────────────────────────────

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
    """Current best: scalar Gram energy gate."""
    def __init__(self, d_model, n_heads, dropout=0.1, decay=0.99, point_dim=4, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.decay = decay
        self.point_dim = point_dim
        self.pairs, self.plucker_dim = make_plucker_pairs(point_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W1_write = nn.Linear(d_model, point_dim * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, point_dim * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, point_dim * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, point_dim * n_heads, bias=False)
        self.mem_value = nn.Linear(d_model, d_model)
        self.mem_gate = nn.Linear(d_model, n_heads)
        self.mem_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        # Use identity for higher dims (model learns structure via projections)
        if point_dim == 4:
            self.register_buffer('J', _J6)
        else:
            self.register_buffer('J', torch.eye(self.plucker_dim))

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
        pd = self.point_dim
        w1 = self.W1_write(x_prev).reshape(B, T, H, pd)
        w2 = self.W2_write(x).reshape(B, T, H, pd)
        write_lines = exterior(w1, w2, self.pairs)
        r1 = self.W1_read(x).reshape(B, T, H, pd)
        r2 = self.W2_read(x).reshape(B, T, H, pd)
        read_lines = exterior(r1, r2, self.pairs)

        J = self.J
        J_write = torch.einsum('bthi,ij->bthj', write_lines, J)
        read_h = read_lines.permute(0, 2, 1, 3)
        Jwrite_h = J_write.permute(0, 2, 1, 3)
        incidence = read_h @ Jwrite_h.transpose(-1, -2)
        incidence_sq = incidence ** 2
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=0).bool()
        incidence_sq = incidence_sq.masked_fill(causal, 0.0)

        # Apply temporal weighting: λ^(t-s) for each (t, s) pair
        # When λ > 1 this amplifies older memories; when λ < 1 it decays them
        if self.decay != 1.0:
            positions = torch.arange(T, device=x.device, dtype=x.dtype)
            # weights[t, s] = λ^(t-s), shape (T, T)
            weights = self.decay ** (positions.unsqueeze(1) - positions.unsqueeze(0))
            weights = weights.tril(diagonal=-1)  # causal: only s < t
            incidence_sq = incidence_sq * weights.unsqueeze(0).unsqueeze(0)

        mem_score = incidence_sq.sum(dim=-1)  # (B, H, T)

        # Normalize per-position to prevent unbounded growth when λ > 1
        if self.decay > 1.0:
            mem_score = mem_score / (mem_score.amax(dim=-1, keepdim=True).clamp(min=1e-8))

        mem_val = self.mem_value(x)
        gate = torch.sigmoid(self.mem_gate(x))  # (B, T, H)
        scale = self.mem_scale.reshape(1, H, 1)
        mem_score_t = mem_score.permute(0, 2, 1)
        gated = torch.sigmoid(mem_score_t * scale.permute(0, 2, 1)) * gate
        gated = gated.mean(dim=-1, keepdim=True)

        return self.out(seq_out + gated * mem_val)


class MultiScaleMemoryAttention(nn.Module):
    """Online Gram memory with multi-scale write lines.

    Instead of only pairing consecutive tokens (offset=1), pairs at
    offsets {1, 2, 4, 8}. This captures structure at sentence-level
    (offset=1-2) and paragraph-level (offset=4-8) simultaneously.
    Inspired by Grassmann Flows (Dec 2025) multi-resolution approach.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, decay=0.99, point_dim=4,
                 offsets=(1, 2, 4, 8), **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.decay = decay
        self.point_dim = point_dim
        self.offsets = offsets
        self.pairs, self.plucker_dim = make_plucker_pairs(point_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Shared write/read projections across offsets
        self.W1_write = nn.Linear(d_model, point_dim * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, point_dim * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, point_dim * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, point_dim * n_heads, bias=False)
        self.mem_value = nn.Linear(d_model, d_model)
        self.mem_gate = nn.Linear(d_model, n_heads)
        self.mem_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        if point_dim == 4:
            self.register_buffer('J', _J6)
        else:
            self.register_buffer('J', torch.eye(self.plucker_dim))

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Standard attention
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        std_attn = std_attn.masked_fill(mask, float('-inf'))
        std_attn = self.drop(F.softmax(std_attn, dim=-1))
        seq_out = (std_attn @ v).transpose(1, 2).reshape(B, T, D)

        # Multi-scale write lines
        pd = self.point_dim
        J = self.J
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=0).bool()

        # Read lines (same for all scales)
        r1 = self.W1_read(x).reshape(B, T, H, pd)
        r2 = self.W2_read(x).reshape(B, T, H, pd)
        read_lines = exterior(r1, r2, self.pairs)
        read_h = read_lines.permute(0, 2, 1, 3)  # (B, H, T, plucker_dim)

        # Accumulate incidence scores across offsets
        total_incidence_sq = torch.zeros(B, H, T, T, device=x.device)

        for offset in self.offsets:
            if offset >= T:
                continue  # skip offsets larger than sequence
            # Shift x by offset positions
            x_past = torch.cat([
                torch.zeros(B, offset, D, device=x.device),
                x[:, :-offset]
            ], dim=1)
            w1 = self.W1_write(x_past).reshape(B, T, H, pd)
            w2 = self.W2_write(x).reshape(B, T, H, pd)
            write_lines = exterior(w1, w2, self.pairs)

            J_write = torch.einsum('bthi,ij->bthj', write_lines, J)
            Jwrite_h = J_write.permute(0, 2, 1, 3)
            incidence = read_h @ Jwrite_h.transpose(-1, -2)
            inc_sq = incidence ** 2
            inc_sq = inc_sq.masked_fill(causal, 0.0)
            total_incidence_sq = total_incidence_sq + inc_sq

        # Average across active offsets
        n_active = sum(1 for o in self.offsets if o < T)
        if n_active > 0:
            total_incidence_sq = total_incidence_sq / n_active

        # Temporal decay
        if self.decay != 1.0:
            positions = torch.arange(T, device=x.device, dtype=x.dtype)
            weights = self.decay ** (positions.unsqueeze(1) - positions.unsqueeze(0))
            weights = weights.tril(diagonal=-1)
            total_incidence_sq = total_incidence_sq * weights.unsqueeze(0).unsqueeze(0)

        mem_score = total_incidence_sq.sum(dim=-1)  # (B, H, T)

        mem_val = self.mem_value(x)
        gate = torch.sigmoid(self.mem_gate(x))
        scale = self.mem_scale.reshape(1, H, 1)
        mem_score_t = mem_score.permute(0, 2, 1)
        gated = torch.sigmoid(mem_score_t * scale.permute(0, 2, 1)) * gate
        gated = gated.mean(dim=-1, keepdim=True)

        return self.out(seq_out + gated * mem_val)


class GramMLPAttention(nn.Module):
    """Gram matrix with MLP readout — vector-valued memory instead of scalar gate.

    Instead of collapsing the Gram to a scalar score, extracts the upper triangle
    (21 features for 6×6) and projects through a small MLP to produce a d_head
    vector per head. This lets the model read rich structure from the Gram rather
    than getting a single yes/no gate.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, decay=0.99, point_dim=4, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.decay = decay
        self.point_dim = point_dim
        self.pairs, self.plucker_dim = make_plucker_pairs(point_dim)

        n_gram = self.plucker_dim * (self.plucker_dim + 1) // 2  # 21 for 6×6
        self.n_gram_features = n_gram

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W1_write = nn.Linear(d_model, point_dim * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, point_dim * n_heads, bias=False)

        # MLP: upper triangle of Gram → d_head vector per head
        self.gram_mlp = nn.Sequential(
            nn.Linear(n_gram, self.d_head),
            nn.GELU(),
            nn.Linear(self.d_head, self.d_head),
        )
        self.mem_gate = nn.Linear(d_model, n_heads)

        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        if point_dim == 4:
            self.register_buffer('J', _J6)
        else:
            self.register_buffer('J', torch.eye(self.plucker_dim))

        triu_i, triu_j = torch.triu_indices(self.plucker_dim, self.plucker_dim)
        self.register_buffer('triu_i', triu_i)
        self.register_buffer('triu_j', triu_j)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        pd = self.plucker_dim

        # Standard attention
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        std_attn = std_attn.masked_fill(mask, float('-inf'))
        std_attn = self.drop(F.softmax(std_attn, dim=-1))
        seq_out = std_attn @ v  # (B, H, T, dh)

        # Write lines from bigrams
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, self.point_dim)
        w2 = self.W2_write(x).reshape(B, T, H, self.point_dim)
        write_lines = exterior(w1, w2, self.pairs)  # (B, T, H, pd)
        wl = write_lines.permute(0, 2, 1, 3)  # (B, H, T, pd)

        # Outer products for all positions
        outer = wl.unsqueeze(-1) * wl.unsqueeze(-2)  # (B, H, T, pd, pd)

        # Sequential scan: M_t = decay * M_{t-1} + outer_{t-1}
        grams = torch.zeros(B, H, T, pd, pd, device=x.device)
        M = torch.zeros(B, H, pd, pd, device=x.device)
        for t in range(T):
            grams[:, :, t] = M
            M = self.decay * M + outer[:, :, t]

        # Extract upper triangle → (B, H, T, n_gram_features)
        gram_features = grams[:, :, :, self.triu_i, self.triu_j]

        # MLP readout → (B, H, T, dh)
        mem_out = self.gram_mlp(gram_features)

        # Input-dependent gate per head
        gate = torch.sigmoid(self.mem_gate(x))  # (B, T, H)
        gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (B, H, T, 1)

        # Standard attention output + gated memory vector per head
        combined = seq_out + gate * mem_out
        combined = combined.transpose(1, 2).reshape(B, T, D)
        return self.out(combined)


class DualPathAttention(nn.Module):
    """
    Dual-pathway: standard Q·K attention + Plücker incidence attention.

    Both pathways produce (B, H, T, T) attention matrices that get softmaxed
    and used to route values. The two outputs are combined with a learned gate.

    This lets the geometry actually ROUTE INFORMATION between tokens rather
    than just producing a scalar score.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, decay=0.99, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Standard pathway: Q, K, V
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Geometric pathway: write lines (bigram), read lines, separate values
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.geo_value = nn.Linear(d_model, d_model)

        # Per-head gate: how much geometric vs standard
        self.path_gate = nn.Linear(d_model, n_heads)
        # Scale for incidence logits
        self.incidence_scale = nn.Parameter(torch.full((n_heads,), 1.0))

        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J6', _J6)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # === Standard pathway ===
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_logits = (q @ k.transpose(-1, -2)) * self.scale  # (B, H, T, T)

        # === Geometric pathway ===
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)  # (B, T, H, 6)

        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)  # (B, T, H, 6)

        J = self.J6
        J_write = torch.einsum('bthi,ij->bthj', write_lines, J)
        read_h = read_lines.permute(0, 2, 1, 3)     # (B, H, T, 6)
        Jwrite_h = J_write.permute(0, 2, 1, 3)      # (B, H, T, 6)

        # Plücker incidence as attention logits
        geo_logits = read_h @ Jwrite_h.transpose(-1, -2)  # (B, H, T, T)
        scale = self.incidence_scale.reshape(1, H, 1, 1)
        geo_logits = geo_logits * scale

        # Causal mask (shared)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Standard attention → standard values
        std_attn = self.drop(F.softmax(std_logits.masked_fill(mask, float('-inf')), dim=-1))
        std_out = (std_attn @ v).transpose(1, 2).reshape(B, T, D)  # (B, T, D)

        # Geometric attention → geometric values
        geo_attn = self.drop(F.softmax(geo_logits.masked_fill(mask, float('-inf')), dim=-1))
        geo_v = self.geo_value(x).reshape(B, T, H, dh).permute(0, 2, 1, 3)  # (B, H, T, dh)
        geo_out = (geo_attn @ geo_v).transpose(1, 2).reshape(B, T, D)  # (B, T, D)

        # Per-head gate: interpolate between standard and geometric
        gate = torch.sigmoid(self.path_gate(x))  # (B, T, H)
        gate = gate.mean(dim=-1, keepdim=True)    # (B, T, 1)

        combined = (1 - gate) * std_out + gate * geo_out

        return self.out(combined)

# ── Model ────────────────────────────────────────────────────────────────────

ATTN_CLASSES = {
    "standard": StandardAttention,
    "online_mem": OnlineMemoryAttention,
    "multi_scale": MultiScaleMemoryAttention,
    "gram_mlp": GramMLPAttention,
    "dual_path": DualPathAttention,
}

class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_type, dropout=0.1, **kw):
        super().__init__()
        self.attn = ATTN_CLASSES[attn_type](d_model, n_heads, dropout=dropout, **kw)
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
    def __init__(self, vocab_size, cfg, attn_type, **kw):
        super().__init__()
        self.cfg = cfg
        self.attn_type = attn_type
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, attn_type, cfg.dropout, **kw)
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

# ── Data (cached) ────────────────────────────────────────────────────────────

_token_cache = {}

def load_tokens_cached(split, cfg):
    if split in _token_cache:
        return _token_cache[split]
    cache_file = cfg.cache_dir / f"wikitext_{split}_tokens.pt"
    if cache_file.exists():
        tokens = torch.load(cache_file, weights_only=True)
    else:
        enc = tiktoken.get_encoding("gpt2")
        fname = {"train": "train.parquet", "val": "val.parquet", "test": "test.parquet"}
        table = pq.read_table(cfg.data_dir / fname[split])
        text = "\n".join(t for t in table['text'].to_pylist() if t.strip())
        tokens = torch.tensor(enc.encode(text, allowed_special=set()), dtype=torch.long)
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, cache_file)
        print(f"  Cached {split} tokens: {len(tokens):,} → {cache_file}")
    _token_cache[split] = tokens
    return tokens

def make_batches(data, seq_len, batch_size):
    n = len(data) - 1
    n_seqs = (n // seq_len // batch_size) * batch_size
    data = data[:n_seqs * seq_len + 1]
    x = data[:-1].reshape(n_seqs, seq_len)
    y = data[1:].reshape(n_seqs, seq_len)
    perm = torch.randperm(n_seqs)
    return [(x[perm[i:i+batch_size]], y[perm[i:i+batch_size]])
            for i in range(0, n_seqs, batch_size)
            if i + batch_size <= n_seqs]

# ── Checkpoint init ──────────────────────────────────────────────────────────

def load_standard_checkpoint(model, device):
    """Load standard checkpoint and transfer shared weights."""
    ckpt_path = Path("checkpoints/lm_standard.pt")
    if not ckpt_path.exists():
        print("  No standard checkpoint found, training from scratch")
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    std_state = ckpt["model"]
    model_state = model.state_dict()

    loaded = 0
    for key in std_state:
        if key in model_state and std_state[key].shape == model_state[key].shape:
            model_state[key] = std_state[key]
            loaded += 1

    model.load_state_dict(model_state)
    total = len(model_state)
    print(f"  Loaded {loaded}/{total} params from standard checkpoint")
    return True

# ── Training ─────────────────────────────────────────────────────────────────

def get_lr(step, cfg, total_steps):
    """Cosine schedule with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))

def train(attn_type, cfg, device, from_checkpoint=True, data_frac=1.0, **kw):
    enc = tiktoken.get_encoding("gpt2")
    train_data = load_tokens_cached("train", cfg)
    if data_frac < 1.0:
        n = int(len(train_data) * data_frac)
        train_data = train_data[:n]
        print(f"  Using {data_frac*100:.0f}% of training data ({n:,} tokens)")
    val_data = load_tokens_cached("val", cfg)

    model = LM(enc.n_vocab, cfg, attn_type, **kw).to(device)
    print(f"\n  {attn_type}: {model.count_params():,} params")

    if from_checkpoint and attn_type != "standard":
        loaded = load_standard_checkpoint(model, device)
        if loaded:
            cfg = Config()  # reset to avoid mutating
            cfg.n_epochs = 3  # fine-tune only
            cfg.lr = 1e-4    # lower LR for fine-tuning
            print(f"  Fine-tuning for {cfg.n_epochs} epochs (lr={cfg.lr})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    n_batches = len(make_batches(train_data, cfg.seq_len, cfg.batch_size))
    total_steps = n_batches * cfg.n_epochs

    losses, ppls = [], []
    best_ppl = float('inf')
    patience_counter = 0
    global_step = 0

    for ep in range(cfg.n_epochs):
        t0 = time.time()
        model.train()
        batches = make_batches(train_data, cfg.seq_len, cfg.batch_size)
        ep_loss = 0.0
        for xb, yb in batches:
            lr = get_lr(global_step, cfg, total_steps)
            for pg in opt.param_groups:
                pg['lr'] = lr

            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb).view(-1, enc.n_vocab), yb.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); ep_loss += loss.item()
            global_step += 1
        avg = ep_loss / len(batches); losses.append(avg)

        model.eval()
        with torch.no_grad():
            vb = make_batches(val_data, cfg.seq_len, cfg.batch_size)
            vl = sum(F.cross_entropy(model(x.to(device)).view(-1, enc.n_vocab),
                     y.to(device).view(-1)).item() for x, y in vb) / len(vb)
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

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "type": attn_type,
                "losses": losses, "ppls": ppls, "vocab": enc.n_vocab},
               f"checkpoints/lm_{attn_type}.pt")
    return model, losses, ppls

# ── Generation ───────────────────────────────────────────────────────────────

def generate(model, enc, prompt, max_tok=60, temp=0.8, top_k=40):
    model.eval()
    dev = next(model.parameters()).device
    idx = torch.tensor([enc.encode(prompt, allowed_special=set())[-model.cfg.seq_len:]],
                       dtype=torch.long, device=dev)
    with torch.no_grad():
        for _ in range(max_tok):
            logits = model(idx[:, -model.cfg.seq_len:])[:, -1] / temp
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return enc.decode(idx[0].tolist())

# ── Main ─────────────────────────────────────────────────────────────────────

PROMPTS = [
    "The president of the United States",
    "In the year 1945 ,",
    "The cat sat on the",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=list(ATTN_CLASSES.keys()), nargs="?", default="dual_path")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch (no checkpoint)")
    parser.add_argument("--fast", action="store_true", help="2-layer screening model")
    parser.add_argument("--baseline", action="store_true", help="Also train standard baseline")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--decay", type=float, default=0.99, help="Memory decay rate")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--point-dim", type=int, default=4, help="Point dimension (4=6D Plücker, 5=10D, 6=15D)")
    parser.add_argument("--data-frac", type=float, default=1.0, help="Fraction of training data to use")
    parser.add_argument("--layers", type=int, default=None, help="Override number of layers")
    parser.add_argument("--d-model", type=int, default=None, help="Override model dimension")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = Config()

    if args.fast:
        cfg.n_layers = 2
        cfg.d_model = 128
        cfg.n_heads = 4
        print("  FAST MODE: 2 layers, d=128, 4 heads")

    if args.layers:
        cfg.n_layers = args.layers
    if args.d_model:
        cfg.d_model = args.d_model
        cfg.n_heads = max(1, args.d_model // 32)  # ~32 per head

    if args.seq_len:
        cfg.seq_len = args.seq_len
        if args.seq_len >= 256:
            cfg.batch_size = 64  # reduce batch for longer seqs

    if args.epochs:
        cfg.n_epochs = args.epochs

    kw = {"decay": args.decay, "point_dim": args.point_dim}
    _, plucker_dim = make_plucker_pairs(args.point_dim)

    print(f"{'='*60}")
    print(f"  {args.variant} | bs={cfg.batch_size} seq={cfg.seq_len} layers={cfg.n_layers}")
    print(f"  point_dim={args.point_dim} plucker_dim={plucker_dim}")
    print(f"  from_checkpoint={not args.from_scratch}")
    print(f"{'='*60}")

    # Train baseline if requested
    if args.baseline:
        print("\n  Training standard baseline...")
        std, _, std_p = train("standard", cfg, device, from_checkpoint=False,
                              data_frac=args.data_frac)
        print(f"  Standard baseline: PPL {min(std_p):.1f}")

    # Train variant
    var, _, var_p = train(args.variant, cfg, device,
                          from_checkpoint=not args.from_scratch,
                          data_frac=args.data_frac, **kw)

    # Results
    print(f"\n{'='*60}")
    print(f"  {args.variant}: best PPL = {min(var_p):.1f}")
    print(f"  Known standard baseline: ~206-209")
    print(f"  Params: {var.count_params():,}")
    print(f"{'='*60}")

    # Quick generation samples
    var = var.cpu()
    enc = tiktoken.get_encoding("gpt2")
    print("\n  SAMPLES:")
    for prompt in PROMPTS:
        print(f"  \"{prompt}\"")
        print(f"    → {generate(var, enc, prompt)}\n")

if __name__ == "__main__":
    main()
