"""
exp_assoc_mem.py — Online associative memory for language modeling

Instead of seeding M₀ at sequence start (which doesn't help), this gives the
model a persistent associative memory it can read/write at every position.

Architecture:
  - Standard Q·K attention (unchanged)
  - Online Gram accumulation within sequence (unchanged)
  - Triadic associative memory: write hidden states during training,
    query at every position during forward pass → additive memory signal

The triadic memory stores (key_SDR, layer_SDR, value_SDR) triples.
Keys are derived from the model's Plücker write lines (geometric fingerprint).
Values are hidden state representations.

Batched querying: instead of one-at-a-time triadic queries, we batch all
positions in a sequence into a single matrix multiply against the indicator
matrices.

Usage:
  uv run python exp_assoc_mem.py --fast --baseline --from-scratch
  uv run python exp_assoc_mem.py --fast --from-scratch    # assoc mem only
  uv run python exp_assoc_mem.py --baseline --from-scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import pyarrow.parquet as pq
import numpy as np
import time
import math
import sys
import argparse
from pathlib import Path
from itertools import combinations

sys.path.insert(0, "/Volumes/PRO-G40/Code/TDGA/src")

# ── Config ───────────────────────────────────────────────────────────────────

class Config:
    d_model = 192
    n_heads = 6
    n_layers = 4
    dropout = 0.1
    seq_len = 128
    batch_size = 128
    n_epochs = 10
    lr = 3e-4
    grad_clip = 1.0
    warmup_steps = 50
    patience = 3
    min_delta = 0.5
    data_dir = Path("data/wikitext")
    cache_dir = Path("data/cache")
    # Associative memory
    sdr_n = 5000         # SDR dimensionality
    sdr_p = 50           # active bits per SDR
    decay = 0.99         # Gram decay
    write_interval = 8   # store to assoc memory every N positions
    mem_start_epoch = 1  # start reading from memory at this epoch

# ── Batched Triadic Memory ────────────────────────────────────────────────────

class BatchedTriadicMemory:
    """Triadic associative memory with batched GPU queries.

    Stores (key, layer, value) triples as indicator matrices.
    Supports querying many keys at once via batched matrix multiplies.
    """

    def __init__(self, N, P, device="cpu"):
        self.N = N
        self.P = P
        self.device = torch.device(device)
        # Indicator matrices: (num_triples, N)
        self._key_rows = []
        self._layer_rows = []
        self._val_rows = []
        # Materialized on device
        self._key_mat = None
        self._layer_mat = None
        self._val_mat = None
        self._dirty = False
        self.n_stored = 0

    def store_batch(self, key_sdrs, layer_sdr, val_sdrs):
        """Store a batch of (key, layer, value) triples.

        key_sdrs: list of np.ndarray (sorted indices), one per item
        layer_sdr: np.ndarray (sorted indices), shared across batch
        val_sdrs: list of np.ndarray (sorted indices), one per item
        """
        N = self.N
        layer_row = torch.zeros(N, dtype=torch.float32)
        layer_row[torch.from_numpy(layer_sdr.astype(np.int64))] = 1.0

        for key_sdr, val_sdr in zip(key_sdrs, val_sdrs):
            key_row = torch.zeros(N, dtype=torch.float32)
            key_row[torch.from_numpy(key_sdr.astype(np.int64))] = 1.0
            val_row = torch.zeros(N, dtype=torch.float32)
            val_row[torch.from_numpy(val_sdr.astype(np.int64))] = 1.0
            self._key_rows.append(key_row)
            self._layer_rows.append(layer_row)
            self._val_rows.append(val_row)
            self.n_stored += 1

        self._dirty = True

    def _rebuild(self):
        if not self._dirty or not self._key_rows:
            return

        for attr, buf in [("_key_mat", self._key_rows),
                          ("_layer_mat", self._layer_rows),
                          ("_val_mat", self._val_rows)]:
            new = torch.stack(buf)
            existing = getattr(self, attr)
            if existing is not None:
                existing_cpu = existing.cpu() if existing.device.type != "cpu" else existing
                combined = torch.cat([existing_cpu, new], dim=0)
            else:
                combined = new
            setattr(self, attr, combined.to(self.device))
            buf.clear()

        self._dirty = False

    def query_batch(self, key_dense_batch, layer_sdr):
        """Query with a batch of dense key vectors + shared layer SDR.

        key_dense_batch: (Q, N) float tensor on device — dense SDR representations
        layer_sdr: np.ndarray of sorted indices

        Returns: (Q, N) float tensor — summed votes for each query
        """
        self._rebuild()
        if self._key_mat is None or self._key_mat.shape[0] == 0:
            return None

        # Layer overlap: (T_stored,) — same for all queries
        layer_oh = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        layer_oh[torch.from_numpy(layer_sdr.astype(np.int64)).to(self.device)] = 1.0
        layer_overlap = self._layer_mat @ layer_oh  # (T_stored,)

        # Key overlap: (Q, T_stored) — batched
        key_overlap = key_dense_batch @ self._key_mat.T  # (Q, T_stored)

        # Combined weights: (Q, T_stored)
        weights = key_overlap * layer_overlap.unsqueeze(0)

        # Vote into value space: (Q, N)
        sums = weights @ self._val_mat  # (Q, N)

        return sums

    def sums_to_sdrs(self, sums):
        """Convert vote sums to SDR indices (top-P per query)."""
        _, topk = torch.topk(sums, self.P, dim=-1)
        return topk  # (Q, P) — indices on device


# ── SDR encoding via random projection ────────────────────────────────────────

class RandomProjectionEncoder:
    """Encode continuous vectors as dense binary SDR vectors for batched queries."""

    def __init__(self, input_dim, n=5000, p=50, seed=42):
        self.n = n
        self.p = p
        rng = np.random.default_rng(seed)
        self.R = rng.standard_normal((input_dim, n)).astype(np.float32)
        norms = np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12
        self.R = self.R / norms
        self.R_torch = None  # lazy init on device
        self.R_pinv = np.linalg.pinv(self.R)
        self.R_pinv_torch = None

    def to_device(self, device):
        self.R_torch = torch.from_numpy(self.R).to(device)
        self.R_pinv_torch = torch.from_numpy(self.R_pinv).to(device)
        return self

    def encode_batch_dense(self, vecs_torch):
        """(B, input_dim) tensor → (B, N) dense binary SDR vectors on device."""
        projected = vecs_torch @ self.R_torch  # (B, N)
        # Top-P per row → binary
        _, topk = torch.topk(projected, self.p, dim=-1)  # (B, P)
        dense = torch.zeros_like(projected)
        dense.scatter_(1, topk, 1.0)
        return dense

    def encode_single(self, vec_np):
        """Single numpy vector → sorted SDR indices."""
        projected = vec_np.astype(np.float32) @ self.R
        indices = np.argsort(-projected)[:self.p]
        return np.sort(indices).astype(np.uint32)

    def decode_batch(self, dense_sdrs):
        """(B, N) dense binary → (B, input_dim) approximate reconstruction."""
        return dense_sdrs @ self.R_pinv_torch


# ── Plücker primitives ──────────────────────────────────────────────────────

_PAIRS4 = list(combinations(range(4), 2))  # 6 pairs for P³

_J6 = torch.tensor([
    [0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
    [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],
], dtype=torch.float32)

def exterior(p1, p2):
    parts = [p1[...,i]*p2[...,j] - p1[...,j]*p2[...,i] for i,j in _PAIRS4]
    L = torch.stack(parts, dim=-1)
    return L / L.norm(dim=-1, keepdim=True).clamp(min=1e-12)


# ── Attention with Online Associative Memory ─────────────────────────────────

class AssocMemoryAttention(nn.Module):
    """Online Gram memory + persistent associative memory.

    The online Gram accumulates within-sequence relational structure (as before).
    The associative memory provides cross-sequence retrieval: at each position,
    query with current hidden state → get back a value vector from any past
    sequence that had similar geometric structure.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, decay=0.99, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.decay = decay

        # Standard attention
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Plücker lines for online Gram
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)

        # Online Gram gating
        self.mem_value = nn.Linear(d_model, d_model)
        self.mem_gate = nn.Linear(d_model, n_heads)
        self.mem_scale = nn.Parameter(torch.full((n_heads,), 0.1))

        # Associative memory signal (from triadic recall)
        self.assoc_proj = nn.Linear(d_model, d_model)
        self.assoc_gate = nn.Parameter(torch.tensor(0.01))  # start small

        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J', _J6)

        # Cache write lines for storage
        self._cached_write_lines = None

    def forward(self, x, assoc_values=None):
        """
        x: (B, T, D)
        assoc_values: optional (B, T, D) — recalled values from associative memory
        """
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Standard Q·K attention
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        std_attn = std_attn.masked_fill(mask, float('-inf'))
        std_attn = self.drop(F.softmax(std_attn, dim=-1))
        seq_out = (std_attn @ v).transpose(1, 2).reshape(B, T, D)

        # Write lines from bigrams
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)  # (B, T, H, 6)
        self._cached_write_lines = write_lines.detach()

        # Read lines
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)  # (B, T, H, 6)

        # Online Gram incidence (within-sequence)
        J = self.J
        J_write = torch.einsum('bthi,ij->bthj', write_lines, J)
        read_h = read_lines.permute(0, 2, 1, 3)
        Jwrite_h = J_write.permute(0, 2, 1, 3)
        incidence = read_h @ Jwrite_h.transpose(-1, -2)
        incidence_sq = incidence ** 2
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=0).bool()
        incidence_sq = incidence_sq.masked_fill(causal, 0.0)

        if self.decay != 1.0:
            positions = torch.arange(T, device=x.device, dtype=x.dtype)
            weights = self.decay ** (positions.unsqueeze(1) - positions.unsqueeze(0))
            weights = weights.tril(diagonal=-1)
            incidence_sq = incidence_sq * weights.unsqueeze(0).unsqueeze(0)

        mem_score = incidence_sq.sum(dim=-1)  # (B, H, T)

        # Gate and combine online Gram
        mem_val = self.mem_value(x)
        gate = torch.sigmoid(self.mem_gate(x))  # (B, T, H)
        scale = self.mem_scale.reshape(1, H, 1)
        mem_score_t = mem_score.permute(0, 2, 1)  # (B, T, H)
        gated = torch.sigmoid(mem_score_t * scale.permute(0, 2, 1)) * gate
        gated = gated.mean(dim=-1, keepdim=True)

        combined = seq_out + gated * mem_val

        # Add associative memory signal
        if assoc_values is not None:
            combined = combined + self.assoc_gate * self.assoc_proj(assoc_values)

        return self.out(combined)


# ── Model ────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_cls, dropout=0.1, **kw):
        super().__init__()
        self.attn = attn_cls(d_model, n_heads, dropout=dropout, **kw)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, assoc_values=None):
        x = x + self.attn(self.ln1(x), assoc_values=assoc_values)
        x = x + self.ffn(self.ln2(x))
        return x


class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, assoc_values=None):
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


class LM(nn.Module):
    def __init__(self, vocab_size, cfg, attn_cls, **kw):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, attn_cls, cfg.dropout, **kw)
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

    def forward(self, idx, assoc_values_per_layer=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for i, block in enumerate(self.blocks):
            av = assoc_values_per_layer[i] if assoc_values_per_layer is not None else None
            x = block(x, assoc_values=av)
        return self.head(self.ln_f(x))

    def get_hidden_states(self, idx):
        """Get hidden states after each block (for writing to memory)."""
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x.detach())
        return states

    def get_cached_write_lines(self):
        lines = []
        for block in self.blocks:
            if hasattr(block.attn, '_cached_write_lines') and block.attn._cached_write_lines is not None:
                lines.append(block.attn._cached_write_lines)
            else:
                lines.append(None)
        return lines

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Associative Memory Manager ───────────────────────────────────────────────

class AssocMemoryManager:
    """Manages the triadic associative memory for all layers.

    Keys: Plücker write lines (6D per head, flattened per position)
    Values: Hidden states (d_model per position)
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.n_layers = cfg.n_layers

        # Key encoder: uses flattened write lines (n_heads * 6 = 36 for 6 heads)
        key_dim = cfg.n_heads * 6
        self.key_encoder = RandomProjectionEncoder(
            key_dim, n=cfg.sdr_n, p=cfg.sdr_p, seed=100
        ).to_device(device)

        # Value encoder: uses hidden states (d_model)
        self.val_encoder = RandomProjectionEncoder(
            cfg.d_model, n=cfg.sdr_n, p=cfg.sdr_p, seed=200
        ).to_device(device)

        # Fixed layer SDRs
        rng = np.random.default_rng(300)
        self.layer_sdrs = [
            np.sort(rng.permutation(cfg.sdr_n)[:cfg.sdr_p]).astype(np.uint32)
            for _ in range(cfg.n_layers)
        ]

        # One triadic memory per layer (keeps them separate, faster queries)
        self.memories = [
            BatchedTriadicMemory(N=cfg.sdr_n, P=cfg.sdr_p, device=str(device))
            for _ in range(cfg.n_layers)
        ]

    def write(self, write_lines_per_layer, hidden_states_per_layer, interval=8):
        """Write to associative memory from cached write lines + hidden states.

        Subsamples positions by interval to control memory growth.
        write_lines_per_layer: list of (B, T, H, 6) tensors
        hidden_states_per_layer: list of (B, T, D) tensors
        """
        for layer_idx in range(self.n_layers):
            wl = write_lines_per_layer[layer_idx]
            hs = hidden_states_per_layer[layer_idx]
            if wl is None or hs is None:
                continue

            B, T, H, _ = wl.shape

            # Flatten write lines per position: (B, T, H*6)
            wl_flat = wl.reshape(B, T, H * 6)

            # Subsample positions
            positions = list(range(0, T, interval))
            if not positions:
                continue

            for pos in positions:
                # Encode keys for all batch items at this position
                keys_at_pos = wl_flat[:, pos, :]  # (B, H*6)
                vals_at_pos = hs[:, pos, :]  # (B, D)

                # Encode as SDRs (numpy, one at a time for now)
                key_sdrs = []
                val_sdrs = []
                for b in range(B):
                    key_sdrs.append(self.key_encoder.encode_single(
                        keys_at_pos[b].cpu().numpy()))
                    val_sdrs.append(self.val_encoder.encode_single(
                        vals_at_pos[b].cpu().numpy()))

                self.memories[layer_idx].store_batch(
                    key_sdrs, self.layer_sdrs[layer_idx], val_sdrs)

    def read(self, write_lines_per_layer):
        """Read from associative memory for all positions.

        Returns list of (B, T, D) tensors per layer — recalled values.
        """
        results = []
        for layer_idx in range(self.n_layers):
            wl = write_lines_per_layer[layer_idx]
            mem = self.memories[layer_idx]

            if wl is None or mem.n_stored == 0:
                results.append(None)
                continue

            B, T, H, _ = wl.shape
            wl_flat = wl.reshape(B, T, H * 6)

            # Encode all positions as dense SDR keys: (B*T, N)
            wl_2d = wl_flat.reshape(B * T, H * 6)
            key_dense = self.key_encoder.encode_batch_dense(wl_2d)  # (B*T, N)

            # Batched query: (B*T, N) → (B*T, N) vote sums
            vote_sums = mem.query_batch(key_dense, self.layer_sdrs[layer_idx])

            if vote_sums is None:
                results.append(None)
                continue

            # Decode votes back to value space: (B*T, N) → (B*T, D)
            # Use top-P to get SDR, then decode
            # But we can also decode directly from the soft votes (smoother)
            recalled_vals = self.val_encoder.decode_batch(vote_sums)  # (B*T, D)

            # Normalize to prevent scale explosion
            recalled_vals = recalled_vals / (recalled_vals.norm(dim=-1, keepdim=True).clamp(min=1e-8))

            results.append(recalled_vals.reshape(B, T, -1))

        return results

    @property
    def total_stored(self):
        return sum(m.n_stored for m in self.memories)


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


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(step, cfg, total_steps):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


# ── Training ──────────────────────────────────────────────────────────────────

def train(attn_cls, cfg, device, from_checkpoint=False, use_assoc=False,
          label="assoc_mem", **kw):
    enc = tiktoken.get_encoding("gpt2")
    train_data = load_tokens_cached("train", cfg)
    val_data = load_tokens_cached("val", cfg)

    model = LM(enc.n_vocab, cfg, attn_cls, **kw).to(device)
    print(f"\n  {label}: {model.count_params():,} params")

    assoc = AssocMemoryManager(cfg, device) if use_assoc else None

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
        read_batches = 0

        for xb, yb in batches:
            lr = get_lr(global_step, cfg, total_steps)
            for pg in opt.param_groups:
                pg['lr'] = lr

            xb, yb = xb.to(device), yb.to(device)

            # Read from associative memory (if enabled and past warmup)
            assoc_values = None
            if use_assoc and assoc is not None and ep >= cfg.mem_start_epoch:
                with torch.no_grad():
                    # Need write lines from current input to form queries
                    # Do a no-grad forward to get write lines
                    _ = model(xb)
                    wl_per_layer = model.get_cached_write_lines()
                    assoc_values = assoc.read(wl_per_layer)
                    if any(v is not None for v in assoc_values):
                        read_batches += 1

            # Forward with optional associative memory values
            logits = model(xb, assoc_values_per_layer=assoc_values)
            loss = F.cross_entropy(logits.view(-1, enc.n_vocab), yb.view(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss += loss.item()
            global_step += 1

            # Write to associative memory
            if use_assoc and assoc is not None:
                with torch.no_grad():
                    wl_per_layer = model.get_cached_write_lines()
                    # Get hidden states for values
                    hs_per_layer = model.get_hidden_states(xb)
                    assoc.write(wl_per_layer, hs_per_layer,
                               interval=cfg.write_interval)

        avg = ep_loss / len(batches)
        losses.append(avg)

        # Validation (without assoc memory for fair comparison)
        model.eval()
        with torch.no_grad():
            vb = make_batches(val_data, cfg.seq_len, cfg.batch_size)
            vl = sum(F.cross_entropy(model(x.to(device)).view(-1, enc.n_vocab),
                     y.to(device).view(-1)).item() for x, y in vb) / len(vb)
        ppl = math.exp(min(vl, 20))
        ppls.append(ppl)

        elapsed = time.time() - t0
        improved = "★" if ppl < best_ppl - cfg.min_delta else ""
        mem_info = ""
        if assoc:
            mem_info = f" read={read_batches}/{len(batches)} stored={assoc.total_stored}"
        print(f"    ep {ep+1:2d}: loss={avg:.3f}  ppl={ppl:.1f}  ({elapsed:.1f}s){mem_info} {improved}")

        if ppl < best_ppl - cfg.min_delta:
            best_ppl = ppl
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"    Early stopping (no improvement for {cfg.patience} epochs)")
                break

    # Evaluate WITH assoc memory on val set
    if use_assoc and assoc is not None and assoc.total_stored > 0:
        model.eval()
        with torch.no_grad():
            vb = make_batches(val_data, cfg.seq_len, cfg.batch_size)
            assoc_losses = []
            for x, y in vb:
                x, y = x.to(device), y.to(device)
                _ = model(x)
                wl = model.get_cached_write_lines()
                av = assoc.read(wl)
                vl = F.cross_entropy(model(x, assoc_values_per_layer=av).view(-1, enc.n_vocab),
                                     y.view(-1)).item()
                assoc_losses.append(vl)
            assoc_vl = sum(assoc_losses) / len(assoc_losses)
        assoc_ppl = math.exp(min(assoc_vl, 20))
        print(f"\n    Val PPL (no assoc):   {best_ppl:.1f}")
        print(f"    Val PPL (with assoc): {assoc_ppl:.1f}")
        print(f"    Assoc delta:          {assoc_ppl - best_ppl:+.1f}")

    return model, losses, ppls


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="2-layer screening model")
    parser.add_argument("--baseline", action="store_true", help="Also train online_mem baseline")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--write-interval", type=int, default=8,
                        help="Write to memory every N positions (controls memory growth)")
    parser.add_argument("--decay", type=float, default=0.99)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = Config()
    cfg.n_epochs = args.epochs
    cfg.decay = args.decay
    cfg.write_interval = args.write_interval

    if args.seq_len:
        cfg.seq_len = args.seq_len

    if args.bs:
        cfg.batch_size = args.bs

    if args.fast:
        cfg.n_layers = 2
        cfg.d_model = 128
        cfg.n_heads = 4
        print("  FAST MODE: 2 layers, d=128, 4 heads")
    else:
        if cfg.batch_size > 64:
            cfg.batch_size = 64

    print(f"{'='*60}")
    print(f"  Online Associative Memory LM")
    print(f"  layers={cfg.n_layers} d={cfg.d_model} heads={cfg.n_heads}")
    print(f"  seq={cfg.seq_len} bs={cfg.batch_size} epochs={cfg.n_epochs}")
    print(f"  decay={cfg.decay} write_interval={cfg.write_interval}")
    print(f"  SDR: N={cfg.sdr_n} P={cfg.sdr_p}")
    print(f"{'='*60}")

    kw = {"decay": cfg.decay}

    # Baseline: online_mem without associative memory
    if args.baseline:
        print("\n  Training online_mem baseline...")
        _, _, base_ppls = train(AssocMemoryAttention, cfg, device,
                                from_checkpoint=not args.from_scratch,
                                use_assoc=False, label="online_mem", **kw)
        print(f"  Online mem baseline: PPL {min(base_ppls):.1f}")

    # Associative memory version
    print("\n  Training with online associative memory...")
    _, _, assoc_ppls = train(AssocMemoryAttention, cfg, device,
                             from_checkpoint=not args.from_scratch,
                             use_assoc=True, label="assoc_mem", **kw)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"  Assoc mem: best PPL = {min(assoc_ppls):.1f}")
    if args.baseline:
        print(f"  Online mem baseline: PPL {min(base_ppls):.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
