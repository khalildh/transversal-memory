"""
exp_assoc_mem.py — Triadic associative memory for inference-time LM boosting

Cognition-inspired: encode experiences during training, recall at decision time.

Architecture:
  1. Train online_mem model normally (Plücker geometry within-sequence)
  2. Build memory: scan training data, encode write lines as SDRs,
     store (context_SDR, layer_SDR, token_SDR) triples in TorchTriadicMemory
  3. Eval: query triadic memory → recalled token SDR → decode via codebook
     → interpolate with model logits via λ

Uses TorchTriadicMemory (sparse COO backend) from TDGA — the real triadic
memory with N×N×N cube semantics via sparse matrices. Query reinforcement
comes from P² address pairs all pointing to the same z-bits.

Usage:
  uv run python exp_assoc_mem.py --fast --from-scratch          # full pipeline
  uv run python exp_assoc_mem.py --fast --eval-only             # skip training
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
from tdga.torch_indicator_memory import TorchIndicatorMemory

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
    # Memory
    sdr_n = 1000         # SDR dimensionality (cube side length)
    sdr_p = 10           # active bits per SDR
    decay = 0.99         # Gram decay
    store_interval = 8   # store every N positions

# ── Plücker primitives ──────────────────────────────────────────────────────

_PAIRS4 = list(combinations(range(4), 2))

_J6 = torch.tensor([
    [0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
    [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],
], dtype=torch.float32)

def exterior(p1, p2):
    parts = [p1[...,i]*p2[...,j] - p1[...,j]*p2[...,i] for i,j in _PAIRS4]
    L = torch.stack(parts, dim=-1)
    return L / L.norm(dim=-1, keepdim=True).clamp(min=1e-12)


# ── SDR encoding via random projection ────────────────────────────────────────

class RandomProjectionEncoder:
    """Encode continuous vectors as sparse SDR index arrays via random projection + top-k."""

    def __init__(self, input_dim, n=1000, p=10, seed=42):
        self.n = n
        self.p = p
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((input_dim, n)).astype(np.float32)
        norms = np.linalg.norm(R, axis=0, keepdims=True) + 1e-12
        self._R = R / norms

    def encode(self, vec):
        """Single numpy vector → sorted uint32 SDR indices."""
        projected = vec.astype(np.float32) @ self._R
        indices = np.argsort(-projected)[:self.p]
        return np.sort(indices).astype(np.uint32)

    def encode_batch(self, vecs):
        """(B, dim) numpy array → list of sorted uint32 SDR index arrays."""
        projected = vecs.astype(np.float32) @ self._R  # (B, N)
        indices = np.argsort(-projected, axis=-1)[:, :self.p]  # (B, P)
        return [np.sort(row).astype(np.uint32) for row in indices]


# ── Token SDR codebook ───────────────────────────────────────────────────────

class TokenCodebook:
    """Deterministic random SDR for each token in vocab.

    Each token gets a fixed random SDR. At decode time, query the triadic
    memory → get recalled SDR sums → compare with all token SDRs → distribution.
    """

    def __init__(self, vocab_size, sdr_n, sdr_p, seed=12345):
        self.vocab_size = vocab_size
        self.sdr_n = sdr_n
        self.sdr_p = sdr_p
        rng = np.random.default_rng(seed)
        # Pre-generate all token SDRs as sparse index arrays
        self.token_sdrs = []
        for _ in range(vocab_size):
            idx = np.sort(rng.choice(sdr_n, size=sdr_p, replace=False)).astype(np.uint32)
            self.token_sdrs.append(idx)
        # Dense codebook for batch decoding: (V, N) float32
        self._dense = np.zeros((vocab_size, sdr_n), dtype=np.float32)
        for i, sdr in enumerate(self.token_sdrs):
            self._dense[i, sdr] = 1.0
        self._dense_torch = None  # lazy device init

    def to_device(self, device):
        self._dense_torch = torch.from_numpy(self._dense).to(device)
        return self

    def get_sdr(self, token_id):
        """Get SDR for a single token."""
        return self.token_sdrs[token_id]

    def decode_sums(self, sums_np):
        """(N,) numpy sums → (V,) overlap scores."""
        return self._dense @ sums_np  # (V, N) @ (N,) → (V,)

    def decode_sums_batch_torch(self, sums_torch):
        """(B, N) tensor → (B, V) overlap scores on device."""
        return sums_torch @ self._dense_torch.T


# ── Online Memory Attention (from exp_fast.py) ──────────────────────────────

class OnlineMemoryAttention(nn.Module):
    """Standard Q·K attention + causal Gram accumulation as scalar gate."""
    def __init__(self, d_model, n_heads, dropout=0.1, decay=0.99, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.decay = decay

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

        self._cached_write_lines = None

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
        self._cached_write_lines = write_lines.detach()

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

        if self.decay != 1.0:
            positions = torch.arange(T, device=x.device, dtype=x.dtype)
            weights = self.decay ** (positions.unsqueeze(1) - positions.unsqueeze(0))
            weights = weights.tril(diagonal=-1)
            incidence_sq = incidence_sq * weights.unsqueeze(0).unsqueeze(0)

        mem_score = incidence_sq.sum(dim=-1)
        mem_val = self.mem_value(x)
        gate = torch.sigmoid(self.mem_gate(x))
        scale = self.mem_scale.reshape(1, H, 1)
        mem_score_t = mem_score.permute(0, 2, 1)
        gated = torch.sigmoid(mem_score_t * scale.permute(0, 2, 1)) * gate
        gated = gated.mean(dim=-1, keepdim=True)

        return self.out(seq_out + gated * mem_val)


# ── Model ────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__()
        self.attn = OnlineMemoryAttention(d_model, n_heads, dropout=dropout, **kw)
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
    def __init__(self, vocab_size, cfg, **kw):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, cfg.dropout, **kw)
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

    def get_all_write_lines(self):
        """Get cached write lines from all layers. List of (B, T, H, 6)."""
        return [block.attn._cached_write_lines for block in self.blocks]

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Triadic SDR Memory ──────────────────────────────────────────────────────

class TriadicSDRMemory:
    """Inference-time associative memory using TorchIndicatorMemory.

    Stores (context_SDR, layer_SDR, token_SDR) triples.
    TorchIndicatorMemory keeps triples separate as indicator rows —
    no interference ceiling, better recall at scale, GPU-friendly matmul queries.

    Build: scan training data → encode write lines as context SDRs →
           store (context, layer, token) triples
    Query: encode current write lines → indicator query → token SDR sums →
           decode via codebook overlap → token distribution
    """

    def __init__(self, key_dim, cfg, vocab_size):
        self.cfg = cfg
        self.key_dim = key_dim
        self.vocab_size = vocab_size
        self.encoder = RandomProjectionEncoder(
            key_dim, n=cfg.sdr_n, p=cfg.sdr_p
        )
        self.codebook = TokenCodebook(vocab_size, cfg.sdr_n, cfg.sdr_p)

        # Layer SDRs: fixed random SDR per layer
        rng = np.random.default_rng(99999)
        self.layer_sdrs = []
        for _ in range(cfg.n_layers):
            idx = np.sort(rng.choice(cfg.sdr_n, size=cfg.sdr_p, replace=False)).astype(np.uint32)
            self.layer_sdrs.append(idx)

        # One TorchIndicatorMemory per layer (CPU for build, can move to device)
        self.memories = [
            TorchIndicatorMemory(cfg.sdr_n, cfg.sdr_p, device="cpu")
            for _ in range(cfg.n_layers)
        ]
        self.n_stored = 0

    def build(self, model, data, cfg, device):
        """Scan training data, store (context_SDR, layer_SDR, token_SDR) triples."""
        model.eval()
        batches = make_batches(data, cfg.seq_len, cfg.batch_size)
        interval = cfg.store_interval
        n_total = 0

        print(f"    Building triadic memory (N={cfg.sdr_n}, P={cfg.sdr_p}, interval={interval})...")
        t0 = time.time()

        with torch.no_grad():
            for bi, (xb, yb) in enumerate(batches):
                xb_dev = xb.to(device)
                _ = model(xb_dev)

                B, T = xb.shape
                all_wl = model.get_all_write_lines()

                # Subsample positions
                positions = list(range(0, T, interval))

                for layer_idx, wl in enumerate(all_wl):
                    # wl: (B, T, H, 6) — get subsampled, flatten heads
                    wl_sub = wl[:, positions, :, :].cpu().numpy()  # (B, n_pos, H, 6)
                    B_actual, n_pos = wl_sub.shape[0], wl_sub.shape[1]
                    wl_flat = wl_sub.reshape(B_actual * n_pos, -1)  # (B*n_pos, H*6)

                    # Encode context as SDRs
                    context_sdrs = self.encoder.encode_batch(wl_flat)

                    # Get target tokens at these positions
                    tok_sub = yb[:, positions].reshape(-1)  # (B*n_pos,)

                    layer_sdr = self.layer_sdrs[layer_idx]

                    for i in range(len(context_sdrs)):
                        token_sdr = self.codebook.get_sdr(tok_sub[i].item())
                        self.memories[layer_idx].store(
                            context_sdrs[i], layer_sdr, token_sdr
                        )

                n_total += B * len(positions)

                if (bi + 1) % 10 == 0:
                    print(f"      batch {bi+1}/{len(batches)}, triples={n_total * cfg.n_layers}")

        # Flush pending stores
        for mem in self.memories:
            mem._rebuild()

        self.n_stored = n_total
        elapsed = time.time() - t0
        print(f"    Memory built: {n_total:,} entries × {cfg.n_layers} layers in {elapsed:.1f}s")

        # Report memory usage (indicator matrices: 3 × (T, N) per layer)
        total_bytes = 0
        for mem in self.memories:
            for attr in ['_x_mat', '_y_mat', '_z_mat']:
                mat = getattr(mem, attr, None)
                if mat is not None:
                    total_bytes += mat.nelement() * mat.element_size()
        print(f"    Indicator storage: {total_bytes / 1e6:.1f} MB")

    def query_token_distribution(self, model, xb_dev, verbose=False):
        """Query indicator memory for token predictions — batched via matmul.

        Per-sequence: mean-pool write lines → encode SDR → batch indicator query →
        recalled token SDR sums → decode via codebook overlap → distribution.

        Uses direct matmul against indicator matrices instead of per-sequence
        .query() calls. Same math, ~100x faster.

        Returns: (B, V) numpy array of overlap scores.
        """
        B = xb_dev.shape[0]
        all_wl = model.get_all_write_lines()
        N = self.cfg.sdr_n
        P = self.cfg.sdr_p

        # Accumulate token SDR sums across layers
        token_sums = np.zeros((B, N), dtype=np.float32)

        for layer_idx, wl in enumerate(all_wl):
            mem = self.memories[layer_idx]
            mem._rebuild()  # ensure matrices are materialized

            if mem._x_mat is None or mem._x_mat.shape[0] == 0:
                continue

            # Mean-pool write lines per sequence: (B, T, H, 6) → (B, H*6)
            wl_pooled = wl.mean(dim=1).reshape(B, -1).cpu().numpy()

            # Encode all B sequences as SDRs → build dense query matrix
            context_sdrs = self.encoder.encode_batch(wl_pooled)  # list of (P,) arrays
            # Build (B, N) dense binary query vectors
            x_query = np.zeros((B, N), dtype=np.float32)
            for i, sdr in enumerate(context_sdrs):
                x_query[i, sdr] = 1.0

            # Build (N,) one-hot for layer SDR
            y_onehot = np.zeros(N, dtype=np.float32)
            y_onehot[self.layer_sdrs[layer_idx]] = 1.0

            # Batched indicator query:
            #   overlap_x[b, t] = x_query[b] · x_ind[t]  → (B, T_stored)
            #   overlap_y[t]    = y_ind[t] · y_onehot     → (T_stored,)
            #   weights[b, t]   = overlap_x[b,t] * overlap_y[t]  → (B, T_stored)
            #   sums[b]         = weights[b] @ z_ind       → (B, N)
            x_ind = mem._x_mat.cpu().numpy()  # (T_stored, N)
            y_ind = mem._y_mat.cpu().numpy()
            z_ind = mem._z_mat.cpu().numpy()

            overlap_x = x_query @ x_ind.T       # (B, T_stored)
            overlap_y = y_ind @ y_onehot         # (T_stored,)
            weights = overlap_x * overlap_y      # (B, T_stored) — conjunction
            sums = weights @ z_ind               # (B, N)

            token_sums += sums

        if verbose:
            active = (token_sums > 0).sum(axis=1)
            print(f"      Token sums: mean active bits={active.mean():.1f} "
                  f"max_sum={token_sums.max():.1f}")

        # Decode: sums @ codebook → (B, V) overlap scores
        scores = token_sums @ self.codebook._dense.T
        return scores


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


# ── Training (standard — no memory involvement) ─────────────────────────────

def train_model(cfg, device, label="online_mem"):
    enc = tiktoken.get_encoding("gpt2")
    train_data = load_tokens_cached("train", cfg)
    if hasattr(cfg, 'data_frac') and cfg.data_frac < 1.0:
        n = int(len(train_data) * cfg.data_frac)
        train_data = train_data[:n]
        print(f"    Using {cfg.data_frac*100:.0f}% of training data ({n:,} tokens)")
    val_data = load_tokens_cached("val", cfg)

    model = LM(enc.n_vocab, cfg, decay=cfg.decay).to(device)
    print(f"\n  {label}: {model.count_params():,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    n_batches = len(make_batches(train_data, cfg.seq_len, cfg.batch_size))
    total_steps = n_batches * cfg.n_epochs

    best_ppl = float('inf')
    best_state = None
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
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss += loss.item()
            global_step += 1

        avg = ep_loss / len(batches)

        model.eval()
        with torch.no_grad():
            vb = make_batches(val_data, cfg.seq_len, cfg.batch_size)
            vl = sum(F.cross_entropy(model(x.to(device)).view(-1, enc.n_vocab),
                     y.to(device).view(-1)).item() for x, y in vb) / len(vb)
        ppl = math.exp(min(vl, 20))

        elapsed = time.time() - t0
        improved = "★" if ppl < best_ppl - cfg.min_delta else ""
        print(f"    ep {ep+1:2d}: loss={avg:.3f}  ppl={ppl:.1f}  ({elapsed:.1f}s) {improved}")

        if ppl < best_ppl - cfg.min_delta:
            best_ppl = ppl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"    Early stopping (no improvement for {cfg.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = Path("checkpoints/lm_assoc_mem.pt")
    torch.save({
        "model": model.state_dict(),
        "best_ppl": best_ppl,
        "vocab": enc.n_vocab,
    }, ckpt_path)
    print(f"    Saved checkpoint: {ckpt_path} (PPL {best_ppl:.1f})")

    return model, best_ppl


# ── Evaluation with triadic memory ──────────────────────────────────────────

def eval_with_memory(model, memory, cfg, device, lam=0.1):
    """Evaluate model with triadic memory interpolation.

    Per sequence: triadic query → token distribution (B, V)
    Broadcast to all positions, interpolate with model probs.
    """
    enc = tiktoken.get_encoding("gpt2")
    val_data = load_tokens_cached("val", cfg)
    batches = make_batches(val_data, cfg.seq_len, cfg.batch_size)

    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for xb, yb in batches:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # (B, T, V)
            B, T, V = logits.shape

            if lam == 0.0:
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1))
            else:
                # Triadic memory query → (B, V) overlap scores
                mem_scores = memory.query_token_distribution(
                    model, xb, verbose=(n_batches == 0)
                )  # numpy (B, V)
                mem_scores_t = torch.from_numpy(mem_scores).float().to(device)

                # Normalize to probabilities
                mem_probs = F.softmax(mem_scores_t, dim=-1)  # (B, V)

                # Broadcast to all positions: (B, 1, V) → (B, T, V)
                mem_probs = mem_probs.unsqueeze(1).expand(B, T, V)
                model_probs = F.softmax(logits, dim=-1)

                combined = (1 - lam) * model_probs + lam * mem_probs
                combined_log = torch.log(combined + 1e-10)

                loss = F.nll_loss(combined_log.view(-1, V), yb.view(-1))

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    ppl = math.exp(min(avg_loss, 20))
    return ppl


def sweep_lambda(model, memory, cfg, device):
    lambdas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    print(f"\n    Sweeping λ ({len(lambdas)} values)...")

    results = []
    for lam in lambdas:
        t0 = time.time()
        ppl = eval_with_memory(model, memory, cfg, device, lam=lam)
        elapsed = time.time() - t0
        results.append((lam, ppl))
        print(f"      λ={lam:.2f}  PPL={ppl:.1f}  ({elapsed:.1f}s)")

    best_lam, best_ppl = min(results, key=lambda x: x[1])
    print(f"\n    Best: λ={best_lam:.2f}  PPL={best_ppl:.1f}")
    return best_lam, best_ppl, results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="2-layer screening model")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--sdr-n", type=int, default=1000, help="SDR dimensionality")
    parser.add_argument("--sdr-p", type=int, default=10, help="SDR active bits")
    parser.add_argument("--store-interval", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--data-frac", type=float, default=1.0,
                        help="Fraction of training data to use (0.1 = 10%%)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = Config()
    cfg.n_epochs = args.epochs
    cfg.decay = args.decay
    cfg.sdr_n = args.sdr_n
    cfg.sdr_p = args.sdr_p
    cfg.store_interval = args.store_interval
    cfg.data_frac = args.data_frac

    if args.seq_len:
        cfg.seq_len = args.seq_len

    if args.fast:
        cfg.n_layers = 2
        cfg.d_model = 128
        cfg.n_heads = 4
        print("  FAST MODE: 2 layers, d=128, 4 heads")

    key_dim = cfg.n_heads * 6

    print(f"{'='*60}")
    print(f"  Triadic Associative Memory LM (inference-time)")
    print(f"  layers={cfg.n_layers} d={cfg.d_model} heads={cfg.n_heads}")
    print(f"  seq={cfg.seq_len} bs={cfg.batch_size} epochs={cfg.n_epochs}")
    print(f"  SDR: N={cfg.sdr_n} P={cfg.sdr_p} key_dim={key_dim}")
    print(f"  Memory: interval={cfg.store_interval} data_frac={cfg.data_frac}")
    print(f"{'='*60}")

    # ── Step 1: Train model (or load checkpoint) ──
    ckpt_path = Path("checkpoints/lm_assoc_mem.pt")
    if args.eval_only and ckpt_path.exists():
        print("\n  Step 1: Loading checkpoint...")
        enc = tiktoken.get_encoding("gpt2")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = LM(ckpt["vocab"], cfg, decay=cfg.decay).to(device)
        model.load_state_dict(ckpt["model"])
        base_ppl = ckpt["best_ppl"]
        print(f"    Loaded {ckpt_path} (PPL {base_ppl:.1f})")
    else:
        print("\n  Step 1: Train online_mem model...")
        model, base_ppl = train_model(cfg, device)
    print(f"\n  Baseline PPL (no memory): {base_ppl:.1f}")

    # ── Step 2: Build triadic memory ──
    print(f"\n  Step 2: Build triadic memory from training data...")
    enc = tiktoken.get_encoding("gpt2")
    train_data = load_tokens_cached("train", cfg)
    if cfg.data_frac < 1.0:
        train_data = train_data[:int(len(train_data) * cfg.data_frac)]
    memory = TriadicSDRMemory(key_dim, cfg, enc.n_vocab)
    memory.codebook.to_device(device)
    memory.build(model, train_data, cfg, device)

    # ── Step 3: Evaluate with triadic memory interpolation ──
    print(f"\n  Step 3: Evaluate with triadic memory interpolation...")
    best_lam, best_ppl, results = sweep_lambda(model, memory, cfg, device)

    # ── Results ──
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"  Baseline (no memory):  PPL {base_ppl:.1f}")
    print(f"  With triadic memory:   PPL {best_ppl:.1f} (λ={best_lam:.2f})")
    delta = best_ppl - base_ppl
    pct = delta / base_ppl * 100
    print(f"  Delta:                 {delta:+.1f} ({pct:+.1f}%)")
    print(f"  Memory entries:        {memory.n_stored:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
