"""
exp_triadic_seed.py — Triadic-seeded online Gram memory for language modeling

Two-tier memory architecture:
  Tier 1 (fast): Online Gram accumulation within each 128-token sequence
  Tier 2 (slow): Triadic associative memory stores Gram matrices across sequences

At end of each sequence, the accumulated Gram matrix is encoded as an SDR and
stored as (context_SDR, layer_SDR, gram_SDR) in triadic memory.

At start of each new sequence, the first tokens' embeddings form a context SDR,
triadic memory is queried, and the recalled Gram seeds the online memory.

This gives the model "structural episodic memory" — it starts each sequence
knowing what relational geometry similar past sequences had.

Usage:
  uv run python exp_triadic_seed.py                   # triadic-seeded online memory
  uv run python exp_triadic_seed.py --baseline         # compare with non-seeded
  uv run python exp_triadic_seed.py --fast             # 2-layer screening
  uv run python exp_triadic_seed.py --no-seed          # online_mem only (no triadic)
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
    patience = 3         # early stopping patience
    min_delta = 0.5      # min PPL improvement
    data_dir = Path("data/wikitext")
    cache_dir = Path("data/cache")
    # Triadic seeding
    sdr_n = 5000         # SDR dimensionality
    sdr_p = 50           # active bits per SDR
    ctx_warmup = 16      # tokens used for context SDR
    seed_start_epoch = 1 # enable seeding from this epoch (0-indexed)
    decay = 0.99         # Gram decay

# ── SDR encoding via random projection ───────────────────────────────────────

class RandomProjectionSDR:
    """Encode continuous vectors as SDRs via random projection."""

    def __init__(self, input_dim, n=5000, p=50, seed=42):
        self.n = n
        self.p = p
        rng = np.random.default_rng(seed)
        self.R = rng.standard_normal((input_dim, n)).astype(np.float32)
        norms = np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12
        self.R = self.R / norms
        self.R_pinv = np.linalg.pinv(self.R)  # (n, input_dim)

    def encode(self, vec):
        """Continuous vector → SDR (top-P indices)."""
        projected = vec.astype(np.float32) @ self.R
        indices = np.argsort(-projected)[:self.p]
        return np.sort(indices).astype(np.uint32)

    def decode(self, sdr_indices):
        """SDR indices → approximate continuous vector."""
        binary = np.zeros(self.n, dtype=np.float32)
        binary[sdr_indices] = 1.0
        return binary @ self.R_pinv


# ── Triadic Gram Store ───────────────────────────────────────────────────────

class TriadicGramStore:
    """Encapsulates triadic memory for storing/recalling Gram matrices."""

    def __init__(self, cfg):
        gram_dim = cfg.n_heads * 36  # 6×6 per head, flattened
        ctx_dim = cfg.d_model

        self.ctx_encoder = RandomProjectionSDR(ctx_dim, n=cfg.sdr_n, p=cfg.sdr_p, seed=100)
        self.gram_encoder = RandomProjectionSDR(gram_dim, n=cfg.sdr_n, p=cfg.sdr_p, seed=200)

        # Fixed layer SDRs (one per layer, deterministic)
        rng = np.random.default_rng(300)
        self.layer_sdrs = [
            np.sort(rng.permutation(cfg.sdr_n)[:cfg.sdr_p]).astype(np.uint32)
            for _ in range(cfg.n_layers)
        ]

        # Use CPU for triadic memory to avoid MPS<->numpy issues
        self.memory = TorchIndicatorMemory(N=cfg.sdr_n, P=cfg.sdr_p, device="cpu", exact=False)
        self.n_stored = 0
        self.cfg = cfg

    def store(self, ctx_vec_np, gram_flat_np, layer_idx):
        """Store (context, layer, gram) triple. All inputs numpy."""
        ctx_sdr = self.ctx_encoder.encode(ctx_vec_np)
        gram_sdr = self.gram_encoder.encode(gram_flat_np)
        self.memory.store(ctx_sdr, self.layer_sdrs[layer_idx], gram_sdr)
        self.n_stored += 1

    def recall(self, ctx_vec_np, layer_idx):
        """Recall gram given context and layer. Returns flat numpy or None."""
        if self.n_stored == 0:
            return None
        ctx_sdr = self.ctx_encoder.encode(ctx_vec_np)
        recalled_sdr = self.memory.query(ctx_sdr, self.layer_sdrs[layer_idx], None)
        if recalled_sdr is None or len(recalled_sdr) == 0:
            return None
        if isinstance(recalled_sdr, torch.Tensor):
            recalled_sdr = recalled_sdr.cpu().numpy()
        return self.gram_encoder.decode(recalled_sdr)


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


# ── Attention ────────────────────────────────────────────────────────────────

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gram_seed=None):
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
    """Online Gram memory with optional triadic seed injection."""

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
        # Learned scale for triadic seed contribution (init small)
        self.seed_scale = nn.Parameter(torch.full((n_heads,), 0.01))
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J', _J6)

        # Cache for write_lines (extracted after forward for Gram storage)
        self._cached_write_lines = None

    def forward(self, x, gram_seed=None):
        """
        x: (B, T, D)
        gram_seed: optional (B, H, 6, 6) — recalled Gram from triadic memory
        """
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

        # Write lines from bigrams
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)  # (B, T, H, 6)

        # Cache for Gram extraction
        self._cached_write_lines = write_lines.detach()

        # Read lines
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)  # (B, T, H, 6)

        # Online incidence
        J = self.J
        J_write = torch.einsum('bthi,ij->bthj', write_lines, J)
        read_h = read_lines.permute(0, 2, 1, 3)
        Jwrite_h = J_write.permute(0, 2, 1, 3)
        incidence = read_h @ Jwrite_h.transpose(-1, -2)
        incidence_sq = incidence ** 2
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=0).bool()
        incidence_sq = incidence_sq.masked_fill(causal, 0.0)

        # Temporal decay
        if self.decay != 1.0:
            positions = torch.arange(T, device=x.device, dtype=x.dtype)
            weights = self.decay ** (positions.unsqueeze(1) - positions.unsqueeze(0))
            weights = weights.tril(diagonal=-1)
            incidence_sq = incidence_sq * weights.unsqueeze(0).unsqueeze(0)

        mem_score = incidence_sq.sum(dim=-1)  # (B, H, T)

        # Triadic seed injection: add score from recalled Gram
        if gram_seed is not None:
            # gram_seed: (B, H, 6, 6)
            # read_lines: (B, T, H, 6) → read_h: (B, H, T, 6)
            J_read = torch.einsum('bthi,ij->bthj', read_lines, J)  # (B, T, H, 6)
            J_read_h = J_read.permute(0, 2, 1, 3)  # (B, H, T, 6)
            # Score: read @ Gram @ read^T per (b, h, t)
            seed_score = torch.einsum('bhti,bhij,bhtj->bht', J_read_h, gram_seed, J_read_h)
            # Add with learned per-head scale
            seed_s = self.seed_scale.reshape(1, H, 1)
            mem_score = mem_score + seed_s * seed_score

        # Gate and combine
        mem_val = self.mem_value(x)
        gate = torch.sigmoid(self.mem_gate(x))  # (B, T, H)
        scale = self.mem_scale.reshape(1, H, 1)
        mem_score_t = mem_score.permute(0, 2, 1)  # (B, T, H)
        gated = torch.sigmoid(mem_score_t * scale.permute(0, 2, 1)) * gate
        gated = gated.mean(dim=-1, keepdim=True)

        return self.out(seq_out + gated * mem_val)


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

    def forward(self, x, gram_seed=None):
        x = x + self.attn(self.ln1(x), gram_seed=gram_seed)
        x = x + self.ffn(self.ln2(x))
        return x


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

    def forward(self, idx, gram_seeds=None):
        """
        idx: (B, T) token indices
        gram_seeds: optional list of (B, H, 6, 6) per layer
        """
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for i, block in enumerate(self.blocks):
            seed = gram_seeds[i] if gram_seeds is not None else None
            x = block(x, gram_seed=seed)
        return self.head(self.ln_f(x))

    def get_context_embedding(self, idx, n_tokens=16):
        """Get mean-pooled embedding of first n_tokens (no grad needed)."""
        with torch.no_grad():
            T = min(n_tokens, idx.shape[1])
            emb = self.tok_emb(idx[:, :T]) + self.pos_emb(torch.arange(T, device=idx.device))
            return emb.mean(dim=1)  # (B, d_model)

    def get_cached_write_lines(self):
        """Get cached write lines from each OnlineMemoryAttention block."""
        lines = []
        for block in self.blocks:
            if hasattr(block.attn, '_cached_write_lines') and block.attn._cached_write_lines is not None:
                lines.append(block.attn._cached_write_lines)
            else:
                lines.append(None)
        return lines

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


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(step, cfg, total_steps):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


# ── Training with triadic seeding ────────────────────────────────────────────

def extract_and_store_grams(model, xb, ctx_vec, store, device):
    """Extract Gram matrices from cached write lines and store in triadic memory."""
    write_lines_per_layer = model.get_cached_write_lines()

    # Batch-average context vector (single representative per batch)
    ctx_avg = ctx_vec.mean(dim=0).cpu().numpy()  # (d_model,)

    for layer_idx, wl in enumerate(write_lines_per_layer):
        if wl is None:
            continue
        # wl: (B, T, H, 6) — compute Gram per head, average over batch
        # Gram = sum_t write_t outer write_t → (B, H, 6, 6)
        gram = torch.einsum('bthi,bthj->bhij', wl, wl)  # (B, H, 6, 6)
        # Batch average
        gram_avg = gram.mean(dim=0)  # (H, 6, 6)
        gram_flat = gram_avg.cpu().numpy().flatten()  # (H*36,)
        store.store(ctx_avg, gram_flat, layer_idx)


def recall_gram_seeds(model, xb, store, cfg, device):
    """Recall Gram seeds from triadic memory for the current batch."""
    if store.n_stored == 0:
        return None

    # Context from first K token embeddings (batch average)
    ctx_vec = model.get_context_embedding(xb, cfg.ctx_warmup)
    ctx_avg = ctx_vec.mean(dim=0).cpu().numpy()

    gram_seeds = []
    for layer_idx in range(cfg.n_layers):
        recalled = store.recall(ctx_avg, layer_idx)
        if recalled is not None:
            # Reshape from flat (H*36,) to (H, 6, 6)
            g = torch.tensor(recalled, dtype=torch.float32).reshape(cfg.n_heads, 6, 6)
            # Symmetrize for stability
            g = (g + g.transpose(-1, -2)) / 2
            # Broadcast to batch: (1, H, 6, 6) → (B, H, 6, 6) via broadcasting
            gram_seeds.append(g.unsqueeze(0).expand(xb.shape[0], -1, -1, -1).to(device))
        else:
            gram_seeds.append(None)

    # If all layers returned None, return None
    if all(g is None for g in gram_seeds):
        return None
    # Replace None with zeros
    for i in range(len(gram_seeds)):
        if gram_seeds[i] is None:
            gram_seeds[i] = torch.zeros(xb.shape[0], cfg.n_heads, 6, 6, device=device)

    return gram_seeds


def train_model(cfg, device, use_seed=True, from_checkpoint=True, label="triadic_seed"):
    """Train with optional triadic seeding."""
    enc = tiktoken.get_encoding("gpt2")
    train_data = load_tokens_cached("train", cfg)
    val_data = load_tokens_cached("val", cfg)

    model = LM(enc.n_vocab, cfg, OnlineMemoryAttention, decay=cfg.decay).to(device)
    print(f"\n  {label}: {model.count_params():,} params")

    if from_checkpoint:
        loaded = load_standard_checkpoint(model, device)
        if loaded:
            cfg_train = Config()
            cfg_train.n_epochs = cfg.n_epochs
            cfg_train.lr = 1e-4
            cfg_train.warmup_steps = cfg.warmup_steps
            cfg_train.patience = cfg.patience
            cfg_train.min_delta = cfg.min_delta
            cfg_train.seq_len = cfg.seq_len
            cfg_train.batch_size = cfg.batch_size
            cfg_train.data_dir = cfg.data_dir
            cfg_train.cache_dir = cfg.cache_dir
            print(f"  Fine-tuning with lr={cfg_train.lr}")
        else:
            cfg_train = cfg
    else:
        cfg_train = cfg

    store = TriadicGramStore(cfg) if use_seed else None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg_train.lr, weight_decay=0.01)
    n_batches = len(make_batches(train_data, cfg.seq_len, cfg.batch_size))
    total_steps = n_batches * cfg_train.n_epochs

    losses, ppls = [], []
    best_ppl = float('inf')
    patience_counter = 0
    global_step = 0

    for ep in range(cfg_train.n_epochs):
        t0 = time.time()
        model.train()
        batches = make_batches(train_data, cfg.seq_len, cfg.batch_size)
        ep_loss = 0.0
        seeded_batches = 0

        for xb, yb in batches:
            lr = get_lr(global_step, cfg_train, total_steps)
            for pg in opt.param_groups:
                pg['lr'] = lr

            xb, yb = xb.to(device), yb.to(device)

            # Recall gram seeds (if triadic seeding enabled and past warmup epoch)
            gram_seeds = None
            if use_seed and store is not None and ep >= cfg.seed_start_epoch:
                gram_seeds = recall_gram_seeds(model, xb, store, cfg, device)
                if gram_seeds is not None:
                    seeded_batches += 1

            # Forward with optional gram seeds
            logits = model(xb, gram_seeds=gram_seeds)
            loss = F.cross_entropy(logits.view(-1, enc.n_vocab), yb.view(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
            opt.step()
            ep_loss += loss.item()
            global_step += 1

            # Store Grams in triadic memory (every epoch, including epoch 0)
            if use_seed and store is not None:
                with torch.no_grad():
                    ctx_vec = model.get_context_embedding(xb, cfg.ctx_warmup)
                    extract_and_store_grams(model, xb, ctx_vec, store, device)

        avg = ep_loss / len(batches)
        losses.append(avg)

        # Validation
        model.eval()
        with torch.no_grad():
            vb = make_batches(val_data, cfg.seq_len, cfg.batch_size)
            # Validate without seeding (fair comparison)
            vl = sum(F.cross_entropy(model(x.to(device)).view(-1, enc.n_vocab),
                     y.to(device).view(-1)).item() for x, y in vb) / len(vb)
        ppl = math.exp(min(vl, 20))
        ppls.append(ppl)

        elapsed = time.time() - t0
        improved = "★" if ppl < best_ppl - cfg_train.min_delta else ""
        seed_info = f" seeded={seeded_batches}/{len(batches)}" if use_seed else ""
        stored_info = f" stored={store.n_stored}" if store else ""
        print(f"    ep {ep+1:2d}: loss={avg:.3f}  ppl={ppl:.1f}  ({elapsed:.1f}s){seed_info}{stored_info} {improved}")

        if ppl < best_ppl - cfg_train.min_delta:
            best_ppl = ppl
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg_train.patience:
                print(f"    Early stopping (no improvement for {cfg_train.patience} epochs)")
                break

    # Also evaluate WITH seeding on val set
    if use_seed and store is not None and store.n_stored > 0:
        model.eval()
        with torch.no_grad():
            vb = make_batches(val_data, cfg.seq_len, cfg.batch_size)
            seeded_losses = []
            for x, y in vb:
                x, y = x.to(device), y.to(device)
                gs = recall_gram_seeds(model, x, store, cfg, device)
                vl = F.cross_entropy(model(x, gram_seeds=gs).view(-1, enc.n_vocab),
                                     y.view(-1)).item()
                seeded_losses.append(vl)
            seeded_vl = sum(seeded_losses) / len(seeded_losses)
        seeded_ppl = math.exp(min(seeded_vl, 20))
        print(f"\n    Val PPL (no seed):   {best_ppl:.1f}")
        print(f"    Val PPL (with seed): {seeded_ppl:.1f}")
        print(f"    Seed delta:          {seeded_ppl - best_ppl:+.1f}")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "type": label,
                "losses": losses, "ppls": ppls, "vocab": enc.n_vocab},
               f"checkpoints/lm_{label}.pt")

    return model, losses, ppls, store


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Also train non-seeded baseline")
    parser.add_argument("--no-seed", action="store_true", help="Train online_mem without seeding")
    parser.add_argument("--fast", action="store_true", help="2-layer screening model")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--seed-start", type=int, default=1, help="Epoch to start seeding (0-indexed)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = Config()
    cfg.n_epochs = args.epochs
    cfg.decay = args.decay
    cfg.seed_start_epoch = args.seed_start

    if args.fast:
        cfg.n_layers = 2
        cfg.d_model = 128
        cfg.n_heads = 4
        print("  FAST MODE: 2 layers, d=128, 4 heads")

    print(f"{'='*60}")
    print(f"  Triadic-Seeded Online Gram Memory LM")
    print(f"  layers={cfg.n_layers} d={cfg.d_model} heads={cfg.n_heads}")
    print(f"  seq={cfg.seq_len} bs={cfg.batch_size} epochs={cfg.n_epochs}")
    print(f"  decay={cfg.decay} seed_start_epoch={cfg.seed_start_epoch}")
    print(f"  SDR: N={cfg.sdr_n} P={cfg.sdr_p}")
    print(f"{'='*60}")

    if args.no_seed or args.baseline:
        print("\n  Training online_mem (no triadic seeding)...")
        _, _, base_ppls, _ = train_model(cfg, device, use_seed=False,
                                          from_checkpoint=not args.from_scratch,
                                          label="online_mem")
        print(f"  Online mem baseline: PPL {min(base_ppls):.1f}")

    if not args.no_seed:
        print("\n  Training triadic-seeded online_mem...")
        _, _, seed_ppls, store = train_model(cfg, device, use_seed=True,
                                              from_checkpoint=not args.from_scratch,
                                              label="triadic_seed")
        print(f"\n  Triadic-seeded: PPL {min(seed_ppls):.1f}")
        if store:
            print(f"  Triadic memory: {store.n_stored} triples stored")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Known baselines:")
    print(f"    Standard attention:     PPL ~209")
    print(f"    Online Gram memory:     PPL ~206")
    if args.no_seed or args.baseline:
        print(f"  This run:")
        print(f"    Online mem (no seed):   PPL {min(base_ppls):.1f}")
    if not args.no_seed:
        print(f"    Triadic-seeded:         PPL {min(seed_ppls):.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
