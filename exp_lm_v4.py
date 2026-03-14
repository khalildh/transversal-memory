"""
exp_lm_v4.py — Hybrid standard + Plücker attention for language modeling

Trains two small transformers on WikiText-2:
  1. Standard dot-product attention (baseline)
  2. Hybrid attention: standard Q·K logits + Plücker incidence bias

The hypothesis: geometric line-incidence information is COMPLEMENTARY
to standard attention. Rather than replacing Q·K entirely (which failed
with PPL 2151), we ADD a Plücker bias term:

  attn_logits = standard_qk_logits + plucker_bias * learned_scale

Each head learns:
  - Standard W_q, W_k projections for dot-product logits
  - Two P³ projections W1_p, W2_p (4 dims each) for Plücker lines
  - A learnable scalar scale (init 0.1) controlling bias strength

The Plücker inner product p @ J6 @ q measures whether two lines in P³
meet (value ≈ 0) or are skew (value ≠ 0). This signed scalar becomes
the additive bias to standard attention.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tiktoken
import pyarrow.parquet as pq
import time
import math
from pathlib import Path
from itertools import combinations

# ── Config ───────────────────────────────────────────────────────────────────

class Config:
    # Model
    d_model = 192
    n_heads = 6
    n_layers = 4
    dropout = 0.1
    seq_len = 128
    # Training
    batch_size = 64
    n_epochs = 10
    lr = 3e-4
    grad_clip = 1.0
    # Data
    data_dir = Path("data/wikitext")


# ── Plücker primitives ──────────────────────────────────────────────────────

_PAIRS_P3 = list(combinations(range(4), 2))
_PAIR_I = [i for i, j in _PAIRS_P3]  # [0,0,0,1,1,2]
_PAIR_J = [j for i, j in _PAIRS_P3]  # [1,2,3,2,3,3]

# Hodge dual matrix for plucker_inner: p · J6 · q
_J6_CONST = torch.tensor([
    [0, 0, 0,  0,  0, 1],
    [0, 0, 0,  0, -1, 0],
    [0, 0, 0,  1,  0, 0],
    [0, 0, 1,  0,  0, 0],
    [0,-1, 0,  0,  0, 0],
    [1, 0, 0,  0,  0, 0],
], dtype=torch.float32)


def plucker_lines(p1, p2):
    """
    Batched exterior product → Plücker 6-vectors.
    p1, p2: (..., 4)
    Returns: (..., 6) normalized
    """
    parts = []
    for i, j in _PAIRS_P3:
        parts.append(p1[..., i] * p2[..., j] - p1[..., j] * p2[..., i])
    lines = torch.stack(parts, dim=-1)
    return lines / lines.norm(dim=-1, keepdim=True).clamp(min=1e-12)


# ── Attention layers ─────────────────────────────────────────────────────────

class StandardAttention(nn.Module):
    """Multi-head dot-product attention."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, dh)

        attn = (q @ k.transpose(-1, -2)) * self.scale  # (B, H, T, T)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class HybridAttention(nn.Module):
    """
    Multi-head hybrid attention: standard Q·K + Plücker incidence bias.

    Keeps full standard dot-product attention but adds a geometric bias
    from Plücker line incidence. Each head has:
      - Standard Q, K, V projections (same as dot-product)
      - Two P³ projections W1_p, W2_p mapping tokens to lines
      - A learnable scale parameter (init 0.1) for the Plücker bias

    Combined: attn_logits = (Q·K / sqrt(d_head)) + plucker_inner * scale
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Standard Q, K, V projections
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Plücker line projections: token → two points in P³, per head
        self.W1_p = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_p = nn.Linear(d_model, 4 * n_heads, bias=False)

        # Learnable scale for Plücker bias, one per head, init to 0.1
        self.plucker_scale = nn.Parameter(torch.full((n_heads,), 0.1))

        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.register_buffer('J6', _J6_CONST)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # ── Standard Q·K logits ──
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, dh)
        std_logits = (q @ k.transpose(-1, -2)) * self.scale  # (B, H, T, T)

        # ── Plücker incidence bias ──
        p1 = self.W1_p(x).reshape(B, T, H, 4)  # (B, T, H, 4)
        p2 = self.W2_p(x).reshape(B, T, H, 4)

        # Plücker lines: (B, T, H, 6)
        lines = plucker_lines(p1, p2)

        # Incidence matrix via Hodge dual: (B, H, T, T)
        lines_h = lines.permute(0, 2, 1, 3)         # (B, H, T, 6)
        Jlines = lines_h @ self.J6.T                 # (B, H, T, 6)
        plucker_bias = lines_h @ Jlines.transpose(-1, -2)  # (B, H, T, T)

        # Scale by learnable per-head scalar
        scale = self.plucker_scale.reshape(1, H, 1, 1)
        attn_logits = std_logits + plucker_bias * scale

        # Causal mask + softmax
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_logits = attn_logits.masked_fill(mask, float('-inf'))
        attn = self.drop(F.softmax(attn_logits, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ── Transformer blocks ──────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, attn_type="standard", dropout=0.1):
        super().__init__()
        AttnClass = HybridAttention if attn_type == "plucker" else StandardAttention
        self.attn = AttnClass(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SmallLM(nn.Module):
    def __init__(self, vocab_size, cfg, attn_type="standard"):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.Sequential(*[
            TransformerBlock(cfg.d_model, cfg.n_heads, attn_type, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data loading ─────────────────────────────────────────────────────────────

def load_wikitext(split, data_dir, enc):
    """Load and tokenize a WikiText-2 split."""
    fname = {"train": "train.parquet", "val": "val.parquet", "test": "test.parquet"}
    table = pq.read_table(data_dir / fname[split])
    texts = table['text'].to_pylist()
    # Join all text, filter empty lines
    full_text = "\n".join(t for t in texts if t.strip())
    tokens = enc.encode(full_text, allowed_special=set())
    return torch.tensor(tokens, dtype=torch.long)


def make_batches(data, seq_len, batch_size):
    """Create (input, target) batches from flat token array."""
    n_tokens = len(data) - 1
    n_seqs = n_tokens // seq_len
    n_seqs = (n_seqs // batch_size) * batch_size  # trim to full batches

    data = data[:n_seqs * seq_len + 1]
    x = data[:-1].reshape(n_seqs, seq_len)
    y = data[1:].reshape(n_seqs, seq_len)

    # Shuffle and batch
    perm = torch.randperm(n_seqs)
    batches = []
    for i in range(0, n_seqs, batch_size):
        idx = perm[i:i+batch_size]
        if len(idx) == batch_size:
            batches.append((x[idx], y[idx]))
    return batches


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(attn_type, cfg, device):
    """Train a model and return (model, train_losses, val_ppls)."""
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    print(f"\nLoading data...", flush=True)
    train_data = load_wikitext("train", cfg.data_dir, enc)
    val_data = load_wikitext("val", cfg.data_dir, enc)
    print(f"  Train: {len(train_data):,} tokens", flush=True)
    print(f"  Val:   {len(val_data):,} tokens", flush=True)

    model = SmallLM(vocab_size, cfg, attn_type).to(device)
    n_params = model.count_params()
    print(f"  Model ({attn_type}): {n_params:,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    train_losses = []
    val_ppls = []

    for epoch in range(cfg.n_epochs):
        t0 = time.time()

        # Train
        model.train()
        batches = make_batches(train_data, cfg.seq_len, cfg.batch_size)
        epoch_loss = 0.0

        for step, (xb, yb) in enumerate(batches):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

            # Sync MPS every 10 steps to prevent command queue buildup
            if device.type == "mps" and step % 10 == 0:
                torch.mps.synchronize()

            if step % 50 == 0:
                print(f"    step {step}/{len(batches)} loss={loss.item():.3f}", flush=True)

        avg_loss = epoch_loss / len(batches)
        train_losses.append(avg_loss)

        # Validate
        model.eval()
        val_batches = make_batches(val_data, cfg.seq_len, cfg.batch_size)
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_batches:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += F.cross_entropy(
                    logits.view(-1, logits.size(-1)), yb.view(-1)).item()
        val_loss /= len(val_batches)
        val_ppl = math.exp(min(val_loss, 20))  # cap to avoid overflow
        val_ppls.append(val_ppl)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{cfg.n_epochs}: "
              f"train_loss={avg_loss:.3f}  val_ppl={val_ppl:.1f}  ({elapsed:.1f}s)", flush=True)

    # Checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_name = "lm_v4_standard" if attn_type == "standard" else "lm_v4_plucker"
    ckpt_path = ckpt_dir / f"{ckpt_name}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "attn_type": attn_type,
        "cfg": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
        "train_losses": train_losses,
        "val_ppls": val_ppls,
        "vocab_size": vocab_size,
    }, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    return model, train_losses, val_ppls


def load_model(ckpt_name, attn_type, device):
    """Load a checkpointed model and its training history."""
    ckpt_path = Path("checkpoints") / f"{ckpt_name}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config()
    for k, v in ckpt["cfg"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = SmallLM(ckpt["vocab_size"], cfg, attn_type).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["train_losses"], ckpt["val_ppls"]


def generate(model, enc, prompt, max_tokens=60, temperature=0.8, top_k=40):
    """Autoregressive generation from a prompt."""
    model.eval()
    device = next(model.parameters()).device
    seq_len = model.cfg.seq_len
    tokens = enc.encode(prompt, allowed_special=set())
    idx = torch.tensor([tokens[-seq_len:]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -seq_len:]
            logits = model(idx_cond)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)

    return enc.decode(idx[0].tolist())


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()

    # Train or load standard attention
    std_ckpt = Path("checkpoints/lm_v4_standard.pt")
    if std_ckpt.exists():
        print("=" * 60)
        print("MODEL 1: Standard dot-product attention (loading checkpoint)")
        print("=" * 60)
        # Load on CPU to save GPU memory for hybrid training
        std_model, std_losses, std_ppls = load_model("lm_v4_standard", "standard", "cpu")
        print(f"  Loaded from {std_ckpt}")
        print(f"  Params: {std_model.count_params():,}")
        print(f"  Best val PPL: {min(std_ppls):.1f}", flush=True)
    else:
        print("=" * 60)
        print("MODEL 1: Standard dot-product attention (baseline)")
        print("=" * 60)
        std_model, std_losses, std_ppls = train_model("standard", cfg, device)
        # Move to CPU to free GPU for hybrid training
        std_model = std_model.cpu()

    # Train or load hybrid attention
    plk_ckpt = Path("checkpoints/lm_v4_plucker.pt")
    if plk_ckpt.exists():
        print("\n" + "=" * 60)
        print("MODEL 2: Hybrid standard + Plücker attention (loading checkpoint)")
        print("=" * 60)
        plk_model, plk_losses, plk_ppls = load_model("lm_v4_plucker", "plucker", "cpu")
        print(f"  Loaded from {plk_ckpt}")
        print(f"  Params: {plk_model.count_params():,}")
        print(f"  Best val PPL: {min(plk_ppls):.1f}", flush=True)
    else:
        print("\n" + "=" * 60)
        print("MODEL 2: Hybrid standard + Plücker attention")
        print("=" * 60)
        plk_model, plk_losses, plk_ppls = train_model("plucker", cfg, device)
        # Move to CPU for generation
        plk_model = plk_model.cpu()

    # ── Comparison table ──
    print("\n" + "=" * 60)
    print("COMPARISON: Standard vs Hybrid (Standard + Plücker bias)")
    print("=" * 60)
    print(f"  {'':20s} {'Standard':>10s} {'Hybrid':>10s}")
    print(f"  {'-'*42}")
    print(f"  {'Params':20s} {std_model.count_params():>10,d} {plk_model.count_params():>10,d}")
    print(f"  {'Final train loss':20s} {std_losses[-1]:>10.3f} {plk_losses[-1]:>10.3f}")
    print(f"  {'Final val PPL':20s} {std_ppls[-1]:>10.1f} {plk_ppls[-1]:>10.1f}")
    print(f"  {'Best val PPL':20s} {min(std_ppls):>10.1f} {min(plk_ppls):>10.1f}")

    # Per-epoch table
    print(f"\n  {'Epoch':>5s}  {'Std Loss':>9s}  {'Std PPL':>9s}  {'Hyb Loss':>9s}  {'Hyb PPL':>9s}")
    print(f"  {'-'*48}")
    for i in range(cfg.n_epochs):
        print(f"  {i+1:5d}  {std_losses[i]:9.3f}  {std_ppls[i]:9.1f}  {plk_losses[i]:9.3f}  {plk_ppls[i]:9.1f}")

    ratio = min(plk_ppls) / min(std_ppls)
    if ratio < 0.95:
        print(f"\n  >> Hybrid attention WINS ({ratio:.2f}x perplexity)")
    elif ratio > 1.05:
        print(f"\n  >> Standard attention wins ({ratio:.2f}x perplexity)")
    else:
        print(f"\n  >> Roughly equivalent ({ratio:.2f}x)")

    # ── Generation samples ──
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES (temperature=0.8, top_k=40, max_tokens=60)")
    print("=" * 60)

    enc = tiktoken.get_encoding("gpt2")
    prompts = [
        "The president of the United States",
        "In the year 1945 ,",
        "The cat sat on the",
        "Scientists discovered that",
        "The album was released in",
    ]

    for prompt in prompts:
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  {'-'*56}")

        std_out = generate(std_model, enc, prompt, max_tokens=60, temperature=0.8, top_k=40)
        print(f"  Standard: {std_out}")

        hyb_out = generate(plk_model, enc, prompt, max_tokens=60, temperature=0.8, top_k=40)
        print(f"  Hybrid:   {hyb_out}")

    # Print learned Plücker scales
    print("\n" + "=" * 60)
    print("LEARNED PLUCKER SCALE PARAMETERS (per head, per layer)")
    print("=" * 60)
    for i, block in enumerate(plk_model.blocks):
        scales = block.attn.plucker_scale.data.cpu().numpy()
        scales_str = ", ".join(f"{s:.4f}" for s in scales)
        print(f"  Layer {i}: [{scales_str}]")
