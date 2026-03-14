"""
exp_lm_v2.py — Plücker KERNEL attention vs standard attention for language modeling

Architectural fix #1: "Plücker kernel attention"
Changes from exp_lm.py's PluckerAttention:
  1. Asymmetric Q/K lines: separate projections for query and key lines
     (W1_q, W2_q for query line; W1_k, W2_k for key line)
  2. Signed inner product as logit: raw plucker_inner(L_query_i, L_key_j)
     scaled by 1/sqrt(6), NOT -log|.|
  3. Standard softmax + causal mask

The hypothesis: the symmetric incidence scoring was the bottleneck.
Asymmetric Q/K with signed logits lets the model learn directional
attention patterns, analogous to how Q·K works in standard attention.
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

def log(msg=""):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()

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


class PluckerKernelAttention(nn.Module):
    """
    Multi-head Plücker KERNEL attention (v2).

    Key changes from v1:
      - Asymmetric Q/K: separate line projections for queries and keys
      - Signed inner product as logit (not -log|.|)
      - Scale by 1/sqrt(6) (the Plücker vector dimension)

    Each head maps each token to a QUERY line and a KEY line in P³
    via 4 separate projections. The attention logit between token i
    (query) and token j (key) is:

        logit(i,j) = plucker_inner(L_query_i, L_key_j) / sqrt(6)
                   = (L_query_i @ J6 @ L_key_j) / sqrt(6)
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 6 ** -0.5  # 1/sqrt(6) for Plücker 6-vectors

        # Asymmetric Q/K: 4 projection matrices
        # Query line: W1_q, W2_q map token → two R⁴ points → query line
        self.W1_q = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_q = nn.Linear(d_model, 4 * n_heads, bias=False)
        # Key line: W1_k, W2_k map token → two R⁴ points → key line
        self.W1_k = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_k = nn.Linear(d_model, 4 * n_heads, bias=False)

        # Value + output (standard, same as dot-product attention)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.register_buffer('J6', _J6_CONST)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads

        # Query lines: each token gets two points per head → exterior product
        p1_q = self.W1_q(x).reshape(B, T, H, 4)  # (B, T, H, 4)
        p2_q = self.W2_q(x).reshape(B, T, H, 4)
        L_query = plucker_lines(p1_q, p2_q)       # (B, T, H, 6)

        # Key lines: separate projections
        p1_k = self.W1_k(x).reshape(B, T, H, 4)
        p2_k = self.W2_k(x).reshape(B, T, H, 4)
        L_key = plucker_lines(p1_k, p2_k)         # (B, T, H, 6)

        # Attention logits via signed Plücker inner product
        # plucker_inner(L_q, L_k) = L_q @ J6 @ L_k^T
        L_q_h = L_query.permute(0, 2, 1, 3)       # (B, H, T, 6)
        L_k_h = L_key.permute(0, 2, 1, 3)         # (B, H, T, 6)

        JL_k = L_k_h @ self.J6.T                  # (B, H, T, 6)
        attn_logits = (L_q_h @ JL_k.transpose(-1, -2)) * self.scale  # (B, H, T, T)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_logits = attn_logits.masked_fill(mask, float('-inf'))

        attn = self.drop(F.softmax(attn_logits, dim=-1))

        # Value aggregation (standard)
        v = self.v_proj(x).reshape(B, T, H, self.d_head).permute(0, 2, 1, 3)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ── Transformer blocks ──────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, attn_type="standard", dropout=0.1):
        super().__init__()
        if attn_type == "plucker":
            self.attn = PluckerKernelAttention(d_model, n_heads, dropout)
        else:
            self.attn = StandardAttention(d_model, n_heads, dropout)
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

    log(f"\nLoading data...")
    train_data = load_wikitext("train", cfg.data_dir, enc)
    val_data = load_wikitext("val", cfg.data_dir, enc)
    log(f"  Train: {len(train_data):,} tokens")
    log(f"  Val:   {len(val_data):,} tokens")

    model = SmallLM(vocab_size, cfg, attn_type).to(device)
    n_params = model.count_params()
    log(f"  Model ({attn_type}): {n_params:,} params")

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

            # Sync MPS every step to prevent deadlocks
            if device.type == "mps":
                torch.mps.synchronize()

            if step % 50 == 0:
                log(f"    step {step}/{len(batches)} loss={loss.item():.3f}")

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
        log(f"  Epoch {epoch+1:2d}/{cfg.n_epochs}: "
            f"train_loss={avg_loss:.3f}  val_ppl={val_ppl:.1f}  ({elapsed:.1f}s)")

        # Clear MPS cache to avoid OOM crashes
        if device.type == "mps":
            torch.mps.empty_cache()

    # Checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"lm_v2_{attn_type}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "attn_type": attn_type,
        "cfg": {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
        "train_losses": train_losses,
        "val_ppls": val_ppls,
        "vocab_size": vocab_size,
    }, ckpt_path)
    log(f"  Saved checkpoint: {ckpt_path}")

    return model, train_losses, val_ppls


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
    # Use MPS if available, override with DEVICE env var
    import os
    device_name = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    log(f"Device: {device}")

    cfg = Config()

    # Train standard attention
    log("=" * 60)
    log("MODEL 1: Standard dot-product attention")
    log("=" * 60)
    std_model, std_losses, std_ppls = train_model("standard", cfg, device)

    # Train Plücker kernel attention
    log("\n" + "=" * 60)
    log("MODEL 2: Plücker kernel attention (v2 — asymmetric Q/K, signed logits)")
    log("=" * 60)
    plk_model, plk_losses, plk_ppls = train_model("plucker", cfg, device)

    # ── Comparison table ──────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("COMPARISON")
    log("=" * 60)
    log(f"  {'':20s} {'Standard':>10s} {'Plücker v2':>10s}")
    log(f"  {'-'*42}")
    log(f"  {'Params':20s} {std_model.count_params():>10,d} {plk_model.count_params():>10,d}")
    log(f"  {'Final train loss':20s} {std_losses[-1]:>10.3f} {plk_losses[-1]:>10.3f}")
    log(f"  {'Final val PPL':20s} {std_ppls[-1]:>10.1f} {plk_ppls[-1]:>10.1f}")
    log(f"  {'Best val PPL':20s} {min(std_ppls):>10.1f} {min(plk_ppls):>10.1f}")

    log(f"\n  Epoch-by-epoch val PPL:")
    log(f"  {'Epoch':>7s} {'Standard':>10s} {'Plücker v2':>12s}")
    for i in range(cfg.n_epochs):
        log(f"  {i+1:>7d} {std_ppls[i]:>10.1f} {plk_ppls[i]:>12.1f}")

    ratio = min(plk_ppls) / min(std_ppls)
    if ratio < 0.95:
        log(f"\n  >> Plücker kernel attention WINS ({ratio:.2f}x perplexity)")
    elif ratio > 1.05:
        log(f"\n  >> Standard attention wins ({ratio:.2f}x perplexity)")
    else:
        log(f"\n  >> Roughly equivalent ({ratio:.2f}x)")

    # ── Text generation samples ───────────────────────────────────────────────
    enc = tiktoken.get_encoding("gpt2")
    prompts = [
        "The president of the United States",
        "In the year 1945 ,",
        "The cat sat on the",
        "Scientists discovered that",
        "The album was released in",
    ]

    log("\n" + "=" * 60)
    log("GENERATION SAMPLES")
    log("=" * 60)

    for prompt in prompts:
        log(f"\nPrompt: \"{prompt}\"")
        log(f"  Standard:   {generate(std_model, enc, prompt, max_tokens=60, temperature=0.8, top_k=40)}")
        log(f"  Plücker v2: {generate(plk_model, enc, prompt, max_tokens=60, temperature=0.8, top_k=40)}")
