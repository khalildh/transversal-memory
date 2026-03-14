"""
exp_lm_v3.py — Plücker BIGRAM attention vs standard attention for language modeling

Trains two small transformers on WikiText-2:
  1. Standard dot-product attention (baseline)
  2. Plücker bigram attention (query-lines encode token pairs, key-lines single tokens)

Plücker bigram attention:
  - Query for position i: concatenate x_i and x_{i-1} (zeros for i=0),
    project to two R⁴ points via W1_q and W2_q, exterior product → 6-vector query line.
  - Key for position j: project to two R⁴ points via W1_k and W2_k,
    exterior product → 6-vector key line.
  - Attention logits: raw signed plucker_inner(query_i, key_j) / sqrt(6).
  - Standard softmax + causal mask.
  - Values: standard linear projection.

This gives degree-4 interaction and encodes local sequential (bigram) context
into the geometric representation.
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
    batch_size = 32
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


class PluckerBigramAttention(nn.Module):
    """
    Multi-head Plücker bigram attention.

    Query-lines encode PAIRS of consecutive tokens (bigrams):
      - For position i, concatenate x_i and x_{i-1} (zeros for i=0)
      - Project to two R⁴ points via W1_q and W2_q (4 × 2*d_model each per head)
      - Exterior product → 6-vector query line

    Key-lines encode single tokens:
      - For position j, project to two R⁴ points via W1_k and W2_k (4 × d_model each per head)
      - Exterior product → 6-vector key line

    Attention logits: raw signed plucker_inner(query_i, key_j) / sqrt(6).
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 6.0 ** -0.5  # 1/sqrt(6) for Plucker 6-vectors

        # Query projections: from bigram (2*d_model) to two R⁴ points per head
        self.W1_q = nn.Linear(2 * d_model, 4 * n_heads, bias=False)
        self.W2_q = nn.Linear(2 * d_model, 4 * n_heads, bias=False)

        # Key projections: from single token (d_model) to two R⁴ points per head
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

        # Build bigram inputs for queries: [x_i; x_{i-1}]
        # x_prev is x shifted right by 1, with zeros for position 0
        x_prev = torch.zeros_like(x)
        x_prev[:, 1:, :] = x[:, :-1, :]
        bigram_input = torch.cat([x, x_prev], dim=-1)  # (B, T, 2*D)

        # Query lines from bigrams
        q_p1 = self.W1_q(bigram_input).reshape(B, T, H, 4)  # (B, T, H, 4)
        q_p2 = self.W2_q(bigram_input).reshape(B, T, H, 4)
        q_lines = plucker_lines(q_p1, q_p2)  # (B, T, H, 6)

        # Key lines from single tokens
        k_p1 = self.W1_k(x).reshape(B, T, H, 4)  # (B, T, H, 4)
        k_p2 = self.W2_k(x).reshape(B, T, H, 4)
        k_lines = plucker_lines(k_p1, k_p2)  # (B, T, H, 6)

        # Compute signed Plücker inner product via Hodge dual: q @ J6 @ k^T
        q_h = q_lines.permute(0, 2, 1, 3)  # (B, H, T, 6)
        k_h = k_lines.permute(0, 2, 1, 3)  # (B, H, T, 6)

        Jk = k_h @ self.J6.T  # (B, H, T, 6)
        attn_logits = (q_h @ Jk.transpose(-1, -2)) * self.scale  # (B, H, T, T)

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
            AttnClass = PluckerBigramAttention
        else:
            AttnClass = StandardAttention
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


# ── Logging ──────────────────────────────────────────────────────────────────

LOG_FILE = None

def log(msg=""):
    """Print and flush, also write to log file."""
    print(msg)
    sys.stdout.flush()
    if LOG_FILE:
        LOG_FILE.write(msg + "\n")
        LOG_FILE.flush()


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
        n_batches = len(batches)

        for step, (xb, yb) in enumerate(batches):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % 50 == 0:
                log(f"    step {step+1}/{n_batches}  loss={loss.item():.3f}")

        avg_loss = epoch_loss / n_batches
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
        import gc; gc.collect()

    # Checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"lm_v3_{attn_type}.pt"
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
    LOG_FILE = open("/tmp/lm_v3_progress.log", "w")
    import os
    import gc
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
    if force_cpu:
        device = torch.device("cpu")
        torch.set_num_threads(4)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"Device: {device}")

    cfg = Config()

    # Train standard attention
    log("=" * 60)
    log("MODEL 1: Standard dot-product attention")
    log("=" * 60)
    std_model, std_losses, std_ppls = train_model("standard", cfg, device)

    # Free memory before training second model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Train Plücker bigram attention
    log("\n" + "=" * 60)
    log("MODEL 2: Plücker bigram attention")
    log("=" * 60)
    plk_model, plk_losses, plk_ppls = train_model("plucker", cfg, device)

    # ── Comparison table ──────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("COMPARISON")
    log("=" * 60)
    log(f"  {'':20s} {'Standard':>10s} {'Plücker':>10s}")
    log(f"  {'-'*40}")
    log(f"  {'Params':20s} {std_model.count_params():>10,d} {plk_model.count_params():>10,d}")
    log(f"  {'Final train loss':20s} {std_losses[-1]:>10.3f} {plk_losses[-1]:>10.3f}")
    log(f"  {'Final val PPL':20s} {std_ppls[-1]:>10.1f} {plk_ppls[-1]:>10.1f}")
    log(f"  {'Best val PPL':20s} {min(std_ppls):>10.1f} {min(plk_ppls):>10.1f}")

    ratio = min(plk_ppls) / min(std_ppls)
    if ratio < 0.95:
        log(f"\n  >> Plücker bigram attention WINS ({ratio:.2f}x perplexity)")
    elif ratio > 1.05:
        log(f"\n  >> Standard attention wins ({ratio:.2f}x perplexity)")
    else:
        log(f"\n  >> Roughly equivalent ({ratio:.2f}x)")

    # ── Epoch-by-epoch table ──────────────────────────────────────────────
    log(f"\n  Epoch-by-epoch val PPL:")
    log(f"  {'Epoch':>6s}  {'Standard':>10s}  {'Plücker':>10s}")
    for ep in range(cfg.n_epochs):
        log(f"  {ep+1:6d}  {std_ppls[ep]:10.1f}  {plk_ppls[ep]:10.1f}")

    # ── Generation samples ────────────────────────────────────────────────
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
        log(f"\n  Prompt: \"{prompt}\"")
        log(f"  {'─'*56}")

        std_out = generate(std_model, enc, prompt, max_tokens=60, temperature=0.8, top_k=40)
        log(f"  Standard: {std_out}")

        plk_out = generate(plk_model, enc, prompt, max_tokens=60, temperature=0.8, top_k=40)
        log(f"  Plücker:  {plk_out}")

    log("\n" + "=" * 60)
    log("DONE")
    log("=" * 60)
    LOG_FILE.close()
