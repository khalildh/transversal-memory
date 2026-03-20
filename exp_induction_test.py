"""
exp_induction_test.py — Test whether Gram memory acts as a geometric induction head

Hypothesis: The online Gram memory provides a continuous, geometric analogue of
induction head behavior. Induction heads (Olsson et al., 2022) implement in-context
pattern completion — they detect "I've seen token A followed by B before, and now
I see A again, so predict B." They require at least 2 layers to form (one head to
copy, one to match). A 1-layer model structurally cannot form induction heads.

If Gram memory substitutes for induction heads, it should:
  1. Help most in 1-layer models (confirmed: -10.1% PPL)
  2. Selectively help on "induction positions" — tokens where the correct next
     token could be predicted by pattern-matching against earlier context
  3. Show diminishing benefit at 4 layers (confirmed: -0.1% PPL)

This script measures per-token loss broken down by token type:
  - "Induction positions": token t where token[t] appeared at some earlier
    position s, and token[s+1] == token[t+1] (pattern completion opportunity)
  - "Non-induction positions": all other positions

If Gram memory selectively reduces loss on induction positions relative to
non-induction positions, that's direct evidence for the geometric induction
head hypothesis.

Usage:
  uv run python exp_induction_test.py                  # 1-layer analysis
  uv run python exp_induction_test.py --layers 2       # 2-layer comparison
  uv run python exp_induction_test.py --layers 4       # 4-layer comparison
  uv run python exp_induction_test.py --all             # sweep 1,2,4 layers
  uv run python exp_induction_test.py --skip-train      # eval only (needs checkpoints)
"""

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import pyarrow.parquet as pq
import time
import math
import argparse
import json
from pathlib import Path
from itertools import combinations

# ── Config ───────────────────────────────────────────────────────────────────

class Config:
    d_model = 96      # small for 1-layer (matches ideas.md 1-layer experiments)
    n_heads = 4
    n_layers = 1
    dropout = 0.1
    seq_len = 128
    batch_size = 128
    n_epochs = 15      # train to convergence for clean analysis
    lr = 3e-4
    grad_clip = 1.0
    warmup_steps = 50
    patience = 3
    min_delta = 1.0
    data_dir = Path("data/wikitext")
    cache_dir = Path("data/cache")

# ── Plücker primitives ──────────────────────────────────────────────────────

_PAIRS4 = list(combinations(range(4), 2))

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
    """Online Gram memory with scalar gate (current best)."""
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

ATTN_CLASSES = {
    "standard": StandardAttention,
    "online_mem": OnlineMemoryAttention,
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

# ── Data ─────────────────────────────────────────────────────────────────────

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
        print(f"  Cached {split} tokens: {len(tokens):,} -> {cache_file}")
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

# ── Induction position detection ─────────────────────────────────────────────

def find_induction_positions(x_batch, y_batch):
    """Identify positions where an induction head would help.

    A position t is an "induction position" if:
      - token x[t] appeared at some earlier position s (s < t)
      - the token that followed s (i.e., x[s+1] or y[s]) equals y[t]
        (the correct next-token prediction at position t)

    This is the prefix-matching pattern that induction heads exploit:
    ... [A] [B] ... [A] [?]  →  predict [B]

    Returns a boolean mask of shape (B, T) where True = induction position.
    """
    B, T = x_batch.shape
    induction_mask = torch.zeros(B, T, dtype=torch.bool)

    for b in range(B):
        x = x_batch[b]   # (T,)
        y = y_batch[b]    # (T,) — the target (next token)

        for t in range(1, T):
            current_tok = x[t].item()
            target_tok = y[t].item()

            # Look for earlier occurrences of current_tok
            for s in range(t):
                if x[s].item() == current_tok and y[s].item() == target_tok:
                    induction_mask[b, t] = True
                    break  # one match suffices

    return induction_mask


def find_induction_positions_fast(x_batch, y_batch):
    """Vectorized version of induction position detection.

    For each position t, checks if there exists s < t such that
    x[s] == x[t] AND y[s] == y[t]. This identifies positions where
    copying the previous completion of the same token would be correct.
    """
    B, T = x_batch.shape
    # Encode (input_token, target_token) as a single integer for fast matching
    # Use a large multiplier to avoid collisions
    M = 100003  # prime larger than vocab size (~50k for GPT-2)
    combined = x_batch.long() * M + y_batch.long()  # (B, T)

    induction_mask = torch.zeros(B, T, dtype=torch.bool)

    for b in range(B):
        seen = set()
        for t in range(T):
            key = combined[b, t].item()
            if key in seen:
                induction_mask[b, t] = True
            seen.add(key)

    return induction_mask


# ── Per-token evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_by_position_type(model, batches, device):
    """Compute per-token cross-entropy loss, split by induction vs non-induction.

    Returns dict with:
      - total_loss, total_count: overall
      - induction_loss, induction_count: loss at induction positions
      - non_induction_loss, non_induction_count: loss at non-induction positions
      - induction_ppl, non_induction_ppl: perplexity for each type
    """
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    V = enc.n_vocab

    total_loss = 0.0
    total_count = 0
    induction_loss = 0.0
    induction_count = 0
    non_induction_loss = 0.0
    non_induction_count = 0

    for xb, yb in batches:
        xb_dev, yb_dev = xb.to(device), yb.to(device)
        logits = model(xb_dev)  # (B, T, V)

        # Per-token loss (no reduction)
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, V), yb_dev.reshape(-1), reduction='none'
        ).reshape(xb.shape)  # (B, T)

        # Find induction positions (on CPU for the token matching)
        ind_mask = find_induction_positions_fast(xb, yb).to(device)
        non_ind_mask = ~ind_mask

        # Accumulate
        total_loss += per_token_loss.sum().item()
        total_count += per_token_loss.numel()

        induction_loss += (per_token_loss * ind_mask.float()).sum().item()
        induction_count += ind_mask.sum().item()

        non_induction_loss += (per_token_loss * non_ind_mask.float()).sum().item()
        non_induction_count += non_ind_mask.sum().item()

    results = {
        "total_loss": total_loss / total_count,
        "total_ppl": math.exp(min(total_loss / total_count, 20)),
        "total_count": total_count,
        "induction_count": induction_count,
        "non_induction_count": non_induction_count,
        "induction_frac": induction_count / total_count if total_count > 0 else 0,
    }

    if induction_count > 0:
        avg_ind = induction_loss / induction_count
        results["induction_loss"] = avg_ind
        results["induction_ppl"] = math.exp(min(avg_ind, 20))
    else:
        results["induction_loss"] = float('nan')
        results["induction_ppl"] = float('nan')

    if non_induction_count > 0:
        avg_non = non_induction_loss / non_induction_count
        results["non_induction_loss"] = avg_non
        results["non_induction_ppl"] = math.exp(min(avg_non, 20))
    else:
        results["non_induction_loss"] = float('nan')
        results["non_induction_ppl"] = float('nan')

    return results


# ── Training ─────────────────────────────────────────────────────────────────

def get_lr(step, cfg, total_steps):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_model(attn_type, cfg, device, data_frac=0.5):
    enc = tiktoken.get_encoding("gpt2")
    train_data = load_tokens_cached("train", cfg)

    # Use 50% data by default for faster iteration
    n = int(len(train_data) * data_frac)
    train_data = train_data[:n]
    print(f"  Using {data_frac*100:.0f}% of training data ({n:,} tokens)")

    model = LM(enc.n_vocab, cfg, attn_type).to(device)
    print(f"  {attn_type}: {model.count_params():,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    n_batches = len(make_batches(train_data, cfg.seq_len, cfg.batch_size))
    total_steps = n_batches * cfg.n_epochs

    val_data = load_tokens_cached("val", cfg)
    best_ppl = float('inf')
    patience_counter = 0
    global_step = 0
    losses, ppls = [], []

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
        losses.append(avg)

        model.eval()
        with torch.no_grad():
            vb = make_batches(val_data, cfg.seq_len, cfg.batch_size)
            vl = sum(F.cross_entropy(model(x.to(device)).view(-1, enc.n_vocab),
                     y.to(device).view(-1)).item() for x, y in vb) / len(vb)
        ppl = math.exp(min(vl, 20))
        ppls.append(ppl)

        elapsed = time.time() - t0
        improved = " *" if ppl < best_ppl - cfg.min_delta else ""
        print(f"    ep {ep+1:2d}: loss={avg:.3f}  ppl={ppl:.1f}  ({elapsed:.1f}s){improved}")

        if ppl < best_ppl - cfg.min_delta:
            best_ppl = ppl
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"    Early stopping (no improvement for {cfg.patience} epochs)")
                break

    # Save checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_name = f"induction_{attn_type}_{cfg.n_layers}L.pt"
    torch.save({
        "model": model.state_dict(),
        "type": attn_type,
        "losses": losses,
        "ppls": ppls,
        "vocab": enc.n_vocab,
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
    }, ckpt_dir / ckpt_name)
    print(f"  Saved {ckpt_name}")

    return model, losses, ppls


def load_model(attn_type, cfg, device):
    """Load a trained model from checkpoint."""
    ckpt_name = f"induction_{attn_type}_{cfg.n_layers}L.pt"
    ckpt_path = Path("checkpoints") / ckpt_name
    if not ckpt_path.exists():
        return None

    enc = tiktoken.get_encoding("gpt2")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = LM(ckpt["vocab"], cfg, attn_type).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded {ckpt_name} (best PPL: {min(ckpt['ppls']):.1f})")
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def run_analysis(n_layers, device, skip_train=False, data_frac=0.5):
    """Train standard + online_mem at given depth, then do induction analysis."""
    cfg = Config()
    cfg.n_layers = n_layers

    # Scale model size with depth (match ideas.md experiments)
    if n_layers == 1:
        cfg.d_model = 96
        cfg.n_heads = 4
    elif n_layers == 2:
        cfg.d_model = 128
        cfg.n_heads = 4
    elif n_layers >= 4:
        cfg.d_model = 192
        cfg.n_heads = 6

    print(f"\n{'='*60}")
    print(f"  INDUCTION HEAD ANALYSIS: {n_layers}-layer, d={cfg.d_model}, h={cfg.n_heads}")
    print(f"{'='*60}")

    enc = tiktoken.get_encoding("gpt2")
    val_data = load_tokens_cached("val", cfg)
    val_batches = make_batches(val_data, cfg.seq_len, cfg.batch_size)

    results = {}

    for attn_type in ["standard", "online_mem"]:
        print(f"\n  --- {attn_type} ---")

        if skip_train:
            model = load_model(attn_type, cfg, device)
            if model is None:
                print(f"  No checkpoint found for {attn_type} {n_layers}L, training...")
                model, _, _ = train_model(attn_type, cfg, device, data_frac=data_frac)
        else:
            model, _, _ = train_model(attn_type, cfg, device, data_frac=data_frac)

        # Per-token analysis
        print(f"\n  Evaluating per-token loss by position type...")
        r = evaluate_by_position_type(model, val_batches, device)
        results[attn_type] = r

        print(f"    Total PPL:          {r['total_ppl']:.1f}")
        print(f"    Induction PPL:      {r['induction_ppl']:.1f}  ({r['induction_count']:,} positions, {r['induction_frac']*100:.1f}%)")
        print(f"    Non-induction PPL:  {r['non_induction_ppl']:.1f}  ({r['non_induction_count']:,} positions)")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Comparison
    print(f"\n  {'='*60}")
    print(f"  COMPARISON ({n_layers}-layer)")
    print(f"  {'='*60}")

    std = results["standard"]
    mem = results["online_mem"]

    total_delta = (mem["total_ppl"] - std["total_ppl"]) / std["total_ppl"] * 100
    ind_delta = (mem["induction_ppl"] - std["induction_ppl"]) / std["induction_ppl"] * 100 if not math.isnan(std["induction_ppl"]) else float('nan')
    non_delta = (mem["non_induction_ppl"] - std["non_induction_ppl"]) / std["non_induction_ppl"] * 100

    print(f"  {'Position type':<20} {'Standard':>10} {'Online mem':>10} {'Delta':>8}")
    print(f"  {'-'*50}")
    print(f"  {'Total':<20} {std['total_ppl']:>10.1f} {mem['total_ppl']:>10.1f} {total_delta:>+7.1f}%")
    print(f"  {'Induction':<20} {std['induction_ppl']:>10.1f} {mem['induction_ppl']:>10.1f} {ind_delta:>+7.1f}%")
    print(f"  {'Non-induction':<20} {std['non_induction_ppl']:>10.1f} {mem['non_induction_ppl']:>10.1f} {non_delta:>+7.1f}%")
    print(f"  {'Induction fraction':<20} {std['induction_frac']*100:>9.1f}%")

    # The key test: does Gram memory help MORE on induction positions?
    selectivity = ind_delta - non_delta  # negative = selective induction benefit
    print(f"\n  Selectivity (ind_delta - non_delta): {selectivity:+.1f}%")
    if selectivity < -1.0:
        print(f"  -> Gram memory SELECTIVELY helps on induction positions")
        print(f"     (supports geometric induction head hypothesis)")
    elif selectivity > 1.0:
        print(f"  -> Gram memory helps MORE on non-induction positions")
        print(f"     (geometry provides general context, not pattern-matching)")
    else:
        print(f"  -> No selective benefit (within noise)")

    return {
        "n_layers": n_layers,
        "standard": std,
        "online_mem": mem,
        "total_delta_pct": total_delta,
        "induction_delta_pct": ind_delta,
        "non_induction_delta_pct": non_delta,
        "selectivity": selectivity,
    }


def main():
    parser = argparse.ArgumentParser(description="Induction head analysis for Gram memory")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers (default: 1)")
    parser.add_argument("--all", action="store_true", help="Sweep 1, 2, 4 layers")
    parser.add_argument("--skip-train", action="store_true", help="Load from checkpoints only")
    parser.add_argument("--data-frac", type=float, default=0.5, help="Training data fraction")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")

    if args.all:
        layer_configs = [1, 2, 4]
    else:
        layer_configs = [args.layers]

    all_results = []
    for nl in layer_configs:
        r = run_analysis(nl, device, skip_train=args.skip_train, data_frac=args.data_frac)
        all_results.append(r)

    # Summary table across depths
    if len(all_results) > 1:
        print(f"\n\n{'='*60}")
        print(f"  DEPTH SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Layers':<8} {'Total Δ':>10} {'Induction Δ':>12} {'Non-ind Δ':>10} {'Selectivity':>12}")
        print(f"  {'-'*54}")
        for r in all_results:
            print(f"  {r['n_layers']:<8} {r['total_delta_pct']:>+9.1f}% {r['induction_delta_pct']:>+11.1f}% "
                  f"{r['non_induction_delta_pct']:>+9.1f}% {r['selectivity']:>+11.1f}%")

        print(f"\n  Hypothesis: if Gram memory acts as geometric induction head,")
        print(f"  selectivity should be most negative at 1 layer and approach")
        print(f"  zero at 4 layers (where real induction heads can form).")

    # Save results
    results_path = Path("checkpoints/induction_analysis.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
