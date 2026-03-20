"""
exp_induction_test.py — Test whether Gram memory acts as a geometric induction head

Hypothesis: The online Gram memory provides a continuous, geometric analogue of
induction head behavior. Induction heads (Olsson et al., 2022) implement in-context
pattern completion — they detect "I've seen token A followed by B before, and now
I see A again, so predict B." They require at least 2 layers to form (one head to
copy, one to match). A 1-layer model structurally cannot form induction heads.

If Gram memory substitutes for induction heads, it should:
  1. Help most in 1-layer models (confirmed: -10.1% PPL on WikiText-2)
  2. Selectively help on "induction positions" — tokens where the correct next
     token could be predicted by pattern-matching against earlier context
  3. Show diminishing benefit at 4 layers (confirmed: -0.1% PPL)

Methodology (follows Olsson et al.):
  - Generate sequences with REPEATED random token subsequences:
    [random prefix] [A B C D ...] [random middle] [A B C D ...]
  - The second occurrence of the repeated pattern creates "induction positions"
    where a model that remembers the first occurrence can predict the next token
  - A 1-layer model cannot form induction heads to exploit this
  - If Gram memory helps specifically on these positions, it's acting as a
    geometric induction head

No external data or tokenizer needed — pure synthetic sequences.

Usage:
  uv run python exp_induction_test.py                  # 1-layer analysis
  uv run python exp_induction_test.py --layers 2       # 2-layer comparison
  uv run python exp_induction_test.py --layers 4       # 4-layer comparison
  uv run python exp_induction_test.py --all            # sweep 1,2,4 layers
"""

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import argparse
import json
from pathlib import Path
from itertools import combinations

# ── Config ───────────────────────────────────────────────────────────────────

class Config:
    vocab_size = 64     # small vocab — patterns are learnable
    d_model = 96
    n_heads = 4
    n_layers = 1
    dropout = 0.1
    seq_len = 128
    batch_size = 64
    n_epochs = 50
    lr = 3e-4
    grad_clip = 1.0
    warmup_steps = 200
    patience = 10
    min_delta = 0.1

# ── Synthetic data with induction patterns ───────────────────────────────────

def make_bigram_matrix(vocab_size, seed=42):
    """Create a fixed bigram transition matrix with ~5 successors per token."""
    bigram_probs = torch.zeros(vocab_size, vocab_size)
    for t in range(vocab_size):
        successors = torch.randint(0, vocab_size, (5,),
                                   generator=torch.Generator().manual_seed(seed + t))
        bigram_probs[t, successors] = 1.0
    bigram_probs = bigram_probs + 0.01
    bigram_probs = bigram_probs / bigram_probs.sum(dim=1, keepdim=True)
    return bigram_probs


def generate_bigram_sequences(n_seqs, seq_len, bigram_probs, seed=42):
    """Generate pure bigram sequences for training."""
    rng = torch.Generator().manual_seed(seed)
    vocab_size = bigram_probs.shape[0]
    x_all = torch.zeros(n_seqs, seq_len + 1, dtype=torch.long)

    for i in range(n_seqs):
        tok = torch.randint(0, vocab_size, (1,), generator=rng).item()
        x_all[i, 0] = tok
        for j in range(1, seq_len + 1):
            tok = torch.multinomial(bigram_probs[tok], 1, generator=rng).item()
            x_all[i, j] = tok

    y = x_all[:, 1:seq_len + 1].contiguous()
    x = x_all[:, :seq_len].contiguous()
    return x, y


def generate_mixed_training_data(n_seqs, seq_len, bigram_probs, repeat_frac=0.5,
                                  seed=42):
    """Generate training data mixing pure bigram and repeated-half sequences.

    repeat_frac of sequences have structure: [bigram half] [exact copy]
    The rest are pure bigram sequences.

    This gives the model the OPPORTUNITY to learn induction (from the repeated
    sequences) while also learning the base bigram distribution.
    """
    rng = torch.Generator().manual_seed(seed)
    vocab_size = bigram_probs.shape[0]
    half = seq_len // 2

    x_all = torch.zeros(n_seqs, seq_len + 1, dtype=torch.long)
    induction_mask = torch.zeros(n_seqs, seq_len, dtype=torch.bool)
    is_repeat_seq = torch.zeros(n_seqs, dtype=torch.bool)

    for i in range(n_seqs):
        # Decide if this is a repeated or pure bigram sequence
        do_repeat = (torch.rand(1, generator=rng).item() < repeat_frac)

        # Generate tokens from bigram distribution
        tok = torch.randint(0, vocab_size, (1,), generator=rng).item()
        tokens = [tok]
        for _ in range(seq_len):
            tok = torch.multinomial(bigram_probs[tok], 1, generator=rng).item()
            tokens.append(tok)

        if do_repeat:
            # Overwrite second half with copy of first half
            first_half = tokens[:half]
            for j in range(half):
                tokens[half + j] = first_half[j]
            # Make the target after the second half also match
            if 2 * half < seq_len + 1:
                tokens[2 * half] = first_half[0]

            # Mark induction positions in the second half
            # (position half+1 through half+half-1 in x-space, since target
            # at position t is tokens[t+1])
            for j in range(1, half):
                if half + j < seq_len:
                    induction_mask[i, half + j] = True
            is_repeat_seq[i] = True

        x_all[i] = torch.tensor(tokens[:seq_len + 1], dtype=torch.long)

    y = x_all[:, 1:seq_len + 1].contiguous()
    x = x_all[:, :seq_len].contiguous()
    return x, y, induction_mask, is_repeat_seq


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
    """Online Gram memory with scalar gate."""
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

# ── Per-token evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_by_position_type(model, x_batches, y_batches, mask_batches, device,
                              vocab_size):
    """Compute per-token loss split by induction vs non-induction positions."""
    model.eval()

    total_loss = 0.0
    total_count = 0
    induction_loss = 0.0
    induction_count = 0
    non_induction_loss = 0.0
    non_induction_count = 0

    for xb, yb, mb in zip(x_batches, y_batches, mask_batches):
        xb_dev = xb.to(device)
        yb_dev = yb.to(device)
        mb_dev = mb.to(device)
        logits = model(xb_dev)

        per_token_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), yb_dev.reshape(-1), reduction='none'
        ).reshape(xb.shape)

        non_mb = ~mb_dev

        total_loss += per_token_loss.sum().item()
        total_count += per_token_loss.numel()

        induction_loss += (per_token_loss * mb_dev.float()).sum().item()
        induction_count += mb_dev.sum().item()

        non_induction_loss += (per_token_loss * non_mb.float()).sum().item()
        non_induction_count += non_mb.sum().item()

    results = {
        "total_loss": total_loss / max(total_count, 1),
        "total_ppl": math.exp(min(total_loss / max(total_count, 1), 20)),
        "total_count": total_count,
        "induction_count": induction_count,
        "non_induction_count": non_induction_count,
        "induction_frac": induction_count / max(total_count, 1),
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
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


def make_batches_from_tensors(x, y, mask, batch_size):
    """Split pre-generated tensors into batches."""
    n = x.shape[0]
    batches_x, batches_y, batches_m = [], [], []
    for i in range(0, n, batch_size):
        if i + batch_size <= n:
            batches_x.append(x[i:i+batch_size])
            batches_y.append(y[i:i+batch_size])
            batches_m.append(mask[i:i+batch_size])
    return batches_x, batches_y, batches_m


def train_model(attn_type, cfg, device, train_x, train_y, val_x, val_y):
    """Train a model on mixed bigram + repeated sequences."""
    model = LM(cfg.vocab_size, cfg, attn_type).to(device)
    print(f"  {attn_type}: {model.count_params():,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    n_batches = train_x.shape[0] // cfg.batch_size
    total_steps = n_batches * cfg.n_epochs

    best_ppl = float('inf')
    patience_counter = 0
    global_step = 0
    losses, ppls = [], []

    for ep in range(cfg.n_epochs):
        t0 = time.time()
        model.train()

        perm = torch.randperm(train_x.shape[0])
        sx = train_x[perm]
        sy = train_y[perm]

        ep_loss = 0.0
        n_batch = 0
        for i in range(0, sx.shape[0], cfg.batch_size):
            if i + cfg.batch_size > sx.shape[0]:
                break
            xb = sx[i:i+cfg.batch_size].to(device)
            yb = sy[i:i+cfg.batch_size].to(device)

            lr = get_lr(global_step, cfg, total_steps)
            for pg in opt.param_groups:
                pg['lr'] = lr

            loss = F.cross_entropy(model(xb).view(-1, cfg.vocab_size), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss += loss.item()
            n_batch += 1
            global_step += 1

        avg = ep_loss / max(n_batch, 1)
        losses.append(avg)

        # Validate on pure bigram sequences (same distribution as training)
        model.eval()
        with torch.no_grad():
            vl = 0.0
            n_vb = 0
            for i in range(0, val_x.shape[0], cfg.batch_size):
                if i + cfg.batch_size > val_x.shape[0]:
                    break
                vx = val_x[i:i+cfg.batch_size].to(device)
                vy = val_y[i:i+cfg.batch_size].to(device)
                vl += F.cross_entropy(model(vx).view(-1, cfg.vocab_size),
                                      vy.view(-1)).item()
                n_vb += 1
            vl /= max(n_vb, 1)
        ppl = math.exp(min(vl, 20))
        ppls.append(ppl)

        elapsed = time.time() - t0
        improved = " *" if ppl < best_ppl - cfg.min_delta else ""
        print(f"    ep {ep+1:2d}: loss={avg:.3f}  ppl={ppl:.1f}  lr={lr:.2e}  ({elapsed:.1f}s){improved}")

        if ppl < best_ppl - cfg.min_delta:
            best_ppl = ppl
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"    Early stopping (no improvement for {cfg.patience} epochs)")
                break

    # Restore best model
    if best_ppl < float('inf'):
        model.load_state_dict(best_state)
        model = model.to(device)

    print(f"  Best bigram PPL: {best_ppl:.1f}")
    return model, losses, ppls


# ── Main analysis ────────────────────────────────────────────────────────────

def run_analysis(n_layers, device):
    """Train standard + online_mem at given depth, then do induction analysis."""
    cfg = Config()
    cfg.n_layers = n_layers

    # Scale model size with depth
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

    # Create shared bigram distribution
    bigram_probs = make_bigram_matrix(cfg.vocab_size, seed=42)
    entropy = -(bigram_probs * (bigram_probs + 1e-10).log()).sum(dim=1).mean().item()
    print(f"  Bigram entropy: {entropy:.2f} nats -> theoretical PPL floor: {math.exp(entropy):.1f}")
    print(f"  (On induction positions, optimal PPL = 1.0 if model can copy)")

    # Training data: 50% repeated-half, 50% pure bigram
    n_train = cfg.batch_size * 128
    n_val = cfg.batch_size * 16

    print(f"  Generating {n_train} training sequences (50% repeated, 50% bigram)...")
    train_x, train_y, train_mask, train_repeat = generate_mixed_training_data(
        n_train, cfg.seq_len, bigram_probs, repeat_frac=0.5, seed=42)
    print(f"    Repeated sequences: {train_repeat.sum().item()}/{n_train}")

    # Validation: ALL repeated-half sequences (to maximize induction signal)
    print(f"  Generating {n_val} validation sequences (all repeated halves)...")
    val_x, val_y, val_mask, _ = generate_mixed_training_data(
        n_val, cfg.seq_len, bigram_probs, repeat_frac=1.0, seed=123)

    ind_count = val_mask.sum().item()
    ind_total = val_mask.numel()
    print(f"  Val induction positions: {ind_count:,} / {ind_total:,} ({ind_count/ind_total*100:.1f}%)")

    val_bx, val_by, val_bm = make_batches_from_tensors(val_x, val_y, val_mask,
                                                         cfg.batch_size)

    results = {}

    for attn_type in ["standard", "online_mem"]:
        print(f"\n  --- {attn_type} ---")

        # Train on mixed data
        model, _, ppls = train_model(attn_type, cfg, device,
                                     train_x, train_y, val_x, val_y)

        # Evaluate per-token on validation (repeated-half sequences)
        print(f"  Evaluating per-token loss by position type...")
        r = evaluate_by_position_type(model, val_bx, val_by, val_bm, device,
                                      cfg.vocab_size)
        r["best_val_ppl"] = min(ppls)
        results[attn_type] = r

        print(f"    Best val PPL:       {r['best_val_ppl']:.2f}")
        print(f"    Total PPL:          {r['total_ppl']:.2f}  (per-token eval)")
        print(f"    Induction PPL:      {r['induction_ppl']:.2f}  ({r['induction_count']:,} pos, {r['induction_frac']*100:.1f}%)")
        print(f"    Non-induction PPL:  {r['non_induction_ppl']:.2f}  ({r['non_induction_count']:,} pos)")

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
    ind_delta = float('nan')
    if not math.isnan(std["induction_ppl"]) and std["induction_ppl"] > 0:
        ind_delta = (mem["induction_ppl"] - std["induction_ppl"]) / std["induction_ppl"] * 100
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
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
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
        r = run_analysis(nl, device)
        all_results.append(r)

    # Summary table across depths
    if len(all_results) > 1:
        print(f"\n\n{'='*60}")
        print(f"  DEPTH SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Layers':<8} {'Total D':>10} {'Induction D':>12} {'Non-ind D':>10} {'Selectivity':>12}")
        print(f"  {'-'*54}")
        for r in all_results:
            print(f"  {r['n_layers']:<8} {r['total_delta_pct']:>+9.1f}% {r['induction_delta_pct']:>+11.1f}% "
                  f"{r['non_induction_delta_pct']:>+9.1f}% {r['selectivity']:>+11.1f}%")

        print(f"\n  Hypothesis: if Gram memory acts as geometric induction head,")
        print(f"  selectivity should be most negative at 1 layer and approach")
        print(f"  zero at 4 layers (where real induction heads can form).")

    # Save results
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    results_path = ckpt_dir / "induction_analysis.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
