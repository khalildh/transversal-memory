"""
exp_arc_local.py — Local ARC-like test for Gram bias on grid transformations.

Synthetic task mimicking ARC structure:
  - Small grids (5x5) with 10 colors
  - Tokenized like mdlARC: colors 0-9, <start>=10, <next_line>=11, <io_sep>=12, <end>=13
  - Each sequence: <start> input_grid <io_sep> output_grid <end>
  - Model must predict output tokens given input + separator

Transformations (randomly selected per batch):
  1. Horizontal flip
  2. Vertical flip
  3. 90° rotation
  4. Color swap (swap two specific colors)
  5. Transpose

Tests whether Gram bias helps learn spatial transformations vs standard attention.
Runs in minutes on CPU.

Usage:
  uv run python exp_arc_local.py                    # all variants
  uv run python exp_arc_local.py standard eigen_bias # specific variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from itertools import combinations

# ── Config ───────────────────────────────────────────────────────────────────

GRID_SIZE = 5       # 5x5 grids
N_COLORS = 10       # ARC has 10 colors (0-9)
VOCAB = 14          # 10 colors + 4 special tokens
START = 10
NEXT_LINE = 11
IO_SEP = 12
END = 13

# Sequence length: <start> + 5*(5+1) + <io_sep> + 5*(5+1) + <end> = 1 + 30 + 1 + 30 + 1 = 63
SEQ_LEN = 1 + GRID_SIZE * (GRID_SIZE + 1) + 1 + GRID_SIZE * (GRID_SIZE + 1) + 1

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
BATCH = 128
N_STEPS = 3000
LR = 3e-4
EVAL_EVERY = 200

N_TRANSFORMS = 5  # number of distinct transformations


# ── Data generation ──────────────────────────────────────────────────────────

def grid_to_tokens(grid):
    """Flatten a (H, W) grid to tokens with <next_line> delimiters."""
    tokens = []
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            tokens.append(grid[r, c].item())
        tokens.append(NEXT_LINE)
    return tokens


def apply_transform(grid, transform_id):
    """Apply one of N_TRANSFORMS transformations to a grid."""
    if transform_id == 0:
        return grid.flip(1)          # horizontal flip
    elif transform_id == 1:
        return grid.flip(0)          # vertical flip
    elif transform_id == 2:
        return grid.rot90(1, [0, 1]) # 90° clockwise
    elif transform_id == 3:
        # color swap: swap color 1 and color 2
        out = grid.clone()
        out[grid == 1] = 2
        out[grid == 2] = 1
        return out
    elif transform_id == 4:
        return grid.t()              # transpose
    else:
        return grid


def make_arc_batch(batch_size):
    """Generate batch of (input_grid → transformed_output_grid) sequences.

    Each sequence:
        <start> input_tokens <io_sep> output_tokens <end>

    Returns:
        x: (batch, SEQ_LEN) input tokens
        y: (batch, SEQ_LEN) target tokens (-100 for input portion, real for output)
        transform_ids: (batch,) which transform was applied
    """
    x = torch.full((batch_size, SEQ_LEN), 0, dtype=torch.long)
    y = torch.full((batch_size, SEQ_LEN), -100, dtype=torch.long)
    transform_ids = torch.randint(0, N_TRANSFORMS, (batch_size,))

    for i in range(batch_size):
        # Random input grid
        in_grid = torch.randint(0, N_COLORS, (GRID_SIZE, GRID_SIZE))
        out_grid = apply_transform(in_grid, transform_ids[i].item())

        in_tokens = grid_to_tokens(in_grid)
        out_tokens = grid_to_tokens(out_grid)

        # Build sequence
        seq = [START] + in_tokens + [IO_SEP] + out_tokens + [END]
        assert len(seq) == SEQ_LEN, f"seq len {len(seq)} != {SEQ_LEN}"

        x[i] = torch.tensor(seq, dtype=torch.long)

        # Target: only predict output tokens (after IO_SEP)
        # IO_SEP is at position 1 + len(in_tokens)
        sep_pos = 1 + len(in_tokens)
        # Predict output tokens at positions sep_pos+1 through SEQ_LEN-1
        for j in range(sep_pos + 1, SEQ_LEN):
            y[i, j] = x[i, j]

    return x, y, transform_ids


# ── Plücker primitives ───────────────────────────────────────────────────────

_PAIRS = list(combinations(range(4), 2))

_J6 = torch.tensor([
    [0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
    [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],
], dtype=torch.float32)

def exterior(p1, p2):
    parts = [p1[...,i]*p2[...,j] - p1[...,j]*p2[...,i] for i,j in _PAIRS]
    L = torch.stack(parts, dim=-1)
    return L / L.norm(dim=-1, keepdim=True).clamp(min=1e-12)


# ── Attention variants ───────────────────────────────────────────────────────

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class EigenBiasAttention(nn.Module):
    """Standard attention + causal Gram-mediated Plücker incidence bias."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.bias_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer('J6', _J6)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Standard Q·K
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, kk, v = qkv[0], qkv[1], qkv[2]
        std_logits = (q @ kk.transpose(-1, -2)) * self.scale

        # Plücker lines from consecutive token pairs
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)

        J = self.J6
        Jw = torch.einsum('bthi,ij->bthj', write_lines, J).permute(0, 2, 1, 3)
        rd = read_lines.permute(0, 2, 1, 3)

        # Causal Gram-mediated incidence
        outer = Jw.unsqueeze(-1) * Jw.unsqueeze(-2)  # (B, H, T, 6, 6)
        M_causal = outer.cumsum(dim=2)
        rd_M = torch.einsum('bhti,bhtij->bhtj', rd, M_causal)
        bias_raw = torch.einsum('bhti,bhsi->bhts', rd_M, Jw)

        scale = self.bias_scale.reshape(1, H, 1, 1)
        bias = bias_raw.abs() * scale

        logits = std_logits + bias
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        logits = logits.masked_fill(mask, float('-inf'))
        attn = F.softmax(logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class GramBiasAttention(nn.Module):
    """Standard attention + direct pairwise Plücker incidence bias (no Gram accumulation)."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.bias_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer('J6', _J6)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, kk, v = qkv[0], qkv[1], qkv[2]
        std_logits = (q @ kk.transpose(-1, -2)) * self.scale

        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)

        J = self.J6
        Jw = torch.einsum('bthi,ij->bthj', write_lines, J).permute(0, 2, 1, 3)
        rd = read_lines.permute(0, 2, 1, 3)

        # Direct pairwise incidence (no cumulative Gram)
        bias_raw = torch.einsum('bhti,bhsi->bhts', rd, Jw)
        scale = self.bias_scale.reshape(1, H, 1, 1)
        bias = bias_raw.abs() * scale

        logits = std_logits + bias
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        logits = logits.masked_fill(mask, float('-inf'))
        attn = F.softmax(logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


ATTN_MAP = {
    "standard": StandardAttention,
    "eigen_bias": EigenBiasAttention,
    "gram_bias": GramBiasAttention,
}


# ── Model ─────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_type):
        super().__init__()
        self.attn = ATTN_MAP[attn_type](d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GridTransformer(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, seq_len, attn_type):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, attn_type) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
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


# ── Training ─────────────────────────────────────────────────────────────────

def evaluate(model, n_batches=20):
    """Evaluate: fraction of output tokens predicted correctly."""
    model.eval()
    dev = next(model.parameters()).device
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, _ = make_arc_batch(BATCH)
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            mask = y != -100
            total += mask.sum().item()
            correct += ((preds == y) & mask).sum().item()
    return correct / max(total, 1)


def evaluate_full_grid(model, n_batches=20):
    """Evaluate: fraction of sequences where ALL output tokens are correct."""
    model.eval()
    dev = next(model.parameters()).device
    perfect = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, _ = make_arc_batch(BATCH)
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            mask = y != -100
            for i in range(BATCH):
                m = mask[i]
                if m.any():
                    total += 1
                    if (preds[i][m] == y[i][m]).all():
                        perfect += 1
    return perfect / max(total, 1)


def train_variant(attn_type, device):
    """Train a variant and return (acc_history, final_token_acc, final_grid_acc, time)."""
    model = GridTransformer(VOCAB, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN, attn_type).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {attn_type}: {params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    accs = []
    t0 = time.time()

    for step in range(1, N_STEPS + 1):
        model.train()
        x, y, _ = make_arc_batch(BATCH)
        x, y = x.to(device), y.to(device)
        logits = model(x).view(-1, VOCAB)
        loss = F.cross_entropy(logits, y.view(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            tok_acc = evaluate(model)
            grid_acc = evaluate_full_grid(model)
            accs.append((step, tok_acc, grid_acc))
            elapsed = time.time() - t0
            print(f"    step {step:4d}: loss={loss.item():.3f}  tok_acc={tok_acc:.3f}  grid_acc={grid_acc:.3f}  ({elapsed:.1f}s)")

    final_tok = evaluate(model, n_batches=50)
    final_grid = evaluate_full_grid(model, n_batches=50)
    total_time = time.time() - t0
    return accs, final_tok, final_grid, total_time


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    variants = sys.argv[1:] if len(sys.argv) > 1 else ["standard", "eigen_bias", "gram_bias"]
    results = {}

    print(f"Device: {device}")
    print(f"Task: ARC-like grid transformation ({GRID_SIZE}x{GRID_SIZE}, {N_COLORS} colors, {N_TRANSFORMS} transforms)")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Config: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")

    for v in variants:
        if v not in ATTN_MAP:
            print(f"Unknown variant: {v}. Choose from: {list(ATTN_MAP.keys())}")
            continue
        print(f"\n{'='*60}")
        print(f"  VARIANT: {v}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        accs, final_tok, final_grid, elapsed = train_variant(v, device)
        results[v] = (accs, final_tok, final_grid, elapsed)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY — ARC Grid Transformation")
        print(f"{'='*60}")
        print(f"  {'Variant':15s} {'Tok Acc':>10s} {'Grid Acc':>10s} {'Time':>8s}")
        print(f"  {'-'*50}")
        for v, (accs, final_tok, final_grid, elapsed) in results.items():
            print(f"  {v:15s} {final_tok:10.3f} {final_grid:10.3f} {elapsed:7.1f}s")

        # Learning curves
        print(f"\n  Learning curves (token accuracy / grid accuracy):")
        all_steps = sorted(set(s for v in results for s, _, _ in results[v][0]))
        header = f"  {'Step':>6s}"
        for v in results:
            header += f"  {v:>20s}"
        print(header)
        for step in all_steps:
            row = f"  {step:6d}"
            for v in results:
                entry = [(t, g) for s, t, g in results[v][0] if s == step]
                if entry:
                    t, g = entry[0]
                    row += f"  {t:.3f}/{g:.3f}        "[:20]
                else:
                    row += f"  {'---':>20s}"
            print(row)


if __name__ == "__main__":
    main()
