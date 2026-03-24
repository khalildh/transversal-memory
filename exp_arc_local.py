"""
exp_arc_local.py — Local ARC-like test for Gram bias on grid transformations.

HARD VERSION: Multi-example rule inference.
  - Each sequence has 2 demonstration pairs + 1 test pair
  - The transform is drawn from a LARGE family (parameterized, not memorizable)
  - Model must infer the rule from demos and apply it to the test input
  - This is the core ARC capability: few-shot abstract reasoning

Transform families (parameterized — hundreds of variants):
  1. Color permutation: randomly permute a subset of colors
  2. Crop+tile: extract a sub-region and tile it
  3. Conditional replace: if cell == color_a, replace with color_b
  4. Row/col shift: cyclically shift rows or columns by k
  5. Mask+fill: cells matching a color get filled with a pattern from another region

Tokenization matches mdlARC: colors 0-9, <start>=10, <next_line>=11, <io_sep>=12, <end>=13

Usage:
  uv run python exp_arc_local.py                    # all variants
  uv run python exp_arc_local.py standard eigen_bias # specific variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import random
from itertools import combinations

# ── Config ───────────────────────────────────────────────────────────────────

GRID_SIZE = 5       # 5x5 grids
N_COLORS = 6        # fewer colors = more structure to find
VOCAB = 14          # 10 colors + 4 special tokens
START = 10
NEXT_LINE = 11
IO_SEP = 12
END = 13

N_DEMOS = 2         # number of demonstration pairs per sequence
# Sequence: N_DEMOS * (<start> + grid + <io_sep> + grid + <end>) + 1 test pair
# Per pair: 1 + 5*(5+1) + 1 + 5*(5+1) + 1 = 63
TOKENS_PER_PAIR = 1 + GRID_SIZE * (GRID_SIZE + 1) + 1 + GRID_SIZE * (GRID_SIZE + 1) + 1
SEQ_LEN = (N_DEMOS + 1) * TOKENS_PER_PAIR  # 3 pairs * 63 = 189

D_MODEL = 96
N_HEADS = 4
N_LAYERS = 3
BATCH = 64
N_STEPS = 5000
LR = 3e-4
EVAL_EVERY = 250


# ── Transform families ───────────────────────────────────────────────────────

def random_color_permutation():
    """Return a transform that permutes colors (parameterized by the permutation)."""
    # Permute colors 0..N_COLORS-1
    perm = list(range(N_COLORS))
    random.shuffle(perm)
    # Make sure it's not identity
    while perm == list(range(N_COLORS)):
        random.shuffle(perm)
    def transform(grid):
        out = grid.clone()
        for src, dst in enumerate(perm):
            out[grid == src] = dst
        return out
    return transform


def random_conditional_replace():
    """If cell == color_a, replace with color_b. Parameterized by (a, b)."""
    a = random.randint(0, N_COLORS - 1)
    b = random.randint(0, N_COLORS - 1)
    while b == a:
        b = random.randint(0, N_COLORS - 1)
    def transform(grid):
        out = grid.clone()
        out[grid == a] = b
        return out
    return transform


def random_row_shift():
    """Cyclically shift all rows by k positions. Parameterized by k."""
    k = random.randint(1, GRID_SIZE - 1)
    def transform(grid):
        return grid.roll(shifts=k, dims=1)
    return transform


def random_col_shift():
    """Cyclically shift all columns by k positions. Parameterized by k."""
    k = random.randint(1, GRID_SIZE - 1)
    def transform(grid):
        return grid.roll(shifts=k, dims=0)
    return transform


def random_row_col_swap():
    """Swap two specific rows and two specific columns. Parameterized by (r1,r2,c1,c2)."""
    r1, r2 = random.sample(range(GRID_SIZE), 2)
    c1, c2 = random.sample(range(GRID_SIZE), 2)
    def transform(grid):
        out = grid.clone()
        out[r1], out[r2] = grid[r2].clone(), grid[r1].clone()
        out2 = out.clone()
        out2[:, c1], out2[:, c2] = out[:, c2].clone(), out[:, c1].clone()
        return out2
    return transform


def random_border_fill():
    """Fill the border cells with a specific color. Parameterized by color."""
    c = random.randint(0, N_COLORS - 1)
    def transform(grid):
        out = grid.clone()
        out[0, :] = c
        out[-1, :] = c
        out[:, 0] = c
        out[:, -1] = c
        return out
    return transform


def random_reflect_axis():
    """Reflect along a random axis. Parameterized by axis (0=vert, 1=horiz) + optional transpose."""
    choice = random.randint(0, 3)
    def transform(grid):
        if choice == 0:
            return grid.flip(0)
        elif choice == 1:
            return grid.flip(1)
        elif choice == 2:
            return grid.t()
        else:
            return grid.rot90(1, [0, 1])
    return transform


TRANSFORM_FAMILIES = [
    random_color_permutation,
    random_conditional_replace,
    random_row_shift,
    random_col_shift,
    random_row_col_swap,
    random_border_fill,
    random_reflect_axis,
]


def sample_transform():
    """Sample a random parameterized transform."""
    family = random.choice(TRANSFORM_FAMILIES)
    return family()


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


def make_arc_batch(batch_size):
    """Generate batch of multi-example ARC-like sequences.

    Each sequence has N_DEMOS demonstration pairs + 1 test pair, all sharing
    the same (randomly sampled) transform.

    Sequence structure:
        <start> demo1_in <io_sep> demo1_out <end>
        <start> demo2_in <io_sep> demo2_out <end>
        <start> test_in  <io_sep> test_out  <end>

    Loss is only on the test output tokens (last pair, after <io_sep>).
    """
    x = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    y = torch.full((batch_size, SEQ_LEN), -100, dtype=torch.long)

    for i in range(batch_size):
        transform = sample_transform()
        pos = 0

        for pair_idx in range(N_DEMOS + 1):
            in_grid = torch.randint(0, N_COLORS, (GRID_SIZE, GRID_SIZE))
            out_grid = transform(in_grid)

            in_tokens = grid_to_tokens(in_grid)
            out_tokens = grid_to_tokens(out_grid)

            seq = [START] + in_tokens + [IO_SEP] + out_tokens + [END]
            assert len(seq) == TOKENS_PER_PAIR

            for j, tok in enumerate(seq):
                x[i, pos + j] = tok

            # Only compute loss on the TEST pair's output tokens
            if pair_idx == N_DEMOS:
                sep_offset = 1 + len(in_tokens)  # position of IO_SEP within this pair
                for j in range(sep_offset + 1, TOKENS_PER_PAIR):
                    y[i, pos + j] = x[i, pos + j]

            pos += TOKENS_PER_PAIR

    return x, y


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


ATTN_MAP = {
    "standard": StandardAttention,
    "eigen_bias": EigenBiasAttention,
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
    """Token-level accuracy on output portion."""
    model.eval()
    dev = next(model.parameters()).device
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = make_arc_batch(BATCH)
            x, y = x.to(dev), y.to(dev)
            preds = model(x).argmax(dim=-1)
            mask = y != -100
            total += mask.sum().item()
            correct += ((preds == y) & mask).sum().item()
    return correct / max(total, 1)


def evaluate_full_grid(model, n_batches=20):
    """Grid-level accuracy: all output tokens correct."""
    model.eval()
    dev = next(model.parameters()).device
    perfect = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = make_arc_batch(BATCH)
            x, y = x.to(dev), y.to(dev)
            preds = model(x).argmax(dim=-1)
            mask = y != -100
            for i in range(BATCH):
                m = mask[i]
                if m.any():
                    total += 1
                    if (preds[i][m] == y[i][m]).all():
                        perfect += 1
    return perfect / max(total, 1)


def train_variant(attn_type, device):
    model = GridTransformer(VOCAB, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN, attn_type).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {attn_type}: {params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    accs = []
    t0 = time.time()

    for step in range(1, N_STEPS + 1):
        model.train()
        x, y = make_arc_batch(BATCH)
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
            print(f"    step {step:4d}: loss={loss.item():.3f}  tok={tok_acc:.3f}  grid={grid_acc:.3f}  ({elapsed:.1f}s)")

    final_tok = evaluate(model, n_batches=50)
    final_grid = evaluate_full_grid(model, n_batches=50)
    total_time = time.time() - t0
    return accs, final_tok, final_grid, total_time


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    random.seed(42)

    variants = sys.argv[1:] if len(sys.argv) > 1 else ["standard", "eigen_bias"]
    results = {}

    print(f"Device: {device}")
    print(f"Task: ARC-like few-shot rule inference ({N_DEMOS} demos + 1 test)")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, {N_COLORS} colors, 7 transform families (parameterized)")
    print(f"Seq length: {SEQ_LEN} tokens")
    print(f"Config: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")

    for v in variants:
        if v not in ATTN_MAP:
            print(f"Unknown variant: {v}. Choose from: {list(ATTN_MAP.keys())}")
            continue
        print(f"\n{'='*60}")
        print(f"  VARIANT: {v}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        random.seed(42)
        accs, final_tok, final_grid, elapsed = train_variant(v, device)
        results[v] = (accs, final_tok, final_grid, elapsed)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY — ARC Few-Shot Rule Inference")
        print(f"{'='*60}")
        print(f"  {'Variant':15s} {'Tok Acc':>10s} {'Grid Acc':>10s} {'Time':>8s}")
        print(f"  {'-'*50}")
        for v, (accs, final_tok, final_grid, elapsed) in results.items():
            print(f"  {v:15s} {final_tok:10.3f} {final_grid:10.3f} {elapsed:7.1f}s")

        print(f"\n  Learning curves:")
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
                    row += f"  {t:6.3f}/{g:.3f}      "[:20]
                else:
                    row += f"  {'---':>20s}"
            print(row)


if __name__ == "__main__":
    main()
