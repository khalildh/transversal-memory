"""
exp_arc_real.py — Test Gram bias on REAL ARC-AGI tasks.

Loads actual ARC training tasks, trains a small transformer to predict
output grids given demonstration pairs + test input. Compares standard
attention vs eigen_bias (causal Gram-mediated incidence).

Each training step samples a random task, uses its train pairs as demos
and one pair as the "test" (leave-one-out within the task's train set).
Evaluation uses the actual held-out test pairs.

Uses mdlARC-style tokenization: colors 0-9, <start>=10, <next_line>=11,
<io_sep>=12, <end>=13.

Usage:
  uv run python exp_arc_real.py                    # all variants
  uv run python exp_arc_real.py standard            # single variant
  uv run python exp_arc_real.py eigen_bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import time
import sys
import random
from itertools import combinations

# ── Config ───────────────────────────────────────────────────────────────────

MAX_SEQ_LEN = 512   # filter to tasks that fit
VOCAB = 14           # 10 colors + 4 special tokens
START = 10
NEXT_LINE = 11
IO_SEP = 12
END = 13

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
BATCH = 16           # smaller batch — variable-length real data
N_STEPS = 6000
LR = 3e-4
EVAL_EVERY = 500
ARC_DIR = "data/ARC-AGI/data/training"


# ── Data loading ─────────────────────────────────────────────────────────────

def grid_to_tokens(grid):
    """Flatten a 2D grid to tokens with <next_line> delimiters."""
    tokens = []
    for row in grid:
        tokens.extend(row)
        tokens.append(NEXT_LINE)
    return tokens


def pair_to_tokens(example):
    """Encode an input/output pair: <start> input <io_sep> output <end>."""
    in_tokens = grid_to_tokens(example['input'])
    out_tokens = grid_to_tokens(example['output'])
    return [START] + in_tokens + [IO_SEP] + out_tokens + [END]


def pair_token_length(example):
    """Compute token length of a pair without materializing."""
    ir = len(example['input'])
    ic = len(example['input'][0])
    orr = len(example['output'])
    oc = len(example['output'][0])
    return 1 + ir * (ic + 1) + 1 + orr * (oc + 1) + 1


def load_tasks(arc_dir, max_seq_len):
    """Load ARC tasks that fit within max_seq_len."""
    tasks = []
    for fname in sorted(os.listdir(arc_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(arc_dir, fname)) as f:
            task = json.load(f)

        # Check if all pairs fit
        all_examples = task['train'] + task['test']
        max_pair = max(pair_token_length(ex) for ex in all_examples)
        n_pairs = len(task['train']) + 1  # demos + 1 test
        total_seq = n_pairs * max_pair

        if total_seq <= max_seq_len and len(task['train']) >= 2:
            tasks.append({
                'name': fname.replace('.json', ''),
                'train': task['train'],
                'test': task['test'],
                'max_pair_len': max_pair,
            })

    return tasks


def encode_sequence(demo_examples, test_example, max_seq_len):
    """Encode demos + test into a padded sequence.

    Returns:
        tokens: (max_seq_len,) padded token ids
        targets: (max_seq_len,) with -100 for non-prediction positions
        length: actual sequence length
    """
    tokens = []
    for ex in demo_examples:
        tokens.extend(pair_to_tokens(ex))

    # Test pair: encode input, separator, then output (to be predicted)
    test_in_tokens = grid_to_tokens(test_example['input'])
    test_out_tokens = grid_to_tokens(test_example['output'])
    test_pair = [START] + test_in_tokens + [IO_SEP] + test_out_tokens + [END]
    test_start = len(tokens)
    tokens.extend(test_pair)

    length = len(tokens)
    if length > max_seq_len:
        return None, None, 0

    # Targets: only predict test output tokens
    targets = [-100] * max_seq_len
    sep_pos = test_start + 1 + len(test_in_tokens)  # position of IO_SEP
    for i in range(sep_pos + 1, length):
        targets[i] = tokens[i]

    # Pad tokens
    padded = tokens + [0] * (max_seq_len - length)

    return padded, targets, length


class ARCDataset:
    """Wraps ARC tasks for training and evaluation."""

    def __init__(self, tasks):
        self.tasks = tasks

    def sample_train_batch(self, batch_size):
        """Sample a training batch using leave-one-out within each task."""
        all_tokens = []
        all_targets = []

        for _ in range(batch_size):
            task = random.choice(self.tasks)
            train_pairs = task['train']

            # Leave-one-out: pick one train pair as test, rest as demos
            test_idx = random.randint(0, len(train_pairs) - 1)
            demos = [p for i, p in enumerate(train_pairs) if i != test_idx]
            test_pair = train_pairs[test_idx]

            tokens, targets, length = encode_sequence(demos, test_pair, MAX_SEQ_LEN)
            if tokens is None:
                # Fallback: just use first 2 demos and last as test
                demos = train_pairs[:2]
                test_pair = train_pairs[-1]
                tokens, targets, length = encode_sequence(demos, test_pair, MAX_SEQ_LEN)
                if tokens is None:
                    # Skip this sample, use a simpler task
                    task = min(self.tasks, key=lambda t: t['max_pair_len'])
                    demos = task['train'][:2]
                    test_pair = task['train'][-1]
                    tokens, targets, length = encode_sequence(demos, test_pair, MAX_SEQ_LEN)

            all_tokens.append(tokens)
            all_targets.append(targets)

        return (
            torch.tensor(all_tokens, dtype=torch.long),
            torch.tensor(all_targets, dtype=torch.long),
        )

    def get_eval_set(self):
        """Get evaluation set: use actual test pairs."""
        eval_items = []
        for task in self.tasks:
            for test_ex in task['test']:
                tokens, targets, length = encode_sequence(
                    task['train'], test_ex, MAX_SEQ_LEN
                )
                if tokens is not None:
                    eval_items.append((tokens, targets, task['name']))
        return eval_items


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

        outer = Jw.unsqueeze(-1) * Jw.unsqueeze(-2)
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


class ARCTransformer(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, max_seq_len, attn_type):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
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


# ── Training & Evaluation ────────────────────────────────────────────────────

def evaluate_token_acc(model, dataset, device):
    """Token-level accuracy on the held-out test pairs."""
    model.eval()
    eval_items = dataset.get_eval_set()
    correct = total = 0

    with torch.no_grad():
        # Process in small batches
        for i in range(0, len(eval_items), 16):
            batch_items = eval_items[i:i+16]
            tokens = torch.tensor([t[0] for t in batch_items], dtype=torch.long, device=device)
            targets = torch.tensor([t[1] for t in batch_items], dtype=torch.long, device=device)

            logits = model(tokens)
            preds = logits.argmax(dim=-1)
            mask = targets != -100
            total += mask.sum().item()
            correct += ((preds == targets) & mask).sum().item()

    return correct / max(total, 1)


def evaluate_grid_acc(model, dataset, device):
    """Grid-level accuracy: all output tokens correct for a test pair."""
    model.eval()
    eval_items = dataset.get_eval_set()
    perfect = total = 0

    with torch.no_grad():
        for tokens, targets, name in eval_items:
            t = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt = torch.tensor([targets], dtype=torch.long, device=device)
            logits = model(t)
            preds = logits.argmax(dim=-1)
            mask = tgt != -100
            m = mask[0]
            if m.any():
                total += 1
                if (preds[0][m] == tgt[0][m]).all():
                    perfect += 1

    return perfect / max(total, 1), total


def train_variant(attn_type, dataset, device):
    model = ARCTransformer(VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, attn_type).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {attn_type}: {params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    history = []
    t0 = time.time()

    for step in range(1, N_STEPS + 1):
        model.train()
        x, y = dataset.sample_train_batch(BATCH)
        x, y = x.to(device), y.to(device)
        logits = model(x).view(-1, VOCAB)
        loss = F.cross_entropy(logits, y.view(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            tok_acc = evaluate_token_acc(model, dataset, device)
            grid_acc, n_eval = evaluate_grid_acc(model, dataset, device)
            history.append((step, tok_acc, grid_acc))
            elapsed = time.time() - t0
            print(f"    step {step:4d}: loss={loss.item():.3f}  tok={tok_acc:.3f}  grid={grid_acc:.3f} ({n_eval} tasks)  ({elapsed:.1f}s)")

    final_tok = evaluate_token_acc(model, dataset, device)
    final_grid, n_eval = evaluate_grid_acc(model, dataset, device)
    total_time = time.time() - t0
    return history, final_tok, final_grid, total_time, n_eval


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    random.seed(42)

    print(f"Device: {device}")
    print(f"Loading ARC tasks from {ARC_DIR}...")

    tasks = load_tasks(ARC_DIR, MAX_SEQ_LEN)
    print(f"Loaded {len(tasks)} tasks (max_seq_len={MAX_SEQ_LEN})")

    dataset = ARCDataset(tasks)
    eval_items = dataset.get_eval_set()
    print(f"Eval set: {len(eval_items)} test pairs")
    print(f"Config: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")

    variants = sys.argv[1:] if len(sys.argv) > 1 else ["standard", "eigen_bias"]
    results = {}

    for v in variants:
        if v not in ATTN_MAP:
            print(f"Unknown: {v}. Choose from: {list(ATTN_MAP.keys())}")
            continue
        print(f"\n{'='*60}")
        print(f"  VARIANT: {v}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        random.seed(42)
        hist, final_tok, final_grid, elapsed, n_eval = train_variant(v, dataset, device)
        results[v] = (hist, final_tok, final_grid, elapsed)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY — Real ARC Tasks ({len(tasks)} tasks, {len(eval_items)} eval pairs)")
        print(f"{'='*60}")
        print(f"  {'Variant':15s} {'Tok Acc':>10s} {'Grid Acc':>10s} {'Time':>8s}")
        print(f"  {'-'*50}")
        for v, (hist, ft, fg, el) in results.items():
            print(f"  {v:15s} {ft:10.3f} {fg:10.3f} {el:7.1f}s")

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
