"""
exp_arc_real.py — Test Gram bias on REAL ARC-AGI tasks.

Matches mdlARC training setup:
  - 3D RoPE positional encoding (x=col, y=row, z=section)
  - Per-task embeddings
  - 8x dihedral augmentation + color permutation
  - RMSNorm + SiLU gated FFN
  - AdamW (betas 0.9, 0.95) with warmup + WSD decay
  - Packed variable-length sequences (no padding waste)

Trains on ARC training set, evaluates on ARC evaluation set.
Autoregressive generation for proper eval (no teacher forcing).

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
import math
import random
import numpy as np
from itertools import combinations

# ── Config ───────────────────────────────────────────────────────────────────

MAX_SEQ_LEN = 512
VOCAB = 14           # 10 colors + 4 special tokens
START = 10
NEXT_LINE = 11
IO_SEP = 12
END = 13

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
BATCH = 8
N_STEPS = 3000
LR = 3e-4
SS_START = 0.0       # scheduled sampling: start ratio (0 = full teacher forcing)
SS_END = 0.5         # scheduled sampling: end ratio (0.5 = 50% model predictions)
WARMUP_PCT = 0.02
WSD_DECAY_START = 0.80
LR_FLOOR = 0.01
EVAL_EVERY = 500
MAX_TIME = 600
MAX_EVAL = 10
ARC_TRAIN_DIR = "data/ARC-AGI/data/training"
ARC_EVAL_DIR = "data/ARC-AGI/data/evaluation"
MIN_SEQ_LEN = 0

# ── Dihedral transforms (8 symmetries of the square) ────────────────────────

def dihedral_transform(grid, idx):
    g = np.array(grid)
    if idx == 0:   return g
    elif idx == 1: return np.rot90(g, 1)
    elif idx == 2: return np.rot90(g, 2)
    elif idx == 3: return np.rot90(g, 3)
    elif idx == 4: return np.fliplr(g)
    elif idx == 5: return np.flipud(g)
    elif idx == 6: return g.T
    elif idx == 7: return np.rot90(g, 2).T
    return g


def transform_example(example, dih_idx):
    return {
        'input': dihedral_transform(example['input'], dih_idx).tolist(),
        'output': dihedral_transform(example['output'], dih_idx).tolist(),
    }


# ── Color permutation augmentation ──────────────────────────────────────────

def random_color_permutation():
    """Return a mapping array for color permutation (colors 0-9)."""
    perm = list(range(10))
    random.shuffle(perm)
    return perm


def apply_color_perm_to_tokens(tokens, perm):
    """Apply color permutation to token list. Only affects color tokens (0-9)."""
    return [perm[t] if 0 <= t < 10 else t for t in tokens]


# ── 3D position computation ─────────────────────────────────────────────────

def compute_3d_positions(tokens):
    """Compute (x, y, z) positions for each token.
    x=column, y=row, z=section (0=special, 1=input, 2=sep, 3=output, 4=end).
    """
    positions = []
    x, y, z = 0, 0, 0
    for tok in tokens:
        if tok == START:
            x, y = 0, 0
            z = 1
            positions.append((0, 0, 0))
        elif tok == IO_SEP:
            x, y = 0, 0
            positions.append((0, 0, 2))
            z = 3
        elif tok == END:
            positions.append((0, 0, 4))
            x, y = 0, 0
        elif tok == NEXT_LINE:
            positions.append((x, y, z))
            y += 1
            x = 0
        else:
            positions.append((x, y, z))
            x += 1
    return positions


# ── Data loading ─────────────────────────────────────────────────────────────

def grid_to_tokens(grid):
    tokens = []
    for row in grid:
        tokens.extend(int(c) for c in row)
        tokens.append(NEXT_LINE)
    return tokens


def pair_to_tokens(example):
    in_tokens = grid_to_tokens(example['input'])
    out_tokens = grid_to_tokens(example['output'])
    return [START] + in_tokens + [IO_SEP] + out_tokens + [END]


def pair_token_length(example):
    ir = len(example['input'])
    ic = len(example['input'][0])
    orr = len(example['output'])
    oc = len(example['output'][0])
    return 1 + ir * (ic + 1) + 1 + orr * (oc + 1) + 1


def load_tasks(arc_dir, max_seq_len, min_seq_len=MIN_SEQ_LEN):
    tasks = []
    for fname in sorted(os.listdir(arc_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(arc_dir, fname)) as f:
            task = json.load(f)

        all_examples = task['train'] + task['test']
        max_pair = max(pair_token_length(ex) for ex in all_examples)
        n_pairs = len(task['train']) + 1
        total_seq = n_pairs * max_pair

        if min_seq_len <= total_seq <= max_seq_len and len(task['train']) >= 2:
            tasks.append({
                'name': fname.replace('.json', ''),
                'train': task['train'],
                'test': task['test'],
                'max_pair_len': max_pair,
                'total_seq': total_seq,
            })
    return tasks


def encode_sequence_packed(demo_examples, test_example):
    """Encode demos + test into a token list with 3D positions (no padding).

    Returns: tokens, targets, pos_3d, or None if invalid.
    """
    tokens = []
    for ex in demo_examples:
        tokens.extend(pair_to_tokens(ex))

    test_in_tokens = grid_to_tokens(test_example['input'])
    test_out_tokens = grid_to_tokens(test_example['output'])
    test_pair = [START] + test_in_tokens + [IO_SEP] + test_out_tokens + [END]
    test_start = len(tokens)
    tokens.extend(test_pair)

    length = len(tokens)

    # Targets: only predict test output tokens
    targets = [-100] * length
    sep_pos = test_start + 1 + len(test_in_tokens)
    for i in range(sep_pos + 1, length):
        targets[i] = tokens[i]

    pos_3d = compute_3d_positions(tokens)
    return tokens, targets, pos_3d, length


class ARCDataset:
    """ARC dataset with dihedral + color augmentation and per-task IDs."""

    def __init__(self, train_tasks, eval_tasks):
        self.train_tasks = train_tasks
        self.test_tasks = eval_tasks
        self.task_to_id = {t['name']: i for i, t in enumerate(train_tasks)}
        self.n_tasks = len(train_tasks)
        print(f"  Train tasks: {len(self.train_tasks)}, Eval tasks: {len(self.test_tasks)}")

    def sample_train_batch(self, batch_size):
        """Sample packed batch with dihedral + color augmentation."""
        all_tokens = []
        all_targets = []
        all_pos3d = []
        all_task_ids = []
        seq_lens = []

        for _ in range(batch_size):
            task = random.choice(self.train_tasks)
            task_id = self.task_to_id[task['name']]
            train_pairs = task['train']

            # Random dihedral + color augmentation
            dih_idx = random.randint(0, 7)
            color_perm = random_color_permutation()

            test_idx = random.randint(0, len(train_pairs) - 1)
            demos = [transform_example(p, dih_idx)
                     for i, p in enumerate(train_pairs) if i != test_idx]
            test_pair = transform_example(train_pairs[test_idx], dih_idx)

            tokens, targets, pos3d, length = encode_sequence_packed(
                demos, test_pair)

            if length > MAX_SEQ_LEN:
                # Fallback to smallest task
                task = min(self.train_tasks, key=lambda t: t['max_pair_len'])
                task_id = self.task_to_id[task['name']]
                demos = [transform_example(p, dih_idx)
                         for p in task['train'][:2]]
                test_pair = transform_example(task['train'][-1], dih_idx)
                tokens, targets, pos3d, length = encode_sequence_packed(
                    demos, test_pair)

            # Apply color permutation
            tokens = apply_color_perm_to_tokens(tokens, color_perm)
            targets = [color_perm[t] if 0 <= t < 10 else t for t in targets]

            all_tokens.append(tokens)
            all_targets.append(targets)
            all_pos3d.append(pos3d)
            all_task_ids.append(task_id)
            seq_lens.append(length)

        # Pack: pad to max length in this batch (not global MAX_SEQ_LEN)
        max_len = max(seq_lens)
        padded_tokens = []
        padded_targets = []
        padded_pos3d = []
        for i in range(batch_size):
            pad_n = max_len - seq_lens[i]
            padded_tokens.append(all_tokens[i] + [0] * pad_n)
            padded_targets.append(all_targets[i] + [-100] * pad_n)
            padded_pos3d.append(all_pos3d[i] + [(0, 0, 0)] * pad_n)

        return (
            torch.tensor(padded_tokens, dtype=torch.long),
            torch.tensor(padded_targets, dtype=torch.long),
            torch.tensor(padded_pos3d, dtype=torch.long),
            torch.tensor(all_task_ids, dtype=torch.long),
        )

    def get_eval_set(self):
        """Eval on unseen tasks (no augmentation, task_id=-1)."""
        eval_items = []
        for task in self.test_tasks:
            for test_ex in task['test']:
                tokens, targets, pos3d, length = encode_sequence_packed(
                    task['train'], test_ex)
                if length <= MAX_SEQ_LEN:
                    eval_items.append((tokens, targets, pos3d, task['name']))
                    if len(eval_items) >= MAX_EVAL:
                        return eval_items
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


# ── 3D Rotary Positional Encoding ───────────────────────────────────────────

class RotaryEmbedding3D(nn.Module):
    """3D RoPE: splits head dim across x, y, z axes with sinusoidal encoding."""

    def __init__(self, d_head, max_x=32, max_y=32, max_z=5):
        super().__init__()
        # Split head dim into 3 parts for x, y, z
        self.d_x = d_head // 3
        self.d_y = d_head // 3
        self.d_z = d_head - 2 * (d_head // 3)  # remainder goes to z

        # Precompute frequency bases
        self.register_buffer('freq_x', self._freqs(self.d_x, max_x))
        self.register_buffer('freq_y', self._freqs(self.d_y, max_y))
        self.register_buffer('freq_z', self._freqs(self.d_z, max_z))

    def _freqs(self, dim, max_pos):
        """Precompute cos/sin tables: (max_pos, dim)."""
        half = dim // 2
        if half == 0:
            return None
        theta = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
        positions = torch.arange(max_pos, dtype=torch.float32)
        angles = positions.unsqueeze(1) * theta.unsqueeze(0)  # (max_pos, half)
        return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (max_pos, half, 2)

    def _apply_rope(self, x, freqs, positions):
        """Apply rotary encoding to x using positions as lookup."""
        if freqs is None or x.shape[-1] == 0:
            return x
        half = x.shape[-1] // 2
        if half == 0:
            return x
        # positions: (B, T) → lookup freqs: (B, T, half, 2)
        pos_clamped = positions.clamp(0, freqs.shape[0] - 1)
        cs = freqs[pos_clamped]  # (B, T, half, 2)
        cos_t = cs[..., 0]  # (B, T, half)
        sin_t = cs[..., 1]

        x1 = x[..., :half]
        x2 = x[..., half:2*half]
        out1 = x1 * cos_t - x2 * sin_t
        out2 = x1 * sin_t + x2 * cos_t
        if x.shape[-1] > 2 * half:
            return torch.cat([out1, out2, x[..., 2*half:]], dim=-1)
        return torch.cat([out1, out2], dim=-1)

    def forward(self, q, k, pos_3d):
        """Apply 3D RoPE to q and k.
        q, k: (B, H, T, d_head)
        pos_3d: (B, T, 3)
        """
        B, H, T, D = q.shape
        px = pos_3d[:, :, 0]  # (B, T)
        py = pos_3d[:, :, 1]
        pz = pos_3d[:, :, 2]

        # Split q/k along head dim
        qx, qy, qz = q[..., :self.d_x], q[..., self.d_x:self.d_x+self.d_y], q[..., self.d_x+self.d_y:]
        kx, ky, kz = k[..., :self.d_x], k[..., self.d_x:self.d_x+self.d_y], k[..., self.d_x+self.d_y:]

        # Apply RoPE per axis (broadcast over heads)
        px_h = px.unsqueeze(1).expand(-1, H, -1)
        py_h = py.unsqueeze(1).expand(-1, H, -1)
        pz_h = pz.unsqueeze(1).expand(-1, H, -1)

        qx = self._apply_rope(qx, self.freq_x, px_h)
        kx = self._apply_rope(kx, self.freq_x, px_h)
        qy = self._apply_rope(qy, self.freq_y, py_h)
        ky = self._apply_rope(ky, self.freq_y, py_h)
        qz = self._apply_rope(qz, self.freq_z, pz_h)
        kz = self._apply_rope(kz, self.freq_z, pz_h)

        q_out = torch.cat([qx, qy, qz], dim=-1)
        k_out = torch.cat([kx, ky, kz], dim=-1)
        return q_out, k_out


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ── Attention variants ───────────────────────────────────────────────────────

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding3D(self.d_head)

    def forward(self, x, pos_3d=None, token_ids=None):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if pos_3d is not None:
            q, k = self.rope(q, k, pos_3d)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class EigenBiasAttention(nn.Module):
    """Standard attention + per-pair Gram bias.

    Resets the Gram at each <start> token so each demo/test pair gets its
    own geometric summary. For the test pair, the Gram starts fresh —
    the geometric signal comes from within the test pair itself, while
    cross-pair pattern matching is handled by standard Q·K attention.
    """
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
        self.rope = RotaryEmbedding3D(self.d_head)
        self.register_buffer('J6', _J6)

    def _pair_cumsum(self, outer, token_ids):
        """Cumulative sum that resets at each <start> token — vectorized.

        Uses segment IDs to mask out cross-pair contributions.
        outer: (B, H, T, 6, 6), token_ids: (B, T)
        Returns: (B, H, T, 6, 6) — per-pair causal Gram
        """
        B, H, T, d, _ = outer.shape

        # Assign each position a pair/segment ID by counting <start> tokens
        is_start = (token_ids == START).float()  # (B, T)
        seg_id = is_start.cumsum(dim=1)  # (B, T) — increments at each <start>

        # Build (B, T, T) mask: same_pair[t, s] = (seg_id[t] == seg_id[s]) & (s < t)
        seg_q = seg_id.unsqueeze(2)  # (B, T, 1)
        seg_k = seg_id.unsqueeze(1)  # (B, 1, T)
        same_pair = (seg_q == seg_k)  # (B, T, T)

        # Causal: s < t
        causal = torch.tril(torch.ones(T, T, device=outer.device, dtype=torch.bool), diagonal=-1)
        pair_causal = same_pair & causal.unsqueeze(0)  # (B, T, T)

        # M_causal[t] = sum of outer[s] for s in same pair and s < t
        # outer: (B, H, T, d, d) → (B, 1, T, d*d) for matmul with mask
        outer_flat = outer.reshape(B, H, T, d * d)  # (B, H, T, d²)

        # pair_causal: (B, T, T) → (B, 1, T, T) broadcast over heads
        mask_float = pair_causal.unsqueeze(1).float()  # (B, 1, T, T)

        # M_causal[t] = mask[t, :] @ outer_flat → (B, H, T, d²)
        M_flat = torch.einsum('bhts,bhsd->bhtd', mask_float.expand(-1, H, -1, -1), outer_flat)
        return M_flat.reshape(B, H, T, d, d)

    def forward(self, x, pos_3d=None, token_ids=None):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, kk, v = qkv[0], qkv[1], qkv[2]

        if pos_3d is not None:
            q, kk = self.rope(q, kk, pos_3d)

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

        outer = Jw.unsqueeze(-1) * Jw.unsqueeze(-2)  # (B, H, T, 6, 6)

        if token_ids is not None:
            M_causal = self._pair_cumsum(outer, token_ids)
        else:
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

class SiLUGatedFFN(nn.Module):
    """SiLU-gated feed-forward (matches mdlARC)."""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w_gate = nn.Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_type):
        super().__init__()
        self.attn = ATTN_MAP[attn_type](d_model, n_heads)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SiLUGatedFFN(d_model)

    def forward(self, x, pos_3d=None, token_ids=None):
        x = x + self.attn(self.ln1(x), pos_3d, token_ids)
        x = x + self.ffn(self.ln2(x))
        return x


class ARCTransformer(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, attn_type, n_tasks=0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.n_tasks = n_tasks
        if n_tasks > 0:
            self.task_emb = nn.Embedding(n_tasks, d_model)
        else:
            self.task_emb = None
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, attn_type) for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, pos_3d=None, task_ids=None, use_pair_gram=False):
        B, T = idx.shape
        x = self.tok_emb(idx)

        if task_ids is not None and self.task_emb is not None:
            valid = task_ids >= 0
            if valid.any():
                te = torch.zeros(B, x.shape[-1], device=idx.device)
                te[valid] = self.task_emb(task_ids[valid])
                x = x + te.unsqueeze(1)

        token_ids = idx if use_pair_gram else None
        for block in self.blocks:
            x = block(x, pos_3d, token_ids)
        return self.head(self.ln_f(x))


# ── LR Schedule: warmup + WSD decay ─────────────────────────────────────────

def get_lr(step, total_steps, base_lr):
    """Linear warmup, constant, then cosine decay to floor."""
    warmup_steps = int(total_steps * WARMUP_PCT)
    decay_start = int(total_steps * WSD_DECAY_START)

    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    elif step < decay_start:
        return base_lr
    else:
        progress = (step - decay_start) / max(total_steps - decay_start, 1)
        return base_lr * (LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress)))


# ── Training & Evaluation ────────────────────────────────────────────────────

N_SAMPLES = 2        # best-of-N attempts per eval task
SAMPLE_TEMP = 0.8    # temperature for sampling (1.0 = full, lower = more greedy)

def generate_one(model, prompt, prompt_pos, task_ids, n_to_gen, device, greedy=False):
    """Generate one sequence autoregressively. Returns (tokens, log_prob)."""
    seq = prompt.clone()
    seq_pos = prompt_pos.clone()
    generated = []
    total_logprob = 0.0
    cur_x, cur_y, cur_z = 0, 0, 3

    for _ in range(n_to_gen):
        logits = model(seq, seq_pos, task_ids)
        logits_last = logits[0, -1]  # (VOCAB,)

        if greedy:
            next_tok = logits_last.argmax().item()
        else:
            probs = F.softmax(logits_last / SAMPLE_TEMP, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()

        # Accumulate log probability for scoring
        log_probs = F.log_softmax(logits_last, dim=-1)
        total_logprob += log_probs[next_tok].item()

        generated.append(next_tok)

        if next_tok == NEXT_LINE:
            pos = (cur_x, cur_y, cur_z)
            cur_y += 1
            cur_x = 0
        elif next_tok == END:
            pos = (0, 0, 4)
        else:
            pos = (cur_x, cur_y, cur_z)
            cur_x += 1

        next_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        pos_tensor = torch.tensor([[[pos[0], pos[1], pos[2]]]], dtype=torch.long, device=device)
        seq = torch.cat([seq, next_tensor], dim=1)
        seq_pos = torch.cat([seq_pos, pos_tensor], dim=1)

        if next_tok == END:
            break

    return generated, total_logprob


def evaluate_grid_acc(model, dataset, device):
    """Best-of-N autoregressive generation eval."""
    model.eval()
    eval_items = dataset.get_eval_set()
    perfect = total = 0
    tok_correct = tok_total = 0

    with torch.no_grad():
        for tokens, targets, pos3d, name in eval_items:
            tgt = torch.tensor(targets, dtype=torch.long)
            mask = tgt != -100
            if not mask.any():
                continue

            gen_start = mask.nonzero(as_tuple=True)[0][0].item()
            n_to_gen = mask.sum().item()
            gt = [targets[i] for i in range(len(targets)) if targets[i] != -100]

            prompt = torch.tensor([tokens[:gen_start]], dtype=torch.long, device=device)
            prompt_pos = torch.tensor([pos3d[:gen_start]], dtype=torch.long, device=device)
            task_ids = torch.tensor([-1], dtype=torch.long, device=device)

            # Generate N candidates, keep best by log-probability
            best_gen = None
            best_logprob = float('-inf')

            # Always include one greedy attempt
            gen, lp = generate_one(model, prompt, prompt_pos, task_ids,
                                   n_to_gen, device, greedy=True)
            if lp > best_logprob:
                best_gen, best_logprob = gen, lp

            # N-1 sampled attempts
            for _ in range(N_SAMPLES - 1):
                gen, lp = generate_one(model, prompt, prompt_pos, task_ids,
                                       n_to_gen, device, greedy=False)
                if lp > best_logprob:
                    best_gen, best_logprob = gen, lp

            total += 1
            n_compare = min(len(best_gen), len(gt))
            matches = sum(1 for i in range(n_compare) if best_gen[i] == gt[i])
            tok_correct += matches
            tok_total += len(gt)
            if best_gen == gt:
                perfect += 1

    tok_acc = tok_correct / max(tok_total, 1)
    grid_acc = perfect / max(total, 1)
    return tok_acc, grid_acc, total, tok_correct, tok_total, perfect


def train_variant(attn_type, dataset, device):
    model = ARCTransformer(
        VOCAB, D_MODEL, N_HEADS, N_LAYERS,
        attn_type, n_tasks=dataset.n_tasks
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {attn_type}: {params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.01)
    history = []
    t0 = time.time()

    for step in range(1, N_STEPS + 1):
        # LR schedule
        lr = get_lr(step, N_STEPS, LR)
        for pg in opt.param_groups:
            pg['lr'] = lr

        model.train()
        x, y, pos3d, task_ids = dataset.sample_train_batch(BATCH)
        x, y = x.to(device), y.to(device)
        pos3d = pos3d.to(device)
        task_ids = task_ids.to(device)

        # Scheduled sampling: linearly ramp up from SS_START to SS_END
        ss_ratio = SS_START + (SS_END - SS_START) * step / N_STEPS

        if ss_ratio > 0:
            # For output positions (where y != -100), sometimes replace
            # the input token with the model's own prediction
            with torch.no_grad():
                pred_logits = model(x, pos3d, task_ids, use_pair_gram=True)
                pred_tokens = pred_logits.argmax(dim=-1)  # (B, T)

            # Create mask: positions where we COULD substitute (output tokens)
            # We substitute at position t, which means x[t+1] gets the model's
            # prediction of x[t] instead of ground truth
            output_mask = (y != -100)  # (B, T)
            # Random coin flip per position
            coin = torch.rand_like(output_mask.float()) < ss_ratio
            substitute = output_mask & coin  # (B, T)

            # Shift: if we substitute at position t, replace x[t+1] with pred[t]
            if substitute.any():
                x_mixed = x.clone()
                # For positions where substitute[t] is True, set x[t+1] = pred[t]
                sub_shifted = substitute[:, :-1]  # (B, T-1)
                x_mixed[:, 1:][sub_shifted] = pred_tokens[:, :-1][sub_shifted]
                x = x_mixed

        logits = model(x, pos3d, task_ids, use_pair_gram=True).view(-1, VOCAB)
        loss = F.cross_entropy(logits, y.view(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            tok_acc, grid_acc, n_eval, tc, tt, gc = evaluate_grid_acc(model, dataset, device)
            history.append((step, tok_acc, grid_acc))
            elapsed = time.time() - t0
            print(f"    step {step:4d}: loss={loss.item():.3f}  tok={tc}/{tt}({tok_acc:.3f})  grid={gc}/{n_eval}  ({elapsed:.1f}s)")
            if elapsed > MAX_TIME:
                print(f"    Time limit ({MAX_TIME}s) reached, stopping early")
                break

    final_tok, final_grid, n_eval, tc, tt, gc = evaluate_grid_acc(model, dataset, device)
    total_time = time.time() - t0
    return history, final_tok, final_grid, total_time, n_eval


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    print(f"Device: {device}")

    train_tasks = load_tasks(ARC_TRAIN_DIR, MAX_SEQ_LEN)
    eval_tasks = load_tasks(ARC_EVAL_DIR, MAX_SEQ_LEN)
    print(f"Train: {len(train_tasks)} tasks, Eval: {len(eval_tasks)} tasks (seq {MIN_SEQ_LEN}-{MAX_SEQ_LEN})")

    dataset = ARCDataset(train_tasks, eval_tasks)
    eval_items = dataset.get_eval_set()
    print(f"Eval pairs: {len(eval_items)}")
    print(f"Config: d={D_MODEL}, h={N_HEADS}, L={N_LAYERS}, steps={N_STEPS}")
    print(f"Features: 3D RoPE, per-task emb, 8x dihedral, color perm, RMSNorm, SiLU gated FFN")
    print(f"Optimizer: AdamW (0.9, 0.95), warmup+WSD, packed batches")

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
        np.random.seed(42)
        hist, final_tok, final_grid, elapsed, n_eval = train_variant(v, dataset, device)
        results[v] = (hist, final_tok, final_grid, elapsed)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY — Real ARC Tasks (train={len(train_tasks)}, eval={len(eval_tasks)}, {len(eval_items)} eval pairs)")
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
