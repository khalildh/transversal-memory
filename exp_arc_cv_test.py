"""
exp_arc_cv_test.py — Cross-validate embeddings on training pairs + fast scoring.

For each embedding, score training pairs in leave-one-out fashion.
The embedding whose transversals best predict held-out pairs gets used.
Then score test candidates with the selected embeddings.

Uses MPS-accelerated tensor generation (no Python cartesian loop).
"""

import json
import os
import numpy as np
import torch
import time
from transversal_memory import P3Memory

N_COLORS = 10
J6 = np.array([[0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
                [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0]], dtype=np.float32)


def make_line(sv, tv, W1, W2):
    combined = np.concatenate([sv, tv])
    p1 = W1 @ combined; p2 = W2 @ combined
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    L = np.array([p1[i]*p2[j]-p1[j]*p2[i] for i,j in pairs], dtype=np.float32)
    n = np.linalg.norm(L)
    return L/n if n > 1e-10 else None


def compute_trans(lines, n_trans=200, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    if len(lines) < 4: return []
    trans = []; att = 0
    while len(trans) < n_trans and att < n_trans*10:
        att += 1; idx = rng.choice(len(lines), size=4, replace=False)
        mem = P3Memory(); mem.store([lines[idx[i]] for i in range(3)])
        for T, res in mem.query_generative(lines[idx[3]]):
            n = np.linalg.norm(T)
            if n > 1e-10 and res < 1e-6: trans.append(T/n)
    return trans


# ── Embeddings ───────────────────────────────────────────────────────────────

def emb_hist_color(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    diff = np.array([(np.sum(out==i)-np.sum(inp==i))/max(inp.size,1)
                     for i in range(N_COLORS)], dtype=np.float32)
    return np.concatenate([in_oh, out_oh, diff])

def emb_color_only(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    return np.concatenate([in_oh, out_oh])

def emb_pos_color(r, c, in_c, out_c, inp, out, H, W):
    pos = np.array([r/max(H-1,1), c/max(W-1,1)], dtype=np.float32)
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    return np.concatenate([pos, in_oh, out_oh])

def emb_all(r, c, in_c, out_c, inp, out, H, W):
    pos = np.array([r/max(H-1,1), c/max(W-1,1)], dtype=np.float32)
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    ih = np.array([np.sum(inp==i)/max(inp.size,1) for i in range(N_COLORS)], dtype=np.float32)
    oh = np.array([np.sum(out==i)/max(out.size,1) for i in range(N_COLORS)], dtype=np.float32)
    return np.concatenate([pos, in_oh, out_oh, ih, oh])

def emb_row_features(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    in_row_hist = np.array([np.sum(inp[r]==i) for i in range(N_COLORS)], dtype=np.float32) / W
    in_row_uniform = np.float32(len(set(inp[r].flatten())) == 1)
    in_row_nunique = np.float32(len(set(inp[r].flatten())) / max(W, 1))
    out_row_hist = np.array([np.sum(out[r]==i) for i in range(N_COLORS)], dtype=np.float32) / W
    out_row_uniform = np.float32(len(set(out[r].flatten())) == 1)
    out_row_nunique = np.float32(len(set(out[r].flatten())) / max(W, 1))
    return np.concatenate([in_oh, out_oh, in_row_hist, [in_row_uniform, in_row_nunique],
                           out_row_hist, [out_row_uniform, out_row_nunique]])

def emb_col_features(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    in_col_hist = np.array([np.sum(inp[:,c]==i) for i in range(N_COLORS)], dtype=np.float32) / H
    in_col_uniform = np.float32(len(set(inp[:,c].flatten())) == 1)
    out_col_hist = np.array([np.sum(out[:,c]==i) for i in range(N_COLORS)], dtype=np.float32) / H
    out_col_uniform = np.float32(len(set(out[:,c].flatten())) == 1)
    return np.concatenate([in_oh, out_oh, in_col_hist, [in_col_uniform],
                           out_col_hist, [out_col_uniform]])

def emb_color_count(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    in_count = np.float32(np.sum(inp == in_c) / max(inp.size, 1))
    out_count = np.float32(np.sum(out == out_c) / max(out.size, 1))
    in_mode = np.float32(in_c == np.argmax(np.bincount(inp.flatten(), minlength=N_COLORS)))
    out_mode = np.float32(out_c == np.argmax(np.bincount(out.flatten(), minlength=N_COLORS)))
    return np.concatenate([in_oh, out_oh, [in_count, out_count, in_mode, out_mode]])

def emb_diagonal(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    main_diag = np.float32((r - c + max(H,W)) / (2*max(H,W)))
    anti_diag = np.float32((r + c) / (H + W - 2 + 1e-6))
    on_main = np.float32(r == c)
    on_anti = np.float32(r + c == min(H,W) - 1)
    pos = np.array([r/max(H-1,1), c/max(W-1,1)], dtype=np.float32)
    return np.concatenate([in_oh, out_oh, pos, [main_diag, anti_diag, on_main, on_anti]])


ALL_EMBEDDINGS = [
    ('hist_color', emb_hist_color, 30),
    ('color_only', emb_color_only, 20),
    ('pos_color', emb_pos_color, 22),
    ('all', emb_all, 42),
    ('row_feat', emb_row_features, 44),
    ('col_feat', emb_col_features, 42),
    ('color_count', emb_color_count, 24),
    ('diagonal', emb_diagonal, 26),
]


# ── Cross-validation on training pairs ───────────────────────────────────────

def cv_score_embedding(task, name, emb_fn, dim, adj_pairs, H, W):
    """Leave-one-out CV: for each training pair, use the others' transversals
    to score the held-out pair. Return mean score (lower = more incident = better)."""
    rng_proj = np.random.RandomState(hash(name) % 2**31)
    W1 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1
    W2 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1

    pairs = task['train']
    total_score = 0.0
    n_eval = 0

    for hold_idx in range(len(pairs)):
        # Transversals from non-held-out pairs
        trans = []
        for i, pair in enumerate(pairs):
            if i == hold_idx:
                continue
            inp, out = np.array(pair['input']), np.array(pair['output'])
            lines = []
            for r, c, r2, c2 in adj_pairs:
                ea = emb_fn(r, c, inp[r,c], out[r,c], inp, out, H, W)
                eb = emb_fn(r2, c2, inp[r2,c2], out[r2,c2], inp, out, H, W)
                L = make_line(ea, eb, W1, W2)
                if L is not None: lines.append(L)
            trans.extend(compute_trans(lines, 100, np.random.default_rng(42+i)))

        if not trans:
            continue

        Tm = np.stack(trans)
        JTm = J6 @ Tm.T

        # Score held-out pair
        held = pairs[hold_idx]
        inp, out = np.array(held['input']), np.array(held['output'])
        lines = []
        for r, c, r2, c2 in adj_pairs:
            ea = emb_fn(r, c, inp[r,c], out[r,c], inp, out, H, W)
            eb = emb_fn(r2, c2, inp[r2,c2], out[r2,c2], inp, out, H, W)
            L = make_line(ea, eb, W1, W2)
            if L is not None: lines.append(L)

        if lines:
            Lm = np.stack(lines)
            inner = Lm @ JTm
            total_score += np.sum(np.log(np.abs(inner) + 1e-10))
            n_eval += 1

    return total_score / max(n_eval, 1)


# ── Fast candidate generation ────────────────────────────────────────────────

def generate_indices_fast(nc, hw, device):
    """Generate all candidate index tensors without Python cartesian loop."""
    n_total = nc ** hw
    indices = torch.zeros(n_total, hw, dtype=torch.long, device=device)
    for pos in range(hw):
        repeat_each = nc ** (hw - 1 - pos)
        tile_count = nc ** pos
        col = torch.arange(nc, device=device).repeat_interleave(repeat_each)
        indices[:, pos] = col.repeat(tile_count)
    return indices


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    arc_dir = "data/ARC-AGI/data/training"
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    test_tasks = [
        # Previously rank 1
        '0d3d703e', '25d8a9c8', '74dd1130',
        # Previously close
        '25ff71a9', '794b24be',
        # Previously unsolved
        '6e02f1e3', 'a85d4709', 'ed36ccf7',
    ]

    for task_name in test_tasks:
        with open(f"{arc_dir}/{task_name}.json") as f:
            task = json.load(f)

        H = len(task['train'][0]['input'])
        W = len(task['train'][0]['input'][0])
        test_inp = np.array(task['test'][0]['input'])
        test_out = np.array(task['test'][0]['output'])
        used_colors = sorted(set(
            c for p in task['train'] + task['test']
            for g in [p['input'], p['output']]
            for row in g for c in row))
        nc = len(used_colors)
        n_total = nc ** (H * W)
        color_to_idx = {c: i for i, c in enumerate(used_colors)}

        adj_pairs = [(r,c,r+dr,c+dc) for r in range(H) for c in range(W)
                     for dr,dc in [(0,1),(1,0)] if r+dr<H and c+dc<W]

        if n_total > 200_000_000:
            print(f"{task_name} ({nc} colors): {n_total:,} too large, skipping")
            continue

        print(f"\n{task_name} ({nc} colors, {n_total:,} candidates)")

        # Step 1: CV score each embedding
        cv_scores = []
        for name, emb_fn, dim in ALL_EMBEDDINGS:
            score = cv_score_embedding(task, name, emb_fn, dim, adj_pairs, H, W)
            cv_scores.append((score, name, emb_fn, dim))
        cv_scores.sort()  # lower = more incident = better

        print("  CV scores (lower=better):")
        for score, name, _, _ in cv_scores:
            print(f"    {name:15s}: {score:.1f}")

        # Step 2: Score candidates with ALL 8 embeddings
        best_embs = [(s, n, fn, d) for s, n, fn, d in cv_scores]  # all of them
        print(f"  Using all {len(best_embs)} embeddings")

        # Generate indices fast
        t0 = time.time()
        indices = generate_indices_fast(nc, H*W, device).reshape(-1, H, W)
        n = indices.shape[0]

        scores = torch.zeros(n, device=device, dtype=torch.float32)
        for _, name, emb_fn, dim in best_embs:
            rng_proj = np.random.RandomState(hash(name) % 2**31)
            W1 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1
            W2 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1

            trans = []
            for i, pair in enumerate(task['train']):
                inp, out = np.array(pair['input']), np.array(pair['output'])
                lines = []
                for r, c, r2, c2 in adj_pairs:
                    ea = emb_fn(r, c, inp[r,c], out[r,c], inp, out, H, W)
                    eb = emb_fn(r2, c2, inp[r2,c2], out[r2,c2], inp, out, H, W)
                    L = make_line(ea, eb, W1, W2)
                    if L is not None: lines.append(L)
                trans.extend(compute_trans(lines, 200, np.random.default_rng(42+i)))

            if not trans: continue
            JTm = J6 @ np.stack(trans).T

            for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
                lt = np.zeros((nc, nc, 6), dtype=np.float32)
                for ia in range(nc):
                    for ib in range(nc):
                        ea = emb_fn(r, c, test_inp[r,c], used_colors[ia],
                                    test_inp, test_inp, H, W)
                        eb = emb_fn(r2, c2, test_inp[r2,c2], used_colors[ib],
                                    test_inp, test_inp, H, W)
                        L = make_line(ea, eb, W1, W2)
                        if L is not None: lt[ia, ib] = L
                flat_inner = lt.reshape(nc*nc, 6) @ JTm
                st = torch.tensor(
                    np.sum(np.log(np.abs(flat_inner)+1e-10), axis=1).reshape(nc,nc),
                    dtype=torch.float32, device=device)
                scores += st[indices[:, r, c], indices[:, r2, c2]]

        # Find rank
        correct_flat = [color_to_idx[test_out[r,c]] for r in range(H) for c in range(W)]
        correct_idx = None
        for i in range(n):
            if indices[i].reshape(-1).cpu().tolist() == correct_flat:
                correct_idx = i; break
        cs = scores[correct_idx].item()
        rank = int((scores < cs).sum().item()) + 1

        elapsed = time.time() - t0
        print(f"  CV-selected rank: {rank:,}/{n_total:,} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
