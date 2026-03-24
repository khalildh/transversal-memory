"""
exp_arc_seed_sweep.py — Test if multiple random seeds improve ARC solving.

For each task, run the multi-embedding transversal solver with different
random projection seeds and combine scores across all seeds.

Usage:
  uv run python exp_arc_seed_sweep.py
  uv run python exp_arc_seed_sweep.py --task 794b24be
"""

import json
import os
import argparse
import time
import numpy as np
import torch
from itertools import product as cartesian
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


EMBEDDINGS = [
    ('hist_color', emb_hist_color, 30),
    ('color_only', emb_color_only, 20),
    ('pos_color', emb_pos_color, 22),
    ('all', emb_all, 42),
]


def score_one_seed(task, seed, n_trans_per_pair=200):
    """Build transversals + score tables for one random seed.
    Returns list of (nc, nc) score tables per adj pair."""
    H = len(task['train'][0]['input'])
    W = len(task['train'][0]['input'][0])
    test_inp = np.array(task['test'][0]['input'])
    used_colors = sorted(set(
        c for p in task['train'] + task['test']
        for g in [p['input'], p['output']]
        for row in g for c in row))
    nc = len(used_colors)

    adj_pairs = [(r,c,r+dr,c+dc) for r in range(H) for c in range(W)
                 for dr,dc in [(0,1),(1,0)] if r+dr<H and c+dc<W]

    all_tables = []  # list of lists of (nc, nc) numpy arrays

    for name, emb_fn, dim in EMBEDDINGS:
        rng_proj = np.random.RandomState(seed + hash(name) % 2**31)
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
            trans.extend(compute_trans(lines, n_trans_per_pair,
                                       np.random.default_rng(seed + 42 + i)))

        if not trans:
            continue

        JTm = J6 @ np.stack(trans).T

        tables = []
        for r, c, r2, c2 in adj_pairs:
            lt = np.zeros((nc, nc, 6), dtype=np.float32)
            for ia in range(nc):
                for ib in range(nc):
                    ea = emb_fn(r, c, test_inp[r,c], used_colors[ia],
                                test_inp, test_inp, H, W)
                    eb = emb_fn(r2, c2, test_inp[r2,c2], used_colors[ib],
                                test_inp, test_inp, H, W)
                    L = make_line(ea, eb, W1, W2)
                    if L is not None:
                        lt[ia, ib] = L
            flat_inner = lt.reshape(nc*nc, 6) @ JTm
            st = np.sum(np.log(np.abs(flat_inner) + 1e-10),
                        axis=1).reshape(nc, nc).astype(np.float32)
            tables.append(st)
        all_tables.append(tables)

    return all_tables, adj_pairs, used_colors, nc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="*",
                        default=["25ff71a9", "794b24be", "0d3d703e"])
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    arc_dir = "data/ARC-AGI/data/training"
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    for task_name in args.task:
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

        if n_total > 200_000_000:
            print(f"{task_name}: {n_total:,} candidates — using sampling")
            continue

        print(f"\n{task_name} ({nc} colors, {n_total:,} candidates)")

        # Test individual seeds and cumulative
        indices = torch.tensor(
            list(cartesian(range(nc), repeat=H*W)),
            dtype=torch.long, device=device).reshape(-1, H, W)
        n = indices.shape[0]

        correct_idx_grid = [[color_to_idx[test_out[r,c]] for c in range(W)]
                            for r in range(H)]

        cumulative_scores = torch.zeros(n, device=device, dtype=torch.float32)

        for seed in range(args.seeds):
            t0 = time.time()
            all_tables, adj_pairs, _, _ = score_one_seed(task, seed * 1000)

            # Score with this seed's tables
            seed_scores = torch.zeros(n, device=device, dtype=torch.float32)
            for tables in all_tables:
                for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
                    st = torch.tensor(tables[ap_idx], dtype=torch.float32,
                                      device=device)
                    seed_scores += st[indices[:, r, c], indices[:, r2, c2]]

            # Individual seed rank
            cs = seed_scores[0].item()  # placeholder
            for i in range(n):
                grid = [used_colors[indices[i, r, c].item()]
                        for r in range(H) for c in range(W)]
                if grid == [test_out[r,c] for r in range(H) for c in range(W)]:
                    cs = seed_scores[i].item()
                    break
            rank_single = int((seed_scores < cs).sum().item()) + 1

            # Cumulative
            cumulative_scores += seed_scores
            cc = cumulative_scores[i].item()  # correct answer's cumulative score
            rank_cumul = int((cumulative_scores < cc).sum().item()) + 1

            elapsed = time.time() - t0
            print(f"  seed {seed}: rank {rank_single:,} alone, "
                  f"rank {rank_cumul:,} cumulative ({elapsed:.1f}s)")

        print(f"  Final (cumulative {args.seeds} seeds): rank {rank_cumul:,}/{n_total:,}")


if __name__ == "__main__":
    main()
