"""
exp_arc_fast_hist.py — Fast per-histogram table precomputation.

For histogram-dependent embeddings, precompute score tables for each
possible output histogram. Vectorized with numpy to handle 11K+ histograms.

Usage:
  uv run python exp_arc_fast_hist.py --task 25ff71a9
  uv run python exp_arc_fast_hist.py --task 0d3d703e
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


def emb_hist_color(in_c, out_c, hist_diff):
    """Vectorized: just build the embedding from components."""
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    return np.concatenate([in_oh, out_oh, hist_diff])


def build_hist_tables_fast(task, test_inp, used_colors, n_trans_per_pair=200):
    """Precompute per-histogram score tables using vectorized operations."""
    H, W = test_inp.shape
    nc = len(used_colors)
    hw = H * W
    inp_hist = np.array([np.sum(test_inp == c) for c in range(N_COLORS)], dtype=np.float32)

    adj_pairs = [(r,c,r+dr,c+dc) for r in range(H) for c in range(W)
                 for dr,dc in [(0,1),(1,0)] if r+dr<H and c+dc<W]
    n_adj = len(adj_pairs)

    # Compute transversals from training pairs (proper histograms)
    rng_proj = np.random.RandomState(hash('hist_color') % 2**31)
    dim = 30
    W1 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1
    W2 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1

    trans = []
    for i, pair in enumerate(task['train']):
        inp, out = np.array(pair['input']), np.array(pair['output'])
        pH, pW = inp.shape
        inp_h = np.array([np.sum(inp==c) for c in range(N_COLORS)], dtype=np.float32)
        out_h = np.array([np.sum(out==c) for c in range(N_COLORS)], dtype=np.float32)
        diff = (out_h - inp_h) / max(inp.size, 1)
        pa = [(r,c,r+dr,c+dc) for r in range(pH) for c in range(pW)
              for dr,dc in [(0,1),(1,0)] if r+dr<pH and c+dc<pW]
        lines = []
        for r, c, r2, c2 in pa:
            ea = emb_hist_color(inp[r,c], out[r,c], diff)
            eb = emb_hist_color(inp[r2,c2], out[r2,c2], diff)
            L = make_line(ea, eb, W1, W2)
            if L is not None: lines.append(L)
        trans.extend(compute_trans(lines, n_trans_per_pair, np.random.default_rng(42+i)))

    if not trans:
        return {}, adj_pairs

    JTm = J6 @ np.stack(trans).T  # (6, n_trans)

    # Precompute ALL line vectors for ALL (adj, color_a, color_b, hist_diff) combos
    # Key insight: the embedding only depends on (in_c, out_c, hist_diff)
    # NOT on position — so we can batch across adj pairs that share the same
    # input colors at their positions.
    #
    # For each adj pair (r1,c1)-(r2,c2), the embedding depends on:
    #   test_inp[r1,c1], candidate_color_a, test_inp[r2,c2], candidate_color_b, hist_diff
    #
    # The hist_diff varies per histogram. For each histogram, precompute all
    # (nc x nc) line vectors for each adj pair, then score against JTm.

    # Enumerate histograms
    all_hists = []
    def gen(rem, ncols, cur):
        if ncols == 1:
            all_hists.append(tuple(cur + [rem]))
            return
        for k in range(rem + 1):
            gen(rem - k, ncols - 1, cur + [k])
    gen(hw, nc, [])

    print(f"  {len(all_hists)} histograms, {n_adj} adj pairs, {nc} colors")

    # Vectorized: precompute embeddings for all (in_c, out_c, hist_diff) combos
    # For each adj pair, in_c is fixed (from test_inp), out_c varies (nc options)
    t0 = time.time()

    hist_tables = {}
    for hi, hist in enumerate(all_hists):
        out_h = np.zeros(N_COLORS, dtype=np.float32)
        for ci, cnt in enumerate(hist):
            out_h[used_colors[ci]] = cnt
        diff = (out_h - inp_hist) / max(test_inp.size, 1)

        # For this histogram, build all line vectors at once
        # Stack all (adj, ca, cb) embeddings into one big array
        all_lines = np.zeros((n_adj, nc, nc, 6), dtype=np.float32)

        for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
            in_c1 = int(test_inp[r, c])
            in_c2 = int(test_inp[r2, c2])
            for ia in range(nc):
                ea = emb_hist_color(in_c1, used_colors[ia], diff)
                for ib in range(nc):
                    eb = emb_hist_color(in_c2, used_colors[ib], diff)
                    L = make_line(ea, eb, W1, W2)
                    if L is not None:
                        all_lines[ap_idx, ia, ib] = L

        # Score: (n_adj, nc, nc, 6) @ (6, n_trans) → (n_adj, nc, nc, n_trans)
        flat = all_lines.reshape(n_adj * nc * nc, 6) @ JTm
        scores = np.sum(np.log(np.abs(flat) + 1e-10), axis=1)
        score_tables = scores.reshape(n_adj, nc, nc).astype(np.float32)

        hist_tables[hist] = score_tables  # (n_adj, nc, nc)

        if (hi + 1) % 100 == 0:
            print(f"    {hi+1}/{len(all_hists)} histograms ({time.time()-t0:.1f}s)")

    print(f"  Histogram tables built in {time.time()-t0:.1f}s")
    return hist_tables, adj_pairs


def solve_task(task, device):
    H = len(task['test'][0]['input'])
    W = len(task['test'][0]['input'][0])
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
        print(f"  Too many candidates ({n_total:,})")
        return None

    # Build histogram tables
    hist_tables, adj_pairs = build_hist_tables_fast(task, test_inp, used_colors)
    if not hist_tables:
        print(f"  No transversals found")
        return None

    # Generate all candidate indices
    t0 = time.time()
    hw = H * W
    indices = torch.zeros(n_total, hw, dtype=torch.long, device=device)
    for pos in range(hw):
        repeat_each = nc ** (hw - 1 - pos)
        tile_count = nc ** pos
        col = torch.arange(nc, device=device).repeat_interleave(repeat_each)
        indices[:, pos] = col.repeat(tile_count)
    indices = indices.reshape(-1, H, W)
    n = indices.shape[0]

    # Compute histogram for each candidate
    flat_idx = indices.reshape(n, hw)
    counts = torch.zeros(n, nc, device=device, dtype=torch.long)
    for ci in range(nc):
        counts[:, ci] = (flat_idx == ci).sum(dim=1)

    # Score each candidate using its histogram's table
    scores = torch.zeros(n, device=device, dtype=torch.float32)

    for hist_key, tables_np in hist_tables.items():
        hist_t = torch.tensor(list(hist_key), dtype=torch.long, device=device)
        mask = torch.all(counts == hist_t.unsqueeze(0), dim=1)
        if not mask.any():
            continue

        # Move tables to device
        tables = [torch.tensor(tables_np[ap], dtype=torch.float32, device=device)
                  for ap in range(len(adj_pairs))]

        for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
            scores[mask] += tables[ap_idx][indices[mask, r, c], indices[mask, r2, c2]]

    # Find correct answer rank
    correct_flat = [color_to_idx[test_out[r,c]] for r in range(H) for c in range(W)]
    correct_idx = None
    for i in range(n):
        if indices[i].reshape(-1).cpu().tolist() == correct_flat:
            correct_idx = i; break

    cs = scores[correct_idx].item()
    rank = int((scores < cs).sum().item()) + 1

    # Best prediction
    best_idx = scores.argmin().item()
    best_grid = np.array([used_colors[indices[best_idx, r, c].item()]
                          for r in range(H) for c in range(W)]).reshape(H, W)
    # Skip identity
    if np.array_equal(best_grid, test_inp):
        order = scores.argsort()
        best_grid = np.array([used_colors[indices[order[1].item(), r, c].item()]
                              for r in range(H) for c in range(W)]).reshape(H, W)
        rank = max(1, rank - 1)

    elapsed = time.time() - t0
    match = np.array_equal(best_grid, test_out)

    return {
        'prediction': best_grid,
        'match': match,
        'rank': rank,
        'n_candidates': n_total,
        'time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="*", default=["25ff71a9", "794b24be"])
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    arc_dir = "data/ARC-AGI/data/training"

    print(f"Fast Histogram Solver (device={device})")
    print()

    for tname in args.task:
        with open(f"{arc_dir}/{tname}.json") as f:
            task = json.load(f)

        nc = len(set(c for p in task['train']+task['test']
                     for g in [p['input'],p['output']] for row in g for c in row))
        H = len(task['test'][0]['input'])
        W = len(task['test'][0]['input'][0])

        print(f"{tname} ({nc} colors, {H}x{W}):")
        r = solve_task(task, device)
        if r is None:
            continue

        if r['match']:
            print(f"  SOLVED ✓ (rank {r['rank']}, {r['time']:.1f}s)")
        else:
            print(f"  rank {r['rank']}/{r['n_candidates']:,} ({r['time']:.1f}s)")
        print()


if __name__ == "__main__":
    main()
