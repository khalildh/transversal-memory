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
        c for p in task['train']
        for g in [p['input'], p['output']]
        for row in g for c in row
    ) | set(c for row in task['test'][0]['input'] for c in row))
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

    # Score in chunks to avoid OOM
    t0 = time.time()
    hw = H * W
    chunk_size = min(n_total, 5_000_000)

    # Score correct answer first
    correct_grid_idx = np.array([color_to_idx[test_out[r,c]] for r in range(H) for c in range(W)])
    correct_hist = tuple(int(np.sum(correct_grid_idx == ci)) for ci in range(nc))
    correct_tables = hist_tables.get(correct_hist)
    correct_score = 0.0
    if correct_tables is not None:
        for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
            correct_score += float(correct_tables[ap_idx][correct_grid_idx[r*W+c], correct_grid_idx[r2*W+c2]])

    better_count = 0
    best_score = float('inf')
    best_grid = test_out.copy()

    n_chunks = (n_total + chunk_size - 1) // chunk_size
    for chunk_i in range(n_chunks):
        start = chunk_i * chunk_size
        end = min(start + chunk_size, n_total)
        n_chunk = end - start

        # Generate indices for this chunk (vectorized)
        global_idx = torch.arange(start, end, device=device, dtype=torch.long)
        indices = torch.zeros(n_chunk, hw, dtype=torch.long, device=device)
        for pos in range(hw):
            divisor = nc ** (hw - 1 - pos)
            indices[:, pos] = (global_idx // divisor) % nc

        indices_hw = indices.reshape(n_chunk, H, W)

        # Compute histogram for each candidate
        flat_idx = indices  # already (n_chunk, hw)
        counts = torch.zeros(n_chunk, nc, device=device, dtype=torch.long)
        for ci in range(nc):
            counts[:, ci] = (flat_idx == ci).sum(dim=1)

        # Score: for each candidate, look up its histogram's tables
        scores = torch.zeros(n_chunk, device=device, dtype=torch.float32)
        counts_cpu = counts.cpu().numpy()
        for hist_key, tables_np in hist_tables.items():
            hist_arr = np.array(hist_key)
            mask_np = np.all(counts_cpu == hist_arr, axis=1)
            if not mask_np.any():
                continue
            idxs = np.where(mask_np)[0]
            idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
            sub_hw = indices_hw[idxs_t]
            sub_scores = torch.zeros(len(idxs), device=device, dtype=torch.float32)
            for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
                st = torch.tensor(tables_np[ap_idx], dtype=torch.float32, device=device)
                sub_scores += st[sub_hw[:, r, c], sub_hw[:, r2, c2]]
            scores[idxs_t] = sub_scores

        # Count how many beat correct
        better_count += int((scores < correct_score).sum().item())

        # Track best
        chunk_best_idx = scores.argmin().item()
        if scores[chunk_best_idx].item() < best_score:
            best_score = scores[chunk_best_idx].item()
            best_vals = indices[chunk_best_idx].cpu().numpy()
            best_grid = np.array([used_colors[v] for v in best_vals]).reshape(H, W)

        if (chunk_i + 1) % 5 == 0 or chunk_i == n_chunks - 1:
            print(f"    chunk {chunk_i+1}/{n_chunks}: {better_count} better so far ({time.time()-t0:.1f}s)")

    rank = better_count + 1
    # Skip identity
    if np.array_equal(best_grid, test_inp):
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
