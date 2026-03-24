"""
exp_arc_bp_then_score.py — BP locks confident cells, full scorer searches uncertain ones.

1. BP produces initial grid (~78-97% correct)
2. Identify uncertain cells from BP belief gaps
3. Enumerate all color combos on uncertain cells (lock the rest)
4. Score each variant with full multi-embedding transversal scorer
5. Pick the best

Usage:
  uv run python exp_arc_bp_then_score.py
"""

import json
import os
import argparse
import time
import numpy as np
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


# ── Embeddings ───────────────────────────────────────────────────────────────

def emb_color_only(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    return np.concatenate([in_oh, out_oh])

def emb_pos_color(r, c, in_c, out_c, inp, out, H, W):
    pos = np.array([r/max(H-1,1), c/max(W-1,1)], dtype=np.float32)
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    return np.concatenate([pos, in_oh, out_oh])

def emb_hist_color(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    diff = np.array([(np.sum(out==i)-np.sum(inp==i))/max(inp.size,1)
                     for i in range(N_COLORS)], dtype=np.float32)
    return np.concatenate([in_oh, out_oh, diff])

def emb_all(r, c, in_c, out_c, inp, out, H, W):
    pos = np.array([r/max(H-1,1), c/max(W-1,1)], dtype=np.float32)
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    ih = np.array([np.sum(inp==i)/max(inp.size,1) for i in range(N_COLORS)], dtype=np.float32)
    oh = np.array([np.sum(out==i)/max(out.size,1) for i in range(N_COLORS)], dtype=np.float32)
    return np.concatenate([pos, in_oh, out_oh, ih, oh])

# BP uses fast embeddings (no histogram dependency)
BP_EMBEDDINGS = [
    ('color_only', emb_color_only, 20),
    ('pos_color', emb_pos_color, 22),
]

# Full scorer uses ALL embeddings (proper histogram per candidate)
SCORE_EMBEDDINGS = [
    ('hist_color', emb_hist_color, 30),
    ('color_only', emb_color_only, 20),
    ('pos_color', emb_pos_color, 22),
    ('all', emb_all, 42),
]


# ── BP ───────────────────────────────────────────────────────────────────────

def build_bp_potentials(task, test_inp, used_colors, embeddings):
    H, W = test_inp.shape
    nc = len(used_colors)
    adj_pairs = [(r,c,r+dr,c+dc) for r in range(H) for c in range(W)
                 for dr,dc in [(0,1),(1,0)] if r+dr<H and c+dc<W]
    potentials = [np.zeros((nc, nc), dtype=np.float32) for _ in adj_pairs]

    for name, emb_fn, dim in embeddings:
        rng_proj = np.random.RandomState(hash(name) % 2**31)
        W1 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1
        W2 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1
        trans = []
        for i, pair in enumerate(task['train']):
            inp, out = np.array(pair['input']), np.array(pair['output'])
            pH, pW = inp.shape
            pa = [(r,c,r+dr,c+dc) for r in range(pH) for c in range(pW)
                  for dr,dc in [(0,1),(1,0)] if r+dr<pH and c+dc<pW]
            lines = []
            for r, c, r2, c2 in pa:
                ea = emb_fn(r, c, inp[r,c], out[r,c], inp, out, pH, pW)
                eb = emb_fn(r2, c2, inp[r2,c2], out[r2,c2], inp, out, pH, pW)
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
            flat = lt.reshape(nc*nc, 6) @ JTm
            potentials[ap_idx] += np.sum(np.log(np.abs(flat)+1e-10),
                                          axis=1).reshape(nc, nc).astype(np.float32)
    return adj_pairs, potentials


def run_bp(H, W, nc, adj_pairs, potentials, n_iters=30):
    cell_idx = lambda r, c: r * W + c
    neighbors = [[] for _ in range(H*W)]
    for ap_idx, (r,c,r2,c2) in enumerate(adj_pairs):
        i, j = cell_idx(r,c), cell_idx(r2,c2)
        neighbors[i].append((j, ap_idx, True))
        neighbors[j].append((i, ap_idx, False))

    rng = np.random.RandomState(42)
    messages = {}
    for ap_idx, (r,c,r2,c2) in enumerate(adj_pairs):
        i, j = cell_idx(r,c), cell_idx(r2,c2)
        messages[(i,j)] = rng.randn(nc).astype(np.float32) * 0.01
        messages[(j,i)] = rng.randn(nc).astype(np.float32) * 0.01

    for _ in range(n_iters):
        new_msg = {}
        for ap_idx, (r1,c1,r2,c2) in enumerate(adj_pairs):
            i, j = cell_idx(r1,c1), cell_idx(r2,c2)
            pot = potentials[ap_idx]
            inc_i = sum(messages[(nb,i)] for nb,_,_ in neighbors[i]
                       if nb != j and (nb,i) in messages)
            if isinstance(inc_i, int): inc_i = np.zeros(nc, dtype=np.float32)
            msg_ij = np.min(pot + inc_i[:,None], axis=0)
            msg_ij -= msg_ij.min()
            inc_j = sum(messages[(nb,j)] for nb,_,_ in neighbors[j]
                       if nb != i and (nb,j) in messages)
            if isinstance(inc_j, int): inc_j = np.zeros(nc, dtype=np.float32)
            msg_ji = np.min(pot + inc_j[None,:], axis=1)
            msg_ji -= msg_ji.min()
            new_msg[(i,j)] = 0.5*messages.get((i,j),msg_ij) + 0.5*msg_ij
            new_msg[(j,i)] = 0.5*messages.get((j,i),msg_ji) + 0.5*msg_ji
        messages = new_msg

    beliefs = np.zeros((H, W, nc), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            i = cell_idx(r,c)
            for nb,_,_ in neighbors[i]:
                if (nb,i) in messages: beliefs[r,c] += messages[(nb,i)]
    return beliefs


# ── Full scorer ──────────────────────────────────────────────────────────────

def score_full(grid, test_inp, task, embeddings):
    """Score a complete grid with ALL embeddings using proper histograms."""
    H, W = grid.shape
    adj_pairs = [(r,c,r+dr,c+dc) for r in range(H) for c in range(W)
                 for dr,dc in [(0,1),(1,0)] if r+dr<H and c+dc<W]
    total = 0.0
    for name, emb_fn, dim in embeddings:
        rng_proj = np.random.RandomState(hash(name) % 2**31)
        W1 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1
        W2 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1
        trans = []
        for i, pair in enumerate(task['train']):
            inp, out = np.array(pair['input']), np.array(pair['output'])
            pH, pW = inp.shape
            pa = [(r,c,r+dr,c+dc) for r in range(pH) for c in range(pW)
                  for dr,dc in [(0,1),(1,0)] if r+dr<pH and c+dc<pW]
            lines = []
            for r, c, r2, c2 in pa:
                ea = emb_fn(r, c, inp[r,c], out[r,c], inp, out, pH, pW)
                eb = emb_fn(r2, c2, inp[r2,c2], out[r2,c2], inp, out, pH, pW)
                L = make_line(ea, eb, W1, W2)
                if L is not None: lines.append(L)
            trans.extend(compute_trans(lines, 200, np.random.default_rng(42+i)))
        if not trans: continue
        JTm = J6 @ np.stack(trans).T
        lines = []
        for r, c, r2, c2 in adj_pairs:
            ea = emb_fn(r, c, test_inp[r,c], grid[r,c], test_inp, grid, H, W)
            eb = emb_fn(r2, c2, test_inp[r2,c2], grid[r2,c2], test_inp, grid, H, W)
            L = make_line(ea, eb, W1, W2)
            if L is not None: lines.append(L)
        if lines:
            Lm = np.stack(lines)
            inner = Lm @ JTm
            total += np.sum(np.log(np.abs(inner) + 1e-10))
    return total


# ── Main solver ──────────────────────────────────────────────────────────────

def solve_task(task, max_uncertain=15):
    H = len(task['test'][0]['input'])
    W = len(task['test'][0]['input'][0])
    test_inp = np.array(task['test'][0]['input'])
    test_out = np.array(task['test'][0]['output'])
    used_colors = sorted(set(
        c for p in task['train'] + task['test']
        for g in [p['input'], p['output']]
        for row in g for c in row))
    nc = len(used_colors)

    # Step 1: BP
    print(f"  BP...")
    adj_pairs, potentials = build_bp_potentials(task, test_inp, used_colors, BP_EMBEDDINGS)
    beliefs = run_bp(H, W, nc, adj_pairs, potentials)
    bp_grid_idx = beliefs.argmin(axis=2)
    bp_grid = np.array([[used_colors[bp_grid_idx[r,c]]
                         for c in range(W)] for r in range(H)])
    bp_acc = np.mean(bp_grid == test_out)

    # Step 2: Find uncertain cells
    sorted_b = np.sort(beliefs, axis=2)
    uncertainty = sorted_b[:,:,1] - sorted_b[:,:,0]
    # Take the most uncertain cells, capped at max_uncertain
    flat_unc = [(uncertainty[r,c], r, c) for r in range(H) for c in range(W)]
    flat_unc.sort()
    uncertain_cells = [(r, c) for _, r, c in flat_unc[:max_uncertain]]
    n_unc = len(uncertain_cells)
    n_variants = nc ** n_unc

    print(f"  BP acc: {bp_acc:.3f}, uncertain cells: {n_unc}, variants: {n_variants:,}")

    if n_variants > 5_000_000:
        # Too many — reduce uncertain cells
        n_unc = max(1, int(np.log(5_000_000) / np.log(nc)))
        uncertain_cells = uncertain_cells[:n_unc]
        n_variants = nc ** n_unc
        print(f"  Reduced to {n_unc} uncertain, {n_variants:,} variants")

    # Step 3: Build precomputed score tables (same as fast solver)
    color_to_idx = {c: i for i, c in enumerate(used_colors)}
    print(f"  Building precomputed score tables...")
    adj_list = [(r,c,r+dr,c+dc) for r in range(H) for c in range(W)
                for dr,dc in [(0,1),(1,0)] if r+dr<H and c+dc<W]

    # Accumulate score tables across all embeddings
    score_tables = [np.zeros((nc, nc), dtype=np.float32) for _ in adj_list]

    for name, emb_fn, dim in SCORE_EMBEDDINGS:
        rng_proj = np.random.RandomState(hash(name) % 2**31)
        W1 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1
        W2 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1
        trans = []
        for i, pair in enumerate(task['train']):
            inp, out = np.array(pair['input']), np.array(pair['output'])
            pH, pW = inp.shape
            pa = [(r,c,r+dr,c+dc) for r in range(pH) for c in range(pW)
                  for dr,dc in [(0,1),(1,0)] if r+dr<pH and c+dc<pW]
            lines = []
            for r, c, r2, c2 in pa:
                ea = emb_fn(r, c, inp[r,c], out[r,c], inp, out, pH, pW)
                eb = emb_fn(r2, c2, inp[r2,c2], out[r2,c2], inp, out, pH, pW)
                L = make_line(ea, eb, W1, W2)
                if L is not None: lines.append(L)
            trans.extend(compute_trans(lines, 200, np.random.default_rng(42+i)))
        if not trans: continue
        JTm = J6 @ np.stack(trans).T
        for ap_idx, (r, c, r2, c2) in enumerate(adj_list):
            lt = np.zeros((nc, nc, 6), dtype=np.float32)
            for ia in range(nc):
                for ib in range(nc):
                    ea = emb_fn(r, c, test_inp[r,c], used_colors[ia],
                                test_inp, test_inp, H, W)
                    eb = emb_fn(r2, c2, test_inp[r2,c2], used_colors[ib],
                                test_inp, test_inp, H, W)
                    L = make_line(ea, eb, W1, W2)
                    if L is not None: lt[ia, ib] = L
            flat = lt.reshape(nc*nc, 6) @ JTm
            score_tables[ap_idx] += np.sum(np.log(np.abs(flat)+1e-10),
                                            axis=1).reshape(nc,nc).astype(np.float32)

    # Precompute: which adj pairs touch uncertain cells?
    unc_set = set(uncertain_cells)
    # Split adj pairs into fixed (both cells locked) and variable (at least one uncertain)
    fixed_score = 0.0
    variable_adj = []  # (ap_idx, r, c, r2, c2)
    bp_idx = np.array([[color_to_idx[bp_grid[r,c]] for c in range(W)] for r in range(H)])

    for ap_idx, (r, c, r2, c2) in enumerate(adj_list):
        if (r,c) in unc_set or (r2,c2) in unc_set:
            variable_adj.append((ap_idx, r, c, r2, c2))
        else:
            fixed_score += score_tables[ap_idx][bp_idx[r,c], bp_idx[r2,c2]]

    print(f"  Fixed adj pairs: {len(adj_list)-len(variable_adj)}, variable: {len(variable_adj)}")

    # Step 4: Enumerate variants, score only variable adj pairs
    print(f"  Scoring {n_variants:,} variants (fast table lookup)...")
    best_grid = bp_grid.copy()
    best_score = float('inf')
    n_scored = 0

    for combo in cartesian(range(nc), repeat=n_unc):
        # Build grid index for this variant
        grid_idx = bp_idx.copy()
        for k, (r, c) in enumerate(uncertain_cells):
            grid_idx[r, c] = combo[k]

        # Score: fixed + variable
        s = fixed_score
        for ap_idx, r, c, r2, c2 in variable_adj:
            s += score_tables[ap_idx][grid_idx[r,c], grid_idx[r2,c2]]

        n_scored += 1
        if s < best_score:
            best_score = s
            best_combo = combo

    # Build best grid
    best_grid = bp_grid.copy()
    for k, (r, c) in enumerate(uncertain_cells):
        best_grid[r, c] = used_colors[best_combo[k]]

    match = np.array_equal(best_grid, test_out)
    cell_acc = np.mean(best_grid == test_out)

    # Score correct answer
    correct_idx = np.array([[color_to_idx[test_out[r,c]] for c in range(W)] for r in range(H)])
    correct_score = fixed_score
    for ap_idx, r, c, r2, c2 in variable_adj:
        correct_score += score_tables[ap_idx][correct_idx[r,c], correct_idx[r2,c2]]

    return {
        'prediction': best_grid,
        'correct': test_out,
        'match': match,
        'cell_acc': cell_acc,
        'bp_acc': bp_acc,
        'n_uncertain': n_unc,
        'n_scored': n_scored,
        'best_score': best_score,
        'correct_score': correct_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="*",
                        default=["0d3d703e", "25ff71a9", "794b24be", "aabf363d"])
    parser.add_argument("--max-uncertain", type=int, default=12)
    args = parser.parse_args()

    arc_dirs = ["data/ARC-AGI/data/training", "data/ARC-AGI/data/evaluation"]

    print(f"BP → Full Score solver (max {args.max_uncertain} uncertain cells)")
    print()

    solved = 0
    for tname in args.task:
        task = None
        for d in arc_dirs:
            p = os.path.join(d, f"{tname}.json")
            if os.path.exists(p):
                with open(p) as f: task = json.load(f)
                break
        if not task: print(f"  {tname}: not found"); continue

        H = len(task['test'][0]['input'])
        W = len(task['test'][0]['input'][0])
        nc = len(set(c for p in task['train']+task['test']
                     for g in [p['input'],p['output']] for row in g for c in row))

        print(f"{tname} ({nc} colors, {H}x{W}):")
        t0 = time.time()
        r = solve_task(task, max_uncertain=args.max_uncertain)
        elapsed = time.time() - t0

        if r['match']:
            print(f"  SOLVED ✓ ({elapsed:.1f}s)")
            solved += 1
        else:
            print(f"  bp={r['bp_acc']:.3f} → final={r['cell_acc']:.3f} "
                  f"({int(r['cell_acc']*H*W)}/{H*W}) ({elapsed:.1f}s)")
            print(f"  scored {r['n_scored']:,} variants on {r['n_uncertain']} uncertain cells")
            print(f"  best_score={r['best_score']:.2f}, correct_score={r['correct_score']:.2f}")
            pred, correct = r['prediction'], r['correct']
            mismatches = [(r_, c_) for r_ in range(H) for c_ in range(W)
                          if pred[r_, c_] != correct[r_, c_]]
            for r_, c_ in mismatches[:5]:
                print(f"    ({r_},{c_}): predicted {pred[r_,c_]}, correct {correct[r_,c_]}")
        print()

    print(f"SOLVED: {solved}/{len(args.task)}")


if __name__ == "__main__":
    main()
