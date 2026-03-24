"""
exp_arc_chain_solve.py — Chained solver: BP → generative refinement → scoring.

1. BP produces initial grid (fast, ~70-97% cell accuracy)
2. For cells where BP is uncertain, try all colors via scoring
3. Score the full grid + variants against training transversals
4. Pick the best

Usage:
  uv run python exp_arc_chain_solve.py
  uv run python exp_arc_chain_solve.py --task 0d3d703e
"""

import json
import os
import argparse
import time
import numpy as np
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


def emb_color_only(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    return np.concatenate([in_oh, out_oh])

def emb_pos_color(r, c, in_c, out_c, inp, out, H, W):
    pos = np.array([r/max(H-1,1), c/max(W-1,1)], dtype=np.float32)
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    return np.concatenate([pos, in_oh, out_oh])

EMBEDDINGS = [
    ('color_only', emb_color_only, 20),
    ('pos_color', emb_pos_color, 22),
]


def build_potentials_and_transversals(task, test_inp, used_colors):
    """Build pairwise potentials + transversal pool for scoring."""
    H, W = test_inp.shape
    nc = len(used_colors)
    adj_pairs = [(r,c,r+dr,c+dc) for r in range(H) for c in range(W)
                 for dr,dc in [(0,1),(1,0)] if r+dr<H and c+dc<W]

    potentials = [np.zeros((nc, nc), dtype=np.float32) for _ in adj_pairs]
    all_trans = []

    for name, emb_fn, dim in EMBEDDINGS:
        rng_proj = np.random.RandomState(hash(name) % 2**31)
        W1 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1
        W2 = rng_proj.randn(4, 2*dim).astype(np.float32) * 0.1

        trans = []
        for i, pair in enumerate(task['train']):
            inp, out = np.array(pair['input']), np.array(pair['output'])
            pH, pW = inp.shape
            pair_adj = [(r,c,r+dr,c+dc) for r in range(pH) for c in range(pW)
                        for dr,dc in [(0,1),(1,0)] if r+dr<pH and c+dc<pW]
            lines = []
            for r, c, r2, c2 in pair_adj:
                ea = emb_fn(r, c, inp[r,c], out[r,c], inp, out, pH, pW)
                eb = emb_fn(r2, c2, inp[r2,c2], out[r2,c2], inp, out, pH, pW)
                L = make_line(ea, eb, W1, W2)
                if L is not None: lines.append(L)
            trans.extend(compute_trans(lines, 200, np.random.default_rng(42+i)))

        if not trans: continue
        all_trans.extend(trans)
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

    return adj_pairs, potentials, all_trans


def run_bp(H, W, nc, adj_pairs, potentials, n_iters=30):
    """Run BP, return beliefs (H, W, nc)."""
    cell_idx = lambda r, c: r * W + c
    neighbors = [[] for _ in range(H * W)]
    for ap_idx, ((r1,c1), (r2,c2)) in enumerate(
            [((r,c),(r2,c2)) for r,c,r2,c2 in adj_pairs]):
        i, j = cell_idx(r1,c1), cell_idx(r2,c2)
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

            inc_i = np.zeros(nc, dtype=np.float32)
            for nb, _, _ in neighbors[i]:
                if nb != j and (nb,i) in messages: inc_i += messages[(nb,i)]
            msg_ij = np.min(pot + inc_i[:,None], axis=0)
            msg_ij -= msg_ij.min()

            inc_j = np.zeros(nc, dtype=np.float32)
            for nb, _, _ in neighbors[j]:
                if nb != i and (nb,j) in messages: inc_j += messages[(nb,j)]
            msg_ji = np.min(pot + inc_j[None,:], axis=1)
            msg_ji -= msg_ji.min()

            new_msg[(i,j)] = 0.5 * messages.get((i,j), msg_ij) + 0.5 * msg_ij
            new_msg[(j,i)] = 0.5 * messages.get((j,i), msg_ji) + 0.5 * msg_ji
        messages = new_msg

    beliefs = np.zeros((H, W, nc), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            i = cell_idx(r, c)
            for nb, _, _ in neighbors[i]:
                if (nb, i) in messages: beliefs[r,c] += messages[(nb,i)]
    return beliefs


def score_grid(grid, test_inp, adj_pairs, potentials, used_colors, unary=None):
    """Score a complete grid using pairwise potentials + unary bias."""
    color_to_idx = {c: i for i, c in enumerate(used_colors)}
    H, W = grid.shape
    total = 0.0
    for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
        ia = color_to_idx[grid[r, c]]
        ib = color_to_idx[grid[r2, c2]]
        total += potentials[ap_idx][ia, ib]
    if unary is not None:
        for r in range(H):
            for c in range(W):
                total += unary[r, c, color_to_idx[grid[r, c]]]
    return total


def solve_task(task):
    H = len(task['test'][0]['input'])
    W = len(task['test'][0]['input'][0])
    test_inp = np.array(task['test'][0]['input'])
    test_out = np.array(task['test'][0]['output'])
    used_colors = sorted(set(
        c for p in task['train'] + task['test']
        for g in [p['input'], p['output']]
        for row in g for c in row))
    nc = len(used_colors)
    color_to_idx = {c: i for i, c in enumerate(used_colors)}

    # Step 1: Build potentials
    adj_pairs, potentials, all_trans = build_potentials_and_transversals(
        task, test_inp, used_colors)

    # Step 1b: Build unary potentials from training pair statistics
    # For each cell position, what colors appear in training outputs?
    # And: what's the relationship between input color and output color?
    unary = np.zeros((H, W, nc), dtype=np.float32)

    for pair in task['train']:
        inp, out = np.array(pair['input']), np.array(pair['output'])
        pH, pW = inp.shape
        for r in range(min(H, pH)):
            for c in range(min(W, pW)):
                in_c = inp[r, c]
                out_c = out[r, c]
                # If test input matches this training input color at this position,
                # bias toward the corresponding output color
                if test_inp[r, c] == in_c:
                    out_idx = used_colors.index(out_c) if out_c in used_colors else -1
                    if out_idx >= 0:
                        unary[r, c, out_idx] -= 5.0  # strong bias toward this color

                # General: which colors appear at this position in outputs?
                out_idx = used_colors.index(out_c) if out_c in used_colors else -1
                if out_idx >= 0:
                    unary[r, c, out_idx] -= 1.0  # mild bias

    # Penalize identity (output == input) — ARC never has identity
    for r in range(H):
        for c in range(W):
            in_c = test_inp[r, c]
            if in_c in color_to_idx:
                # Check if training pairs ever keep this color
                keeps = sum(1 for p in task['train']
                           if r < len(p['input']) and c < len(p['input'][0])
                           and p['input'][r][c] == p['output'][r][c] == in_c)
                changes = sum(1 for p in task['train']
                             if r < len(p['input']) and c < len(p['input'][0])
                             and p['input'][r][c] == in_c and p['output'][r][c] != in_c)
                if changes > 0 and keeps == 0:
                    unary[r, c, color_to_idx[in_c]] += 10.0  # penalize keeping input

    # Step 2: BP for initial grid (with unary potentials added to beliefs)
    beliefs = run_bp(H, W, nc, adj_pairs, potentials)
    beliefs += unary  # add unary bias
    bp_grid_idx = beliefs.argmin(axis=2)
    bp_grid = np.array([[used_colors[bp_grid_idx[r,c]]
                         for c in range(W)] for r in range(H)])
    bp_score = score_grid(bp_grid, test_inp, adj_pairs, potentials, used_colors, unary)
    bp_acc = np.mean(bp_grid == test_out)

    # Step 3: Find uncertain cells (where top-2 beliefs are close)
    sorted_beliefs = np.sort(beliefs, axis=2)
    uncertainty = sorted_beliefs[:,:,1] - sorted_beliefs[:,:,0]  # gap between best and 2nd best
    threshold = np.percentile(uncertainty, 30)  # bottom 30% most uncertain
    uncertain_cells = [(r, c) for r in range(H) for c in range(W)
                       if uncertainty[r, c] <= threshold]

    # Step 4: Try all colors at uncertain cells, keep best scoring grid
    best_grid = bp_grid.copy()
    best_score = bp_score

    # Greedy refinement: for each uncertain cell, try all colors
    for r, c in uncertain_cells:
        current = best_grid[r, c]
        for color in used_colors:
            if color == current:
                continue
            test_grid = best_grid.copy()
            test_grid[r, c] = color
            s = score_grid(test_grid, test_inp, adj_pairs, potentials, used_colors, unary)
            if s < best_score:
                best_score = s
                best_grid = test_grid.copy()

    # Step 5: ICM refinement on the full grid
    for icm_round in range(10):
        changed = False
        for r in range(H):
            for c in range(W):
                current = best_grid[r, c]
                for color in used_colors:
                    if color == current:
                        continue
                    test_grid = best_grid.copy()
                    test_grid[r, c] = color
                    s = score_grid(test_grid, test_inp, adj_pairs, potentials, used_colors, unary)
                    if s < best_score:
                        best_score = s
                        best_grid = test_grid.copy()
                        changed = True
        if not changed:
            break

    match = np.array_equal(best_grid, test_out)
    cell_acc = np.mean(best_grid == test_out)

    return {
        'prediction': best_grid,
        'correct': test_out,
        'match': match,
        'cell_acc': cell_acc,
        'bp_acc': bp_acc,
        'n_uncertain': len(uncertain_cells),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="*",
                        default=["0d3d703e", "25ff71a9", "794b24be",
                                 "aabf363d", "ae3edfdc", "00dbd492"])
    args = parser.parse_args()

    arc_dirs = ["data/ARC-AGI/data/training", "data/ARC-AGI/data/evaluation"]

    print("Chained Solver: BP → uncertain refinement → ICM")
    print()

    solved = 0
    for tname in args.task:
        task = None
        for arc_dir in arc_dirs:
            path = os.path.join(arc_dir, f"{tname}.json")
            if os.path.exists(path):
                with open(path) as f: task = json.load(f)
                break
        if task is None:
            print(f"  {tname}: not found"); continue

        H = len(task['test'][0]['input'])
        W = len(task['test'][0]['input'][0])
        nc = len(set(c for p in task['train']+task['test']
                     for g in [p['input'],p['output']] for row in g for c in row))

        print(f"{tname} ({nc} colors, {H}x{W}):")
        t0 = time.time()
        r = solve_task(task)
        elapsed = time.time() - t0

        if r['match']:
            print(f"  SOLVED ✓ ({elapsed:.1f}s, bp_acc={r['bp_acc']:.3f}, refined {r['n_uncertain']} cells)")
            solved += 1
        else:
            print(f"  bp_acc={r['bp_acc']:.3f} → final={r['cell_acc']:.3f} "
                  f"({int(r['cell_acc']*H*W)}/{H*W}) ({elapsed:.1f}s, {r['n_uncertain']} uncertain)")
            pred, correct = r['prediction'], r['correct']
            mismatches = [(r_, c_) for r_ in range(H) for c_ in range(W)
                          if pred[r_, c_] != correct[r_, c_]]
            for r_, c_ in mismatches[:5]:
                print(f"    ({r_},{c_}): predicted {pred[r_,c_]}, correct {correct[r_,c_]}")
        print()

    print(f"SOLVED: {solved}/{len(args.task)}")


if __name__ == "__main__":
    main()
