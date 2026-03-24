"""
exp_arc_gen_solve.py — ARC solver via generative transversal retrieval.

Instead of scoring candidates, GENERATE the output directly:
  1. Store training pair lines in P3Memory
  2. For each test cell, query with the test input line
  3. The transversal output encodes the answer
  4. Decode by finding which output color produces the closest line

Usage:
  uv run python exp_arc_gen_solve.py
  uv run python exp_arc_gen_solve.py --task aabf363d
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


def emb_pos_color(r, c, in_c, out_c, inp, out, H, W):
    pos = np.array([r/max(H-1,1), c/max(W-1,1)], dtype=np.float32)
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    return np.concatenate([pos, in_oh, out_oh])


def emb_color_only(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    return np.concatenate([in_oh, out_oh])


EMBEDDINGS = [
    ('color_only', emb_color_only, 20),
    ('pos_color', emb_pos_color, 22),
]


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

    adj_pairs = [(r, c, r+dr, c+dc)
                 for r in range(H) for c in range(W)
                 for dr, dc in [(0, 1), (1, 0)]
                 if r+dr < H and c+dc < W]

    # For each embedding, for each cell, generate the output color
    # by querying transversal memory with the test input line
    votes = np.zeros((H, W, nc), dtype=np.float32)

    for name, emb_fn, dim in EMBEDDINGS:
        rng_proj = np.random.RandomState(hash(name) % 2**31)
        W1 = rng_proj.randn(4, 2 * dim).astype(np.float32) * 0.1
        W2 = rng_proj.randn(4, 2 * dim).astype(np.float32) * 0.1

        # Collect all training pair lines
        train_lines = []
        for pair in task['train']:
            inp, out = np.array(pair['input']), np.array(pair['output'])
            pH, pW = inp.shape
            for r in range(pH):
                for c in range(pW):
                    for dr, dc in [(0, 1), (1, 0)]:
                        r2, c2 = r + dr, c + dc
                        if r2 < pH and c2 < pW:
                            ea = emb_fn(r, c, inp[r,c], out[r,c], inp, out, pH, pW)
                            eb = emb_fn(r2, c2, inp[r2,c2], out[r2,c2], inp, out, pH, pW)
                            L = make_line(ea, eb, W1, W2)
                            if L is not None:
                                train_lines.append(L)

        if len(train_lines) < 4:
            continue

        # For each adjacency pair in the test grid, try each output color
        # combination — query transversal, see which color combo produces
        # the line closest to the transversal
        for r, c, r2, c2 in adj_pairs:
            # Sample multiple transversals from training lines
            rng = np.random.default_rng(42 + r * W + c)
            transversals = []
            for _ in range(50):
                idx = rng.choice(len(train_lines), size=4, replace=False)
                mem = P3Memory()
                mem.store([train_lines[idx[i]] for i in range(3)])
                for T, res in mem.query_generative(train_lines[idx[3]]):
                    n = np.linalg.norm(T)
                    if n > 1e-10 and res < 1e-6:
                        transversals.append(T / n)

            if not transversals:
                continue

            Tm = np.stack(transversals)
            JTm = J6 @ Tm.T  # (6, n_trans)

            # For each candidate color at (r,c) and (r2,c2),
            # compute the line and its incidence with transversals
            for ia, ca in enumerate(used_colors):
                for ib, cb in enumerate(used_colors):
                    ea = emb_fn(r, c, test_inp[r,c], ca, test_inp, test_inp, H, W)
                    eb = emb_fn(r2, c2, test_inp[r2,c2], cb, test_inp, test_inp, H, W)
                    L = make_line(ea, eb, W1, W2)
                    if L is not None:
                        inner = L @ JTm
                        score = np.sum(np.log(np.abs(inner) + 1e-10))
                        # Lower score = more incident = better
                        votes[r, c, ia] += score
                        votes[r2, c2, ib] += score

    # Decode: pick lowest-vote color per cell
    decoded_idx = votes.argmin(axis=2)
    decoded = np.array([[used_colors[decoded_idx[r, c]]
                         for c in range(W)] for r in range(H)])

    match = np.array_equal(decoded, test_out)
    cell_acc = np.mean(decoded == test_out)

    return {
        'prediction': decoded,
        'correct': test_out,
        'match': match,
        'cell_acc': cell_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="*",
                        default=["aabf363d", "ae3edfdc", "25ff71a9",
                                 "794b24be", "0d3d703e", "00dbd492"])
    args = parser.parse_args()

    arc_dirs = ["data/ARC-AGI/data/training", "data/ARC-AGI/data/evaluation"]

    print("Generative Transversal ARC Solver")
    print()

    solved = 0
    for tname in args.task:
        task = None
        for arc_dir in arc_dirs:
            path = os.path.join(arc_dir, f"{tname}.json")
            if os.path.exists(path):
                with open(path) as f:
                    task = json.load(f)
                break
        if task is None:
            print(f"  {tname}: not found")
            continue

        H = len(task['test'][0]['input'])
        W = len(task['test'][0]['input'][0])
        nc = len(set(c for p in task['train']+task['test']
                     for g in [p['input'],p['output']] for row in g for c in row))

        print(f"{tname} ({nc} colors, {H}x{W}):")
        t0 = time.time()
        r = solve_task(task)
        elapsed = time.time() - t0

        if r['match']:
            print(f"  SOLVED ✓ ({elapsed:.1f}s)")
            solved += 1
        else:
            print(f"  cell_acc={r['cell_acc']:.3f} ({int(r['cell_acc']*H*W)}/{H*W}) ({elapsed:.1f}s)")
            pred, correct = r['prediction'], r['correct']
            mismatches = [(r_, c_) for r_ in range(H) for c_ in range(W)
                          if pred[r_, c_] != correct[r_, c_]]
            for r_, c_ in mismatches[:5]:
                print(f"    ({r_},{c_}): predicted {pred[r_,c_]}, correct {correct[r_,c_]}")
        print()

    print(f"SOLVED: {solved}/{len(args.task)}")


if __name__ == "__main__":
    main()
