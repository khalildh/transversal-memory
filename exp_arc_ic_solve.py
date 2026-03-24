"""
exp_arc_ic_solve.py — ARC solver via Interaction Calculus-inspired reduction.

Models each cell as a SUP (superposition) of possible colors.
Adjacent cells interact via Plücker score tables — interactions that
score poorly get eliminated (annihilated). When a cell's superposition
collapses to a single value, it propagates constraints to neighbors.

This is constraint propagation (AC-3) + branching, motivated by the
structural parallel between IC's SUP/DUP/annihilation and Plücker
geometry's transversal/dual-projection/incidence.

Usage:
  uv run python exp_arc_ic_solve.py
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


def build_score_tables(task, test_inp, used_colors):
    H, W = test_inp.shape
    nc = len(used_colors)
    adj_pairs = [(r,c,r+dr,c+dc) for r in range(H) for c in range(W)
                 for dr,dc in [(0,1),(1,0)] if r+dr<H and c+dc<W]
    tables = [np.zeros((nc, nc), dtype=np.float32) for _ in adj_pairs]

    for name, emb_fn, dim in EMBEDDINGS:
        rng = np.random.RandomState(hash(name) % 2**31)
        W1 = rng.randn(4, 2*dim).astype(np.float32) * 0.1
        W2 = rng.randn(4, 2*dim).astype(np.float32) * 0.1
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
            tables[ap_idx] += np.sum(np.log(np.abs(flat)+1e-10),
                                      axis=1).reshape(nc,nc).astype(np.float32)
    return adj_pairs, tables


# ── IC-inspired constraint propagation ───────────────────────────────────────

def ic_solve(H, W, nc, adj_pairs, tables, used_colors):
    """
    IC reduction:
      - Each cell starts as SUP{0, 1, ..., nc-1} (superposition of all colors)
      - INTERACT: for each adj pair, eliminate color combos that score poorly
      - COLLAPSE: when a cell has 1 option, it's determined
      - BRANCH (DUP): when stuck, split on the smallest-domain cell

    Returns the solved grid or best guess.
    """
    # Build neighbor map
    cell_adj = {}  # (r,c) → [(ap_idx, r2, c2, is_first)]
    for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
        cell_adj.setdefault((r,c), []).append((ap_idx, r2, c2, True))
        cell_adj.setdefault((r2,c2), []).append((ap_idx, r, c, False))

    def propagate(domains):
        """AC-3: remove values from domains that have no support."""
        queue = [(r, c) for r in range(H) for c in range(W)]
        changed = True
        while changed:
            changed = False
            for r, c in list(queue):
                if len(domains[r][c]) <= 1:
                    continue
                for ap_idx, nr, nc_, is_first in cell_adj.get((r,c), []):
                    table = tables[ap_idx]
                    nb_domain = domains[nr][nc_]
                    if not nb_domain:
                        continue

                    # For each color in my domain, check if ANY neighbor color
                    # gives a good score (not in the worst quartile)
                    new_domain = set()
                    for my_c in domains[r][c]:
                        # Best score this color can achieve with any neighbor color
                        if is_first:
                            scores = [table[my_c, nb_c] for nb_c in nb_domain]
                        else:
                            scores = [table[nb_c, my_c] for nb_c in nb_domain]
                        if scores:
                            new_domain.add(my_c)

                    if len(new_domain) < len(domains[r][c]):
                        domains[r][c] = new_domain
                        changed = True

        return domains

    def score_assignment(assignment):
        """Score a complete assignment."""
        total = 0.0
        for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
            if assignment[r][c] is not None and assignment[r2][c2] is not None:
                total += tables[ap_idx][assignment[r][c], assignment[r2][c2]]
        return total

    def solve_recursive(domains, depth=0):
        """Branch and bound with constraint propagation."""
        domains = propagate(domains)

        # Check for empty domains (contradiction)
        for r in range(H):
            for c in range(W):
                if len(domains[r][c]) == 0:
                    return None, float('inf')

        # Check if solved (all domains singleton)
        all_single = all(len(domains[r][c]) == 1
                         for r in range(H) for c in range(W))
        if all_single:
            assignment = [[list(domains[r][c])[0] for c in range(W)]
                          for r in range(H)]
            return assignment, score_assignment(assignment)

        # BRANCH (DUP): pick cell with smallest domain > 1
        min_size = nc + 1
        branch_cell = None
        for r in range(H):
            for c in range(W):
                d = len(domains[r][c])
                if 1 < d < min_size:
                    min_size = d
                    branch_cell = (r, c)

        if branch_cell is None:
            return None, float('inf')

        if depth > 20:
            # Too deep — return best guess
            assignment = [[min(domains[r][c]) if domains[r][c] else 0
                           for c in range(W)] for r in range(H)]
            return assignment, score_assignment(assignment)

        r, c = branch_cell
        best_result = None
        best_score = float('inf')

        # Try each color in the domain, sorted by how good they score locally
        local_scores = []
        for color in domains[r][c]:
            s = sum(min(tables[ap][color, nb_c] if is_first else tables[ap][nb_c, color]
                       for nb_c in domains[nr][nc_])
                   for ap, nr, nc_, is_first in cell_adj.get((r,c), [])
                   if domains[nr][nc_])
            local_scores.append((s, color))
        local_scores.sort()

        for _, color in local_scores[:3]:  # try top 3 only for speed
            new_domains = [[set(domains[r_][c_]) for c_ in range(W)]
                           for r_ in range(H)]
            new_domains[r][c] = {color}
            result, score = solve_recursive(new_domains, depth + 1)
            if result is not None and score < best_score:
                best_score = score
                best_result = result

        return best_result, best_score

    # Initialize: all cells start as superposition of all colors
    initial_domains = [[set(range(nc)) for _ in range(W)] for _ in range(H)]

    result, score = solve_recursive(initial_domains)

    if result is None:
        # Fallback: greedy
        result = [[0] * W for _ in range(H)]

    grid = np.array([[used_colors[result[r][c]] for c in range(W)]
                      for r in range(H)])
    return grid


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

    adj_pairs, tables = build_score_tables(task, test_inp, used_colors)
    grid = ic_solve(H, W, nc, adj_pairs, tables, used_colors)

    match = np.array_equal(grid, test_out)
    cell_acc = np.mean(grid == test_out)
    return {
        'prediction': grid,
        'correct': test_out,
        'match': match,
        'cell_acc': cell_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="*",
                        default=["0d3d703e", "25ff71a9", "794b24be",
                                 "aabf363d", "ae3edfdc"])
    args = parser.parse_args()

    arc_dirs = ["data/ARC-AGI/data/training", "data/ARC-AGI/data/evaluation"]

    print("IC-Inspired Constraint Propagation Solver")
    print()

    solved = 0
    for tname in args.task:
        task = None
        for d in arc_dirs:
            p = os.path.join(d, f"{tname}.json")
            if os.path.exists(p):
                with open(p) as f: task = json.load(f)
                break
        if not task: continue

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
