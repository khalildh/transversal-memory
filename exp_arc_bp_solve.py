"""
exp_arc_bp_solve.py — ARC solver via belief propagation on Plücker MRF.

The multi-transversal scoring defines a pairwise MRF:
  score(grid) = Σ table[adj][color_a][color_b]

Belief propagation finds the minimum-energy grid configuration
without enumerating candidates. O(nc² × H × W × iterations).

Pipeline:
  1. Build multi-embedding transversal score tables (same as fast solver)
  2. Run loopy BP on the grid graph with tables as pairwise potentials
  3. Decode: each cell picks its best color from the converged beliefs
  4. Output = the decoded grid

Usage:
  uv run python exp_arc_bp_solve.py --task ae3edfdc
  uv run python exp_arc_bp_solve.py --task 00dbd492
  uv run python exp_arc_bp_solve.py --all
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


# ── Plücker primitives ──────────────────────────────────────────────────────

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

def emb_row_features(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    in_rh = np.array([np.sum(inp[r]==i) for i in range(N_COLORS)], dtype=np.float32) / W
    in_ru = np.float32(len(set(inp[r].flatten())) == 1)
    in_rn = np.float32(len(set(inp[r].flatten())) / max(W, 1))
    out_rh = np.array([np.sum(out[r]==i) for i in range(N_COLORS)], dtype=np.float32) / W
    out_ru = np.float32(len(set(out[r].flatten())) == 1)
    out_rn = np.float32(len(set(out[r].flatten())) / max(W, 1))
    return np.concatenate([in_oh, out_oh, in_rh, [in_ru, in_rn], out_rh, [out_ru, out_rn]])

def emb_col_features(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    in_ch = np.array([np.sum(inp[:,c]==i) for i in range(N_COLORS)], dtype=np.float32) / H
    in_cu = np.float32(len(set(inp[:,c].flatten())) == 1)
    out_ch = np.array([np.sum(out[:,c]==i) for i in range(N_COLORS)], dtype=np.float32) / H
    out_cu = np.float32(len(set(out[:,c].flatten())) == 1)
    return np.concatenate([in_oh, out_oh, in_ch, [in_cu], out_ch, [out_cu]])

EMBEDDINGS = [
    ('color_only', emb_color_only, 20),
    ('pos_color', emb_pos_color, 22),
    ('row_feat', emb_row_features, 44),
    ('col_feat', emb_col_features, 42),
]


# ── Build pairwise potentials ────────────────────────────────────────────────

def build_potentials(task, test_inp, used_colors):
    """Build pairwise potential tables for the MRF.

    Returns:
      adj_pairs: list of ((r1,c1), (r2,c2)) edges
      potentials: list of (nc, nc) arrays, one per edge
    """
    H, W = test_inp.shape
    nc = len(used_colors)

    adj_pairs = []
    for r in range(H):
        for c in range(W):
            for dr, dc in [(0, 1), (1, 0)]:
                r2, c2 = r + dr, c + dc
                if r2 < H and c2 < W:
                    adj_pairs.append(((r, c), (r2, c2)))

    # Accumulate potentials from all embeddings
    potentials = [np.zeros((nc, nc), dtype=np.float32) for _ in adj_pairs]

    for name, emb_fn, dim in EMBEDDINGS:
        rng_proj = np.random.RandomState(hash(name) % 2**31)
        W1 = rng_proj.randn(4, 2 * dim).astype(np.float32) * 0.1
        W2 = rng_proj.randn(4, 2 * dim).astype(np.float32) * 0.1

        # Transversals from training pairs
        trans = []
        for i, pair in enumerate(task['train']):
            inp, out = np.array(pair['input']), np.array(pair['output'])
            pH, pW = inp.shape
            pair_adj = [(r, c, r+dr, c+dc) for r in range(pH) for c in range(pW)
                        for dr, dc in [(0,1),(1,0)] if r+dr < pH and c+dc < pW]
            lines = []
            for r, c, r2, c2 in pair_adj:
                ea = emb_fn(r, c, inp[r,c], out[r,c], inp, out, pH, pW)
                eb = emb_fn(r2, c2, inp[r2,c2], out[r2,c2], inp, out, pH, pW)
                L = make_line(ea, eb, W1, W2)
                if L is not None:
                    lines.append(L)
            trans.extend(compute_trans(lines, 200, np.random.default_rng(42 + i)))

        if not trans:
            continue

        JTm = J6 @ np.stack(trans).T  # (6, n_trans)

        # Build score table per edge
        for ap_idx, ((r1, c1), (r2, c2)) in enumerate(adj_pairs):
            lt = np.zeros((nc, nc, 6), dtype=np.float32)
            for ia in range(nc):
                for ib in range(nc):
                    ea = emb_fn(r1, c1, test_inp[r1, c1], used_colors[ia],
                                test_inp, test_inp, H, W)
                    eb = emb_fn(r2, c2, test_inp[r2, c2], used_colors[ib],
                                test_inp, test_inp, H, W)
                    L = make_line(ea, eb, W1, W2)
                    if L is not None:
                        lt[ia, ib] = L
            flat_inner = lt.reshape(nc * nc, 6) @ JTm
            potentials[ap_idx] += np.sum(
                np.log(np.abs(flat_inner) + 1e-10),
                axis=1).reshape(nc, nc).astype(np.float32)

    return adj_pairs, potentials


# ── Belief Propagation ───────────────────────────────────────────────────────

def belief_propagation(H, W, nc, adj_pairs, potentials, n_iters=30, damping=0.5):
    """Min-sum loopy BP on grid graph with pairwise potentials.

    We want to MINIMIZE the energy (lower = more incident = better).
    Messages: m_{i→j}(x_j) = min_{x_i} [potential(x_i, x_j) + Σ_{k≠j} m_{k→i}(x_i)]

    Returns: beliefs (H, W, nc) — lower = better for each color at each cell.
    """
    n_cells = H * W

    # Build adjacency: for each cell, list of (neighbor_cell, edge_idx, is_first)
    cell_idx = lambda r, c: r * W + c
    neighbors = [[] for _ in range(n_cells)]
    for ap_idx, ((r1, c1), (r2, c2)) in enumerate(adj_pairs):
        i = cell_idx(r1, c1)
        j = cell_idx(r2, c2)
        neighbors[i].append((j, ap_idx, True))   # i is first in potential
        neighbors[j].append((i, ap_idx, False))   # j is second in potential

    # Messages: msg[i][j] = (nc,) message from cell i to cell j
    # Initialize with small random noise for symmetry breaking
    rng = np.random.RandomState(42)
    messages = {}
    for ap_idx, ((r1, c1), (r2, c2)) in enumerate(adj_pairs):
        i, j = cell_idx(r1, c1), cell_idx(r2, c2)
        messages[(i, j)] = rng.randn(nc).astype(np.float32) * 0.01
        messages[(j, i)] = rng.randn(nc).astype(np.float32) * 0.01

    for iteration in range(n_iters):
        new_messages = {}
        for ap_idx, ((r1, c1), (r2, c2)) in enumerate(adj_pairs):
            i, j = cell_idx(r1, c1), cell_idx(r2, c2)
            pot = potentials[ap_idx]  # (nc, nc) — pot[xi, xj]

            # Message i → j: for each xj, minimize over xi
            # cost(xi) = pot[xi, xj] + sum of incoming messages to i (except from j)
            incoming_i = np.zeros(nc, dtype=np.float32)
            for nb, _, _ in neighbors[i]:
                if nb != j and (nb, i) in messages:
                    incoming_i += messages[(nb, i)]

            # m_{i→j}(xj) = min_{xi} [pot[xi, xj] + incoming_i[xi]]
            cost_i = pot + incoming_i[:, None]  # (nc_xi, nc_xj)
            msg_ij = np.min(cost_i, axis=0)     # (nc_xj,)
            msg_ij -= msg_ij.min()  # normalize

            # Message j → i: for each xi, minimize over xj
            incoming_j = np.zeros(nc, dtype=np.float32)
            for nb, _, _ in neighbors[j]:
                if nb != i and (nb, j) in messages:
                    incoming_j += messages[(nb, j)]

            cost_j = pot + incoming_j[None, :]  # (nc_xi, nc_xj)
            msg_ji = np.min(cost_j, axis=1)     # (nc_xi,)
            msg_ji -= msg_ji.min()

            # Damping
            new_messages[(i, j)] = damping * messages.get((i, j), msg_ij) + (1 - damping) * msg_ij
            new_messages[(j, i)] = damping * messages.get((j, i), msg_ji) + (1 - damping) * msg_ji

        messages = new_messages

    # Compute beliefs: belief[cell](x) = sum of incoming messages
    beliefs = np.zeros((H, W, nc), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            i = cell_idx(r, c)
            for nb, _, _ in neighbors[i]:
                if (nb, i) in messages:
                    beliefs[r, c] += messages[(nb, i)]

    return beliefs


# ── Solver ───────────────────────────────────────────────────────────────────

def solve_task(task, n_iters=30):
    H = len(task['test'][0]['input'])
    W = len(task['test'][0]['input'][0])
    test_inp = np.array(task['test'][0]['input'])
    test_out = np.array(task['test'][0]['output'])
    used_colors = sorted(set(
        c for p in task['train'] + task['test']
        for g in [p['input'], p['output']]
        for row in g for c in row))
    nc = len(used_colors)

    print(f"  Building potentials ({len(EMBEDDINGS)} embeddings)...")
    t0 = time.time()
    adj_pairs, potentials = build_potentials(task, test_inp, used_colors)
    build_time = time.time() - t0

    print(f"  Running BP ({n_iters} iterations, {H}x{W} grid, {nc} colors)...")
    t0 = time.time()
    beliefs = belief_propagation(H, W, nc, adj_pairs, potentials, n_iters)
    bp_time = time.time() - t0

    # Decode: pick best color per cell
    decoded_idx = beliefs.argmin(axis=2)
    decoded = np.array([[used_colors[decoded_idx[r, c]]
                         for c in range(W)] for r in range(H)])

    # Re-decode with unary bias from beliefs
    decoded_idx = beliefs.argmin(axis=2)

    # ICM refinement with BOTH pairwise potentials AND unary beliefs
    print(f"  Running ICM refinement...")
    t0 = time.time()
    color_to_idx = {c: i for i, c in enumerate(used_colors)}
    grid_idx = np.array([[color_to_idx[decoded[r, c]]
                          for c in range(W)] for r in range(H)])

    # Build neighbor lookup: for each cell, list of (edge_idx, is_first)
    cell_edges = [[[] for _ in range(W)] for _ in range(H)]
    for ap_idx, ((r1, c1), (r2, c2)) in enumerate(adj_pairs):
        cell_edges[r1][c1].append((ap_idx, r2, c2, True))
        cell_edges[r2][c2].append((ap_idx, r1, c1, False))

    for icm_iter in range(50):
        changed = 0
        for r in range(H):
            for c in range(W):
                current = grid_idx[r, c]
                # Compute energy contribution of current color
                best_color = current
                best_energy = 0.0
                for ap_idx, nr, nc_, is_first in cell_edges[r][c]:
                    nb_color = grid_idx[nr, nc_]
                    if is_first:
                        best_energy += potentials[ap_idx][current, nb_color]
                    else:
                        best_energy += potentials[ap_idx][nb_color, current]

                # Try all other colors
                for ci in range(len(used_colors)):
                    if ci == current:
                        continue
                    energy = 0.0
                    for ap_idx, nr, nc_, is_first in cell_edges[r][c]:
                        nb_color = grid_idx[nr, nc_]
                        if is_first:
                            energy += potentials[ap_idx][ci, nb_color]
                        else:
                            energy += potentials[ap_idx][nb_color, ci]
                    if energy < best_energy:
                        best_energy = energy
                        best_color = ci

                if best_color != current:
                    grid_idx[r, c] = best_color
                    changed += 1

        if changed == 0:
            break

    icm_time = time.time() - t0
    decoded = np.array([[used_colors[grid_idx[r, c]]
                         for c in range(W)] for r in range(H)])

    match = np.array_equal(decoded, test_out)
    cell_acc = np.mean(decoded == test_out)

    return {
        'prediction': decoded,
        'correct': test_out,
        'match': match,
        'cell_acc': cell_acc,
        'build_time': build_time,
        'bp_time': bp_time,
        'icm_time': icm_time,
        'n_edges': len(adj_pairs),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="*",
                        default=["aabf363d", "ae3edfdc", "00dbd492",
                                 "25ff71a9", "794b24be", "0d3d703e"])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    arc_dirs = ["data/ARC-AGI/data/training", "data/ARC-AGI/data/evaluation"]

    if args.all:
        task_names = []
        for arc_dir in arc_dirs:
            for fname in sorted(os.listdir(arc_dir)):
                if not fname.endswith('.json'): continue
                with open(os.path.join(arc_dir, fname)) as f:
                    task = json.load(f)
                if len(task['train']) < 2: continue
                all_same = all(
                    len(p['input']) == len(p['output']) and
                    len(p['input'][0]) == len(p['output'][0])
                    for p in task['train'] + task['test'])
                if not all_same: continue
                H = len(task['test'][0]['input'])
                W = len(task['test'][0]['input'][0])
                if H > 30 or W > 30: continue
                task_names.append(fname.replace('.json', ''))
    else:
        task_names = args.task

    print(f"Belief Propagation ARC Solver")
    print(f"Embeddings: {len(EMBEDDINGS)}, BP iters: {args.iters}")
    print(f"Tasks: {len(task_names)}")
    print()

    solved = 0
    for tname in task_names:
        # Try both dirs
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
        r = solve_task(task, n_iters=args.iters)

        if r['match']:
            print(f"  SOLVED ✓ (build={r['build_time']:.1f}s, bp={r['bp_time']:.1f}s, icm={r['icm_time']:.1f}s)")
            solved += 1
        else:
            print(f"  cell_acc={r['cell_acc']:.3f} ({int(r['cell_acc']*H*W)}/{H*W} cells)")
            print(f"  build={r['build_time']:.1f}s, bp={r['bp_time']:.1f}s, icm={r['icm_time']:.1f}s")
            # Show a few mismatched cells
            pred, correct = r['prediction'], r['correct']
            mismatches = [(r_, c_) for r_ in range(H) for c_ in range(W)
                          if pred[r_, c_] != correct[r_, c_]]
            if mismatches and len(mismatches) <= 20:
                for r_, c_ in mismatches[:5]:
                    print(f"    ({r_},{c_}): predicted {pred[r_,c_]}, correct {correct[r_,c_]}")
        print()

    print(f"{'='*60}")
    print(f"SOLVED: {solved}/{len(task_names)}")


if __name__ == "__main__":
    main()
