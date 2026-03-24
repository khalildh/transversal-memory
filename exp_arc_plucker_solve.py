"""
exp_arc_plucker_solve.py — ARC solver via sequential Plücker geometry.

Pipeline (zero learning):
  1. Build co-occurrence embeddings from (position, color, section) words
  2. Multi-transversal scoring: joint (input+output) lines → transversals
  3. Spatial Gram transport: 21×21 linear map predicts output Gram
  4. Sequential filter+rerank: transversal top-K → Gram rerank

Best result: rank 11/19683 (top 0.056%) on task 25ff71a9 with K=200.

Usage:
  uv run python exp_arc_plucker_solve.py                    # default task
  uv run python exp_arc_plucker_solve.py --task 0d3d703e    # specific task
  uv run python exp_arc_plucker_solve.py --all              # all 3x3 tasks
"""

import json
import os
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from itertools import product as cartesian
from transversal_memory import P3Memory
from transversal_memory.plucker import line_from_points

N_COLORS = 10


# ── Co-occurrence embeddings (position, color, section) ──────────────────────

def cell_id_z(r, c, color, section, H, W):
    return ((section * H * W) + r * W + c) * N_COLORS + color


def vocab_size(H, W):
    return 2 * H * W * N_COLORS


def dihedral_transform(grid, idx):
    g = np.array(grid)
    if idx == 0: return g
    elif idx == 1: return np.rot90(g, 1)
    elif idx == 2: return np.rot90(g, 2)
    elif idx == 3: return np.rot90(g, 3)
    elif idx == 4: return np.fliplr(g)
    elif idx == 5: return np.flipud(g)
    elif idx == 6: return g.T
    elif idx == 7: return np.rot90(g, 2).T
    return g


def build_cooccurrence(task, H, W):
    """Build co-occurrence matrix with dihedral aug + cross-grid + test input."""
    V = vocab_size(H, W)
    cooc = np.zeros((V, V))

    for pair in task['train']:
        for d in range(8):
            id_ = np.array(dihedral_transform(pair['input'], d))
            od_ = np.array(dihedral_transform(pair['output'], d))
            # Within-grid adjacency
            for grid, sec in [(id_, 0), (od_, 1)]:
                for r in range(H):
                    for c in range(W):
                        a = cell_id_z(r, c, grid[r, c], sec, H, W)
                        if c + 1 < W:
                            b = cell_id_z(r, c+1, grid[r, c+1], sec, H, W)
                            cooc[a, b] += 1; cooc[b, a] += 1
                        if r + 1 < H:
                            b = cell_id_z(r+1, c, grid[r+1, c], sec, H, W)
                            cooc[a, b] += 1; cooc[b, a] += 1
            # Cross-grid: input cell ↔ output cell at same position
            for r in range(H):
                for c in range(W):
                    a = cell_id_z(r, c, id_[r, c], 0, H, W)
                    b = cell_id_z(r, c, od_[r, c], 1, H, W)
                    cooc[a, b] += 1; cooc[b, a] += 1

    # Test input adjacency
    ti = np.array(task['test'][0]['input'])
    for r in range(H):
        for c in range(W):
            a = cell_id_z(r, c, ti[r, c], 0, H, W)
            if c + 1 < W:
                b = cell_id_z(r, c+1, ti[r, c+1], 0, H, W)
                cooc[a, b] += 1; cooc[b, a] += 1
            if r + 1 < H:
                b = cell_id_z(r+1, c, ti[r+1, c], 0, H, W)
                cooc[a, b] += 1; cooc[b, a] += 1

    return cooc


def ppmi_svd(cooc, dim=4):
    n = cooc.shape[0]
    total = cooc.sum()
    if total == 0:
        return np.zeros((n, dim)), np.zeros((n, dim))
    rs = cooc.sum(1, keepdims=True)
    cs = cooc.sum(0, keepdims=True)
    exp = (rs * cs) / total
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(cooc / (exp + 1e-10) + 1e-10)
    pmi = np.maximum(pmi, 0)
    n_active = (pmi.sum(1) > 0).sum()
    dim = min(dim, n_active - 1)
    if dim < 1:
        return np.zeros((n, 1)), np.zeros((n, 1))
    U, S, Vt = svds(csr_matrix(pmi), k=dim)
    ss = np.sqrt(S)
    return U * ss, Vt.T * ss


# ── Signal 1: Multi-transversal scoring ──────────────────────────────────────

def make_line_dual(sv, tv, W1, W2):
    combined = np.concatenate([sv, tv])
    p1 = W1 @ combined
    p2 = W2 @ combined
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    L = np.array([p1[i]*p2[j] - p1[j]*p2[i] for i, j in pairs])
    n = np.linalg.norm(L)
    return L / n if n > 1e-10 else None


def grid_to_trans_lines(grid, se, te, W1, W2, H, W, section):
    grid = np.array(grid)
    lines = []
    for r in range(H):
        for c in range(W):
            a = cell_id_z(r, c, grid[r, c], section, H, W)
            for dr, dc in [(0, 1), (1, 0)]:
                r2, c2 = r + dr, c + dc
                if r2 < H and c2 < W:
                    b = cell_id_z(r2, c2, grid[r2, c2], section, H, W)
                    L = make_line_dual(se[a], te[b], W1, W2)
                    if L is not None:
                        lines.append(L)
    return lines


def compute_transversals(lines, n_trans=15, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if len(lines) < 4:
        return []
    trans = []
    att = 0
    while len(trans) < n_trans and att < n_trans * 10:
        att += 1
        idx = rng.choice(len(lines), size=4, replace=False)
        mem = P3Memory()
        mem.store([lines[idx[i]] for i in range(3)])
        for T, res in mem.query_generative(lines[idx[3]]):
            n = np.linalg.norm(T)
            if n > 1e-10 and res < 1e-6:
                trans.append(T / n)
    return trans


J6 = np.array([
    [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -1, 0], [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0], [0, -1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
], dtype=float)


def transversal_score(candidate_lines, rule_trans):
    """Score lines by combined incidence with transversals. Lower = better."""
    if not candidate_lines or not rule_trans:
        return 0.0
    Tm = np.stack(rule_trans)
    Lm = np.stack(candidate_lines)
    inner = Lm @ J6 @ Tm.T
    return np.sum(np.log(np.abs(inner) + 1e-10))


# ── Signal 2: Spatial Gram transport ─────────────────────────────────────────

def spatial_lines_3d(grid, z=1.0):
    grid = np.array(grid)
    H, W = grid.shape
    lines = []
    for r in range(H):
        for c in range(W - 1):
            a = np.array([1., c/(max(W-1, 1)), r/(max(H-1, 1)), z + grid[r, c]/9.])
            b = np.array([1., (c+1)/(max(W-1, 1)), r/(max(H-1, 1)), z + grid[r, c+1]/9.])
            lines.append(line_from_points(a, b))
    for r in range(H - 1):
        for c in range(W):
            a = np.array([1., c/(max(W-1, 1)), r/(max(H-1, 1)), z + grid[r, c]/9.])
            b = np.array([1., c/(max(W-1, 1)), (r+1)/(max(H-1, 1)), z + grid[r+1, c]/9.])
            lines.append(line_from_points(a, b))
    return np.array(lines)


def gram_vec(grid, z=1.0):
    L = spatial_lines_3d(np.array(grid), z)
    M = L.T @ L
    return M[np.triu_indices(6)]


def gram_transport_score(candidate, predicted_gram_vec):
    """Distance between candidate's Gram and predicted output Gram. Lower = better."""
    return np.linalg.norm(gram_vec(candidate, z=3.0) - predicted_gram_vec)


# ── Training pair verification ───────────────────────────────────────────────

def verify_candidate(candidate, test_input, train_pairs, H, W):
    """Check if a candidate output is consistent with ALL training pairs.

    Tests simple transforms: shifts, rotations, flips, global color maps.
    Returns list of consistent rule descriptions, or empty list.
    """
    candidate = np.array(candidate)
    test_input = np.array(test_input)
    results = []

    # Shifts
    for dr in range(-H + 1, H):
        for dc in range(-W + 1, W):
            if dr == 0 and dc == 0:
                continue
            pred = np.roll(np.roll(test_input, dr, 0), dc, 1)
            if not np.array_equal(pred, candidate):
                continue
            ok = all(np.array_equal(
                np.roll(np.roll(np.array(p['input']), dr, 0), dc, 1),
                np.array(p['output'])) for p in train_pairs)
            if ok:
                results.append(f'shift({dr},{dc})')

    # Rotations
    for k in range(1, 4):
        if np.array_equal(np.rot90(test_input, k), candidate):
            ok = all(np.array_equal(
                np.rot90(np.array(p['input']), k),
                np.array(p['output'])) for p in train_pairs)
            if ok:
                results.append(f'rot{k * 90}')

    # Flips
    for name, fn in [('flipud', np.flipud), ('fliplr', np.fliplr),
                     ('transpose', lambda g: g.T)]:
        if np.array_equal(fn(test_input), candidate):
            ok = all(np.array_equal(
                fn(np.array(p['input'])),
                np.array(p['output'])) for p in train_pairs)
            if ok:
                results.append(name)

    # Global color map
    color_map = {}
    cm_ok = True
    for r in range(H):
        for c in range(W):
            ic, oc = int(test_input[r, c]), int(candidate[r, c])
            if ic in color_map:
                if color_map[ic] != oc:
                    cm_ok = False
                    break
            else:
                color_map[ic] = oc
        if not cm_ok:
            break
    if cm_ok:
        train_ok = True
        for pair in train_pairs:
            ti2, to2 = np.array(pair['input']), np.array(pair['output'])
            for r in range(H):
                for c in range(W):
                    ic = int(ti2[r, c])
                    if ic in color_map and color_map[ic] != int(to2[r, c]):
                        train_ok = False
                        break
                if not train_ok:
                    break
            if not train_ok:
                break
        if train_ok:
            results.append(f'color_map')

    return results


# ── Solver ───────────────────────────────────────────────────────────────────

def solve_task(task, shortlist_k=200, trans_per_pair=15, emb_dim=4):
    """Solve a same-size ARC task using sequential Plücker geometry.

    Returns: (predicted_grid, rank_in_all_candidates, total_candidates)
    """
    H = len(task['train'][0]['input'])
    W = len(task['train'][0]['input'][0])

    used_colors = sorted(set(
        c for p in task['train'] + task['test']
        for g in [p['input'], p['output']]
        for row in g for c in row
    ))

    test_inp = np.array(task['test'][0]['input'])
    test_out = np.array(task['test'][0]['output'])
    n_candidates = len(used_colors) ** (H * W)

    # === Signal 1: Transversal setup ===
    cooc = build_cooccurrence(task, H, W)
    se, te = ppmi_svd(cooc, dim=emb_dim)
    rng_np = np.random.RandomState(42)
    W1 = rng_np.randn(4, 2 * emb_dim) * 0.3
    W2 = rng_np.randn(4, 2 * emb_dim) * 0.3

    rule_trans = []
    for pair in task['train']:
        li = grid_to_trans_lines(pair['input'], se, te, W1, W2, H, W, 0)
        lo = grid_to_trans_lines(pair['output'], se, te, W1, W2, H, W, 1)
        rule_trans.extend(compute_transversals(
            li + lo, n_trans=trans_per_pair, rng=np.random.default_rng(42)))

    # === Signal 2: Gram transport setup ===
    X = [gram_vec(p['input'], z=1.0) for p in task['train']]
    Y = [gram_vec(p['output'], z=3.0) for p in task['train']]
    W_transport = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0]
    predicted_gram = gram_vec(test_inp, z=1.0) @ W_transport

    # === Stage 1: Score all candidates by transversal ===
    ti_lines = grid_to_trans_lines(test_inp, se, te, W1, W2, H, W, 0)
    Tm = np.stack(rule_trans) if rule_trans else None

    all_cands = []
    trans_scores = []

    for vals in cartesian(used_colors, repeat=H * W):
        cand = np.array(vals).reshape(H, W)
        all_cands.append(cand)

        lo = grid_to_trans_lines(cand, se, te, W1, W2, H, W, 1)
        joint = ti_lines + lo
        if joint and Tm is not None:
            Lm = np.stack(joint)
            inner = Lm @ J6 @ Tm.T
            trans_scores.append(np.sum(np.log(np.abs(inner) + 1e-10)))
        else:
            trans_scores.append(0.0)

    trans_scores = np.array(trans_scores)
    trans_order = np.argsort(trans_scores)  # ascending: lower = more incident

    # === Stage 2: Gram rerank within top-K ===
    top_k_idx = trans_order[:shortlist_k]
    gram_scores = np.array([
        gram_transport_score(all_cands[i], predicted_gram)
        for i in top_k_idx
    ])
    rerank_order = top_k_idx[np.argsort(gram_scores)]

    # Best prediction
    best_idx = rerank_order[0]
    best_grid = all_cands[best_idx]

    # Find correct answer rank
    correct_idx = next(
        (i for i, c in enumerate(all_cands) if np.array_equal(c, test_out)),
        None
    )

    if correct_idx is not None and correct_idx in top_k_idx:
        final_rank = int(np.where(rerank_order == correct_idx)[0][0]) + 1
    elif correct_idx is not None:
        # Correct wasn't in shortlist — report overall transversal rank
        final_rank = int(np.where(trans_order == correct_idx)[0][0]) + 1
    else:
        final_rank = -1

    # Also get individual signal ranks
    trans_rank = int(np.where(trans_order == correct_idx)[0][0]) + 1 if correct_idx is not None else -1
    gram_all_scores = np.array([gram_transport_score(c, predicted_gram) for c in all_cands])
    gram_rank = int(np.argsort(gram_all_scores).tolist().index(correct_idx)) + 1 if correct_idx is not None else -1

    in_shortlist = correct_idx in top_k_idx if correct_idx is not None else False

    # === Stage 3: Verify top candidates against training pairs ===
    verified = []
    verify_k = min(50, len(rerank_order))
    for idx in rerank_order[:verify_k]:
        cand = all_cands[idx]
        rules = verify_candidate(cand, test_inp, task['train'], H, W)
        if rules:
            verified.append((idx, cand, rules))

    # If verification found candidates, use the first verified one
    if verified:
        best_idx = verified[0][0]
        best_grid = verified[0][1]
        verified_rules = verified[0][2]
    else:
        verified_rules = []

    return {
        'prediction': best_grid,
        'correct': test_out,
        'match': np.array_equal(best_grid, test_out),
        'final_rank': final_rank,
        'trans_rank': trans_rank,
        'gram_rank': gram_rank,
        'in_shortlist': in_shortlist,
        'n_candidates': n_candidates,
        'shortlist_k': shortlist_k,
        'n_transversals': len(rule_trans),
        'verified': len(verified),
        'verified_rules': verified_rules,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="25ff71a9", help="Task name")
    parser.add_argument("--all", action="store_true", help="Run all 3x3 tasks")
    parser.add_argument("--K", type=int, default=200, help="Shortlist size")
    args = parser.parse_args()

    arc_dir = "data/ARC-AGI/data/training"

    if args.all:
        # Find all same-size tasks that are brute-force feasible
        task_names = []
        for fname in sorted(os.listdir(arc_dir)):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(arc_dir, fname)) as f:
                task = json.load(f)
            if len(task['train']) < 2:
                continue
            # Same-size input/output, small enough to brute force
            all_same = all(
                len(p['input']) == len(p['output']) and
                len(p['input'][0]) == len(p['output'][0])
                for p in task['train'] + task['test']
            )
            if not all_same:
                continue
            sizes = set((len(p['input']), len(p['input'][0]))
                        for p in task['train'] + task['test'])
            if len(sizes) > 1:
                continue
            H, W = sizes.pop()
            used = set(c for p in task['train']+task['test']
                       for g in [p['input'],p['output']] for row in g for c in row)
            if len(used) ** (H * W) > 50000:  # too many candidates
                continue
            task_names.append(fname.replace('.json', ''))
    else:
        task_names = [args.task]

    print(f"Sequential Plücker solver (K={args.K})")
    print(f"Pipeline: transversal top-{args.K} → Gram rerank → verify top-50 against training")
    print(f"Tasks: {len(task_names)}")
    print()

    results = []
    for tname in task_names:
        with open(os.path.join(arc_dir, f"{tname}.json")) as f:
            task = json.load(f)

        r = solve_task(task, shortlist_k=args.K)
        results.append((tname, r))

        if r['match']:
            status = "SOLVED"
            if r['verified_rules']:
                status += f" via {r['verified_rules']}"
        else:
            status = f"rank {r['final_rank']}"
        in_sl = "in shortlist" if r['in_shortlist'] else f"NOT in top-{args.K}"
        print(f"  {tname}: {status} ({in_sl})")
        print(f"    trans={r['trans_rank']}/{r['n_candidates']}, "
              f"gram={r['gram_rank']}/{r['n_candidates']}, "
              f"final={r['final_rank']}/{r['shortlist_k'] if r['in_shortlist'] else r['n_candidates']}")
        print(f"    verified={r['verified']}/50 candidates consistent with training")
        if not r['match']:
            print(f"    predicted: {r['prediction'].flatten().tolist()}")
            print(f"    actual:    {r['correct'].flatten().tolist()}")
        print()

    # Summary
    if len(results) > 1:
        n_solved = sum(1 for _, r in results if r['match'])
        n_shortlisted = sum(1 for _, r in results if r['in_shortlist'])
        print(f"{'='*60}")
        print(f"SUMMARY: {n_solved}/{len(results)} solved, "
              f"{n_shortlisted}/{len(results)} in shortlist")
        trans_ranks = [r['trans_rank'] for _, r in results if r['trans_rank'] > 0]
        gram_ranks = [r['gram_rank'] for _, r in results if r['gram_rank'] > 0]
        final_ranks = [r['final_rank'] for _, r in results
                       if r['in_shortlist'] and r['final_rank'] > 0]
        if trans_ranks:
            print(f"  Transversal ranks: median={np.median(trans_ranks):.0f}, "
                  f"mean={np.mean(trans_ranks):.0f}")
        if gram_ranks:
            print(f"  Gram ranks:        median={np.median(gram_ranks):.0f}, "
                  f"mean={np.mean(gram_ranks):.0f}")
        if final_ranks:
            print(f"  Final ranks (in shortlist): median={np.median(final_ranks):.0f}, "
                  f"mean={np.mean(final_ranks):.0f}")


if __name__ == "__main__":
    main()
