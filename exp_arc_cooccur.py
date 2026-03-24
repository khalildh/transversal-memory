"""
exp_arc_cooccur.py — Co-occurrence based Plücker lines for ARC grids.

Treats colors as "words" and adjacency as "co-occurrence."
Builds PPMI + SVD embeddings from grid adjacency patterns,
then uses multi-transversal pipeline from the word association approach.

Usage:
  uv run python exp_arc_cooccur.py
"""

import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from transversal_memory import P3Memory
from transversal_memory.plucker import plucker_inner


N_COLORS = 10


# ── Co-occurrence from grids (position+color as "words") ─────────────────────

def cell_id(r, c, color, H, W):
    """Unique ID for (row, col, color) tuple."""
    return (r * W + c) * N_COLORS + color


def n_cell_ids(H, W):
    return H * W * N_COLORS


def grid_cooccurrence(grid, vocab_size):
    """Build (position,color) co-occurrence matrix from adjacencies."""
    grid = np.array(grid)
    H, W = grid.shape
    cooc = np.zeros((vocab_size, vocab_size))
    for r in range(H):
        for c in range(W):
            a = cell_id(r, c, grid[r, c], H, W)
            if c + 1 < W:
                b = cell_id(r, c+1, grid[r, c+1], H, W)
                cooc[a, b] += 1
                cooc[b, a] += 1
            if r + 1 < H:
                b = cell_id(r+1, c, grid[r+1, c], H, W)
                cooc[a, b] += 1
                cooc[b, a] += 1
    return cooc


def grids_cooccurrence(grids, H, W):
    """Accumulate co-occurrence across multiple grids."""
    vocab_size = n_cell_ids(H, W)
    cooc = np.zeros((vocab_size, vocab_size))
    for g in grids:
        cooc += grid_cooccurrence(g, vocab_size)
    return cooc


def ppmi_svd(cooc, dim=4):
    """PPMI + SVD on co-occurrence matrix → embeddings."""
    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)
    total = cooc.sum()

    n = cooc.shape[0]
    if total == 0:
        return np.zeros((n, dim)), np.zeros((n, dim))

    expected = (row_sum * col_sum) / total
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(cooc / (expected + 1e-10) + 1e-10)
    pmi = np.maximum(pmi, 0)

    # Only keep rows/cols with any signal
    active = pmi.sum(axis=1) > 0
    n_active = active.sum()
    dim = min(dim, n_active - 1)
    if dim < 1:
        return np.zeros((n, 1)), np.zeros((n, 1))

    U, S, Vt = svds(csr_matrix(pmi), k=dim)
    sqrt_S = np.sqrt(S)
    src_emb = U * sqrt_S
    tgt_emb = Vt.T * sqrt_S

    return src_emb, tgt_emb


# ── Plücker lines from color embeddings ──────────────────────────────────────

def make_line_dual(src_vec, tgt_vec, W1, W2):
    """Plücker line from two embeddings via dual projection."""
    combined = np.concatenate([src_vec, tgt_vec])
    p1 = W1 @ combined
    p2 = W2 @ combined

    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    L = np.array([p1[i]*p2[j] - p1[j]*p2[i] for i, j in pairs])
    norm = np.linalg.norm(L)
    if norm < 1e-10:
        return None
    return L / norm


def grid_to_lines(grid, src_emb, tgt_emb, W1, W2, H_ref, W_ref):
    """Convert grid adjacencies to Plücker lines using (pos,color) embeddings."""
    grid = np.array(grid)
    H, W = grid.shape
    lines = []
    for r in range(H):
        for c in range(W):
            id_a = cell_id(r, c, grid[r, c], H_ref, W_ref)
            # Horizontal
            if c + 1 < W:
                id_b = cell_id(r, c+1, grid[r, c+1], H_ref, W_ref)
                L = make_line_dual(src_emb[id_a], tgt_emb[id_b], W1, W2)
                if L is not None:
                    lines.append(L)
            # Vertical
            if r + 1 < H:
                id_b = cell_id(r+1, c, grid[r+1, c], H_ref, W_ref)
                L = make_line_dual(src_emb[id_a], tgt_emb[id_b], W1, W2)
                if L is not None:
                    lines.append(L)
    return lines


# ── Multi-transversal ────────────────────────────────────────────────────────

def compute_transversals(lines, n_transversals=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if len(lines) < 4:
        return []

    transversals = []
    attempts = 0
    while len(transversals) < n_transversals and attempts < n_transversals * 10:
        attempts += 1
        idx = rng.choice(len(lines), size=4, replace=False)
        four = [lines[i] for i in idx]

        mem = P3Memory()
        mem.store(four[:3])
        tvs = mem.query_generative(four[3])

        for T, resid in tvs:
            norm = np.linalg.norm(T)
            if norm > 1e-10 and resid < 1e-6:
                transversals.append(T / norm)

    return transversals


def score_against_transversals(candidate_lines, transversals):
    """Score lines by combined incidence with transversals. Lower = more incident."""
    if not transversals or not candidate_lines:
        return np.zeros(len(candidate_lines))

    J6 = np.array([
        [0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
        [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],
    ], dtype=float)

    T_mat = np.stack(transversals)
    L_mat = np.stack(candidate_lines)
    inner = L_mat @ J6 @ T_mat.T
    scores = np.sum(np.log(np.abs(inner) + 1e-10), axis=1)
    return scores


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open("data/ARC-AGI/data/training/25ff71a9.json") as f:
        task = json.load(f)

    print("Task: 25ff71a9 (3 colors, spatial transform)")
    print(f"Train: {len(task['train'])}, Test: {len(task['test'])}")

    # Get grid dimensions (assume all same size for this task)
    H = len(task['train'][0]['input'])
    W_grid = len(task['train'][0]['input'][0])
    vocab_size = n_cell_ids(H, W_grid)
    print(f"Grid: {H}x{W_grid}, vocab: {vocab_size} (pos,color) words")

    # Step 1: Build co-occurrence from ALL training grids (input + output)
    all_grids = []
    for pair in task['train']:
        all_grids.append(pair['input'])
        all_grids.append(pair['output'])

    cooc = grids_cooccurrence(all_grids, H, W_grid)
    n_nonzero = (cooc > 0).sum()
    print(f"\nCo-occurrence: {n_nonzero} nonzero entries in {vocab_size}x{vocab_size} matrix")

    # Step 2: PPMI + SVD → (pos,color) embeddings
    DIM = 4
    src_emb, tgt_emb = ppmi_svd(cooc, dim=DIM)
    print(f"Embeddings: dim={DIM}, shape={src_emb.shape}")

    # Step 3: Dual projection matrices
    rng = np.random.RandomState(42)
    W1 = rng.randn(4, 2 * DIM) * 0.3
    W2 = rng.randn(4, 2 * DIM) * 0.3

    # Step 4: Grid → Plücker lines
    print(f"\n=== Lines per training pair ===")
    for i, pair in enumerate(task['train']):
        lines_in = grid_to_lines(pair['input'], src_emb, tgt_emb, W1, W2, H, W_grid)
        lines_out = grid_to_lines(pair['output'], src_emb, tgt_emb, W1, W2, H, W_grid)
        print(f"  Train {i}: {len(lines_in)} input, {len(lines_out)} output")

    # Step 5: Multi-transversal on joint lines
    print(f"\n=== Multi-transversal (joint lines per training pair) ===")
    trans_grams = []
    for i, pair in enumerate(task['train']):
        lines_in = grid_to_lines(pair['input'], src_emb, tgt_emb, W1, W2, H, W_grid)
        lines_out = grid_to_lines(pair['output'], src_emb, tgt_emb, W1, W2, H, W_grid)
        joint = lines_in + lines_out
        trans = compute_transversals(joint, n_transversals=30, rng=np.random.default_rng(42 + i))
        print(f"  Train {i}: {len(trans)} transversals from {len(joint)} joint lines")

        if trans:
            TM = np.array(trans)
            M = TM.T @ TM
            trans_grams.append(M[np.triu_indices(6)])

    # Consistency
    if len(trans_grams) >= 2:
        tg = np.array(trans_grams)
        mean_tg = tg.mean(axis=0)
        print(f"\n  Transversal Gram consistency:")
        for i, g in enumerate(trans_grams):
            err = np.linalg.norm(g - mean_tg) / (np.linalg.norm(mean_tg) + 1e-10)
            print(f"    Train {i}: deviation = {err:.4f}")

    # Step 6: Test prediction
    print(f"\n=== Test prediction ===")
    # Use first training pair's transversals as the rule
    pair0 = task['train'][0]
    lines_in0 = grid_to_lines(pair0['input'], src_emb, tgt_emb, W1, W2, H, W_grid)
    lines_out0 = grid_to_lines(pair0['output'], src_emb, tgt_emb, W1, W2, H, W_grid)
    rule_trans = compute_transversals(lines_in0 + lines_out0, n_transversals=30,
                                      rng=np.random.default_rng(42))

    for ti, test in enumerate(task['test']):
        lines_test_in = grid_to_lines(test['input'], src_emb, tgt_emb, W1, W2, H, W_grid)
        lines_test_out = grid_to_lines(test['output'], src_emb, tgt_emb, W1, W2, H, W_grid)

        # Score test lines against rule transversals
        if rule_trans and lines_test_out and lines_test_in:
            scores_out = score_against_transversals(lines_test_out, rule_trans)
            scores_in = score_against_transversals(lines_test_in, rule_trans)

            print(f"  Test {ti}:")
            print(f"    Output lines mean incidence: {np.mean(scores_out):.4f}")
            print(f"    Input lines mean incidence:  {np.mean(scores_in):.4f}")
            print(f"    Output more incident? {np.mean(scores_out) < np.mean(scores_in)}")

        # Test transversal Gram
        test_joint = lines_test_in + lines_test_out
        test_trans = compute_transversals(test_joint, n_transversals=30,
                                          rng=np.random.default_rng(42))
        if test_trans and trans_grams:
            TT = np.array(test_trans)
            M_test = TT.T @ TT
            test_tg = M_test[np.triu_indices(6)]
            err = np.linalg.norm(mean_tg - test_tg) / (np.linalg.norm(test_tg) + 1e-10)
            print(f"    Transversal Gram prediction err: {err:.4f}")

    # Step 7: Can we use transversals to discriminate correct vs wrong output?
    print(f"\n=== Discriminating correct vs wrong output ===")
    test = task['test'][0]
    test_inp = np.array(test['input'])
    test_out_correct = np.array(test['output'])

    # Generate some wrong outputs
    wrong_outputs = []
    rng_w = np.random.RandomState(123)
    # Wrong 1: random grid
    wrong_outputs.append(("random", rng_w.randint(0, 3, test_out_correct.shape)))
    # Wrong 2: copy input
    wrong_outputs.append(("copy_input", test_inp.copy()))
    # Wrong 3: all zeros
    wrong_outputs.append(("all_zeros", np.zeros_like(test_out_correct)))
    # Wrong 4: swap colors
    swapped = test_out_correct.copy()
    swapped[swapped == 0] = 99
    swapped[swapped == 1] = 0
    swapped[swapped == 99] = 1
    wrong_outputs.append(("swapped", swapped))

    if rule_trans:
        lines_correct = grid_to_lines(test_out_correct, src_emb, tgt_emb, W1, W2, H, W_grid)
        if lines_correct:
            score_correct = np.mean(score_against_transversals(lines_correct, rule_trans))
            print(f"  Correct output:  score = {score_correct:.4f}")

        for name, wrong in wrong_outputs:
            lines_wrong = grid_to_lines(wrong, src_emb, tgt_emb, W1, W2, H, W_grid)
            if lines_wrong:
                score_wrong = np.mean(score_against_transversals(lines_wrong, rule_trans))
                print(f"  {name:15s}: score = {score_wrong:.4f} {'✓' if score_wrong > score_correct else '✗'}")


if __name__ == "__main__":
    main()
