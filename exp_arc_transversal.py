"""
exp_arc_transversal.py — Multi-transversal approach for ARC grids.

Instead of grid adjacency lines (which are degenerate), encode each cell
as an embedding vector and use the word-association pipeline:
  1. Cell embedding: learnable or fixed encoding of (row, col, color)
  2. Plücker lines from cell pairs via dual projection
  3. Multi-transversal: sample 4-tuples of lines, find transversals
  4. Score candidates by combined Plücker inner product

Test on a single ARC task to see if transversals exist and are meaningful.

Usage:
  uv run python exp_arc_transversal.py
"""

import json
import numpy as np
from transversal_memory import P3Memory, plucker_inner
from transversal_memory.plucker import random_projection_dual


# ── Cell embeddings ──────────────────────────────────────────────────────────

def cell_embedding(r, c, color, H, W, dim=32):
    """Encode a grid cell as a vector in R^dim.

    Uses a mix of position and color features spread across dim dimensions
    so that different cells produce linearly independent vectors.
    """
    rng = np.random.RandomState(42)  # fixed random features

    # Normalized position
    x = c / max(W - 1, 1)
    y = r / max(H - 1, 1)

    # One-hot color (10 colors)
    color_vec = np.zeros(10)
    color_vec[int(color)] = 1.0

    # Combine: [x, y, color_onehot, random_features]
    base = np.concatenate([[x, y], color_vec])  # 12 dims

    # Project to dim via random matrix (fixed seed)
    proj = rng.randn(len(base), dim) * 0.3
    vec = base @ proj

    # Add position-color interaction features
    interact = rng.randn(10, dim) * 0.2
    vec += interact[int(color)] * (x + y + 1)

    # Normalize
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec


def grid_to_embeddings(grid):
    """Convert grid to dict of {(r,c): embedding}."""
    grid = np.array(grid)
    H, W = grid.shape
    embeddings = {}
    for r in range(H):
        for c in range(W):
            embeddings[(r, c)] = cell_embedding(r, c, grid[r, c], H, W)
    return embeddings


# ── Plücker lines from cell pairs ───────────────────────────────────────────

def make_cell_line(emb1, emb2, W1, W2):
    """Make a Plücker line from two cell embeddings via dual projection."""
    combined = np.concatenate([emb1, emb2])
    p1 = W1 @ combined  # R^4
    p2 = W2 @ combined  # R^4

    # Exterior product
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    L = np.array([p1[i]*p2[j] - p1[j]*p2[i] for i, j in pairs])
    norm = np.linalg.norm(L)
    if norm < 1e-10:
        return None
    return L / norm


def grid_lines(grid, W1, W2):
    """Make Plücker lines from all adjacent cell pairs."""
    grid = np.array(grid)
    H, W = grid.shape
    embs = grid_to_embeddings(grid)

    lines = []
    labels = []
    for r in range(H):
        for c in range(W):
            # Horizontal neighbor
            if c + 1 < W:
                L = make_cell_line(embs[(r,c)], embs[(r,c+1)], W1, W2)
                if L is not None:
                    lines.append(L)
                    labels.append(f"h({r},{c})-({r},{c+1})")
            # Vertical neighbor
            if r + 1 < H:
                L = make_cell_line(embs[(r,c)], embs[(r+1,c)], W1, W2)
                if L is not None:
                    lines.append(L)
                    labels.append(f"v({r},{c})-({r+1},{c})")
    return lines, labels


# ── Multi-transversal ────────────────────────────────────────────────────────

def compute_transversals(lines, n_transversals=20, rng=None):
    """Sample 4-tuples of lines, find transversals of each."""
    if rng is None:
        rng = np.random.default_rng(42)

    if len(lines) < 4:
        return []

    transversals = []
    attempts = 0
    max_attempts = n_transversals * 10

    while len(transversals) < n_transversals and attempts < max_attempts:
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


def score_lines_against_transversals(candidate_lines, transversals):
    """Score each candidate line by combined inner product with all transversals.
    Lower = more incident = more related."""
    if not transversals:
        return np.ones(len(candidate_lines))

    T_mat = np.stack(transversals)  # (n_trans, 6)
    L_mat = np.stack(candidate_lines)  # (n_cand, 6)

    # Plücker inner product via J6 (Hodge dual)
    J6 = np.array([
        [0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
        [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],
    ], dtype=float)

    # inner[i,j] = L_mat[i] @ J6 @ T_mat[j]
    inner = L_mat @ J6 @ T_mat.T  # (n_cand, n_trans)

    # Combined score: sum of log(|inner| + eps) — lower = more incident
    scores = np.sum(np.log(np.abs(inner) + 1e-10), axis=1)
    return scores


# ── Main: test on one ARC task ──────────────────────────────────────────────

def main():
    DIM = 32
    rng = np.random.RandomState(42)
    W1 = rng.randn(4, 2 * DIM) * 0.1
    W2 = rng.randn(4, 2 * DIM) * 0.1

    with open("data/ARC-AGI/data/training/25ff71a9.json") as f:
        task = json.load(f)

    print("Task: 25ff71a9 (3 colors, spatial transform)")
    print(f"Train pairs: {len(task['train'])}, Test pairs: {len(task['test'])}")

    # Step 1: Get lines from each training pair
    print("\n=== Lines from training pairs ===")
    for i, pair in enumerate(task['train']):
        lines_in, _ = grid_lines(pair['input'], W1, W2)
        lines_out, _ = grid_lines(pair['output'], W1, W2)
        print(f"  Train {i}: {len(lines_in)} input lines, {len(lines_out)} output lines")

    # Step 2: Find transversals within input lines of first training pair
    pair = task['train'][0]
    lines_in, labels_in = grid_lines(pair['input'], W1, W2)
    lines_out, labels_out = grid_lines(pair['output'], W1, W2)

    print(f"\n=== Transversals within input lines ===")
    trans_in = compute_transversals(lines_in, n_transversals=20, rng=np.random.default_rng(42))
    print(f"  Input transversals: {len(trans_in)}")

    print(f"\n=== Transversals within output lines ===")
    trans_out = compute_transversals(lines_out, n_transversals=20, rng=np.random.default_rng(42))
    print(f"  Output transversals: {len(trans_out)}")

    print(f"\n=== Transversals within joint (input+output) lines ===")
    joint_lines = lines_in + lines_out
    trans_joint = compute_transversals(joint_lines, n_transversals=20, rng=np.random.default_rng(42))
    print(f"  Joint transversals: {len(trans_joint)}")

    # Step 3: Cross-grid transversals (2 input + 2 output lines)
    print(f"\n=== Cross-grid transversals (2 input + 2 output) ===")
    cross_trans = []
    n_attempts = 0
    cross_rng = np.random.default_rng(42)
    while len(cross_trans) < 20 and n_attempts < 200:
        n_attempts += 1
        idx_in = cross_rng.choice(len(lines_in), size=2, replace=False)
        idx_out = cross_rng.choice(len(lines_out), size=2, replace=False)
        four = [lines_in[idx_in[0]], lines_in[idx_in[1]],
                lines_out[idx_out[0]], lines_out[idx_out[1]]]

        mem = P3Memory()
        mem.store(four[:3])
        tvs = mem.query_generative(four[3])
        for T, resid in tvs:
            norm = np.linalg.norm(T)
            if norm > 1e-10 and resid < 1e-6:
                cross_trans.append(T / norm)
    print(f"  Cross transversals: {len(cross_trans)} (from {n_attempts} attempts)")

    # Step 4: Consistency check — are transversal Grams consistent across training pairs?
    if trans_joint:
        print(f"\n=== Joint transversal Gram consistency ===")
        trans_grams = []
        for pi, pair in enumerate(task['train']):
            li, _ = grid_lines(pair['input'], W1, W2)
            lo, _ = grid_lines(pair['output'], W1, W2)
            tj = compute_transversals(li + lo, n_transversals=20, rng=np.random.default_rng(42))
            if tj:
                TJ = np.array(tj)
                M = TJ.T @ TJ
                trans_grams.append(M[np.triu_indices(6)])
                print(f"  Train {pi}: {len(tj)} transversals")

        if len(trans_grams) >= 2:
            tg = np.array(trans_grams)
            mean_tg = tg.mean(axis=0)
            for i, g in enumerate(trans_grams):
                err = np.linalg.norm(g - mean_tg) / (np.linalg.norm(mean_tg) + 1e-10)
                print(f"  Train {i} deviation: {err:.4f}")

    # Step 5: Multi-transversal scoring
    # Use training joint transversals to score test output lines
    if trans_joint:
        print(f"\n=== Multi-transversal scoring ===")
        # Average transversal Gram from training pairs as the rule
        mean_trans_gram = np.mean(trans_grams, axis=0) if trans_grams else None

        for ti, test in enumerate(task['test']):
            li_test, _ = grid_lines(test['input'], W1, W2)
            lo_test, _ = grid_lines(test['output'], W1, W2)

            # Score test output lines against training transversals
            scores_out = score_lines_against_transversals(lo_test, trans_joint)
            scores_in = score_lines_against_transversals(li_test, trans_joint)

            print(f"  Test {ti}:")
            print(f"    Output lines mean score: {np.mean(scores_out):.4f} (lower=more incident)")
            print(f"    Input lines mean score:  {np.mean(scores_in):.4f}")
            print(f"    Output more incident? {np.mean(scores_out) < np.mean(scores_in)}")

            # Test joint transversal Gram
            tj_test = compute_transversals(li_test + lo_test, n_transversals=20, rng=np.random.default_rng(42))
            if tj_test and mean_trans_gram is not None:
                TT = np.array(tj_test)
                M_test = TT.T @ TT
                test_tg = M_test[np.triu_indices(6)]
                err = np.linalg.norm(mean_trans_gram - test_tg) / (np.linalg.norm(test_tg) + 1e-10)
                print(f"    Joint transversal Gram prediction err: {err:.4f}")


if __name__ == "__main__":
    main()
