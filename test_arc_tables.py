"""
test_arc_tables.py — Correctness tests comparing C and Python implementations.

Usage:
    uv run python test_arc_tables.py
"""

import numpy as np
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Python reference implementations
from exp_arc_fast_solve import (
    emb_hist_color, emb_color_only, emb_pos_color, emb_all,
    emb_row_features, emb_col_features, emb_color_count, emb_diagonal,
    precompute_cell_embeddings, build_score_tables_vec,
    EMBEDDINGS, N_COLORS, J6, PLUCKER_PAIRS,
)

# Import C wrapper
from arc_tables_wrapper import (
    precompute_cell_embeddings_c, build_score_tables_c,
    build_hist_score_tables_c, get_embedding_dim, EMB_NAME_TO_ID,
)


def make_test_grid():
    """Create a small test grid and parameters."""
    rng = np.random.RandomState(42)
    H, W = 3, 3
    inp = rng.randint(0, 5, size=(H, W))
    used_colors = sorted(set(inp.flatten()))
    nc = len(used_colors)
    return inp, used_colors, nc, H, W


def test_embedding_dims():
    """Verify C reports correct dimensions for all embedding types."""
    print("Testing embedding dimensions...")
    for name, _, expected_dim in EMBEDDINGS:
        c_dim = get_embedding_dim(name)
        assert c_dim == expected_dim, f"{name}: C dim={c_dim}, expected={expected_dim}"
    print("  PASS: all 8 embedding dimensions match")


def test_precompute_embeddings():
    """Compare C vs Python precomputed embeddings for all 8 types."""
    print("Testing precompute_cell_embeddings...")
    inp, used_colors, nc, H, W = make_test_grid()

    py_emb_funcs = {
        'hist_color': emb_hist_color,
        'color_only': emb_color_only,
        'pos_color': emb_pos_color,
        'all': emb_all,
        'row_feat': emb_row_features,
        'col_feat': emb_col_features,
        'color_count': emb_color_count,
        'diagonal': emb_diagonal,
    }

    for name, emb_fn, dim in EMBEDDINGS:
        # Python reference
        py_embs = precompute_cell_embeddings(emb_fn, inp, used_colors, H, W)

        # C implementation
        c_embs = precompute_cell_embeddings_c(name, inp, used_colors, H, W)

        assert py_embs.shape == c_embs.shape, \
            f"{name}: shape mismatch: py={py_embs.shape} vs c={c_embs.shape}"

        max_diff = np.max(np.abs(py_embs - c_embs))
        if max_diff > 1e-6:
            # Find first mismatch
            idx = np.unravel_index(np.argmax(np.abs(py_embs - c_embs)), py_embs.shape)
            print(f"  FAIL {name}: max_diff={max_diff:.8f} at index {idx}")
            print(f"    Python: {py_embs[idx[0], idx[1], :]}")
            print(f"    C:      {c_embs[idx[0], idx[1], :]}")
            return False
        print(f"  PASS {name}: max_diff={max_diff:.2e}")

    return True


def test_build_score_tables():
    """Compare C vs Python score table building."""
    print("Testing build_score_tables...")
    inp, used_colors, nc, H, W = make_test_grid()

    adj_pairs = [
        (r, c, r + dr, c + dc)
        for r in range(H) for c in range(W)
        for dr, dc in [(0, 1), (1, 0)]
        if r + dr < H and c + dc < W
    ]

    for name, emb_fn, dim in EMBEDDINGS:
        # Generate random projection matrices (same seed as solver)
        rng_proj = np.random.RandomState(hash(name) % 2**31)
        W1 = rng_proj.randn(4, 2 * dim).astype(np.float32) * 0.1
        W2 = rng_proj.randn(4, 2 * dim).astype(np.float32) * 0.1

        # Create some fake transversals
        rng_t = np.random.RandomState(123)
        n_trans = 50
        trans = rng_t.randn(n_trans, 6).astype(np.float32)
        # Normalize them
        norms = np.linalg.norm(trans, axis=1, keepdims=True)
        trans = trans / norms
        JTm = (J6.astype(np.float64) @ trans.T.astype(np.float64)).astype(np.float32)

        # Python reference
        py_embs = precompute_cell_embeddings(emb_fn, inp, used_colors, H, W)
        py_scores = build_score_tables_vec(py_embs, adj_pairs, W1, W2, JTm, nc, H, W)

        # C implementation
        c_embs = precompute_cell_embeddings_c(name, inp, used_colors, H, W)
        c_scores = build_score_tables_c(c_embs, adj_pairs, W1, W2, JTm, nc, H, W)

        assert py_scores.shape == c_scores.shape, \
            f"{name}: shape mismatch: py={py_scores.shape} vs c={c_scores.shape}"

        # Allow slightly larger tolerance due to float32 accumulation differences.
        # Python vectorized path accumulates differently than C element-wise path,
        # so for 200 transversals we can see ~1e-3 relative drift.
        max_diff = np.max(np.abs(py_scores - c_scores))
        rel_diff = max_diff / (np.max(np.abs(py_scores)) + 1e-10)
        if rel_diff > 1e-3:
            idx = np.unravel_index(np.argmax(np.abs(py_scores - c_scores)), py_scores.shape)
            print(f"  FAIL {name}: max_diff={max_diff:.6f} rel_diff={rel_diff:.6f} at {idx}")
            print(f"    Python: {py_scores[idx]:.8f}")
            print(f"    C:      {c_scores[idx]:.8f}")
            return False
        print(f"  PASS {name}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")

    return True


def test_hist_score_tables():
    """Compare C vs Python histogram score table building."""
    print("Testing build_hist_score_tables...")
    inp, used_colors, nc, H, W = make_test_grid()
    inp_hist = np.array([np.sum(inp == c) for c in range(N_COLORS)], dtype=np.float32)
    inp_size = max(inp.size, 1)

    adj_pairs = [
        (r, c, r + dr, c + dc)
        for r in range(H) for c in range(W)
        for dr, dc in [(0, 1), (1, 0)]
        if r + dr < H and c + dc < W
    ]
    n_adj = len(adj_pairs)

    # Use hist_color embedding
    name = 'hist_color'
    dim = 30
    rng_proj = np.random.RandomState(hash(name) % 2**31)
    W1 = rng_proj.randn(4, 2 * dim).astype(np.float32) * 0.1
    W2 = rng_proj.randn(4, 2 * dim).astype(np.float32) * 0.1

    rng_t = np.random.RandomState(123)
    n_trans = 50
    trans = rng_t.randn(n_trans, 6).astype(np.float32)
    norms = np.linalg.norm(trans, axis=1, keepdims=True)
    trans = trans / norms
    JTm = (J6.astype(np.float64) @ trans.T.astype(np.float64)).astype(np.float32)

    # Test a few histogram diffs
    for seed in [0, 1, 2]:
        rng_h = np.random.RandomState(seed)
        # Random output histogram
        out_h = np.zeros(N_COLORS, dtype=np.float32)
        for ci in range(nc):
            out_h[used_colors[ci]] = rng_h.randint(0, H * W)
        diff = (out_h - inp_hist) / inp_size

        # Python reference: build embeddings manually for hist_color with this diff
        py_embs = np.zeros((H * W, nc, 30), dtype=np.float32)
        for r in range(H):
            for c in range(W):
                in_c = int(inp[r, c])
                for ci in range(nc):
                    in_oh = np.zeros(N_COLORS, dtype=np.float32)
                    in_oh[in_c] = 1.0
                    out_oh = np.zeros(N_COLORS, dtype=np.float32)
                    out_oh[used_colors[ci]] = 1.0
                    py_embs[r * W + c, ci] = np.concatenate([in_oh, out_oh, diff])

        py_scores = build_score_tables_vec(py_embs, adj_pairs, W1, W2, JTm, nc, H, W)

        # C implementation
        c_scores = build_hist_score_tables_c(
            diff, inp, used_colors, adj_pairs,
            nc, H, W, W1, W2, JTm, n_trans,
        )

        max_diff = np.max(np.abs(py_scores - c_scores))
        rel_diff = max_diff / (np.max(np.abs(py_scores)) + 1e-10)
        if rel_diff > 1e-4:
            idx = np.unravel_index(np.argmax(np.abs(py_scores - c_scores)), py_scores.shape)
            print(f"  FAIL hist seed={seed}: max_diff={max_diff:.6f} rel={rel_diff:.6f} at {idx}")
            print(f"    Python: {py_scores[idx]:.8f}")
            print(f"    C:      {c_scores[idx]:.8f}")
            return False
        print(f"  PASS hist seed={seed}: max_diff={max_diff:.2e}, rel={rel_diff:.2e}")

    return True


def test_performance():
    """Quick performance comparison."""
    import time
    inp, used_colors, nc, H, W = make_test_grid()
    # Use larger grid for timing
    H2, W2 = 5, 5
    inp2 = np.random.RandomState(99).randint(0, 6, size=(H2, W2))
    uc2 = [0, 1, 2, 3, 4, 5]
    nc2 = len(uc2)
    adj_pairs = [
        (r, c, r + dr, c + dc)
        for r in range(H2) for c in range(W2)
        for dr, dc in [(0, 1), (1, 0)]
        if r + dr < H2 and c + dc < W2
    ]

    name = 'color_only'
    dim = 20
    rng_proj = np.random.RandomState(hash(name) % 2**31)
    proj_W1 = rng_proj.randn(4, 2 * dim).astype(np.float32) * 0.1
    proj_W2 = rng_proj.randn(4, 2 * dim).astype(np.float32) * 0.1
    rng_t = np.random.RandomState(123)
    n_trans = 200
    trans = rng_t.randn(n_trans, 6).astype(np.float32)
    norms = np.linalg.norm(trans, axis=1, keepdims=True)
    trans = trans / norms
    JTm = (J6.astype(np.float64) @ trans.T.astype(np.float64)).astype(np.float32)

    # Filter out nan/inf from JTm
    valid = np.all(np.isfinite(JTm), axis=0)
    if not valid.all():
        JTm = np.ascontiguousarray(JTm[:, valid])
        n_trans = JTm.shape[1]

    # Python timing
    t0 = time.time()
    for _ in range(5):
        py_embs = precompute_cell_embeddings(emb_color_only, inp2, uc2, H2, W2)
        py_scores = build_score_tables_vec(
            py_embs, adj_pairs, proj_W1, proj_W2, JTm, nc2, H2, W2)
    py_time = (time.time() - t0) / 5

    # C timing
    t0 = time.time()
    for _ in range(5):
        c_embs = precompute_cell_embeddings_c(name, inp2, uc2, H2, W2)
        c_scores = build_score_tables_c(
            c_embs, adj_pairs, proj_W1, proj_W2, JTm, nc2, H2, W2)
    c_time = (time.time() - t0) / 5

    print(f"\nPerformance (5x5 grid, {nc2} colors, {n_trans} transversals):")
    print(f"  Python: {py_time*1000:.1f} ms")
    print(f"  C:      {c_time*1000:.1f} ms")
    print(f"  Speedup: {py_time/c_time:.1f}x")


if __name__ == '__main__':
    all_pass = True
    test_embedding_dims()
    all_pass &= test_precompute_embeddings()
    all_pass &= test_build_score_tables()
    all_pass &= test_hist_score_tables()
    if all_pass:
        test_performance()
    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
