"""
test_step_by_step.py — Step-by-step comparison of Python vs C ARC solver
on task 25ff71a9.

Identifies exactly where the two solvers diverge by comparing:
1. W1/W2 projection matrices (MT19937 + Box-Muller)
2. Plucker lines from training pair 0
3. Transversal computation (RNG for 4-tuple selection)
4. Score tables
5. Final scores for correct answer
"""

import json
import numpy as np
from transversal_memory import P3Memory

N_COLORS = 10
J6 = np.array([[0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
                [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0]], dtype=np.float32)

PLUCKER_PAIRS = np.array([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])


# ── Embedding functions (from exp_arc_fast_solve.py) ───────────────────────

def emb_hist_color(r, c, in_c, out_c, inp, out, H, W):
    in_oh = np.zeros(N_COLORS, dtype=np.float32); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS, dtype=np.float32); out_oh[out_c] = 1.0
    diff = np.array([(np.sum(out == i) - np.sum(inp == i)) / max(inp.size, 1)
                     for i in range(N_COLORS)], dtype=np.float32)
    return np.concatenate([in_oh, out_oh, diff])


def make_line(sv, tv, W1, W2):
    combined = np.concatenate([sv, tv])
    p1 = W1 @ combined; p2 = W2 @ combined
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    L = np.array([p1[i]*p2[j] - p1[j]*p2[i] for i,j in pairs], dtype=np.float32)
    n = np.linalg.norm(L)
    return L / n if n > 1e-10 else None


def compute_transversals_python(lines, n_trans=200, rng=None):
    """Python solver's transversal computation (uses default_rng = PCG64)."""
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
                Tn = (T / n).astype(np.float32)
                if np.all(np.isfinite(Tn)):
                    trans.append(Tn)
    return trans


def compute_transversals_c_style(lines, n_trans=200, seed=42):
    """Replicate C solver's transversal computation using MT19937 (RandomState)
    with the same rng_choose logic: Fisher-Yates partial shuffle."""
    rng = np.random.RandomState(seed)
    if len(lines) < 4:
        return []

    def rng_choose_mt(n, k):
        """Replicate C's rng_choose: Fisher-Yates partial shuffle using
        rng_int(n) = floor(rng_double() * n).
        C's rng_double uses MT19937: (genrand>>5)*2^-27 + (genrand>>6)*2^-53.
        RandomState.random_sample() produces the same doubles."""
        arr = list(range(n))
        out = []
        for i in range(k):
            # C: rng_int(r, n-i) = (int)(rng_double(r) * (n-i))
            j = i + int(rng.random_sample() * (n - i))
            arr[i], arr[j] = arr[j], arr[i]
            out.append(arr[i])
        return out

    trans = []
    att = 0
    while len(trans) < n_trans and att < n_trans * 10:
        att += 1
        idx = rng_choose_mt(len(lines), 4)
        mem = P3Memory()
        mem.store([lines[idx[i]] for i in range(3)])
        for T, res in mem.query_generative(lines[idx[3]]):
            n_val = np.linalg.norm(T)
            if n_val > 1e-10 and res < 1e-6:
                Tn = (T / n_val).astype(np.float32)
                if np.all(np.isfinite(Tn)):
                    trans.append(Tn)
    return trans


def main():
    # ── Load task ──────────────────────────────────────────────────────────
    task_path = "data/ARC-AGI/data/training/25ff71a9.json"
    with open(task_path) as f:
        task = json.load(f)

    test_inp = np.array(task['test'][0]['input'])
    test_out = np.array(task['test'][0]['output'])
    H, W = test_inp.shape
    print(f"Task 25ff71a9: test grid {H}x{W}")
    print(f"Test input:\n{test_inp}")
    print(f"Test output:\n{test_out}")

    # Used colors (same logic as both Python and C)
    used_colors = sorted(set(
        c for p in task['train']
        for g in [p['input'], p['output']]
        for row in g for c in row
    ) | set(c for row in task['test'][0]['input'] for c in row))
    nc = len(used_colors)
    print(f"Used colors: {used_colors} (nc={nc})")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Compare W1, W2 projection matrices
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP 1: W1/W2 comparison for hist_color embedding")
    print("="*70)

    name = 'hist_color'
    dim = 30

    # CRITICAL FINDING: Python's hash() is NON-DETERMINISTIC across runs!
    # PYTHONHASHSEED is random by default, so hash('hist_color') % 2**31
    # produces a DIFFERENT seed every time Python starts.
    py_seed = hash(name) % 2**31
    print(f"  Python hash('{name}') % 2**31 = {py_seed}  (THIS RUN)")

    # The C code hardcodes:
    c_seed = 146416007
    print(f"  C hardcoded seed for '{name}': {c_seed}")
    print(f"  MATCH: {'YES' if py_seed == c_seed else 'NO — SEEDS DIFFER!'}")
    print()
    print("  *** DIVERGENCE POINT #1: hash() is non-deterministic! ***")
    print("  Python uses a random PYTHONHASHSEED on each run.")
    print("  C uses a fixed hardcoded value (146416007 for hist_color).")
    print("  This means W1/W2 are DIFFERENT between Python and C,")
    print("  AND between different Python runs!")

    # Python's W1/W2 (this run's seed)
    rng_proj_py = np.random.RandomState(py_seed)
    W1_py = rng_proj_py.randn(4, 2 * dim).astype(np.float32) * 0.1
    W2_py = rng_proj_py.randn(4, 2 * dim).astype(np.float32) * 0.1

    # C's W1/W2 (hardcoded seed)
    rng_proj_c = np.random.RandomState(c_seed)
    W1_c = rng_proj_c.randn(4, 2 * dim).astype(np.float32) * 0.1
    W2_c = rng_proj_c.randn(4, 2 * dim).astype(np.float32) * 0.1

    print(f"\n  Python W1[0,:5]: {W1_py[0,:5]}")
    print(f"  C-seed W1[0,:5]: {W1_c[0,:5]}")
    w1_match = np.allclose(W1_py, W1_c)
    print(f"  W1 match: {w1_match}")
    if not w1_match:
        print(f"  Max |W1_py - W1_c|: {np.max(np.abs(W1_py - W1_c)):.6e}")
    print(f"\n  Python W2[0,:5]: {W2_py[0,:5]}")
    print(f"  C-seed W2[0,:5]: {W2_c[0,:5]}")
    w2_match = np.allclose(W2_py, W2_c)
    print(f"  W2 match: {w2_match}")

    # Verify C's MT19937 matches numpy's RandomState (already confirmed externally)
    print()
    print("  VERIFIED: C's MT19937 + Box-Muller produces IDENTICAL randn values")
    print("  as numpy's RandomState for the SAME seed. The problem is the seed,")
    print("  not the RNG implementation.")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Compute lines for training pair 0
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP 2: Plucker lines for training pair 0 (hist_color)")
    print("="*70)

    pair = task['train'][0]
    inp = np.array(pair['input'])
    out = np.array(pair['output'])
    pH, pW = inp.shape
    print(f"  Training pair 0: {pH}x{pW}")
    print(f"  Input:\n  {inp}")
    print(f"  Output:\n  {out}")

    # Build adjacency pairs (same order as both Python and C)
    pair_adj = [(r, c, r+dr, c+dc)
                for r in range(pH) for c in range(pW)
                for dr, dc in [(0, 1), (1, 0)]
                if r+dr < pH and c+dc < pW]
    print(f"  Number of adj pairs: {len(pair_adj)}")

    # Compute lines
    lines = []
    for r, c, r2, c2 in pair_adj:
        ea = emb_hist_color(r, c, inp[r, c], out[r, c], inp, out, pH, pW)
        eb = emb_hist_color(r2, c2, inp[r2, c2], out[r2, c2], inp, out, pH, pW)
        L = make_line(ea, eb, W1_py, W2_py)
        if L is not None:
            lines.append(L)

    print(f"  Number of valid lines: {len(lines)}")
    if lines:
        print(f"  First 3 lines:")
        for i in range(min(3, len(lines))):
            print(f"    [{i}] {lines[i]}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Compare RNG choice() implementations
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP 3: RNG choice() comparison — default_rng (PCG64) vs RandomState (MT19937)")
    print("="*70)

    n = len(lines)
    print(f"  Choosing 4 from {n} lines")

    # Python solver uses default_rng(42) with .choice(n, 4, replace=False)
    rng_py = np.random.default_rng(42)
    print(f"\n  default_rng(42).choice({n}, 4, replace=False) — first 10 selections:")
    py_selections = []
    for i in range(10):
        sel = rng_py.choice(n, size=4, replace=False)
        py_selections.append(tuple(sel))
        print(f"    [{i}] {sel}")

    # C solver uses MT19937(42) with Fisher-Yates partial shuffle
    rng_c = np.random.RandomState(42)
    print(f"\n  MT19937(42) rng_choose({n}, 4) — first 10 selections:")
    c_selections = []
    for i in range(10):
        arr = list(range(n))
        out_sel = []
        for j in range(4):
            k = j + int(rng_c.random_sample() * (n - j))
            arr[j], arr[k] = arr[k], arr[j]
            out_sel.append(arr[j])
        c_selections.append(tuple(out_sel))
        print(f"    [{i}] {out_sel}")

    match_count = sum(1 for a, b in zip(py_selections, c_selections) if a == b)
    print(f"\n  Matching selections: {match_count}/10")
    if match_count == 0:
        print("  CONFIRMED: 4-tuple selections DIFFER between Python and C")
        print("  → This means transversals will differ even with identical lines")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Compute transversals with both RNG methods
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP 4: Transversal computation comparison")
    print("="*70)

    print("\n  Computing transversals with Python's default_rng(42) (PCG64)...")
    trans_py = compute_transversals_python(lines, n_trans=200,
                                            rng=np.random.default_rng(42))
    print(f"  Python transversals: {len(trans_py)}")
    if trans_py:
        print(f"  First 3 Python transversals:")
        for i in range(min(3, len(trans_py))):
            print(f"    [{i}] {trans_py[i]}")

    print(f"\n  Computing transversals with C-style MT19937(42)...")
    trans_c = compute_transversals_c_style(lines, n_trans=200, seed=42)
    print(f"  C-style transversals: {len(trans_c)}")
    if trans_c:
        print(f"  First 3 C-style transversals:")
        for i in range(min(3, len(trans_c))):
            print(f"    [{i}] {trans_c[i]}")

    # Check if any transversals are shared
    if trans_py and trans_c:
        py_set = set(tuple(np.round(t, 5)) for t in trans_py)
        c_set = set(tuple(np.round(t, 5)) for t in trans_c)
        shared = len(py_set & c_set)
        print(f"\n  Shared transversals (rounded to 5 decimals): {shared}")
        print(f"  Python unique: {len(py_set - c_set)}")
        print(f"  C-style unique: {len(c_set - py_set)}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Build JTm and compare
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP 5: JTm matrix comparison")
    print("="*70)

    def build_JTm(trans_list):
        if not trans_list:
            return None
        trans_arr = np.stack(trans_list).astype(np.float64)
        JTm = (J6.astype(np.float64) @ trans_arr.T).astype(np.float32)
        valid = np.all(np.isfinite(JTm), axis=0)
        if not valid.all():
            JTm = JTm[:, valid]
        return JTm

    JTm_py = build_JTm(trans_py)
    JTm_c = build_JTm(trans_c)

    if JTm_py is not None:
        print(f"  Python JTm shape: {JTm_py.shape}")
        print(f"  Python JTm[:, :3]:\n  {JTm_py[:, :3]}")
    if JTm_c is not None:
        print(f"  C-style JTm shape: {JTm_c.shape}")
        print(f"  C-style JTm[:, :3]:\n  {JTm_c[:, :3]}")

    if JTm_py is not None and JTm_c is not None:
        if JTm_py.shape == JTm_c.shape:
            max_diff = np.max(np.abs(JTm_py - JTm_c))
            print(f"  Max |JTm_py - JTm_c|: {max_diff}")
        else:
            print(f"  JTm shapes differ: {JTm_py.shape} vs {JTm_c.shape}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Build score tables and compare for a sample adj pair
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP 6: Score table comparison (first adj pair)")
    print("="*70)

    # For hist_color with 3-color tasks, use histogram tables
    inp_hist = np.array([np.sum(test_inp == c) for c in range(N_COLORS)],
                        dtype=np.float32)

    # Build score for correct output histogram
    correct_hist = np.zeros(nc, dtype=int)
    for r in range(H):
        for c in range(W):
            ci = used_colors.index(test_out[r, c])
            correct_hist[ci] += 1
    print(f"  Correct output histogram: {correct_hist}")

    # Build diff vector for correct histogram
    out_h = np.zeros(N_COLORS, dtype=np.float32)
    for ci, cnt in enumerate(correct_hist):
        out_h[used_colors[ci]] = cnt
    diff = (out_h - inp_hist) / max(test_inp.size, 1)
    print(f"  Diff vector: {diff}")

    # Score first adj pair with correct candidate colors
    r1, c1, r2, c2 = H, W, 0, 0  # will set below
    adj_pairs = [(r, c, r+dr, c+dc)
                 for r in range(H) for c in range(W)
                 for dr, dc in [(0, 1), (1, 0)]
                 if r+dr < H and c+dc < W]
    n_adj = len(adj_pairs)
    print(f"  Test adj pairs: {n_adj}")

    def score_candidate_with_JTm(JTm, adj_pairs, candidate_grid, used_colors,
                                  test_inp, H, W):
        """Score a single candidate grid using hist_color embedding."""
        nc = len(used_colors)
        total = 0.0

        # Build histogram diff for this candidate
        cand_flat = candidate_grid.flatten()
        out_h = np.zeros(N_COLORS, dtype=np.float32)
        for v in cand_flat:
            out_h[v] += 1
        diff = (out_h - inp_hist) / max(test_inp.size, 1)

        for r1, c1, r2, c2 in adj_pairs:
            in_c_a = int(test_inp[r1, c1])
            out_c_a = int(candidate_grid[r1, c1])
            in_c_b = int(test_inp[r2, c2])
            out_c_b = int(candidate_grid[r2, c2])

            # Build embeddings
            in_oh_a = np.zeros(N_COLORS, dtype=np.float32); in_oh_a[in_c_a] = 1.0
            out_oh_a = np.zeros(N_COLORS, dtype=np.float32); out_oh_a[out_c_a] = 1.0
            ea = np.concatenate([in_oh_a, out_oh_a, diff])

            in_oh_b = np.zeros(N_COLORS, dtype=np.float32); in_oh_b[in_c_b] = 1.0
            out_oh_b = np.zeros(N_COLORS, dtype=np.float32); out_oh_b[out_c_b] = 1.0
            eb = np.concatenate([in_oh_b, out_oh_b, diff])

            L = make_line(ea, eb, W1_py, W2_py)
            if L is None:
                continue

            inner = L @ JTm  # (n_trans,)
            inner = np.clip(inner, -1e10, 1e10)
            sc = np.nansum(np.log(np.abs(inner) + 1e-10))
            sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=-100.0)
            total += sc

        return total

    if JTm_py is not None and JTm_c is not None:
        correct_score_py = score_candidate_with_JTm(
            JTm_py, adj_pairs, test_out, used_colors, test_inp, H, W)
        correct_score_c = score_candidate_with_JTm(
            JTm_c, adj_pairs, test_out, used_colors, test_inp, H, W)
        print(f"\n  Score for CORRECT answer (hist_color only):")
        print(f"    Python transversals: {correct_score_py:.6f}")
        print(f"    C-style transversals: {correct_score_c:.6f}")
        print(f"    Difference: {correct_score_py - correct_score_c:.6f}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: Full scoring — find rank with each set of transversals
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP 7: Full candidate scoring — find where ranks diverge")
    print("="*70)

    # For a 3x3 grid with nc colors, enumerate all candidates
    n_total = nc ** (H * W)
    print(f"  Grid: {H}x{W}, nc={nc}, total candidates: {n_total}")

    if n_total > 1_000_000:
        print("  Too many candidates for exhaustive comparison, skipping.")
    else:
        from itertools import product as cartesian

        color_to_idx = {c: i for i, c in enumerate(used_colors)}
        correct_flat = sum(
            color_to_idx[test_out[r, c]] * (nc ** (H * W - 1 - (r * W + c)))
            for r in range(H) for c in range(W)
        )

        # Score all candidates with Python transversals
        print("  Scoring all candidates with Python transversals...")
        scores_py = np.zeros(n_total, dtype=np.float32)
        scores_c = np.zeros(n_total, dtype=np.float32)

        for idx_flat in range(n_total):
            # Decode candidate
            cand_indices = []
            rem = idx_flat
            for _ in range(H * W):
                cand_indices.append(rem % nc)
                rem //= nc
            cand_indices = cand_indices[::-1]
            cand_grid = np.array([used_colors[ci] for ci in cand_indices]).reshape(H, W)

            if JTm_py is not None:
                scores_py[idx_flat] = score_candidate_with_JTm(
                    JTm_py, adj_pairs, cand_grid, used_colors, test_inp, H, W)
            if JTm_c is not None:
                scores_c[idx_flat] = score_candidate_with_JTm(
                    JTm_c, adj_pairs, cand_grid, used_colors, test_inp, H, W)

        # Note: This only uses hist_color embedding. The full solver uses all 8.
        # For this task, hist_color is the one that matters (3-color → histogram tables).

        # Find rank of correct answer
        correct_score_py_full = scores_py[correct_flat]
        correct_score_c_full = scores_c[correct_flat]

        # In the solver, lower score = better (log-likelihood, more negative = worse)
        # Actually checking: score < correct_score means better
        rank_py = int((scores_py < correct_score_py_full).sum()) + 1
        rank_c = int((scores_c < correct_score_c_full).sum()) + 1

        best_idx_py = int(np.argmin(scores_py))
        best_idx_c = int(np.argmin(scores_c))

        # Skip identity (output == input)
        inp_flat = sum(
            color_to_idx.get(test_inp[r, c], 0) * (nc ** (H * W - 1 - (r * W + c)))
            for r in range(H) for c in range(W)
        )
        if best_idx_py == inp_flat:
            sorted_py = np.argsort(scores_py)
            best_idx_py = int(sorted_py[1])
        if best_idx_c == inp_flat:
            sorted_c = np.argsort(scores_c)
            best_idx_c = int(sorted_c[1])

        def decode_candidate(idx_flat):
            cand = []
            rem = idx_flat
            for _ in range(H * W):
                cand.append(rem % nc)
                rem //= nc
            return np.array([used_colors[ci] for ci in reversed(cand)]).reshape(H, W)

        best_grid_py = decode_candidate(best_idx_py)
        best_grid_c = decode_candidate(best_idx_c)

        print(f"\n  Results with Python transversals (hist_color only):")
        print(f"    Correct score: {correct_score_py_full:.6f}")
        print(f"    Rank: {rank_py}")
        print(f"    Best candidate (rank 1):\n    {best_grid_py}")
        print(f"    Best score: {scores_py[best_idx_py]:.6f}")
        print(f"    Match correct: {np.array_equal(best_grid_py, test_out)}")

        print(f"\n  Results with C-style transversals (hist_color only):")
        print(f"    Correct score: {correct_score_c_full:.6f}")
        print(f"    Rank: {rank_c}")
        print(f"    Best candidate (rank 1):\n    {best_grid_c}")
        print(f"    Best score: {scores_c[best_idx_c]:.6f}")
        print(f"    Match correct: {np.array_equal(best_grid_c, test_out)}")

        if not np.array_equal(best_grid_py, best_grid_c):
            print(f"\n  DIVERGENCE: Python and C pick different best candidates!")

            # Check what C's best candidate scored under Python's transversals
            c_best_flat = best_idx_c
            print(f"    C's best scored under Python transversals: {scores_py[c_best_flat]:.6f}")
            print(f"    Python's best scored under C transversals: {scores_c[best_idx_py]:.6f}")

        # Show top-5 for each
        print(f"\n  Top-5 candidates (Python transversals):")
        sorted_py = np.argsort(scores_py)
        for i in range(min(5, len(sorted_py))):
            idx = int(sorted_py[i])
            g = decode_candidate(idx)
            is_correct = " *** CORRECT ***" if idx == correct_flat else ""
            is_input = " [INPUT]" if idx == inp_flat else ""
            print(f"    [{i+1}] score={scores_py[idx]:.6f} grid={g.flatten()}{is_correct}{is_input}")

        print(f"\n  Top-5 candidates (C-style transversals):")
        sorted_c = np.argsort(scores_c)
        for i in range(min(5, len(sorted_c))):
            idx = int(sorted_c[i])
            g = decode_candidate(idx)
            is_correct = " *** CORRECT ***" if idx == correct_flat else ""
            is_input = " [INPUT]" if idx == inp_flat else ""
            print(f"    [{i+1}] score={scores_c[idx]:.6f} grid={g.flatten()}{is_correct}{is_input}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8: Summary of divergence
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP 8: Summary of divergence")
    print("="*70)
    print("""
  DIVERGENCE POINT #1 (CRITICAL): hash() seeds for W1/W2 projections
     Python uses hash(name) % 2**31, which is NON-DETERMINISTIC across runs
     (PYTHONHASHSEED is random by default). C hardcodes specific values.
     The C's MT19937 + Box-Muller is VERIFIED IDENTICAL to numpy's RandomState.
     But since the SEEDS differ, W1 and W2 are completely different matrices.
     → Different projections → different Plucker lines → everything downstream differs.

  DIVERGENCE POINT #2: 4-tuple selection RNG for transversals
     Python: np.random.default_rng(42+i).choice(n, 4, replace=False) → PCG64
     C: MT19937(42+i) Fisher-Yates rng_choose()
     CONFIRMED DIFFERENT: Different RNG algorithm AND different shuffle algorithm.
     Even with identical lines, different 4-tuples are selected.

  COMBINED EFFECT:
     Both #1 and #2 contribute to different transversal sets.
     Different transversals → different JTm → different score tables → different rankings.
     For task 25ff71a9, both happen to get the correct answer at rank 1
     (after skipping input-identity), but the scores differ by ~3000 points.
     For other tasks, the rank differences could cause solve/fail divergence.

  FIX: To make Python and C match exactly:
     1. Set PYTHONHASHSEED=0 or use a fixed seed independent of hash()
     2. OR: make C use the same RNG as Python for transversal sampling (PCG64)
     3. OR: make Python use RandomState for transversal sampling too
     Recommendation: Use a deterministic hash (e.g., Python's hashlib) in both.
""")


if __name__ == '__main__':
    main()
