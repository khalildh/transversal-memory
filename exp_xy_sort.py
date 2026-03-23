"""
exp_xy_sort.py — Can Plücker geometry help with X+Y sorting?

X+Y sorting: given X = {x₁,...,xₙ} and Y = {y₁,...,yₙ}, sort all n² sums
x_i + y_j. Best known is O(n² log n); whether O(n²) suffices is open.

Key idea: Plücker coordinates are bilinear (they encode products via a∧b).
To get sums, we lift via exponentiation: e^{x_i} · e^{y_j} = e^{x_i + y_j}.

This script explores three approaches:
  1. Direct embedding — encode (x_i, y_j) as Plücker lines, check if Gram
     energy ranking correlates with sum ordering
  2. Exponential lifting — use e^x coordinates so products encode sums
  3. Transversal partitioning — use transversals of 4 "pivot" lines as
     batch partition boundaries (like quicksort pivots)

The question: does the geometry provide any structural shortcut over
plain O(n² log n) comparison sort?
"""

import numpy as np
import time
from itertools import product as cartesian
from scipy.stats import spearmanr, kendalltau

from transversal_memory.plucker import (
    line_from_points, plucker_inner, hodge_dual, _J6,
    plucker_relation, is_valid_line,
)
from transversal_memory.solver import solve_p3
from transversal_memory.memory import GramMemory


# ── Helpers ──────────────────────────────────────────────────────────────────

def true_xy_order(X, Y):
    """Ground truth: all n² sums, sorted."""
    sums = [(x + y, i, j) for i, x in enumerate(X) for j, y in enumerate(Y)]
    sums.sort()
    return sums


def embed_product(x, y):
    """
    Embed (x, y) as a Plücker line via two points in P³.
    Points: a = (1, x, 0, 0), b = (0, 0, 1, y)
    This gives p = (0, 1, y, x, xy, 0) — encodes PRODUCT xy in p₁₃.
    """
    a = np.array([1.0, x, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0, y])
    return line_from_points(a, b)


def embed_exp_sum(x, y):
    """
    Embed (x, y) as a Plücker line using exponential lifting.
    Points: a = (1, e^x, 0, 0), b = (0, 0, 1, e^y)
    This gives p = (0, 1, e^y, e^x, e^{x+y}, 0) — p₁₃ encodes the SUM.
    """
    a = np.array([1.0, np.exp(x), 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0, np.exp(y)])
    return line_from_points(a, b)


def embed_additive(x, y):
    """
    Embed (x, y) with additive structure in the line direction.
    Points: a = (1, 0, x, y), b = (0, 1, x, y)
    The line direction is fixed but the "moment" encodes (x, y).
    """
    a = np.array([1.0, 0.0, x, y])
    b = np.array([0.0, 1.0, x, y])
    return line_from_points(a, b)


def embed_sum_axis(x, y):
    """
    Embed (x, y) so that a specific Plücker coordinate equals x+y.
    Points: a = (1, x+y, 0, 0), b = (0, 0, 1, 1)
    Gives p = (0, 1, 1, x+y, x+y, 0) — but all lines are co-planar.
    """
    s = x + y
    a = np.array([1.0, s, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0, 1.0])
    return line_from_points(a, b)


# ── Experiment 1: Gram energy ranking vs true sum order ──────────────────────

def exp1_gram_energy_ranking(n=10, seed=42):
    """
    Store all n² Plücker lines in a GramMemory, then rank by energy.
    Compare energy ranking to true sum ranking.
    """
    print(f"\n{'='*70}")
    print(f"Experiment 1: Gram energy ranking (n={n})")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    X = np.sort(rng.standard_normal(n))
    Y = np.sort(rng.standard_normal(n))

    true_order = true_xy_order(X, Y)
    true_sums = [s for s, i, j in true_order]
    true_pairs = [(i, j) for s, i, j in true_order]

    for name, embed_fn in [("product", embed_product),
                            ("exp_sum", embed_exp_sum),
                            ("additive", embed_additive),
                            ("sum_axis", embed_sum_axis)]:
        gram = GramMemory()
        lines = {}
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                p = embed_fn(x, y)
                gram.store_line(p)
                lines[(i, j)] = p

        # Check Plücker validity
        n_valid = sum(1 for p in lines.values() if is_valid_line(p, tol=1e-4))

        # Rank by energy
        scored = []
        for (i, j), p in lines.items():
            scored.append((gram.score(p), X[i] + Y[j], i, j))
        scored.sort(reverse=True)  # high energy first

        energy_sums = [s for _, s, _, _ in scored]
        true_sums_sorted = sorted([X[i] + Y[j] for i in range(n) for j in range(n)])

        rho, _ = spearmanr(
            [s for _, s, _, _ in scored],
            list(range(len(scored)))  # rank by energy
        )
        tau, _ = kendalltau(energy_sums, true_sums_sorted)

        print(f"\n  {name} embedding:")
        print(f"    Valid Plücker lines: {n_valid}/{n*n}")
        print(f"    Spearman ρ (energy rank vs sum rank): {rho:.4f}")
        print(f"    Kendall τ (energy order vs true order): {tau:.4f}")
        print(f"    Top-5 by energy: {[(round(s,3), i, j) for _, s, i, j in scored[:5]]}")
        print(f"    Top-5 by sum:    {[(round(s,3), i, j) for s, i, j in true_order[-5:][::-1]]}")


# ── Experiment 2: Incidence structure of X+Y lines ──────────────────────────

def exp2_incidence_structure(n=8, seed=42):
    """
    Study the incidence (meeting) pattern of X+Y Plücker lines.
    Two lines meet iff plucker_inner = 0. The pattern of near-incidences
    might reveal the sorted order.
    """
    print(f"\n{'='*70}")
    print(f"Experiment 2: Incidence structure (n={n})")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    X = np.sort(rng.standard_normal(n))
    Y = np.sort(rng.standard_normal(n))

    for name, embed_fn in [("product", embed_product),
                            ("exp_sum", embed_exp_sum)]:
        lines = []
        sums = []
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                lines.append(embed_fn(x, y))
                sums.append(x + y)

        N = len(lines)
        # Compute pairwise Plücker inner products
        PI = np.zeros((N, N))
        for a in range(N):
            for b in range(a+1, N):
                PI[a, b] = PI[b, a] = plucker_inner(lines[a], lines[b])

        # How does |plucker_inner(L_ij, L_kl)| relate to |sum_ij - sum_kl|?
        diffs = []
        inners = []
        for a in range(N):
            for b in range(a+1, N):
                diffs.append(abs(sums[a] - sums[b]))
                inners.append(abs(PI[a, b]))

        rho, _ = spearmanr(diffs, inners)
        print(f"\n  {name} embedding:")
        print(f"    Spearman ρ (|sum diff| vs |Plücker inner|): {rho:.4f}")
        print(f"    Mean |inner product|: {np.mean(inners):.6f}")
        n_meet = sum(1 for x in inners if x < 1e-6)
        print(f"    Pairs that meet (inner < 1e-6): {n_meet}/{len(inners)}")


# ── Experiment 3: Transversal partitioning ───────────────────────────────────

def exp3_transversal_partition(n=12, seed=42):
    """
    Use transversal computation as batch partitioning:
    1. Pick 4 "pivot" lines corresponding to known sums
    2. Find their 2 transversals
    3. Use plucker_inner(T, L) sign/magnitude to partition remaining lines

    If the transversal bisects the sum space, this is like quicksort
    with O(1) pivot computation that partitions n² elements.
    """
    print(f"\n{'='*70}")
    print(f"Experiment 3: Transversal partitioning (n={n})")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    X = np.sort(rng.standard_normal(n))
    Y = np.sort(rng.standard_normal(n))

    all_lines = []
    all_sums = []
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            all_lines.append(embed_exp_sum(x, y))
            all_sums.append(x + y)

    all_lines = np.array(all_lines)
    all_sums = np.array(all_sums)
    N = len(all_lines)

    # Pick 4 pivots: the lines at the 20th, 40th, 60th, 80th percentile sums
    sorted_idx = np.argsort(all_sums)
    pivot_positions = [int(N * p) for p in [0.2, 0.4, 0.6, 0.8]]
    pivot_idx = [sorted_idx[p] for p in pivot_positions]
    pivot_lines = [all_lines[i] for i in pivot_idx]
    pivot_sums = [all_sums[i] for i in pivot_idx]

    print(f"\n  Pivot sums: {[round(s, 3) for s in pivot_sums]}")

    # Build constraint matrix and find transversals
    A = np.stack([hodge_dual(p) for p in pivot_lines])  # (4, 6)
    _, S, Vt = np.linalg.svd(A, full_matrices=True)
    v1, v2 = Vt[-1], Vt[-2]

    transversals = solve_p3(v1, v2)
    print(f"  Found {len(transversals)} transversals")

    for t_idx, (T, resid) in enumerate(transversals):
        print(f"  Transversal {t_idx}: residual = {resid:.2e}")
        print(f"    Valid line: {is_valid_line(T, tol=1e-4)}")

        # Compute signed Plücker inner product of T with all lines
        # Use: plucker_inner(T, L) = T @ J6 @ L
        JT = T @ _J6  # (6,)
        signed_inner = all_lines @ JT  # (N,)

        # Does sign of inner product partition the sums?
        pos_mask = signed_inner > 0
        neg_mask = signed_inner <= 0

        if pos_mask.any() and neg_mask.any():
            pos_sums = all_sums[pos_mask]
            neg_sums = all_sums[neg_mask]
            print(f"    Partition: {pos_mask.sum()} positive, {neg_mask.sum()} negative")
            print(f"    Pos sums: mean={np.mean(pos_sums):.3f}, range=[{np.min(pos_sums):.3f}, {np.max(pos_sums):.3f}]")
            print(f"    Neg sums: mean={np.mean(neg_sums):.3f}, range=[{np.min(neg_sums):.3f}, {np.max(neg_sums):.3f}]")

            # Measure partition quality: what fraction of pos sums > all neg sums?
            median_sum = np.median(all_sums)
            pos_above = np.mean(pos_sums > median_sum)
            neg_below = np.mean(neg_sums <= median_sum)
            print(f"    Quality: {pos_above:.1%} of pos partition above median, "
                  f"{neg_below:.1%} of neg partition below median")

            # Spearman correlation between |inner product| and distance from median
            rho, _ = spearmanr(np.abs(signed_inner), np.abs(all_sums - median_sum))
            print(f"    Spearman ρ (|inner| vs |sum - median|): {rho:.4f}")
        else:
            print(f"    Degenerate partition: all on one side")


# ── Experiment 4: Eigenstructure encodes X and Y separately ──────────────────

def exp4_eigenstructure(n=15, seed=42):
    """
    The Gram matrix of all n² lines has eigenstructure.
    For the exp_sum embedding, the eigenvectors should reflect the
    factored structure X × Y. Check if the top eigenvectors separate
    the x-indices and y-indices.
    """
    print(f"\n{'='*70}")
    print(f"Experiment 4: Gram eigenstructure (n={n})")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    X = np.sort(rng.standard_normal(n))
    Y = np.sort(rng.standard_normal(n))

    gram = GramMemory()
    lines = []
    sums = []
    xi_list = []
    yj_list = []

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            p = embed_exp_sum(x, y)
            gram.store_line(p)
            lines.append(p)
            sums.append(x + y)
            xi_list.append(x)
            yj_list.append(y)

    lines = np.array(lines)
    sums = np.array(sums)
    xi_arr = np.array(xi_list)
    yj_arr = np.array(yj_list)

    # Eigenstructure
    evals = gram.eigenvalues()
    evecs = gram.principal_axes(k=6)  # (6, 6)

    print(f"\n  Eigenvalues: {[f'{v:.2f}' for v in evals]}")
    print(f"  Variance explained by top-1: {evals[0]/evals.sum():.1%}")
    print(f"  Variance explained by top-2: {evals[:2].sum()/evals.sum():.1%}")

    # Project all lines onto each eigenvector
    for k in range(min(3, len(evecs))):
        projections = lines @ evecs[k]  # (N,)
        rho_sum, _ = spearmanr(projections, sums)
        rho_x, _ = spearmanr(projections, xi_arr)
        rho_y, _ = spearmanr(projections, yj_arr)
        print(f"\n  Eigenvector {k} (λ={evals[k]:.2f}):")
        print(f"    Spearman ρ with x+y: {rho_sum:.4f}")
        print(f"    Spearman ρ with x:   {rho_x:.4f}")
        print(f"    Spearman ρ with y:   {rho_y:.4f}")


# ── Experiment 5: Computational comparison ───────────────────────────────────

def exp5_timing(max_n=50, seed=42):
    """
    Time the Plücker-based approach vs naive sort for various n.
    """
    print(f"\n{'='*70}")
    print(f"Experiment 5: Timing comparison")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)

    for n in [10, 20, 30, 40, 50]:
        X = rng.standard_normal(n)
        Y = rng.standard_normal(n)

        # Naive: compute all sums and sort
        t0 = time.perf_counter()
        sums = [X[i] + Y[j] for i in range(n) for j in range(n)]
        sums.sort()
        t_naive = time.perf_counter() - t0

        # Plücker: encode all lines, score via Gram, sort by score
        t0 = time.perf_counter()
        gram = GramMemory()
        lines = []
        for i in range(n):
            for j in range(n):
                p = embed_exp_sum(X[i], Y[j])
                gram.store_line(p)
                lines.append(p)
        scores = [gram.score(p) for p in lines]
        order = sorted(range(len(scores)), key=lambda k: -scores[k])
        t_plucker = time.perf_counter() - t0

        print(f"\n  n={n}: naive {t_naive*1000:.1f}ms, Plücker {t_plucker*1000:.1f}ms "
              f"(ratio: {t_plucker/t_naive:.1f}x)")


# ── Experiment 6: Can transversal geometry do better than O(n² log n)? ────────

def exp6_theoretical_analysis():
    """
    Theoretical analysis: what would a Plücker-based X+Y sort look like?
    """
    print(f"\n{'='*70}")
    print(f"Experiment 6: Theoretical analysis")
    print(f"{'='*70}")

    print("""
  X+Y Sorting via Plücker Geometry — Analysis

  SETUP:
    X = {x₁,...,xₙ}, Y = {y₁,...,yₙ} → n² sums to sort.
    Embed (xᵢ, yⱼ) as Plücker line pᵢⱼ in P³.

  APPROACH 1: Gram Energy Ranking
    Store all n² lines → 6×6 Gram M = Σ pᵢⱼ⊗pᵢⱼ.
    Score each line: E(pᵢⱼ) = pᵢⱼᵀ M pᵢⱼ.
    Sort by energy → approximate sum ordering.

    Complexity: O(n²) to build M + O(n²) to score all + O(n² log n) to sort scores.
    No improvement: we still need O(n² log n) for the final sort.

    But: if energy ~ sum, we get an approximate sort in O(n²) that could
    seed a nearly-sorted refinement pass (O(n² · inversions)).

  APPROACH 2: Transversal Partitioning
    Pick 4 lines at known quantiles → find 2 transversals (O(1)).
    Use plucker_inner(T, L) to partition all n² lines (O(n²)).
    Recurse on each partition.

    If each partition splits evenly: O(log n) levels × O(n²) per level = O(n² log n).
    No improvement over comparison sort — same recursion depth.

    But: what if we use 4 CAREFULLY CHOSEN lines from different (x,y) "quadrants"
    to get a 4-way partition? Then O(log₄ n) depth... but still O(n² log n).

  APPROACH 3: Sorted Matrix Structure
    The X+Y matrix is a Young tableau (rows and columns sorted).
    Each row of the matrix corresponds to lines through a fixed xᵢ;
    each column to lines through a fixed yⱼ.

    These form two "rulings" of a quadric surface in P³.
    The intersection pattern of the rulings encodes the sorted structure.

    Key insight: for the exp_sum embedding, the lines
      Lᵢ* = {pᵢⱼ : j=1..n}  (fixed xᵢ, varying yⱼ)
    form a 1-parameter family parameterised by e^{yⱼ}.
    Similarly Lⱼ* for fixed yⱼ.

    HOWEVER: extracting the sort order from the ruling structure still
    requires comparing O(n²) lines, giving O(n² log n).

  FUNDAMENTAL OBSTACLE:
    The Plücker inner product is a SINGLE scalar comparison between two lines.
    It does not provide "batch comparisons" — each evaluation is O(1) but you
    need Ω(n² log n) evaluations (information-theoretic lower bound for
    sorting n² items, unless you exploit the sum structure).

    The sum structure (Young tableau monotonicity) reduces the entropy
    from log₂(n²!) to about n² log n, but Plücker geometry doesn't
    seem to provide a way to exploit this below O(n² log n).

  WHAT PLÜCKER GEOMETRY DOES BUY:
    1. O(n²) APPROXIMATE sort via Gram energy (no log factor for the sketch)
    2. Compact O(36) summary of all n² pairs (the Gram matrix)
    3. O(1) "is this sum near the median?" queries via transversal incidence
    4. Natural factored structure: the two rulings correspond to X and Y
    5. Top-k and threshold queries without full sort
    """)


# ── Experiment 7: Batch eigenprojection sort ─────────────────────────────────

def exp7_batch_eigenprojection(n=20, seed=42):
    """
    Batch comparison via Gram eigenstructure:

    1. Encode all n² lines as (n², 6) matrix L — O(n²)
    2. Build Gram M = L^T L — O(n² × 36) = O(n²)
    3. Eigendecompose M — O(6³) = O(1)
    4. Batch project: L @ V — O(n² × 6) = O(n²)
       → Each line gets 6 coordinates in the eigenbasis
    5. Eigenvecs 1,2 separate x and y → recover (i,j) indices → radix structure
    6. Use recovered structure for a structured merge

    Key insight: the batch matmul L @ V simultaneously "compares" all n²
    lines against 6 directions. This is 6n² scalar operations that yield
    the full factored structure — equivalent to O(n² log n) pairwise
    comparisons in information content.
    """
    print(f"\n{'='*70}")
    print(f"Experiment 7: Batch eigenprojection sort (n={n})")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    X = np.sort(rng.standard_normal(n))
    Y = np.sort(rng.standard_normal(n))

    # Step 1: Encode all lines (batch)
    t0 = time.perf_counter()
    L = np.zeros((n * n, 6))
    true_sums = np.zeros(n * n)
    for i in range(n):
        for j in range(n):
            L[i * n + j] = embed_exp_sum(X[i], Y[j])
            true_sums[i * n + j] = X[i] + Y[j]
    t_encode = time.perf_counter() - t0

    # Step 2: Build Gram — one matmul
    t0 = time.perf_counter()
    M = L.T @ L  # (6, 6)
    t_gram = time.perf_counter() - t0

    # Step 3: Eigendecompose
    t0 = time.perf_counter()
    evals, evecs = np.linalg.eigh(M)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]  # (6, 6) columns are eigenvectors
    t_eig = time.perf_counter() - t0

    # Step 4: Batch project — one matmul
    t0 = time.perf_counter()
    P = L @ evecs  # (n², 6) — projections onto eigenbasis
    t_proj = time.perf_counter() - t0

    print(f"\n  Timing: encode {t_encode*1000:.1f}ms, gram {t_gram*1000:.3f}ms, "
          f"eig {t_eig*1000:.3f}ms, project {t_proj*1000:.3f}ms")

    # Step 5: Which eigenvector best separates x? y? x+y?
    xi_arr = np.repeat(X, n)
    yj_arr = np.tile(Y, n)

    print(f"\n  Eigenprojection correlations:")
    best_x_col, best_y_col, best_sum_col = 0, 0, 0
    best_x_rho, best_y_rho, best_sum_rho = 0, 0, 0
    for k in range(6):
        rho_x, _ = spearmanr(P[:, k], xi_arr)
        rho_y, _ = spearmanr(P[:, k], yj_arr)
        rho_s, _ = spearmanr(P[:, k], true_sums)
        print(f"    EV{k} (λ={evals[k]:.2f}): ρ_x={rho_x:.4f}, ρ_y={rho_y:.4f}, ρ_{'{x+y}'}={rho_s:.4f}")
        if abs(rho_x) > abs(best_x_rho):
            best_x_col, best_x_rho = k, rho_x
        if abs(rho_y) > abs(best_y_rho):
            best_y_col, best_y_rho = k, rho_y
        if abs(rho_s) > abs(best_sum_rho):
            best_sum_col, best_sum_rho = k, rho_s

    # Step 6: Recover factored structure via the two best eigenvectors
    proj_x = P[:, best_x_col] * np.sign(best_x_rho)  # flip if anti-correlated
    proj_y = P[:, best_y_col] * np.sign(best_y_rho)

    # Cluster into n groups by x-projection (bucket sort)
    t0 = time.perf_counter()
    x_order = np.argsort(proj_x)
    # Each consecutive block of n elements should correspond to one x_i value
    recovered_x_ranks = np.zeros(n * n, dtype=int)
    for rank, idx_val in enumerate(x_order):
        recovered_x_ranks[idx_val] = rank // n

    y_order = np.argsort(proj_y)
    recovered_y_ranks = np.zeros(n * n, dtype=int)
    for rank, idx_val in enumerate(y_order):
        recovered_y_ranks[idx_val] = rank // n

    # Check recovery accuracy
    true_x_ranks = np.repeat(np.arange(n), n)
    true_y_ranks = np.tile(np.arange(n), n)
    x_acc = np.mean(recovered_x_ranks == true_x_ranks)
    y_acc = np.mean(recovered_y_ranks == true_y_ranks)
    t_recover = time.perf_counter() - t0

    print(f"\n  Factor recovery (batch argsort):")
    print(f"    X-index accuracy: {x_acc:.1%}")
    print(f"    Y-index accuracy: {y_acc:.1%}")
    print(f"    Recovery time: {t_recover*1000:.3f}ms")

    # Step 7: Structured merge using recovered factors
    # Once we know (i, j) for each element, we have n sorted rows.
    # Merge via a priority queue (heap).
    import heapq
    t0 = time.perf_counter()

    # Build n rows, each sorted by y-projection within the x-bucket
    rows = [[] for _ in range(n)]
    for flat_idx in range(n * n):
        xi = recovered_x_ranks[flat_idx]
        rows[xi].append((proj_y[flat_idx], flat_idx))
    for row in rows:
        row.sort()  # sort each row by y-projection

    # Merge n sorted rows
    heap = []
    for i in range(n):
        if rows[i]:
            val, flat_idx = rows[i][0]
            heapq.heappush(heap, (val, i, 0, flat_idx))

    merged_order = []
    while heap:
        val, row_i, col_j, flat_idx = heapq.heappop(heap)
        merged_order.append(flat_idx)
        if col_j + 1 < len(rows[row_i]):
            next_val, next_flat = rows[row_i][col_j + 1]
            heapq.heappush(heap, (next_val, row_i, col_j + 1, next_flat))

    t_merge = time.perf_counter() - t0

    # Evaluate: how close is the merged order to the true sort?
    merged_sums = true_sums[merged_order]
    true_sorted = np.sort(true_sums)
    tau, _ = kendalltau(merged_sums, true_sorted)
    n_inversions = sum(1 for a in range(len(merged_sums))
                       for b in range(a+1, min(a+20, len(merged_sums)))
                       if merged_sums[a] > merged_sums[b])

    print(f"\n  Structured merge result:")
    print(f"    Kendall τ with true order: {tau:.4f}")
    print(f"    Local inversions (window=20): {n_inversions}")
    print(f"    Merge time: {t_merge*1000:.3f}ms")

    # Compare total time
    t_total_plucker = t_encode + t_gram + t_eig + t_proj + t_recover + t_merge
    t0 = time.perf_counter()
    naive_sorted = sorted(range(n*n), key=lambda k: true_sums[k])
    t_naive = time.perf_counter() - t0
    print(f"\n  Total: Plücker batch {t_total_plucker*1000:.1f}ms vs naive sort {t_naive*1000:.1f}ms")


# ── Experiment 8: Weighted eigenprojection for direct sum approximation ──────

def exp8_weighted_eigenprojection(n=20, seed=42):
    """
    Instead of recovering (i,j) factors, find a LINEAR COMBINATION of
    eigenprojections that directly approximates x+y.

    Key idea: if we can find weights w such that L @ V @ w ≈ true_sums,
    then sorting by this score gives the answer. The weights w can be
    found by least-squares regression on a SMALL sample, then applied
    to ALL n² elements.

    This is a "batch comparison" because L @ (V @ w) is a single matmul
    that simultaneously scores all n² elements.
    """
    print(f"\n{'='*70}")
    print(f"Experiment 8: Weighted eigenprojection (n={n})")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    X = np.sort(rng.standard_normal(n))
    Y = np.sort(rng.standard_normal(n))

    # Encode
    L = np.zeros((n * n, 6))
    true_sums = np.zeros(n * n)
    for i in range(n):
        for j in range(n):
            L[i * n + j] = embed_exp_sum(X[i], Y[j])
            true_sums[i * n + j] = X[i] + Y[j]

    # Gram + eigen
    M = L.T @ L
    evals, evecs = np.linalg.eigh(M)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    # Batch project
    P = L @ evecs  # (n², 6)

    # Learn weights from a SMALL sample (O(sqrt(n²)) = O(n) samples)
    n_sample = min(2 * n, n * n)
    sample_idx = rng.choice(n * n, n_sample, replace=False)
    P_sample = P[sample_idx]
    s_sample = true_sums[sample_idx]

    # Least squares: find w such that P_sample @ w ≈ s_sample
    w, residuals, rank, sv = np.linalg.lstsq(P_sample, s_sample, rcond=None)

    # Apply to ALL elements (the batch comparison)
    t0 = time.perf_counter()
    approx_sums = P @ w  # (n²,) — one matmul scores everything
    t_batch = time.perf_counter() - t0

    # Sort by approximate sums
    approx_order = np.argsort(approx_sums)
    true_order = np.argsort(true_sums)

    rho, _ = spearmanr(approx_sums, true_sums)
    tau, _ = kendalltau(approx_order, true_order)

    # Count exact-position matches
    exact_matches = np.sum(approx_order == true_order)

    print(f"\n  Learned weights w: {[f'{wi:.3f}' for wi in w]}")
    print(f"  Trained on {n_sample}/{n*n} samples ({n_sample/(n*n):.0%})")
    print(f"  Spearman ρ (approx vs true sums): {rho:.6f}")
    print(f"  Kendall τ (approx order vs true order): {tau:.6f}")
    print(f"  Exact position matches: {exact_matches}/{n*n} ({exact_matches/(n*n):.1%})")
    print(f"  Batch scoring time: {t_batch*1000:.4f}ms")

    # How many elements are displaced by at most k positions?
    for k in [0, 1, 3, 5, 10]:
        displaced = np.sum(np.abs(np.argsort(approx_order).astype(int)
                                  - np.argsort(true_order).astype(int)) <= k)
        print(f"  Within ±{k} of true position: {displaced}/{n*n} ({displaced/(n*n):.1%})")


# ── Experiment 9: Multi-transversal batch partition ──────────────────────────

def exp9_multi_transversal_batch(n=20, seed=42):
    """
    Use MULTIPLE sets of 4 pivot lines → multiple transversal pairs.
    Each transversal gives a signed partition of all n² lines via a
    single matmul (batch comparison).

    With k transversals, we get k sign bits per element — a k-bit
    "geometric hash" computed in O(k × n²) time.
    Elements with the same hash are in the same partition cell.

    If k = O(log n), we get O(n) partition cells, each of expected
    size O(n). Sorting within cells: O(n log n) each, total O(n² log n).

    But: if the geometric hash correlates with sum order, cells are
    nearly sorted → fewer inversions → faster refinement.
    """
    print(f"\n{'='*70}")
    print(f"Experiment 9: Multi-transversal batch partition (n={n})")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    X = np.sort(rng.standard_normal(n))
    Y = np.sort(rng.standard_normal(n))

    # Encode all lines
    L = np.zeros((n * n, 6))
    true_sums = np.zeros(n * n)
    for i in range(n):
        for j in range(n):
            L[i * n + j] = embed_exp_sum(X[i], Y[j])
            true_sums[i * n + j] = X[i] + Y[j]
    N = n * n

    # Generate multiple transversal "hyperplanes"
    n_transversals = int(np.ceil(2 * np.log2(n)))
    print(f"\n  Using {n_transversals} transversal directions (≈ 2·log₂(n))")

    # Strategy: pick pivots at various quantile combinations
    sorted_idx = np.argsort(true_sums)
    transversal_dirs = []  # each is a (6,) vector for batch comparison

    for t in range(n_transversals):
        # Pick 4 pivots at random quantiles
        quantiles = rng.uniform(0.1, 0.9, 4)
        pivot_pos = [int(N * q) for q in sorted(quantiles)]
        pivot_idx = [sorted_idx[p] for p in pivot_pos]
        pivot_lines = L[pivot_idx]

        # Find transversals
        A = np.stack([hodge_dual(p) for p in pivot_lines])
        _, S, Vt = np.linalg.svd(A, full_matrices=True)
        v1, v2 = Vt[-1], Vt[-2]
        sols = solve_p3(v1, v2)

        for T, resid in sols:
            if is_valid_line(T, tol=1e-2):
                # The "comparison direction" is J6 @ T
                transversal_dirs.append(T @ _J6)

    n_dirs = len(transversal_dirs)
    print(f"  Valid transversal directions found: {n_dirs}")

    if n_dirs == 0:
        print("  No valid transversals found — skipping")
        return

    # Batch comparison: compute all signs simultaneously
    T_mat = np.array(transversal_dirs)  # (n_dirs, 6)
    t0 = time.perf_counter()
    signs = (L @ T_mat.T > 0).astype(int)  # (N, n_dirs) — sign bits
    t_batch = time.perf_counter() - t0

    # Convert sign vectors to hash codes
    hash_codes = np.zeros(N, dtype=int)
    for k in range(n_dirs):
        hash_codes += signs[:, k] << k

    n_cells = len(set(hash_codes))
    print(f"  Partition cells: {n_cells} (max possible: {2**n_dirs})")
    print(f"  Batch comparison time: {t_batch*1000:.4f}ms")

    # Analyze cells: are elements within each cell close in sum value?
    from collections import Counter
    cell_counts = Counter(hash_codes)
    cell_ranges = {}
    for code in cell_counts:
        mask = hash_codes == code
        cell_sums = true_sums[mask]
        cell_ranges[code] = (np.min(cell_sums), np.max(cell_sums),
                             np.max(cell_sums) - np.min(cell_sums))

    ranges = [r[2] for r in cell_ranges.values()]
    total_range = np.max(true_sums) - np.min(true_sums)

    print(f"  Cell size: mean={np.mean(list(cell_counts.values())):.1f}, "
          f"max={max(cell_counts.values())}")
    print(f"  Cell sum range: mean={np.mean(ranges):.3f}, "
          f"max={np.max(ranges):.3f} (total range: {total_range:.3f})")
    print(f"  Mean cell range / total range: {np.mean(ranges)/total_range:.1%}")

    # Sort within cells, concatenate → approximate sort
    t0 = time.perf_counter()
    # Sort cells by their minimum sum, then sort within each cell
    cell_order = sorted(cell_ranges.keys(),
                        key=lambda c: cell_ranges[c][0])
    final_order = []
    for code in cell_order:
        mask = hash_codes == code
        cell_indices = np.where(mask)[0]
        cell_sums_local = true_sums[cell_indices]
        sorted_within = cell_indices[np.argsort(cell_sums_local)]
        final_order.extend(sorted_within)
    final_order = np.array(final_order)
    t_sort = time.perf_counter() - t0

    # Evaluate
    final_sums = true_sums[final_order]
    tau, _ = kendalltau(final_sums, np.sort(true_sums))
    rho, _ = spearmanr(final_sums, np.sort(true_sums))
    print(f"\n  Result after cell-sort:")
    print(f"    Kendall τ: {tau:.6f}")
    print(f"    Spearman ρ: {rho:.6f}")
    print(f"    Sort time: {t_sort*1000:.3f}ms")


# ── Experiment 10: The full pipeline — eigenfactored batch sort ──────────────

def exp10_full_pipeline(seed=42):
    """
    The complete batch-comparison X+Y sort using Plücker eigenstructure:

    1. Embed all n² pairs as lines: L ∈ R^{n²×6}           — O(n²)
    2. Gram matrix: M = L^T L ∈ R^{6×6}                     — O(n²)
    3. Eigendecompose M → V ∈ R^{6×6}                        — O(1)
    4. Batch project: P = L @ V ∈ R^{n²×6}                  — O(n²)
    5. Identify x-factor & y-factor eigenvectors              — O(n) probe
    6. Bucket by x-factor projection (radix digit 1)          — O(n²)
    7. Within each bucket, sort by y-factor (radix digit 2)   — O(n² total)
    8. Merge n sorted rows via heap                           — O(n² log n)

    Steps 1-7 are all O(n²). Step 8 is the unavoidable O(n² log n).
    The geometry recovers the Young tableau structure in O(n²) batch ops,
    then we merge the tableau rows.

    Question: can we also do step 8 in a batch-geometric way?
    Idea: use a learned sum-direction w to approximately sort,
    then fix inversions. The number of inversions bounds the refinement cost.
    """
    print(f"\n{'='*70}")
    print(f"Experiment 10: Full eigenfactored batch sort pipeline")
    print(f"{'='*70}")

    for n in [10, 20, 30, 50, 75, 100]:
        rng = np.random.default_rng(seed)
        X = np.sort(rng.standard_normal(n))
        Y = np.sort(rng.standard_normal(n))
        N = n * n

        # ── Naive baseline ──
        t0 = time.perf_counter()
        naive_sums = np.array([X[i] + Y[j] for i in range(n) for j in range(n)])
        naive_order = np.argsort(naive_sums)
        t_naive = time.perf_counter() - t0

        # ── Plücker batch pipeline ──
        t_start = time.perf_counter()

        # Step 1: Encode (vectorised)
        # Pre-compute exponentials
        ex = np.exp(X)  # (n,)
        ey = np.exp(Y)  # (n,)
        # Build unnormalised Plücker vectors directly
        # p = (0, 1, e^y, e^x, e^{x+y}, 0) for each (i,j)
        L = np.zeros((N, 6))
        L[:, 1] = 1.0
        L[:, 2] = np.tile(ey, n)         # e^{y_j} repeated n times
        L[:, 3] = np.repeat(ex, n)       # e^{x_i} each n times
        L[:, 4] = np.repeat(ex, n) * np.tile(ey, n)  # e^{x_i + y_j}
        # Normalise
        norms = np.linalg.norm(L, axis=1, keepdims=True)
        L /= norms

        # Steps 2-3: Gram + eigen
        M = L.T @ L
        evals, evecs = np.linalg.eigh(M)
        idx_sort = np.argsort(evals)[::-1]
        evals = evals[idx_sort]
        evecs = evecs[:, idx_sort]

        # Step 4: Batch project
        P = L @ evecs  # (N, 6)

        # Step 5: Identify factor eigenvectors
        # Use a small probe: first n elements have x=X[0], varying y
        # and elements [0, n, 2n, ...] have y=Y[0], varying x
        probe_x = np.arange(0, N, n)[:n]  # elements with j=0 (varying i)
        probe_y = np.arange(n)             # elements with i=0 (varying j)

        x_col, y_col = 0, 1
        best_x_var, best_y_var = 0, 0
        for k in range(min(4, 6)):
            x_var = np.var(P[probe_x, k])
            y_var = np.var(P[probe_y, k])
            # The x-factor eigenvector has high variance across varying-x probe
            # and low variance across varying-y probe (and vice versa)
            x_score = x_var / (y_var + 1e-20)
            y_score = y_var / (x_var + 1e-20)
            if x_score > best_x_var:
                best_x_var = x_score
                x_col = k
            if y_score > best_y_var:
                best_y_var = y_score
                y_col = k

        # Step 6: Bucket by x-factor (argsort → groups of n)
        x_proj = P[:, x_col]
        x_order = np.argsort(x_proj)
        # Assign bucket: element at position k in x_order gets bucket k // n
        x_bucket = np.empty(N, dtype=int)
        x_bucket[x_order] = np.arange(N) // n

        # Step 7: Within each bucket, sort by y-factor projection
        y_proj = P[:, y_col]
        rows = [[] for _ in range(n)]
        for flat_idx in range(N):
            rows[x_bucket[flat_idx]].append((y_proj[flat_idx], flat_idx))
        for row in rows:
            row.sort()

        # Step 8: Merge n sorted rows via heap
        import heapq
        heap = []
        for i in range(n):
            if rows[i]:
                val, flat_idx = rows[i][0]
                heapq.heappush(heap, (val, i, 0, flat_idx))

        plucker_order = np.empty(N, dtype=int)
        out_pos = 0
        while heap:
            val, row_i, col_j, flat_idx = heapq.heappop(heap)
            plucker_order[out_pos] = flat_idx
            out_pos += 1
            if col_j + 1 < len(rows[row_i]):
                nv, nf = rows[row_i][col_j + 1]
                heapq.heappush(heap, (nv, row_i, col_j + 1, nf))

        t_plucker = time.perf_counter() - t_start

        # ── Also try: direct sort by learned sum-direction ──
        t0 = time.perf_counter()
        # The sum e^{x+y} is in coordinate 4 (before normalisation).
        # After normalisation, we can recover it: L[:, 4] * norms[:, 0]
        # Or: learn w from the full projection
        # Use top-3 eigenvectors, fit w from O(n) samples
        n_sample = 2 * n
        sample_idx = rng.choice(N, n_sample, replace=False)
        true_sums_all = np.repeat(X, n) + np.tile(Y, n)
        w, _, _, _ = np.linalg.lstsq(P[sample_idx, :3],
                                       true_sums_all[sample_idx], rcond=None)
        approx_sums = P[:, :3] @ w
        direct_order = np.argsort(approx_sums)
        t_direct = time.perf_counter() - t0

        # ── Evaluate ──
        plucker_sums = true_sums_all[plucker_order]
        direct_sums = true_sums_all[direct_order]
        true_sorted = np.sort(true_sums_all)

        tau_heap, _ = kendalltau(plucker_sums, true_sorted)
        tau_direct, _ = kendalltau(direct_sums, true_sorted)

        # Count inversions in a window
        def local_inversions(arr, window=10):
            count = 0
            for i in range(len(arr)):
                for j in range(i+1, min(i+window, len(arr))):
                    if arr[i] > arr[j]:
                        count += 1
            return count

        inv_heap = local_inversions(plucker_sums, window=5)
        inv_direct = local_inversions(direct_sums, window=5)

        print(f"\n  n={n} (N={N}):")
        print(f"    Naive sort:     {t_naive*1000:.2f}ms")
        print(f"    Plücker heap:   {t_plucker*1000:.2f}ms  τ={tau_heap:.4f}  local_inv={inv_heap}")
        print(f"    Plücker direct: {t_direct*1000:.2f}ms  τ={tau_direct:.4f}  local_inv={inv_direct}")
        print(f"    Eigen x-col={x_col}, y-col={y_col}, "
              f"explained={evals[:2].sum()/evals.sum():.1%}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    exp1_gram_energy_ranking(n=10)
    exp2_incidence_structure(n=8)
    exp3_transversal_partition(n=12)
    exp4_eigenstructure(n=15)
    exp5_timing(max_n=50)
    exp6_theoretical_analysis()
    exp7_batch_eigenprojection(n=20)
    exp8_weighted_eigenprojection(n=20)
    exp9_multi_transversal_batch(n=20)
    exp10_full_pipeline()
