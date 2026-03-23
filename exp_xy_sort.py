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


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    exp1_gram_energy_ranking(n=10)
    exp2_incidence_structure(n=8)
    exp3_transversal_partition(n=12)
    exp4_eigenstructure(n=15)
    exp5_timing(max_n=50)
    exp6_theoretical_analysis()
