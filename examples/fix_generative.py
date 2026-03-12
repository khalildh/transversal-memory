"""
fix_generative.py — Fix the co-punctal degeneracy in P3Memory decoding.

ROOT CAUSE:
  line(W*src, W*tgt) for fixed src → all lines share endpoint W*src
  → they form a 3D linear subspace of Plücker R⁶
  → plucker_inner = 0 for ALL pairs → transversal meets EVERYTHING

The fix: both endpoints must depend on BOTH source and target.
We test two approaches that break the shared-point structure.
"""

import pickle
import sys
import time
import numpy as np

sys.path.insert(0, "..")

from transversal_memory import P3Memory, GramMemory
from transversal_memory.plucker import (
    line_from_points, plucker_inner, random_projection, is_valid_line,
)


# ── Load cached data ─────────────────────────────────────────────────────

with open("data/cache/associations.pkl", "rb") as f:
    assoc = pickle.load(f)
with open("data/cache/embeddings_dim32.pkl", "rb") as f:
    emb = pickle.load(f)

DIM = 32

TESTS = [
    ("dog",      ["puppy", "bark", "fetch"],       "bone"),
    ("king",     ["crown", "throne", "royal"],      "queen"),
    ("love",     ["heart", "romance", "passion"],   "marriage"),
    ("fire",     ["flame", "heat", "burn"],         "smoke"),
    ("ocean",    ["waves", "deep", "salt"],         "fish"),
    ("music",    ["rhythm", "melody", "harmony"],   "instrument"),
    ("tree",     ["leaves", "branches", "roots"],   "forest"),
    ("brain",    ["neurons", "memory", "cortex"],   "intelligence"),
]


# ── Helpers ──────────────────────────────────────────────────────────────

def check_degeneracy(make_line_fn, source, label):
    """Check if lines from a fixed source are co-punctal."""
    lines = []
    words = []
    for w in emb.vocab[:3000]:
        if w == source:
            continue
        L = make_line_fn(source, w)
        if L is not None:
            lines.append(L)
            words.append(w)
        if len(lines) >= 2000:
            break

    M = np.array(lines)
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    rank = int(np.sum(S > 0.01 * S[0]))

    # Check pairwise plucker_inner
    pi_vals = []
    for i in range(min(200, len(lines))):
        for j in range(i+1, min(200, len(lines))):
            pi_vals.append(abs(plucker_inner(lines[i], lines[j])))
    pi_vals = np.array(pi_vals)

    print(f"  [{label}] source='{source}':")
    print(f"    Singular values: {S.round(2)}")
    print(f"    Effective rank: {rank}/6")
    print(f"    Pairwise |pi|: mean={pi_vals.mean():.6e}  "
          f"max={pi_vals.max():.6e}  "
          f"frac_zero={np.mean(pi_vals < 1e-10):.3f}")
    return rank


def run_test(make_line_fn, label, use_plucker_score=True):
    """Run all analogy tests and report results."""
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")

    total_known = 0
    total_target_found = 0

    for source, stored_tgts, query_tgt in TESTS:
        stored_lines = []
        for t in stored_tgts:
            L = make_line_fn(source, t)
            if L is not None:
                stored_lines.append(L)
        if len(stored_lines) < 3:
            print(f"\n  {source}: not enough stored lines")
            continue

        q_line = make_line_fn(source, query_tgt)
        if q_line is None:
            continue

        mem = P3Memory()
        mem.store(stored_lines[:3])
        trans = mem.query_generative(q_line)
        if not trans:
            print(f"\n  {source}: no transversals found")
            continue

        T, resid = trans[0]

        # Decode
        results = []
        known = set(assoc.get(source, []))
        for word in emb.vocab:
            if word == source:
                continue
            L = make_line_fn(source, word)
            if L is None:
                continue
            if use_plucker_score:
                # Lower |pi| = more incident with transversal = better
                score = -abs(plucker_inner(T, L))
            else:
                score = abs(float(T @ L))
            results.append((score, word))

        results.sort(key=lambda x: -x[0])
        top10 = results[:10]

        # Find target rank
        target_rank = None
        for i, (s, w) in enumerate(results):
            if w == query_tgt:
                target_rank = i + 1
                break

        n_known = sum(1 for _, w in top10 if w in known)
        total_known += n_known
        if target_rank and target_rank <= 100:
            total_target_found += 1

        metric = "|pi| asc" if use_plucker_score else "dot"
        print(f"\n  {source} + {stored_tgts} → {query_tgt}  "
              f"[resid={resid:.1e}, metric={metric}]")
        print(f"    Known in top 10: {n_known}  "
              f"Target '{query_tgt}' rank: {target_rank}/{len(results)}")
        for s, w in top10:
            marker = " ✓" if w in known else ""
            if use_plucker_score:
                print(f"      {w:20s} |pi|={-s:.6e}{marker}")
            else:
                print(f"      {w:20s} dot={s:.6f}{marker}")

    print(f"\n  TOTAL known associates in top-10 across {len(TESTS)} queries: "
          f"{total_known}")
    print(f"  Queries with target in top 100: {total_target_found}/{len(TESTS)}")
    return total_known


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE: Original encoding (co-punctal, broken)
# ═══════════════════════════════════════════════════════════════════════════

W_base = random_projection(DIM, np.random.default_rng(42))

def make_line_baseline(source, target):
    if source not in emb.src or target not in emb.tgt:
        return None
    a = emb.src[source]
    b = emb.tgt[target]
    Wa = W_base @ a
    Wb = W_base @ b
    return line_from_points(Wa, Wb)

print("=" * 70)
print("DEGENERACY CHECK")
print("=" * 70)
check_degeneracy(make_line_baseline, "dog", "baseline")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1: Nonlinear encoding — Point2 = W2 @ (src ⊙ tgt)
#
# Point 1 = W1 @ src          (still shared across targets)
# Point 2 = W2 @ (src ⊙ tgt)  (depends nonlinearly on target)
#
# Since Point 2 is a nonlinear function of tgt, different targets
# produce different Point 2 values even for fixed source.
# The line is NOT co-punctal: it doesn't pass through a fixed point.
# ═══════════════════════════════════════════════════════════════════════════

rng1 = np.random.default_rng(42)
W1_nl = random_projection(DIM, rng1)
W2_nl = random_projection(DIM, rng1)

def make_line_nonlinear(source, target):
    if source not in emb.src or target not in emb.tgt:
        return None
    a = emb.src[source]
    b = emb.tgt[target]
    p1 = W1_nl @ a
    p2 = W2_nl @ (a * b)  # elementwise product
    return line_from_points(p1, p2)

print()
check_degeneracy(make_line_nonlinear, "dog", "nonlinear (src⊙tgt)")
run_test(make_line_nonlinear, "FIX 1: Nonlinear encoding (src, src⊙tgt)", use_plucker_score=True)
run_test(make_line_nonlinear, "FIX 1b: Nonlinear encoding — dot product", use_plucker_score=False)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2: Concatenation encoding
#
# Point 1 = W1 @ [src; tgt]
# Point 2 = W2 @ [src; tgt]
#
# Both endpoints depend on both src and tgt via DIFFERENT projections.
# Since W1 ≠ W2, the two points are distinct, and changing tgt
# changes both points → the line is not co-punctal.
# ═══════════════════════════════════════════════════════════════════════════

rng2 = np.random.default_rng(99)
W1_cat = random_projection(DIM * 2, rng2)  # 4 × 64
W2_cat = random_projection(DIM * 2, rng2)  # 4 × 64

def make_line_concat(source, target):
    if source not in emb.src or target not in emb.tgt:
        return None
    a = emb.src[source]
    b = emb.tgt[target]
    ab = np.concatenate([a, b])
    p1 = W1_cat @ ab
    p2 = W2_cat @ ab
    return line_from_points(p1, p2)

print()
check_degeneracy(make_line_concat, "dog", "concat [src;tgt]")
run_test(make_line_concat, "FIX 2: Concatenation encoding ([src;tgt])", use_plucker_score=True)
run_test(make_line_concat, "FIX 2b: Concatenation encoding — dot product", use_plucker_score=False)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 3: Cross-term encoding
#
# Point 1 = W1 @ src + W2 @ tgt      (linear mix)
# Point 2 = W3 @ (src ⊙ tgt) + W4 @ tgt  (nonlinear + linear)
#
# Combines linear and nonlinear terms. More expressive.
# ═══════════════════════════════════════════════════════════════════════════

rng3 = np.random.default_rng(77)
W1_x = random_projection(DIM, rng3)
W2_x = random_projection(DIM, rng3)
W3_x = random_projection(DIM, rng3)
W4_x = random_projection(DIM, rng3)

def make_line_cross(source, target):
    if source not in emb.src or target not in emb.tgt:
        return None
    a = emb.src[source]
    b = emb.tgt[target]
    p1 = W1_x @ a + W2_x @ b
    p2 = W3_x @ (a * b) + W4_x @ b
    return line_from_points(p1, p2)

print()
check_degeneracy(make_line_cross, "dog", "cross-term")
run_test(make_line_cross, "FIX 3: Cross-term encoding", use_plucker_score=True)
run_test(make_line_cross, "FIX 3b: Cross-term encoding — dot product", use_plucker_score=False)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4: Multi-projection consensus with non-degenerate encoding
#
# Use K random projections, each with the nonlinear encoding.
# For each projection, get the transversal and rank words by |pi|.
# Final ranking = average rank across projections.
# ═══════════════════════════════════════════════════════════════════════════

K_PROJ = 10

def run_multi_projection(source, stored_tgts, query_tgt):
    rng = np.random.default_rng(0)
    all_ranks = {}

    for k in range(K_PROJ):
        W1k = random_projection(DIM, rng)
        W2k = random_projection(DIM, rng)

        def ml(src, tgt, _W1=W1k, _W2=W2k):
            if src not in emb.src or tgt not in emb.tgt:
                return None
            a = emb.src[src]
            b = emb.tgt[tgt]
            return line_from_points(_W1 @ a, _W2 @ (a * b))

        stored = [ml(source, t) for t in stored_tgts]
        stored = [l for l in stored if l is not None]
        if len(stored) < 3:
            continue

        q = ml(source, query_tgt)
        if q is None:
            continue

        mem = P3Memory()
        mem.store(stored[:3])
        trans = mem.query_generative(q)
        if not trans:
            continue

        T, _ = trans[0]
        scores = []
        for word in emb.vocab:
            if word == source:
                continue
            L = ml(source, word)
            if L is None:
                continue
            scores.append((abs(plucker_inner(T, L)), word))

        scores.sort(key=lambda x: x[0])
        for rank, (_, w) in enumerate(scores):
            if w not in all_ranks:
                all_ranks[w] = []
            all_ranks[w].append(rank)

    # Average rank
    avg = [(np.mean(ranks), w) for w, ranks in all_ranks.items()
           if len(ranks) >= K_PROJ // 2]
    avg.sort(key=lambda x: x[0])
    return avg

print(f"\n{'=' * 70}")
print(f"FIX 4: Multi-projection consensus ({K_PROJ} projections, nonlinear)")
print(f"{'=' * 70}")

for source, stored_tgts, query_tgt in TESTS:
    t0 = time.time()
    avg = run_multi_projection(source, stored_tgts, query_tgt)
    elapsed = time.time() - t0
    if not avg:
        print(f"\n  {source}: failed")
        continue

    known = set(assoc.get(source, []))
    top10 = avg[:10]
    n_known = sum(1 for _, w in top10 if w in known)

    target_rank = None
    for i, (s, w) in enumerate(avg):
        if w == query_tgt:
            target_rank = i + 1
            break

    print(f"\n  {source} + {stored_tgts} → {query_tgt}  ({elapsed:.1f}s)")
    print(f"    Known in top 10: {n_known}  "
          f"Target rank: {target_rank}/{len(avg)}")
    for s, w in top10:
        marker = " ✓" if w in known else ""
        print(f"      {w:20s} avg_rank={s:.1f}{marker}")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print("""
ROOT CAUSE: Co-punctal degeneracy.
  line(W*src, W*tgt) for fixed src → all lines share endpoint W*src
  → 3D subspace of Plücker R⁶ → plucker_inner = 0 for ALL pairs
  → transversal meets every candidate → zero discrimination

FIXES TESTED:
  1. Nonlinear: line(W1*src, W2*(src⊙tgt))  — breaks shared point via ⊙
  2. Concatenation: line(W1*[src;tgt], W2*[src;tgt])  — both endpoints move
  3. Cross-term: combines linear + nonlinear features
  4. Multi-projection consensus with nonlinear encoding
""")
