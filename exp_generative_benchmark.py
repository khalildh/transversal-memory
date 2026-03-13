"""
exp_generative_benchmark.py — Generative associate expansion benchmark
=======================================================================

Task: Given a source word + K seed associates, generate/rank N words.
      Measure how many are actual (held-out) associates.

Key question: With very few seeds (K=3-5), can geometry outperform
embeddings? At K=3, covariance estimation breaks down (need 33+ samples
for 32D Mahalanobis), so geometry in 6D Plücker space might have an edge.

Methods:
  1. Cosine to source (trivial baseline — ignores seeds entirely)
  2. Cosine to centroid of seeds
  3. Max cosine to any seed
  4. Centroid + source blend
  5. Multi-seed Gram^0.05 (geometry)
  6. Autoregressive Gram expansion (generative)
  7. Mahalanobis (when enough seeds)

Evaluation: same 200 test words as evaluate.py, varying K from 3 to 20.
"""

import pickle
import time
import numpy as np
from pathlib import Path

from transversal_memory.plucker import (
    random_projection_dual,
    batch_encode_lines_dual,
    project_to_line_dual,
)


# ── Config ──────────────────────────────────────────────────────────────

N_SEEDS_GEO = 50
SEED_SPACING = 10
GRAM_POWER = 0.05
DIM = 32
DATA_DIR = Path("data/cache")

# Test sizes (number of seed associates)
K_VALUES = [3, 5, 7, 10, 15, 20]


# ── Data ────────────────────────────────────────────────────────────────

with open(DATA_DIR / "associations.pkl", "rb") as f:
    assoc = pickle.load(f)
with open(DATA_DIR / "embeddings_dim32.pkl", "rb") as f:
    emb = pickle.load(f)


# ── Geometry infrastructure ─────────────────────────────────────────────

GEO_SEED_LIST = [i * SEED_SPACING for i in range(N_SEEDS_GEO)]
PROJECTIONS = []
for seed in GEO_SEED_LIST:
    rng = np.random.default_rng(seed)
    W1, W2 = random_projection_dual(DIM, rng)
    PROJECTIONS.append((W1, W2))


def _power_gram(gram, power):
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


def build_gram_ensemble(source, seeds):
    """Build multi-seed Gram^0.05 from source + seed associates."""
    src_vec = emb.src[source]
    per_seed = []
    for W1, W2 in PROJECTIONS:
        lines = []
        for t in seeds:
            if t in emb.tgt:
                L = project_to_line_dual(src_vec, emb.tgt[t], W1, W2)
                if np.linalg.norm(L) > 1e-12:
                    lines.append(L)
        if len(lines) < 2:
            continue
        arr = np.stack(lines)
        gram = _power_gram(arr.T @ arr, GRAM_POWER)
        per_seed.append({"W1": W1, "W2": W2, "gram": gram})
    return per_seed, src_vec


def score_gram_ensemble(src_vec, per_seed, tgt_mat):
    """Score candidate matrix with multi-seed Gram^0.05 ensemble."""
    N = tgt_mat.shape[0]
    acc = np.zeros(N)
    for sd in per_seed:
        lines = batch_encode_lines_dual(src_vec, tgt_mat, sd["W1"], sd["W2"])
        acc += np.sum((lines @ sd["gram"]) * lines, axis=1)
    return acc / len(per_seed) if per_seed else acc


# ── Methods ─────────────────────────────────────────────────────────────

def method_cosine_source(source, seeds, exclude, tgt_mat, all_words):
    """Baseline: cosine to source embedding (ignores seeds)."""
    src_vec = emb.src[source]
    scores = tgt_mat @ src_vec
    return scores


def method_cosine_centroid(source, seeds, exclude, tgt_mat, all_words):
    """Cosine to centroid of seed embeddings."""
    vecs = [emb.tgt[s] for s in seeds if s in emb.tgt]
    if not vecs:
        return np.zeros(tgt_mat.shape[0])
    centroid = np.mean(vecs, axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    return tgt_mat @ centroid


def method_max_cosine(source, seeds, exclude, tgt_mat, all_words):
    """Max cosine similarity to any seed associate."""
    vecs = [emb.tgt[s] for s in seeds if s in emb.tgt]
    if not vecs:
        return np.zeros(tgt_mat.shape[0])
    seed_mat = np.stack(vecs)
    sim = tgt_mat @ seed_mat.T
    return sim.max(axis=1)


def method_source_centroid_blend(source, seeds, exclude, tgt_mat, all_words):
    """Blend of cosine to source + cosine to centroid (0.5/0.5)."""
    s1 = method_cosine_source(source, seeds, exclude, tgt_mat, all_words)
    s2 = method_cosine_centroid(source, seeds, exclude, tgt_mat, all_words)
    # Rank-based blend (normalize to [0,1])
    r1 = np.argsort(np.argsort(-s1)).astype(float) / len(s1)
    r2 = np.argsort(np.argsort(-s2)).astype(float) / len(s2)
    return -(r1 + r2)  # negate because lower rank = better


def method_gram_ensemble(source, seeds, exclude, tgt_mat, all_words):
    """Multi-seed Gram^0.05 geometric scoring."""
    per_seed, src_vec = build_gram_ensemble(source, seeds)
    if not per_seed:
        return np.zeros(tgt_mat.shape[0])
    return score_gram_ensemble(src_vec, per_seed, tgt_mat)


def method_mahalanobis(source, seeds, exclude, tgt_mat, all_words):
    """Mahalanobis distance to centroid (only works with enough seeds)."""
    vecs = [emb.tgt[s] for s in seeds if s in emb.tgt]
    if len(vecs) < DIM + 1:
        # Fall back to cosine centroid when too few samples
        centroid = np.mean(vecs, axis=0) if vecs else np.zeros(DIM)
        centroid /= np.linalg.norm(centroid) + 1e-12
        return tgt_mat @ centroid

    mat = np.stack(vecs)
    centroid = np.mean(mat, axis=0)
    cov = np.cov(mat.T) + 0.001 * np.eye(DIM)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        centroid /= np.linalg.norm(centroid) + 1e-12
        return tgt_mat @ centroid

    diff = tgt_mat - centroid[None, :]
    maha = np.sum((diff @ inv_cov) * diff, axis=1)
    return -maha  # negate: lower distance = better


def method_rrf_blend(source, seeds, exclude, tgt_mat, all_words):
    """RRF blend of cosine-source + cosine-centroid + max-cosine + Gram."""
    K = 13
    methods = [
        method_cosine_source,
        method_cosine_centroid,
        method_max_cosine,
        method_gram_ensemble,
    ]
    N = tgt_mat.shape[0]
    rrf = np.zeros(N)
    for m in methods:
        raw = m(source, seeds, exclude, tgt_mat, all_words)
        order = np.argsort(-raw)
        ranks = np.empty(N, dtype=int)
        ranks[order] = np.arange(N)
        rrf += 1.0 / (K + ranks)
    return rrf


# ── Evaluation ──────────────────────────────────────────────────────────

def select_test_words(n=200, min_assoc=15):
    rng = np.random.default_rng(54321)
    candidates = [
        w for w in sorted(assoc.keys())
        if w in emb.src and len(assoc[w]) >= min_assoc
    ]
    idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    idx.sort()
    return [candidates[i] for i in idx]


def evaluate_method(method_fn, test_words, K, rng_seed=12345):
    """Evaluate a method at a given number of seed associates."""
    p_at_10 = []
    p_at_50 = []
    r_at_50 = []

    for i, source in enumerate(test_words):
        associates = [a for a in assoc[source] if a in emb.tgt]
        if len(associates) < K + 5:
            continue

        # Deterministic split: first K as seeds, rest as test
        rng = np.random.default_rng(rng_seed + i)
        perm = rng.permutation(len(associates))
        seed_idx = perm[:K]
        test_idx = perm[K:]
        seeds = [associates[j] for j in seed_idx]
        test_set = {associates[j] for j in test_idx}

        exclude = set(seeds) | {source}
        all_words = [w for w in emb.vocab if w not in exclude and w in emb.tgt]
        tgt_mat = np.stack([emb.tgt[w] for w in all_words])

        scores = method_fn(source, seeds, exclude, tgt_mat, all_words)

        order = np.argsort(-scores)
        ranked = [(float(scores[idx]), all_words[idx]) for idx in order]

        # Metrics
        top10 = {w for _, w in ranked[:10]}
        top50 = {w for _, w in ranked[:50]}
        p_at_10.append(len(top10 & test_set) / 10)
        p_at_50.append(len(top50 & test_set) / 50)
        r_at_50.append(len(top50 & test_set) / len(test_set) if test_set else 0)

    return {
        "p@10": np.mean(p_at_10) if p_at_10 else 0,
        "p@50": np.mean(p_at_50) if p_at_50 else 0,
        "r@50": np.mean(r_at_50) if r_at_50 else 0,
        "n_eval": len(p_at_10),
    }


# ── Main ────────────────────────────────────────────────────────────────

METHODS = {
    "cosine_src":     method_cosine_source,
    "cosine_cent":    method_cosine_centroid,
    "max_cosine":     method_max_cosine,
    "src+cent_blend": method_source_centroid_blend,
    "mahalanobis":    method_mahalanobis,
    "gram_0.05":      method_gram_ensemble,
    "rrf_blend":      method_rrf_blend,
}


def main():
    print("=" * 80)
    print("Generative Associate Expansion Benchmark")
    print(f"Multi-seed Gram^{GRAM_POWER} ({N_SEEDS_GEO} seeds) vs embedding baselines")
    print(f"Vocab: {len(emb.vocab):,} words, {DIM}D SVD embeddings")
    print("=" * 80)

    test_words = select_test_words(n=200)
    print(f"Test words: {len(test_words)}")

    # Header
    print(f"\n{'K':>3} | ", end="")
    for name in METHODS:
        print(f"{name:>14}", end="")
    print()
    print("-" * (6 + 14 * len(METHODS)))

    results_table = {}

    for K in K_VALUES:
        print(f"{K:>3} | ", end="", flush=True)
        row = {}
        for name, fn in METHODS.items():
            t0 = time.time()
            r = evaluate_method(fn, test_words, K)
            elapsed = time.time() - t0
            row[name] = r
            print(f"{r['p@10']:>13.4f}", end="", flush=True)
        print()
        results_table[K] = row

    # Detailed summary for each K
    for K in K_VALUES:
        print(f"\n{'='*60}")
        print(f"K={K} seed associates — detailed metrics")
        print(f"{'='*60}")
        print(f"  {'Method':<16} {'p@10':>8} {'p@50':>8} {'r@50':>8}")
        print(f"  {'-'*44}")
        row = results_table[K]
        for name in METHODS:
            r = row[name]
            print(f"  {name:<16} {r['p@10']:>8.4f} {r['p@50']:>8.4f} {r['r@50']:>8.4f}")

    # Final comparison
    print(f"\n{'='*80}")
    print("SUMMARY — p@10 across K values")
    print(f"{'='*80}")
    print(f"  {'Method':<16}", end="")
    for K in K_VALUES:
        print(f" {'K='+str(K):>8}", end="")
    print()
    print(f"  {'-'*16}", end="")
    for K in K_VALUES:
        print(f" {'--------':>8}", end="")
    print()
    for name in METHODS:
        print(f"  {name:<16}", end="")
        for K in K_VALUES:
            v = results_table[K][name]["p@10"]
            print(f" {v:>8.4f}", end="")
        print()

    # Find where geometry wins
    print(f"\n  Geometry advantage (gram_0.05 - best embedding method):")
    for K in K_VALUES:
        geo = results_table[K]["gram_0.05"]["p@10"]
        emb_methods = ["cosine_src", "cosine_cent", "max_cosine",
                       "src+cent_blend", "mahalanobis"]
        best_emb = max(results_table[K][m]["p@10"] for m in emb_methods)
        diff = geo - best_emb
        winner = "GEOMETRY" if diff > 0 else "EMBEDDING"
        print(f"    K={K:>2}: {diff:>+.4f} ({winner})")


if __name__ == "__main__":
    main()
