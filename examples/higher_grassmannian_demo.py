"""
examples/higher_grassmannian_demo.py
=====================================
Compare generation quality across different Grassmannians:
  G(2,4) = P³:  6D,  need 4 lines
  G(2,5) = P⁴: 10D,  need 8 lines
  G(2,6) = P⁵: 15D,  need 13 lines

Higher D = more Plücker coordinates = richer signatures = (hopefully)
better discrimination at 67K vocabulary scale.
"""

import pickle
import sys
import time
import numpy as np

sys.path.insert(0, "..")

from transversal_memory.higher_grass import (
    plucker_dim, lines_needed,
    project_to_line_dual_general,
    random_projection_dual_general,
    plucker_inner_general,
    HigherGramMemory, HigherP3Memory,
)
from transversal_memory.plucker import (
    project_to_line_dual, random_projection_dual, plucker_inner,
)
from transversal_memory import P3Memory

# ── Load cached data ─────────────────────────────────────────────────────

with open("data/cache/associations.pkl", "rb") as f:
    assoc = pickle.load(f)
with open("data/cache/embeddings_dim32.pkl", "rb") as f:
    emb = pickle.load(f)

DIM = 32


def semantic_similarity(word1, word2):
    if word1 not in emb.tgt or word2 not in emb.tgt:
        return 0.0
    v1, v2 = emb.tgt[word1], emb.tgt[word2]
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / d) if d > 0 else 0.0


# ── Test functions ───────────────────────────────────────────────────────

def test_single_transversal(source, targets, n_proj, rng_seed=99):
    """
    Single transversal generation for G(2, n_proj+1).
    Store D-2 lines, query with 1, decode vocabulary.
    """
    D = plucker_dim(n_proj)
    K = lines_needed(n_proj)  # lines to store (D-2) + 1 query = D-1

    if len(targets) < K + 1:
        return None

    rng = np.random.default_rng(rng_seed)
    W1, W2 = random_projection_dual_general(DIM, n_proj, rng)

    # Encode lines
    target_lines = {}
    for t in targets:
        if t in emb.src and t in emb.tgt:
            a, b = emb.src[source], emb.tgt[t]
            L = project_to_line_dual_general(a, b, W1, W2, n_proj)
            if np.linalg.norm(L) > 1e-12:
                target_lines[t] = L

    valid_targets = list(target_lines.keys())
    if len(valid_targets) < K + 1:
        return None

    # Store K lines, query with (K+1)th
    stored = [target_lines[valid_targets[i]] for i in range(K)]
    query = target_lines[valid_targets[K]]

    mem = HigherP3Memory(n_proj=n_proj)
    mem.store(stored)
    tvs = mem.query_generative(query)

    if not tvs:
        return None

    T, resid = tvs[0]

    # Decode: rank vocabulary by |plucker_inner_general(T, L)|
    exclude = set(valid_targets[:K+1]) | {source}
    results = []
    known = set(targets)

    for word in emb.vocab:
        if word in exclude:
            continue
        if word not in emb.tgt:
            continue
        a, b = emb.src[source], emb.tgt[word]
        L = project_to_line_dual_general(a, b, W1, W2, n_proj)
        if np.linalg.norm(L) > 1e-12:
            pi = abs(plucker_inner_general(T, L, n_proj))
            results.append((pi, word))

    results.sort(key=lambda x: x[0])

    top10 = results[:10]
    top10_words = [w for _, w in top10]
    sims = [semantic_similarity(source, w) for w in top10_words]
    n_known = sum(1 for w in top10_words if w in known)

    return {
        "n_proj": n_proj,
        "D": D,
        "K": K,
        "resid": resid,
        "top10": top10,
        "sims": sims,
        "mean_sim": np.mean(sims),
        "n_known": n_known,
        "n_candidates": len(results),
    }


def test_multi_transversal(source, targets, n_proj, n_transversals=20,
                           rng_seed=99):
    """
    Multi-transversal generation for G(2, n_proj+1).
    Sample random subsets, compute transversals, rank by combined score.
    """
    D = plucker_dim(n_proj)
    K = lines_needed(n_proj)

    if len(targets) < K + 1:
        return None

    rng = np.random.default_rng(rng_seed)
    W1, W2 = random_projection_dual_general(DIM, n_proj, rng)

    # Encode all target lines
    target_lines = {}
    for t in targets:
        if t in emb.src and t in emb.tgt:
            a, b = emb.src[source], emb.tgt[t]
            L = project_to_line_dual_general(a, b, W1, W2, n_proj)
            if np.linalg.norm(L) > 1e-12:
                target_lines[t] = L

    valid_targets = list(target_lines.keys())
    if len(valid_targets) < K + 1:
        return None

    # Compute multiple transversals
    sample_rng = np.random.default_rng(42)
    transversals = []
    attempts = 0

    while len(transversals) < n_transversals and attempts < n_transversals * 5:
        attempts += 1
        idx = sample_rng.choice(len(valid_targets), size=K + 1, replace=False)
        subset = [valid_targets[i] for i in idx]
        lines = [target_lines[t] for t in subset]

        mem = HigherP3Memory(n_proj=n_proj)
        mem.store(lines[:K])
        tvs = mem.query_generative(lines[K])

        for T, resid in tvs:
            transversals.append(T)

    if len(transversals) < 2:
        return None

    # Decode: rank by sum_log of |plucker_inner| across transversals
    exclude = set(targets) | {source}
    eps = 1e-20
    results = []
    known = set(targets)

    for word in emb.vocab:
        if word in exclude or word not in emb.tgt:
            continue
        a, b = emb.src[source], emb.tgt[word]
        L = project_to_line_dual_general(a, b, W1, W2, n_proj)
        if np.linalg.norm(L) < 1e-12:
            continue

        pis = [abs(plucker_inner_general(T, L, n_proj)) for T in transversals]
        score = sum(np.log(pi + eps) for pi in pis)
        results.append((score, word))

    results.sort(key=lambda x: x[0])

    top10 = results[:10]
    top10_words = [w for _, w in top10]
    sims = [semantic_similarity(source, w) for w in top10_words]

    return {
        "n_proj": n_proj,
        "D": D,
        "K": K,
        "n_transversals": len(transversals),
        "top10": top10,
        "sims": sims,
        "mean_sim": np.mean(sims),
        "n_candidates": len(results),
    }


def test_gram_discrimination(source, targets, n_proj, rng_seed=99):
    """
    GramMemory discrimination: held-out vs random scores in G(2, n+1).
    Higher D should give better separation.
    """
    D = plucker_dim(n_proj)
    rng = np.random.default_rng(rng_seed)
    W1, W2 = random_projection_dual_general(DIM, n_proj, rng)

    split = max(3, int(len(targets) * 0.75))
    train = targets[:split]
    held_out = targets[split:]

    gram = HigherGramMemory(n_proj=n_proj)
    for t in train:
        if t in emb.tgt:
            a, b = emb.src[source], emb.tgt[t]
            L = project_to_line_dual_general(a, b, W1, W2, n_proj)
            if np.linalg.norm(L) > 1e-12:
                gram.store_line(L)

    if gram.n_lines < 3:
        return None

    # Score held-out
    h_scores = []
    for t in held_out:
        if t in emb.tgt:
            a, b = emb.src[source], emb.tgt[t]
            L = project_to_line_dual_general(a, b, W1, W2, n_proj)
            if np.linalg.norm(L) > 1e-12:
                h_scores.append(gram.score(L))

    # Score random
    sample_rng = np.random.default_rng(0)
    non_assoc = [w for w in emb.vocab if w not in set(targets) and w != source]
    idx = sample_rng.choice(len(non_assoc), size=min(200, len(non_assoc)),
                            replace=False)
    r_scores = []
    for i in idx:
        w = non_assoc[i]
        if w in emb.tgt:
            a, b = emb.src[source], emb.tgt[w]
            L = project_to_line_dual_general(a, b, W1, W2, n_proj)
            if np.linalg.norm(L) > 1e-12:
                r_scores.append(gram.score(L))

    if not h_scores or not r_scores:
        return None

    return {
        "n_proj": n_proj,
        "D": D,
        "h_mean": np.mean(h_scores),
        "r_mean": np.mean(r_scores),
        "separated": np.mean(h_scores) > np.mean(r_scores),
        "n_train": gram.n_lines,
        "rank": np.sum(gram.eigenvalues() > 1e-10),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Higher Grassmannian Comparison")
    print("=" * 70)

    test_words = ["king", "fire", "ocean", "music", "brain", "doctor",
                  "mountain", "guitar"]

    # ── Test 1: GramMemory discrimination across dimensions ──────────
    print("\n" + "=" * 70)
    print("TEST 1: GramMemory Discrimination (held-out vs random)")
    print("=" * 70)

    for n_proj in [3, 4, 5]:
        D = plucker_dim(n_proj)
        print(f"\n  G(2,{n_proj+1}): D={D}")

        n_sep = 0
        n_total = 0
        for source in test_words:
            if source not in assoc or source not in emb.src:
                continue
            result = test_gram_discrimination(source, assoc[source], n_proj)
            if result is None:
                continue
            n_total += 1
            if result["separated"]:
                n_sep += 1
            sep = "✓" if result["separated"] else "✗"
            print(f"    {source:12s}  held={result['h_mean']:.4f}  "
                  f"rand={result['r_mean']:.4f}  rank={result['rank']:2d}/{D}  {sep}")

        if n_total:
            print(f"    Separated: {n_sep}/{n_total}")

    # ── Test 2: Single transversal generation ────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Single Transversal Generation")
    print("=" * 70)

    for n_proj in [3, 4, 5]:
        D = plucker_dim(n_proj)
        K = lines_needed(n_proj)
        print(f"\n  G(2,{n_proj+1}): D={D}, need {K} stored + 1 query")

        sims_all = []
        for source in test_words:
            if source not in assoc or source not in emb.src:
                continue
            if len(assoc[source]) < K + 1:
                print(f"    {source:12s}  not enough associates "
                      f"({len(assoc[source])} < {K+1})")
                continue

            t0 = time.time()
            result = test_single_transversal(source, assoc[source], n_proj)
            elapsed = time.time() - t0

            if result is None:
                print(f"    {source:12s}  no transversal ({elapsed:.1f}s)")
                continue

            sims_all.append(result["mean_sim"])
            top5 = [w for _, w in result["top10"][:5]]
            print(f"    {source:12s}  sim={result['mean_sim']:.3f}  "
                  f"({elapsed:.1f}s)  {', '.join(top5)}")

        if sims_all:
            print(f"    Mean similarity: {np.mean(sims_all):.3f}")

    # ── Test 3: Multi-transversal generation ─────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: Multi-Transversal Generation (10T)")
    print("=" * 70)

    for n_proj in [3, 4, 5]:
        D = plucker_dim(n_proj)
        K = lines_needed(n_proj)
        n_t = 10
        print(f"\n  G(2,{n_proj+1}): D={D}, {n_t} transversals "
              f"(each needs {K+1} lines)")

        sims_all = []
        for source in test_words:
            if source not in assoc or source not in emb.src:
                continue
            if len(assoc[source]) < K + 1:
                continue

            t0 = time.time()
            result = test_multi_transversal(
                source, assoc[source], n_proj, n_transversals=n_t)
            elapsed = time.time() - t0

            if result is None:
                print(f"    {source:12s}  failed ({elapsed:.1f}s)")
                continue

            sims_all.append(result["mean_sim"])
            top5 = [w for _, w in result["top10"][:5]]
            print(f"    {source:12s}  sim={result['mean_sim']:.3f}  "
                  f"{result['n_transversals']}T  ({elapsed:.1f}s)  "
                  f"{', '.join(top5)}")

        if sims_all:
            print(f"    Mean similarity: {np.mean(sims_all):.3f}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
