"""
examples/multi_transversal_generation.py
=========================================
Multiple transversals as intersected constraints for generation.

The single-transversal approach fails at 67K scale because 1 scalar
constraint (|⟨T, L⟩| ≈ 0) has too many accidental near-zeros among
67K candidates.

Fix: sample many different 4-tuples from a word's associates, compute
a transversal from each, then rank candidates by their COMBINED score
across all transversals. An accidental near-zero on one transversal
is unlikely to be near-zero on all of them.

Results: with 20 transversals, the generated words are semantically
coherent for most concepts (bassist/trumpeter for "music",
boulder/crevasse for "mountain", etc.) even though they may not be
in the original associate list.
"""

import pickle
import sys
import time
import numpy as np

sys.path.insert(0, "..")

from transversal_memory import P3Memory, plucker_inner
from transversal_memory.plucker import random_projection_dual
from transversal_memory.cooccurrence import SVDEmbeddings


# ── Load cached data ─────────────────────────────────────────────────────

with open("data/cache/associations.pkl", "rb") as f:
    assoc = pickle.load(f)
with open("data/cache/embeddings_dim32.pkl", "rb") as f:
    emb = pickle.load(f)

DIM = 32
W1, W2 = random_projection_dual(DIM, np.random.default_rng(99))


# ── Core: compute multiple transversals ──────────────────────────────────

def compute_transversals(source, targets, n_transversals=20, rng=None):
    """
    Sample random 4-tuples from targets, compute a transversal from each.
    Returns list of transversal 6-vectors.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Pre-compute all lines
    target_lines = {}
    for t in targets:
        L = emb.make_line_dual(source, t, W1, W2)
        if L is not None:
            target_lines[t] = L

    valid_targets = list(target_lines.keys())
    if len(valid_targets) < 4:
        return []

    transversals = []
    attempts = 0
    max_attempts = n_transversals * 5

    while len(transversals) < n_transversals and attempts < max_attempts:
        attempts += 1
        idx = rng.choice(len(valid_targets), size=4, replace=False)
        four = [valid_targets[i] for i in idx]
        lines = [target_lines[t] for t in four]

        mem = P3Memory()
        mem.store(lines[:3])
        tvs = mem.query_generative(lines[3])

        for T, resid in tvs:
            transversals.append(T)

    return transversals


def rank_by_multi_transversal(source, transversals, exclude=set(),
                              method="sum_log"):
    """
    Rank all vocab words by combined Plücker inner product across
    multiple transversals.

    Methods:
      - "sum_log": sum of log(|⟨T_i, L⟩| + eps)
      - "max": max |⟨T_i, L⟩| — most conservative
      - "mean": mean |⟨T_i, L⟩|
    """
    skip = exclude | {source}
    eps = 1e-20

    results = []
    for word in emb.vocab:
        if word in skip:
            continue
        L = emb.make_line_dual(source, word, W1, W2)
        if L is None:
            continue

        pis = [abs(plucker_inner(T, L)) for T in transversals]

        if method == "sum_log":
            score = sum(np.log(pi + eps) for pi in pis)
        elif method == "max":
            score = max(pis)
        elif method == "mean":
            score = np.mean(pis)
        else:
            raise ValueError(f"Unknown method: {method}")

        results.append((score, word))

    results.sort(key=lambda x: x[0])
    return results


def semantic_similarity(word1, word2):
    """Cosine similarity between two words in target embedding space."""
    if word1 not in emb.tgt or word2 not in emb.tgt:
        return 0.0
    v1 = emb.tgt[word1]
    v2 = emb.tgt[word2]
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / d) if d > 0 else 0.0


def evaluate_semantic_quality(source, generated_words, k=10):
    """
    Evaluate whether generated words are semantically related to source,
    using embedding cosine similarity as a proxy.
    Compare generated top-k vs random baseline.
    """
    gen = generated_words[:k]
    gen_sims = [semantic_similarity(source, w) for _, w in gen]

    # Random baseline
    rng = np.random.default_rng(0)
    vocab_list = list(emb.vocab)
    rand_idx = rng.choice(len(vocab_list), size=200, replace=False)
    rand_sims = [semantic_similarity(source, vocab_list[i]) for i in rand_idx]

    return {
        "gen_mean": np.mean(gen_sims) if gen_sims else 0,
        "gen_median": np.median(gen_sims) if gen_sims else 0,
        "rand_mean": np.mean(rand_sims),
        "rand_median": np.median(rand_sims),
        "gen_sims": gen_sims,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Multi-Transversal Generation")
    print("=" * 70)

    # ── Test 1: Effect of transversal count ──────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: Effect of transversal count (king)")
    print("=" * 70)

    source = "king"
    known = set(assoc[source])
    targets = assoc[source]

    for n_t in [1, 5, 10, 20, 40]:
        transversals = compute_transversals(
            source, targets, n_transversals=n_t,
            rng=np.random.default_rng(42))

        if not transversals:
            continue

        ranked = rank_by_multi_transversal(
            source, transversals, exclude=known, method="sum_log")
        top10 = [w for _, w in ranked[:10]]
        eval_result = evaluate_semantic_quality(source, ranked[:10])

        print(f"\n  {n_t:2d}T (got {len(transversals):2d}): "
              f"sim={eval_result['gen_mean']:.3f} "
              f"(rand={eval_result['rand_mean']:.3f})")
        print(f"    {', '.join(top10)}")

    # ── Test 2: Generation across many concepts ──────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Multi-transversal generation (20T, sum_log)")
    print("=" * 70)
    print("  Semantic similarity: generated top-10 vs random baseline")

    test_words = ["king", "ocean", "fire", "love", "brain", "music",
                  "dog", "tree", "computer", "war", "money", "dream",
                  "doctor", "mountain", "guitar", "river", "school",
                  "sun", "food", "car"]

    all_gen_sims = []
    all_rand_sims = []

    for source in test_words:
        if source not in assoc or source not in emb.src:
            continue
        targets = assoc[source]
        known = set(targets)

        transversals = compute_transversals(
            source, targets, n_transversals=20,
            rng=np.random.default_rng(42))

        if len(transversals) < 2:
            continue

        ranked = rank_by_multi_transversal(
            source, transversals, exclude=known, method="sum_log")
        top10 = [w for _, w in ranked[:10]]
        ev = evaluate_semantic_quality(source, ranked[:10])
        all_gen_sims.append(ev["gen_mean"])
        all_rand_sims.append(ev["rand_mean"])

        ratio = ev["gen_mean"] / ev["rand_mean"] if ev["rand_mean"] > 0 else 0
        print(f"\n  {source:12s}  sim={ev['gen_mean']:.3f}  "
              f"rand={ev['rand_mean']:.3f}  ratio={ratio:.1f}x")
        print(f"    {', '.join(top10)}")

    if all_gen_sims:
        print(f"\n  {'OVERALL':12s}  sim={np.mean(all_gen_sims):.3f}  "
              f"rand={np.mean(all_rand_sims):.3f}  "
              f"ratio={np.mean(all_gen_sims)/np.mean(all_rand_sims):.1f}x")

    # ── Test 3: Continuous generation (fixed seed pool) ──────────────
    print("\n" + "=" * 70)
    print("TEST 3: Continuous Generation (fixed associate pool)")
    print("=" * 70)
    print("  Use ALL known associates for transversals (never slide noise in)")
    print("  Generate 20 new words per concept")

    for source in ["king", "fire", "ocean", "brain", "music",
                   "mountain", "guitar", "doctor"]:
        if source not in assoc or source not in emb.src:
            continue

        targets = assoc[source]
        known = set(targets)

        t0 = time.time()
        transversals = compute_transversals(
            source, targets, n_transversals=30,
            rng=np.random.default_rng(42))
        t_trans = time.time() - t0

        if len(transversals) < 5:
            continue

        t0 = time.time()
        ranked = rank_by_multi_transversal(
            source, transversals, exclude=known, method="sum_log")
        t_rank = time.time() - t0

        top20 = ranked[:20]
        words = [w for _, w in top20]
        sims = [semantic_similarity(source, w) for w in words]

        print(f"\n  {source} ({len(transversals)}T, {t_trans:.1f}s + "
              f"{t_rank:.1f}s)")

        # Print in two rows of 10
        for row_start in [0, 10]:
            row_words = words[row_start:row_start+10]
            row_sims = sims[row_start:row_start+10]
            items = [f"{w}({s:.2f})" for w, s in zip(row_words, row_sims)]
            print(f"    {', '.join(items)}")

    # ── Test 4: Chained generation with re-sampling ──────────────────
    print("\n" + "=" * 70)
    print("TEST 4: Chained generation (grow pool with good words only)")
    print("=" * 70)
    print("  Generate a word, check its similarity, add to pool if good")

    for source in ["king", "music", "mountain", "fire", "brain", "doctor"]:
        if source not in assoc or source not in emb.src:
            continue

        targets = list(assoc[source])
        known = set(targets)
        pool = list(targets)  # mutable copy
        seen = set(targets) | {source}
        generated = []
        sim_threshold = 0.15  # add to pool if sim > this

        for step in range(15):
            transversals = compute_transversals(
                source, pool, n_transversals=20,
                rng=np.random.default_rng(42 + step))

            if len(transversals) < 3:
                break

            ranked = rank_by_multi_transversal(
                source, transversals, exclude=seen, method="sum_log")

            if not ranked:
                break

            new_word = ranked[0][1]
            sim = semantic_similarity(source, new_word)
            seen.add(new_word)

            # Only add to pool if semantically related
            if sim > sim_threshold:
                pool.append(new_word)
                generated.append(f"{new_word}({sim:.2f})+")
            else:
                generated.append(f"{new_word}({sim:.2f})")

        print(f"\n  {source}:")
        # Print 5 per line
        for i in range(0, len(generated), 5):
            print(f"    {', '.join(generated[i:i+5])}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
