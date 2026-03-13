"""
evaluate.py — Fixed evaluation harness for transversal memory experiments
=========================================================================

DO NOT MODIFY THIS FILE. It defines the ground-truth metrics and baselines.

Evaluation protocol:
  1. Sample N source words (with >= min_associates associates)
  2. For each source word:
     a. Hold out 25% of associates as test set
     b. Pass train set to experiment's build() function
     c. Experiment returns ranked vocabulary list
     d. Measure where held-out associates land in the ranking
  3. Aggregate metrics across all source words

Primary metric: mean_precision_at_10 (higher is better)

Baselines:
  - cosine: rank by cos(source_embedding, target_embedding)
  - random: expected precision = n_associates / vocab_size ≈ 0.0004
"""

import pickle
import time
import numpy as np
from pathlib import Path


# ── Fixed constants ──────────────────────────────────────────────────────────

DATA_DIR = Path("data/cache")
N_TEST_WORDS = 200          # number of source words to evaluate
MIN_ASSOCIATES = 15         # minimum associates per source word
HOLDOUT_FRACTION = 0.25     # fraction of associates held out for testing
RNG_SEED = 12345            # fixed seed for reproducible train/test splits
VOCAB_SAMPLE_SEED = 54321   # fixed seed for selecting test words


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load associations and embeddings. Returns (assoc, emb)."""
    with open(DATA_DIR / "associations.pkl", "rb") as f:
        assoc = pickle.load(f)
    with open(DATA_DIR / "embeddings_dim32.pkl", "rb") as f:
        emb = pickle.load(f)
    return assoc, emb


def select_test_words(assoc, emb, n=N_TEST_WORDS, min_assoc=MIN_ASSOCIATES):
    """
    Select N test words deterministically.
    Requirements: word must be in both assoc and emb.src, with >= min_assoc associates.
    """
    rng = np.random.default_rng(VOCAB_SAMPLE_SEED)
    candidates = [
        w for w in sorted(assoc.keys())
        if w in emb.src and len(assoc[w]) >= min_assoc
    ]
    idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    idx.sort()
    return [candidates[i] for i in idx]


def split_associates(associates, seed):
    """
    Split associates into train/test sets deterministically.
    Returns (train_list, test_set).
    """
    rng = np.random.default_rng(seed)
    n = len(associates)
    n_test = max(1, int(n * HOLDOUT_FRACTION))
    perm = rng.permutation(n)
    test_idx = set(perm[:n_test])
    train = [associates[i] for i in range(n) if i not in test_idx]
    test = {associates[i] for i in test_idx}
    return train, test


# ── Cosine baseline ──────────────────────────────────────────────────────────

def cosine_baseline(source, emb, exclude=set()):
    """
    Rank all vocabulary by cosine similarity to source embedding.
    Returns list of (score, word) sorted descending (best first).
    """
    if source not in emb.src:
        return []
    src_vec = emb.src[source]
    results = []
    for w in emb.vocab:
        if w in exclude or w not in emb.tgt:
            continue
        sim = float(np.dot(src_vec, emb.tgt[w]))
        results.append((sim, w))
    results.sort(key=lambda x: -x[0])
    return results


def cosine_baseline_batch(source, emb, exclude=set()):
    """
    Vectorised cosine baseline. Returns list of (score, word), best first.
    """
    if source not in emb.src:
        return []
    src_vec = emb.src[source]
    words = [w for w in emb.vocab if w not in exclude and w in emb.tgt]
    tgt_mat = np.stack([emb.tgt[w] for w in words])
    scores = tgt_mat @ src_vec  # (N,) — embeddings are already normalised
    order = np.argsort(-scores)
    return [(float(scores[i]), words[i]) for i in order]


# ── Metrics ──────────────────────────────────────────────────────────────────

def precision_at_k(ranked_words, test_set, k):
    """Fraction of top-k results that are in test_set."""
    top_k = [w for _, w in ranked_words[:k]]
    return sum(1 for w in top_k if w in test_set) / k if k > 0 else 0.0


def recall_at_k(ranked_words, test_set, k):
    """Fraction of test_set found in top-k results."""
    top_k = {w for _, w in ranked_words[:k]}
    return sum(1 for w in test_set if w in top_k) / len(test_set) if test_set else 0.0


def mean_reciprocal_rank(ranked_words, test_set):
    """
    Mean reciprocal rank of test set items.
    MRR = mean(1/rank_i) for each test item i.
    """
    word_to_rank = {}
    for rank, (_, w) in enumerate(ranked_words, 1):
        if w in test_set and w not in word_to_rank:
            word_to_rank[w] = rank
    if not test_set:
        return 0.0
    rrs = []
    for w in test_set:
        if w in word_to_rank:
            rrs.append(1.0 / word_to_rank[w])
        else:
            rrs.append(0.0)
    return float(np.mean(rrs))


def median_rank(ranked_words, test_set):
    """Median rank of test set items in the ranking."""
    word_to_rank = {}
    for rank, (_, w) in enumerate(ranked_words, 1):
        if w in test_set and w not in word_to_rank:
            word_to_rank[w] = rank
    ranks = [word_to_rank.get(w, len(ranked_words)) for w in test_set]
    return float(np.median(ranks)) if ranks else float(len(ranked_words))


def compute_metrics(ranked_words, test_set):
    """Compute all metrics for a single source word."""
    return {
        "p@10": precision_at_k(ranked_words, test_set, 10),
        "p@50": precision_at_k(ranked_words, test_set, 50),
        "r@50": recall_at_k(ranked_words, test_set, 50),
        "r@200": recall_at_k(ranked_words, test_set, 200),
        "mrr": mean_reciprocal_rank(ranked_words, test_set),
        "median_rank": median_rank(ranked_words, test_set),
    }


# ── Main evaluation loop ────────────────────────────────────────────────────

def evaluate(build_fn, rank_fn, verbose=True):
    """
    Run the full evaluation protocol.

    build_fn(source, train_associates, emb) -> state
        Called once per source word with training associates.
        Returns arbitrary state passed to rank_fn.

    rank_fn(source, state, emb, exclude) -> [(score, word), ...]
        Called once per source word. Must return ranked vocabulary.
        Lower-ranked (earlier) items are considered better matches.
        `exclude` is the set of words to skip (train associates + source).

    Returns dict with aggregated metrics and per-word details.
    """
    assoc, emb = load_data()
    test_words = select_test_words(assoc, emb)

    if verbose:
        print(f"Evaluating on {len(test_words)} words "
              f"(vocab={len(emb.vocab)}, holdout={HOLDOUT_FRACTION:.0%})")

    all_metrics = []
    baseline_metrics = []
    per_word = []
    total_time = 0.0

    for i, source in enumerate(test_words):
        associates = assoc[source]
        train, test = split_associates(associates, RNG_SEED + i)
        exclude = set(train) | {source}

        # Experiment
        t0 = time.time()
        state = build_fn(source, train, emb)
        ranked = rank_fn(source, state, emb, exclude)
        elapsed = time.time() - t0
        total_time += elapsed

        m = compute_metrics(ranked, test)
        all_metrics.append(m)

        # Cosine baseline
        baseline_ranked = cosine_baseline_batch(source, emb, exclude)
        bm = compute_metrics(baseline_ranked, test)
        baseline_metrics.append(bm)

        per_word.append({
            "word": source,
            "n_train": len(train),
            "n_test": len(test),
            "time": elapsed,
            "metrics": m,
            "baseline": bm,
        })

        if verbose and (i + 1) % 50 == 0:
            avg_p10 = np.mean([m["p@10"] for m in all_metrics])
            avg_base = np.mean([m["p@10"] for m in baseline_metrics])
            print(f"  [{i+1}/{len(test_words)}] p@10={avg_p10:.4f} "
                  f"(baseline={avg_base:.4f})")

    # Aggregate
    result = {
        "n_words": len(test_words),
        "total_time": total_time,
        "mean_time_per_word": total_time / len(test_words),
    }

    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        base_vals = [m[key] for m in baseline_metrics]
        result[f"mean_{key}"] = float(np.mean(vals))
        result[f"baseline_{key}"] = float(np.mean(base_vals))

    result["per_word"] = per_word

    if verbose:
        print("\n" + "=" * 60)
        print(f"{'Metric':<20s} {'Experiment':>12s} {'Cosine NN':>12s} {'Lift':>8s}")
        print("-" * 60)
        for key in ["p@10", "p@50", "r@50", "r@200", "mrr", "median_rank"]:
            exp_val = result[f"mean_{key}"]
            base_val = result[f"baseline_{key}"]
            if key == "median_rank":
                lift = f"{base_val/exp_val:.1f}x" if exp_val > 0 else "n/a"
            else:
                lift = f"{exp_val/base_val:.1f}x" if base_val > 0 else "n/a"
            print(f"  {key:<18s} {exp_val:>12.4f} {base_val:>12.4f} {lift:>8s}")
        print(f"\n  Time: {total_time:.1f}s total, "
              f"{total_time/len(test_words)*1000:.0f}ms/word")
        print("=" * 60)

    return result


# ── Standalone: run cosine baseline only ─────────────────────────────────────

if __name__ == "__main__":
    print("Running cosine-NN baseline evaluation...")
    print("=" * 60)

    def build_cosine(source, train, emb):
        return None

    def rank_cosine(source, state, emb, exclude):
        return cosine_baseline_batch(source, emb, exclude)

    result = evaluate(build_cosine, rank_cosine)
    print(f"\nPrimary metric (mean p@10): {result['mean_p@10']:.6f}")
