"""
examples/word_associations_multiseed.py
========================================
Upgraded word associations using multi-seed Gram^0.05 ensemble.

Demonstrates both discriminative and generative modes on the full
Overmann association dataset (~67K vocab, SVD embeddings).

Key improvements over word_associations.py:
  - N_SEEDS random projections instead of 1
  - Gram^0.05 eigenvalue compression
  - Full-scale vocabulary (67K words)
  - Side-by-side comparison with single-seed baseline

These techniques improved pure geometry from p@10=0.011 to 0.1065
in the discriminative ranking benchmark (9.7x improvement).
"""

import pickle
import sys
import time
import numpy as np

sys.path.insert(0, "..")

from transversal_memory import GramMemory, plucker_inner
from transversal_memory.plucker import (
    random_projection, random_projection_dual,
    batch_encode_lines_dual, project_to_line, project_to_line_dual,
)
from transversal_memory.cooccurrence import SVDEmbeddings


# ── Config ──────────────────────────────────────────────────────────────

N_SEEDS = 50
SEED_SPACING = 10
GRAM_POWER = 0.05
DIM = 32


# ── Load data ───────────────────────────────────────────────────────────

with open("data/cache/associations.pkl", "rb") as f:
    assoc = pickle.load(f)
with open("data/cache/embeddings_dim32.pkl", "rb") as f:
    emb = pickle.load(f)

SEED_LIST = [i * SEED_SPACING for i in range(N_SEEDS)]

# Precompute projections
PROJECTIONS_DUAL = []
for seed in SEED_LIST:
    rng = np.random.default_rng(seed)
    W1, W2 = random_projection_dual(DIM, rng)
    PROJECTIONS_DUAL.append((W1, W2))

# Single-seed baseline projection
W_SINGLE = random_projection(DIM, np.random.default_rng(42))


# ── Multi-seed Gram infrastructure ─────────────────────────────────────

def _power_gram(gram, power):
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


def build_multiseed(source, targets):
    """Build per-seed Gram^0.05 matrices."""
    src_vec = emb.src[source]
    per_seed = []
    for W1, W2 in PROJECTIONS_DUAL:
        lines = []
        for t in targets:
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


def score_multiseed(src_vec, per_seed, candidate_words):
    """Score candidates with multi-seed ensemble. Returns [(score, word)]."""
    cand_words = [w for w in candidate_words if w in emb.tgt]
    if not cand_words or not per_seed:
        return []
    tgt_mat = np.stack([emb.tgt[w] for w in cand_words])
    N = len(cand_words)
    acc = np.zeros(N)
    for sd in per_seed:
        lines = batch_encode_lines_dual(src_vec, tgt_mat, sd["W1"], sd["W2"])
        acc += np.sum((lines @ sd["gram"]) * lines, axis=1)
    acc /= len(per_seed)
    scored = list(zip(acc.tolist(), cand_words))
    scored.sort(key=lambda x: -x[0])
    return scored


def build_singleseed(source, targets):
    """Build single-seed GramMemory (baseline)."""
    gram = GramMemory()
    for t in targets:
        if t in emb.tgt:
            L = project_to_line(emb.src[source], emb.tgt[t], W_SINGLE)
            if np.linalg.norm(L) > 1e-12:
                gram.store_line(L)
    return gram


def score_singleseed(source, gram, candidate_words):
    """Score candidates with single-seed GramMemory."""
    scored = []
    src_vec = emb.src[source]
    for w in candidate_words:
        if w in emb.tgt:
            L = project_to_line(src_vec, emb.tgt[w], W_SINGLE)
            if np.linalg.norm(L) > 1e-12:
                scored.append((gram.score(L), w))
    scored.sort(key=lambda x: -x[0])
    return scored


# ── Select test words ──────────────────────────────────────────────────

def get_test_words(n=20, min_assocs=15):
    """Get words with enough associates for meaningful testing."""
    words = []
    for source, targets in assoc.items():
        if source in emb.src and len(targets) >= min_assocs:
            filtered = [t for t in targets if t in emb.tgt]
            if len(filtered) >= min_assocs:
                words.append((source, filtered))
                if len(words) >= n:
                    break
    return words


# ═════════════════════════════════════════════════════════════════════════
# Demo 1: Discriminative scoring comparison
# ═════════════════════════════════════════════════════════════════════════

def demo_discriminative():
    print("=" * 70)
    print("DEMO 1: Discriminative Scoring — Single-seed vs Multi-seed Gram^0.05")
    print("=" * 70)

    test_words = get_test_words(n=10, min_assocs=15)

    for source, targets in test_words[:5]:
        n_train = int(0.75 * len(targets))
        train = targets[:n_train]
        test = targets[n_train:]

        # Build both
        gram_single = build_singleseed(source, train)
        per_seed, src_vec = build_multiseed(source, train)

        # Score test associates and random non-associates
        non_assocs = [w for w in list(emb.vocab)[:1000]
                      if w not in set(targets) and w != source and w in emb.tgt][:len(test)]

        print(f"\n  '{source}' (train={len(train)}, test={len(test)}):")

        # Single-seed scores
        test_s1 = [gram_single.score(project_to_line(emb.src[source], emb.tgt[w], W_SINGLE))
                   for w in test if w in emb.tgt]
        non_s1 = [gram_single.score(project_to_line(emb.src[source], emb.tgt[w], W_SINGLE))
                  for w in non_assocs if w in emb.tgt]

        # Multi-seed scores
        test_scored = score_multiseed(src_vec, per_seed, test)
        non_scored = score_multiseed(src_vec, per_seed, non_assocs)
        test_sm = [s for s, _ in test_scored] if test_scored else [0]
        non_sm = [s for s, _ in non_scored] if non_scored else [0]

        if test_s1 and non_s1:
            sep1 = np.mean(test_s1) - np.mean(non_s1)
            print(f"    Single-seed:   assoc={np.mean(test_s1):.4f}  "
                  f"non-assoc={np.mean(non_s1):.4f}  gap={sep1:.4f}")
        if test_sm and non_sm:
            sepm = np.mean(test_sm) - np.mean(non_sm)
            print(f"    Multi-seed:    assoc={np.mean(test_sm):.4f}  "
                  f"non-assoc={np.mean(non_sm):.4f}  gap={sepm:.4f}")


# ═════════════════════════════════════════════════════════════════════════
# Demo 2: Full-vocabulary ranking
# ═════════════════════════════════════════════════════════════════════════

def demo_ranking():
    print("\n\n" + "=" * 70)
    print("DEMO 2: Full-Vocabulary Ranking (67K words)")
    print("=" * 70)

    test_words = get_test_words(n=20, min_assocs=15)

    single_ranks = []
    multi_ranks = []
    single_top10 = 0
    multi_top10 = 0
    single_top100 = 0
    multi_top100 = 0

    t0 = time.time()

    for source, targets in test_words:
        n_train = int(0.75 * len(targets))
        train = targets[:n_train]
        test = targets[n_train:]
        exclude = set(train) | {source}

        # Single-seed ranking
        gram_single = build_singleseed(source, train)
        candidates = [w for w in emb.vocab if w not in exclude and w in emb.tgt]
        scored_single = score_singleseed(source, gram_single, candidates)

        # Multi-seed ranking
        per_seed, src_vec = build_multiseed(source, train)
        scored_multi = score_multiseed(src_vec, per_seed, candidates)

        # Find ranks for each test word
        for tw in test:
            # Single
            for i, (s, w) in enumerate(scored_single):
                if w == tw:
                    r = i + 1
                    single_ranks.append(r)
                    if r <= 10: single_top10 += 1
                    if r <= 100: single_top100 += 1
                    break
            # Multi
            for i, (s, w) in enumerate(scored_multi):
                if w == tw:
                    r = i + 1
                    multi_ranks.append(r)
                    if r <= 10: multi_top10 += 1
                    if r <= 100: multi_top100 += 1
                    break

    elapsed = time.time() - t0

    n_s = len(single_ranks) if single_ranks else 1
    n_m = len(multi_ranks) if multi_ranks else 1

    print(f"\n  Evaluated {len(test_words)} words, "
          f"{n_s} held-out associates ({elapsed:.1f}s)\n")

    print(f"  {'Metric':<20} {'Single-seed':>15} {'Multi-seed':>15} {'Improvement':>15}")
    print(f"  {'-'*65}")
    print(f"  {'Median rank':<20} {int(np.median(single_ranks)):>15,} "
          f"{int(np.median(multi_ranks)):>15,} "
          f"{np.median(single_ranks)/max(np.median(multi_ranks),1):>14.0f}x")
    print(f"  {'Mean rank':<20} {int(np.mean(single_ranks)):>15,} "
          f"{int(np.mean(multi_ranks)):>15,} "
          f"{np.mean(single_ranks)/max(np.mean(multi_ranks),1):>14.1f}x")
    print(f"  {'Top-10 hits':<20} {single_top10:>14} ({100*single_top10/n_s:.1f}%) "
          f"{multi_top10:>8} ({100*multi_top10/n_m:.1f}%)")
    print(f"  {'Top-100 hits':<20} {single_top100:>14} ({100*single_top100/n_s:.1f}%) "
          f"{multi_top100:>8} ({100*multi_top100/n_m:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════
# Demo 3: Top-10 qualitative comparison
# ═════════════════════════════════════════════════════════════════════════

def demo_top10():
    print("\n\n" + "=" * 70)
    print("DEMO 3: Top-10 Predictions — Qualitative Comparison")
    print("=" * 70)

    showcase = ["king", "ocean", "dog", "music", "brain"]

    for source in showcase:
        if source not in assoc or source not in emb.src:
            continue
        targets = [t for t in assoc[source] if t in emb.tgt]
        if len(targets) < 10:
            continue

        n_train = int(0.75 * len(targets))
        train = targets[:n_train]
        test_set = set(targets[n_train:])
        exclude = set(train) | {source}
        candidates = [w for w in emb.vocab if w not in exclude and w in emb.tgt]

        # Single-seed
        gram_single = build_singleseed(source, train)
        top_single = score_singleseed(source, gram_single, candidates)[:10]

        # Multi-seed
        per_seed, src_vec = build_multiseed(source, train)
        top_multi = score_multiseed(src_vec, per_seed, candidates)[:10]

        all_assocs = set(targets)
        print(f"\n  '{source}' (trained on {len(train)} associates):")
        print(f"    {'Single-seed top 10:':<25}  {'Multi-seed top 10:':<25}")
        for i in range(10):
            w1 = top_single[i][1] if i < len(top_single) else "—"
            w2 = top_multi[i][1] if i < len(top_multi) else "—"
            m1 = "✓" if w1 in all_assocs else " "
            m2 = "✓" if w2 in all_assocs else " "
            h1 = "★" if w1 in test_set else ""
            h2 = "★" if w2 in test_set else ""
            print(f"    {i+1:>2}. {w1:<20} {m1}{h1:<3}  "
                  f"{i+1:>2}. {w2:<20} {m2}{h2}")


# ═════════════════════════════════════════════════════════════════════════
# Demo 4: Cross-word relational comparison
# ═════════════════════════════════════════════════════════════════════════

def demo_cross_word():
    print("\n\n" + "=" * 70)
    print("DEMO 4: Cross-word Relational Similarity (Multi-seed)")
    print("=" * 70)

    words = ["king", "queen", "prince", "ocean", "river", "lake",
             "dog", "cat", "horse", "music", "art", "dance"]
    available = [w for w in words if w in assoc and w in emb.src]

    # Build multi-seed raw Grams (no power transform) for comparison
    # Power transform compresses eigenvalues → all matrices look identical
    # Raw Gram preserves discriminative relational structure
    grams = {}
    for w in available:
        targets = [t for t in assoc[w] if t in emb.tgt]
        if len(targets) >= 5:
            src_vec = emb.src[w]
            all_grams = []
            for W1, W2 in PROJECTIONS_DUAL:
                lines = []
                for t in targets:
                    if t in emb.tgt:
                        L = project_to_line_dual(src_vec, emb.tgt[t], W1, W2)
                        if np.linalg.norm(L) > 1e-12:
                            lines.append(L)
                if len(lines) >= 2:
                    arr = np.stack(lines)
                    raw_gram = arr.T @ arr
                    # Normalize by trace for comparability
                    tr = np.trace(raw_gram)
                    if tr > 1e-12:
                        all_grams.append(raw_gram / tr)
            if all_grams:
                grams[w] = np.mean(all_grams, axis=0)

    if len(grams) < 2:
        print("  Not enough words with associations in vocabulary.")
        return

    avail = sorted(grams.keys())
    print(f"\n  Cosine similarity between averaged Gram matrices:\n")
    print(f"  {'':>12}", end="")
    for w in avail:
        print(f"  {w[:8]:>8}", end="")
    print()

    for w1 in avail:
        print(f"  {w1:>12}", end="")
        g1 = grams[w1].flatten()
        n1 = np.linalg.norm(g1)
        for w2 in avail:
            g2 = grams[w2].flatten()
            n2 = np.linalg.norm(g2)
            sim = float(g1 @ g2 / (n1 * n2)) if n1 > 0 and n2 > 0 else 0
            print(f"  {sim:>8.3f}", end="")
        print()


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print(f"Word Associations — Multi-seed Gram^{GRAM_POWER} ({N_SEEDS} seeds)")
    print(f"Vocabulary: {len(emb.vocab):,} words, {DIM}D SVD embeddings")
    print("=" * 70)

    demo_discriminative()
    demo_ranking()
    demo_top10()
    demo_cross_word()

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
