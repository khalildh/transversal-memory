"""
examples/full_dataset_demo.py
==============================
Full pipeline on the Overmann WordAssociations dataset (65K source words).

Loads the CSV, builds PPMI + SVD embeddings from the association norms
themselves (no GloVe), then demonstrates both retrieval modes.

Uses scipy sparse SVD (memory-efficient) and pickle checkpointing.
"""

import csv
import os
import pickle
import sys
import time
import numpy as np

sys.path.insert(0, "..")

from transversal_memory import P3Memory, GramMemory
from transversal_memory.cooccurrence import (
    CooccurrenceMatrix,
    SVDEmbeddings,
)
from transversal_memory.plucker import random_projection


# ── Config ───────────────────────────────────────────────────────────────

CSV_PATH = "data/WordAssociations/WordAssociations.csv"
CACHE_DIR = "data/cache"
EMBED_DIM = 32          # SVD embedding dimension
MIN_ASSOCIATES = 5      # skip source words with fewer associates


# ── Load ─────────────────────────────────────────────────────────────────

def load_associations(path):
    """Load the CSV into {source: [target, ...]}."""
    cache = os.path.join(CACHE_DIR, "associations.pkl")
    if os.path.exists(cache):
        print(f"  Loading cached associations from {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    associations = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            source = row[0].strip().lower()
            targets = [t.strip().lower() for t in row[1:] if t.strip()]
            if len(targets) >= MIN_ASSOCIATES:
                associations[source] = targets

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(associations, f)
    print(f"  Cached associations to {cache}")
    return associations


def build_embeddings(assoc):
    """Build or load cached SVD embeddings."""
    cache = os.path.join(CACHE_DIR, f"embeddings_dim{EMBED_DIM}.pkl")
    if os.path.exists(cache):
        print(f"  Loading cached embeddings from {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("  Building co-occurrence matrix (sparse)...")
    t0 = time.time()
    co = CooccurrenceMatrix()
    co.add_many(assoc, position_decay=True)
    C = co.build(weighting="ppmi")
    vocab_size = len(co.vocab)

    from scipy.sparse import issparse
    if issparse(C):
        nnz = C.nnz
        density = nnz / (vocab_size * vocab_size)
        print(f"  PPMI matrix: {vocab_size:,} × {vocab_size:,}  "
              f"nnz={nnz:,}  density={density:.5f}  ({time.time()-t0:.1f}s)")
    else:
        density = np.count_nonzero(C) / C.size
        print(f"  PPMI matrix: {vocab_size:,} × {vocab_size:,}  "
              f"density={density:.5f}  ({time.time()-t0:.1f}s)")

    print(f"  Running truncated SVD (dim={EMBED_DIM})...")
    t0 = time.time()
    emb = co.svd_embeddings(dim=EMBED_DIM, role="both")
    print(f"  SVD done ({time.time()-t0:.1f}s)")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(emb, f)
    print(f"  Cached embeddings to {cache}")
    return emb


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Transversal Memory — Full Overmann Dataset")
    print("=" * 70)

    # ── Step 1: Load ─────────────────────────────────────────────────
    print("\nStep 1: Load associations")
    t0 = time.time()
    assoc = load_associations(CSV_PATH)
    n_pairs = sum(len(v) for v in assoc.values())
    print(f"  {len(assoc):,} source words, {n_pairs:,} associations "
          f"({time.time()-t0:.1f}s)")

    # ── Step 2: Build embeddings ─────────────────────────────────────
    print("\nStep 2: Build SVD embeddings")
    t0 = time.time()
    emb = build_embeddings(assoc)
    ve = emb.variance_explained()
    er = emb.effective_rank()
    print(f"  dim={EMBED_DIM}, vocab={len(emb.vocab):,}  ({time.time()-t0:.1f}s)")
    print(f"  Top singular values: {emb.singular_values[:8].round(1)}")
    print(f"  Variance explained (first 8): {ve[:8].round(3)}")
    print(f"  Effective rank (95%): {er}")

    W = random_projection(EMBED_DIM, np.random.default_rng(42))

    # ── Step 3: Nearest neighbours sanity check ──────────────────────
    print("\n" + "=" * 70)
    print("Nearest Neighbours (sanity check)")
    print("=" * 70)

    test_words = ["dog", "music", "ocean", "computer", "love",
                  "doctor", "mountain", "king"]
    for word in test_words:
        if word not in emb.src:
            continue
        nn_src = emb.nearest(word, k=5, space="source")
        nn_tgt = emb.nearest(word, k=5, space="target")
        src_str = ", ".join(f"{w}({s:.2f})" for s, w in nn_src)
        tgt_str = ", ".join(f"{w}({s:.2f})" for s, w in nn_tgt)
        print(f"\n  {word}")
        print(f"    source (cue):   {src_str}")
        print(f"    target (assoc): {tgt_str}")

    # ── Step 4: GramMemory — discriminative scoring ──────────────────
    print("\n" + "=" * 70)
    print("GramMemory — Discriminative Scoring")
    print("=" * 70)

    test_sources = ["dog", "music", "ocean", "love", "king",
                    "fire", "brain", "guitar"]
    for source in test_sources:
        if source not in assoc or source not in emb.src:
            continue

        targets = assoc[source]
        gram = GramMemory()

        split = max(3, int(len(targets) * 0.75))
        train_targets = targets[:split]
        held_out = targets[split:]

        for tgt in train_targets:
            line = emb.make_line(source, tgt, W)
            if line is not None:
                gram.store_line(line)

        held_scores = []
        for tgt in held_out:
            line = emb.make_line(source, tgt, W)
            if line is not None:
                held_scores.append(gram.score(line))

        rng = np.random.default_rng(0)
        non_assoc_words = [w for w in emb.vocab
                          if w not in set(targets) and w != source]
        sample = rng.choice(len(non_assoc_words),
                           size=min(200, len(non_assoc_words)),
                           replace=False)
        random_scores = []
        for idx in sample:
            line = emb.make_line(source, non_assoc_words[idx], W)
            if line is not None:
                random_scores.append(gram.score(line))

        if held_scores and random_scores:
            h_mean = np.mean(held_scores)
            r_mean = np.mean(random_scores)
            sep = "✓" if h_mean > r_mean else "✗"
            print(f"\n  {source:15s}  train={len(train_targets):2d}  "
                  f"held-out={len(held_out):2d}")
            print(f"    held-out mean:  {h_mean:.4f}")
            print(f"    non-assoc mean: {r_mean:.4f}  {sep}")

            evals = gram.eigenvalues()
            print(f"    eigenvalues: {evals[:6].round(3)}")

            axes = gram.principal_axes(k=3)
            for ax_i in range(min(3, len(axes))):
                ax = axes[ax_i]
                ax_scores = []
                for tgt in train_targets:
                    line = emb.make_line(source, tgt, W)
                    if line is not None:
                        ax_scores.append((abs(float(line @ ax)), tgt))
                ax_scores.sort(key=lambda x: -x[0])
                top = ", ".join(w for _, w in ax_scores[:6])
                print(f"    axis {ax_i+1}: {top}")

    # ── Step 5: Cross-word similarity ────────────────────────────────
    print("\n" + "=" * 70)
    print("Gram Matrix Similarity — Relational Pattern Comparison")
    print("=" * 70)

    compare_groups = [
        ("Animals",  ["dog", "cat", "horse", "bird"]),
        ("Music",    ["guitar", "piano", "drum", "violin"]),
        ("Emotions", ["anger", "fear", "sadness", "joy"]),
        ("Body",     ["brain", "heart", "hand", "eye"]),
    ]

    for group_name, words in compare_groups:
        grams = {}
        for w in words:
            if w not in assoc or w not in emb.src:
                continue
            g = GramMemory()
            for tgt in assoc[w]:
                line = emb.make_line(w, tgt, W)
                if line is not None:
                    g.store_line(line)
            if g.n_lines > 0:
                grams[w] = g

        if len(grams) < 2:
            continue

        print(f"\n  {group_name}:")
        gwords = list(grams.keys())
        header = "".join(f"{w:>12s}" for w in gwords)
        print(f"  {'':12s}{header}")
        for w1 in gwords:
            row = ""
            for w2 in gwords:
                sim = grams[w1].compare(grams[w2])
                row += f"{sim:12.3f}"
            print(f"  {w1:12s}{row}")

    # ── Step 6: P3Memory — generative retrieval ──────────────────────
    print("\n" + "=" * 70)
    print("P3Memory — Generative Retrieval (Dual Projection)")
    print("=" * 70)
    print("  Using dual projection: line(W1·[src;tgt], W2·[src;tgt])")
    print("  Both endpoints depend on both source and target,")
    print("  breaking the co-punctal degeneracy of single-projection encoding.")

    from transversal_memory.plucker import plucker_inner, random_projection_dual
    W1_dual, W2_dual = random_projection_dual(EMBED_DIM, np.random.default_rng(99))

    analogy_tests = [
        ("dog",   ["puppy", "bark", "fetch"],     "bone"),
        ("music", ["rhythm", "melody", "harmony"], "instrument"),
        ("ocean", ["waves", "deep", "salt"],       "fish"),
        ("king",  ["crown", "throne", "royal"],    "queen"),
        ("love",  ["heart", "romance", "passion"], "marriage"),
        ("fire",  ["flame", "heat", "burn"],       "smoke"),
        ("tree",  ["leaves", "branches", "roots"], "forest"),
        ("brain", ["neurons", "memory", "cortex"], "intelligence"),
    ]

    for source, stored_targets, query_target in analogy_tests:
        if source not in emb.src:
            continue

        stored_lines = []
        for tgt in stored_targets:
            line = emb.make_line_dual(source, tgt, W1_dual, W2_dual)
            if line is not None:
                stored_lines.append(line)

        if len(stored_lines) < 3:
            continue

        q_line = emb.make_line_dual(source, query_target, W1_dual, W2_dual)
        if q_line is None:
            continue

        mem = P3Memory()
        mem.store(stored_lines[:3])
        transversals = mem.query_generative(q_line)

        if not transversals:
            print(f"\n  {source}: no transversals found")
            continue

        T, resid = transversals[0]

        # Decode by Plücker inner product: lower |pi| = more incident = better
        results = []
        known = set(assoc.get(source, []))
        for word in emb.vocab:
            if word == source:
                continue
            line = emb.make_line_dual(source, word, W1_dual, W2_dual)
            if line is None:
                continue
            pi = abs(plucker_inner(T, line))
            results.append((pi, word))

        results.sort(key=lambda x: x[0])  # ascending: lower |pi| = better
        top = results[:10]

        # Find target rank
        target_rank = next(
            (i+1 for i, (_, w) in enumerate(results) if w == query_target),
            None)

        stored_str = ", ".join(stored_targets)
        n_known = sum(1 for _, w in top if w in known)
        print(f"\n  {source} + [{stored_str}] → query: {query_target}")
        print(f"    Plücker residual: {resid:.2e}  "
              f"Target rank: {target_rank}/{len(results)}  "
              f"Known in top 10: {n_known}")
        for pi_val, word in top:
            marker = "✓" if word in known else " "
            print(f"      {word:20s} |pi|={pi_val:.2e}  {marker}")

    # ── Step 7: Held-out prediction across many words ────────────────
    print("\n" + "=" * 70)
    print("Held-out Prediction — Batch Evaluation")
    print("=" * 70)

    rng = np.random.default_rng(123)
    eligible = [w for w in assoc
                if len(assoc[w]) >= 10 and w in emb.src]
    sample_sources = rng.choice(eligible,
                                size=min(500, len(eligible)),
                                replace=False)

    n_separated = 0
    n_tested = 0
    margins = []

    for source in sample_sources:
        targets = assoc[source]
        split = max(3, int(len(targets) * 0.75))
        train_targets = targets[:split]
        held_out = targets[split:]

        gram = GramMemory()
        for tgt in train_targets:
            line = emb.make_line(source, tgt, W)
            if line is not None:
                gram.store_line(line)

        if gram.n_lines < 3:
            continue

        h_scores = []
        for tgt in held_out:
            line = emb.make_line(source, tgt, W)
            if line is not None:
                h_scores.append(gram.score(line))

        non_assoc = [w for w in emb.vocab
                     if w not in set(targets) and w != source]
        idx = rng.choice(len(non_assoc),
                        size=min(100, len(non_assoc)),
                        replace=False)
        r_scores = []
        for i in idx:
            line = emb.make_line(source, non_assoc[i], W)
            if line is not None:
                r_scores.append(gram.score(line))

        if h_scores and r_scores:
            h_mean = np.mean(h_scores)
            r_mean = np.mean(r_scores)
            n_tested += 1
            if h_mean > r_mean:
                n_separated += 1
                margins.append(h_mean - r_mean)

    print(f"\n  Tested: {n_tested} source words")
    print(f"  Separated (held-out > non-assoc): {n_separated}/{n_tested} "
          f"({100*n_separated/n_tested:.1f}%)")
    if margins:
        print(f"  Mean margin (when separated): {np.mean(margins):.4f}")
        print(f"  Median margin: {np.median(margins):.4f}")
    print(f"\n  (random baseline = 50%)")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
