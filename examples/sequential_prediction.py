"""
examples/sequential_prediction.py
==================================
Can transversal memory predict the next word in a sequence?

Approach: encode each bigram (wᵢ → wᵢ₊₁) as a Plücker line.
Use the geometric structure to predict what comes next.

Two modes:
  1. GramMemory: accumulate transition lines, score candidates
  2. P3Memory: store last 3 transitions, decode transversal

Uses the Overmann associations as ground truth sequences —
each source word's associate list is treated as an ordered sequence.
"""

import pickle
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, "..")

from transversal_memory import P3Memory, GramMemory, plucker_inner
from transversal_memory.plucker import random_projection, random_projection_dual
from transversal_memory.cooccurrence import SVDEmbeddings


# ── Load cached data ─────────────────────────────────────────────────────

with open("data/cache/associations.pkl", "rb") as f:
    assoc = pickle.load(f)
with open("data/cache/embeddings_dim32.pkl", "rb") as f:
    emb = pickle.load(f)

DIM = 32
W = random_projection(DIM, np.random.default_rng(42))
W1_dual, W2_dual = random_projection_dual(DIM, np.random.default_rng(99))


# ── Helpers ──────────────────────────────────────────────────────────────

def make_line_single(w1, w2):
    """Single projection — for GramMemory (discriminative)."""
    return emb.make_line(w1, w2, W)

def make_line_dual(w1, w2):
    """Dual projection — for P3Memory (generative)."""
    return emb.make_line_dual(w1, w2, W1_dual, W2_dual)


def get_sequences(min_len=8, max_seqs=None):
    """
    Extract sequences from association data.
    Each source word's associate list is an ordered sequence.
    """
    seqs = []
    for source, targets in assoc.items():
        if len(targets) >= min_len:
            # Filter to words in vocabulary
            filtered = [t for t in targets if t in emb.src and t in emb.tgt]
            if len(filtered) >= min_len:
                seqs.append(filtered)
                if max_seqs and len(seqs) >= max_seqs:
                    break
    return seqs


# ═══════════════════════════════════════════════════════════════════════════
# MODE 1: GramMemory — accumulate transition pattern, score candidates
# ═══════════════════════════════════════════════════════════════════════════

def predict_gram(sequence, context_size=None):
    """
    Given a sequence [w1, w2, ..., wn], predict wn from [w1, ..., wn-1].

    Encodes each consecutive bigram (wi, wi+1) as a Plücker line,
    accumulates them into a GramMemory, then scores all candidates
    for the final transition.

    context_size: if set, only use the last N transitions.
    """
    if len(sequence) < 3:
        return None

    target = sequence[-1]
    context = sequence[:-1]

    if context_size:
        context = context[-context_size:]

    gram = GramMemory()
    for i in range(len(context) - 1):
        line = make_line_single(context[i], context[i+1])
        if line is not None:
            gram.store_line(line)

    if gram.n_lines < 2:
        return None

    # Score candidates: what word best continues from the last context word?
    last_word = context[-1]
    scores = []
    for word in emb.vocab:
        if word == last_word:
            continue
        line = make_line_single(last_word, word)
        if line is not None:
            scores.append((gram.score(line), word))

    scores.sort(key=lambda x: -x[0])

    # Find target rank
    target_rank = None
    for i, (s, w) in enumerate(scores):
        if w == target:
            target_rank = i + 1
            break

    return {
        "target": target,
        "target_rank": target_rank,
        "n_candidates": len(scores),
        "top": scores[:10],
        "context": context,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODE 2: P3Memory — store last 3 transitions, decode transversal
# ═══════════════════════════════════════════════════════════════════════════

def predict_p3(sequence):
    """
    Given a sequence [w1, w2, ..., wn], use the last 4 transitions
    to predict wn.

    Store transitions (w_{n-4}→w_{n-3}), (w_{n-3}→w_{n-2}), (w_{n-2}→w_{n-1}).
    The transversal should point toward the pattern that continues to wn.
    Decode by finding which line(w_{n-1}, candidate) is most incident with T.
    """
    if len(sequence) < 5:
        return None

    target = sequence[-1]
    # Last 5 words give us 4 transitions; store 3, query with last context word
    tail = sequence[-5:]

    stored_lines = []
    for i in range(3):
        line = make_line_dual(tail[i], tail[i+1])
        if line is not None:
            stored_lines.append(line)

    if len(stored_lines) < 3:
        return None

    # Query: transition from second-to-last to last word
    q_line = make_line_dual(tail[3], target)
    if q_line is None:
        return None

    mem = P3Memory()
    mem.store(stored_lines[:3])
    transversals = mem.query_generative(q_line)

    if not transversals:
        return None

    T, resid = transversals[0]

    # Decode: which line(tail[3], candidate) is most incident with T?
    last_word = tail[3]
    results = []
    for word in emb.vocab:
        if word == last_word:
            continue
        line = make_line_dual(last_word, word)
        if line is None:
            continue
        pi = abs(plucker_inner(T, line))
        results.append((pi, word))

    results.sort(key=lambda x: x[0])  # ascending: lower |pi| = better

    target_rank = None
    for i, (s, w) in enumerate(results):
        if w == target:
            target_rank = i + 1
            break

    return {
        "target": target,
        "target_rank": target_rank,
        "n_candidates": len(results),
        "top": results[:10],
        "context": tail[:-1],
        "residual": resid,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODE 3: Autoregressive generation — keep predicting
# ═══════════════════════════════════════════════════════════════════════════

def generate_sequence(seed_words, n_steps=10, mode="gram", context_size=5):
    """
    Given seed words, autoregressively generate the next n_steps words.

    mode: "gram" uses GramMemory, "p3" uses P3Memory
    """
    sequence = list(seed_words)

    for step in range(n_steps):
        if mode == "gram":
            gram = GramMemory()
            ctx = sequence[-context_size:] if context_size else sequence
            for i in range(len(ctx) - 1):
                line = make_line_single(ctx[i], ctx[i+1])
                if line is not None:
                    gram.store_line(line)

            if gram.n_lines < 1:
                break

            last_word = sequence[-1]
            best_score = -1
            best_word = None
            for word in emb.vocab:
                if word == last_word or word in sequence:
                    continue
                line = make_line_single(last_word, word)
                if line is not None:
                    score = gram.score(line)
                    if score > best_score:
                        best_score = score
                        best_word = word

            if best_word is None:
                break
            sequence.append(best_word)

        elif mode == "p3":
            if len(sequence) < 4:
                break

            tail = sequence[-4:]
            stored_lines = []
            for i in range(3):
                line = make_line_dual(tail[i], tail[i+1])
                if line is not None:
                    stored_lines.append(line)

            if len(stored_lines) < 3:
                break

            mem = P3Memory()
            mem.store(stored_lines[:3])

            # Try a dummy query to get the transversal structure
            # Use a random line as query to get the transversal
            last_word = sequence[-1]
            best_pi = float('inf')
            best_word = None

            # Get transversal from a self-referential query
            # (store 3 transitions, the transversal encodes the pattern)
            # Actually: we need a 4th line to query with.
            # Use the first transition again as a "reference"
            ref_line = stored_lines[0]
            transversals = mem.query_generative(ref_line)

            if not transversals:
                break

            T, _ = transversals[0]

            for word in emb.vocab:
                if word == last_word or word in sequence:
                    continue
                line = make_line_dual(last_word, word)
                if line is None:
                    continue
                pi = abs(plucker_inner(T, line))
                if pi < best_pi:
                    best_pi = pi
                    best_word = word

            if best_word is None:
                break
            sequence.append(best_word)

    return sequence


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Sequential Prediction with Transversal Memory")
    print("=" * 70)

    # ── Test 1: Predict held-out last word ────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: Predict last word from context (GramMemory)")
    print("=" * 70)
    print("  Sequence = a source word's associate list (ordered)")
    print("  Context = first N-1 associates, target = last associate")

    sequences = get_sequences(min_len=10, max_seqs=200)
    print(f"  Using {len(sequences)} sequences with >= 10 words")

    # Evaluate GramMemory prediction
    ranks = []
    top10_hits = 0
    top100_hits = 0

    for seq in sequences[:200]:
        result = predict_gram(seq, context_size=6)
        if result and result["target_rank"]:
            ranks.append(result["target_rank"])
            if result["target_rank"] <= 10:
                top10_hits += 1
            if result["target_rank"] <= 100:
                top100_hits += 1

    if ranks:
        print(f"\n  Evaluated: {len(ranks)} sequences")
        print(f"  Median target rank: {int(np.median(ranks))} / ~67K")
        print(f"  Mean target rank:   {int(np.mean(ranks))} / ~67K")
        print(f"  Top-10 hits:  {top10_hits}/{len(ranks)} "
              f"({100*top10_hits/len(ranks):.1f}%)")
        print(f"  Top-100 hits: {top100_hits}/{len(ranks)} "
              f"({100*top100_hits/len(ranks):.1f}%)")
        print(f"  (random baseline: median rank ~33K, top-10 = 0.015%)")

    # Show some examples
    print("\n  Examples:")
    for seq in sequences[:10]:
        result = predict_gram(seq, context_size=6)
        if result:
            ctx = result["context"][-4:]
            target = result["target"]
            rank = result["target_rank"]
            top3 = [w for _, w in result["top"][:3]]
            hit = "✓" if rank and rank <= 100 else " "
            print(f"    ...{' → '.join(ctx)} → [{target}]  "
                  f"rank={rank}  top3={top3}  {hit}")

    # ── Test 2: P3Memory generative prediction ────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Predict last word from context (P3Memory)")
    print("=" * 70)

    p3_ranks = []
    p3_top10 = 0
    p3_top100 = 0
    p3_fail = 0

    for seq in sequences[:100]:
        result = predict_p3(seq)
        if result is None:
            p3_fail += 1
            continue
        if result["target_rank"]:
            p3_ranks.append(result["target_rank"])
            if result["target_rank"] <= 10:
                p3_top10 += 1
            if result["target_rank"] <= 100:
                p3_top100 += 1

    if p3_ranks:
        print(f"\n  Evaluated: {len(p3_ranks)} sequences "
              f"({p3_fail} failed to find transversals)")
        print(f"  Median target rank: {int(np.median(p3_ranks))} / ~67K")
        print(f"  Mean target rank:   {int(np.mean(p3_ranks))} / ~67K")
        print(f"  Top-10 hits:  {p3_top10}/{len(p3_ranks)} "
              f"({100*p3_top10/len(p3_ranks):.1f}%)")
        print(f"  Top-100 hits: {p3_top100}/{len(p3_ranks)} "
              f"({100*p3_top100/len(p3_ranks):.1f}%)")

    # Show examples
    print("\n  Examples:")
    for seq in sequences[:15]:
        result = predict_p3(seq)
        if result:
            ctx = result["context"][-4:]
            target = result["target"]
            rank = result["target_rank"]
            top3 = [w for _, w in result["top"][:3]]
            hit = "✓" if rank and rank <= 100 else " "
            print(f"    ...{' → '.join(ctx)} → [{target}]  "
                  f"rank={rank}  top3={top3}  {hit}")

    # ── Test 3: Autoregressive generation ─────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: Autoregressive Generation (GramMemory)")
    print("=" * 70)
    print("  Seed with 3-4 words, generate 10 more.")

    seeds = [
        ["king", "crown", "throne"],
        ["ocean", "waves", "deep"],
        ["fire", "flame", "heat"],
        ["love", "heart", "romance"],
        ["dog", "puppy", "bark"],
        ["brain", "neurons", "memory"],
        ["music", "rhythm", "melody"],
        ["tree", "leaves", "branches"],
    ]

    for seed in seeds:
        # Check all seed words exist
        if not all(w in emb.src and w in emb.tgt for w in seed):
            continue
        t0 = time.time()
        generated = generate_sequence(seed, n_steps=10, mode="gram",
                                      context_size=5)
        elapsed = time.time() - t0
        new_words = generated[len(seed):]
        print(f"\n  Seed: {' → '.join(seed)}")
        print(f"  Generated: {' → '.join(new_words)}  ({elapsed:.1f}s)")

        # Check how many are known associates of the first seed word
        known = set(assoc.get(seed[0], []))
        n_known = sum(1 for w in new_words if w in known)
        print(f"  Known associates of '{seed[0]}': {n_known}/{len(new_words)}")

    # ── Test 4: Compare with actual associate lists ───────────────────
    print("\n" + "=" * 70)
    print("TEST 4: Generated vs Actual Associate Lists")
    print("=" * 70)

    for source in ["king", "ocean", "fire", "love", "brain"]:
        if source not in assoc:
            continue
        actual = assoc[source][:12]
        seed = actual[:3]
        generated = generate_sequence(seed, n_steps=9, mode="gram",
                                      context_size=5)
        gen_words = generated[len(seed):]

        known = set(assoc[source])
        gen_known = [w for w in gen_words if w in known]

        print(f"\n  {source}:")
        print(f"    Actual:    {' → '.join(actual)}")
        print(f"    Seed:      {' → '.join(seed)}")
        print(f"    Generated: {' → '.join(gen_words)}")
        print(f"    Known associates in generated: "
              f"{len(gen_known)}/{len(gen_words)} {gen_known}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
