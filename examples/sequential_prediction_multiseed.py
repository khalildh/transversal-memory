"""
examples/sequential_prediction_multiseed.py
=============================================
Upgraded sequential prediction using multi-seed Gram^0.05 ensemble.

Key improvements over sequential_prediction.py:
  - N_SEEDS random projections instead of 1
  - Gram^0.05 eigenvalue compression (fractional power)
  - Averaged scores across all seeds

These two techniques improved pure geometry from p@10=0.011 to 0.1065
in the discriminative ranking benchmark (9.7x improvement).
"""

import pickle
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, "..")

from transversal_memory import P3Memory, GramMemory, plucker_inner
from transversal_memory.plucker import (
    random_projection, random_projection_dual,
    batch_encode_lines_dual, project_to_line_dual,
)
from transversal_memory.cooccurrence import SVDEmbeddings


# ── Config ──────────────────────────────────────────────────────────────

N_SEEDS = 50           # number of random projections
SEED_SPACING = 10      # spacing between seeds (avoids correlated projections)
GRAM_POWER = 0.05      # eigenvalue compression factor
DIM = 32


# ── Load cached data ────────────────────────────────────────────────────

with open("data/cache/associations.pkl", "rb") as f:
    assoc = pickle.load(f)
with open("data/cache/embeddings_dim32.pkl", "rb") as f:
    emb = pickle.load(f)


# ── Multi-seed Gram infrastructure ──────────────────────────────────────

SEED_LIST = [i * SEED_SPACING for i in range(N_SEEDS)]

# Precompute projection matrices
PROJECTIONS = []
for seed in SEED_LIST:
    rng = np.random.default_rng(seed)
    W1, W2 = random_projection_dual(DIM, rng)
    PROJECTIONS.append((W1, W2))


def _power_gram(gram, power):
    """Apply fractional power to Gram eigenvalues."""
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-10)
    transformed = eigvals ** power
    return eigvecs @ np.diag(transformed) @ eigvecs.T


def encode_line(src_vec, tgt_vec, W1, W2):
    """Encode a single line in G(2,4) via dual projection."""
    L = project_to_line_dual(src_vec, tgt_vec, W1, W2)
    if np.linalg.norm(L) > 1e-12:
        return L
    return None


def build_multiseed_gram(src_word, target_words):
    """
    Build per-seed Gram^0.05 matrices from (src, target) line pairs.
    Returns list of {W1, W2, gram} dicts.
    """
    src_vec = emb.src[src_word]
    per_seed = []

    for W1, W2 in PROJECTIONS:
        lines_list = []
        for t in target_words:
            if t in emb.tgt:
                L = encode_line(src_vec, emb.tgt[t], W1, W2)
                if L is not None:
                    lines_list.append(L)
        if len(lines_list) < 2:
            continue
        lines_arr = np.stack(lines_list)
        gram = lines_arr.T @ lines_arr
        gram = _power_gram(gram, GRAM_POWER)
        per_seed.append({"W1": W1, "W2": W2, "gram": gram})

    return per_seed, src_vec


def score_candidates_multiseed(src_vec, per_seed, candidate_words):
    """
    Score all candidates using multi-seed Gram^0.05 ensemble.
    Returns list of (score, word) sorted descending.
    """
    # Get candidate vectors
    cand_words = []
    cand_vecs = []
    for w in candidate_words:
        if w in emb.tgt:
            cand_words.append(w)
            cand_vecs.append(emb.tgt[w])

    if not cand_vecs or not per_seed:
        return []

    tgt_mat = np.stack(cand_vecs)
    N = len(cand_words)
    accumulated = np.zeros(N)

    for sd in per_seed:
        lines = batch_encode_lines_dual(src_vec, tgt_mat, sd["W1"], sd["W2"])
        gram_scores = np.sum((lines @ sd["gram"]) * lines, axis=1)
        accumulated += gram_scores

    accumulated /= len(per_seed)

    scored = list(zip(accumulated, cand_words))
    scored.sort(key=lambda x: -x[0])
    return scored


# ── Helpers ─────────────────────────────────────────────────────────────

def get_sequences(min_len=8, max_seqs=None):
    seqs = []
    for source, targets in assoc.items():
        if len(targets) >= min_len:
            filtered = [t for t in targets if t in emb.src and t in emb.tgt]
            if len(filtered) >= min_len:
                seqs.append((source, filtered))
                if max_seqs and len(seqs) >= max_seqs:
                    break
    return seqs


# ═════════════════════════════════════════════════════════════════════════
# TEST 1: Predict held-out associate (discriminative)
# ═════════════════════════════════════════════════════════════════════════

def predict_multiseed_gram(source, sequence, context_size=None):
    """
    Given source word and its associates [w1,...,wn],
    train on [w1,...,wn-1], predict wn.
    """
    if len(sequence) < 3:
        return None

    target = sequence[-1]
    train = sequence[:-1]
    if context_size:
        train = train[-context_size:]

    per_seed, src_vec = build_multiseed_gram(source, train)
    if not per_seed:
        return None

    # Score all vocab
    candidates = [w for w in emb.vocab if w != source and w not in train]
    scored = score_candidates_multiseed(src_vec, per_seed, candidates)

    # Find target rank
    target_rank = None
    for i, (s, w) in enumerate(scored):
        if w == target:
            target_rank = i + 1
            break

    return {
        "target": target,
        "target_rank": target_rank,
        "n_candidates": len(scored),
        "top": scored[:10],
        "context": train,
    }


# ═════════════════════════════════════════════════════════════════════════
# TEST 2: Autoregressive generation
# ═════════════════════════════════════════════════════════════════════════

def generate_sequence_multiseed(source, seed_words, n_steps=10):
    """Generate next words using multi-seed Gram^0.05."""
    sequence = list(seed_words)

    for step in range(n_steps):
        # Build Gram from all words so far
        per_seed, src_vec = build_multiseed_gram(source, sequence)
        if not per_seed:
            break

        # Score candidates
        candidates = [w for w in emb.vocab if w != source and w not in sequence]
        scored = score_candidates_multiseed(src_vec, per_seed, candidates)

        if not scored:
            break
        sequence.append(scored[0][1])  # best word

    return sequence


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(f"Sequential Prediction — Multi-seed Gram^{GRAM_POWER} ({N_SEEDS} seeds)")
    print("=" * 70)

    # ── Test 1: Predict held-out last associate ─────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: Predict last associate from context (multi-seed Gram)")
    print("=" * 70)

    sequences = get_sequences(min_len=10, max_seqs=200)
    print(f"  Using {len(sequences)} sequences with >= 10 words")

    ranks = []
    top10_hits = 0
    top100_hits = 0
    t0 = time.time()

    for idx, (source, seq) in enumerate(sequences[:200]):
        result = predict_multiseed_gram(source, seq)
        if result and result["target_rank"]:
            ranks.append(result["target_rank"])
            if result["target_rank"] <= 10:
                top10_hits += 1
            if result["target_rank"] <= 100:
                top100_hits += 1

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/200] median_rank={int(np.median(ranks)) if ranks else 'N/A'} "
                  f"top10={top10_hits} top100={top100_hits} ({elapsed:.0f}s)")

    elapsed = time.time() - t0

    if ranks:
        print(f"\n  Evaluated: {len(ranks)} sequences ({elapsed:.1f}s)")
        print(f"  Median target rank: {int(np.median(ranks))} / ~67K")
        print(f"  Mean target rank:   {int(np.mean(ranks))} / ~67K")
        print(f"  Top-10 hits:  {top10_hits}/{len(ranks)} "
              f"({100*top10_hits/len(ranks):.1f}%)")
        print(f"  Top-100 hits: {top100_hits}/{len(ranks)} "
              f"({100*top100_hits/len(ranks):.1f}%)")
        print(f"  (Single-seed baseline: median ~21683, top-10=0%, top-100=0%)")
        print(f"  (Random baseline: median ~33K)")

    # Show examples
    print("\n  Examples:")
    for source, seq in sequences[:10]:
        result = predict_multiseed_gram(source, seq)
        if result:
            ctx = result["context"][-4:]
            target = result["target"]
            rank = result["target_rank"]
            top3 = [w for _, w in result["top"][:3]]
            hit = "✓" if rank and rank <= 100 else " "
            print(f"    [{source}] ...{' → '.join(ctx)} → [{target}]  "
                  f"rank={rank}  top3={top3}  {hit}")

    # ── Test 2: Autoregressive generation ───────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Autoregressive Generation (multi-seed Gram)")
    print("=" * 70)

    for source in ["king", "ocean", "fire", "love", "brain"]:
        if source not in assoc:
            continue
        actual = assoc[source][:12]
        seed = actual[:3]

        # Check seed words exist
        if not all(w in emb.src and w in emb.tgt for w in seed):
            continue

        t0 = time.time()
        generated = generate_sequence_multiseed(source, seed, n_steps=9)
        elapsed = time.time() - t0
        gen_words = generated[len(seed):]

        known = set(assoc[source])
        gen_known = [w for w in gen_words if w in known]

        print(f"\n  {source}:")
        print(f"    Actual:    {' → '.join(actual)}")
        print(f"    Seed:      {' → '.join(seed)}")
        print(f"    Generated: {' → '.join(gen_words)}  ({elapsed:.1f}s)")
        print(f"    Known associates: {len(gen_known)}/{len(gen_words)} {gen_known}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
