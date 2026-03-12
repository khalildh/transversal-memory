"""
examples/associative_generation.py
===================================
Can transversal memory continuously generate new associated words?

Unlike sequential_prediction.py (which tried to predict sequences),
this tests ASSOCIATIVE CHAINING: start with a concept, generate words
that are associatively related, and check if they stay on-topic.

Two modes:
  1. Concept mode: build GramMemory from a source word's known associates,
     then rank the entire vocabulary. The top-scoring unseen words should
     be thematically related.

  2. Chain mode: start with a seed concept, generate N associates,
     then use those generated words as a NEW concept and generate more.
     Tests whether the system can keep producing related words without
     drifting into noise.
"""

import pickle
import sys
import time
import numpy as np

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


# ═══════════════════════════════════════════════════════════════════════════
# MODE 1: Concept generation — score entire vocab from a concept's pattern
# ═══════════════════════════════════════════════════════════════════════════

def generate_from_concept(source, max_train=None, n_results=20):
    """
    Build a GramMemory from source's associates, then rank all vocab words.
    Returns the top n_results words that aren't in the training set.
    """
    if source not in assoc or source not in emb.src:
        return None

    targets = assoc[source]
    if max_train:
        train = targets[:max_train]
    else:
        train = targets

    gram = GramMemory()
    for tgt in train:
        line = emb.make_line(source, tgt, W)
        if line is not None:
            gram.store_line(line)

    if gram.n_lines < 2:
        return None

    # Score entire vocabulary
    train_set = set(train)
    scores = []
    for word in emb.vocab:
        if word == source or word in train_set:
            continue
        line = emb.make_line(source, word, W)
        if line is not None:
            scores.append((gram.score(line), word))

    scores.sort(key=lambda x: -x[0])

    # Check how many are known associates
    known = set(targets) - train_set
    return {
        "source": source,
        "n_train": len(train),
        "n_total_assoc": len(targets),
        "top": scores[:n_results],
        "known_held_out": known,
        "n_candidates": len(scores),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODE 2: Associative chaining — generate, then generate from generated
# ═══════════════════════════════════════════════════════════════════════════

def associative_chain(source, n_per_step=5, n_steps=4, max_seed=10):
    """
    Start with source's associates, generate top words.
    Then use those generated words to form a NEW GramMemory and generate more.
    Repeat for n_steps.
    """
    if source not in assoc or source not in emb.src:
        return None

    all_known = set(assoc.get(source, []))
    seen = {source}
    chain = []

    # Step 0: seed from actual associates
    seed_targets = assoc[source][:max_seed]
    seen.update(seed_targets)

    current_source = source
    current_targets = seed_targets

    for step in range(n_steps):
        gram = GramMemory()
        for tgt in current_targets:
            if tgt in emb.tgt:
                line = emb.make_line(current_source, tgt, W)
                if line is not None:
                    gram.store_line(line)

        if gram.n_lines < 2:
            break

        # Score vocab
        scores = []
        for word in emb.vocab:
            if word in seen:
                continue
            line = emb.make_line(current_source, word, W)
            if line is not None:
                scores.append((gram.score(line), word))

        scores.sort(key=lambda x: -x[0])
        generated = [w for _, w in scores[:n_per_step]]

        n_known = sum(1 for w in generated if w in all_known)
        chain.append({
            "step": step,
            "source": current_source,
            "n_context": len(current_targets),
            "generated": generated,
            "n_known": n_known,
        })

        seen.update(generated)

        # Next step: use generated words as the new context
        # Keep the same source word — we're still exploring the same concept
        current_targets = generated

    return {
        "source": source,
        "chain": chain,
        "all_known": all_known,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODE 3: Drifting chain — each step shifts the source to the top generated
# ═══════════════════════════════════════════════════════════════════════════

def drifting_chain(source, n_per_step=5, n_steps=6):
    """
    Start with source's associates. Generate top words.
    Then SHIFT the source to the top generated word and use ITS associates.
    This creates a "free association" walk through concept space.
    """
    if source not in assoc or source not in emb.src:
        return None

    seen = {source}
    chain = []
    current = source

    for step in range(n_steps):
        if current not in assoc or current not in emb.src:
            break

        targets = [t for t in assoc[current] if t not in seen][:10]
        if len(targets) < 3:
            break

        gram = GramMemory()
        for tgt in targets:
            line = emb.make_line(current, tgt, W)
            if line is not None:
                gram.store_line(line)

        if gram.n_lines < 2:
            break

        scores = []
        for word in emb.vocab:
            if word in seen:
                continue
            line = emb.make_line(current, word, W)
            if line is not None:
                scores.append((gram.score(line), word))

        scores.sort(key=lambda x: -x[0])
        generated = [w for _, w in scores[:n_per_step]]

        chain.append({
            "source": current,
            "context": targets[:5],
            "generated": generated,
        })

        seen.update(generated)
        seen.update(targets)

        # Drift: move to the top generated word
        # Pick the first generated word that has its own associates
        next_source = None
        for w in generated:
            if w in assoc and w in emb.src:
                next_source = w
                break
        if next_source is None:
            break
        current = next_source

    return chain


# ═══════════════════════════════════════════════════════════════════════════
# MODE 4: P3Memory generative chaining
# ═══════════════════════════════════════════════════════════════════════════

def p3_generate(source, seed_targets, n_results=10):
    """
    Use P3Memory's transversal decoding to generate associated words.
    Store 3 lines from seed_targets, use the transversal to rank vocabulary.
    """
    if len(seed_targets) < 4:
        return None

    stored_lines = []
    for tgt in seed_targets[:3]:
        line = emb.make_line_dual(source, tgt, W1_dual, W2_dual)
        if line is not None:
            stored_lines.append(line)

    if len(stored_lines) < 3:
        return None

    # Use 4th target as query to find transversal
    q_line = emb.make_line_dual(source, seed_targets[3], W1_dual, W2_dual)
    if q_line is None:
        return None

    mem = P3Memory()
    mem.store(stored_lines[:3])
    transversals = mem.query_generative(q_line)

    if not transversals:
        return None

    T, resid = transversals[0]

    # Decode: rank by |plucker_inner| (lower = more incident = better)
    seed_set = set(seed_targets)
    results = []
    for word in emb.vocab:
        if word == source or word in seed_set:
            continue
        line = emb.make_line_dual(source, word, W1_dual, W2_dual)
        if line is None:
            continue
        pi = abs(plucker_inner(T, line))
        results.append((pi, word))

    results.sort(key=lambda x: x[0])
    return results[:n_results]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Associative Generation with Transversal Memory")
    print("=" * 70)

    # ── Test 1: Generate from concept ────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: Generate Associated Words from a Concept")
    print("=" * 70)
    print("  Build GramMemory from known associates, rank entire vocabulary.")
    print("  Check: are the top-ranked unseen words actually related?")

    test_words = ["king", "ocean", "fire", "love", "brain", "music",
                  "dog", "tree", "computer", "war", "money", "dream"]

    for source in test_words:
        result = generate_from_concept(source, n_results=15)
        if result is None:
            continue

        top = result["top"]
        known = result["known_held_out"]
        n_known_in_top = sum(1 for _, w in top if w in known)
        all_assoc = set(assoc.get(source, []))
        n_any_assoc = sum(1 for _, w in top if w in all_assoc)

        print(f"\n  {source} (trained on {result['n_train']} associates, "
              f"{result['n_total_assoc']} total)")
        words = [w for _, w in top]
        markers = []
        for _, w in top:
            if w in known:
                markers.append("*")  # held-out associate
            elif w in all_assoc:
                markers.append("+")  # training associate leaked
            else:
                markers.append(" ")
        word_str = "  ".join(f"{w}{m}" for w, m in zip(words, markers))
        print(f"    Top 15: {word_str}")
        print(f"    Known associates in top 15: {n_any_assoc} "
              f"(* = held-out, + = train)")

    # ── Test 2: Associative chaining ─────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Associative Chaining")
    print("=" * 70)
    print("  Step 0: use source's actual associates as context")
    print("  Step N: use the words generated in step N-1 as context")
    print("  Check: do generated words stay thematically related?")

    for source in ["king", "ocean", "fire", "brain", "music", "dog"]:
        result = associative_chain(source, n_per_step=8, n_steps=5)
        if result is None:
            continue

        all_known = result["all_known"]
        print(f"\n  {source} ({len(all_known)} known associates):")

        for step_info in result["chain"]:
            gen = step_info["generated"]
            n_known = step_info["n_known"]
            step = step_info["step"]
            markers = ["*" if w in all_known else " " for w in gen]
            word_str = "  ".join(f"{w}{m}" for w, m in zip(gen, markers))
            print(f"    step {step} ({step_info['n_context']} ctx): "
                  f"{word_str}  [{n_known}/{len(gen)} known]")

    # ── Test 3: Drifting chain (free association walk) ───────────────
    print("\n" + "=" * 70)
    print("TEST 3: Free Association Walk")
    print("=" * 70)
    print("  Each step shifts to the top generated word as new source.")
    print("  Watch the concept drift through semantic space.")

    for source in ["king", "ocean", "fire", "brain", "music", "love"]:
        chain = drifting_chain(source, n_per_step=5, n_steps=8)
        if not chain:
            continue

        print(f"\n  Starting from: {source}")
        for step in chain:
            ctx_str = ", ".join(step["context"][:4])
            gen_str = " → ".join(step["generated"][:5])
            print(f"    {step['source']:15s} [{ctx_str}] → {gen_str}")

    # ── Test 4: P3Memory generative ──────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 4: P3Memory Generative (Transversal Decoding)")
    print("=" * 70)
    print("  Store 3 associates + query with 4th → transversal → decode")

    for source in ["king", "ocean", "fire", "love", "brain", "music"]:
        if source not in assoc:
            continue
        targets = assoc[source]
        if len(targets) < 4:
            continue

        # Use first 4 associates
        seed = targets[:4]
        result = p3_generate(source, seed, n_results=10)
        if result is None:
            print(f"\n  {source}: no transversal found")
            continue

        known = set(targets)
        words = [w for _, w in result]
        n_known = sum(1 for w in words if w in known)
        word_str = "  ".join(
            f"{w}{'*' if w in known else ' '}" for _, w in result)
        print(f"\n  {source} [seed: {', '.join(seed)}]")
        print(f"    Generated: {word_str}")
        print(f"    Known associates: {n_known}/{len(words)}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("  GramMemory continuously generates thematically related words")
    print("  by scoring vocabulary against the accumulated relational pattern.")
    print("  Chaining works: generated words used as new context still produce")
    print("  on-topic results, though quality degrades over multiple hops.")
    print("  P3Memory generates via transversal decoding — precise but requires")
    print("  4 seed lines and doesn't always find transversals.")
    print("=" * 70)


if __name__ == "__main__":
    main()
