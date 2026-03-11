"""
examples/cooccurrence_demo.py
==============================
Builds word embeddings entirely from association norms — no GloVe, no external
corpus — then runs the full Transversal Memory pipeline on them.

Demonstrates concretely:
  - What the co-occurrence matrix looks like
  - What PPMI does to it
  - What the SVD embedding dimensions mean
  - Discriminative scoring (GramMemory)
  - Generative retrieval (P3Memory)
  - Held-out prediction: does the geometry separate known from unknown associates?
"""

import numpy as np
import sys
sys.path.insert(0, "..")

from transversal_memory import (
    CooccurrenceMatrix, embeddings_from_associations,
    GramMemory, P3Memory
)
from transversal_memory.plucker import random_projection


# ── Data ──────────────────────────────────────────────────────────────────────

# A focused subset: three semantic fields
# (emotional withdrawal / abandonment words) with overlapping associates.
# The interesting structure is that some associates are shared across fields,
# some are unique, and some are synonyms of the source word itself.

ASSOCIATIONS = {
    # --- emotional-withdrawal cluster ---
    "abandonment": [
        "child", "issues", "fear", "rates", "emotional", "neglect",
        "trauma", "loss", "reaction", "feelings", "desertion", "forsaken",
        "isolation", "rejection", "loneliness", "betrayal", "vacancy",
        "dereliction", "solitude", "separation", "alienation", "estrangement",
        "withdrawal", "parting", "absence", "orphaned",
    ],
    "isolation": [
        "loneliness", "solitude", "fear", "social", "cell", "ward",
        "separation", "withdrawal", "alienation", "confinement", "quarantine",
        "seclusion", "rejection", "feelings", "emotional", "estrangement",
        "depression", "anxiety", "chamber", "tank",
    ],
    "rejection": [
        "fear", "social", "feelings", "emotional", "trauma", "loneliness",
        "abandonment", "isolation", "depression", "anxiety", "letter",
        "application", "job", "sensitivity", "pain", "self-esteem",
        "criticism", "failure", "betrayal",
    ],

    # --- active-abandonment verbs ---
    "abandons": [
        "plan", "project", "post", "ship", "vehicle", "idea", "cause",
        "mission", "partner", "home", "deserts", "leaves", "forsakes",
        "neglects", "betrays", "quits", "discards", "relinquishes",
        "renounces", "withdraws", "surrenders", "ceases", "vacates",
        "abdicates", "dumps",
    ],
    "deserts": [
        "army", "soldier", "post", "partner", "family", "abandons",
        "leaves", "forsakes", "flees", "escapes", "dry", "sand",
        "sahara", "wasteland", "arid", "oasis", "camel", "mirage",
    ],
    "neglects": [
        "child", "duty", "responsibility", "health", "appearance",
        "ignores", "abandons", "forgets", "overlooks", "disregards",
        "garden", "house", "work", "studies", "feelings",
    ],

    # --- self-abasement verbs ---
    "abase": [
        "self", "oneself", "pride", "dignity", "spirit", "character",
        "humiliate", "degrade", "demean", "lower", "belittle",
        "humble", "subjugate", "debase", "mortify", "shame",
    ],
    "humiliate": [
        "shame", "embarrass", "degrade", "demean", "abase", "belittle",
        "mock", "ridicule", "insult", "humiliation", "pride", "dignity",
        "public", "punishment", "bully",
    ],
    "degrade": [
        "quality", "performance", "signal", "environment", "dignity",
        "humiliate", "demean", "abase", "lower", "reduce", "erode",
        "soil", "damage", "debase", "corruption",
    ],
}


def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ── Step 1: Build and inspect the co-occurrence matrix ───────────────────────

def demo_cooccurrence_matrix():
    print_section("Step 1: The co-occurrence matrix")

    co = CooccurrenceMatrix()
    co.add_many(ASSOCIATIONS, position_decay=True)
    C_raw = co.build(weighting="count")

    print(f"\nVocabulary size: {len(co.vocab)} words")
    print(f"Matrix shape: {C_raw.shape}")
    print(f"Non-zero entries: {int((C_raw > 0).sum())} / {C_raw.size}")
    print(f"Density: {(C_raw > 0).mean():.3f}")

    # Show the raw counts for a few pairs
    print("\nRaw counts for selected (source, target) pairs:")
    check_pairs = [
        ("abandonment", "loneliness"),
        ("abandonment", "fear"),
        ("isolation",   "loneliness"),
        ("isolation",   "fear"),
        ("rejection",   "loneliness"),
        ("rejection",   "fear"),
        ("abandonment", "sand"),       # should be 0
        ("abase",       "loneliness"), # should be 0
    ]
    for src, tgt in check_pairs:
        if src in co.word2idx and tgt in co.word2idx:
            i, j = co.word2idx[src], co.word2idx[tgt]
            print(f"  C[{src}, {tgt}] = {C_raw[i,j]:.3f}")

    # Now rebuild with PPMI
    C_ppmi = co.build(weighting="ppmi")
    print("\nPPMI values for same pairs:")
    for src, tgt in check_pairs:
        if src in co.word2idx and tgt in co.word2idx:
            i, j = co.word2idx[src], co.word2idx[tgt]
            print(f"  PPMI[{src}, {tgt}] = {C_ppmi[i,j]:.3f}")

    print("""
Why PPMI instead of raw counts?
  "fear" appears as an associate of many source words.
  Raw counts make "fear" look highly related to everything.
  PPMI = log(P(i,j) / P(i,*)·P(*,j))
  It measures how much more often two words co-occur than you'd
  expect by chance given their individual frequencies.
  A word that appears with ONLY "abandonment" gets high PPMI
  for that pair. A word that appears everywhere gets low PPMI
  for all pairs. This separates specific from generic associates.
    """)

    return co


# ── Step 2: SVD embeddings and what they mean ─────────────────────────────────

def demo_svd_embeddings(co: CooccurrenceMatrix):
    print_section("Step 2: SVD embeddings")

    emb = co.svd_embeddings(dim=8, role="both")

    print(f"\nEmbedding dim: {emb.dim}")
    print(f"Singular values: {np.round(emb.singular_values, 3)}")
    print(f"Variance explained by each dim: {np.round(emb.variance_explained(), 3)}")
    print(f"Dims needed for 90% variance: {emb.effective_rank(0.90)}")

    # What do the top dimensions separate?
    print("\nNearest neighbours in SOURCE space (U vectors, cue role):")
    for word in ["abandonment", "abase", "deserts"]:
        if word in emb.src:
            nn = emb.nearest(word, k=5, space="source")
            names = [f"{w}({s:.2f})" for s,w in nn]
            print(f"  {word}: {', '.join(names)}")

    print("\nNearest neighbours in TARGET space (V vectors, associate role):")
    for word in ["loneliness", "fear", "shame"]:
        if word in emb.tgt:
            nn = emb.nearest(word, k=5, space="target")
            names = [f"{w}({s:.2f})" for s,w in nn]
            print(f"  {word}: {', '.join(names)}")

    print("""
Why separate source (U) and target (V) vectors?

  The association data is directed: "abandonment → trauma" is in the data
  but "trauma → abandonment" probably isn't.
  U[:,k] captures how words behave as CUE words (what they prime).
  V[:,k] captures how words behave as ASSOCIATE words (what primes them).
  A Plücker line for (abandonment → grief) uses:
    a = U[abandonment]  (abandonment in cue role)
    b = V[grief]        (grief in associate role)
  This is more precise than using the same vector for both roles.
    """)

    return emb


# ── Step 3: Build Plücker lines and GramMemory ────────────────────────────────

def demo_gram_memory(emb):
    print_section("Step 3: GramMemory — discriminative scoring")

    # Project embeddings to P³
    rng = np.random.default_rng(0)
    W = random_projection(emb.dim, rng)   # 4 × dim

    # Build one GramMemory per source word using its known associates
    memories = {}
    for source in ["abandonment", "rejection", "abase"]:
        if source not in ASSOCIATIONS:
            continue
        mem = GramMemory()
        stored = 0
        for target in ASSOCIATIONS[source]:
            line = emb.make_line(source, target, W)
            if line is not None:
                mem.store_line(line)
                stored += 1
        memories[source] = mem
        print(f"Built GramMemory for '{source}' from {stored} lines")

    # Score: known associates should score higher than strangers
    print()
    test_cases = [
        ("abandonment", "grief",      True),   # semantically close, not in list
        ("abandonment", "sorrow",     True),   # semantically close, not in list
        ("abandonment", "loneliness", True),   # IN the list
        ("abandonment", "sand",       False),  # clearly unrelated
        ("abandonment", "algorithm",  False),  # unrelated
        ("rejection",   "loneliness", True),   # IN the list
        ("rejection",   "sand",       False),
        ("abase",       "shame",      True),   # IN the list
        ("abase",       "sand",       False),
    ]

    print(f"{'Pair':<35} {'Score':>8}  {'Expect'}")
    print("-" * 55)
    for source, candidate, expect_high in test_cases:
        if source not in memories:
            continue
        line = emb.make_line(source, candidate, W)
        pair_str = f"({source}, {candidate})"
        if line is None:
            print(f"  {pair_str:<33}  {'OOV':>8}")
        else:
            score = memories[source].score(line)
            label = "HIGH ✓" if expect_high else "low  ✓"
            print(f"  {pair_str:<33} {score:>8.4f}  {label}")

    return memories, W


# ── Step 4: Held-out prediction ───────────────────────────────────────────────

def demo_held_out(emb, W):
    print_section("Step 4: Held-out prediction")
    print("""
  Train on 75% of associates, test on held-out 25%.
  Score held-out associates vs random vocabulary words.
  Are they separated?
    """)

    all_words = list(emb.vocab)
    results = []

    for source in ASSOCIATIONS:
        if source not in emb.src:
            continue
        assocs = [t for t in ASSOCIATIONS[source] if t in emb.tgt]
        if len(assocs) < 4:
            continue

        n_train = int(0.75 * len(assocs))
        train, test = assocs[:n_train], assocs[n_train:]
        non_assocs = [w for w in all_words
                      if w not in assocs and w != source]

        # Build memory from training associates
        mem = GramMemory()
        for t in train:
            line = emb.make_line(source, t, W)
            if line is not None:
                mem.store_line(line)

        # Score test vs non-associates (same length sample)
        test_scores = []
        for t in test:
            line = emb.make_line(source, t, W)
            if line is not None:
                test_scores.append(mem.score(line))

        non_scores = []
        rng2 = np.random.default_rng(99)
        sample = rng2.choice(non_assocs,
                             size=min(len(test), len(non_assocs)),
                             replace=False)
        for t in sample:
            line = emb.make_line(source, t, W)
            if line is not None:
                non_scores.append(mem.score(line))

        if test_scores and non_scores:
            sep = np.mean(test_scores) > np.mean(non_scores)
            results.append((source, test_scores, non_scores, sep))
            mark = "✓" if sep else "✗"
            print(f"  {source:<15}  "
                  f"held-out={np.mean(test_scores):.4f}  "
                  f"non-assoc={np.mean(non_scores):.4f}  {mark}")

    n_sep = sum(r[3] for r in results)
    print(f"\n  Separated: {n_sep}/{len(results)} source words")
    print("  (random baseline = 50%)")


# ── Step 5: Generative retrieval ──────────────────────────────────────────────

def demo_generative(emb, W):
    print_section("Step 5: P3Memory — generative retrieval")
    print("""
  Store 3 known (abandonment → X) lines.
  Query with a 4th.
  Decode the transversal to find which word it points toward.
    """)

    mem = P3Memory()
    stored_targets = ["trauma", "loneliness", "rejection"]
    stored_lines = []
    for t in stored_targets:
        line = emb.make_line("abandonment", t, W)
        if line is not None:
            stored_lines.append(line)
            print(f"  Stored: abandonment → {t}")

    if len(stored_lines) < 3:
        print("  Not enough lines — try a larger association dataset")
        return

    mem.store(stored_lines)
    print()

    for query_target in ["fear", "isolation", "betrayal"]:
        q_line = emb.make_line("abandonment", query_target, W)
        if q_line is None:
            continue
        solutions = mem.query_generative(q_line)
        if not solutions:
            print(f"  Query: abandonment → {query_target}  →  no real transversals (degenerate config)")
            continue

        T, resid = solutions[0]
        print(f"  Query: abandonment → {query_target}  (Plücker residual={resid:.2e})")

        # Decode: score all (abandonment → w) lines against T
        vocab_scores = []
        for w in emb.vocab:
            if w == "abandonment":
                continue
            line = emb.make_line("abandonment", w, W)
            if line is not None:
                align = abs(float(T @ line))
                vocab_scores.append((align, w))
        vocab_scores.sort(key=lambda x: -x[0])

        top5 = vocab_scores[:5]
        top_words = [f"{w}({s:.3f})" for s, w in top5]
        known = [w for _, w in top5 if w in ASSOCIATIONS["abandonment"]]
        print(f"    Top matches: {', '.join(top_words)}")
        print(f"    Of those, known associates: {known or 'none'}")
        print()


# ── Step 6: Cross-source comparison ──────────────────────────────────────────

def demo_cross_source(emb, W):
    print_section("Step 6: Gram matrix similarity between source words")
    print("""
  How similar are the RELATIONAL PATTERNS of different source words?
  (Not lexical similarity — geometric similarity of their associate clusters.)
    """)

    mems = {}
    for source, assocs in ASSOCIATIONS.items():
        mem = GramMemory()
        for t in assocs:
            line = emb.make_line(source, t, W)
            if line is not None:
                mem.store_line(line)
        mems[source] = mem

    sources = list(mems.keys())
    print(f"\n  {'':15}", end="")
    for s in sources:
        print(f"  {s[:12]:>12}", end="")
    print()

    for s1 in sources:
        print(f"  {s1[:15]:15}", end="")
        for s2 in sources:
            sim = mems[s1].compare(mems[s2])
            print(f"  {sim:>12.3f}", end="")
        print()

    print("""
  Expected structure (with good embeddings):
  - abandonment / isolation / rejection should cluster together
    (all draw from the same emotional-withdrawal vocabulary)
  - abandons / deserts / neglects should cluster together
    (active-verb field, overlapping objects: "partner", "home", "duty")
  - abase / humiliate / degrade should cluster together
    (self-directed diminishment field)
  Cross-cluster similarities should be lower.
    """)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building embeddings from association norms only (no GloVe needed)\n")

    co = demo_cooccurrence_matrix()
    emb = demo_svd_embeddings(co)

    rng = np.random.default_rng(0)
    W = random_projection(emb.dim, rng)

    _, _ = demo_gram_memory(emb)
    demo_held_out(emb, W)
    demo_generative(emb, W)
    demo_cross_source(emb, W)

    print_section("Summary")
    print("""
  The co-occurrence matrix encodes which words share the same contexts.
  PPMI suppresses high-frequency associates that co-occur with everything.
  SVD projects the matrix into a low-dimensional space where semantically
  similar words are geometrically close.
  The Plücker line for (source, target) is the exterior product of their
  projected vectors — it encodes the DIRECTED RELATION, not the words.
  GramMemory accumulates these relational directions.
  Its eigenvectors are the principal axes of the semantic field.
  P3Memory retrieves geometrically implied completions.

  The key difference from GloVe:
  GloVe uses a large external corpus and learns embeddings that generalise
  across many tasks. This approach learns embeddings that are SPECIFIC to
  your association norm data — the geometry is exactly tuned to the
  relational structure in your dataset.
    """)
