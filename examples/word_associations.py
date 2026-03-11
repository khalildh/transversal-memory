"""
examples/word_associations.py
==============================
Demonstrates both memory modes on a word association dataset.

Mode 1 — Discriminative (GramMemory):
  Build one Gram matrix per source word from all its associates.
  Score and rank candidate words.
  Find principal relational axes.
  Compare relational patterns across source words.

Mode 2 — Generative (P3Memory via WordMemory.analogy):
  Store 3 associates as a triple.
  Query with a 4th.
  Decode the transversal back to the nearest word in vocabulary.

Does NOT require GloVe. Uses random embeddings by default.
Set GLOVE_PATH to use real embeddings.
"""

import numpy as np

from transversal_memory.embeddings import (
    WordMemory, random_embeddings, load_glove
)

GLOVE_PATH = None   # e.g. "glove.6B.50d.txt"

# ── Data ──────────────────────────────────────────────────────────────────────

ASSOCIATIONS = {
    "abandonment": [
        "child", "issues", "fear", "rates", "emotional", "neglect",
        "trauma", "loss", "reaction", "feelings", "desertion", "forsaken",
        "isolation", "rejection", "loneliness", "betrayal", "vacancy",
        "disuse", "dereliction", "solitude", "separation", "alienation",
        "estrangement", "withdrawal", "parting", "dismissal", "absence",
        "orphaned",
    ],
    "abandons": [
        "plan", "project", "post", "ship", "vehicle", "idea", "cause",
        "mission", "partner", "home", "deserts", "leaves", "forsakes",
        "neglects", "betrays", "quits", "discards", "relinquishes",
        "renounces", "withdraws", "surrenders", "ceases", "vacates",
        "evacuates", "abdicates", "forfeits", "discontinues", "retreats",
        "releases", "dumps",
    ],
    "abase": [
        "self", "oneself", "pride", "dignity", "spirit", "character",
        "nature", "ego", "status", "identity", "humiliate", "degrade",
        "demean", "lower", "belittle", "disparage", "humble", "subjugate",
        "debase", "mortify", "shame", "demote", "downgrade", "depreciate",
        "diminish",
    ],
}


def build_vocabulary(associations: dict) -> list[str]:
    """All unique words across source words and their associates."""
    vocab = set(associations.keys())
    for targets in associations.values():
        vocab.update(targets)
    return sorted(vocab)


def demo_discriminative(wm: WordMemory, vocab: list[str]):
    print("=" * 65)
    print("Mode 1: Discriminative — GramMemory energy scoring")
    print("=" * 65)

    # Score known associate vs random word
    print("\n  Scoring candidates for 'abandonment':")
    test_words = ["grief", "longing", "algebra", "velocity", "sorrow",
                  "rejection", "comfort"]
    for word in test_words:
        s = wm.score("abandonment", word)
        if s is not None:
            known = "✓ known" if word in ASSOCIATIONS["abandonment"] else "  unseen"
            print(f"    {word:<15} score={s:.4f}  {known}")

    # Rank vocabulary
    print("\n  Top 10 words by fit to 'abandonment' relational pattern:")
    ranked = wm.rank("abandonment", candidates=vocab, top_k=10)
    for i, (score, word) in enumerate(ranked):
        known = "✓" if word in ASSOCIATIONS["abandonment"] else " "
        print(f"    {i+1:>2}. {word:<20} {score:.4f}  {known}")


def demo_principal_axes(wm: WordMemory):
    print("\n" + "=" * 65)
    print("Principal relational axes of 'abandonment'")
    print("=" * 65)

    clusters = wm.cluster_associates("abandonment", k=3)
    if clusters:
        print("\n  Associates grouped by dominant axis:\n")
        for i, group in enumerate(clusters):
            print(f"  Axis {i+1}: {', '.join(group)}")
        print()
        print("  (With GloVe embeddings, axes typically correspond to:")
        print("   emotional consequences / synonyms / co-occurrence contexts)")


def demo_cross_word_comparison(wm: WordMemory):
    print("\n" + "=" * 65)
    print("Cross-word comparison — relational pattern similarity")
    print("=" * 65)

    pairs = [
        ("abandonment", "abandons"),
        ("abandonment", "abase"),
        ("abandons",    "abase"),
    ]
    print()
    for a, b in pairs:
        sim = wm.compare(a, b)
        if sim is not None:
            print(f"  similarity({a}, {b}) = {sim:.4f}")

    print()
    print("  (abandonment/abandons should be higher than abandonment/abase")
    print("   since they share more of the same relational field)")


def demo_generative(wm: WordMemory, vocab: list[str]):
    print("\n" + "=" * 65)
    print("Mode 2: Generative — analogy via transversal retrieval")
    print("=" * 65)

    print("""
  Store: (abandonment, trauma), (abandonment, loneliness),
         (abandonment, rejection)   [emotional-consequence associates]
  Query: (abandonment, fear)
  Decode: what word best completes the transversal?
    """)

    stored_assocs = ["trauma", "loneliness", "rejection"]
    query_assoc   = "fear"

    results = wm.analogy(
        source="abandonment",
        associates=stored_assocs,
        query_target=query_assoc,
        n_candidates=10,
    )

    print("  Top candidates:\n")
    for i, (score, word) in enumerate(results):
        known = "✓ known associate" if word in ASSOCIATIONS["abandonment"] else ""
        print(f"  {i+1:>2}. {word:<20} align={score:.4f}  {known}")

    print()
    print("  With random embeddings the ranking will be random.")
    print("  With GloVe, emotional-consequence words should rank higher.")


def demo_held_out_prediction(wm: WordMemory, vocab: list[str]):
    print("\n" + "=" * 65)
    print("Held-out prediction — score associates vs non-associates")
    print("=" * 65)

    # Split each word's associates: 80% train, 20% test
    for source in ASSOCIATIONS:
        assocs = ASSOCIATIONS[source]
        n_train = int(0.8 * len(assocs))
        train   = assocs[:n_train]
        test    = assocs[n_train:]

        # Build a fresh memory with training associates only
        from transversal_memory.embeddings import WordMemory as WM
        wm_partial = WM(wm.embeddings, wm.W)
        wm_partial.add_associations(source, train)

        # Score test associates vs non-associates
        non_assocs = [w for w in vocab
                      if w not in assocs and w != source][:len(test)]

        test_scores     = [wm_partial.score(source, w) for w in test
                           if wm_partial.score(source, w) is not None]
        non_assoc_scores = [wm_partial.score(source, w) for w in non_assocs
                            if wm_partial.score(source, w) is not None]

        if test_scores and non_assoc_scores:
            print(f"\n  '{source}':")
            print(f"    held-out associates:  mean={np.mean(test_scores):.4f}")
            print(f"    non-associates:        mean={np.mean(non_assoc_scores):.4f}")
            sep = np.mean(test_scores) > np.mean(non_assoc_scores)
            print(f"    separated: {'✓' if sep else '✗ (need better embeddings)'}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading embeddings...")
    if GLOVE_PATH:
        embeddings = load_glove(GLOVE_PATH, max_words=50000)
    else:
        print("(Using random embeddings. Set GLOVE_PATH for meaningful results.)")
        vocab_words = build_vocabulary(ASSOCIATIONS)
        embeddings  = random_embeddings(vocab_words, dim=50, seed=0)

    vocab = build_vocabulary(ASSOCIATIONS)
    print(f"Vocabulary: {len(vocab)} words\n")

    wm = WordMemory(embeddings)
    for source, targets in ASSOCIATIONS.items():
        n = wm.add_associations(source, targets)
        print(f"  Stored {n} associates for '{source}'")

    demo_discriminative(wm, vocab)
    demo_principal_axes(wm)
    demo_cross_word_comparison(wm)
    demo_generative(wm, vocab)
    demo_held_out_prediction(wm, vocab)

    print("\n" + "=" * 65)
    print("To use GloVe embeddings:")
    print("  1. Download: https://nlp.stanford.edu/projects/glove/")
    print("  2. Set GLOVE_PATH at the top of this file")
    print("  3. Re-run — separation scores should be clearly above chance")
    print("=" * 65)
