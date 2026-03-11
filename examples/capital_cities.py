"""
examples/capital_cities.py
============================
Concrete analogy test:
  Paris:France :: Berlin:Germany :: Rome:Italy → Madrid:Spain

Demonstrates:
  - Building item embeddings
  - Constructing lines as item-pair exterior products
  - Storing a triple in P3Memory
  - Querying and decoding the transversal
  - Comparing generative vs discriminative retrieval
"""

import numpy as np
import sys
sys.path.insert(0, "..")

from transversal_memory import P3Memory, GramMemory, project_to_line, random_projection
from transversal_memory.embeddings import WordMemory, random_embeddings, load_glove

GLOVE_PATH = None   # set to use real embeddings

# ── Data ──────────────────────────────────────────────────────────────────────

CAPITALS = {
    "paris":   "france",
    "berlin":  "germany",
    "rome":    "italy",
    "madrid":  "spain",
    "london":  "uk",
    "vienna":  "austria",
    "lisbon":  "portugal",
    "warsaw":  "poland",
}

# Negative examples (not capitals)
NEGATIVES = {
    "barcelona": "spain",    # city but not capital
    "munich":    "germany",  # city but not capital
    "milan":     "italy",    # city but not capital
    "frankfurt": "germany",
}


def demo_gram_memory(embeddings: dict, W: np.ndarray):
    """
    GramMemory (discriminative):
    Store all known capital pairs, score candidates.
    """
    print("=" * 60)
    print("GramMemory: discriminative scoring")
    print("=" * 60)

    # Build a GramMemory with all known capital-of lines
    # The "source" concept here is the capital-of relation itself
    # We encode it as lines from city → country
    mem = GramMemory()

    print("\nStoring capital-of lines:")
    for city, country in list(CAPITALS.items())[:5]:  # train on 5
        if city in embeddings and country in embeddings:
            line = project_to_line(embeddings[city], embeddings[country], W)
            mem.store_line(line)
            print(f"  ({city}, {country})")

    # Score held-out capital pairs vs negatives
    print("\nScoring held-out pairs (higher = more capital-like):")
    test_pairs = list(CAPITALS.items())[5:]  # last 3 as test
    all_test = test_pairs + list(NEGATIVES.items())

    results = []
    for city, country in all_test:
        if city in embeddings and country in embeddings:
            line = project_to_line(embeddings[city], embeddings[country], W)
            score = mem.score(line)
            is_capital = (city, country) in CAPITALS.items()
            results.append((score, city, country, is_capital))

    results.sort(key=lambda x: -x[0])
    for score, city, country, is_cap in results:
        label = "capital ✓" if is_cap else "not capital"
        print(f"  ({city:<12}, {country:<10}) score={score:.4f}  {label}")


def demo_p3_memory(embeddings: dict, W: np.ndarray):
    """
    P3Memory (generative):
    Store 3 capital pairs, query with Madrid → should retrieve Spain.
    """
    print("\n" + "=" * 60)
    print("P3Memory: generative retrieval")
    print("=" * 60)

    vocab_cities    = list(CAPITALS.keys()) + list(NEGATIVES.keys())
    vocab_countries = list(CAPITALS.values()) + list(NEGATIVES.values())
    vocab_all       = list(embeddings.keys())

    # Store Paris:France, Berlin:Germany, Rome:Italy
    stored_pairs = [("paris","france"), ("berlin","germany"), ("rome","italy")]
    stored_lines = []
    for city, country in stored_pairs:
        if city in embeddings and country in embeddings:
            line = project_to_line(embeddings[city], embeddings[country], W)
            stored_lines.append(line)

    if len(stored_lines) < 3:
        print("  Not enough embeddings for this demo")
        return

    mem = P3Memory()
    mem.store(stored_lines)
    print(f"\nStored: {[f'{c}:{n}' for c,n in stored_pairs]}")

    # Query with Madrid:Spain
    if "madrid" in embeddings and "spain" in embeddings:
        q_line = project_to_line(embeddings["madrid"], embeddings["spain"], W)
        print(f"Query: (madrid, spain)")

        solutions = mem.query_generative(q_line)
        print(f"\nTransversal found: {len(solutions)} candidate(s)")
        print(f"Plücker residuals: {[f'{r:.2e}' for _, r in solutions]}")

        # Decode: find the (city, country) pair nearest to the transversal
        print("\nDecoding transversal to nearest word-pairs:")
        if solutions:
            T, resid = solutions[0]
            best_pairs = []
            for city in vocab_cities:
                for country in vocab_countries:
                    if city not in embeddings or country not in embeddings:
                        continue
                    if city == country:
                        continue
                    line = project_to_line(embeddings[city],
                                           embeddings[country], W)
                    align = abs(float(T @ line))
                    best_pairs.append((align, city, country))
            best_pairs.sort(key=lambda x: -x[0])
            for align, city, country in best_pairs[:5]:
                is_cap = CAPITALS.get(city) == country
                print(f"  ({city:<12}, {country:<10}) align={align:.4f}"
                      f"  {'✓ capital' if is_cap else ''}")


def demo_principal_axes(embeddings: dict, W: np.ndarray):
    """
    What are the principal axes of the capital-of GramMemory?
    With good embeddings these should reflect geographic clusters.
    """
    print("\n" + "=" * 60)
    print("Principal axes of capital-of GramMemory")
    print("=" * 60)

    mem = GramMemory()
    for city, country in CAPITALS.items():
        if city in embeddings and country in embeddings:
            line = project_to_line(embeddings[city], embeddings[country], W)
            mem.store_line(line)

    evals = mem.eigenvalues()
    print(f"\n  Eigenvalues: {evals.round(4)}")
    print(f"  Effective rank ≈ {int(np.sum(evals > 0.01 * evals[0]))}")
    print()
    print("  (With GloVe embeddings, 2-3 dominant eigenvalues typically reflect")
    print("   Western European, Eastern European, and other geographic clusters)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading embeddings...")
    all_words = (list(CAPITALS.keys()) + list(CAPITALS.values())
               + list(NEGATIVES.keys()) + list(NEGATIVES.values()))

    if GLOVE_PATH:
        embeddings = load_glove(GLOVE_PATH, max_words=100000)
        # Filter to known words
        missing = [w for w in all_words if w not in embeddings]
        if missing:
            print(f"  Missing from GloVe: {missing}")
    else:
        print("(Using random embeddings. Set GLOVE_PATH for meaningful results.)")
        embeddings = random_embeddings(all_words, dim=50, seed=42)

    dim = next(iter(embeddings.values())).shape[0]
    W   = random_projection(dim, np.random.default_rng(0))

    demo_gram_memory(embeddings, W)
    demo_p3_memory(embeddings, W)
    demo_principal_axes(embeddings, W)

    print("\n" + "=" * 60)
    print("Note: with random embeddings, no meaningful structure is preserved.")
    print("The architecture is correct; the signal requires real embeddings.")
    print("=" * 60)
