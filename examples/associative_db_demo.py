#!/usr/bin/env python3
"""
Demo: Unified Associative Database
===================================

Tests TDGA triadic memory + transversal geometry together.

Operations tested:
    1. Store facts in both triadic + geometric layers
    2. Exact triadic recall (2 → 1)
    3. Cosine similarity search
    4. Geometric subject similarity
    5. Generative expansion (geometric)
    6. Hybrid search (cosine + triadic)
"""

import sys
import time
sys.path.insert(0, "/Volumes/PRO-G40/Code/TDGA/src")
sys.path.insert(0, "/Volumes/PRO-G40/Code/transversal-memory")

from tdga.sample_facts import Fact
from associative_db import AssociativeDB


def make_facts():
    """Sample knowledge graph with clear relational patterns."""
    return [
        # Capitals
        Fact("Paris", "capital_of", "France", "Paris is the capital of France", "2024-01-01"),
        Fact("Berlin", "capital_of", "Germany", "Berlin is the capital of Germany", "2024-01-01"),
        Fact("Madrid", "capital_of", "Spain", "Madrid is the capital of Spain", "2024-01-01"),
        Fact("Rome", "capital_of", "Italy", "Rome is the capital of Italy", "2024-01-01"),
        Fact("Tokyo", "capital_of", "Japan", "Tokyo is the capital of Japan", "2024-01-01"),
        Fact("London", "capital_of", "United Kingdom", "London is the capital of the UK", "2024-01-01"),

        # Rivers
        Fact("Seine", "flows_through", "Paris", "The Seine flows through Paris", "2024-01-01"),
        Fact("Thames", "flows_through", "London", "The Thames flows through London", "2024-01-01"),
        Fact("Tiber", "flows_through", "Rome", "The Tiber flows through Rome", "2024-01-01"),
        Fact("Spree", "flows_through", "Berlin", "The Spree flows through Berlin", "2024-01-01"),

        # Languages
        Fact("France", "speaks", "French", "France speaks French", "2024-01-01"),
        Fact("Germany", "speaks", "German", "Germany speaks German", "2024-01-01"),
        Fact("Spain", "speaks", "Spanish", "Spain speaks Spanish", "2024-01-01"),
        Fact("Italy", "speaks", "Italian", "Italy speaks Italian", "2024-01-01"),
        Fact("Japan", "speaks", "Japanese", "Japan speaks Japanese", "2024-01-01"),

        # Continents
        Fact("France", "located_in", "Europe", "France is in Europe", "2024-01-01"),
        Fact("Germany", "located_in", "Europe", "Germany is in Europe", "2024-01-01"),
        Fact("Japan", "located_in", "Asia", "Japan is in Asia", "2024-01-01"),

        # Cuisine
        Fact("Paris", "famous_for", "croissants", "Paris is famous for croissants", "2024-01-01"),
        Fact("Rome", "famous_for", "pasta", "Rome is famous for pasta", "2024-01-01"),
        Fact("Tokyo", "famous_for", "sushi", "Tokyo is famous for sushi", "2024-01-01"),
    ]


def main():
    print("=" * 60)
    print("Unified Associative Database Demo")
    print("TDGA triadic memory + transversal geometry")
    print("=" * 60)

    # Load
    t0 = time.time()
    print("\n[1] Loading TDGA pipeline + MiniLM embeddings...")
    db = AssociativeDB.from_checkpoints(
        embed_model="/Volumes/PRO-G40/Code/TDGA/results/embed_sorted.pt",
        device="mps",
    )
    print(f"    Loaded in {time.time()-t0:.1f}s")

    # Store facts
    facts = make_facts()
    t0 = time.time()
    print(f"\n[2] Storing {len(facts)} facts...")
    db.store_batch(facts)
    db.finalize_signatures()
    print(f"    Stored in {time.time()-t0:.1f}s")
    print(f"    {db.n_facts} facts, {db.n_subjects} subjects")

    # ── Triadic recall ──
    print("\n" + "─" * 60)
    print("[3] TRIADIC RECALL (exact pattern completion)")
    print("─" * 60)

    queries = [
        ("Paris", "capital_of", None, "What is Paris the capital of?"),
        (None, "capital_of", "France", "What is the capital of France?"),
        ("Seine", "flows_through", None, "Where does the Seine flow?"),
        ("France", "speaks", None, "What does France speak?"),
    ]

    for s, r, o, desc in queries:
        print(f"\n  Q: {desc}")
        try:
            results = db.recall(subject=s, relation=r, object=o)
            if results:
                for sf, overlap in results[:3]:
                    f = sf.fact
                    missing = f.object if o is None else f.subject
                    print(f"    → {missing} (overlap={overlap:.0f})")
            else:
                print("    → (no results)")
        except Exception as e:
            print(f"    → Error: {e}")

    # ── Cosine search ──
    print("\n" + "─" * 60)
    print("[4] COSINE SIMILARITY SEARCH")
    print("─" * 60)

    search_queries = [
        "European capital cities",
        "rivers in major cities",
        "Asian food culture",
    ]

    for q in search_queries:
        print(f"\n  Q: '{q}'")
        results = db.search(q, k=5)
        for sf, sim in results:
            f = sf.fact
            print(f"    {sim:.3f}  {f.subject} {f.relation} {f.object}")

    # ── Geometric similarity ──
    print("\n" + "─" * 60)
    print("[5] GEOMETRIC SUBJECT SIMILARITY (Gram^0.05)")
    print("─" * 60)

    for subj in ["Paris", "France", "Seine"]:
        if subj in [s for s in db.subjects()]:
            similar = db.geometric_similar(subj, k=5)
            print(f"\n  Subjects similar to '{subj}':")
            for other, sim in similar:
                print(f"    {sim:.4f}  {other}")

    # ── Geometric expansion ──
    print("\n" + "─" * 60)
    print("[6] GEOMETRIC EXPANSION (generate new associations)")
    print("─" * 60)

    # For cities with multiple facts, expand to find what fits
    for subj in ["Paris", "Rome", "France"]:
        if subj in db.subjects():
            expanded = db.expand(subj, k=8)
            print(f"\n  Expansion for '{subj}' (what else fits its pattern?):")
            for word, score in expanded:
                print(f"    {score:.4f}  {word}")

    # ── Hybrid search ──
    print("\n" + "─" * 60)
    print("[7] HYBRID SEARCH (cosine + triadic expansion)")
    print("─" * 60)

    for q in ["capital of European country", "famous food in Asia"]:
        print(f"\n  Q: '{q}'")
        results = db.hybrid_search(q, k=5)
        for sf, score, method in results:
            f = sf.fact
            print(f"    {score:.3f} [{method:7s}]  {f.subject} {f.relation} {f.object}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
