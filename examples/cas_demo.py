"""
examples/cas_demo.py
=====================
Demonstrate the content-addressable store built on Plücker geometry.

Tests:
1. Store and retrieve files by content hash
2. Deduplication (same content → same hash)
3. Similarity search (find files with similar content)
4. Fragment search (find files containing a substring)
5. Integrity verification via Plücker relation
6. Content fingerprinting (eigenstructure as signature)
7. Corruption detection
"""

import sys
import os
import numpy as np

sys.path.insert(0, "..")

from transversal_memory.cas import ContentStore, ContentSignature


def main():
    print("=" * 70)
    print("Content-Addressable Store — Plücker Geometry Demo")
    print("=" * 70)

    store = ContentStore(chunk_size=16, dim=32, seed=42)

    # ── Test 1: Store and retrieve ───────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: Store and Retrieve")
    print("=" * 70)

    files = {
        "hello.txt": b"Hello, world! This is a test file for the content-addressable store.",
        "python.py": b"def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        "readme.md": b"# Transversal Memory\n\nA content-addressable memory system based on projective geometry.\n",
        "data.csv": b"name,age,city\nAlice,30,London\nBob,25,Paris\nCharlie,35,Tokyo\nDiana,28,Berlin\n",
        "poem.txt": b"Shall I compare thee to a summer's day?\nThou art more lovely and more temperate.\nRough winds do shake the darling buds of May,\nAnd summer's lease hath all too short a date.\n",
    }

    hashes = {}
    for name, content in files.items():
        h = store.put(content, label=name)
        hashes[name] = h
        print(f"  Stored: {name:15s} ({len(content):4d} bytes) → {h[:16]}...")

    print(f"\n  Total items: {store.n_items}")

    # Retrieve
    print("\n  Retrieval test:")
    for name, h in hashes.items():
        data = store.get(h)
        match = data == files[name]
        print(f"    {name:15s} → {'✓ exact match' if match else '✗ MISMATCH'}")

    # ── Test 2: Deduplication ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Deduplication")
    print("=" * 70)

    h1 = store.put(files["hello.txt"], label="hello_copy")
    print(f"  Re-stored hello.txt → {h1[:16]}...")
    print(f"  Same hash as original: {h1 == hashes['hello.txt']}")
    print(f"  Store size unchanged: {store.n_items} items")

    # ── Test 3: Content signatures ───────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: Content Signatures (Gram eigenstructure)")
    print("=" * 70)

    for name in files:
        sig = store.get_signature(hashes[name])
        evals = sig.eigenvalues()
        nonzero = np.sum(evals > 1e-10)
        print(f"\n  {name:15s}  lines={sig.n_lines:2d}  "
              f"rank={nonzero}  "
              f"evals={evals[:4].round(3)}")

    # ── Test 4: Similarity search ────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 4: Similarity Search")
    print("=" * 70)

    # Add more files for interesting comparisons
    similar_files = {
        "hello2.txt": b"Hello, world! This is another test file for the content store.",
        "python2.py": b"def factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n",
        "poem2.txt": b"Shall I compare thee to a winter's night?\nThou art more dark and more intemperate.\nCold winds do break the frozen buds of March,\nAnd winter's hold hath all too long a date.\n",
        "data2.csv": b"name,age,city\nEve,31,Madrid\nFrank,27,Rome\nGrace,33,Vienna\nHenry,29,Dublin\n",
        "random.bin": os.urandom(200),
    }

    for name, content in similar_files.items():
        store.put(content, label=name)

    print(f"\n  Store now has {store.n_items} items\n")

    # Query with variants of existing content
    queries = [
        ("hello variant", b"Hello, world! This is a modified test file for content storage."),
        ("python variant", b"def fibonacci(n):\n    a, b = 0, 1\n    for i in range(n):\n        a, b = b, a + b\n    return a\n"),
        ("poem variant", b"Shall I compare thee to a summer's day?\nThou art more lovely and more temperate.\n"),
        ("csv variant", b"name,age,city\nAlice,30,London\nBob,25,Paris\n"),
        ("random query", os.urandom(200)),
    ]

    for query_name, query_data in queries:
        results = store.find_similar(query_data, top_k=3)
        print(f"  Query: {query_name}")
        for r in results:
            print(f"    {r['label']:15s}  sim={r['similarity']:.4f}  "
                  f"({r['size']} bytes)")
        print()

    # ── Test 5: Pairwise similarity matrix ───────────────────────────
    print("=" * 70)
    print("TEST 5: Pairwise Similarity Matrix")
    print("=" * 70)

    labels, sim_mat = store.similarity_matrix()
    # Print a subset
    show = min(10, len(labels))
    header = "".join(f"{l[:8]:>10s}" for l in labels[:show])
    print(f"\n  {'':15s}{header}")
    for i in range(show):
        row = "".join(f"{sim_mat[i,j]:10.3f}" for j in range(show))
        print(f"  {labels[i]:15s}{row}")

    # ── Test 6: Integrity verification ───────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 6: Integrity Verification")
    print("=" * 70)

    for name in ["hello.txt", "python.py", "poem.txt"]:
        result = store.verify(hashes[name])
        print(f"\n  {name}:")
        print(f"    Hash valid:      {result['hash_valid']}")
        print(f"    Geometry valid:  {result['geometric_valid']}")
        print(f"    Mean score:      {result['mean_score']:.6f}")
        print(f"    Max Plücker res: {result['max_plucker_residual']:.2e}")

    # ── Test 7: Corruption detection ─────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 7: Corruption Detection")
    print("=" * 70)

    original = files["poem.txt"]
    sig_original = ContentSignature(original, chunk_size=16, dim=32)

    # Corrupt at different levels
    corruptions = [
        ("1 byte flip", bytearray(original)),
        ("10 byte flip", bytearray(original)),
        ("50 byte flip", bytearray(original)),
        ("completely different", b"This is completely different content that has nothing to do with poetry."),
    ]

    corruptions[0][1][10] ^= 0xFF  # flip 1 byte
    for i in range(10):
        corruptions[1][1][i * 15 % len(original)] ^= 0xFF
    for i in range(50):
        corruptions[2][1][i * 3 % len(original)] ^= 0xFF

    print(f"\n  Original poem: {len(original)} bytes, "
          f"{sig_original.n_lines} lines")

    for desc, corrupted in corruptions:
        corrupted_bytes = bytes(corrupted) if isinstance(corrupted, bytearray) else corrupted
        sig_corrupt = ContentSignature(corrupted_bytes, chunk_size=16, dim=32)
        sim = sig_original.similarity(sig_corrupt)
        hash_match = hashlib.sha256(corrupted_bytes).hexdigest() == \
                     hashlib.sha256(original).hexdigest()
        print(f"\n  {desc}:")
        print(f"    Hash match:  {hash_match}")
        print(f"    Similarity:  {sim:.4f}")
        print(f"    Detected:    {'✓' if sim < 0.999 else '✗'}")

    # ── Test 8: Signature serialisation ──────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 8: Signature Serialisation")
    print("=" * 70)

    sig = store.get_signature(hashes["hello.txt"])
    raw = sig.to_bytes()
    restored = ContentSignature.from_bytes(raw, sig.content_hash)

    print(f"  Original eigenvalues:  {sig.eigenvalues()[:4].round(4)}")
    print(f"  Restored eigenvalues:  {restored.eigenvalues()[:4].round(4)}")
    print(f"  Similarity:            {sig.similarity(restored):.6f}")
    print(f"  Serialised size:       {len(raw)} bytes "
          f"(vs {len(files['hello.txt'])} original)")

    # ── Test 9: Fragment search ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 9: Fragment Search")
    print("=" * 70)
    print("  Search for stored files by a content fragment")

    fragments = [
        ("fibonacci code", b"def fibonacci(n):\n    a, b = 0, 1\n"),
        ("csv header", b"name,age,city\n"),
        ("Shakespeare", b"Shall I compare thee"),
        ("Hello world", b"Hello, world!"),
    ]

    for desc, fragment in fragments:
        results = store.find_by_fragment(fragment, top_k=3)
        print(f"\n  Fragment: \"{desc}\"")
        for r in results:
            print(f"    {r['label']:15s}  score={r['score']:.6f}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


# Need this import for corruption test
import hashlib

if __name__ == "__main__":
    main()
