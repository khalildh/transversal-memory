"""
examples/cas_benchmark.py
==========================
Benchmark: single-seed CAS vs multi-seed Gram^0.05 CAS

Tests:
  1. Similarity search — can it find the right file variant?
  2. Fragment search — can it find the file containing a substring?
  3. Corruption detection — can it detect modified content?
  4. Cross-type discrimination — does it separate different file types?
  5. Near-duplicate detection — can it find slightly modified files?
"""

import os
import sys
import time
import hashlib
import numpy as np

sys.path.insert(0, "..")

from transversal_memory.cas import ContentStore, ContentSignature
from transversal_memory.cas_multiseed import MultiSeedContentStore, MultiSeedSignature


# ── Test data ──────────────────────────────────────────────────────────

# Original files
FILES = {
    "fibonacci.py": b"""def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)

if __name__ == "__main__":
    for i in range(20):
        print(f"fib({i}) = {fibonacci(i)}")
""",
    "factorial.py": b"""def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def fact_recursive(n):
    if n <= 1:
        return 1
    return n * fact_recursive(n - 1)

if __name__ == "__main__":
    for i in range(15):
        print(f"{i}! = {factorial(i)}")
""",
    "sorting.py": b"""def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)
""",
    "poem_shakespeare.txt": b"""Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date.
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimmed;
And every fair from fair sometime declines,
By chance, or nature's changing course, untrimmed;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st,
Nor shall death brag thou wand'rest in his shade,
When in eternal lines to Time thou grow'st.
So long as men can breathe, or eyes can see,
So long lives this, and this gives life to thee.
""",
    "poem_frost.txt": b"""Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could
To where it bent in the undergrowth;
Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same.
""",
    "data_users.csv": b"""name,age,city,occupation
Alice,30,London,engineer
Bob,25,Paris,designer
Charlie,35,Tokyo,manager
Diana,28,Berlin,scientist
Eve,32,Madrid,developer
Frank,27,Rome,analyst
Grace,33,Vienna,architect
Henry,29,Dublin,researcher
""",
    "data_products.csv": b"""name,price,category,stock
Widget,9.99,hardware,150
Gadget,24.99,electronics,75
Doohickey,4.99,misc,300
Thingamajig,14.99,tools,120
Whatchamacallit,19.99,electronics,50
""",
    "config.json": b"""{
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "myapp_production"
    },
    "cache": {
        "type": "redis",
        "ttl": 3600
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}""",
    "random_binary": os.urandom(500),
}

# Variants of originals (for similarity testing)
VARIANTS = {
    "fibonacci_v2.py": b"""def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def fib_memoized(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memoized(n-1) + fib_memoized(n-2)
    return memo[n]

if __name__ == "__main__":
    for i in range(25):
        print(f"fib({i}) = {fibonacci(i)}")
""",
    "poem_shakespeare_v2.txt": b"""Shall I compare thee to a winter's night?
Thou art more dark and more intemperate.
Cold winds do break the frozen buds of March,
And winter's hold hath all too long a date.
Sometime too cold the eye of heaven hides,
And often is his pale complexion dimmed;
And every fair from fair sometime declines,
By chance, or nature's changing course, untrimmed.
""",
    "data_users_v2.csv": b"""name,age,city,occupation
Alice,31,London,engineer
Bob,26,Paris,designer
Charlie,36,Tokyo,manager
Diana,29,Berlin,scientist
Eve,33,Sydney,developer
""",
}


def print_header(title):
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")


# ═════════════════════════════════════════════════════════════════════════
# TEST 1: Similarity Search
# ═════════════════════════════════════════════════════════════════════════

def test_similarity_search(store_single, store_multi, hashes):
    print_header("TEST 1: Similarity Search — Find the right variant")

    queries = [
        ("fibonacci_v2.py", VARIANTS["fibonacci_v2.py"], "fibonacci.py"),
        ("shakespeare_v2", VARIANTS["poem_shakespeare_v2.txt"], "poem_shakespeare.txt"),
        ("users_v2.csv", VARIANTS["data_users_v2.csv"], "data_users.csv"),
    ]

    single_correct = 0
    multi_correct = 0

    for query_name, query_data, expected_label in queries:
        print(f"\n  Query: {query_name} → expected: {expected_label}")

        # Single-seed
        results_s = store_single.find_similar(query_data, top_k=3)
        top1_s = results_s[0]["label"] if results_s else "?"
        correct_s = top1_s == expected_label
        if correct_s:
            single_correct += 1

        # Multi-seed
        results_m = store_multi.find_similar(query_data, top_k=3)
        top1_m = results_m[0]["label"] if results_m else "?"
        correct_m = top1_m == expected_label
        if correct_m:
            multi_correct += 1

        print(f"    Single-seed: {top1_s:<20} sim={results_s[0]['similarity']:.4f}  "
              f"{'✓' if correct_s else '✗'}")
        print(f"    Multi-seed:  {top1_m:<20} sim={results_m[0]['similarity']:.4f}  "
              f"{'✓' if correct_m else '✗'}")

        # Show full top-3
        for tag, results in [("    S", results_s), ("    M", results_m)]:
            ranked = " | ".join(f"{r['label']}:{r['similarity']:.3f}" for r in results[:3])
            print(f"      {tag}: {ranked}")

    print(f"\n  Score: Single={single_correct}/{len(queries)}, "
          f"Multi={multi_correct}/{len(queries)}")


# ═════════════════════════════════════════════════════════════════════════
# TEST 2: Fragment Search
# ═════════════════════════════════════════════════════════════════════════

def test_fragment_search(store_single, store_multi, hashes):
    print_header("TEST 2: Fragment Search — Find file by substring")

    fragments = [
        ("python def", b"def fibonacci(n):\n    a, b = 0, 1\n", "fibonacci.py"),
        ("csv header", b"name,age,city,occupation\n", "data_users.csv"),
        ("shakespeare", b"Shall I compare thee to a summer's day?", "poem_shakespeare.txt"),
        ("json config", b'"database": {\n        "host": "localhost"', "config.json"),
        ("quicksort", b"def quicksort(arr):\n    if len(arr) <= 1:", "sorting.py"),
    ]

    single_correct = 0
    multi_correct = 0

    for desc, fragment, expected in fragments:
        results_s = store_single.find_by_fragment(fragment, top_k=3)
        results_m = store_multi.find_by_fragment(fragment, top_k=3)

        top1_s = results_s[0]["label"] if results_s else "?"
        top1_m = results_m[0]["label"] if results_m else "?"

        cs = top1_s == expected
        cm = top1_m == expected
        if cs: single_correct += 1
        if cm: multi_correct += 1

        print(f"\n  Fragment: '{desc}' → expected: {expected}")
        print(f"    Single: {top1_s:<20} {'✓' if cs else '✗'}")
        print(f"    Multi:  {top1_m:<20} {'✓' if cm else '✗'}")

    print(f"\n  Score: Single={single_correct}/{len(fragments)}, "
          f"Multi={multi_correct}/{len(fragments)}")


# ═════════════════════════════════════════════════════════════════════════
# TEST 3: Corruption Detection
# ═════════════════════════════════════════════════════════════════════════

def test_corruption_detection(store_single, store_multi, hashes):
    print_header("TEST 3: Corruption Detection")

    original = FILES["poem_shakespeare.txt"]

    corruptions = []
    # 1 byte flip
    c1 = bytearray(original)
    c1[50] ^= 0xFF
    corruptions.append(("1 byte", bytes(c1)))

    # 5 byte flip
    c5 = bytearray(original)
    for i in range(5):
        c5[i * 50 % len(original)] ^= 0xFF
    corruptions.append(("5 bytes", bytes(c5)))

    # 20 byte flip
    c20 = bytearray(original)
    for i in range(20):
        c20[i * 25 % len(original)] ^= 0xFF
    corruptions.append(("20 bytes", bytes(c20)))

    # Completely different
    corruptions.append(("different", b"This is completely different content about databases and SQL queries."))

    print(f"\n  Original: poem_shakespeare.txt ({len(original)} bytes)")
    print(f"\n  {'Corruption':<15} {'Single sim':>12} {'Multi sim':>12} {'Detected':>10}")
    print(f"  {'-'*50}")

    # Get signatures for original
    sig_s = store_single.get_signature(hashes["single"]["poem_shakespeare.txt"])
    sig_m = store_multi.get_signature(hashes["multi"]["poem_shakespeare.txt"])

    for desc, corrupted in corruptions:
        # Single-seed
        sig_cs = ContentSignature(corrupted, chunk_size=16, dim=32)
        sim_s = sig_s.similarity(sig_cs)

        # Multi-seed
        sig_cm = MultiSeedSignature(
            corrupted, chunk_size=16, dim=32, n_seeds=50,
            projections=store_multi._projections)
        sim_m = sig_m.similarity(sig_cm)

        detected = "✓" if sim_m < 0.99 else "✗"
        print(f"  {desc:<15} {sim_s:>12.4f} {sim_m:>12.4f} {detected:>10}")


# ═════════════════════════════════════════════════════════════════════════
# TEST 4: Cross-type Discrimination
# ═════════════════════════════════════════════════════════════════════════

def test_cross_type_discrimination(store_single, store_multi, hashes):
    print_header("TEST 4: Pairwise Similarity Matrix")

    for label, store in [("Single-seed", store_single), ("Multi-seed", store_multi)]:
        labels, mat = store.similarity_matrix()
        n = len(labels)

        # Compute within-type and across-type averages
        types = {}
        for lbl in labels:
            if lbl.endswith('.py'):
                types[lbl] = 'python'
            elif lbl.endswith('.txt'):
                types[lbl] = 'text'
            elif lbl.endswith('.csv'):
                types[lbl] = 'csv'
            elif lbl.endswith('.json'):
                types[lbl] = 'json'
            else:
                types[lbl] = 'binary'

        within = []
        across = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = mat[i, j]
                if types[labels[i]] == types[labels[j]]:
                    within.append(sim)
                else:
                    across.append(sim)

        within_mean = np.mean(within) if within else 0
        across_mean = np.mean(across) if across else 0
        gap = within_mean - across_mean

        print(f"\n  {label}:")
        print(f"    Within-type mean similarity:  {within_mean:.4f}")
        print(f"    Across-type mean similarity:  {across_mean:.4f}")
        print(f"    Discrimination gap:           {gap:.4f}")

        # Show matrix for a subset
        show = min(9, n)
        short_labels = [l[:12] for l in labels[:show]]
        print(f"\n    {'':>13}", end="")
        for l in short_labels:
            print(f" {l:>12}", end="")
        print()
        for i in range(show):
            print(f"    {short_labels[i]:>13}", end="")
            for j in range(show):
                v = mat[i, j]
                print(f" {v:>12.3f}", end="")
            print()


# ═════════════════════════════════════════════════════════════════════════
# TEST 5: Near-duplicate detection
# ═════════════════════════════════════════════════════════════════════════

def test_near_duplicate(store_single, store_multi, hashes):
    print_header("TEST 5: Near-Duplicate Detection")

    # Generate 5 near-duplicates of fibonacci.py with small changes
    original = FILES["fibonacci.py"]
    near_dupes = []
    for i in range(5):
        d = bytearray(original)
        # Change a few characters
        pos = (i * 37 + 13) % len(d)
        d[pos:pos+3] = b'xyz'
        near_dupes.append(bytes(d))

    # Generate 5 unrelated files
    unrelated = [os.urandom(len(original)) for _ in range(5)]

    print(f"  Similarity to fibonacci.py:\n")
    print(f"  {'Type':<20} {'Single':>10} {'Multi':>10}")
    print(f"  {'-'*42}")

    sig_s = store_single.get_signature(hashes["single"]["fibonacci.py"])
    sig_m = store_multi.get_signature(hashes["multi"]["fibonacci.py"])

    dupe_sims_s, dupe_sims_m = [], []
    rand_sims_s, rand_sims_m = [], []

    for i, d in enumerate(near_dupes):
        cs = ContentSignature(d, chunk_size=16, dim=32)
        cm = MultiSeedSignature(d, chunk_size=16, dim=32, n_seeds=50,
                                projections=store_multi._projections)
        ss = sig_s.similarity(cs)
        sm = sig_m.similarity(cm)
        dupe_sims_s.append(ss)
        dupe_sims_m.append(sm)
        print(f"  near-dupe {i+1:<12} {ss:>10.4f} {sm:>10.4f}")

    for i, d in enumerate(unrelated):
        cs = ContentSignature(d, chunk_size=16, dim=32)
        cm = MultiSeedSignature(d, chunk_size=16, dim=32, n_seeds=50,
                                projections=store_multi._projections)
        ss = sig_s.similarity(cs)
        sm = sig_m.similarity(cm)
        rand_sims_s.append(ss)
        rand_sims_m.append(sm)
        print(f"  random {i+1:<15} {ss:>10.4f} {sm:>10.4f}")

    gap_s = np.mean(dupe_sims_s) - np.mean(rand_sims_s)
    gap_m = np.mean(dupe_sims_m) - np.mean(rand_sims_m)
    print(f"\n  Mean near-dupe:  Single={np.mean(dupe_sims_s):.4f}  Multi={np.mean(dupe_sims_m):.4f}")
    print(f"  Mean random:     Single={np.mean(rand_sims_s):.4f}  Multi={np.mean(rand_sims_m):.4f}")
    print(f"  Gap (dupe-rand): Single={gap_s:.4f}  Multi={gap_m:.4f}")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("CAS Benchmark: Single-seed vs Multi-seed Gram^0.05")
    print("=" * 70)

    # Build both stores
    print("\nBuilding stores...")
    t0 = time.time()
    store_single = ContentStore(chunk_size=16, dim=32, seed=42)
    t1 = time.time()
    store_multi = MultiSeedContentStore(chunk_size=16, dim=32, n_seeds=50)
    t2 = time.time()

    hashes = {"single": {}, "multi": {}}
    for name, content in FILES.items():
        hashes["single"][name] = store_single.put(content, label=name)
        hashes["multi"][name] = store_multi.put(content, label=name)

    t3 = time.time()
    print(f"  Single-seed: {len(FILES)} items stored ({t3-t1:.2f}s)")
    print(f"  Multi-seed:  {len(FILES)} items stored ({t3-t2:.2f}s)")

    test_similarity_search(store_single, store_multi, hashes)
    test_fragment_search(store_single, store_multi, hashes)
    test_corruption_detection(store_single, store_multi, hashes)
    test_cross_type_discrimination(store_single, store_multi, hashes)
    test_near_duplicate(store_single, store_multi, hashes)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
