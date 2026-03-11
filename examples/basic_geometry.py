"""
examples/basic_geometry.py
===========================
Demonstrates the core geometric fact:
4 lines in general position in P³ → exactly 2 transversal lines.

No embeddings, no ML. Pure projective geometry.
"""

import numpy as np
import sys
sys.path.insert(0, "..")

from transversal_memory import (
    random_line, P3Memory, find_transversals,
    plucker_inner, is_valid_line, lines_meet
)


def demo_four_lines():
    print("=" * 60)
    print("4 lines in P³ → exactly 2 transversals")
    print("=" * 60)

    rng = np.random.default_rng(42)
    lines = [random_line(rng) for _ in range(4)]

    # Ground truth from constraint-matrix solver
    transversals, disc = find_transversals(lines)

    print(f"\nDiscriminant: {disc:.6f}  →  2 real transversals\n")

    for i, T in enumerate(transversals):
        print(f"Transversal T{i+1}:")
        print(f"  Valid Plücker line: {is_valid_line(T)}")
        for j, L in enumerate(lines):
            ip = plucker_inner(T, L)
            print(f"  meets L{j+1}: {abs(ip):.2e}  {'✓' if abs(ip)<1e-6 else '✗'}")
        print()

    # T1 and T2 are skew
    ip_12 = plucker_inner(transversals[0], transversals[1])
    print(f"T1 meets T2? inner={ip_12:.4f}  "
          f"→ {'skew (as expected)' if abs(ip_12) > 0.01 else 'they meet'}")


def demo_p3_memory():
    print("\n" + "=" * 60)
    print("P3Memory: store triple → query → retrieve transversals")
    print("=" * 60)

    rng = np.random.default_rng(7)
    L1, L2, L3, L4 = [random_line(rng) for _ in range(4)]

    mem = P3Memory()
    mem.store([L1, L2, L3])
    solutions = mem.query_generative(L4)

    print(f"\nStored: L1, L2, L3")
    print(f"Query:  L4")
    print(f"Found {len(solutions)} transversal(s):\n")

    for i, (T, resid) in enumerate(solutions):
        v = mem.verify(T, L4)
        print(f"  T{i+1}: Plücker residual={resid:.2e}  all_ok={v['all_ok']}")
        print(f"    meets stored: {v['meets_stored']}")
        print(f"    meets query:  {v['meets_query']}")
        print(f"    valid line:   {v['valid_line']}")


def demo_symmetry():
    print("\n" + "=" * 60)
    print("Symmetry: any line can be the query")
    print("=" * 60)

    rng = np.random.default_rng(99)
    lines = [random_line(rng) for _ in range(4)]
    mem = P3Memory()

    print()
    for held_out in range(4):
        stored = [lines[i] for i in range(4) if i != held_out]
        query  = lines[held_out]
        mem.store(stored)
        solutions = mem.query_generative(query)
        # A genuine transversal should verify against all 4 lines
        good = [T for T, r in solutions
                if all(abs(plucker_inner(T, L)) < 1e-4 for L in lines)]
        print(f"  Hold out L{held_out+1}: {len(good)} genuine transversal(s) found  "
              f"{'✓' if good else '✗'}")


if __name__ == "__main__":
    demo_four_lines()
    demo_p3_memory()
    demo_symmetry()
