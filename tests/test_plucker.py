"""
tests/test_plucker.py
======================
Unit tests for the core Plücker geometry and memory classes.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, "..")

from transversal_memory import (
    line_from_points, plucker_inner, plucker_relation,
    is_valid_line, lines_meet, random_line, find_transversals,
    P3Memory, GramMemory, ProjectedMemory, solve_p3
)


# ── Plücker geometry ──────────────────────────────────────────────────────────

class TestPluckerGeometry:

    def test_valid_line_from_points(self):
        """Lines constructed from points must satisfy Plücker relation."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            a = np.append(rng.standard_normal(3), 1.0)
            b = np.append(rng.standard_normal(3), 1.0)
            p = line_from_points(a, b)
            assert is_valid_line(p, tol=1e-8), f"Plücker relation violated: {plucker_relation(p)}"

    def test_random_line_valid(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            L = random_line(rng)
            assert is_valid_line(L, tol=1e-8)

    def test_coplanar_lines_meet(self):
        """Lines in the same plane must meet."""
        # Two lines through a common point
        rng = np.random.default_rng(2)
        pt = np.append(rng.standard_normal(3), 1.0)
        b  = np.append(rng.standard_normal(3), 1.0)
        c  = np.append(rng.standard_normal(3), 1.0)
        L1 = line_from_points(pt, b)
        L2 = line_from_points(pt, c)
        assert lines_meet(L1, L2, tol=1e-6), "Lines through common point should meet"

    def test_skew_lines_dont_meet(self):
        """Generic random lines in P³ are skew (don't meet)."""
        rng = np.random.default_rng(3)
        n_skew = 0
        for _ in range(100):
            L1 = random_line(rng)
            L2 = random_line(rng)
            if not lines_meet(L1, L2, tol=1e-4):
                n_skew += 1
        assert n_skew > 90, f"Expected most random pairs to be skew, got {n_skew}/100"

    def test_normalization(self):
        """Line from points should be unit norm in direction part."""
        rng = np.random.default_rng(4)
        for _ in range(20):
            a = np.append(rng.standard_normal(3), 1.0)
            b = np.append(rng.standard_normal(3), 1.0)
            p = line_from_points(a, b)
            assert abs(np.linalg.norm(p) - 1.0) < 1e-8


# ── Transversal computation ───────────────────────────────────────────────────

class TestTransversals:

    def test_four_lines_two_transversals(self):
        """4 lines in general position → exactly 2 transversals."""
        rng = np.random.default_rng(42)
        n_success = 0
        for _ in range(50):
            lines = [random_line(rng) for _ in range(4)]
            T_list, disc = find_transversals(lines)
            if disc > 1e-4 and len(T_list) == 2:
                # Verify both transversals meet all 4 lines
                for T in T_list:
                    for L in lines:
                        assert abs(plucker_inner(T, L)) < 1e-5, \
                            "Transversal does not meet stored line"
                n_success += 1
        assert n_success >= 35, f"Only {n_success}/50 configurations gave 2 transversals (some are degenerate)"

    def test_transversals_are_skew(self):
        """The two transversals are mutually skew."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            lines = [random_line(rng) for _ in range(4)]
            T_list, disc = find_transversals(lines)
            if len(T_list) == 2 and disc > 1e-4:
                ip = abs(plucker_inner(T_list[0], T_list[1]))
                assert ip > 0.01, f"Transversals should be skew, inner={ip}"

    def test_symmetry(self):
        """Any of the 4 lines can be the query."""
        rng = np.random.default_rng(99)
        lines = [random_line(rng) for _ in range(4)]
        T_truth, disc = find_transversals(lines)
        if disc < 1e-4:
            return

        mem = P3Memory()
        for held_out in range(4):
            stored = [lines[i] for i in range(4) if i != held_out]
            query  = lines[held_out]
            mem.store(stored)
            solutions = mem.query_generative(query)
            good = [T for T, r in solutions
                    if all(abs(plucker_inner(T, L)) < 1e-4 for L in lines)]
            assert len(good) >= 1, f"Symmetry failed: held out L{held_out+1}"


# ── Exact Plücker solver ──────────────────────────────────────────────────────

class TestSolver:

    def test_solve_p3_finds_valid_lines(self):
        """solve_p3 returns lines satisfying the Plücker relation."""
        rng = np.random.default_rng(10)
        for _ in range(100):
            v1 = rng.standard_normal(6)
            v2 = rng.standard_normal(6)
            v1 /= np.linalg.norm(v1); v2 /= np.linalg.norm(v2)
            solutions = solve_p3(v1, v2)
            for T, resid in solutions:
                assert resid < 0.01, f"Plücker residual too large: {resid}"

    def test_solve_p3_at_most_two(self):
        """solve_p3 returns at most 2 solutions."""
        rng = np.random.default_rng(11)
        for _ in range(100):
            v1 = rng.standard_normal(6); v1 /= np.linalg.norm(v1)
            v2 = rng.standard_normal(6); v2 /= np.linalg.norm(v2)
            solutions = solve_p3(v1, v2)
            assert len(solutions) <= 2

    def test_solve_p3_on_known_subspace(self):
        """If v1, v2 are both valid lines, solve_p3 should recover them."""
        rng = np.random.default_rng(12)
        for _ in range(50):
            # Construct two known valid lines
            a = np.append(rng.standard_normal(3), 1.0)
            b = np.append(rng.standard_normal(3), 1.0)
            c = np.append(rng.standard_normal(3), 1.0)
            d = np.append(rng.standard_normal(3), 1.0)
            v1 = line_from_points(a, b)
            v2 = line_from_points(c, d)

            solutions = solve_p3(v1, v2)
            assert len(solutions) >= 1
            for T, resid in solutions:
                assert is_valid_line(T, tol=1e-6), f"Solution not on Grassmannian, resid={resid}"


# ── GramMemory ────────────────────────────────────────────────────────────────

class TestGramMemory:

    def test_stored_lines_high_energy(self):
        """Stored lines should have higher energy than random lines."""
        rng = np.random.default_rng(20)
        mem = GramMemory()
        stored = [random_line(rng) for _ in range(10)]
        for L in stored:
            mem.store_line(L)

        stored_scores  = [mem.score(L) for L in stored]
        random_scores  = [mem.score(random_line(rng)) for _ in range(100)]

        assert np.mean(stored_scores) > np.mean(random_scores), \
            "Stored lines should score higher than random"

    def test_principal_axes_shape(self):
        rng = np.random.default_rng(21)
        mem = GramMemory()
        for _ in range(10):
            mem.store_line(random_line(rng))
        axes = mem.principal_axes(k=3)
        assert axes.shape == (3, 6)

    def test_compare_identical(self):
        """Identical memories should have similarity 1.0."""
        rng = np.random.default_rng(22)
        mem = GramMemory()
        for _ in range(5):
            mem.store_line(random_line(rng))
        assert abs(mem.compare(mem) - 1.0) < 1e-8

    def test_compare_different(self):
        """Different memories should have similarity < 1.0."""
        rng = np.random.default_rng(23)
        m1, m2 = GramMemory(), GramMemory()
        for _ in range(5):
            m1.store_line(random_line(rng))
            m2.store_line(random_line(rng))
        assert m1.compare(m2) < 0.99

    def test_rank_candidates(self):
        """Stored lines should rank higher than random ones."""
        rng = np.random.default_rng(24)
        mem = GramMemory()
        stored = [random_line(rng) for _ in range(5)]
        for L in stored:
            mem.store_line(L)

        candidates = stored + [random_line(rng) for _ in range(20)]
        labels     = [f"stored_{i}" for i in range(5)] + \
                     [f"random_{i}" for i in range(20)]

        ranked = mem.rank_candidates(candidates, labels)
        stored_scores  = [s for s, lbl, _ in ranked if lbl.startswith("stored")]
        random_scores_ = [s for s, lbl, _ in ranked if lbl.startswith("random")]
        assert np.mean(stored_scores) > np.mean(random_scores_),             f"Stored mean {np.mean(stored_scores):.4f} not > random mean {np.mean(random_scores_):.4f}"


# ── P3Memory ──────────────────────────────────────────────────────────────────

class TestP3Memory:

    def test_generative_retrieval(self):
        """P3Memory query returns verified transversals."""
        rng = np.random.default_rng(30)
        n_success = 0
        for _ in range(50):
            lines = [random_line(rng) for _ in range(4)]
            _, disc = find_transversals(lines)
            if disc < 1e-4:
                continue

            mem = P3Memory()
            mem.store(lines[:3])
            solutions = mem.query_generative(lines[3])

            for T, resid in solutions:
                v = mem.verify(T, lines[3])
                if v["all_ok"]:
                    n_success += 1
                    break

        assert n_success >= 25, f"Only {n_success}/50 successful retrievals"

    def test_score_stored_vs_random(self):
        """Stored lines should score higher than random in P3Memory."""
        rng = np.random.default_rng(31)
        lines = [random_line(rng) for _ in range(3)]
        mem = P3Memory()
        mem.store(lines)

        stored_scores = [mem.score(L) for L in lines]
        random_scores = [mem.score(random_line(rng)) for _ in range(50)]

        assert np.mean(stored_scores) > np.mean(random_scores)


# ── ProjectedMemory ───────────────────────────────────────────────────────────

class TestProjectedMemory:

    def test_store_and_query(self):
        """ProjectedMemory should retrieve from the correct slot."""
        rng = np.random.default_rng(40)
        n_items, n_slots = 32, 10
        mem = ProjectedMemory(n_items=n_items, n_slots=n_slots, seed=0)

        items = [rng.standard_normal(n_items) for _ in range(n_items)]

        stored_slots = []
        queries = []
        for i in range(n_slots):
            triple = [(items[3*i % len(items)], items[(3*i+1) % len(items)]),
                      (items[(3*i+2) % len(items)], items[(3*i+3) % len(items)]),
                      (items[(3*i+4) % len(items)], items[(3*i+5) % len(items)])]
            slot = mem.store(triple)
            stored_slots.append(slot)
            queries.append((items[(3*i+6) % len(items)],
                            items[(3*i+7) % len(items)]))

        # At least some queries should find low-violation results
        low_viol = 0
        for i, (q, slot) in enumerate(zip(queries, stored_slots)):
            results = mem.query(q)
            for r in results:
                if r["violation"] < 0.05:
                    low_viol += 1
                    break

        assert low_viol >= 1, "No low-violation results found"


if __name__ == "__main__":
    # Run without pytest
    import traceback
    test_classes = [
        TestPluckerGeometry, TestTransversals, TestSolver,
        TestGramMemory, TestP3Memory, TestProjectedMemory
    ]
    total, passed, failed = 0, 0, 0
    for cls in test_classes:
        obj = cls()
        for name in dir(obj):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                getattr(obj, name)()
                print(f"  ✓ {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {cls.__name__}.{name}: {e}")
                failed += 1
    print(f"\n{passed}/{total} tests passed")
