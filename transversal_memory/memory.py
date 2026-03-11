"""
memory.py — Three memory architectures

P3Memory         : Single P³ slot. Stores one triple. Generative retrieval.
GramMemory       : Unlimited capacity. Discriminative (energy scoring).
ProjectedMemory  : K P³ slots via random projections. High capacity.

All three use Plücker coordinates as the fundamental representation.
Items are n-dimensional vectors; lines are item-pairs projected to P³.
"""

import numpy as np
from typing import Optional, Union
from .plucker import (
    line_from_points, line_from_dm_vec, project_to_line,
    plucker_inner, plucker_relation, is_valid_line, lines_meet,
    random_projection, _IDX_MAP_P3, _PAIRS_P3,
)
from .solver import solve_p3


# ── P3Memory: single slot, generative retrieval ───────────────────────────────

class P3Memory:
    """
    A single P³ memory slot.

    Stores one triple of lines via a 6×6 Gram matrix.
    Retrieval: project M onto null space of query → eigenvectors →
    exact Plücker solve → up to 2 transversals.

    The two retrieved lines:
      - Satisfy the Plücker relation (are valid lines)
      - Meet all 4 input lines (the stored triple + query)
      - Are mutually skew
      - Did not exist before the query
    """

    def __init__(self):
        self.M = np.zeros((6, 6))
        self._stored_ps: list[np.ndarray] = []
        self._stored_lines: list[np.ndarray] = []

    def store(self, lines: list[np.ndarray]) -> None:
        """
        Store a triple of lines.

        lines: list of 3 arrays, each either:
          - A 6-vector in (i,j)-minor Plücker format  (from line_from_points)
          - A 6-vector in [d;m] format                (from find_transversals)
          - Two 4-vectors (homogeneous points)        — not yet supported here
        """
        assert len(lines) == 3, "Store exactly 3 lines per triple"
        self.M[:] = 0.0
        self._stored_ps = []
        self._stored_lines = []

        for L in lines:
            L = np.asarray(L, float)
            # Detect [d;m] format: the Plücker relation will be large
            p = _ensure_minor_format(L)
            self.M += np.outer(p, p)
            self._stored_ps.append(p)
            self._stored_lines.append(L)

    def query_generative(self,
                         query_line: np.ndarray,
                         tol: float = 1e-8
                         ) -> list[tuple[np.ndarray, float]]:
        """
        Query with a 4th line. Returns up to 2 transversals.

        The condition "T meets L" is T · hodge_dual(L) = 0 in the (i,j)-minor
        Plücker format.  We build the 4×6 constraint matrix A where each row is
        hodge_dual(Li), take its 2D null space via SVD, then use the exact
        Plücker solver to find the (up to 2) valid lines in that null space.

        Returns list of (T_plucker_6vec, plücker_residual) sorted by residual.
        """
        from .plucker import hodge_dual
        if len(self._stored_ps) < 3:
            return []

        p4 = _ensure_minor_format(np.asarray(query_line, float))
        all_lines = self._stored_ps + [p4]

        # Constraint matrix: T must be orthogonal to hodge_dual(Li) for all i
        A = np.stack([hodge_dual(p) for p in all_lines])   # (4, 6)
        _, S, Vt = np.linalg.svd(A, full_matrices=True)

        # Null space is the last 2 right singular vectors
        v1 = Vt[-1].copy()
        v2 = Vt[-2].copy()

        return solve_p3(v1, v2, tol=tol)

    def score(self, candidate_line: np.ndarray) -> float:
        """
        Energy score of a candidate line against the stored pattern.
        High = consistent with stored triple. Low = not consistent.
        """
        p = _ensure_minor_format(np.asarray(candidate_line, float))
        return float(p @ self.M @ p)

    def verify(self,
               T: np.ndarray,
               query_line: np.ndarray,
               tol: float = 1e-4) -> dict:
        """
        Check if T is a genuine transversal: meets all stored lines + query.
        T should be in (i,j)-minor format.
        """
        T = np.asarray(T, float)
        p4 = _ensure_minor_format(np.asarray(query_line, float))
        stored_meets = [abs(plucker_inner(T, p)) < tol
                        for p in self._stored_ps]
        query_meets  = abs(plucker_inner(T, p4)) < tol
        valid        = is_valid_line(T, tol=tol)
        return {
            "meets_stored":  stored_meets,
            "meets_query":   query_meets,
            "valid_line":    valid,
            "all_ok":        all(stored_meets) and query_meets and valid,
            "plucker_resid": abs(plucker_relation(T)),
        }

    def reset(self) -> None:
        self.M[:] = 0.0
        self._stored_ps = []
        self._stored_lines = []


# ── GramMemory: discriminative, unlimited capacity ────────────────────────────

class GramMemory:
    """
    Discriminative memory via Gram matrix accumulation.

    Store any number of lines: M = Σ p⊗p.
    No capacity limit — every stored line adds to M.

    Retrieval:
      - score(candidate): T^T M T — how well T fits the stored pattern
      - principal_axes(k): top k eigenvectors of M — dominant relational directions
      - rank_candidates(vocab): score all candidates, return sorted
      - compare(other): cosine similarity between two GramMemory matrices

    Typical use: build one GramMemory per source word from its associate list.
    The eigenvectors then reveal the principal relational axes of that word.
    """

    def __init__(self):
        self.M = np.zeros((6, 6))
        self.n_lines = 0
        self._lines: list[np.ndarray] = []  # kept for inspection

    def store_line(self, line: np.ndarray) -> None:
        """Add one line to the memory."""
        p = _ensure_minor_format(np.asarray(line, float))
        self.M += np.outer(p, p)
        self.n_lines += 1
        self._lines.append(p)

    def store_lines(self, lines: list[np.ndarray]) -> None:
        """Add multiple lines."""
        for L in lines:
            self.store_line(L)

    def score(self, candidate: np.ndarray) -> float:
        """
        Energy score: how consistent is this line with the stored pattern?
        Normalised by trace(M) so scores are comparable across memories
        with different numbers of stored lines.
        """
        p = _ensure_minor_format(np.asarray(candidate, float))
        raw = float(p @ self.M @ p)
        tr = np.trace(self.M)
        return raw / tr if tr > 1e-12 else 0.0

    def score_raw(self, candidate: np.ndarray) -> float:
        """Unnormalised energy score."""
        p = _ensure_minor_format(np.asarray(candidate, float))
        return float(p @ self.M @ p)

    def principal_axes(self, k: int = 3) -> np.ndarray:
        """
        Top k eigenvectors of M, sorted by descending eigenvalue.
        These are the principal relational directions of the stored pattern.

        Returns array of shape (k, 6).
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.M)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, idx[:k]].T

    def eigenvalues(self) -> np.ndarray:
        """All eigenvalues of M, descending."""
        vals = np.linalg.eigvalsh(self.M)
        return np.sort(vals)[::-1]

    def rank_candidates(self,
                        candidates: list[np.ndarray],
                        labels: Optional[list] = None
                        ) -> list[tuple]:
        """
        Score all candidates and return sorted list of (score, label, line).

        candidates: list of Plücker 6-vectors
        labels:     optional list of names/strings

        Returns list of (score, label, line) sorted descending by score.
        """
        if labels is None:
            labels = list(range(len(candidates)))
        scored = [(self.score(c), lbl, c)
                  for c, lbl in zip(candidates, labels)]
        scored.sort(key=lambda x: -x[0])
        return scored

    def compare(self, other: "GramMemory") -> float:
        """
        Cosine similarity between two GramMemory matrices (as vectors).
        Measures how similar two words' relational patterns are.
        """
        a = self.M.flatten()
        b = other.M.flatten()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0

    def reset(self) -> None:
        self.M[:] = 0.0
        self.n_lines = 0
        self._lines = []


# ── ProjectedMemory: K slots, high capacity ───────────────────────────────────

class ProjectedMemory:
    """
    K independent P³ memory slots via random projections.

    Items: n-dimensional vectors (word embeddings, image features, etc.)
    Lines: item-pairs projected to P³ via a slot-specific 4×n matrix Wk.
    Capacity: K independent triple associations.

    Each slot operates identically to P3Memory but in a different random
    projection of the item space. A genuine relational match will produce
    low-violation transversals across many slots; an accidental match will
    not survive multiple random projections.

    Usage:
        mem = ProjectedMemory(n_items=64, n_slots=20)
        mem.store([(a1,b1), (a2,b2), (a3,b3)], label="capital_of")
        results = mem.query((madrid, spain))
    """

    def __init__(self,
                 n_items: int = 64,
                 n_slots: int = 20,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_items = n_items
        self.n_slots = n_slots
        # 4×n_items projection matrices, one per slot
        self.W = np.stack([random_projection(n_items, rng)
                           for _ in range(n_slots)])
        self.slots  = [P3Memory() for _ in range(n_slots)]
        self.labels = [None] * n_slots
        self._next  = 0

    def _project_pair(self, k: int,
                      a: np.ndarray,
                      b: np.ndarray) -> np.ndarray:
        """Project item-pair (a, b) into slot k's P³."""
        return project_to_line(a, b, self.W[k])

    def store(self,
              item_pairs: list[tuple[np.ndarray, np.ndarray]],
              label: Optional[str] = None) -> int:
        """
        Store a triple of item-pairs.

        item_pairs: list of 3 tuples (a, b) where a, b ∈ Rⁿ
        label:      optional name for this association

        Returns the slot index used.
        """
        assert len(item_pairs) == 3
        k = self._next % self.n_slots
        self._next += 1
        projected = [self._project_pair(k, a, b)
                     for a, b in item_pairs]
        self.slots[k].store(projected)
        self.labels[k] = label or f"assoc_{k}"
        return k

    def query(self,
              item_pair: tuple[np.ndarray, np.ndarray],
              tol_violation: float = 0.05
              ) -> list[dict]:
        """
        Query with a new item-pair (a, b).

        Searches all active slots. Returns a list of dicts sorted by
        Plücker violation (lower = better match):
          {slot, label, transversal, violation, score}

        A genuine match has violation < tol_violation.
        Slots where the random projection did not preserve the relational
        structure will produce high-violation results.
        """
        a, b = item_pair
        results = []

        for k in range(self.n_slots):
            if self.slots[k].n_stored_check():
                continue
            q = self._project_pair(k, a, b)
            transversals = self.slots[k].query_generative(q)

            for T, viol in transversals:
                results.append({
                    "slot":        k,
                    "label":       self.labels[k],
                    "transversal": T,
                    "violation":   viol,
                    "score":       self.slots[k].score(q),
                })

        results.sort(key=lambda x: x["violation"])
        return results

    def query_best(self,
                   item_pair: tuple[np.ndarray, np.ndarray],
                   tol_violation: float = 0.05
                   ) -> Optional[dict]:
        """Return the single best result, or None if no good match."""
        results = self.query(item_pair, tol_violation)
        good = [r for r in results if r["violation"] < tol_violation]
        return good[0] if good else None

    @property
    def n_stored(self) -> int:
        return min(self._next, self.n_slots)


# Patch P3Memory to add the helper needed by ProjectedMemory
def _p3_n_stored_check(self):
    return len(self._stored_ps) == 0

P3Memory.n_stored_check = _p3_n_stored_check


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_minor_format(p: np.ndarray) -> np.ndarray:
    """
    Ensure p is in (i,j)-minor Plücker format.
    If it looks like [d;m] format (large Plücker relation), convert it.
    """
    assert p.shape == (6,), f"Expected 6-vector, got shape {p.shape}"
    # Check if it's already in minor format: Plücker relation should be ~0
    rel = (p[_IDX_MAP_P3[(0,1)]]*p[_IDX_MAP_P3[(2,3)]]
         - p[_IDX_MAP_P3[(0,2)]]*p[_IDX_MAP_P3[(1,3)]]
         + p[_IDX_MAP_P3[(0,3)]]*p[_IDX_MAP_P3[(1,2)]])
    if abs(rel) < 0.01:
        return p
    # Looks like [d;m] format — convert
    return line_from_dm_vec(p)
