"""
higher_grass.py — Generalized Grassmannian G(2, n+1) for n > 3

G(2,4) = P³:  6D Plücker space,  1 relation,  4 lines → 2 transversals
G(2,5) = P⁴: 10D Plücker space,  5 relations, 8 lines → 2 transversals
G(2,6) = P⁵: 15D Plücker space, 15 relations, 13 lines → 2 transversals

Higher dimensions give:
  - More Plücker coordinates per line → richer signatures
  - More constraints per transversal → better discrimination
  - But require more input lines (D - 2 lines per query)

This module provides generalized versions of:
  - line_from_points_general: exterior product in R^(n+1)
  - plucker_inner_general: generalized incidence check
  - hodge_dual_general: constraint row construction
  - HigherGramMemory: Gram matrix in D-dimensional Plücker space
  - HigherP3Memory: transversal finding in G(2, n+1)
"""

import numpy as np
from itertools import combinations
from typing import Optional

from .solver import solve_general


# ── Plücker geometry in G(2, n+1) ────────────────────────────────────────

def plucker_dim(n_proj: int) -> int:
    """Plücker embedding dimension D = C(n+1, 2) for G(2, n+1)."""
    return (n_proj + 1) * n_proj // 2


def lines_needed(n_proj: int) -> int:
    """Number of lines needed to leave a 2D null space: D - 2."""
    return plucker_dim(n_proj) - 2


def line_from_points_general(a: np.ndarray, b: np.ndarray,
                             n_proj: int) -> np.ndarray:
    """
    Plücker coordinates of the line through points a, b in P^n.
    a, b ∈ R^(n+1) (homogeneous coordinates).
    Returns a normalised D-vector where D = C(n+1, 2).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1 = n_proj + 1
    assert a.shape == (n1,) and b.shape == (n1,), \
        f"Points must be {n1}-vectors for P^{n_proj}"

    pairs = list(combinations(range(n1), 2))
    p = np.array([a[i]*b[j] - a[j]*b[i] for i, j in pairs])
    n = np.linalg.norm(p)
    return p / n if n > 1e-12 else p


def plucker_relations_general(p: np.ndarray, n_proj: int) -> np.ndarray:
    """
    Evaluate ALL Plücker relations for G(2, n+1).
    Each 4-subset (a,b,c,d) gives one relation:
        p_ab·p_cd - p_ac·p_bd + p_ad·p_bc = 0

    Returns array of residuals (all should be ~0 for a valid line).
    """
    p = np.asarray(p, float)
    n1 = n_proj + 1
    idx_map = {ij: k for k, ij in enumerate(combinations(range(n1), 2))}

    residuals = []
    for a, b, c, d in combinations(range(n1), 4):
        ab, cd = idx_map[(a,b)], idx_map[(c,d)]
        ac, bd = idx_map[(a,c)], idx_map[(b,d)]
        ad, bc = idx_map[(a,d)], idx_map[(b,c)]
        residuals.append(p[ab]*p[cd] - p[ac]*p[bd] + p[ad]*p[bc])

    return np.array(residuals)


def is_valid_line_general(p: np.ndarray, n_proj: int,
                          tol: float = 1e-6) -> bool:
    """Check if p is a valid line in G(2, n+1)."""
    return np.max(np.abs(plucker_relations_general(p, n_proj))) < tol


def plucker_inner_general(p: np.ndarray, q: np.ndarray,
                          n_proj: int) -> float:
    """
    Generalized Plücker inner product for G(2, n+1).

    Two lines meet iff their Plücker inner product vanishes.
    For G(2,4) this is the standard 6D inner product.
    For higher G(2, n+1), use the Hodge dual contraction.
    """
    return float(p @ hodge_dual_general(q, n_proj))


def hodge_dual_general(p: np.ndarray, n_proj: int) -> np.ndarray:
    """
    Hodge dual for G(2, n+1).

    For G(2,4): the standard 6→6 swap with signs.
    For G(2, n+1): construct the D×D matrix J such that
    plucker_inner(p, q) = p · (J · q).

    J is the matrix where J[idx(i,j), idx(k,l)] = sign * delta
    for complementary pairs in the incidence structure.

    For lines (2-planes), the Hodge dual maps a 2-form to a (n-1)-form.
    The inner product via J generalises the P³ case.
    """
    p = np.asarray(p, float)
    n1 = n_proj + 1
    D = plucker_dim(n_proj)
    pairs = list(combinations(range(n1), 2))
    idx_map = {ij: k for k, ij in enumerate(pairs)}

    if n_proj == 3:
        # Use the known P³ formula directly (faster)
        return np.array([p[5], -p[4], p[3], p[2], -p[1], p[0]])

    # General case: build J matrix
    # For each pair of 2-element subsets (i,j) and (k,l),
    # J[(i,j), (k,l)] is nonzero iff {i,j} and {k,l} form a 4-element
    # subset, and then equals the sign of the permutation (i,j,k,l).
    J = np.zeros((D, D))
    for a, b, c, d in combinations(range(n1), 4):
        ab, cd = idx_map[(a,b)], idx_map[(c,d)]
        ac, bd = idx_map[(a,c)], idx_map[(b,d)]
        ad, bc = idx_map[(a,d)], idx_map[(b,c)]
        # The relation: p_ab*q_cd - p_ac*q_bd + p_ad*q_bc = 0
        # So inner product contribution: p_ab * J_cd + ...
        J[ab, cd] += 1; J[cd, ab] += 1
        J[ac, bd] -= 1; J[bd, ac] -= 1
        J[ad, bc] += 1; J[bc, ad] += 1

    return J @ p


def hodge_matrix_general(n_proj: int) -> np.ndarray:
    """
    Build the D×D Hodge dual matrix J for G(2, n+1).
    Cached version — call once, reuse for batch scoring.

    plucker_inner(p, q) = p @ J @ q
    """
    from .plucker import _J6
    if n_proj == 3:
        return _J6.copy()

    n1 = n_proj + 1
    D = plucker_dim(n_proj)
    pairs = list(combinations(range(n1), 2))
    idx_map = {ij: k for k, ij in enumerate(pairs)}

    J = np.zeros((D, D))
    for a, b, c, d in combinations(range(n1), 4):
        ab, cd = idx_map[(a,b)], idx_map[(c,d)]
        ac, bd = idx_map[(a,c)], idx_map[(b,d)]
        ad, bc = idx_map[(a,d)], idx_map[(b,c)]
        J[ab, cd] += 1; J[cd, ab] += 1
        J[ac, bd] -= 1; J[bd, ac] -= 1
        J[ad, bc] += 1; J[bc, ad] += 1

    return J


def batch_encode_lines_dual_general(source: np.ndarray,
                                     targets: np.ndarray,
                                     W1: np.ndarray,
                                     W2: np.ndarray,
                                     n_proj: int) -> np.ndarray:
    """
    Encode all (source, target_i) pairs as Plücker lines in G(2, n+1).

    source  : (d,) source embedding
    targets : (N, d) target embeddings
    W1, W2  : (n+1, 2d) projection matrices
    n_proj  : projective dimension

    Returns (N, D) array of normalised Plücker lines, D = C(n+1, 2).
    """
    N, d = targets.shape
    n1 = n_proj + 1
    D = plucker_dim(n_proj)
    pairs = list(combinations(range(n1), 2))

    # Build (N, 2d) concatenated vectors
    src_tile = np.tile(source, (N, 1))
    ab = np.hstack([src_tile, targets])  # (N, 2d)

    # Project to R^(n+1)
    P1 = ab @ W1.T  # (N, n+1)
    P2 = ab @ W2.T  # (N, n+1)

    # Exterior product for all D pairs
    lines = np.empty((N, D))
    for k, (i, j) in enumerate(pairs):
        lines[:, k] = P1[:, i] * P2[:, j] - P1[:, j] * P2[:, i]

    # Normalise
    norms = np.linalg.norm(lines, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    lines /= norms

    return lines


def batch_score_transversals_general(transversals: np.ndarray,
                                      lines: np.ndarray,
                                      J: np.ndarray,
                                      method: str = "sum_log") -> np.ndarray:
    """
    Score all lines against all transversals in G(2, n+1).

    transversals : (T, D) array
    lines        : (N, D) array
    J            : (D, D) Hodge dual matrix from hodge_matrix_general()
    method       : "sum_log", "mean", or "max"

    Returns (N,) score array.
    """
    Jlines = lines @ J.T  # (N, D)
    pi = np.abs(transversals @ Jlines.T)  # (T, N)

    eps = 1e-20
    if method == "sum_log":
        return np.log(pi + eps).sum(axis=0)
    elif method == "mean":
        return pi.mean(axis=0)
    elif method == "max":
        return pi.max(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")


def project_to_line_general(a: np.ndarray, b: np.ndarray,
                            W: np.ndarray,
                            n_proj: int) -> np.ndarray:
    """
    Project item-pair (a, b) into P^n via a (n+1)×d matrix W,
    then compute Plücker coordinates in G(2, n+1).
    """
    Wa = W @ np.asarray(a, float)
    Wb = W @ np.asarray(b, float)
    return line_from_points_general(Wa, Wb, n_proj)


def project_to_line_dual_general(a: np.ndarray, b: np.ndarray,
                                 W1: np.ndarray, W2: np.ndarray,
                                 n_proj: int) -> np.ndarray:
    """
    Dual projection into G(2, n+1): both endpoints depend on [a; b].
    W1, W2: (n+1)×(2d) matrices.
    """
    ab = np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    p1 = W1 @ ab
    p2 = W2 @ ab
    return line_from_points_general(p1, p2, n_proj)


def random_projection_general(n_items: int, n_proj: int,
                               rng: Optional[np.random.Generator] = None
                               ) -> np.ndarray:
    """Random (n+1)×n_items projection matrix with normalised rows."""
    if rng is None:
        rng = np.random.default_rng()
    n1 = n_proj + 1
    W = rng.standard_normal((n1, n_items))
    for i in range(n1):
        W[i] /= np.linalg.norm(W[i])
    return W


def random_projection_dual_general(n_items: int, n_proj: int,
                                    rng: Optional[np.random.Generator] = None
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """Two independent (n+1)×(2*n_items) projection matrices."""
    if rng is None:
        rng = np.random.default_rng()
    n1 = n_proj + 1
    W1 = rng.standard_normal((n1, 2 * n_items))
    W2 = rng.standard_normal((n1, 2 * n_items))
    for W in (W1, W2):
        for i in range(n1):
            W[i] /= np.linalg.norm(W[i])
    return W1, W2


# ── Memory classes for higher Grassmannians ──────────────────────────────

class HigherGramMemory:
    """
    GramMemory generalized to G(2, n+1).

    D = C(n+1, 2) dimensional Plücker space.
    M is a D×D Gram matrix instead of 6×6.
    """

    def __init__(self, n_proj: int = 3):
        self.n_proj = n_proj
        self.D = plucker_dim(n_proj)
        self.M = np.zeros((self.D, self.D))
        self.n_lines = 0

    def store_line(self, line: np.ndarray) -> None:
        p = np.asarray(line, float)
        assert p.shape == (self.D,), \
            f"Expected {self.D}-vector for G(2,{self.n_proj+1})"
        self.M += np.outer(p, p)
        self.n_lines += 1

    def score(self, candidate: np.ndarray) -> float:
        p = np.asarray(candidate, float)
        raw = float(p @ self.M @ p)
        tr = np.trace(self.M)
        return raw / tr if tr > 1e-12 else 0.0

    def eigenvalues(self) -> np.ndarray:
        vals = np.linalg.eigvalsh(self.M)
        return np.sort(vals)[::-1]

    def principal_axes(self, k: int = 3) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(self.M)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, idx[:k]].T

    def compare(self, other: "HigherGramMemory") -> float:
        a = self.M.flatten()
        b = other.M.flatten()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0

    def reset(self) -> None:
        self.M[:] = 0.0
        self.n_lines = 0


class HigherP3Memory:
    """
    P3Memory generalized to G(2, n+1).

    Stores D-2 lines (instead of 3 for P³).
    Query with 1 more line → 2D null space → solve Plücker quadratic.
    """

    def __init__(self, n_proj: int = 4):
        self.n_proj = n_proj
        self.D = plucker_dim(n_proj)
        self.n_lines_needed = self.D - 2
        self._stored: list[np.ndarray] = []

    def store(self, lines: list[np.ndarray]) -> None:
        """Store D-2 lines."""
        assert len(lines) == self.n_lines_needed, \
            f"Need exactly {self.n_lines_needed} lines for G(2,{self.n_proj+1})"
        self._stored = [np.asarray(L, float) for L in lines]

    def query_generative(self, query_line: np.ndarray,
                         tol: float = 1e-8
                         ) -> list[tuple[np.ndarray, float]]:
        """
        Query with one more line → find transversals in G(2, n+1).

        Builds constraint matrix from Hodge duals of all stored + query lines.
        SVD → 2D null space → solve_general → up to 2 transversals.
        """
        if len(self._stored) < self.n_lines_needed:
            return []

        q = np.asarray(query_line, float)
        all_lines = self._stored + [q]

        # Constraint matrix: each row is hodge_dual(L_i)
        A = np.stack([hodge_dual_general(L, self.n_proj) for L in all_lines])

        _, S, Vt = np.linalg.svd(A, full_matrices=True)

        # Null space is the last 2 right singular vectors
        v1 = Vt[-1].copy()
        v2 = Vt[-2].copy()

        return solve_general(v1, v2, self.n_proj, tol=tol)
