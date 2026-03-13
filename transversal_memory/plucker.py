"""
plucker.py — Core Plücker geometry for lines in P³

A line in P³ is represented as a 6-vector of Plücker coordinates:
    p = (p₀₁, p₀₂, p₀₃, p₁₂, p₁₃, p₂₃)
where p_ij = a_i·b_j - a_j·b_i for two points a, b ∈ R⁴ on the line.

This is the exterior product a∧b, mapping G(2,4) → P⁵ (the Plücker embedding).

Key facts:
  - Two lines L, M meet iff plucker_inner(L, M) = 0
  - A vector is a valid line iff plucker_relation(p) = 0
  - The Grassmannian G(2,4) is a quadric hypersurface in P⁵
  - Schubert calculus: 4 lines in general position → exactly 2 transversals
"""

import numpy as np
from itertools import combinations
from typing import Optional


# ── Index map for P³ (n_proj=3, dim=4, D=6) ──────────────────────────────────

_PAIRS_P3 = list(combinations(range(4), 2))
_IDX_MAP_P3 = {ij: k for k, ij in enumerate(_PAIRS_P3)}

# The single Plücker relation for G(2,4):
#   p₀₁·p₂₃ - p₀₂·p₁₃ + p₀₃·p₁₂ = 0
# Indices in the (i,j)-minor ordering:
_PR_AB, _PR_CD = _IDX_MAP_P3[(0,1)], _IDX_MAP_P3[(2,3)]
_PR_AC, _PR_BD = _IDX_MAP_P3[(0,2)], _IDX_MAP_P3[(1,3)]
_PR_AD, _PR_BC = _IDX_MAP_P3[(0,3)], _IDX_MAP_P3[(1,2)]


# ── Constructors ──────────────────────────────────────────────────────────────

def line_from_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Plücker coordinates of the line through points a, b ∈ R⁴.
    Returns a normalised 6-vector.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == (4,) and b.shape == (4,), "Points must be 4-vectors (homogeneous)"
    p = np.array([a[i]*b[j] - a[j]*b[i] for i, j in _PAIRS_P3])
    n = np.linalg.norm(p)
    return p / n if n > 1e-12 else p


def line_from_direction_moment(d: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Convert the classical Plücker format [d; m] (direction + moment)
    used by find_transversals() into the (i,j)-minor format.

    d: unit direction vector in R³
    m: moment vector = point_on_line × d, in R³
    """
    d, m = np.asarray(d, float), np.asarray(m, float)
    pt = np.cross(m, d) / (np.dot(d, d) + 1e-12)   # closest point to origin
    p1 = np.append(pt, 1.0)
    p2 = np.append(pt + d, 1.0)
    return line_from_points(p1, p2)


def line_from_dm_vec(L: np.ndarray) -> np.ndarray:
    """
    Convert a 6-vector [d₀,d₁,d₂,m₀,m₁,m₂] to (i,j)-minor Plücker coords.
    This is the format produced by the original transversal_memory.py helpers.
    """
    return line_from_direction_moment(L[:3], L[3:])


def project_to_line(a: np.ndarray, b: np.ndarray,
                    W: np.ndarray) -> np.ndarray:
    """
    Project item-pair (a, b) ∈ Rⁿ × Rⁿ into P³ via a 4×n matrix W,
    then compute Plücker coordinates.

    a, b : n-dimensional item vectors
    W    : 4×n projection matrix

    Returns a normalised 6-vector.

    WARNING: For a fixed source a, all lines project_to_line(a, b_i, W)
    share the endpoint W@a, making them co-punctal (plucker_inner = 0
    for ALL pairs). Use project_to_line_dual() for generative retrieval.
    """
    Wa = W @ np.asarray(a, float)
    Wb = W @ np.asarray(b, float)
    return line_from_points(Wa, Wb)


def project_to_line_dual(a: np.ndarray, b: np.ndarray,
                         W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Project item-pair (a, b) into P³ via two DIFFERENT matrices,
    using the concatenation [a; b] for both endpoints.

    Point 1 = W1 @ [a; b]
    Point 2 = W2 @ [a; b]

    Both endpoints depend on both a and b, breaking the co-punctal
    degeneracy that makes single-projection encoding useless for
    generative (transversal) retrieval.

    a, b : n-dimensional item vectors
    W1   : 4×2n projection matrix
    W2   : 4×2n projection matrix (must differ from W1)

    Returns a normalised 6-vector.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = np.concatenate([a, b])
    p1 = W1 @ ab
    p2 = W2 @ ab
    return line_from_points(p1, p2)


def random_projection_dual(n_items: int,
                           rng: Optional[np.random.Generator] = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Two independent 4×(2*n_items) projection matrices for dual encoding.

    Returns (W1, W2) for use with project_to_line_dual().
    """
    if rng is None:
        rng = np.random.default_rng()
    W1 = rng.standard_normal((4, 2 * n_items))
    W2 = rng.standard_normal((4, 2 * n_items))
    for W in (W1, W2):
        for i in range(4):
            W[i] /= np.linalg.norm(W[i])
    return W1, W2


# ── Predicates ────────────────────────────────────────────────────────────────

def plucker_inner(p: np.ndarray, q: np.ndarray) -> float:
    """
    Plücker inner product of two lines.
    Zero iff the lines intersect (or are parallel).

    For lines in (i,j)-minor format the inner product is:
        Σᵢⱼ pᵢⱼ · q_kl  (Hodge contraction)
    which in P³ reduces to:
        p₀₁q₂₃ - p₀₂q₁₃ + p₀₃q₁₂ + p₁₂q₀₃ - p₁₃q₀₂ + p₂₃q₀₁
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    return (p[0]*q[5] - p[1]*q[4] + p[2]*q[3]
          + p[3]*q[2] - p[4]*q[1] + p[5]*q[0])


def plucker_relation(p: np.ndarray) -> float:
    """
    The single Plücker relation for G(2,4):
        p₀₁·p₂₃ - p₀₂·p₁₃ + p₀₃·p₁₂ = 0
    Returns 0 for a valid line.
    """
    p = np.asarray(p, float)
    return (p[_PR_AB]*p[_PR_CD]
          - p[_PR_AC]*p[_PR_BD]
          + p[_PR_AD]*p[_PR_BC])


def is_valid_line(p: np.ndarray, tol: float = 1e-6) -> bool:
    """True if p satisfies the Plücker relation within tolerance."""
    return abs(plucker_relation(p)) < tol


def lines_meet(p: np.ndarray, q: np.ndarray, tol: float = 1e-6) -> bool:
    """True if lines p and q intersect."""
    return abs(plucker_inner(p, q)) < tol


# ── Random constructors ───────────────────────────────────────────────────────

def random_line(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Random line in P³ in general position, as a 6-vector in (i,j)-minor format.
    """
    if rng is None:
        rng = np.random.default_rng()
    a = np.append(rng.standard_normal(3), 1.0)
    b = np.append(rng.standard_normal(3), 1.0)
    return line_from_points(a, b)


def hodge_dual(p: np.ndarray) -> np.ndarray:
    """
    Hodge dual of a Plücker vector in (i,j)-minor format for G(2,4).

    The Plücker inner product satisfies:  <p, q> = p · (J·q)
    where J is this Hodge dual operator.

    J·p = [p₂₃, -p₁₃, p₁₂, p₀₃, -p₀₂, p₀₁]
          = [p[5], -p[4], p[3], p[2], -p[1], p[0]]

    Use as constraint rows:  store J·p, then T·(J·p) = plucker_inner(T, p)
    """
    p = np.asarray(p, float)
    return np.array([p[5], -p[4], p[3], p[2], -p[1], p[0]])


def random_projection(n_items: int,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Random 4×n_items projection matrix with normalised rows.
    Maps n-dimensional item vectors into R⁴ for Plücker embedding.
    """
    if rng is None:
        rng = np.random.default_rng()
    W = rng.standard_normal((4, n_items))
    for i in range(4):
        W[i] /= np.linalg.norm(W[i])
    return W


# ── Batch operations (vectorised over vocabulary) ────────────────────────────

# Hodge dual as a matrix: plucker_inner(p, q) = p @ _J6 @ q
_J6 = np.array([
    [0, 0, 0,  0,  0, 1],
    [0, 0, 0,  0, -1, 0],
    [0, 0, 0,  1,  0, 0],
    [0, 0, 1,  0,  0, 0],
    [0,-1, 0,  0,  0, 0],
    [1, 0, 0,  0,  0, 0],
], dtype=float)


def batch_encode_lines_dual(source: np.ndarray,
                             targets: np.ndarray,
                             W1: np.ndarray,
                             W2: np.ndarray) -> np.ndarray:
    """
    Encode all (source, target_i) pairs as Plücker lines in one shot.

    source  : (d,) source embedding
    targets : (N, d) target embeddings
    W1, W2  : (4, 2d) projection matrices

    Returns (N, 6) array of normalised Plücker lines.
    Lines with near-zero norm are left as zeros.
    """
    N, d = targets.shape
    # Build (N, 2d) concatenated vectors: [source; target_i]
    src_tile = np.tile(source, (N, 1))  # (N, d)
    ab = np.hstack([src_tile, targets])  # (N, 2d)

    # Project to R⁴: (N, 4) each
    P1 = ab @ W1.T  # (N, 4)
    P2 = ab @ W2.T  # (N, 4)

    # Exterior product for all 6 pairs: p_ij = P1[:,i]*P2[:,j] - P1[:,j]*P2[:,i]
    lines = np.empty((N, 6))
    for k, (i, j) in enumerate(_PAIRS_P3):
        lines[:, k] = P1[:, i] * P2[:, j] - P1[:, j] * P2[:, i]

    # Normalise each row
    norms = np.linalg.norm(lines, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    lines /= norms

    return lines


def batch_score_transversals(transversals: np.ndarray,
                              lines: np.ndarray,
                              method: str = "sum_log") -> np.ndarray:
    """
    Score all lines against all transversals in one shot.

    transversals : (T, 6) array of transversal lines
    lines        : (N, 6) array of candidate lines
    method       : "sum_log", "mean", or "max"

    Returns (N,) score array (lower = closer to transversal).
    """
    # Plücker inner products via Hodge dual: (T, 6) @ (6, 6) @ (6, N) → (T, N)
    # Equivalent to: for each T,L pair, T @ _J6 @ L
    Jlines = lines @ _J6.T  # (N, 6) — Hodge-transformed lines
    pi = np.abs(transversals @ Jlines.T)  # (T, N) — |⟨T_i, L_j⟩|

    eps = 1e-20
    if method == "sum_log":
        return np.log(pi + eps).sum(axis=0)  # (N,)
    elif method == "mean":
        return pi.mean(axis=0)
    elif method == "max":
        return pi.max(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Classic P³ transversal finder (SVD + scalar quadratic) ───────────────────

def find_transversals(lines: list,
                      tol: float = 1e-8) -> tuple[list, float]:
    """
    Given 4 lines in general position in P³ (in [d;m] format),
    find the 2 transversal lines that meet all four.

    Uses the constraint-matrix approach:
      - Build 4×6 constraint matrix from the Plücker inner product condition
      - Find 2D null space via SVD
      - Solve the Plücker quadratic in the null space

    Returns (transversals, discriminant) where transversals is a list
    of up to 2 lines in [d;m] format.
    """
    assert len(lines) == 4

    # Build constraint matrix in [d;m] Plücker format
    # plucker_inner([d;m]_T, [d;m]_L) = T[:3]·L[3:] + T[3:]·L[:3]
    A = np.zeros((4, 6))
    for i, L in enumerate(lines):
        L = np.asarray(L, float)
        # Detect format: [d;m] (classical) vs (i,j)-minor
        rel = (L[_IDX_MAP_P3[(0,1)]]*L[_IDX_MAP_P3[(2,3)]]
             - L[_IDX_MAP_P3[(0,2)]]*L[_IDX_MAP_P3[(1,3)]]
             + L[_IDX_MAP_P3[(0,3)]]*L[_IDX_MAP_P3[(1,2)]])
        if abs(rel) < 0.01:
            # (i,j)-minor format: use Hodge dual as constraint row
            A[i] = hodge_dual(L)
        else:
            # [d;m] format: swap d and m
            A[i, :3] = L[3:]
            A[i, 3:] = L[:3]

    _, S, Vt = np.linalg.svd(A)
    v1, v2 = Vt[-1], Vt[-2]

    # Solve Plücker quadratic in [d;m] format: d·m = 0
    d1, m1 = v1[:3], v1[3:]
    d2, m2 = v2[:3], v2[3:]
    c2 = np.dot(d1, m1)
    c1 = np.dot(d1, m2) + np.dot(d2, m1)
    c0 = np.dot(d2, m2)

    disc = c1**2 - 4*c2*c0
    transversals = []

    if abs(c2) < tol:
        if abs(c1) > tol:
            t = -c0 / c1
            T = t*v1 + v2
            n = np.linalg.norm(T[:3])
            if n > tol:
                transversals.append(T / n)
        n = np.linalg.norm(v1[:3])
        if n > tol:
            transversals.append(v1 / n)
    elif disc < -tol:
        pass
    else:
        sq = np.sqrt(max(disc, 0.0))
        for sign in [+1, -1]:
            t = (-c1 + sign*sq) / (2*c2)
            T = t*v1 + v2
            n = np.linalg.norm(T[:3])
            if n > tol:
                transversals.append(T / n)

    return transversals, float(disc)


# ── General Grassmannian index map (for G(2,n+1), n_proj > 3) ─────────────────

def make_index_map_general(n_proj: int) -> tuple[dict, list]:
    """
    Build the (i,j)-pair index map for G(2, n_proj+1).
    Returns (idx_map, pairs) where pairs = list of (i,j) with i < j.
    """
    pairs = list(combinations(range(n_proj + 1), 2))
    idx_map = {ij: k for k, ij in enumerate(pairs)}
    return idx_map, pairs
