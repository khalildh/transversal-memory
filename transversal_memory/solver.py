"""
solver.py — Exact Plücker solver via PCA + quadratic formula

Given a 2D subspace spanned by v1, v2 in Plücker space (R⁶ for P³),
find all T = t·v1 + v2 satisfying the Plücker relation.

The Plücker relation on T = t·v1 + v2 expands to a scalar quadratic:
    α·t² + β·t + γ = 0

where:
    α = plucker_relation(v1)
    β = cross term (symmetric bilinear form of v1, v2)
    γ = plucker_relation(v2)

For P³ there is exactly one Plücker relation, so this is already exact.
For higher Grassmannians G(2,n+1) there are multiple relations, all of which
must vanish simultaneously. In the consistent case they all give the same
quadratic (up to scale). PCA of the coefficient matrix finds the consensus.

The method:
  1. Compute (α_i, β_i, γ_i) for each Plücker relation
  2. Stack into matrix, find first singular vector (PCA)
  3. Solve αt² + βt + γ = 0 with the quadratic formula
  4. Return up to 2 real solutions with their residuals
"""

import numpy as np
from itertools import combinations
from typing import Optional


# ── P³ exact solver (single Plücker relation) ─────────────────────────────────

def solve_p3(v1: np.ndarray, v2: np.ndarray,
             tol: float = 1e-10) -> list[tuple[np.ndarray, float]]:
    """
    Find all T = t·v1 + v2 satisfying the P³ Plücker relation.

    For P³ there is one relation: p₀₁p₂₃ - p₀₂p₁₃ + p₀₃p₁₂ = 0.
    With T = t·v1 + v2 this gives a scalar quadratic — solved exactly.

    Returns list of (T, residual) sorted by residual.
    Residual = |plucker_relation(T)|.
    """
    from .plucker import plucker_relation, _PR_AB, _PR_CD, _PR_AC, _PR_BD, _PR_AD, _PR_BC

    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)

    # α·t² + β·t + γ = 0
    # α = plucker_relation(v1) = v1_ab·v1_cd - v1_ac·v1_bd + v1_ad·v1_bc
    alpha = plucker_relation(v1)

    # γ = plucker_relation(v2)
    gamma = plucker_relation(v2)

    # β = symmetric bilinear form:
    # (v1_ab·v2_cd + v2_ab·v1_cd) - (v1_ac·v2_bd + v2_ac·v1_bd) + (v1_ad·v2_bc + v2_ad·v1_bc)
    beta = ((v1[_PR_AB]*v2[_PR_CD] + v2[_PR_AB]*v1[_PR_CD])
          - (v1[_PR_AC]*v2[_PR_BD] + v2[_PR_AC]*v1[_PR_BD])
          + (v1[_PR_AD]*v2[_PR_BC] + v2[_PR_AD]*v1[_PR_BC]))

    return _solve_quadratic(alpha, beta, gamma, v1, v2, tol)


# ── General solver for G(2,n+1) via PCA ───────────────────────────────────────

def solve_general(v1: np.ndarray, v2: np.ndarray,
                  n_proj: int,
                  n_sample: Optional[int] = None,
                  tol: float = 1e-10) -> list[tuple[np.ndarray, float]]:
    """
    Find all T = t·v1 + v2 satisfying ALL Plücker relations for G(2,n+1).

    For each quadruple (a,b,c,d) there is a Plücker relation:
        p_ab·p_cd - p_ac·p_bd + p_ad·p_bc = 0

    Substituting T = t·v1 + v2 gives a quadratic with coefficients (A_i, B_i, C_i).
    In the consistent case all rows [A_i, B_i, C_i] are proportional.
    PCA finds the dominant direction → consensus (α, β, γ) → quadratic formula.

    n_proj  : dimension of projective space (lines live in Pⁿ)
    n_sample: if set, subsample this many quadruples (for speed when n_proj large)

    Returns list of (T, residual) sorted by residual.
    """
    from .plucker import make_index_map_general

    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    idx_map, _ = make_index_map_general(n_proj)
    n1 = n_proj + 1
    all_quads = list(combinations(range(n1), 4))

    if n_sample and len(all_quads) > n_sample:
        idx = np.random.choice(len(all_quads), n_sample, replace=False)
        all_quads = [all_quads[k] for k in idx]

    rows = []
    for a, b, c, d in all_quads:
        ab = idx_map[(a,b)]; cd = idx_map[(c,d)]
        ac = idx_map[(a,c)]; bd = idx_map[(b,d)]
        ad = idx_map[(a,d)]; bc = idx_map[(b,c)]

        A = v1[ab]*v1[cd] - v1[ac]*v1[bd] + v1[ad]*v1[bc]
        B = (v1[ab]*v2[cd] + v2[ab]*v1[cd]
           - v1[ac]*v2[bd] - v2[ac]*v1[bd]
           + v1[ad]*v2[bc] + v2[ad]*v1[bc])
        C = v2[ab]*v2[cd] - v2[ac]*v2[bd] + v2[ad]*v2[bc]
        rows.append([A, B, C])

    coefs = np.array(rows)
    norms = np.linalg.norm(coefs, axis=1)
    coefs = coefs[norms > 1e-12 * (norms.max() + tol)]

    if len(coefs) == 0:
        return []

    _, _, Vt = np.linalg.svd(coefs, full_matrices=False)
    alpha, beta, gamma = Vt[0]

    return _solve_quadratic(alpha, beta, gamma, v1, v2, tol)


# ── Shared quadratic solve ─────────────────────────────────────────────────────

def _solve_quadratic(alpha: float, beta: float, gamma: float,
                     v1: np.ndarray, v2: np.ndarray,
                     tol: float) -> list[tuple[np.ndarray, float]]:
    """
    Solve α·t² + β·t + γ = 0 and construct T = t·v1 + v2.
    Returns list of (T_normalised, residual).
    """
    from .plucker import plucker_relation

    solutions = []

    def add(t=None, b_zero=False):
        T = v1.copy() if b_zero else t*v1 + v2
        n = np.linalg.norm(T)
        if n < tol:
            return
        T /= n
        resid = abs(plucker_relation(T))
        solutions.append((T, resid))

    if abs(alpha) < tol:
        if abs(beta) > tol:
            add(-gamma / beta)
        add(b_zero=True)
    else:
        disc = beta**2 - 4*alpha*gamma
        if disc >= 0:
            sq = np.sqrt(max(disc, 0.0))
            for sign in [+1, -1]:
                add((-beta + sign*sq) / (2*alpha))

    solutions.sort(key=lambda x: x[1])
    return solutions
