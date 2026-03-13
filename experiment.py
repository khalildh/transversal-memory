"""
experiment.py — The file you modify
=====================================

This is the ONLY file the autonomous agent edits.

It must define two functions:
  build(source, train_associates, emb) -> state
  rank(source, state, emb, exclude) -> [(score, word), ...]

The evaluate.py harness calls build() once per source word with the
training associates, then rank() to get the full vocabulary ranking.

Current approach: multi-transversal generation in G(2,4) with
batch-vectorised scoring.
"""

import numpy as np
from transversal_memory import P3Memory
from transversal_memory.plucker import (
    random_projection_dual,
    batch_encode_lines_dual,
    batch_score_transversals,
)


# ── Tunable parameters ──────────────────────────────────────────────────────

N_PROJ = 3              # Grassmannian: G(2, N_PROJ+1). 3=P³(6D), 5=P⁵(15D)
N_TRANSVERSALS = 20     # number of transversals to sample
SCORING_METHOD = "sum_log"  # "sum_log", "mean", "max"
PROJECTION_SEED = 99    # seed for projection matrices
SAMPLE_SEED = 42        # seed for transversal sampling

DIM = 32                # embedding dimension (must match data)


# ── Projection matrices (computed once) ──────────────────────────────────────

_W1, _W2 = None, None
_current_seed = None


def _get_projections():
    global _W1, _W2, _current_seed
    if _current_seed != PROJECTION_SEED or _W1 is None:
        if N_PROJ == 3:
            rng = np.random.default_rng(PROJECTION_SEED)
            _W1, _W2 = random_projection_dual(DIM, rng)
        else:
            from transversal_memory.higher_grass import (
                random_projection_dual_general,
            )
            rng = np.random.default_rng(PROJECTION_SEED)
            _W1, _W2 = random_projection_dual_general(DIM, N_PROJ, rng)
        _current_seed = PROJECTION_SEED
    return _W1, _W2


# ── Build: compute transversals from training associates ─────────────────────

def build(source, train_associates, emb):
    """
    Encode training associates as Plücker lines and compute transversals.
    Returns dict with transversals and projection matrices.
    """
    W1, W2 = _get_projections()

    # Encode all training associates as lines
    if N_PROJ == 3:
        target_lines = {}
        for t in train_associates:
            if t in emb.tgt:
                L = _encode_line(emb.src[source], emb.tgt[t], W1, W2)
                if L is not None:
                    target_lines[t] = L
    else:
        from transversal_memory.higher_grass import (
            project_to_line_dual_general,
        )
        target_lines = {}
        for t in train_associates:
            if t in emb.tgt:
                L = project_to_line_dual_general(
                    emb.src[source], emb.tgt[t], W1, W2, N_PROJ)
                if np.linalg.norm(L) > 1e-12:
                    target_lines[t] = L

    valid_targets = list(target_lines.keys())

    # Compute transversals by sampling random subsets
    transversals = _compute_transversals(target_lines, valid_targets)

    return {
        "transversals": transversals,
        "W1": W1,
        "W2": W2,
        "source_vec": emb.src[source],
    }


def _encode_line(src_vec, tgt_vec, W1, W2):
    """Encode a single line in G(2, N_PROJ+1)."""
    from transversal_memory.plucker import project_to_line_dual
    L = project_to_line_dual(src_vec, tgt_vec, W1, W2)
    if np.linalg.norm(L) > 1e-12:
        return L
    return None


def _compute_transversals(target_lines, valid_targets):
    """Sample random subsets and compute transversals."""
    if N_PROJ == 3:
        return _compute_transversals_p3(target_lines, valid_targets)
    else:
        return _compute_transversals_higher(target_lines, valid_targets)


def _compute_transversals_p3(target_lines, valid_targets):
    """Transversals in G(2,4): sample 4-tuples, need 3 stored + 1 query."""
    if len(valid_targets) < 4:
        return []

    rng = np.random.default_rng(SAMPLE_SEED)
    transversals = []
    attempts = 0
    max_attempts = N_TRANSVERSALS * 5

    while len(transversals) < N_TRANSVERSALS and attempts < max_attempts:
        attempts += 1
        idx = rng.choice(len(valid_targets), size=4, replace=False)
        lines = [target_lines[valid_targets[i]] for i in idx]

        mem = P3Memory()
        mem.store(lines[:3])
        tvs = mem.query_generative(lines[3])

        for T, resid in tvs:
            transversals.append(T)

    return transversals


def _compute_transversals_higher(target_lines, valid_targets):
    """Transversals in G(2, N_PROJ+1): need D-2 stored + 1 query."""
    from transversal_memory.higher_grass import (
        plucker_dim, lines_needed, HigherP3Memory,
    )

    K = lines_needed(N_PROJ)
    if len(valid_targets) < K + 1:
        return []

    rng = np.random.default_rng(SAMPLE_SEED)
    transversals = []
    attempts = 0
    max_attempts = N_TRANSVERSALS * 5

    while len(transversals) < N_TRANSVERSALS and attempts < max_attempts:
        attempts += 1
        idx = rng.choice(len(valid_targets), size=K + 1, replace=False)
        lines = [target_lines[valid_targets[i]] for i in idx]

        mem = HigherP3Memory(n_proj=N_PROJ)
        mem.store(lines[:K])
        tvs = mem.query_generative(lines[K])

        for T, resid in tvs:
            transversals.append(T)

    return transversals


# ── Rank: score all vocabulary against transversals ──────────────────────────

def rank(source, state, emb, exclude):
    """
    Rank all vocabulary words by combined Plücker inner product
    across all transversals. Returns [(score, word), ...] sorted
    ascending (lowest = closest to transversal).
    """
    transversals = state["transversals"]
    W1, W2 = state["W1"], state["W2"]
    source_vec = state["source_vec"]

    if not transversals:
        return []

    # Get target embeddings, excluding train associates + source
    all_words = [w for w in emb.vocab if w not in exclude and w in emb.tgt]
    tgt_mat = np.stack([emb.tgt[w] for w in all_words])

    if N_PROJ == 3:
        # Batch encode + score in G(2,4)
        lines = batch_encode_lines_dual(source_vec, tgt_mat, W1, W2)
        T_mat = np.stack(transversals)
        scores = batch_score_transversals(T_mat, lines, method=SCORING_METHOD)
    else:
        # Batch encode + score in G(2, N_PROJ+1)
        from transversal_memory.higher_grass import (
            batch_encode_lines_dual_general,
            batch_score_transversals_general,
            hodge_matrix_general,
        )
        lines = batch_encode_lines_dual_general(
            source_vec, tgt_mat, W1, W2, N_PROJ)
        J = hodge_matrix_general(N_PROJ)
        T_mat = np.stack(transversals)
        scores = batch_score_transversals_general(
            T_mat, lines, J, method=SCORING_METHOD)

    # Sort ascending (lowest score = best match)
    order = np.argsort(scores)
    return [(float(scores[i]), all_words[i]) for i in order]


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from evaluate import evaluate
    result = evaluate(build, rank)
    print(f"\nPrimary metric (mean p@10): {result['mean_p@10']:.6f}")
