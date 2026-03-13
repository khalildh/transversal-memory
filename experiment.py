"""
experiment.py — 8-signal RRF with covariance-based scoring
Best: p@10=0.1280 (73.0% lift over cosine NN baseline of 0.074)

Signals (after ablation-driven pruning):
  1. Cosine similarity to source (src embedding)
  2. Cosine similarity to source (tgt embedding)
  3. Max similarity to any associate
  4. Mean similarity to all associates
  5. Reciprocal NN: per-associate rank fusion
  6. Mahalanobis distance to centroid (associate covariance)
  7. Whitened cosine: cosine in covariance-normalized space
  8. Mahalanobis distance to source (associate covariance)

Removed (hurt performance via signal dilution):
  - Cosine to centroid: redundant with Mahalanobis to centroid
  - Top-3 mean: redundant with max + mean signals
"""

import numpy as np

RRF_K = 13
RECIP_K = 32
MAHA_REG = 0.001


def build(source, train_associates, emb):
    assoc_vecs = []
    for t in train_associates:
        if t in emb.tgt:
            assoc_vecs.append(emb.tgt[t])

    centroid = None
    assoc_mat = None
    inv_cov = None
    sqrt_inv_cov = None
    if assoc_vecs:
        centroid = np.mean(assoc_vecs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        assoc_mat = np.stack(assoc_vecs)
        if len(assoc_vecs) >= 5:
            cov = np.cov(assoc_mat.T) + MAHA_REG * np.eye(assoc_mat.shape[1])
            try:
                inv_cov = np.linalg.inv(cov)
                vals, vecs = np.linalg.eigh(inv_cov)
                vals = np.maximum(vals, 0)
                sqrt_inv_cov = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
            except np.linalg.LinAlgError:
                pass

    return {
        "source_vec": emb.src[source],
        "source_tgt": emb.tgt.get(source, None),
        "centroid": centroid,
        "assoc_mat": assoc_mat,
        "inv_cov": inv_cov,
        "sqrt_inv_cov": sqrt_inv_cov,
    }


def _rrf_ranks(raw_scores, ascending=False):
    """Convert raw scores to RRF contribution: 1/(K + rank)."""
    order = np.argsort(raw_scores) if ascending else np.argsort(-raw_scores)
    ranks = np.empty(len(raw_scores), dtype=int)
    ranks[order] = np.arange(len(raw_scores))
    return 1.0 / (RRF_K + ranks)


def _recip_nn(sim_matrix):
    """Per-associate reciprocal rank fusion."""
    N, M = sim_matrix.shape
    recip = np.zeros(N)
    for j in range(M):
        order_j = np.argsort(-sim_matrix[:, j])
        ranks_j = np.empty(N, dtype=float)
        ranks_j[order_j] = np.arange(N, dtype=float)
        recip += 1.0 / (RECIP_K + ranks_j)
    return recip


def rank(source, state, emb, exclude):
    source_vec = state["source_vec"]
    source_tgt = state["source_tgt"]
    centroid = state["centroid"]
    assoc_mat = state["assoc_mat"]
    inv_cov = state["inv_cov"]
    sqrt_inv_cov = state["sqrt_inv_cov"]

    all_words = [w for w in emb.vocab if w not in exclude and w in emb.tgt]
    tgt_mat = np.stack([emb.tgt[w] for w in all_words])
    N = len(all_words)

    rrf_scores = np.zeros(N)

    # Signal 1: Cosine to source (src embedding space)
    rrf_scores += _rrf_ranks(tgt_mat @ source_vec)

    # Signal 2: Cosine to source (tgt embedding space)
    if source_tgt is not None:
        rrf_scores += _rrf_ranks(tgt_mat @ source_tgt)

    if assoc_mat is not None:
        sim_matrix = tgt_mat @ assoc_mat.T

        # Signal 3: Max similarity to any associate
        rrf_scores += _rrf_ranks(sim_matrix.max(axis=1))

        # Signal 4: Mean similarity to all associates
        rrf_scores += _rrf_ranks(sim_matrix.mean(axis=1))

        # Signal 5: Reciprocal nearest-neighbor fusion
        rrf_scores += _rrf_ranks(_recip_nn(sim_matrix))

    # Signal 6: Mahalanobis distance to centroid
    if inv_cov is not None and centroid is not None:
        diff = tgt_mat - centroid[None, :]
        maha_dist = np.sum((diff @ inv_cov) * diff, axis=1)
        rrf_scores += _rrf_ranks(maha_dist, ascending=True)

    # Signal 7: Whitened cosine to centroid
    if sqrt_inv_cov is not None and centroid is not None:
        w_tgt = tgt_mat @ sqrt_inv_cov
        w_cent = centroid @ sqrt_inv_cov
        norms = np.linalg.norm(w_tgt, axis=1) * np.linalg.norm(w_cent)
        w_cos = w_tgt @ w_cent / (norms + 1e-12)
        rrf_scores += _rrf_ranks(w_cos)

    # Signal 8: Mahalanobis distance to source
    if inv_cov is not None:
        diff = tgt_mat - source_vec[None, :]
        maha_src = np.sum((diff @ inv_cov) * diff, axis=1)
        rrf_scores += _rrf_ranks(maha_src, ascending=True)

    order = np.argsort(-rrf_scores)
    return [(float(rrf_scores[i]), all_words[i]) for i in order]


if __name__ == "__main__":
    from evaluate import evaluate
    result = evaluate(build, rank)
    print(f"\nPrimary metric (mean p@10): {result['mean_p@10']:.6f}")
