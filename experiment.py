"""
experiment.py — 7-signal RRF with reciprocal nearest-neighbor fusion
Best: p@10=0.1135 (53.4% lift over cosine NN baseline of 0.074)

Signals:
  1. Cosine similarity to source (src embedding)
  2. Cosine similarity to source (tgt embedding)
  3. Cosine similarity to centroid of associates
  4. Max similarity to any associate
  5. Mean similarity to all associates
  6. Top-3 mean similarity to closest associates
  7. Reciprocal NN: per-associate rank fusion
"""

import numpy as np

RRF_K = 13
RECIP_K = 24


def build(source, train_associates, emb):
    assoc_vecs = []
    for t in train_associates:
        if t in emb.tgt:
            assoc_vecs.append(emb.tgt[t])

    centroid = None
    assoc_mat = None
    if assoc_vecs:
        centroid = np.mean(assoc_vecs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        assoc_mat = np.stack(assoc_vecs)

    return {
        "source_vec": emb.src[source],
        "source_tgt": emb.tgt.get(source, None),
        "centroid": centroid,
        "assoc_mat": assoc_mat,
    }


def _rrf_ranks(raw_scores, ascending=False):
    """Convert raw scores to RRF contribution: 1/(K + rank)."""
    order = np.argsort(raw_scores) if ascending else np.argsort(-raw_scores)
    ranks = np.empty(len(raw_scores), dtype=int)
    ranks[order] = np.arange(len(raw_scores))
    return 1.0 / (RRF_K + ranks)


def _recip_nn(sim_matrix):
    """Per-associate reciprocal rank fusion.

    For each associate, rank all candidates by similarity, then sum
    1/(RECIP_K + rank) across associates. This captures how consistently
    a candidate appears near multiple associates.
    """
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

    all_words = [w for w in emb.vocab if w not in exclude and w in emb.tgt]
    tgt_mat = np.stack([emb.tgt[w] for w in all_words])
    N = len(all_words)

    rrf_scores = np.zeros(N)

    # Signal 1: Cosine to source (src embedding space)
    rrf_scores += _rrf_ranks(tgt_mat @ source_vec)

    # Signal 2: Cosine to source (tgt embedding space)
    if source_tgt is not None:
        rrf_scores += _rrf_ranks(tgt_mat @ source_tgt)

    # Signal 3: Cosine to centroid
    if centroid is not None:
        rrf_scores += _rrf_ranks(tgt_mat @ centroid)

    if assoc_mat is not None:
        sim_matrix = tgt_mat @ assoc_mat.T

        # Signal 4: Max similarity to any associate
        rrf_scores += _rrf_ranks(sim_matrix.max(axis=1))

        # Signal 5: Mean similarity to all associates
        rrf_scores += _rrf_ranks(sim_matrix.mean(axis=1))

        # Signal 6: Top-3 mean similarity
        if assoc_mat.shape[0] >= 3:
            top3 = np.partition(sim_matrix, -3, axis=1)[:, -3:]
            rrf_scores += _rrf_ranks(top3.mean(axis=1))
        else:
            rrf_scores += _rrf_ranks(sim_matrix.mean(axis=1))

        # Signal 7: Reciprocal nearest-neighbor fusion
        rrf_scores += _rrf_ranks(_recip_nn(sim_matrix))

    order = np.argsort(-rrf_scores)
    return [(float(rrf_scores[i]), all_words[i]) for i in order]


if __name__ == "__main__":
    from evaluate import evaluate
    result = evaluate(build, rank)
    print(f"\nPrimary metric (mean p@10): {result['mean_p@10']:.6f}")
