# Ideas for further exploration

## Sequence prediction / Transformer replacement

- **Use the system as a transformer-like sequence predictor**: encode token sequences
  as Plücker lines, use transversals to predict next tokens. The geometric constraint
  (transversal must intersect all stored lines) is analogous to attention over context.

- **Token-level operation**: Replace word-level embeddings with subword token
  embeddings (BPE/SentencePiece). This would test whether the geometric structure
  captures syntactic/sequential patterns, not just semantic associations.

- **General sequence prediction**: Given a sequence [t1, t2, ..., tn], encode
  consecutive pairs as lines, compute transversals, and use them to predict t_{n+1}.
  Compare against n-gram baselines and simple RNNs.

- **Attention as Plücker intersection**: Each attention head could be viewed as
  finding transversal lines through the "context lines" formed by key-value pairs.
  The Plücker inner product is a natural measure of "relevance" between a query
  line and stored context lines.

## Current best results (autoresearch)

- Cosine NN baseline: p@10 = 0.074
- Pure geometric (transversals): p@10 = 0.011
- Linear blend (cos + geometry): p@10 = 0.085 (+14.9%)
- 4-signal RRF (cos+cent+max+avg): p@10 = 0.100 (+35.1%)
- 5-signal RRF (+top-3 mean): p@10 = 0.104 (+40.5%)
- 6-signal RRF (+source tgt space): p@10 = 0.108 (+45.9%)
- 7-signal RRF (+reciprocal NN): p@10 = 0.114 (+53.4%)
- 8-signal RRF (+Mahalanobis): p@10 = 0.119 (+60.8%)
- 10-signal RRF (+whitened cos, maha-src): p@10 = 0.123 (+66.2%)
- **8-signal RRF (ablation-pruned, RECIP_K=32): p@10 = 0.128 (+73.0%)**

## Resolved questions

- **Does higher Grassmannian help the hybrid?** No. Tested G(2,4) [6D],
  G(2,6) [15D], G(2,8) [28D], and G(2,17) [136D]. All add zero unique
  information when the full 32D embedding covariance is exploited. Even
  G(2,17) — which EXPANDS from 32D to 136D and has d-prime=6.7 and only
  r=0.45 correlation with cosine — adds exactly 0.0pp to the ensemble.

- **Can we ensemble across Grassmannian dimensions?** Tested G(2,4)+G(2,6)+G(2,8)
  simultaneously. Zero improvement over embedding-only approach.

- **Gram eigenstructure as features?** PCA of the Gram matrix in Plücker space
  provides no additive signal over embedding-space PCA/covariance.

- **Can transversals generate useful pseudo-associates?** No. Using P3Memory
  to generate transversal-decoded words and adding them to the associate set
  degrades performance from 0.128 to 0.117. The pseudo-associates are too
  noisy and dilute the real associate signal.

- **Can transversals re-rank embedding candidates?** Tested re-ranking the
  top-100 embedding candidates using transversal incidence scores across
  multiple projection seeds. No improvement — the Plücker inner products
  show zero separation between associates and random words (d-prime ≈ 0)
  with single projections, and multi-seed aggregation doesn't fix this.

- **Can Plücker incidence serve as a continuous similarity signal?** Tested
  both pairwise Plücker inner products and Gram matrix energy as RRF signals.
  Neither adds to the ensemble. The fundamental reason: Plücker coordinates
  are quadratic functions of projected embeddings, and the Gram energy is
  quartic. But the embedding covariance (used in Mahalanobis/whitened cosine)
  already captures the 2nd-order structure, which strictly dominates.

- **Does signal pruning help?** Yes — significantly. Removing cosine-to-centroid
  (redundant with Mahalanobis-to-centroid) and top-3-mean (redundant with
  max+mean) improved p@10 from 0.123 to 0.127. Fewer signals = less dilution
  in rank fusion.

- **Which signals matter most?** Ablation study (leave-one-out on 8 signals):
  Mahalanobis-cent is the strongest (-0.003 when removed). Whitened-cos,
  recip-NN, and Mahalanobis-src are medium (-0.001 to -0.0015). Cosine
  and max/mean similarity are weak (±0.0005).

- **Alternative fusion methods?** Tested z-score normalization, min-max,
  log-product, squared RRF, weighted RRF, per-signal K values, CombMNZ,
  Borda count. All degrade performance. Standard RRF with uniform K is optimal.

- **Ledoit-Wolf shrinkage vs ridge?** Ridge regularization (MAHA_REG=0.001)
  outperforms Ledoit-Wolf shrinkage by 0.009pp. With only 12-74 training
  associates in 32D, the simple ridge estimate is more stable.

## Open questions

- **Cross-word geometry**: The Gram matrix comparison (e.g., dog/cat similarity
  = 0.86) captures relational structure. Can this be used for zero-shot transfer
  of associate patterns between related source words?

- **Sequence prediction**: Can transversals predict next tokens in sequences?
  The geometric constraint (must intersect all context lines) is analogous to
  attention. Worth testing on BPE-tokenized text.

- **Generative vs discriminative**: Geometry excels at generation (transversals
  create new relational items) but fails at discrimination (ranking existing
  items). Is there a task where generative retrieval is the primary mode?

- **Higher-order interactions**: The embedding ensemble captures up to 2nd-order
  (covariance) structure. Could non-linear kernels or tensor decompositions
  in the original 32D space (without lossy projection) capture useful higher-order
  interactions?
