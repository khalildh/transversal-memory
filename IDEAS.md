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
- **10-signal RRF (+whitened cos, maha-src): p@10 = 0.123 (+66.2%)**

## Resolved questions

- **Does higher Grassmannian help the hybrid?** No. Gram matrix scores from
  G(2,4), G(2,6), and G(2,8) all add zero unique information when the full
  32D embedding covariance is exploited. The Gram matrix in 15D Plücker space
  is a lossy compression of the 32×32 covariance — strictly dominated.

- **Can we ensemble across Grassmannian dimensions?** Tested G(2,4)+G(2,6)+G(2,8)
  simultaneously. Zero improvement over embedding-only approach.

- **Gram eigenstructure as features?** PCA of the Gram matrix in Plücker space
  provides no additive signal over embedding-space PCA/covariance.

## Open questions

- **Where does geometry uniquely contribute?** Generative retrieval (transversals
  rank 1-3 out of 67K) is geometry's strength. Can this be used as a re-ranker
  for the top-100 candidates from the embedding ensemble?

- **Is the Plücker relation a useful constraint?** The quadratic Plücker relation
  `p₀₁p₂₃ - p₀₂p₁₃ + p₀₃p₁₂ = 0` constrains valid lines. Could enforcing this
  during ranking eliminate false positives that embedding methods miss?

- **Cross-word geometry**: The Gram matrix comparison (e.g., dog/cat similarity
  = 0.86) captures relational structure. Can this be used for zero-shot transfer
  of associate patterns between related source words?

- **Sequence prediction**: Can transversals predict next tokens in sequences?
  The geometric constraint (must intersect all context lines) is analogous to
  attention. Worth testing on BPE-tokenized text.
