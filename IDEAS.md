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
- **7-signal RRF (+reciprocal NN): p@10 = 0.114 (+53.4%)**

## Open questions

- Does higher Grassmannian (G(2,6)) help the hybrid approach?
- Can we ensemble across multiple Grassmannian dimensions?
- Is there a principled way to set alpha (not just grid search)?
- What about using the Gram matrix eigenstructure as additional features?
