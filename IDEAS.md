# Ideas

## Active: Memory-integrated attention

### 1. Memory-augmented attention ← building now

Add an external Gram memory bank to a transformer with hybrid Plücker attention.
Each attention head computes standard Q·K over the sequence *plus* scores each
query against stored Gram matrices using the Plücker inner product. The query
line from token i gets scored against the memory's stored lines — same `p·(★q)`
operation the memory already uses. Output is a weighted combination of sequence
values and memory-retrieved values.

This is a direct extension of hybrid v4 — instead of the Plücker bias coming
only from other tokens in the sequence, it also comes from an external Gram
matrix storing relational knowledge.

**Why it works**: The Plücker inner product is the same operation in both the
memory system and the attention mechanism. No adapter needed.

### 2. Transversal retrieval as context injection

Given a prompt, use the generative mode (4 lines → 2 transversals) to retrieve
related concepts from the Gram memory, then prepend them as additional context
tokens. The transformer processes them alongside the input.

Simpler than option 1 — doesn't require modifying the attention mechanism at all.
Just uses memory as a retrieval augmentation step before inference.

### 3. Online memory accumulation ★ most interesting

As the model processes text, it accumulates Plücker lines into Gram matrices
(the `M = Σ pᵢ⊗pᵢ` operation) in real time. Later tokens can query these
accumulated memories. Similar to Neural Turing Machines or modern Hopfield
networks, but using geometric incidence instead of dot-product addressing.

**Why this is the most interesting**: It creates a differentiable, growing
memory that captures relational structure (not just content) and can be queried
geometrically. The Gram matrix eigenvectors become "principal relational axes"
that emerge during processing — the model discovers what the text is *about*
as it reads.

Key design questions:
- Write head: which token pairs get stored as Plücker lines?
- Read head: how does the current token query the accumulated Gram matrix?
- Forgetting: does M grow forever, or is there a decay/gating mechanism?
- Multi-scale: separate fast (sentence) and slow (document) Gram memories?

### 4. Higher Grassmannian attention

Use G(2,6) with 15D Plücker coordinates instead of G(2,4) with 6D. The higher
Grassmannian provides more geometric room for discrimination (demonstrated by
the 2.6x improvement in generation quality from G(2,4) to G(2,6)). The
attention mechanism would use 15D incidence instead of 6D.

Trade-off: more parameters per head (15D vs 6D coordinates), but potentially
much richer geometric interactions.

## Active: Retrieval and reasoning

### 5. Few-shot relational reasoning

The geometric system has a demonstrated crossover point at K≥10 seed associates
where it outperforms embeddings. Build an interface for few-shot relational
tasks: given 10-20 examples of a relation (e.g., country→capital), the Gram
matrix captures the relational pattern and can score new candidates.

Unique advantage: works with the existing PPMI+SVD embeddings, no neural
network training needed. The geometry does the heavy lifting.

### 6. Compositional transversal chains

Chain multiple transversal operations: output of one query becomes input to
the next. This is the geometric analogue of multi-hop reasoning. Already
demonstrated with music→bassist→multitrack→arpeggio chains.

Could be formalized as a graph traversal where each step uses the
multi-transversal generation to propose next nodes, and the Gram energy
scoring to filter.

### 7. Cross-word geometry transfer

The Gram matrix comparison (e.g., dog/cat similarity = 0.86) captures relational
structure. Can this be used for zero-shot transfer of associate patterns between
related source words? If dog and cat have similar Gram matrices, can we use
dog's relational structure to predict cat's associates?

## Plücker attention LM experiment results

### Results summary

| Variant | Architecture | Best PPL | vs Standard |
|---------|-------------|:--------:|:-----------:|
| Standard | Q·K dot product | **208** | baseline |
| Hybrid v4 | Q·K + learnable Plücker bias | **207** | 1.00x (tied) |
| Bigram v3 | Token-pair query lines, single-token key lines | 215 | 1.03x worse |
| Kernel v2 | Asymmetric Q/K line projections | 252 | 1.21x worse |
| Original v1 | Symmetric -log|incidence| | 2063 | 10.0x worse |

### Key findings

- Plücker geometry **cannot replace** dot-product attention but **can augment** it
- Degree-4 interactions help when they encode context (bigram > kernel)
- The original failure was architectural (symmetry, -log|.|, expressivity), not mathematical
- Learned Plücker projections ≈ random projections for retrieval (geometry is decorative for discriminative tasks)
- Hybrid approach matches standard — geometric incidence captures complementary structure

### What went wrong (v1) and fixes

1. **Symmetric Q/K**: attn(i,j) = attn(j,i) → fixed with separate projections
2. **-log|incidence| scoring**: spiky gradients → fixed with signed inner product
3. **Single-token lines**: too little expressivity → fixed with bigram encoding

## Resolved questions

- **Does higher Grassmannian help the RRF ensemble?** No. G(2,4) through G(2,17)
  all add zero unique information when full 32D embedding covariance is exploited.

- **Can Plücker incidence serve as a continuous similarity signal?** No. Quadratic
  in projected embeddings, dominated by embedding covariance (Mahalanobis).

- **Can an optimized Gram^0.05 multi-seed ensemble add to the RRF?** No.
  Despite reaching p@10=0.1065 standalone (9.7x improvement), it's entirely
  subsumed by embedding-space covariance signals.

- **Signal pruning?** Yes. Removing redundant signals improved p@10 from 0.123
  to 0.128 (+4%). Fewer signals = less dilution in rank fusion.

- **Mahalanobis-cent is the strongest signal** (-0.003 when ablated).

- **Alternative fusion methods?** All degrade vs standard RRF with uniform K.

## Geometry beats embeddings at K≥10

Multi-seed Gram^0.05 outperforms all embedding methods when given ≥10 seed
associates. Crossover at K≈7-10 because 6D Plücker needs fewer samples to
estimate structure than 32D Mahalanobis. This is the system's unique advantage.

## Unified associative database (TDGA + transversal geometry)

Working prototype in `associative_db.py`. Combines TDGA's exact triadic recall
with multi-seed Gram^0.05 geometric similarity and generative expansion.

### Demonstrated
- Exact triadic recall: "Paris capital_of ?" → France (50/50 overlap)
- Geometric subject similarity: France↔Germany 0.54 (shared relational patterns)
- Generative expansion: Paris → Spain, French, Germany, Italy (consistent patterns)
- Cosine search: standard semantic retrieval over stored facts

### Directions

**Few-shot relational reasoning** ← most validated
Feed 10-20 examples of a relation, geometry outperforms embeddings at scoring
new candidates. No training needed — Gram matrices capture relational patterns
from examples alone. This is the K≥10 crossover applied to structured knowledge.

**Knowledge graph completion**
The expansion operation scores unstored (subject, object) pairs by geometric
consistency. This is link prediction without training a GNN — score candidate
triples against accumulated Gram signatures.

**Memory layer for LLM agents**
Triadic recall is O(P²·N) regardless of corpus size. An agent stores facts as
(entity, relation, value) triples with constant-time recall. Geometry adds
"what else might be true?" for reasoning beyond stored facts.

**Incremental knowledge accumulation**
Both layers support one-shot storage, no retraining. Geometric signatures
become more discriminative as more facts arrive (more lines → richer Gram).

**Anomaly / novelty detection**
Score new candidate facts against existing geometric patterns. Low score against
all subject signatures = genuinely novel fact. High score but unstored =
plausible inference.

### Limitations
- Subjects with <2 facts get zero geometric signatures (need ≥2 lines for Gram)
- Requires TDGA checkpoint + sentence-transformers (external dependencies)
- 384D→4D projection is lossy; geometry works because ensemble (50 seeds) recovers info

## Open questions

- **Optimal K regime**: Can the Gram ensemble be made more sample-efficient?
- **Higher-order interactions**: Non-linear kernels or tensor decompositions
  in 32D without lossy projection to capture useful higher-order interactions?
- **Sequence prediction via transversals**: Can geometric constraints predict
  next tokens in BPE-tokenized text?
