# Triadic Gram Memory

A two-tier memory architecture for transformers that combines **online Gram accumulation** (within-sequence relational structure) with **triadic associative memory** (cross-sequence persistent storage). The Gram matrix captures geometric structure; triadic memory stores and recalls it when similar context reappears.

## Architecture overview

```
Sequence N                          Sequence N+K (similar topic)
┌──────────────────────┐            ┌──────────────────────┐
│ Tokens → Transformer │            │ Tokens → Transformer │
│                      │            │                      │
│ Gram accumulates:    │            │ Gram SEEDED from     │
│ M₁ → M₂ → ... → M_T │──STORE──→│ recalled memory:     │
│         ▓▓▓▓▓▓▓▓▓▓▓▓ │    │      │ M₀ = recalled Gram   │
└──────────────────────┘    │      │ M₁ = λM₀ + p₁⊗p₁    │
                            │      │      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
                            ▼      └──────────▲───────────┘
                   ┌──────────────┐           │
                   │   TRIADIC    │───RECALL───┘
                   │   MEMORY     │
                   │  (persistent)│
                   └──────────────┘
```

The online Gram memory (Tier 1) operates within a single sequence: as tokens are processed, consecutive pairs are encoded as Plucker lines and accumulated into a 6x6 Gram matrix per attention head. This matrix compresses the relational structure of the entire past sequence into 36 numbers.

Triadic memory (Tier 2) operates across sequences: at the end of each sequence, the Gram state is encoded as a Sparse Distributed Representation (SDR) and stored alongside the context. When a new sequence begins with similar context, the stored Gram is recalled and used to seed the online memory — so the model starts with structural knowledge instead of zeros.

## The three components of a triple

Triadic memory stores and recalls triples `(x, y, z)`. Given any two elements, it can recall the third. For Gram memory, the three elements are:

### 1. Context SDR — "what am I reading?"

The mean-pooled hidden state of the transformer summarizes the current text as a 192-dimensional vector. This is projected to an SDR via a fixed random matrix.

```
mean(hidden_states) → [0.3, -0.1, 0.7, ..., 0.2]  (192-dim)
                              │
                        R_ctx (192 × 5000)  fixed random matrix
                              │
                      [1.4, -0.8, ..., 2.7]  (5000-dim projection)
                              │
                        top-50 indices
                              │
                      {23, 147, 382, ...}  context SDR
```

Articles about the same topic produce similar hidden states, which produce similar SDRs with high bit overlap. This is what enables recall: a new war article's context SDR overlaps with a stored war article's context SDR.

### 2. Layer SDR — "which level of abstraction?"

Four fixed SDRs, one per transformer layer. These never change — they act as keys that separate layer-0 Grams from layer-3 Grams within the same triadic memory.

```
Layer 0: {12, 89, 234, 567, ...}   (50 random indices, generated once)
Layer 1: {45, 178, 401, 823, ...}
Layer 2: {3, 156, 299, 741, ...}
Layer 3: {67, 201, 488, 912, ...}
```

Different layers capture different levels of abstraction. Layer 0 might encode syntactic patterns while layer 3 encodes semantic structure. Storing them separately allows layer-specific recall.

### 3. Gram SDR — "what relational structure does the text have?"

The accumulated Gram matrices from all attention heads are flattened into a single vector and projected to an SDR.

```
Head 0: M (6×6) = 36 floats ─┐
Head 1: M (6×6) = 36 floats  │
Head 2: M (6×6) = 36 floats  ├─→ flatten → [216 floats]
Head 3: M (6×6) = 36 floats  │         │
Head 4: M (6×6) = 36 floats  │   R_gram (216 × 5000)
Head 5: M (6×6) = 36 floats ─┘         │
                                  top-50 indices
                                        │
                                  {91, 274, 503, ...}  gram SDR
```

The Gram matrix eigenvectors encode the "principal relational axes" of the text. A war article might have dominant axes for conflict, chronology, and geography. A science article might have axes for causation, classification, and measurement. The SDR encoding preserves these structural differences.

## SDR encoding via random projection

The bridge between continuous Gram matrices (36 floats per head) and binary SDRs (50 indices in 5000) is a fixed random projection matrix.

### Encoding (Gram to SDR)

```
gram_vector (216-dim) × R (216 × 5000) = projection (5000-dim)
                                              │
                                        argsort, take top 50
                                              │
                                        SDR: 50 indices
```

The Johnson-Lindenstrauss property guarantees that random projections preserve distances: similar Gram vectors produce similar projections, which share many top-50 indices. Measured overlap:

- Same-topic Gram SDRs: 20.4 +/- 2.9 shared bits
- Cross-topic Gram SDRs: 3.0 +/- 2.5 shared bits
- Gap: 17.4 bits (recall needs roughly 5+ to work reliably)

### Decoding (SDR to Gram)

```
SDR: {91, 274, 503, ...}
          │
    binary vector b (5000-dim, 1s at 50 positions)
          │
    b × R_pinv (5000 × 216)   pseudoinverse, precomputed once
          │
    approximate gram_vector (216-dim)
          │
    reshape to 6 × (6×6) per-head Gram matrices
```

Reconstruction is lossy (cosine similarity ~0.795 with the original), but preserves the dominant eigenvectors. The recalled Gram does not need to be exact — it only needs to point the model in the right structural direction.

## How triadic memory works

Triadic memory is a 3D associative store operating on SDRs. All three elements (context, layer, gram) live in the same N=5000 dimensional space with P=50 active bits each.

### Store

For a triple (x, y, z), set `memory[x_i, y_j, z_k] = 1` for every combination of active bits:

```
x has 50 active bits
y has 50 active bits    →  50 × 50 × 50 = 125,000 cells set per triple
z has 50 active bits
```

### Recall

Given (x, y, ?), accumulate votes for z:

```
For each active bit x_i in x:
  For each active bit y_j in y:
    votes[k] += memory[x_i, y_j, k]   for all k in [0, 5000)
```

Each true z-bit receives 50 × 50 = 2,500 votes. Random bits receive 0-5 votes. Taking the top-50 indices from the vote vector recovers the original z SDR.

### Why this works for Gram recall

When a new sequence has a similar context SDR (many shared bits with a stored context), the voting mechanism naturally recalls the associated Gram SDR. The overlap does not need to be perfect — even 30/50 shared context bits produce enough votes to dominate the noise.

The layer SDR acts as a filter: querying with the wrong layer SDR produces near-zero votes because the stored layer bits don't overlap. In testing, wrong-layer queries return no result at all.

## Lifecycle

### Training — store phase

At the end of each 128-token sequence:

1. Compute `context = mean(hidden_states)` across all positions
2. Encode context as SDR via `R_ctx`
3. For each transformer layer `l`:
   - Flatten the accumulated Gram matrices from all heads (216 floats)
   - Encode as SDR via `R_gram`
   - Store triple: `triadic.store(context_SDR, layer_SDR[l], gram_SDR)`

### Inference — recall phase

At the start of each new sequence:

1. Process the first 16 tokens with a quick forward pass
2. Compute `context = mean(hidden_states[:16])`
3. Encode context as SDR via `R_ctx`
4. For each transformer layer `l`:
   - Query: `triadic.query(context_SDR, layer_SDR[l], ?) → recalled_gram_SDR`
   - Decode recalled SDR back to 216 floats via `R_pinv`
   - Reshape to 6×6 Gram matrices per head
   - Seed the online Gram memory: `M_0 = recalled_Gram`
5. Continue normal processing: `M_t = λ · M_{t-1} + p_t ⊗ p_t`

The model now starts each sequence with a structural prior from past experience instead of a blank memory.

## What the Gram eigenvectors represent

The Gram matrix `M = sum(p_i outer p_i)` accumulates Plucker line outer products. Its eigendecomposition `M = V Lambda V^T` reveals the principal relational axes of the text:

- The eigenvectors V define directions in 6D Plucker space that concentrate the most relational structure
- The eigenvalues Lambda indicate how much structure exists along each axis
- Different topics produce different eigenvector patterns

When a recalled Gram seeds the online memory, it provides these principal axes immediately — the model doesn't need to rediscover them from token patterns. This is analogous to how recognizing a genre (war story, scientific paper, dialogue) immediately activates expectations about relational structure.

## Properties

**No learned parameters in the storage pipeline.** The random projection matrices R_ctx and R_gram are fixed (seeded deterministically). The layer SDRs are fixed. The triadic memory is TDGA's standard implementation. Only the Gram projection weights within the transformer (W1_write, W2_write, W1_read, W2_read) are learned, and those are already trained as part of the online memory attention.

**Constant-time recall.** Triadic memory recall is O(P^2 * N) regardless of how many triples are stored. With P=50 and N=5000, that is 12.5 million operations — roughly 0.1ms on modern hardware.

**Compression ratio.** A 128-token sequence with 192-dim hidden states = 24,576 floats of state. The Gram summary = 216 floats (6 heads × 36). The SDR = 50 integers. Compression: 24,576 → 50 indices, a 490x reduction. Yet the SDR preserves enough structure for 100% topic identification in testing.

**Graceful degradation.** If the recalled Gram doesn't match the current text well, the decay factor λ=0.99 ensures it fades within ~70 tokens, replaced by the actual Gram accumulation. A bad recall is self-correcting.

## Test results

Validated with synthetic data: 5 topics, 10 samples each, 6 attention heads.

| Metric | Result |
|--------|--------|
| SDR same-topic overlap | 20.4 +/- 2.9 bits |
| SDR cross-topic overlap | 3.0 +/- 2.5 bits |
| Overlap gap | 17.4 bits |
| Gram reconstruction cosine | 0.795 |
| Triadic recall accuracy | 10/10 (100%) |
| Recalled Gram same-topic cosine | 0.770 |
| Recalled Gram cross-topic cosine | 0.318 |
| Wrong-layer filtering | Perfect (no recall) |

## Related files

- `exp_mem_attn.py` — Online Gram memory attention (Tier 1). PPL 206 vs standard 209.
- `exp_triadic_gram.py` — Triadic Gram storage/recall test (Tier 2). 100% topic recall.
- `transversal_memory/plucker.py` — Plucker line encoding, exterior products, Hodge dual.
- `associative_db.py` — Unified database combining TDGA triadic + geometric signatures.

## Open questions

- Does Gram seeding improve PPL on WikiText-2? The synthetic test validates the pipeline, but the end-to-end LM experiment hasn't been run yet.
- How many stored triples before triadic memory saturates? With N=5000, P=50, theoretical capacity is ~10,000 triples before cross-talk becomes significant.
- Should the recalled Gram be scaled down (e.g., multiply by 0.5) to avoid over-committing to the recalled structure before seeing the actual text?
- Multi-scale: store Grams at different decay rates (sentence-level λ=0.95 and document-level λ=0.999) as separate triples?
