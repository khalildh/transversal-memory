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

### 3. Online memory accumulation ★ WORKING — first geometry win

**Status: implemented, PPL 206 vs standard 209**

As the model processes text, it accumulates Plücker lines into Gram matrices
(the `M = Σ pᵢ⊗pᵢ` operation) in real time. Later tokens can query these
accumulated memories. Similar to Neural Turing Machines or modern Hopfield
networks, but using geometric incidence instead of dot-product addressing.

**Why it works**: 36 numbers (6×6 Gram) compress the relational structure of
the entire past sequence. Standard attention has no way to compute this — it
sees individual tokens, not structural summaries. The Gram eigenvectors are
emergent "principal relational axes" of the text.

Implementation: `exp_mem_attn.py`, OnlineMemoryAttention class. Memory-efficient
version computes (B,H,T,T) Plücker incidence matrix (same footprint as standard
attention) rather than materializing (B,T,H,6,6) Gram matrices.

#### 3a. Dual-pathway incidence attention ← easiest to test

Instead of collapsing the (B,H,T,T) Plücker incidence matrix to a scalar via
sum-of-squares, use it as a **second attention pathway**. Softmax over the
incidence matrix and use it to route values, just like standard attention does.
Two parallel pathways: Q·K dot-product attention + Plücker incidence attention,
each producing weighted sums of values, combined with a learned gate.

This is the simplest change — the incidence matrix is already computed, we just
need to softmax it and multiply by values instead of squaring and summing.

**Why this should help**: Currently the geometry produces a scalar "how related
is my query to the past?" but throws away *which* past tokens are geometrically
related. The attention pathway preserves this information.

#### 3b. Longer sequences ← TESTED, peaks at 256

4-layer results (10 epochs from scratch):

| Seq len | Standard | Online mem | Delta | % |
|---------|----------|-----------|-------|---|
| 128 | ~244 | ~244 | ~0 | ~0% |
| 256 | 259.1 | **250.8** | **-8.3** | **3.2%** |
| 512 | 282.7 | 279.6 | -3.1 | 1.1% |

2-layer fast screening (10 epochs from scratch, d=128, 4 heads):

| Seq len | Standard | Online mem | Delta | % |
|---------|----------|-----------|-------|---|
| 256 | 432.3 | **421.8** | **-10.5** | **2.4%** |

The advantage peaks at seq=256 then declines at 512. This confirms the 6D
Gram saturation hypothesis: a rank-6, 21-independent-entry matrix can only
encode so much structure. At 256 tokens it captures useful patterns; at 512
it's overwhelmed.

The 2-layer result at seq=256 confirms this isn't a fluke of the 4-layer
architecture — geometry helps at both scales, though more at 4 layers (3.2%
vs 2.4%) where the model has more capacity to refine the geometric signal.

This makes a strong case for higher-dimensional Plücker spaces (idea 4) or
the multi-scale memory with separate fast/slow banks (idea 3f) for longer
sequences.

#### 3c. Tune decay parameter ← TESTED, decay doesn't matter

**Status: tested, all decay values beat standard by ~3.3-3.5%**

Sweep on fast 2-layer model (7 epochs each):

| Decay (λ) | Best PPL | vs Standard (503.4) |
|-----------|----------|---------------------|
| 0.95      | **486.0** | -3.5% (best)       |
| 0.99      | 486.8    | -3.3%               |
| 1.0       | 486.6    | -3.3%               |
| 0.995     | 490.4    | -2.6%               |

**Lesson**: The Gram memory mechanism is robust — decay is not a critical
hyperparameter. The model benefits from geometric structure regardless of
forgetting rate. λ=0.95 is marginally best but all values work.

Longer sequence results (seq=256, fast 2-layer):

| Config | PPL | vs Standard (511.9) |
|--------|-----|---------------------|
| λ=0.95 | 513.3 | +0.3% (worse!) |
| λ=0.99 | 507.6 | -0.8% |

The memory advantage shrank from 3.5% at seq=128 to <1% at seq=256. The 6D
Gram matrix (rank 6, 21 independent entries) likely saturates when
compressing 200+ tokens of relational structure. This motivates both higher-
dimensional Plücker spaces (idea 4) and amplifying memory (3d-ii below).

#### 3d. Amplifying memory (λ > 1) with normalization

**Motivation**: The seq=256 results showed the memory advantage disappearing,
and the decay sweep showed λ doesn't matter much in [0.95, 1.0]. The natural
question: what happens *above* 1.0? If λ < 1 forgets old patterns and λ = 1
weights all patterns equally, then λ > 1 should *amplify* established
patterns — the more a relational direction has been reinforced, the stronger
it becomes.

This reframes the Gram matrix from "running average of relational structure"
to "reinforcement signal for persistent patterns." The topic of a document
*is* the relational structure that persists — recent tokens are noise until
they reinforce the pattern. λ > 1 says "trust what's been confirmed."

**Implementation**: Applied as temporal weights on the incidence matrix
(no loop needed): score_t = Σ_{s<t} λ^(t-s) · incidence(t,s)², with
max-normalization per position when λ > 1 to prevent unbounded scores.

**Status: tested, λ=1.05 hurts at both seq lengths**

| λ | PPL (seq=128) | vs Std (503.4) | PPL (seq=256) | vs Std (511.9) |
|---|---------------|----------------|---------------|----------------|
| 0.95 | **486.0** | **-3.5%** | 513.3 | +0.3% |
| 0.99 | 486.8 | -3.3% | **507.6** | **-0.8%** |
| 1.0 | 486.6 | -3.3% | — | — |
| 1.05 | 496.8 | -1.3% | 513.3 | +0.3% |

**Lesson**: Amplification doesn't help. λ=0.95 and λ=1.05 fail at seq=256
for opposite reasons — fast forgetting loses long-range structure, while
amplification over-weights old patterns and normalization washes out the
signal. λ=0.99 (half-life ≈ 69 tokens) is the sweet spot, but the advantage
still shrinks from 3.3% → 0.8% going 128 → 256 tokens. The real bottleneck
is the 6D Gram space saturating, not the temporal weighting.

#### 3e. Triadic-seeded Gram memory ← TESTED, doesn't help

**Status: fast screening shows +2% improvement (494.8 → 485.2 PPL, 2-layer)**

Two-tier memory: online Gram accumulation (within-sequence) + triadic
associative memory (cross-sequence persistent storage). At end of each
128-token sequence, the accumulated Gram is encoded as an SDR via random
projection and stored as (context_SDR, layer_SDR, gram_SDR) in triadic
memory. At start of next sequence, context from first 16 token embeddings
is used to recall the closest matching Gram, which seeds M₀.

Implementation: `exp_triadic_seed.py`

Fast screening (2-layer, 7 epochs, from scratch):
- Online mem (no seed): PPL 494.8
- Triadic-seeded:       PPL 485.2 (2.0% better)
- Seeded vs non-seeded eval: identical (485.2 both)

Full model (4-layer, 10 epochs, from scratch, bs=64):
- Online mem (no seed): PPL 242.7
- Triadic-seeded:       PPL 244.0 (0.5% worse)
- Seeded vs non-seeded eval: identical (244.0 vs 244.2)
- Overhead: 92s → 128s/epoch as triadic memory grew to 11,640 triples

**Key findings**:
1. Seeding helped the 2-layer model but hurt the 4-layer model
2. The 4-layer model is expressive enough to learn its own structural priors
3. Batch-averaged context + Gram is too coarse — all topics blur together
4. Seeded eval never helps — the learned weights already capture the signal

Per-sequence storage test (2-layer, 7 epochs, from scratch):
- Baseline: PPL 492.7
- Seeded (per-seq, 25% subsample): PPL tracking ~5% better per epoch
  BUT epoch times explode (72s → 454s at 28k triples) as triadic
  memory grows. Per-sequence recall does 128×n_layers queries per batch.
- **Per-seq storage is impractical** without batched triadic queries.

**Optimization ideas to test:**
1. **Per-sequence storage** (not batch-averaged): current approach averages
   context across 128 sequences in a batch, losing topic signal. Store each
   sequence individually for sharper topic recall. Risk: memory grows 128x.
   Mitigation: subsample 25% of sequences.
2. **Contiguous sequence ordering**: WikiText is article-structured. Process
   sequences in order (no shuffle) so consecutive sequences share topic.
   Triadic recall should be strongest when the context actually repeats.
3. **Multi-scale write lines**: Grassmann Flows uses offsets {1,2,4,8,12,16}
   instead of just bigrams (offset=1). This gives the Gram richer structure.
   Easy to add — just loop over offsets in the write line computation.
4. **Larger SDR space**: N=10000, P=100 for better reconstruction fidelity
   (current cos=0.795 is lossy). Trade-off: slower triadic query.
5. **Seed at inference too**: Instead of training-only benefit, build triadic
   memory from training data, then seed during validation. Need separate
   context computation (first 16 tokens of val sequences).
6. **Gradient through seed**: Currently frozen retrieval. Use straight-through
   estimator to let gradients flow through the recalled Gram → teach the
   model to use the seed signal more effectively.
7. **Contrastive seed loss**: Add auxiliary loss that encourages the model's
   online Gram at position T to be close to the recalled seed Gram. This
   regularizes the Gram space to be consistent across related sequences.
8. **Short-sequence test (seq=32)** ← TESTED, no benefit:
   At seq=32 the model barely builds M before the sequence ends, so a
   recalled Gram seed should matter most here. Result: PPL 265.8 seeded
   vs 266.4 baseline (-0.2%, noise). Even with minimal tokens to build
   structure, the model adapts fast enough that cross-sequence memory
   adds no value. This effectively closes the triadic seeding approach —
   it doesn't help at any sequence length (32, 128) or model size (2, 4
   layers). The SDR encode/decode pipeline is too lossy and the model's
   learned projections already handle cold-start efficiently.

#### 3f. Multi-scale memory ← TESTED, promising at seq=256

**Dual decay**: Half heads at λ=0.95 (sentence), half at λ=0.999 (document).
**Learned decay**: Each head learns its own λ via sigmoid(param).

Rapid screening (2-layer, 10% data, ROCm):

**seq=128:**
| Model | vs Standard |
|-------|-------------|
| Online mem (λ=0.99) | -0.6% |
| Dual decay (0.95/0.999) | -0.6% |
| Learned decay | +1.0% (no benefit) |

**seq=256:**
| Model | vs Standard |
|-------|-------------|
| Online mem (λ=0.99) | -1.7% |
| Dual decay (0.95/0.999) | -3.2% |
| **Learned decay** | **-3.5%** |

**seq=512:** Too underfit with 10% data to draw conclusions.

**Comprehensive variant sweep at seq=256, rapid (10% data, 2-layer):**

| Rank | Variant | vs Standard | Key idea |
|------|---------|-------------|----------|
| 1 | **learned_decay** | **-3.5%** | Learned λ per head + additive gate |
| 2 | dual_decay | -3.2% | Fixed fast/slow λ |
| 3 | residual_gram | -2.8% | Multiplicative (1+g)*out |
| 4 | multi_write | -2.3% | 2 write projection pairs |
| 5 | trigram | -1.8% | Trigram write context |
| 6 | online_mem | -1.7% | Single λ=0.99 |
| 7 | abs_inc | -1.4% | |incidence| instead of ² |
| 8 | inc_route | -0.5% | Geometry routes values |
| 9 | learned_power | -0.2% | Learned exponent p |
| 10 | inc_bias | +0.8% | Geometry biases attention |
| 11 | resid_learned | +4.2% | Residual + learned (destructive!) |

**Key findings from variant sweep:**
1. **Additive scalar gate wins**: Geometry as separate gated signal > routing > bias
2. **incidence² is optimal**: Learned power and |incidence| don't improve
3. **Learned per-head decay is the strongest single improvement**
4. **Multiplicative + learned decay is destructive**: Each works alone but
   their combination creates unstable gradient interactions
5. **Wider write context marginal**: Trigrams and multi-write help modestly
6. **J6 matters at seq=256**: Learned decay with identity gets -1.4% vs -3.5%
   with J6. At seq=128 J6 doesn't matter, but at seq=256 the geometric
   structure becomes load-bearing.

**Head count scaling (d=192, 2 layers, seq=256, rapid 10% data):**

| Heads | d_head | Standard | Learned decay | Delta |
|-------|--------|----------|---------------|-------|
| 4 | 32* | ~3184 | ~3074 | -3.5% |
| 6 | 32 | 1730.8 | 1676.6 | -3.1% |
| **8** | **24** | **1777.5** | **1687.5** | **-5.1%** |
| 12 | 16 | 1748.7 | 1693.9 | -3.1% |

*d=128 fast model

8 heads is the sweet spot. At 12 heads (d_head=16), attention quality degrades
and the advantage shrinks. The optimal regime balances d_head≥24 for attention
quality with enough heads for timescale diversity.

**Needs full-data verification at seq=256.**

#### 3g. Gram eigenstructure as features

Instead of scoring read lines against the full Gram matrix, extract the
top-k eigenvectors of M_t and use them as additional "memory keys" that
the standard attention can attend to. This exposes the principal relational
axes directly to the attention mechanism.

More complex to implement (differentiable eigendecomposition needed).

#### 3h. Multi-scale write lines (from Grassmann Flows) ← TESTED, marginal

Instead of only pairing consecutive tokens (offset=1), pair at offsets
{1, 2, 4, 8}. Each offset produces write lines that capture structure
at different timescales. All offsets accumulate into the same Gram.
This is the one useful idea from the Grassmann Flows paper (Dec 2025).

Implementation: `exp_fast.py`, MultiScaleMemoryAttention class.

Results (from scratch):
| Model | 2-layer PPL | 4-layer PPL |
|-------|-------------|-------------|
| Standard | 504.2 | ~244 |
| Online mem (offset=1) | 492.7 | 243.7 |
| Multi-scale ({1,2,4,8}) | 489.1 | 243.4 |

**Finding**: Multi-scale helps at 2 layers (-3.6 PPL) but is negligible at
4 layers (-0.3 PPL). The deeper model already captures multi-scale
structure through its multiple attention layers. Not worth the ~18%
computational overhead for the full model.

Overhead: ~8s/epoch (2-layer), ~16s/epoch (4-layer) — modest.

#### 3i. Dual-pathway incidence attention ← TESTED, negative

**Status: tested, PPL 372 from scratch (1.8x worse than standard)**

Used the (B,H,T,T) Plücker incidence matrix as a second attention pathway
alongside standard Q·K. Each pathway softmaxes its logits and routes values
independently, combined with a learned gate: `(1-g)*std_out + g*geo_out`.

Results: from scratch PPL 372 vs standard ~206 (1.8x worse). From checkpoint
init PPL 209 (matches baseline but geometry isn't helping — the gate likely
learns to suppress the geometric pathway).

**Why it failed**: When geometry competes with standard attention for value
routing, it loses. Standard Q·K is a much better attention pattern for LM.
The successful approach (online memory scalar gate) works because it ADDS
geometric information on top of standard attention rather than replacing any
part of the attention computation.

**Lesson**: Geometry should augment, not compete. Scalar gating > dual pathway.

#### 3j. Online associative memory ← TESTING

**Status: running fast 2-layer screening**

Instead of seeding M₀ once at sequence start (which doesn't help at any
sequence length or model size), give the model a persistent associative
memory that it reads/writes at every position during the forward pass.

Architecture:
- Keys: Plücker write lines (6D per head, flattened to 36D per position)
- Values: hidden states (d_model per position)
- Storage: write every 8 positions during training (controls memory growth)
- Retrieval: batched GPU queries at every position via matrix multiply
  against indicator matrices. All B×T queries computed in one matmul.
- Integration: recalled values → learned linear projection → scalar gate
  (starts small at 0.01, model learns how much to trust the memory)

Key differences from triadic seeding (3e):
1. Query at every position, not just sequence start
2. Keys are geometric (write lines), not semantic (context embeddings)
3. Values are hidden states, not Gram matrices
4. Batched GPU queries instead of one-at-a-time CPU queries
5. The memory is truly associative — "what have I seen before that looks
   geometrically like what I'm seeing now?"

Implementation: `exp_assoc_mem.py`

Concern: double forward pass per batch (one no-grad to get write lines
for queries, then the real forward with recalled values). This doubles
compute when reading is active. Could optimize by caching write lines
from the previous batch instead.

### 4. Higher Grassmannian attention ← TESTED, negative (fast screening)

Use G(2,6) with 15D Plücker coordinates instead of G(2,4) with 6D. The higher
Grassmannian provides more geometric room for discrimination (demonstrated by
the 2.6x improvement in generation quality from G(2,4) to G(2,6)). The
attention mechanism would use 15D incidence instead of 6D.

Trade-off: more parameters per head (15D vs 6D coordinates), but potentially
much richer geometric interactions.

**Fast 2-layer screening (10 epochs, from scratch, ROCm AMD 8060S):**

| Model | Best PPL | Params | J matrix |
|-------|----------|--------|----------|
| Standard (no geometry) | 403.9 | 6,846,080 | — |
| Online mem G(2,4) | **400.5** | 6,896,528 | J6 (Hodge) |
| Online mem G(2,6) | 403.7 | 6,904,720 | Identity |

| Online mem G(2,4) no-J6 | **400.8** | 6,896,528 | Identity |

**Finding**: J6 is NOT the key ingredient (400.5 vs 400.8, noise-level
difference). The exterior product structure itself provides the geometric
signal — the specific bilinear form (J6 vs identity) is irrelevant.

However, G(2,6) with 15D coordinates (403.7) matches standard (403.9)
and loses the G(2,4) advantage. Higher dimensionality actively hurts.

**Why G(2,4) works but G(2,6) doesn't**: The 4D→6D exterior product is a
compact nonlinear feature map (quadratic interactions of 4 coordinates =
6 features). Going to 6D→15D creates 15 features from 6 coordinates,
but the extra dimensions are redundant — the model only has 4 heads with
32 dims each to learn structure. The 6D sweet spot balances expressivity
with learnability.

Full sweep (fast 2-layer, 10 epochs, ROCm):

| Model | Best PPL | Plücker dim |
|-------|----------|-------------|
| Standard | 403.9 | — |
| G(2,4) + J6 | **400.5** | 6D |
| G(2,4) + identity | 400.8 | 6D |
| G(2,5) + identity | 401.0 | 10D |
| G(2,6) + identity | 403.7 | 15D |

**Lesson**: G(2,4) is the sweet spot. Higher Grassmannians dilute the
signal. The geometry's value is in the exterior product as a compact
nonlinear feature map (4 coords → 6 features), not in the Plücker inner
product or higher-dimensional Grassmannian structure.

## Active: Retrieval and reasoning

### 5. Few-shot relational reasoning ← TESTED, negative with neural embeddings

Tested with 384D MiniLM sentence embeddings across 5 relations (country→capital,
animal→sound, material→product, country→language, profession→tool), 25 pairs each.

Results: geometry loses at ALL K values. Best geometry (Gram^0.05 on PCA-16D)
reaches p@10=0.84 vs cosine centroid p@10=0.91. Offset-based Gram (encoding
relational transformation tgt-src) is even worse (p@10=0.43).

Root cause: 384D neural embeddings already capture relational semantics so well
that cosine similarity achieves >0.9 precision. Geometry can only lose information
through the 384D→4D projection. The K≥10 crossover from the word association
benchmark only occurs with weak embeddings (32D PPMI+SVD).

**The geometry's niche is weak embeddings + medium data.** When embeddings are
information-poor (statistical, low-D, non-neural), Plücker projection extracts
complementary structure. With modern sentence transformers, it's strictly dominated.

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
| Standard | Q·K dot product | 209 | baseline |
| **Online memory** | **Q·K + causal Gram accumulation** | **206** | **0.99x (wins!)** |
| Hybrid v4 | Q·K + learnable Plücker bias | 207 | 1.00x (tied) |
| Bigram v3 | Token-pair query lines, single-token key lines | 215 | 1.03x worse |
| Kernel v2 | Asymmetric Q/K line projections | 252 | 1.21x worse |
| Original v1 | Symmetric -log|incidence| | 2063 | 10.0x worse |

### Key findings

- Plücker geometry **cannot replace** dot-product attention but **can augment** it
- **Online Gram accumulation is the first variant to beat standard attention** (PPL 206 vs 209)
  - Causal M_t = Σ_{s<t} decay^{t-s} · p_s⊗p_s creates differentiable growing memory
  - Only 56,652 memory-specific parameters (0.5% overhead)
  - Bigram write lines + separate read projections + sigmoid gating
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

**Few-shot relational reasoning** ← NEGATIVE with neural embeddings
Tested: geometry loses at all K values when using 384D MiniLM embeddings.
Cosine centroid p@10=0.91 vs best geometry p@10=0.84. The K≥10 crossover
only applies with weak (32D PPMI+SVD) embeddings. See §5 above.

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

- **Geometry only helps with weak embeddings**: Can geometry add to neural embedding
  systems in any task, or is the K≥10 crossover specific to co-occurrence vectors?
- **Optimal K regime**: Can the Gram ensemble be made more sample-efficient?
- **Higher-order interactions**: Non-linear kernels or tensor decompositions
  in 32D without lossy projection to capture useful higher-order interactions?
- **Sequence prediction via transversals**: Can geometric constraints predict
  next tokens in BPE-tokenized text?
