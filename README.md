# Transversal Memory

A content-addressable memory system based on projective geometry, specifically
the transversal problem in Schubert calculus: *given 4 lines in P³ in general
position, exactly 2 lines meet all four*.

This repository grew out of noticing a structural correspondence between
Peter Overmann's Triadic Memory (2022) and Schubert calculus on the
Grassmannian G(2,4).

---

## The core idea

In Overmann's triadic memory, items are sparse binary hypervectors and
storage is accumulation into a weight matrix. Retrieval is a dot-product
threshold. The hidden layer encodes all pairwise combinations of input bits —
structurally identical to the Plücker embedding of a line in P³.

This library takes that correspondence literally:

- **Items** are vectors in Rⁿ
- **Relations** are lines in P³, encoded as Plücker 6-vectors (exterior products)
- **Storage** is accumulation into a 6×6 Gram matrix M = Σ p⊗p
- **Retrieval** has two modes:
  - **Generative** (transversal): given 3 stored lines + 1 query → 2 new lines via exact Plücker solve
  - **Discriminative** (energy scoring): given a full M → score any candidate line

The exact Plücker solver is new: given a 2D null space (v1, v2), it finds
T = t·v1 + v2 satisfying all Plücker relations by collecting the quadratic
coefficients from each relation, finding the consensus via PCA, and solving
with the quadratic formula. No grid search, no gradient descent.

---

## Results on the full Overmann dataset

Using the [Overmann WordAssociations](https://github.com/PeterOvermann/WordAssociations)
dataset (64,823 source words, 1.88M associations), the system builds its own
embeddings from the association norms via PPMI + sparse truncated SVD — no
external corpus or pre-trained embeddings required.

### Discriminative scoring (GramMemory)

Train on 75% of each word's associates, test on the held-out 25%.

**Batch evaluation (500 words):** held-out associates score higher than
random non-associates **94.6%** of the time (baseline = 50%).

Per-word examples:

| Source | Train | Held-out | Held-out mean | Non-assoc mean | Separated |
|--------|------:|---------:|--------------:|---------------:|-----------|
| dog    |    34 |       12 |        0.3321 |         0.2874 | ✓         |
| music  |    65 |       22 |        0.3604 |         0.3039 | ✓         |
| ocean  |    48 |       16 |        0.3628 |         0.2769 | ✓         |
| king   |    32 |       11 |        0.3946 |         0.3333 | ✓         |
| brain  |    30 |       10 |        0.3956 |         0.3049 | ✓         |

### Principal relational axes

The eigenvectors of the Gram matrix reveal semantically coherent axes:

```
king:
  axis 1: chivalry, reign, authority, succession, monarch, ruler
  axis 2: queen, realm, regal, empire, throne, kingdom
  axis 3: heir, noble, ceremony, court, coronation, sovereign

brain:
  axis 1: perception, cognition, consciousness, neuroplasticity, cells, thought
  axis 2: think, region, structure, health, tumor, disorder
  axis 3: amygdala, hippocampus, waves, imaging, mind, brainstem

fire:
  axis 1: char, engine, blaze, fighter, wildfire, spark
  axis 2: escape, flame, alarm, hazard, burn, torch
  axis 3: smoke, ember, department, burn, safety, extinguisher
```

### Cross-word relational similarity

Cosine similarity between Gram matrices measures how similar two words'
*relational patterns* are (not lexical similarity):

```
Animals:       dog    cat  horse   bird
  dog        1.000  0.860  0.537  0.663
  cat        0.860  1.000  0.582  0.673
  horse      0.537  0.582  1.000  0.430
  bird       0.663  0.673  0.430  1.000

Emotions:    anger   fear sadness    joy
  anger      1.000  0.901   0.800  0.613
  fear       0.901  1.000   0.904  0.571
  sadness    0.800  0.904   1.000  0.451
  joy        0.613  0.571   0.451  1.000
```

Dog/cat are most similar (0.86); joy is the outlier among emotions (0.45–0.61).

### Nearest neighbours (SVD embeddings)

The PPMI + SVD embeddings themselves are semantically coherent:

```
dog:
  source (cue):   pooch(0.96), dogs(0.95), doggy(0.94), mutt(0.94), beagle(0.92)
  target (assoc): puppy(0.93), cat(0.91), hound(0.90), breed(0.90)

king:
  source (cue):   empress(0.99), throne(0.99), kings(0.99), monarch(0.99)
  target (assoc): queen(0.98), prince(0.97)
```

### Generative retrieval (P3Memory)

Store 3 relational lines, query with a 4th, decode the transversal back
to vocabulary. Uses **dual projection** encoding (see
[COPUNCTAL_FIX.md](COPUNCTAL_FIX.md) for why this is necessary).

| Query | Stored | Target | Rank / 67K |
|-------|--------|--------|-----------|
| king  | crown, throne, royal | queen | **1** |
| fire  | flame, heat, burn | smoke | **1** |
| ocean | waves, deep, salt | fish | **2** |
| love  | heart, romance, passion | marriage | **2** |
| music | rhythm, melody, harmony | instrument | **3** |
| brain | neurons, memory, cortex | intelligence | **3** |

Target word ranks 1–3 out of 67,378 candidates for 6/8 test queries.
Decoding uses the Plücker inner product: `|⟨T, L⟩| → 0` means the
candidate line L is incident with the transversal T.

### Continuous generation (multi-transversal)

A single transversal provides only 1 scalar constraint — not enough to
discriminate 67K candidates (accidental near-zeros dominate). The fix:
sample many different 4-tuples from a word's associates, compute a
transversal from each, and rank candidates by their **combined** score
across all transversals.

With 20–30 transversals, the generated words are semantically coherent —
**10.2x more similar** to the source concept than random baseline,
averaged across 20 test words:

| Concept  | Similarity ratio | Top generated words |
|----------|:----------------:|---------------------|
| tree     | 93x | bud, botany, stalks, inflorescence, leaves, bristlecone, thorns |
| guitar   | 47x | musicians, violin, multitrack, albums, ensembles, flute, bassist |
| music    | 31x | bassist, trumpeter, fingerboard, funk, ensembles, saxophonist |
| mountain | 18x | boulder, abyss, ranges, alp, horizon, boulders, crevasse |
| war      | 14x | gunnery, warfare, competitive, besieger, offensives |
| doctor   | 12x | psychiatry, prevention, hypertension, outpatient, psychosurgery |
| king     | 11x | maharani, empress, duchess, telecommunication, licence |
| food     | 10x | scones, cayenne, pork, capsaicin, pepper, noodle, parmesan |

Chained generation (generate a word, add it to the pool if quality is
high enough, re-sample transversals) produces 15+ consecutive on-topic
words for strong concepts:

```
music:  bassist → multitrack → arpeggio → cantata → amplifiers →
        drummer → recital → singers → fretboard → funk → etude → ...

doctor: psychiatry → chiropody → non-addictive → infirmary →
        inpatient → pneumonectomy → otc → hypodermic → ...
```

---

## How the pipeline works

```
Association Norms          PPMI Matrix              SVD Embeddings
┌──────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│ dog →             │    │         puppy cat │    │                  │
│   [puppy, bark,   │───▶│ dog   [0.82 0.41]│───▶│  C ≈ U · √Σ · Vᵀ│
│    fetch, ...]    │    │ cat   [0.31 0.96]│    │                  │
│ cat →             │    │                   │    │  U = source vecs │
│   [kitten, ...]   │    │ filters generics  │    │  V = target vecs │
└──────────────────┘    └───────────────────┘    └──────────────────┘

   Plücker Embedding                    Two Retrieval Modes
┌──────────────────────┐    ┌─────────────────┬──────────────────────┐
│ For "dog → puppy":   │    │ GramMemory      │ P3Memory             │
│                      │    │ (single proj)   │ (dual projection)    │
│ a = U[dog]    ∈ R³²  │    │                 │                      │
│ b = V[puppy]  ∈ R³²  │    │ p = Wa ∧ Wb     │ p = W1[a;b] ∧ W2[a;b]│
│                      │───▶│ M = Σ pᵢ⊗pᵢ    │ Store 3 + query 1    │
│                      │    │ score = cᵀMc   │ → 2 transversals     │
│ (Plücker 6-vector)   │    │                 │                      │
│                      │    │ 94.6% held-out  │ target ranks 1-3     │
│                      │    │ separation      │ out of 67K vocab     │
└──────────────────────┘    └─────────────────┴──────────────────────┘
```

Key design choices:
- **PPMI** (positive pointwise mutual information) suppresses high-frequency
  generic associates that co-occur with everything
- **Separate U/V vectors** from SVD preserve the directionality of associations
  (U = cue role, V = associate role)
- **Dual projection** for generative retrieval — both endpoints of the Plücker
  line depend on both source and target, avoiding co-punctal degeneracy
  (see [COPUNCTAL_FIX.md](COPUNCTAL_FIX.md))
- **Sparse matrix + truncated SVD** (scipy) handles 67K×67K vocabulary
  in seconds with minimal memory
- **Pickle checkpointing** caches the embeddings so subsequent runs are instant

---

## Mathematical background

A line in P³ is represented by its Plücker coordinates: a 6-vector
p = (p₀₁, p₀₂, p₀₃, p₁₂, p₁₃, p₂₃) where p_ij = a_i·b_j - a_j·b_i
for two points a, b on the line. This is the exterior product a∧b.

Two lines meet iff their Plücker inner product vanishes:
  ⟨p, q⟩ = p₀₁q₂₃ - p₀₂q₁₃ + p₀₃q₁₂ + p₁₂q₀₃ - p₁₃q₀₂ + p₂₃q₀₁ = 0

A valid line satisfies the Plücker relation: p₀₁p₂₃ - p₀₂p₁₃ + p₀₃p₁₂ = 0

The space of all lines in P³ is the Grassmannian G(2,4), embedded in P⁵
via the Plücker map. Schubert calculus on G(2,4) gives the classic result:
4 lines in general position → exactly 2 transversals.

### Key equations

| Equation | Description |
|----------|-------------|
| p = a ∧ b | Plücker embedding of line through points a, b |
| p₀₁p₂₃ − p₀₂p₁₃ + p₀₃p₁₂ = 0 | Plücker relation (defines G(2,4) ⊂ P⁵) |
| ⟨p, q⟩ = p · (★q) = 0 | Incidence: lines p, q meet (★ = Hodge dual) |
| A = [★L₁; ★L₂; ★L₃; ★L₄] | Constraint matrix, null(A) = span{v₁, v₂} |
| αt² + βt + γ = 0 | Plücker quadratic for T = t·v₁ + v₂ |
| M = Σᵢ pᵢpᵢᵀ, score(c) = cᵀMc / tr(M) | Gram energy scoring |

### Dimensional hierarchy

For K-tuple associations using G(2, n+1):
- D = C(n+1, 2) = Plücker dimension
- K = D - 2 lines needed to leave a 2D null space
- G(2,4) = P³ (D=6):  triple associations (K=4)
- G(2,5) = P⁴ (D=10): 8-tuple associations
- G(2,6) = P⁵ (D=15): 13-tuple associations
- G(2,10) = P⁹ (D=45): 43-tuple associations

Higher Grassmannians bind more items per association, not more triples.

### Higher Grassmannians: G(2,6) vs G(2,4)

The 6D Plücker space of G(2,4) is a bottleneck for large vocabularies —
too few dimensions to discriminate 67K candidates. Moving to G(2,6)
with 15D Plücker space dramatically improves generation quality:

| Grassmannian | D  | Lines/query | Mean similarity | Example (music) |
|--------------|----|-------------|:---------------:|-----------------|
| G(2,4) = P³  |  6 | 5           | 0.284           | orleans, empresses, seafloor |
| G(2,6) = P⁵  | 15 | 14          | **0.733**       | strumming, joplin, saxophone, plectrum |

Full results across 8 concepts with 10 multi-transversals:

```
G(2,6) generation:
  music:    strumming, joplin, chordophone, saxophone, plectrum
  guitar:   piano, organist, warbling, ditty, legato
  king:     courtiers, sovereigns, empresses, empress, democracies
  doctor:   outpatient, veterinary, immunotherapy, hemorrhoids
  fire:     brightness, brimstone, cremate, embers
  ocean:    seaside, tides, barnacle, seagulls
```

### Batch vectorisation

Vocabulary ranking is fully vectorised: all 67K lines are encoded in a single
matrix multiply `(N, 2d) @ (2d, n+1)`, then scored against all transversals
with `(T, D) @ (D, N)`. No Python loops over the vocabulary.

| Grassmannian | Before (loop) | After (batch) | Speedup |
|--------------|:-------------:|:-------------:|:-------:|
| G(2,4) 20T   | ~3s           | 0.018s        | ~170×   |
| G(2,6) 10T   | ~50s          | 0.017s        | ~3000×  |

G(2,6) needs 14 lines per query (vs 5 for G(2,4)), but 15D provides enough
geometric room to discriminate 67K items with fewer multi-transversal samples.

G(2,5) performs worse than G(2,4) — likely a Hodge dual formulation issue
in 10D. This is an open problem.

---

## Two memory modes

### Mode 1: Generative (transversal retrieval)

Store a triple of lines. Query with a 4th. Get 2 new lines — the transversals.

```python
from transversal_memory import P3Memory

mem = P3Memory()
mem.store([L1, L2, L3])
T1, T2 = mem.query_generative(L4)
```

The output lines T1, T2 are geometrically valid (satisfy the Plücker relation),
meet all 4 input lines, and are mutually skew. They did not exist before the
query — they are constituted by the intersection of the query with the stored
structure.

### Mode 2: Discriminative (energy scoring)

Store any number of lines. Score candidates.

```python
from transversal_memory import GramMemory

mem = GramMemory()
for word in associates:
    mem.store_line(line(source, word))

score = mem.score(line(source, candidate))
axes  = mem.principal_axes(k=3)   # dominant relational directions
```

### Self-contained embeddings (no GloVe needed)

Build embeddings directly from association norms:

```python
from transversal_memory.cooccurrence import embeddings_from_associations

associations = {
    "dog": ["puppy", "bark", "fetch", "bone", ...],
    "cat": ["kitten", "purr", "whiskers", ...],
    ...
}

emb = embeddings_from_associations(associations, dim=32)
line = emb.make_line("dog", "puppy")  # Plücker 6-vector
```

---

## Installation

```bash
pip install numpy scipy
git clone https://github.com/yourname/transversal-memory
cd transversal-memory
pip install -e .
```

## Examples

```bash
# Pure geometry: 4 lines → 2 transversals (no embeddings)
python examples/basic_geometry.py

# Self-contained pipeline on small dataset (~130 words)
python examples/cooccurrence_demo.py

# Full Overmann dataset (65K words, auto-downloads)
git clone https://github.com/PeterOvermann/WordAssociations data/WordAssociations
python examples/full_dataset_demo.py
# First run builds PPMI + SVD (~4s), subsequent runs load from cache
```

---

## Files

```
transversal_memory/
├── transversal_memory/
│   ├── plucker.py       # Plücker geometry: coords, inner product, Hodge dual
│   ├── solver.py        # Exact Plücker solver via PCA + quadratic formula
│   ├── memory.py        # P3Memory, GramMemory, ProjectedMemory
│   ├── embeddings.py    # Word vector utilities (GloVe, random, make_line)
│   ├── cooccurrence.py  # PPMI + sparse SVD embedding pipeline
│   ├── cas.py           # Content-addressable store via Plücker geometry
│   └── higher_grass.py  # Generalized G(2, n+1) for n > 3
├── examples/
│   ├── basic_geometry.py      # Pure geometry: 4 lines → 2 transversals
│   ├── capital_cities.py      # Analogy: Paris:France :: Madrid:Spain
│   ├── word_associations.py   # Word association scoring + generation
│   ├── cooccurrence_demo.py   # Full pipeline on small dataset
│   ├── full_dataset_demo.py   # Full Overmann dataset (65K words)
│   ├── debug_generative.py    # Diagnosis of co-punctal degeneracy
│   ├── fix_generative.py      # Fix experiments (dual projection)
│   ├── sequential_prediction.py    # Sequence prediction experiments
│   ├── associative_generation.py   # Single-transversal generation (baseline)
│   ├── multi_transversal_generation.py  # Multi-transversal generation (works)
│   ├── cas_demo.py                     # Content-addressable store demo
│   └── higher_grassmannian_demo.py     # G(2,4) vs G(2,5) vs G(2,6) comparison
├── COPUNCTAL_FIX.md           # Detailed writeup of the failure and fix
├── tests/
│   └── test_plucker.py
└── data/                      # (created on first run)
    ├── WordAssociations/      # Overmann dataset (git clone)
    └── cache/                 # Pickled embeddings
```

---

## References

- Overmann, P. (2022). Triadic Memory — A Fundamental Algorithm for Cognitive Computing.
  https://peterovermann.com/TriadicMemory.pdf

- Overmann, P. (2025). Word Associations Dataset.
  https://github.com/PeterOvermann/WordAssociations

- Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.

- Schubert, H. (1879). Calcül der abzählenden Geometrie.

- Levy, O. & Goldberg, Y. (2014). Neural Word Embedding as Implicit Matrix Factorization. NIPS.

---

## What the geometry says about cognition

- Retrieval is intersection, not lookup. The answer doesn't pre-exist in storage.
- The output count is always exactly 2 — forced by the quadratic, not a parameter.
- The two solutions are mutually skew: two coherent completions that cannot
  be simultaneously occupied (formal model of perceptual ambiguity).
- The Plücker relation enforces structural validity: not just any vector,
  but a geometrically coherent one. Analogue of semantic coherence in retrieval.
- The Gram matrix's eigenvectors are the principal relational axes of a concept —
  not hand-labeled but emergent from the geometry of its associations.
- A single transversal is a verifier, not a generator — one scalar constraint
  cannot discriminate a large vocabulary. But multiple transversals intersected
  act as a generator: each eliminates different false positives, and their
  intersection converges on genuinely related items.
