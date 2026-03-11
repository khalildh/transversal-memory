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

## Mathematical background

A line in P³ is represented by its Plücker coordinates: a 6-vector
p = (p₀₁, p₀₂, p₀₃, p₁₂, p₁₃, p₂₃) where p_ij = a_i·b_j - a_j·b_i
for two points a, b on the line. This is the exterior product a∧b.

Two lines meet iff their Plücker inner product vanishes:
  <p, q> = p₀₁q₂₃ - p₀₂q₁₃ + p₀₃q₁₂ + p₁₂q₀₃ - p₁₃q₀₂ + p₂₃q₀₁ = 0

A valid line satisfies the Plücker relation: p₀₁p₂₃ - p₀₂p₁₃ + p₀₃p₁₂ = 0

The space of all lines in P³ is the Grassmannian G(2,4), embedded in P⁵
via the Plücker map. Schubert calculus on G(2,4) gives the classic result:
4 lines in general position → exactly 2 transversals.

### Dimensional hierarchy

For K-tuple associations using G(2, n+1):
- D = C(n+1, 2) = Plücker dimension
- K = D - 2 lines needed to leave a 2D null space
- G(2,4) = P³ (D=6):  triple associations (K=4)
- G(2,5) = P⁴ (D=10): 8-tuple associations
- G(2,10) = P⁹ (D=45): 43-tuple associations

Higher Grassmannians bind more items per association, not more triples.

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

---

## Word association example

```python
from transversal_memory import GramMemory, ProjectedMemory
from transversal_memory.embeddings import load_glove, make_line

glove = load_glove("glove.6B.50d.txt")

# Build discriminative memory for "abandonment"
mem = GramMemory(n_proj=3)
associates = ["child","fear","trauma","loss","loneliness","rejection",
              "isolation","alienation","estrangement","betrayal"]
for word in associates:
    mem.store_line(make_line(glove, "abandonment", word))

# Score a new candidate
print(mem.score(make_line(glove, "abandonment", "grief")))    # should be high
print(mem.score(make_line(glove, "abandonment", "algebra")))  # should be low

# Find principal relational axes
axes = mem.principal_axes(k=3)
# axes[0] ≈ emotional-consequence direction
# axes[1] ≈ synonym direction
# axes[2] ≈ co-occurrence direction
```

---

## Installation

```bash
pip install numpy scipy
git clone https://github.com/yourname/transversal-memory
cd transversal-memory
pip install -e .
```

---

## Files

```
transversal_memory/
├── transversal_memory/
│   ├── plucker.py       # Plücker geometry: coords, inner product, validity
│   ├── solver.py        # Exact Plücker solver via PCA + quadratic formula
│   ├── memory.py        # P3Memory, GramMemory, ProjectedMemory
│   └── embeddings.py    # Word vector utilities
├── examples/
│   ├── basic_geometry.py      # Pure geometry: 4 lines → 2 transversals
│   ├── capital_cities.py      # Analogy: Paris:France :: Madrid:Spain
│   └── word_associations.py   # Full word association scoring + generation
└── tests/
    └── test_plucker.py
```

---

## References

- Overmann, P. (2022). Triadic Memory — A Fundamental Algorithm for Cognitive Computing.
  https://peterovermann.com/TriadicMemory.pdf

- Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.

- Schubert, H. (1879). Calcül der abzählenden Geometrie.

- "Attention Is Not What You Need: Grassmann Flows as an Attention-Free
  Alternative for Sequence Modeling." arXiv:2512.19428 (2025).

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
