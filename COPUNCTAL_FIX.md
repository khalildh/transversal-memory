# The Co-Punctal Degeneracy and Its Fix

This document describes the failure mode discovered when scaling P3Memory's
generative retrieval from ~130 words to 67K words, and the dual projection
fix that resolved it.

---

## The symptom

On the small co-occurrence demo (~130 words), P3Memory's generative retrieval
works: store 3 relational lines, query with a 4th, decode the transversal
back to the vocabulary, and known associates appear in the top results.

On the full Overmann dataset (67,378 words), the decoded results were garbage.
All alignment scores clustered at ~1.0 with no discrimination. The target
word ranked in the thousands or tens of thousands out of 67K candidates.

## The diagnosis

### What we observed

Running `examples/debug_generative.py` revealed:

1. **The Plücker inner product is identically zero** for ALL candidate lines
   from the same source word. Not approximately zero — exactly zero to machine
   precision (`|pi| < 1e-14` for 100% of pairs).

2. **The line matrix has effective rank 3, not 6.** For source="dog", the
   singular values of 2,000 candidate lines were `[27.47, 24.05, 22.13, 0.00, 0.00, 0.00]`.
   Three of the six Plücker dimensions are completely unused.

3. **The dot product has spread but is meaningless.** `|T · L|` had
   mean=0.42, std=0.30, but the top-10 words by dot product were unrelated
   to the source ("microfiber", "agamemnon", "vara", etc.).

### Root cause: shared-point degeneracy

The encoding `line(W·src, W·tgt)` maps a directed relation (source → target)
to the Plücker line through the two projected points `W·src` and `W·tgt`
in P³.

The problem: for a **fixed source word**, the point `W·src` is the same
for every target. All lines `line(W·src, W·tgt_i)` pass through the
common point `W·src`.

In projective geometry, lines through a common point are called **co-punctal**
(or concurrent). Co-punctal lines in P³ have a specific algebraic structure:

- They form a **3-dimensional linear subspace** of the 6-dimensional Plücker
  space. This is why the effective rank is 3, not 6.

- Any two lines through the same point **intersect** at that point. Therefore
  `plucker_inner(L_i, L_j) = 0` identically for all pairs. This is not
  numerical noise — it is an exact algebraic identity.

- The transversal T, computed as a linear combination of the null space
  basis vectors, also lies in this 3D subspace. Therefore T is incident
  with every candidate line from the same source.

- Result: **zero discrimination**. The Plücker inner product cannot
  distinguish any candidate from any other.

### Why it appeared to work at small scale

On the ~130 word demo, the Euclidean dot product `|T · L|` (not the
Plücker inner product) was used for decoding. At small vocabulary sizes,
the numerical spread of the dot product, combined with normalisation
effects, produced enough variation to rank words. But this was coincidental
— the dot product does not respect the Plücker geometry. At 67K words,
it becomes pure noise.

---

## Fixes tested and their results

### Fixes that DON'T work

| Fix | Approach | Why it fails |
|-----|----------|-------------|
| **A** | Use Plücker inner product instead of dot product | `|pi| = 0` for ALL candidates — the degeneracy makes this metric useless |
| **B** | PCA-based projection (preserve more variance) | Better projection, but co-punctal structure is unchanged — all lines still pass through the projected source |
| **C** | Use top 4 SVD dimensions directly (no random projection) | Same co-punctal issue regardless of which 4 dimensions are chosen |
| **D** | Multiple random projections with consensus ranking | Each projection has the same degeneracy; consensus of degenerate results is still degenerate |
| **E** | Higher-dimensional Grassmannian G(2,6) | More Plücker dimensions, but the shared-point problem persists — no transversals found because the constraint system is underdetermined |
| **F** | Soft Plücker constraint (rank by |pi| ascending) | Same as Fix A — |pi| = 0 for everything |

All of these fail because they address the **metric** or **projection quality**
but not the fundamental geometric structure. The co-punctal degeneracy
is not a numerical issue — it is an algebraic identity that holds regardless
of the projection matrix, dimensionality, or scoring function.

### Encoding changes that also DON'T work

| Fix | Encoding | Why it fails |
|-----|----------|-------------|
| **G** | Offset: `line(W(src + α·tgt), W·tgt)` | Point 2 is now shared across targets for fixed source. Just moves the degeneracy to the other endpoint. |
| **H** | Sum/diff: `line(W(src+tgt), W(src−tgt))` | Plücker coords of `line(a+b, a−b)` are proportional to `line(a, b)`. This produces the **exact same line** — the encoding is algebraically invariant. |
| **I** | Nonlinear: `line(W1·src, W2·(src⊙tgt))` | Point 1 = `W1·src` is still fixed for a given source. Lines remain co-punctal through that shared point. Full rank 3/6. |
| **J** | Multi-projection with nonlinear encoding | Consensus of rank-3 results is still rank-3. |

The key insight: **any encoding of the form `line(f(src), g(tgt))` where
`f(src)` is constant for a fixed source will produce co-punctal lines.**
The fix requires both endpoints to depend on the target.

### Fix that WORKS: dual projection (concatenation encoding)

**Encoding:** `line(W1·[src;tgt], W2·[src;tgt])`

Both endpoints are computed from the concatenation `[src; tgt]`, but through
two **different** random projection matrices W1 and W2 (each 4×2n).

Why this works:
- Point 1 = `W1·[src; tgt]` changes when target changes
- Point 2 = `W2·[src; tgt]` also changes when target changes
- Since W1 ≠ W2, the two points are distinct for each target
- No two lines from the same source share an endpoint
- The lines achieve **full rank 6** in Plücker space
- `plucker_inner` has real spread: mean=0.29, max=0.99, fraction zero=0.009

### Results with dual projection

Tested on the full Overmann vocabulary (67,378 candidates):

| Source | Stored lines | Query target | Target rank | Known in top 10 |
|--------|-------------|-------------|------------|----------------|
| king | crown, throne, royal | queen | **1** / 67,378 | 3 |
| fire | flame, heat, burn | smoke | **1** / 67,378 | 4 |
| ocean | waves, deep, salt | fish | **2** / 67,378 | 4 |
| love | heart, romance, passion | marriage | **2** / 67,378 | 4 |
| music | rhythm, melody, harmony | instrument | **3** / 67,378 | 4 |
| brain | neurons, memory, cortex | intelligence | **3** / 67,378 | 4 |
| dog | puppy, bark, fetch | bone | no transversal | — |
| tree | leaves, branches, roots | forest | no transversal | — |

For 6 of 8 test queries, the target word ranks in the top 3 out of 67,378
candidates. This is decoded purely from the geometric structure — no
nearest-neighbour search, no learned scoring function.

The stored lines (the 3 associates used to build the memory) also appear
in the top results, which is expected: the transversal T is geometrically
incident with all 4 input lines, so they should score near zero.

Two queries ("dog" and "tree") failed to find transversals. This happens
when the 4 input lines are not in sufficiently general position — the
constraint matrix does not have a clean 2D null space. This is a known
limitation of the P³ geometry (the Schubert theorem requires general position).

---

## Implementation

The fix is implemented in:

- **`transversal_memory/plucker.py`**: `project_to_line_dual()` and
  `random_projection_dual()` — the new encoding and its projection matrices
- **`transversal_memory/cooccurrence.py`**: `SVDEmbeddings.make_line_dual()` —
  high-level API for the dual encoding
- **`examples/full_dataset_demo.py`**: uses dual projection for the
  generative retrieval section

Usage:

```python
from transversal_memory import P3Memory, plucker_inner
from transversal_memory.plucker import random_projection_dual

# Create dual projection matrices
W1, W2 = random_projection_dual(n_items=32, rng=np.random.default_rng(42))

# Encode relations as lines
line1 = emb.make_line_dual("king", "crown", W1, W2)
line2 = emb.make_line_dual("king", "throne", W1, W2)
line3 = emb.make_line_dual("king", "royal", W1, W2)
query = emb.make_line_dual("king", "queen", W1, W2)

# Store and query
mem = P3Memory()
mem.store([line1, line2, line3])
transversals = mem.query_generative(query)

T, residual = transversals[0]

# Decode: rank candidates by |plucker_inner(T, candidate)|
# Lower = more incident with T = better match
for word in vocabulary:
    candidate = emb.make_line_dual("king", word, W1, W2)
    score = abs(plucker_inner(T, candidate))  # 0 = perfect match
```

### When to use which encoding

| Mode | Encoding | Why |
|------|----------|-----|
| **GramMemory** (discriminative) | `make_line(src, tgt, W)` single projection | Co-punctal degeneracy doesn't affect scoring: `cᵀMc` uses the Gram matrix, not plucker_inner. The rank-3 subspace still carries discriminative signal. |
| **P3Memory** (generative) | `make_line_dual(src, tgt, W1, W2)` dual projection | Decoding requires plucker_inner to discriminate candidates. Only works with full-rank (non-degenerate) lines. |

---

## The geometry in detail

### Why co-punctal lines form a 3D subspace

A line through point p ∈ P³ and point q ∈ P³ has Plücker coordinates
`L = p ∧ q` (exterior product). If p is fixed, then `L = p ∧ q` is
linear in q. Since q ∈ P³ has 4 homogeneous coordinates, and one degree
of freedom is removed (scalar multiples of q give the same line), the
space of lines through p is parametrised by 3 free parameters.

In Plücker R⁶, these lines span a 3-dimensional linear subspace. This
subspace lies entirely within the Grassmannian G(2,4) ⊂ P⁵.

### Why plucker_inner vanishes for co-punctal lines

Two lines L₁ = p ∧ q₁ and L₂ = p ∧ q₂ through a common point p:

```
plucker_inner(L₁, L₂) = plucker_inner(p ∧ q₁, p ∧ q₂)
```

The Plücker inner product of two lines is the determinant of the 4×4
matrix formed by their four defining points. When two of those points
coincide (both lines pass through p), the matrix has two identical rows,
so the determinant is zero.

This is an exact algebraic identity — it holds for ANY choice of
projection matrix W, any embedding dimension, and any vocabulary size.

### Why dual projection breaks it

With dual projection, the two points defining the line are:
```
Point 1 = W1 · [src; tgt]
Point 2 = W2 · [src; tgt]
```

For two different targets tgt₁ and tgt₂ with the same source:
```
L₁: Point1_a = W1·[src;tgt₁],  Point2_a = W2·[src;tgt₁]
L₂: Point1_b = W1·[src;tgt₂],  Point2_b = W2·[src;tgt₂]
```

In general, Point1_a ≠ Point1_b and Point2_a ≠ Point2_b. The four points
are distinct, so the lines do not share an endpoint, and
`plucker_inner(L₁, L₂) ≠ 0` in general.

---

## Diagnostic scripts

- **`examples/debug_generative.py`**: Systematic diagnosis. Tests fixes A–F
  (metric changes, PCA, multi-projection, higher Grassmannian, soft constraints).
  Confirms all fail due to co-punctal degeneracy.

- **`examples/fix_generative.py`**: Tests encoding-level fixes.
  Confirms nonlinear (Fix 1) and cross-term (Fix 3) don't fully break
  the degeneracy. Confirms concatenation/dual projection (Fix 2) achieves
  full rank 6 and produces correct results.
