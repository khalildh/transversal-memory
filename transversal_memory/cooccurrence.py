"""
cooccurrence.py — Build word embeddings directly from co-occurrence data

No external corpus, no GloVe. The association norms themselves are the
co-occurrence signal.

Pipeline:
  1. Build a word × word co-occurrence matrix from (source, target) pairs
  2. Apply PMI or PPMI weighting
  3. SVD → dense left vectors (source role) and right vectors (target role)
  4. Lines are formed between a word's source vector and another's target vector

Why separate source/target vectors?

  The association data is directed: "abandonment → trauma" is in your data
  but "trauma → abandonment" probably isn't. This direction is real signal.
  SVD of an asymmetric matrix C gives:

      C ≈ U Σ Vᵀ

  U[:,k] = the k-th "source" direction  (how words behave as cue words)
  V[:,k] = the k-th "target" direction  (how words behave as associate words)

  A line between word a (as source) and word b (as target) then encodes
  the *directed relation* a→b, not just co-occurrence proximity.

  This is analogous to how word2vec separates input and output embeddings.
"""

import numpy as np
from collections import defaultdict
from typing import Optional


# ── Co-occurrence matrix ──────────────────────────────────────────────────────

class CooccurrenceMatrix:
    """
    Build a sparse word × word co-occurrence matrix from (source, target) pairs.

    Entries can be:
      - raw counts           (weighting="count")
      - PMI                  (weighting="pmi")
      - PPMI (positive PMI)  (weighting="ppmi")  ← recommended default
      - log(1 + count)       (weighting="log")

    Usage:
        co = CooccurrenceMatrix()
        for source, targets in associations.items():
            for target in targets:
                co.add(source, target)
        co.build(weighting="ppmi")
        embeddings = co.svd_embeddings(dim=50)
    """

    def __init__(self):
        self._counts: dict[tuple[str,str], float] = defaultdict(float)
        self._word_set: set[str] = set()

    def add(self, source: str, target: str, weight: float = 1.0) -> None:
        """Record one (source, target) co-occurrence."""
        self._counts[(source, target)] += weight
        self._word_set.add(source)
        self._word_set.add(target)

    def add_many(self,
                 associations: dict[str, list[str]],
                 position_decay: bool = False) -> None:
        """
        Add all associations from a dict of {source: [target, target, ...]}.

        position_decay: if True, weight earlier associates more strongly.
          Many association norm datasets list associates roughly in order of
          production frequency — the first associate is most strongly primed.
          Weight = 1/(1 + position).
        """
        for source, targets in associations.items():
            for i, target in enumerate(targets):
                w = 1.0 / (1.0 + i) if position_decay else 1.0
                self.add(source, target, w)

    def build(self, weighting: str = "ppmi", sparse: bool = False):
        """
        Build the (n_words × n_words) co-occurrence matrix.

        weighting: "count", "log", "pmi", "ppmi"
        sparse:    if True, build as scipy.sparse.csr_matrix (for large vocabs)
        Returns the matrix (also stored as self.C).
        """
        self.vocab = sorted(self._word_set)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        n = len(self.vocab)

        if sparse or n > 5000:
            from scipy.sparse import csr_matrix
            rows, cols, vals = [], [], []
            for (src, tgt), count in self._counts.items():
                rows.append(self.word2idx[src])
                cols.append(self.word2idx[tgt])
                vals.append(count)
            C_raw = csr_matrix((vals, (rows, cols)), shape=(n, n))

            if weighting in ("pmi", "ppmi"):
                C = _apply_pmi_sparse(C_raw, positive=(weighting == "ppmi"))
            elif weighting == "log":
                C_raw.data = np.log1p(C_raw.data)
                C = C_raw
            else:
                C = C_raw
        else:
            C = np.zeros((n, n))
            for (src, tgt), count in self._counts.items():
                i = self.word2idx[src]
                j = self.word2idx[tgt]
                C[i, j] = count

            if weighting == "count":
                pass
            elif weighting == "log":
                C = np.log1p(C)
            elif weighting in ("pmi", "ppmi"):
                C = _apply_pmi(C, positive=(weighting == "ppmi"))
            else:
                raise ValueError(f"Unknown weighting: {weighting!r}")

        self.C = C
        return C

    def svd_embeddings(self,
                       dim: int = 50,
                       role: str = "both"
                       ) -> "SVDEmbeddings":
        """
        Factorise C via truncated SVD and return word embeddings.

        dim  : number of dimensions to keep
        role : "source"  → U vectors only   (cue-word role)
               "target"  → V vectors only   (associate role)
               "both"    → separate U and V  ← recommended
               "average" → (U + V) / 2, symmetric

        Returns an SVDEmbeddings object.
        """
        assert hasattr(self, "C"), "Call build() first"
        dim = min(dim, min(self.C.shape) - 1)

        # Use scipy sparse truncated SVD for large matrices
        from scipy.sparse import issparse, csr_matrix
        from scipy.sparse.linalg import svds

        C = self.C
        if not issparse(C):
            C = csr_matrix(C)
        U, s, Vt = svds(C, k=dim)
        # svds returns smallest singular values first — reverse
        idx = np.argsort(s)[::-1]
        s = s[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]

        # Absorb sqrt(Σ) into both sides (standard for PMI-SVD)
        sq = np.sqrt(s)
        U_emb = U[:, :dim] * sq[np.newaxis, :]   # (n_words, dim)
        V_emb = Vt[:dim, :].T * sq[np.newaxis, :]  # (n_words, dim)

        # Normalise rows
        def norm_rows(M):
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            return M / norms

        U_emb = norm_rows(U_emb)
        V_emb = norm_rows(V_emb)

        if role == "both":
            src_vecs = {w: U_emb[i] for w, i in self.word2idx.items()}
            tgt_vecs = {w: V_emb[i] for w, i in self.word2idx.items()}
        elif role == "source":
            src_vecs = tgt_vecs = {w: U_emb[i] for w, i in self.word2idx.items()}
        elif role == "target":
            src_vecs = tgt_vecs = {w: V_emb[i] for w, i in self.word2idx.items()}
        elif role == "average":
            avg = norm_rows((U_emb + V_emb) / 2)
            src_vecs = tgt_vecs = {w: avg[i] for w, i in self.word2idx.items()}
        else:
            raise ValueError(f"Unknown role: {role!r}")

        return SVDEmbeddings(src_vecs, tgt_vecs, s[:dim], self.vocab)


# ── SVDEmbeddings ─────────────────────────────────────────────────────────────

class SVDEmbeddings:
    """
    Word embeddings from SVD of co-occurrence matrix.

    Holds separate source (U) and target (V) vectors.
    Use make_line() to construct a Plücker line for a directed (src, tgt) pair.
    Use as_dict() to get a plain {word: vector} dict for WordMemory.
    """

    def __init__(self,
                 src_vecs: dict[str, np.ndarray],
                 tgt_vecs: dict[str, np.ndarray],
                 singular_values: np.ndarray,
                 vocab: list[str]):
        self.src = src_vecs   # word → source embedding (U role)
        self.tgt = tgt_vecs   # word → target embedding (V role)
        self.singular_values = singular_values
        self.vocab = vocab
        self.dim = next(iter(src_vecs.values())).shape[0]

    def make_line(self,
                  source: str,
                  target: str,
                  W: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Construct the Plücker line for the directed pair (source → target).

        Uses source's U-vector and target's V-vector.
        If W is given (4×dim matrix), projects to P³ first.
        Otherwise uses the first 4 dimensions directly (fast but lossy).
        """
        from .plucker import project_to_line, line_from_points
        if source not in self.src or target not in self.tgt:
            return None
        a = self.src[source]
        b = self.tgt[target]
        if W is not None:
            return project_to_line(a, b, W)
        else:
            # Use top-4 SVD dimensions directly (already most informative)
            a4 = np.append(a[:4], 0.0)[:4]  # pad/truncate to 4
            b4 = np.append(b[:4], 0.0)[:4]
            return line_from_points(
                np.append(a4, 1.0) / np.linalg.norm(np.append(a4, 1.0)),
                np.append(b4, 1.0) / np.linalg.norm(np.append(b4, 1.0)),
            )

    def make_line_dual(self,
                       source: str,
                       target: str,
                       W1: np.ndarray,
                       W2: np.ndarray) -> Optional[np.ndarray]:
        """
        Construct a non-degenerate Plücker line for (source → target).

        Uses dual projection: both endpoints depend on both source and target.
        This breaks the co-punctal degeneracy that makes single-projection
        encoding useless for generative (transversal) retrieval.

        W1, W2: 4×(2*dim) projection matrices from random_projection_dual().
        """
        from .plucker import project_to_line_dual
        if source not in self.src or target not in self.tgt:
            return None
        a = self.src[source]
        b = self.tgt[target]
        return project_to_line_dual(a, b, W1, W2)

    def similarity(self, a: str, b: str, space: str = "source") -> float:
        """Cosine similarity between two words in source or target space."""
        vecs = self.src if space == "source" else self.tgt
        if a not in vecs or b not in vecs:
            return 0.0
        return float(np.dot(vecs[a], vecs[b]))

    def nearest(self, word: str, k: int = 10,
                space: str = "source") -> list[tuple[float, str]]:
        """Find k nearest words by cosine similarity."""
        vecs = self.src if space == "source" else self.tgt
        if word not in vecs:
            return []
        q = vecs[word]
        scored = [(float(np.dot(q, vecs[w])), w)
                  for w in self.vocab if w != word]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def variance_explained(self) -> np.ndarray:
        """Fraction of variance explained by each singular value."""
        s2 = self.singular_values ** 2
        return s2 / s2.sum()

    def effective_rank(self, threshold: float = 0.95) -> int:
        """Number of dimensions needed to explain `threshold` of variance."""
        cumvar = np.cumsum(self.variance_explained())
        return int(np.searchsorted(cumvar, threshold)) + 1

    def as_dict(self, role: str = "source") -> dict[str, np.ndarray]:
        """Return plain {word: vector} dict for use with WordMemory."""
        return self.src if role == "source" else self.tgt


# ── PMI helper ────────────────────────────────────────────────────────────────

def _apply_pmi(C: np.ndarray, positive: bool = True) -> np.ndarray:
    """
    Pointwise Mutual Information:
        PMI(i,j) = log[ P(i,j) / (P(i,*) · P(*,j)) ]

    With positive=True, negative values are clipped to 0 (PPMI).

    PPMI is standard for distributional semantics and avoids the
    problem that negative PMI is unreliable with sparse data.
    """
    total = C.sum()
    if total < 1e-12:
        return C

    row_sums = C.sum(axis=1, keepdims=True)   # P(i,*)
    col_sums = C.sum(axis=0, keepdims=True)   # P(*,j)

    # Avoid division by zero
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)

    expected = (row_sums * col_sums) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(np.where(C > 0, C / expected, 1e-12))
        pmi = np.where(C > 0, pmi, 0.0)

    if positive:
        pmi = np.maximum(pmi, 0.0)

    return pmi


def _apply_pmi_sparse(C, positive: bool = True):
    """
    Sparse PPMI: operates on scipy.sparse matrix without densifying.
    """
    from scipy.sparse import csr_matrix

    total = C.sum()
    if total < 1e-12:
        return C

    row_sums = np.array(C.sum(axis=1)).flatten()
    col_sums = np.array(C.sum(axis=0)).flatten()

    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)

    C_coo = C.tocoo()
    pmi_data = np.log(
        C_coo.data * total / (row_sums[C_coo.row] * col_sums[C_coo.col])
    )

    if positive:
        pmi_data = np.maximum(pmi_data, 0.0)

    # Drop zeros to keep it sparse
    mask = pmi_data > 0
    result = csr_matrix(
        (pmi_data[mask], (C_coo.row[mask], C_coo.col[mask])),
        shape=C.shape,
    )
    return result


# ── Convenience: build everything from an associations dict ───────────────────

def embeddings_from_associations(
        associations: dict[str, list[str]],
        dim: int = 20,
        weighting: str = "ppmi",
        role: str = "both",
        position_decay: bool = True,
        symmetric: bool = False,
) -> SVDEmbeddings:
    """
    One-shot: build SVD embeddings from a {source: [associates]} dict.

    associations   : your word association data
    dim            : embedding dimension (capped at vocab size - 1)
    weighting      : "count", "log", "pmi", "ppmi"
    role           : "both", "source", "target", "average"
    position_decay : weight earlier associates more (reflects production freq)
    symmetric      : if True, also add reverse associations (tgt → src)
                     makes the co-occurrence matrix symmetric

    Returns SVDEmbeddings ready to use with make_line() or as_dict().
    """
    co = CooccurrenceMatrix()
    co.add_many(associations, position_decay=position_decay)
    if symmetric:
        for src, targets in associations.items():
            for tgt in targets:
                co.add(tgt, src, weight=0.5)  # reverse with lower weight
    co.build(weighting=weighting)
    return co.svd_embeddings(dim=dim, role=role)
