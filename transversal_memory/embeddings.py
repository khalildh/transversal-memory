"""
embeddings.py — Word vector utilities for TransversalMemory

Provides:
  - load_glove():    load GloVe embeddings from a .txt file
  - load_dict():     use a plain Python dict of {word: vector}
  - make_line():     project a (source, target) word-pair to a Plücker line
  - WordMemory:      high-level wrapper for building GramMemory per source word

GloVe vectors available at: https://nlp.stanford.edu/projects/glove/
Recommended: glove.6B.50d.txt (171 MB) or glove.6B.100d.txt

If you don't have GloVe, use random_embeddings() for testing.
"""

import numpy as np
from typing import Optional
from .plucker import project_to_line, random_projection
from .memory import GramMemory, ProjectedMemory


# ── Embedding loaders ─────────────────────────────────────────────────────────

def load_glove(path: str,
               max_words: Optional[int] = None,
               verbose: bool = True) -> dict[str, np.ndarray]:
    """
    Load GloVe embeddings from a text file.

    path      : path to glove.6B.Xd.txt
    max_words : if set, only load the first N words (they're sorted by frequency)

    Returns dict mapping word → normalised embedding vector.
    """
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_words and i >= max_words:
                break
            parts = line.rstrip().split()
            word = parts[0]
            vec  = np.array(parts[1:], dtype=float)
            n    = np.linalg.norm(vec)
            embeddings[word] = vec / n if n > 1e-12 else vec
    if verbose:
        print(f"Loaded {len(embeddings)} embeddings, "
              f"dim={next(iter(embeddings.values())).shape[0]}")
    return embeddings


def load_dict(d: dict) -> dict[str, np.ndarray]:
    """
    Wrap a plain {word: array} dict, normalising all vectors.
    Useful for testing with small hand-crafted embeddings.
    """
    out = {}
    for word, vec in d.items():
        vec = np.asarray(vec, float)
        n   = np.linalg.norm(vec)
        out[word] = vec / n if n > 1e-12 else vec
    return out


def random_embeddings(words: list[str],
                      dim: int = 50,
                      seed: int = 0) -> dict[str, np.ndarray]:
    """
    Generate random normalised embeddings for a word list.
    Useful for testing without a real embedding file.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for word in words:
        vec = rng.standard_normal(dim)
        out[word] = vec / np.linalg.norm(vec)
    return out


# ── Line construction from word pairs ─────────────────────────────────────────

def make_line(embeddings: dict[str, np.ndarray],
              source: str,
              target: str,
              W: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Construct the Plücker 6-vector for the line (source, target).

    If W is provided (4×n matrix), projects both embeddings through W first.
    Otherwise uses a default projection computed from the embedding dimension.

    Returns None if either word is missing from embeddings.
    """
    if source not in embeddings or target not in embeddings:
        return None
    a = embeddings[source]
    b = embeddings[target]

    if W is None:
        n = a.shape[0]
        W = _default_projection(n)

    return project_to_line(a, b, W)


def _default_projection(n: int, seed: int = 42) -> np.ndarray:
    """Cached default 4×n projection matrix."""
    if not hasattr(_default_projection, "_cache"):
        _default_projection._cache = {}
    if n not in _default_projection._cache:
        _default_projection._cache[n] = random_projection(n, np.random.default_rng(seed))
    return _default_projection._cache[n]


# ── WordMemory: high-level wrapper ────────────────────────────────────────────

class WordMemory:
    """
    High-level memory for word association data.

    Builds one GramMemory per source word from its associate list.
    Supports scoring, ranking, principal axis analysis, and cross-word comparison.

    Usage:
        wm = WordMemory(embeddings)
        wm.add_associations("abandonment",
                            ["child","fear","trauma","loss","loneliness"])
        wm.add_associations("abandons",
                            ["plan","project","partner","deserts","leaves"])

        # Score a new candidate
        wm.score("abandonment", "grief")    # → float

        # Rank all words in vocabulary against "abandonment"
        wm.rank("abandonment", top_k=10)

        # Principal relational axes of "abandonment"
        wm.principal_axes("abandonment", k=3)

        # Compare relational patterns of two words
        wm.compare("abandonment", "abase")
    """

    def __init__(self,
                 embeddings: dict[str, np.ndarray],
                 W: Optional[np.ndarray] = None):
        self.embeddings = embeddings
        n = next(iter(embeddings.values())).shape[0]
        self.W = W if W is not None else _default_projection(n)
        self.memories: dict[str, GramMemory] = {}
        self.associations: dict[str, list[str]] = {}

    def add_associations(self,
                         source: str,
                         targets: list[str]) -> int:
        """
        Store all (source, target) lines into source's GramMemory.

        Returns the number of lines successfully stored
        (some targets may be missing from embeddings).
        """
        if source not in self.memories:
            self.memories[source] = GramMemory()
            self.associations[source] = []

        stored = 0
        for target in targets:
            line = make_line(self.embeddings, source, target, self.W)
            if line is not None:
                self.memories[source].store_line(line)
                self.associations[source].append(target)
                stored += 1
        return stored

    def score(self, source: str, candidate: str) -> Optional[float]:
        """
        Score how well (source, candidate) fits the stored pattern for source.
        Returns None if source has no memory or candidate is not in embeddings.
        """
        if source not in self.memories:
            return None
        line = make_line(self.embeddings, source, candidate, self.W)
        if line is None:
            return None
        return self.memories[source].score(line)

    def rank(self,
             source: str,
             candidates: Optional[list[str]] = None,
             top_k: int = 10) -> list[tuple[float, str]]:
        """
        Rank candidate words by their fit to source's relational pattern.

        candidates: words to score. If None, uses all words in embeddings.
        top_k:      return only the top k results.

        Returns list of (score, word) sorted descending.
        """
        if source not in self.memories:
            return []
        if candidates is None:
            candidates = list(self.embeddings.keys())

        scored = []
        for word in candidates:
            s = self.score(source, word)
            if s is not None:
                scored.append((s, word))

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def principal_axes(self,
                       source: str,
                       k: int = 3) -> Optional[np.ndarray]:
        """
        Top k eigenvectors of source's Gram matrix.
        These are the principal relational directions of source's associate cluster.
        Shape: (k, 6).
        """
        if source not in self.memories:
            return None
        return self.memories[source].principal_axes(k)

    def cluster_associates(self,
                           source: str,
                           k: int = 3) -> Optional[list[list[str]]]:
        """
        Group associates of source by their dominant principal axis.
        Each associate is assigned to the axis it scores highest on.

        Returns k lists of associate words, one per axis.
        """
        if source not in self.memories:
            return None
        axes = self.principal_axes(source, k)
        if axes is None:
            return None

        assocs = self.associations.get(source, [])
        clusters: list[list[str]] = [[] for _ in range(k)]

        for word in assocs:
            line = make_line(self.embeddings, source, word, self.W)
            if line is None:
                continue
            scores = [abs(float(line @ ax)) for ax in axes]
            best = int(np.argmax(scores))
            clusters[best].append(word)

        return clusters

    def compare(self, source_a: str, source_b: str) -> Optional[float]:
        """
        Cosine similarity between the Gram matrices of two source words.
        High → similar relational patterns. Low → different.
        """
        ma = self.memories.get(source_a)
        mb = self.memories.get(source_b)
        if ma is None or mb is None:
            return None
        return ma.compare(mb)

    def analogy(self,
                source: str,
                associates: list[str],
                query_target: str,
                n_candidates: int = 10) -> list[tuple[float, str]]:
        """
        Analogy-style query using the transversal architecture.

        Given:   source:associates[0], source:associates[1], source:associates[2]
        Query:   source:query_target
        Returns: ranked list of words that complete the pattern

        This uses P3Memory (generative mode) with the first 3 associates as
        the stored triple, queries with (source, query_target), then decodes
        the resulting transversal back to the nearest word-pair in vocabulary.

        The result is the two words whose (source, word) line is most
        geometrically consistent with the stored triple + query.
        """
        from .memory import P3Memory

        if len(associates) < 3:
            return []

        # Build 3 stored lines from first 3 associates
        stored_lines = []
        for a in associates[:3]:
            line = make_line(self.embeddings, source, a, self.W)
            if line is not None:
                stored_lines.append(line)

        if len(stored_lines) < 3:
            return []

        mem = P3Memory()
        mem.store(stored_lines)

        # Query line
        q_line = make_line(self.embeddings, source, query_target, self.W)
        if q_line is None:
            return []

        transversals = mem.query_generative(q_line)
        if not transversals:
            return []

        T, _ = transversals[0]   # best transversal

        # Decode: find word whose (source, word) line is nearest to T
        results = []
        for word in self.embeddings:
            if word == source:
                continue
            line = make_line(self.embeddings, source, word, self.W)
            if line is None:
                continue
            # Alignment between T and this line
            align = abs(float(T @ line))
            results.append((align, word))

        results.sort(key=lambda x: -x[0])
        return results[:n_candidates]
