"""
cas_multiseed.py — Multi-seed Content-Addressable Store

Upgraded CAS using multi-seed Gram^0.05 ensemble signatures.
Same API as cas.py but with dramatically better similarity discrimination.

Single-seed CAS: 1 random projection → 1 Gram matrix (36 floats)
Multi-seed CAS:  N random projections → N Gram^0.05 matrices (36N floats)

The multi-seed ensemble recovers structural information lost by any single
random projection, improving similarity search from near-random to meaningful.
"""

import hashlib
import numpy as np
from typing import Optional

from .plucker import (
    plucker_relation,
    random_projection_dual,
    project_to_line_dual,
    batch_encode_lines_dual,
)


# ── Config ──────────────────────────────────────────────────────────────────

DEFAULT_N_SEEDS = 50
DEFAULT_SEED_SPACING = 10
DEFAULT_GRAM_POWER = 0.05
DEFAULT_DIM = 32
DEFAULT_CHUNK_SIZE = 256


# ── Helpers ─────────────────────────────────────────────────────────────────

def _bytes_to_vector(data: bytes, dim: int = 32) -> np.ndarray:
    """Deterministically map bytes to a unit vector in R^dim."""
    seed_bytes = hashlib.sha256(data).digest()
    seed = int.from_bytes(seed_bytes[:8], "little")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim)
    norm = np.linalg.norm(raw)
    return raw / norm if norm > 1e-12 else raw


def _chunk_bytes(data: bytes, chunk_size: int = 256) -> list[bytes]:
    """Split data into overlapping chunks."""
    overlap = chunk_size // 2
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(data), step):
        chunk = data[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + b'\x00' * (chunk_size - len(chunk))
        chunks.append(chunk)
    if not chunks:
        chunks = [b'\x00' * chunk_size]
    return chunks


def _power_gram(gram: np.ndarray, power: float) -> np.ndarray:
    """Apply fractional power to Gram eigenvalues."""
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


def _make_projections(n_seeds: int, seed_spacing: int,
                      dim: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Precompute dual projection matrices for all seeds."""
    projs = []
    for i in range(n_seeds):
        rng = np.random.default_rng(i * seed_spacing)
        W1, W2 = random_projection_dual(dim, rng)
        projs.append((W1, W2))
    return projs


# ── Multi-seed Content Signature ───────────────────────────────────────────

class MultiSeedSignature:
    """
    Content fingerprint using multi-seed Gram^0.05 ensemble.

    Stores N_SEEDS Gram matrices (each 6×6), one per random projection.
    Similarity is computed as mean cosine across all seed pairs.

    Properties:
        - Deterministic: same content → same signature
        - Compact: 36 × N_SEEDS floats (1800 for 50 seeds = 14.4KB)
        - Comparable: mean cosine similarity across seeds
        - Much better discrimination than single-seed
    """

    def __init__(self, data: bytes = b"",
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 dim: int = DEFAULT_DIM,
                 n_seeds: int = DEFAULT_N_SEEDS,
                 seed_spacing: int = DEFAULT_SEED_SPACING,
                 gram_power: float = DEFAULT_GRAM_POWER,
                 projections: Optional[list] = None):
        self._dim = dim
        self._chunk_size = chunk_size
        self._n_seeds = n_seeds
        self._gram_power = gram_power
        self._content_hash = hashlib.sha256(data).hexdigest() if data else ""
        self._n_lines = 0

        if projections is not None:
            self._projections = projections
        else:
            self._projections = _make_projections(n_seeds, seed_spacing, dim)

        # Per-seed Gram matrices (for scoring) and raw Grams (for comparison)
        self._grams_powered = []    # Gram^power — for line scoring
        self._grams_raw = []        # Raw Gram — for cosine comparison

        if data:
            self._encode(data)

    def _encode(self, data: bytes) -> None:
        """Encode content into per-seed Gram matrices."""
        chunks = _chunk_bytes(data, self._chunk_size)
        vecs = [_bytes_to_vector(c, self._dim) for c in chunks]

        if len(vecs) < 2:
            # Single chunk: pair with anchor
            anchor = _bytes_to_vector(b"__cas_anchor__", self._dim)
            vecs = [vecs[0], anchor] if vecs else [anchor, anchor]

        # Build vector pairs: consecutive + skip-one
        pairs = []
        for i in range(len(vecs) - 1):
            pairs.append((vecs[i], vecs[i + 1]))
        for i in range(len(vecs) - 2):
            pairs.append((vecs[i], vecs[i + 2]))

        self._n_lines = len(pairs)

        for W1, W2 in self._projections:
            lines = []
            for a, b in pairs:
                L = project_to_line_dual(a, b, W1, W2)
                if np.linalg.norm(L) > 1e-12:
                    lines.append(L)

            if not lines:
                self._grams_powered.append(np.zeros((6, 6)))
                self._grams_raw.append(np.zeros((6, 6)))
                continue

            arr = np.stack(lines)
            raw_gram = arr.T @ arr
            self._grams_raw.append(raw_gram)
            self._grams_powered.append(_power_gram(raw_gram, self._gram_power))

    @property
    def n_lines(self) -> int:
        return self._n_lines

    @property
    def content_hash(self) -> str:
        return self._content_hash

    @property
    def n_seeds(self) -> int:
        return self._n_seeds

    def similarity(self, other: "MultiSeedSignature") -> float:
        """
        Mean cosine similarity across all seed pairs.
        Uses raw Gram (not powered) for comparison — powered matrices
        compress eigenvalues too aggressively for similarity.
        """
        if not self._grams_raw or not other._grams_raw:
            return 0.0

        n = min(len(self._grams_raw), len(other._grams_raw))
        sims = []
        for i in range(n):
            a = self._grams_raw[i].flatten()
            b = other._grams_raw[i].flatten()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-12 and nb > 1e-12:
                sims.append(float(a @ b / (na * nb)))

        return float(np.mean(sims)) if sims else 0.0

    def score_content(self, data: bytes) -> float:
        """
        Score how well content matches this signature.
        Encodes query content and computes mean Gram energy across seeds.
        """
        chunks = _chunk_bytes(data, self._chunk_size)
        vecs = [_bytes_to_vector(c, self._dim) for c in chunks]

        if len(vecs) < 2:
            anchor = _bytes_to_vector(b"__cas_anchor__", self._dim)
            vecs = [vecs[0], anchor] if vecs else [anchor, anchor]

        pairs = []
        for i in range(len(vecs) - 1):
            pairs.append((vecs[i], vecs[i + 1]))
        for i in range(len(vecs) - 2):
            pairs.append((vecs[i], vecs[i + 2]))

        if not pairs or not self._grams_powered:
            return 0.0

        total = 0.0
        count = 0
        for seed_idx, (W1, W2) in enumerate(self._projections):
            if seed_idx >= len(self._grams_powered):
                break
            gram = self._grams_powered[seed_idx]
            for a, b in pairs:
                L = project_to_line_dual(a, b, W1, W2)
                if np.linalg.norm(L) > 1e-12:
                    total += float(L @ gram @ L)
                    count += 1

        return total / count if count > 0 else 0.0

    def integrity_check(self, data: bytes) -> dict:
        """Check content integrity via Plücker relation + Gram consistency."""
        chunks = _chunk_bytes(data, self._chunk_size)
        vecs = [_bytes_to_vector(c, self._dim) for c in chunks]

        if len(vecs) < 2:
            anchor = _bytes_to_vector(b"__cas_anchor__", self._dim)
            vecs = [vecs[0], anchor] if vecs else [anchor, anchor]

        pairs = []
        for i in range(len(vecs) - 1):
            pairs.append((vecs[i], vecs[i + 1]))

        # Check Plücker relation on first seed's lines
        if self._projections:
            W1, W2 = self._projections[0]
            residuals = []
            for a, b in pairs:
                L = project_to_line_dual(a, b, W1, W2)
                residuals.append(abs(plucker_relation(L)))

            return {
                "n_lines": len(pairs),
                "all_valid": all(r < 1e-8 for r in residuals),
                "mean_score": self.score_content(data),
                "max_plucker_residual": max(residuals) if residuals else 0.0,
            }

        return {"n_lines": 0, "all_valid": True, "mean_score": 0.0,
                "max_plucker_residual": 0.0}

    def to_bytes(self) -> bytes:
        """Serialise all Gram matrices."""
        raw = np.stack(self._grams_raw) if self._grams_raw else np.zeros((0, 6, 6))
        powered = np.stack(self._grams_powered) if self._grams_powered else np.zeros((0, 6, 6))
        return raw.tobytes() + powered.tobytes()


# ── Multi-seed Content Store ───────────────────────────────────────────────

class MultiSeedContentStore:
    """
    Content-addressable store using multi-seed Gram^0.05 signatures.

    Drop-in replacement for ContentStore with much better similarity search.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 dim: int = DEFAULT_DIM,
                 n_seeds: int = DEFAULT_N_SEEDS,
                 seed_spacing: int = DEFAULT_SEED_SPACING,
                 gram_power: float = DEFAULT_GRAM_POWER):
        self._chunk_size = chunk_size
        self._dim = dim
        self._n_seeds = n_seeds
        self._gram_power = gram_power

        # Precompute shared projections
        self._projections = _make_projections(n_seeds, seed_spacing, dim)
        self._entries: dict[str, _MSEntry] = {}

    def put(self, data: bytes, label: str = "") -> str:
        """Store content. Returns content hash."""
        content_hash = hashlib.sha256(data).hexdigest()
        if content_hash in self._entries:
            return content_hash

        sig = MultiSeedSignature(
            data, self._chunk_size, self._dim, self._n_seeds,
            gram_power=self._gram_power, projections=self._projections)

        self._entries[content_hash] = _MSEntry(
            content_hash=content_hash, signature=sig,
            data=data, label=label or content_hash[:12], size=len(data))

        return content_hash

    def get(self, content_hash: str) -> Optional[bytes]:
        entry = self._entries.get(content_hash)
        return entry.data if entry else None

    def get_signature(self, content_hash: str) -> Optional[MultiSeedSignature]:
        entry = self._entries.get(content_hash)
        return entry.signature if entry else None

    def contains(self, data: bytes) -> bool:
        return hashlib.sha256(data).hexdigest() in self._entries

    def find_similar(self, data: bytes, top_k: int = 5,
                     threshold: float = 0.0) -> list[dict]:
        """Find stored items most similar to query content."""
        query_sig = MultiSeedSignature(
            data, self._chunk_size, self._dim, self._n_seeds,
            gram_power=self._gram_power, projections=self._projections)

        results = []
        for h, entry in self._entries.items():
            sim = query_sig.similarity(entry.signature)
            if sim >= threshold:
                results.append({
                    "hash": h, "label": entry.label,
                    "similarity": sim, "size": entry.size})

        results.sort(key=lambda x: -x["similarity"])
        return results[:top_k]

    def find_by_fragment(self, fragment: bytes, top_k: int = 5) -> list[dict]:
        """Find stored items that likely contain the given fragment."""
        results = []
        for h, entry in self._entries.items():
            score = entry.signature.score_content(fragment)
            results.append({
                "hash": h, "label": entry.label,
                "score": score, "size": entry.size})

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]

    def verify(self, content_hash: str) -> Optional[dict]:
        entry = self._entries.get(content_hash)
        if entry is None:
            return None
        actual_hash = hashlib.sha256(entry.data).hexdigest()
        geo_check = entry.signature.integrity_check(entry.data)
        return {
            "hash_valid": actual_hash == content_hash,
            "geometric_valid": geo_check["all_valid"],
            "mean_score": geo_check["mean_score"],
            "n_lines": geo_check["n_lines"],
            "max_plucker_residual": geo_check["max_plucker_residual"],
        }

    def similarity_matrix(self) -> tuple[list[str], np.ndarray]:
        entries = list(self._entries.values())
        n = len(entries)
        labels = [e.label for e in entries]
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = entries[i].signature.similarity(entries[j].signature)
                mat[i, j] = sim
                mat[j, i] = sim
        return labels, mat

    @property
    def n_items(self) -> int:
        return len(self._entries)

    def list_items(self) -> list[dict]:
        return [
            {"hash": h, "label": e.label, "size": e.size,
             "n_lines": e.signature.n_lines}
            for h, e in self._entries.items()
        ]


class _MSEntry:
    __slots__ = ("content_hash", "signature", "data", "label", "size")

    def __init__(self, content_hash, signature, data, label, size):
        self.content_hash = content_hash
        self.signature = signature
        self.data = data
        self.label = label
        self.size = size
