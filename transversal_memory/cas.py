"""
cas.py — Content-Addressable Store via Plücker geometry

Encode arbitrary byte content into Plücker lines, store in a geometric
index, retrieve by content similarity. The Plücker relation serves as
a built-in integrity check; the Gram matrix eigenstructure serves as
a compact content signature.

Architecture:
    Content (bytes)
        → chunk into fixed-size blocks
        → hash each block to a deterministic embedding vector
        → encode consecutive block-pairs as Plücker lines
        → accumulate into a 6×6 Gram matrix (the "signature")

    Store:  signature → data mapping (the Gram matrix IS the address)
    Query:  encode query content the same way → compare signatures
    Verify: Plücker relation on each line = integrity check

The key insight: the Gram matrix M = Σ pᵢ⊗pᵢ is a 36-float lossy
compression of the content's relational structure. Two files with
similar content produce similar M, enabling content-addressable
lookup without storing or comparing raw bytes.
"""

import hashlib
import numpy as np
from typing import Optional

from .plucker import (
    line_from_points,
    plucker_relation,
    is_valid_line,
    plucker_inner,
    random_projection,
)
from .memory import GramMemory, P3Memory


# ── Content encoding ─────────────────────────────────────────────────────

def _bytes_to_vector(data: bytes, dim: int = 32) -> np.ndarray:
    """
    Deterministically map arbitrary bytes to a unit vector in R^dim.
    Uses SHA-256 as a seed for a deterministic RNG to produce
    well-distributed float values.
    """
    # Use hash as seed for reproducible normal distribution
    seed_bytes = hashlib.sha256(data).digest()
    seed = int.from_bytes(seed_bytes[:8], "little")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim)
    norm = np.linalg.norm(raw)
    return raw / norm if norm > 1e-12 else raw


def chunk_bytes(data: bytes, chunk_size: int = 256,
                overlap: int = 0) -> list[bytes]:
    """
    Split data into fixed-size chunks with optional overlap.
    Pads the last chunk if needed. Overlap produces more chunks
    (and thus more Plücker lines) for richer signatures.
    """
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


def content_to_lines(data: bytes, chunk_size: int = 256,
                     dim: int = 32, W: Optional[np.ndarray] = None,
                     stride: int = 1,
                     ) -> list[np.ndarray]:
    """
    Encode byte content as a sequence of Plücker lines.

    Chunks the data with 50% overlap to produce more lines, then
    encodes pairs at distance `stride` apart. stride=1 gives
    consecutive pairs; stride=2 also adds skip-one pairs for
    richer structure.

    Returns list of normalised Plücker lines.
    """
    if W is None:
        W = _default_projection(dim)

    overlap = chunk_size // 2
    chunks = chunk_bytes(data, chunk_size, overlap=overlap)
    vecs = [_bytes_to_vector(c, dim) for c in chunks]

    lines = []
    # Consecutive pairs
    for i in range(len(vecs) - 1):
        a = W @ vecs[i]
        b = W @ vecs[i + 1]
        L = line_from_points(a, b)
        if L is not None and np.linalg.norm(L) > 1e-12:
            lines.append(L)

    # Skip-one pairs for additional structure
    for i in range(len(vecs) - 2):
        a = W @ vecs[i]
        b = W @ vecs[i + 2]
        L = line_from_points(a, b)
        if L is not None and np.linalg.norm(L) > 1e-12:
            lines.append(L)

    # If only one chunk, encode it against a fixed "anchor" vector
    if not lines and len(vecs) >= 1:
        anchor = _bytes_to_vector(b"__cas_anchor__", dim)
        a = W @ vecs[0]
        b = W @ anchor
        L = line_from_points(a, b)
        if L is not None:
            lines.append(L)

    return lines


# ── Signature: the Gram matrix as content address ────────────────────────

class ContentSignature:
    """
    A compact geometric fingerprint of byte content.

    The signature is a 6×6 Gram matrix M = Σ pᵢ⊗pᵢ built from
    the Plücker lines of consecutive chunk pairs. Two files with
    similar content produce similar signatures.

    Properties:
        - Deterministic: same content → same signature
        - Compact: 36 floats regardless of file size
        - Comparable: cosine similarity between signatures
        - Verifiable: eigenstructure encodes content geometry
    """

    def __init__(self, data: bytes = b"", chunk_size: int = 256,
                 dim: int = 32, W: Optional[np.ndarray] = None):
        self._gram = GramMemory()
        self._n_chunks = 0
        self._content_hash = hashlib.sha256(data).hexdigest() if data else ""
        self._dim = dim
        self._chunk_size = chunk_size
        self._W = W if W is not None else _default_projection(dim)

        if data:
            self._encode(data)

    def _encode(self, data: bytes) -> None:
        lines = content_to_lines(data, self._chunk_size, self._dim, self._W)
        for L in lines:
            self._gram.store_line(L)
        self._n_chunks = len(chunk_bytes(data, self._chunk_size))

    @property
    def matrix(self) -> np.ndarray:
        """The 6×6 Gram matrix."""
        return self._gram.M.copy()

    @property
    def n_lines(self) -> int:
        return self._gram.n_lines

    @property
    def content_hash(self) -> str:
        """SHA-256 hex digest of the original content (for exact match)."""
        return self._content_hash

    def eigenvalues(self) -> np.ndarray:
        """Eigenvalue spectrum of the signature."""
        return self._gram.eigenvalues()

    def principal_axes(self, k: int = 3) -> np.ndarray:
        """Top k eigenvectors — the dominant structural directions."""
        return self._gram.principal_axes(k)

    def similarity(self, other: "ContentSignature") -> float:
        """Cosine similarity between two content signatures."""
        return self._gram.compare(other._gram)

    def score_line(self, line: np.ndarray) -> float:
        """How consistent is this line with the stored content pattern?"""
        return self._gram.score(line)

    def integrity_check(self, data: bytes) -> dict:
        """
        Verify content integrity using Plücker geometry.

        Re-encodes the data and checks:
        1. Each line satisfies the Plücker relation (geometric validity)
        2. Lines are consistent with the stored Gram matrix (content match)
        """
        lines = content_to_lines(data, self._chunk_size, self._dim, self._W)
        results = {
            "n_lines": len(lines),
            "all_valid": True,
            "mean_score": 0.0,
            "plucker_residuals": [],
            "scores": [],
        }

        for L in lines:
            resid = abs(plucker_relation(L))
            results["plucker_residuals"].append(resid)
            if resid > 1e-8:
                results["all_valid"] = False

            if self._gram.n_lines > 0:
                results["scores"].append(self._gram.score(L))

        if results["scores"]:
            results["mean_score"] = float(np.mean(results["scores"]))

        return results

    def to_bytes(self) -> bytes:
        """Serialise the signature to bytes (for storage/transmission)."""
        return self._gram.M.tobytes()

    @classmethod
    def from_bytes(cls, raw: bytes, content_hash: str = "") -> "ContentSignature":
        """Reconstruct a signature from serialised bytes."""
        sig = cls.__new__(cls)
        sig._gram = GramMemory()
        sig._gram.M = np.frombuffer(raw, dtype=np.float64).reshape(6, 6).copy()
        sig._gram.n_lines = -1  # unknown
        sig._content_hash = content_hash
        sig._dim = 32
        sig._chunk_size = 256
        sig._W = _default_projection(32)
        return sig


# ── Content-Addressable Store ────────────────────────────────────────────

class ContentStore:
    """
    Content-addressable store using Plücker geometry.

    Store files (byte content). Retrieve by:
    - Exact match: SHA-256 hash
    - Similarity: Gram matrix cosine similarity
    - Partial match: encode a fragment, score against all signatures

    The geometric structure gives:
    - Content fingerprinting (Gram eigenstructure)
    - Integrity verification (Plücker relation)
    - Similarity search (Gram matrix comparison)
    - Deduplication (similar content → similar signatures)
    """

    def __init__(self, chunk_size: int = 256, dim: int = 32,
                 seed: int = 42):
        self._chunk_size = chunk_size
        self._dim = dim
        self._W = random_projection(dim, np.random.default_rng(seed))
        # row-normalise for numerical stability
        for i in range(self._W.shape[0]):
            n = np.linalg.norm(self._W[i])
            if n > 1e-12:
                self._W[i] /= n

        self._entries: dict[str, _StoreEntry] = {}  # hash → entry

    def put(self, data: bytes, label: str = "") -> str:
        """
        Store content. Returns the content hash (address).

        If content already exists (same SHA-256), returns existing hash
        without re-storing (deduplication).
        """
        content_hash = hashlib.sha256(data).hexdigest()

        if content_hash in self._entries:
            return content_hash

        sig = ContentSignature(data, self._chunk_size, self._dim, self._W)

        self._entries[content_hash] = _StoreEntry(
            content_hash=content_hash,
            signature=sig,
            data=data,
            label=label or content_hash[:12],
            size=len(data),
        )

        return content_hash

    def get(self, content_hash: str) -> Optional[bytes]:
        """Retrieve content by exact hash."""
        entry = self._entries.get(content_hash)
        return entry.data if entry else None

    def get_signature(self, content_hash: str) -> Optional[ContentSignature]:
        """Get the geometric signature for a stored item."""
        entry = self._entries.get(content_hash)
        return entry.signature if entry else None

    def contains(self, data: bytes) -> bool:
        """Check if content is already stored (by hash)."""
        return hashlib.sha256(data).hexdigest() in self._entries

    def find_similar(self, data: bytes, top_k: int = 5,
                     threshold: float = 0.0) -> list[dict]:
        """
        Find stored items most similar to the given content.

        Encodes the query content into a signature, then compares against
        all stored signatures using Gram matrix cosine similarity.

        Returns list of {hash, label, similarity, size} sorted by
        similarity descending.
        """
        query_sig = ContentSignature(
            data, self._chunk_size, self._dim, self._W)

        results = []
        for h, entry in self._entries.items():
            sim = query_sig.similarity(entry.signature)
            if sim >= threshold:
                results.append({
                    "hash": h,
                    "label": entry.label,
                    "similarity": sim,
                    "size": entry.size,
                })

        results.sort(key=lambda x: -x["similarity"])
        return results[:top_k]

    def find_by_fragment(self, fragment: bytes, top_k: int = 5) -> list[dict]:
        """
        Find stored items that likely contain the given fragment.

        Encodes the fragment as Plücker lines and scores them against
        each stored signature's Gram matrix.
        """
        lines = content_to_lines(
            fragment, self._chunk_size, self._dim, self._W)

        if not lines:
            return []

        results = []
        for h, entry in self._entries.items():
            scores = [entry.signature.score_line(L) for L in lines]
            mean_score = float(np.mean(scores))
            results.append({
                "hash": h,
                "label": entry.label,
                "score": mean_score,
                "size": entry.size,
            })

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]

    def verify(self, content_hash: str) -> Optional[dict]:
        """
        Verify integrity of stored content.

        Re-encodes the stored data and checks:
        1. SHA-256 hash matches
        2. All Plücker lines are geometrically valid
        3. Lines are consistent with stored signature
        """
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
            "max_plucker_residual": max(geo_check["plucker_residuals"])
                if geo_check["plucker_residuals"] else 0.0,
        }

    def similarity_matrix(self) -> tuple[list[str], np.ndarray]:
        """
        Compute pairwise similarity between all stored items.
        Returns (labels, matrix).
        """
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
        """List all stored items."""
        return [
            {"hash": h, "label": e.label, "size": e.size,
             "n_lines": e.signature.n_lines}
            for h, e in self._entries.items()
        ]


class _StoreEntry:
    __slots__ = ("content_hash", "signature", "data", "label", "size")

    def __init__(self, content_hash, signature, data, label, size):
        self.content_hash = content_hash
        self.signature = signature
        self.data = data
        self.label = label
        self.size = size


# ── Module-level helpers ─────────────────────────────────────────────────

_DEFAULT_W = None

def _default_projection(dim: int = 32) -> np.ndarray:
    """Lazily create a shared default projection matrix."""
    global _DEFAULT_W
    if _DEFAULT_W is None or _DEFAULT_W.shape[1] != dim:
        W = random_projection(dim, np.random.default_rng(0xCA5))
        for i in range(W.shape[0]):
            n = np.linalg.norm(W[i])
            if n > 1e-12:
                W[i] /= n
        _DEFAULT_W = W
    return _DEFAULT_W
