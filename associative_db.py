"""
associative_db.py — Unified Associative Database
==================================================

Combines TDGA's triadic memory (exact recall) with transversal memory's
multi-seed Gram^0.05 (geometric similarity + generative expansion).

Architecture:
    TDGA layer:
        - Sentence embeddings (384D, frozen MiniLM)
        - SDR encoding (50 sorted indices in [0,4999])
        - Triadic memory: given 2 of (subject, relation, object), recall 3rd
        - Fixed-time O(P²·N) retrieval, independent of corpus size

    Geometry layer:
        - Multi-seed Gram^0.05 signatures from 384D embeddings
        - Similarity search via Gram matrix comparison
        - Generative expansion: discover new facts consistent with stored patterns

    Combined operations:
        1. store(fact)      — store in both triadic + geometry
        2. recall(s, r, ?)  — exact triadic recall
        3. similar(query)   — geometric similarity search
        4. expand(subject)  — use known facts as seeds, generate new associations
        5. hybrid(query)    — triadic recall + geometric expansion + reranking

Usage:
    db = AssociativeDB.from_checkpoints(
        embed_model="/path/to/embed_sorted.pt",
        device="mps"
    )
    db.store(Fact("Paris", "capital_of", "France", ...))
    db.store(Fact("Berlin", "capital_of", "Germany", ...))

    # Exact recall
    db.recall("Paris", "capital_of")  # → "France"

    # Similarity: what facts look like "Madrid capital_of Spain"?
    db.similar("Madrid capital_of Spain")

    # Expansion: what else relates to "Paris" like its stored facts?
    db.expand("Paris")
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# Add TDGA to path
TDGA_PATH = Path("/Volumes/PRO-G40/Code/TDGA/src")
if str(TDGA_PATH) not in sys.path:
    sys.path.insert(0, str(TDGA_PATH))

from tdga.pipeline import TDGAPipeline
from tdga.torch_indicator_memory import TorchIndicatorMemory
from tdga.sample_facts import Fact
from tdga.sdr import sdr_overlap

# Transversal memory geometry
from transversal_memory.plucker import (
    random_projection_dual,
    batch_encode_lines_dual,
    project_to_line_dual,
)


# ── Config ──────────────────────────────────────────────────────────────

N_SEEDS = 50
SEED_SPACING = 10
GRAM_POWER = 0.05
EMBED_DIM = 384  # MiniLM output dimension


# ── Geometric signature for facts ──────────────────────────────────────

def _make_projections(n_seeds, seed_spacing, dim):
    projs = []
    for i in range(n_seeds):
        rng = np.random.default_rng(i * seed_spacing)
        W1, W2 = random_projection_dual(dim, rng)
        projs.append((W1, W2))
    return projs


def _power_gram(gram, power):
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


# Precompute projections for 384D embeddings
PROJECTIONS = _make_projections(N_SEEDS, SEED_SPACING, EMBED_DIM)


class GeometricSignature:
    """Multi-seed Gram^0.05 signature built from embedding pairs."""

    def __init__(self):
        self.grams_raw = []      # For similarity comparison
        self.grams_powered = []  # For scoring
        self.n_lines = 0

    def add_line(self, src_vec, tgt_vec):
        """Add a single (source, target) embedding pair."""
        if not hasattr(self, '_lines'):
            self._lines = []
        self._lines.append((src_vec.copy(), tgt_vec.copy()))
        self.n_lines += 1

    def finalize(self):
        """Compute Gram matrices from accumulated lines."""
        if not hasattr(self, '_lines') or not self._lines:
            return

        self.grams_raw = []
        self.grams_powered = []

        for W1, W2 in PROJECTIONS:
            plucker_lines = []
            for src, tgt in self._lines:
                L = project_to_line_dual(src, tgt, W1, W2)
                if np.linalg.norm(L) > 1e-12:
                    plucker_lines.append(L)

            if len(plucker_lines) < 2:
                self.grams_raw.append(np.zeros((6, 6)))
                self.grams_powered.append(np.zeros((6, 6)))
                continue

            arr = np.stack(plucker_lines)
            raw = arr.T @ arr
            self.grams_raw.append(raw)
            self.grams_powered.append(_power_gram(raw, GRAM_POWER))

        del self._lines

    def similarity(self, other):
        """Mean cosine similarity across seeds (uses raw Gram)."""
        if not self.grams_raw or not other.grams_raw:
            return 0.0
        n = min(len(self.grams_raw), len(other.grams_raw))
        sims = []
        for i in range(n):
            a = self.grams_raw[i].flatten()
            b = other.grams_raw[i].flatten()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-12 and nb > 1e-12:
                sims.append(float(a @ b / (na * nb)))
        return float(np.mean(sims)) if sims else 0.0

    def score_embedding(self, src_vec, tgt_vec):
        """Score how well an (src, tgt) pair fits this signature."""
        if not self.grams_powered:
            return 0.0
        total = 0.0
        count = 0
        for i, (W1, W2) in enumerate(PROJECTIONS):
            if i >= len(self.grams_powered):
                break
            L = project_to_line_dual(src_vec, tgt_vec, W1, W2)
            if np.linalg.norm(L) > 1e-12:
                total += float(L @ self.grams_powered[i] @ L)
                count += 1
        return total / count if count > 0 else 0.0


# ── Stored fact with both representations ──────────────────────────────

@dataclass
class StoredFact:
    fact: Fact
    subject_sdr: np.ndarray
    relation_sdr: np.ndarray
    object_sdr: np.ndarray
    subject_emb: np.ndarray     # 384D
    relation_emb: np.ndarray    # 384D
    object_emb: np.ndarray      # 384D
    fact_emb: np.ndarray        # 384D (combined text)


# ── Associative Database ───────────────────────────────────────────────

class AssociativeDB:
    """
    Unified database combining triadic memory + geometric signatures.

    Two retrieval modes:
        1. Triadic recall: exact pattern completion (2→1)
        2. Geometric search: similarity + generative expansion
    """

    def __init__(self, pipeline, encoder_fn, device="cpu"):
        self.pipeline = pipeline
        self.encoder_fn = encoder_fn
        self.device = device

        # Triadic memory for exact recall
        self.triadic = TorchIndicatorMemory(
            N=pipeline.config.sdr_dim,
            P=pipeline.config.sdr_bits,
            device="mps" if torch.backends.mps.is_available() else device,
            exact=True,
        )

        # Stored facts
        self.facts: list[StoredFact] = []

        # Per-subject geometric signatures
        self._subject_sigs: dict[str, GeometricSignature] = {}

        # Cached SDRs
        self._text_sdrs: dict[str, np.ndarray] = {}

    @classmethod
    def from_checkpoints(cls, embed_model, flow_model=None,
                         device="cpu", embed_model_name="all-MiniLM-L6-v2"):
        """Create from TDGA checkpoints."""
        from sentence_transformers import SentenceTransformer

        pipeline = TDGAPipeline.from_checkpoint(embed_model, flow_model, device)
        st_model = SentenceTransformer(embed_model_name, device=device)

        def encoder_fn(texts):
            embs = st_model.encode(texts, convert_to_tensor=True,
                                   show_progress_bar=False)
            return embs.to(device)

        return cls(pipeline, encoder_fn, device)

    def _embed_texts(self, texts):
        embs = self.encoder_fn(texts)
        if not isinstance(embs, torch.Tensor):
            embs = torch.tensor(embs, dtype=torch.float32)
        return embs.to(self.device)

    def _embed_to_sdr(self, embs):
        indices = self.pipeline.embed_to_sdr_indices(embs)
        return [indices[i].cpu().numpy().astype(np.uint32) for i in range(indices.shape[0])]

    # ── Store ───────────────────────────────────────────────────────────

    def store(self, fact: Fact):
        """Store a fact in both triadic memory and geometric index."""
        texts = [fact.subject, fact.relation, fact.object]
        fact_text = f"{fact.subject} {fact.relation} {fact.object}"
        all_texts = texts + [fact_text]

        embs = self._embed_texts(all_texts)
        sdrs = self._embed_to_sdr(embs[:3])
        s_sdr, r_sdr, o_sdr = sdrs

        # Triadic store
        self.triadic.store(s_sdr, r_sdr, o_sdr)

        # Cache embeddings as numpy
        s_emb = embs[0].detach().cpu().numpy()
        r_emb = embs[1].detach().cpu().numpy()
        o_emb = embs[2].detach().cpu().numpy()
        f_emb = embs[3].detach().cpu().numpy()

        stored = StoredFact(
            fact=fact,
            subject_sdr=s_sdr, relation_sdr=r_sdr, object_sdr=o_sdr,
            subject_emb=s_emb, relation_emb=r_emb, object_emb=o_emb,
            fact_emb=f_emb,
        )
        self.facts.append(stored)

        # Add to subject's geometric signature
        subj = fact.subject
        if subj not in self._subject_sigs:
            self._subject_sigs[subj] = GeometricSignature()
        # Encode (subject, object) as a line — the relational pattern
        self._subject_sigs[subj].add_line(s_emb, o_emb)

    def store_batch(self, facts: list[Fact]):
        """Store multiple facts efficiently."""
        if not facts:
            return

        all_texts = []
        for f in facts:
            all_texts.extend([f.subject, f.relation, f.object])
            all_texts.append(f"{f.subject} {f.relation} {f.object}")

        embs = self._embed_texts(all_texts)
        sdrs_tensor = self.pipeline.embed_to_sdr_indices(embs)

        for i, fact in enumerate(facts):
            base = i * 4
            s_sdr = sdrs_tensor[base].cpu().numpy().astype(np.uint32)
            r_sdr = sdrs_tensor[base+1].cpu().numpy().astype(np.uint32)
            o_sdr = sdrs_tensor[base+2].cpu().numpy().astype(np.uint32)

            self.triadic.store(s_sdr, r_sdr, o_sdr)

            s_emb = embs[base].detach().cpu().numpy()
            r_emb = embs[base+1].detach().cpu().numpy()
            o_emb = embs[base+2].detach().cpu().numpy()
            f_emb = embs[base+3].detach().cpu().numpy()

            stored = StoredFact(
                fact=fact,
                subject_sdr=s_sdr, relation_sdr=r_sdr, object_sdr=o_sdr,
                subject_emb=s_emb, relation_emb=r_emb, object_emb=o_emb,
                fact_emb=f_emb,
            )
            self.facts.append(stored)

            subj = fact.subject
            if subj not in self._subject_sigs:
                self._subject_sigs[subj] = GeometricSignature()
            self._subject_sigs[subj].add_line(s_emb, o_emb)

    def finalize_signatures(self):
        """Compute Gram matrices for all subjects. Call after batch loading."""
        for sig in self._subject_sigs.values():
            sig.finalize()

    # ── Triadic Recall ──────────────────────────────────────────────────

    def recall(self, subject: str = None, relation: str = None,
               object: str = None) -> list[tuple[StoredFact, float]]:
        """
        Exact triadic recall: given 2 of 3, recall the 3rd.
        Returns matching stored facts ranked by SDR overlap.
        """
        # Determine which element is missing
        given = [subject, relation, object]
        missing_idx = [i for i, x in enumerate(given) if x is None]
        present = [x for x in given if x is not None]

        if len(missing_idx) != 1 or len(present) != 2:
            raise ValueError("Provide exactly 2 of 3 (subject, relation, object)")

        # Encode the two known elements
        embs = self._embed_texts(present)
        sdrs = self._embed_to_sdr(embs)

        # Query triadic memory
        if missing_idx[0] == 0:  # missing subject
            recalled_sdr = self.triadic.query(None, sdrs[0], sdrs[1])
        elif missing_idx[0] == 1:  # missing relation
            recalled_sdr = self.triadic.query(sdrs[0], None, sdrs[1])
        else:  # missing object
            recalled_sdr = self.triadic.query(sdrs[0], sdrs[1], None)

        if recalled_sdr is None or len(recalled_sdr) == 0:
            return []

        # Match against stored facts
        results = []
        field_names = ["subject_sdr", "relation_sdr", "object_sdr"]
        target_field = field_names[missing_idx[0]]

        for sf in self.facts:
            stored_sdr = getattr(sf, target_field)
            overlap = len(np.intersect1d(recalled_sdr, stored_sdr))
            if overlap > 0:
                results.append((sf, float(overlap)))

        results.sort(key=lambda x: -x[1])
        return results

    # ── Cosine Search ───────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[tuple[StoredFact, float]]:
        """Cosine similarity search against all stored fact embeddings."""
        if not self.facts:
            return []

        q_emb = self._embed_texts([query])[0]
        fact_embs = torch.stack([
            torch.tensor(sf.fact_emb) for sf in self.facts
        ]).to(self.device)

        sims = F.cosine_similarity(q_emb.unsqueeze(0), fact_embs)
        top_k = min(k, len(self.facts))
        values, indices = torch.topk(sims, top_k)

        return [
            (self.facts[idx], float(val))
            for idx, val in zip(indices.cpu(), values.cpu())
        ]

    # ── Geometric Similarity ───────────────────────────────────────────

    def geometric_similar(self, subject: str,
                          k: int = 5) -> list[tuple[str, float]]:
        """
        Find subjects with the most similar relational patterns.
        Uses multi-seed Gram matrix cosine similarity.
        """
        if subject not in self._subject_sigs:
            return []

        query_sig = self._subject_sigs[subject]
        results = []
        for other_subj, other_sig in self._subject_sigs.items():
            if other_subj == subject:
                continue
            sim = query_sig.similarity(other_sig)
            results.append((other_subj, sim))

        results.sort(key=lambda x: -x[1])
        return results[:k]

    # ── Geometric Expansion ─────────────────────────────────────────────

    def expand(self, subject: str, candidates: list[str] = None,
               k: int = 10) -> list[tuple[str, float]]:
        """
        Generative expansion: given stored facts about a subject,
        score candidate objects by geometric consistency.

        Uses multi-seed Gram^0.05 to find new objects that fit the
        relational pattern established by existing facts.
        """
        if subject not in self._subject_sigs:
            return []

        sig = self._subject_sigs[subject]
        if not sig.grams_powered:
            sig.finalize()
            if not sig.grams_powered:
                return []

        # Get subject embedding
        s_emb = self._embed_texts([subject])[0].detach().cpu().numpy()

        # Default candidates: all unique objects in the store
        if candidates is None:
            candidates = list(set(
                sf.fact.object for sf in self.facts
                if sf.fact.subject != subject
            ))

        if not candidates:
            return []

        # Score each candidate
        c_embs = self._embed_texts(candidates)
        c_np = c_embs.detach().cpu().numpy()

        # Batch score across all seeds
        scores = np.zeros(len(candidates))
        for seed_idx, (W1, W2) in enumerate(PROJECTIONS):
            if seed_idx >= len(sig.grams_powered):
                break
            gram = sig.grams_powered[seed_idx]
            lines = batch_encode_lines_dual(s_emb, c_np, W1, W2)
            scores += np.sum((lines @ gram) * lines, axis=1)

        scores /= len(sig.grams_powered)

        ranked = sorted(zip(scores.tolist(), candidates), reverse=True)
        return [(word, score) for score, word in ranked[:k]]

    # ── Hybrid Search ───────────────────────────────────────────────────

    def hybrid_search(self, query: str, k: int = 10) -> list[tuple[StoredFact, float, str]]:
        """
        Combined search: cosine similarity + triadic expansion.

        Returns (fact, score, method) tuples.
        """
        results = {}

        # 1. Cosine search
        cosine_results = self.search(query, k=k)
        for sf, sim in cosine_results:
            key = (sf.fact.subject, sf.fact.relation, sf.fact.object)
            results[key] = (sf, sim, "cosine")

        # 2. For each cosine hit, try triadic expansion
        for sf, sim in cosine_results[:3]:
            # Try recalling with subject + relation
            try:
                triadic_results = self.recall(
                    subject=sf.fact.subject,
                    relation=sf.fact.relation)
                for tf, overlap in triadic_results[:3]:
                    key = (tf.fact.subject, tf.fact.relation, tf.fact.object)
                    if key not in results:
                        results[key] = (tf, overlap / 50, "triadic")
            except (ValueError, RuntimeError):
                pass

        ranked = sorted(results.values(), key=lambda x: -x[1])
        return [(sf, score, method) for sf, score, method in ranked[:k]]

    # ── Info ────────────────────────────────────────────────────────────

    @property
    def n_facts(self):
        return len(self.facts)

    @property
    def n_subjects(self):
        return len(self._subject_sigs)

    def subjects(self):
        return list(self._subject_sigs.keys())

    def facts_for(self, subject: str) -> list[StoredFact]:
        return [sf for sf in self.facts if sf.fact.subject == subject]
