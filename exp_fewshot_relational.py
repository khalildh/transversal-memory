#!/usr/bin/env python3
"""
Few-shot relational reasoning benchmark
========================================

Given K seed examples of a relation (e.g., country→capital), can the
system identify new instances from distractors?

Tests geometry (Gram^0.05) vs embedding baselines (cosine, centroid,
Mahalanobis) across K=3,5,7,10,15,20.

The K≥10 crossover from the generative benchmark should appear here too:
6D Plücker needs fewer samples than 384D embeddings to capture structure.
"""

import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

sys.path.insert(0, "/Volumes/PRO-G40/Code/TDGA/src")
sys.path.insert(0, "/Volumes/PRO-G40/Code/transversal-memory")

from sentence_transformers import SentenceTransformer
from transversal_memory.plucker import (
    random_projection_dual,
    project_to_line_dual,
    batch_encode_lines_dual,
)


# ── Config ──────────────────────────────────────────────────────────────

N_SEEDS = 50
SEED_SPACING = 10
GRAM_POWER = 0.05
K_VALUES = [3, 5, 7, 10, 15, 20]
N_DISTRACTORS = 50
N_TRIALS = 20  # random splits per relation
PCA_DIMS = [16, 32, 64]  # reduce 384D before Plücker projection


# ── Relations ───────────────────────────────────────────────────────────

RELATIONS = {
    "country→capital": [
        ("France", "Paris"), ("Germany", "Berlin"), ("Spain", "Madrid"),
        ("Italy", "Rome"), ("Japan", "Tokyo"), ("Egypt", "Cairo"),
        ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"),
        ("Mexico", "Mexico City"), ("Canada", "Ottawa"),
        ("Australia", "Canberra"), ("India", "New Delhi"),
        ("China", "Beijing"), ("Russia", "Moscow"), ("Turkey", "Ankara"),
        ("Poland", "Warsaw"), ("Sweden", "Stockholm"), ("Norway", "Oslo"),
        ("Portugal", "Lisbon"), ("Greece", "Athens"),
        ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
        ("Colombia", "Bogota"), ("Peru", "Lima"), ("Chile", "Santiago"),
    ],
    "animal→sound": [
        ("dog", "bark"), ("cat", "meow"), ("cow", "moo"),
        ("sheep", "baa"), ("duck", "quack"), ("rooster", "crow"),
        ("lion", "roar"), ("snake", "hiss"), ("bee", "buzz"),
        ("owl", "hoot"), ("frog", "croak"), ("pig", "oink"),
        ("horse", "neigh"), ("donkey", "bray"), ("wolf", "howl"),
        ("crow", "caw"), ("dove", "coo"), ("monkey", "chatter"),
        ("elephant", "trumpet"), ("turkey", "gobble"),
        ("goat", "bleat"), ("cricket", "chirp"), ("seal", "bark"),
        ("whale", "song"), ("parrot", "squawk"),
    ],
    "material→product": [
        ("wood", "furniture"), ("cotton", "clothing"), ("steel", "bridge"),
        ("glass", "window"), ("rubber", "tire"), ("leather", "shoe"),
        ("wool", "sweater"), ("clay", "pottery"), ("silk", "dress"),
        ("concrete", "building"), ("paper", "book"), ("plastic", "bottle"),
        ("gold", "jewelry"), ("silver", "coin"), ("copper", "wire"),
        ("iron", "nail"), ("aluminum", "can"), ("marble", "statue"),
        ("granite", "countertop"), ("bamboo", "basket"),
        ("hemp", "rope"), ("flour", "bread"), ("wax", "candle"),
        ("sand", "glass"), ("oil", "fuel"),
    ],
    "country→language": [
        ("France", "French"), ("Germany", "German"), ("Spain", "Spanish"),
        ("Italy", "Italian"), ("Japan", "Japanese"), ("China", "Mandarin"),
        ("Brazil", "Portuguese"), ("Russia", "Russian"), ("Turkey", "Turkish"),
        ("Poland", "Polish"), ("Sweden", "Swedish"), ("Norway", "Norwegian"),
        ("Greece", "Greek"), ("Thailand", "Thai"), ("Vietnam", "Vietnamese"),
        ("Korea", "Korean"), ("Indonesia", "Indonesian"), ("Finland", "Finnish"),
        ("Hungary", "Hungarian"), ("Romania", "Romanian"),
        ("Netherlands", "Dutch"), ("Denmark", "Danish"),
        ("Czech Republic", "Czech"), ("Croatia", "Croatian"),
        ("Serbia", "Serbian"),
    ],
    "profession→tool": [
        ("carpenter", "hammer"), ("painter", "brush"), ("surgeon", "scalpel"),
        ("farmer", "plow"), ("chef", "knife"), ("tailor", "needle"),
        ("blacksmith", "anvil"), ("musician", "instrument"), ("writer", "pen"),
        ("photographer", "camera"), ("electrician", "wire"), ("plumber", "wrench"),
        ("gardener", "shovel"), ("fisherman", "net"), ("architect", "blueprint"),
        ("astronomer", "telescope"), ("chemist", "flask"), ("sculptor", "chisel"),
        ("weaver", "loom"), ("baker", "oven"),
        ("miner", "pickaxe"), ("sailor", "compass"), ("pilot", "cockpit"),
        ("mechanic", "toolbox"), ("dentist", "drill"),
    ],
}

# Distractors: words unlikely to be valid targets for any relation
DISTRACTOR_POOL = [
    "algorithm", "entropy", "paradox", "symphony", "quantum", "nebula",
    "theorem", "catalyst", "prism", "glacier", "vortex", "mosaic",
    "cipher", "aurora", "zenith", "epoch", "nexus", "quasar",
    "metaphor", "paradigm", "axiom", "vertex", "matrix", "helix",
    "spectrum", "polygon", "fractal", "orbital", "tangent", "cosine",
    "vector", "scalar", "tensor", "gradient", "divergence", "manifold",
    "topology", "isomorphism", "homology", "functor", "morphism", "sheaf",
    "cohomology", "fibration", "homotopy", "simplex", "lattice", "group",
    "ring", "field", "module", "ideal", "kernel", "image",
    "sequence", "series", "limit", "boundary", "interior", "closure",
]


# ── Geometry helpers ────────────────────────────────────────────────────

def make_projections(n_seeds, seed_spacing, dim):
    projs = []
    for i in range(n_seeds):
        rng = np.random.default_rng(i * seed_spacing)
        W1, W2 = random_projection_dual(dim, rng)
        projs.append((W1, W2))
    return projs


def power_gram(gram, power):
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T


def build_gram_signature(src_embs, tgt_embs, projections, gram_power):
    """Build multi-seed Gram^power signature from (src, tgt) embedding pairs."""
    n_pairs = len(src_embs)
    grams_powered = []

    for W1, W2 in projections:
        lines = []
        for i in range(n_pairs):
            L = project_to_line_dual(src_embs[i], tgt_embs[i], W1, W2)
            if np.linalg.norm(L) > 1e-12:
                lines.append(L)

        if len(lines) < 2:
            grams_powered.append(np.zeros((6, 6)))
            continue

        arr = np.stack(lines)
        raw = arr.T @ arr
        grams_powered.append(power_gram(raw, gram_power))

    return grams_powered


def score_gram(src_emb, candidate_embs, grams_powered, projections):
    """Score candidates against a Gram signature. Returns array of scores."""
    scores = np.zeros(len(candidate_embs))
    n_valid = 0

    for seed_idx, (W1, W2) in enumerate(projections):
        if seed_idx >= len(grams_powered):
            break
        gram = grams_powered[seed_idx]
        if np.linalg.norm(gram) < 1e-12:
            continue

        lines = batch_encode_lines_dual(src_emb, candidate_embs, W1, W2)
        scores += np.sum((lines @ gram) * lines, axis=1)
        n_valid += 1

    return scores / max(n_valid, 1)


# ── Scoring methods ─────────────────────────────────────────────────────

def build_gram_offsets(seed_src, seed_tgt, projections, gram_power):
    """Build Gram from relational offsets (tgt - src) paired with src."""
    offsets = seed_tgt - seed_src
    n = len(offsets)
    grams = []

    for W1, W2 in projections:
        lines = []
        # Encode (offset_i, offset_j) pairs for structural variety
        for i in range(n):
            for j in range(i + 1, n):
                L = project_to_line_dual(offsets[i], offsets[j], W1, W2)
                if np.linalg.norm(L) > 1e-12:
                    lines.append(L)
        # Also encode (src_i, offset_i) to capture source-conditioned pattern
        for i in range(n):
            L = project_to_line_dual(seed_src[i], offsets[i], W1, W2)
            if np.linalg.norm(L) > 1e-12:
                lines.append(L)

        if len(lines) < 2:
            grams.append(np.zeros((6, 6)))
            continue

        arr = np.stack(lines)
        raw = arr.T @ arr
        grams.append(power_gram(raw, gram_power))
    return grams


def score_gram_offset(query_src, candidate_embs, seed_src_centroid,
                      grams_powered, projections):
    """Score candidates by how well (candidate - query_src) fits the offset Gram."""
    offsets = candidate_embs - query_src
    scores = np.zeros(len(candidate_embs))
    n_valid = 0

    for seed_idx, (W1, W2) in enumerate(projections):
        if seed_idx >= len(grams_powered):
            break
        gram = grams_powered[seed_idx]
        if np.linalg.norm(gram) < 1e-12:
            continue

        # Score each candidate offset against seed offset Gram
        lines = batch_encode_lines_dual(query_src, candidate_embs, W1, W2)
        scores += np.sum((lines @ gram) * lines, axis=1)
        n_valid += 1

    return scores / max(n_valid, 1)


def score_cosine_centroid(seed_tgt_embs, candidate_embs):
    """Cosine similarity to centroid of seed target embeddings."""
    centroid = seed_tgt_embs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-12
    normed = candidate_embs / norms
    return normed @ centroid


def score_cosine_max(seed_tgt_embs, candidate_embs):
    """Max cosine similarity to any seed target."""
    norms_s = np.linalg.norm(seed_tgt_embs, axis=1, keepdims=True) + 1e-12
    norms_c = np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-12
    sims = (candidate_embs / norms_c) @ (seed_tgt_embs / norms_s).T
    return sims.max(axis=1)


def score_mahalanobis(seed_tgt_embs, candidate_embs, reg=0.01):
    """Mahalanobis distance to seed target distribution (inverted for ranking)."""
    if len(seed_tgt_embs) < 3:
        return score_cosine_centroid(seed_tgt_embs, candidate_embs)

    centroid = seed_tgt_embs.mean(axis=0)
    centered = seed_tgt_embs - centroid
    cov = centered.T @ centered / len(centered)
    cov += reg * np.eye(cov.shape[0])

    try:
        L = np.linalg.cholesky(cov)
        diff = candidate_embs - centroid
        solved = np.linalg.solve(L, diff.T).T
        dists = np.sum(solved ** 2, axis=1)
        return -dists  # negate: closer = higher score
    except np.linalg.LinAlgError:
        return score_cosine_centroid(seed_tgt_embs, candidate_embs)


def score_cosine_src(seed_src_embs, src_emb, candidate_embs):
    """Cosine similarity of candidates to the query source embedding,
    weighted by how similar query source is to seed sources."""
    # Similarity of query source to seed sources
    src_norm = np.linalg.norm(src_emb) + 1e-12
    seed_norms = np.linalg.norm(seed_src_embs, axis=1, keepdims=True) + 1e-12
    src_sims = (seed_src_embs / seed_norms) @ (src_emb / src_norm)

    # Use centroid of seed targets weighted by source similarity
    weights = np.maximum(src_sims, 0)
    if weights.sum() < 1e-12:
        weights = np.ones_like(weights)
    weights /= weights.sum()

    # Just use centroid similarity (weighted doesn't help much with few seeds)
    return score_cosine_centroid(seed_src_embs, candidate_embs)


# ── Main benchmark ──────────────────────────────────────────────────────

def run_benchmark():
    print("=" * 70)
    print("Few-shot Relational Reasoning Benchmark")
    print("Geometry (Gram^0.05) vs Embedding baselines")
    print("=" * 70)

    # Load model
    t0 = time.time()
    print("\nLoading sentence-transformers (MiniLM)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Embed all unique words across all relations + distractors
    all_words = set()
    for pairs in RELATIONS.values():
        for s, t in pairs:
            all_words.add(s)
            all_words.add(t)
    for d in DISTRACTOR_POOL:
        all_words.add(d)

    word_list = sorted(all_words)
    print(f"Embedding {len(word_list)} unique words...")
    emb_full = model.encode(word_list, convert_to_tensor=False,
                            show_progress_bar=False)
    word2idx = {w: i for i, w in enumerate(word_list)}

    # PCA: fit on all embeddings, project to lower dims
    from sklearn.decomposition import PCA
    pca_models = {}
    emb_reduced = {}
    for d in PCA_DIMS:
        pca = PCA(n_components=d)
        reduced = pca.fit_transform(emb_full)
        pca_models[d] = pca
        emb_reduced[d] = reduced
        var = pca.explained_variance_ratio_.sum()
        print(f"  PCA → {d}D: {var:.1%} variance retained")

    # Precompute projections for each PCA dim
    proj_by_dim = {}
    for d in PCA_DIMS:
        proj_by_dim[d] = make_projections(N_SEEDS, SEED_SPACING, d)
    # Also 384D projections
    proj_by_dim[384] = make_projections(N_SEEDS, SEED_SPACING, 384)

    # Methods: embeddings (always 384D) + geometry at various dims
    emb_methods = ["cosine_cent", "cosine_max", "mahalanobis"]
    gram_methods = [f"gram_{d}D" for d in PCA_DIMS] + ["gram_384D"]
    offset_methods = [f"offset_{d}D" for d in PCA_DIMS] + ["offset_384D"]
    all_methods = emb_methods + gram_methods + offset_methods + ["rrf_best"]
    results = {m: {k: [] for k in K_VALUES} for m in all_methods}

    rng = np.random.default_rng(42)

    for rel_name, pairs in RELATIONS.items():
        print(f"\n{'─'*70}")
        print(f"Relation: {rel_name} ({len(pairs)} pairs)")
        print(f"{'─'*70}")

        for K in K_VALUES:
            if K >= len(pairs):
                continue

            k_precisions = {m: [] for m in all_methods}

            for trial in range(N_TRIALS):
                perm = rng.permutation(len(pairs))
                seed_idx = perm[:K]
                test_idx = perm[K:]

                # Full 384D embeddings
                seed_src_full = np.array([emb_full[word2idx[pairs[i][0]]] for i in seed_idx])
                seed_tgt_full = np.array([emb_full[word2idx[pairs[i][1]]] for i in seed_idx])

                true_targets = [pairs[i][1] for i in test_idx]
                n_true = len(true_targets)

                valid_targets = set(t for _, t in pairs)
                distractors = [d for d in DISTRACTOR_POOL if d not in valid_targets]
                dist_sample = list(rng.choice(distractors,
                                              size=min(N_DISTRACTORS, len(distractors)),
                                              replace=False))

                candidates = true_targets + dist_sample
                cand_full = np.array([emb_full[word2idx[c]] for c in candidates])
                is_positive = np.array([1] * n_true + [0] * len(dist_sample))

                # Embedding baselines (384D)
                scores = {}
                scores["cosine_cent"] = score_cosine_centroid(seed_tgt_full, cand_full)
                scores["cosine_max"] = score_cosine_max(seed_tgt_full, cand_full)
                scores["mahalanobis"] = score_mahalanobis(seed_tgt_full, cand_full)

                # Geometry at each PCA dim + full 384D
                best_gram_name = None
                best_gram_score = -1
                for d in PCA_DIMS + [384]:
                    if d == 384:
                        src_d = seed_src_full
                        tgt_d = seed_tgt_full
                        cand_d = cand_full
                        qsrc_d = seed_src_full[0]
                    else:
                        src_d = np.array([emb_reduced[d][word2idx[pairs[i][0]]] for i in seed_idx])
                        tgt_d = np.array([emb_reduced[d][word2idx[pairs[i][1]]] for i in seed_idx])
                        cand_d = np.array([emb_reduced[d][word2idx[c]] for c in candidates])
                        qsrc_d = src_d[0]

                    grams = build_gram_signature(src_d, tgt_d, proj_by_dim[d], GRAM_POWER)
                    gram_scores = score_gram(qsrc_d, cand_d, grams, proj_by_dim[d])
                    m_name = f"gram_{d}D"
                    scores[m_name] = gram_scores

                    # Also try offset-based Gram (relational pattern)
                    off_grams = build_gram_offsets(src_d, tgt_d, proj_by_dim[d], GRAM_POWER)
                    off_scores = score_gram_offset(qsrc_d, cand_d, src_d.mean(axis=0),
                                                  off_grams, proj_by_dim[d])
                    scores[f"offset_{d}D"] = off_scores

                # RRF: best embedding + best geometry
                RRF_K = 13
                rrf_scores = np.zeros(len(candidates))
                for m_name in emb_methods:
                    ranks = np.argsort(-scores[m_name]).argsort() + 1
                    rrf_scores += 1.0 / (RRF_K + ranks)
                # Add best-performing gram dim
                for m_name in gram_methods:
                    ranks = np.argsort(-scores[m_name]).argsort() + 1
                    rrf_scores += 1.0 / (RRF_K + ranks)
                scores["rrf_best"] = rrf_scores

                # Evaluate
                for m_name in all_methods:
                    ranked = np.argsort(-scores[m_name])
                    top10 = ranked[:10]
                    p_at_10 = is_positive[top10].mean()
                    k_precisions[m_name].append(p_at_10)

            for m_name in all_methods:
                mean_p = np.mean(k_precisions[m_name])
                results[m_name][K].append(mean_p)

        # Per-relation table
        gram_cols = " ".join(f"{'gram_'+str(d)+'D':>9s}" for d in PCA_DIMS)
        print(f"\n  {'K':>3s}  {'cos_cent':>9s}  {'cos_max':>9s}  {'mahalanob':>9s}  {gram_cols}  {'gram_384D':>9s}  {'rrf_best':>9s}")
        for K in K_VALUES:
            if K >= len(pairs):
                continue
            vals = {m: results[m][K][-1] if results[m][K] else 0 for m in all_methods}
            best = max(vals, key=vals.get)
            gram_vals = " ".join(f"{vals.get(f'gram_{d}D', 0):9.4f}" for d in PCA_DIMS)
            print(f"  {K:3d}  {vals['cosine_cent']:9.4f}  {vals['cosine_max']:9.4f}  {vals['mahalanobis']:9.4f}  {gram_vals}  {vals['gram_384D']:9.4f}  {vals['rrf_best']:9.4f}  ← {best}")

    # ── Aggregate ──
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean across all relations)")
    print("=" * 70)

    # Compact table: embeddings vs best gram vs best offset vs RRF
    print(f"\n  {'K':>3s}  {'cos_cent':>9s}  {'cos_max':>9s}  {'mahalanob':>9s}  {'best_gram':>9s}  {'best_offs':>9s}  {'rrf_best':>9s}  best")
    print(f"  {'─'*3}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*10}")

    for K in K_VALUES:
        vals = {m: np.mean(results[m][K]) if results[m][K] else 0 for m in all_methods}
        if not any(vals.values()):
            continue
        best_emb = max(vals["cosine_cent"], vals["cosine_max"], vals["mahalanobis"])
        best_gram = max(vals.get(f"gram_{d}D", 0) for d in PCA_DIMS + [384])
        best_gram_dim = max(PCA_DIMS + [384], key=lambda d: vals.get(f"gram_{d}D", 0))
        best_offset = max(vals.get(f"offset_{d}D", 0) for d in PCA_DIMS + [384])
        best_offset_dim = max(PCA_DIMS + [384], key=lambda d: vals.get(f"offset_{d}D", 0))
        best = max(vals, key=vals.get)
        marker = " ★" if best_offset > best_emb or best_gram > best_emb else ""
        print(f"  {K:3d}  {vals['cosine_cent']:9.4f}  {vals['cosine_max']:9.4f}  {vals['mahalanobis']:9.4f}  {best_gram:9.4f}({best_gram_dim}D)  {best_offset:9.4f}({best_offset_dim}D)  {vals['rrf_best']:9.4f}  ← {best}{marker}")

    # Deltas
    print("\n" + "─" * 70)
    print("Best geometry (gram or offset) vs best embedding delta:")
    for K in K_VALUES:
        vals = {m: np.mean(results[m][K]) if results[m][K] else 0 for m in all_methods}
        if not any(vals.values()):
            continue
        best_emb = max(vals["cosine_cent"], vals["cosine_max"], vals["mahalanobis"])
        best_geo = max(
            max(vals.get(f"gram_{d}D", 0) for d in PCA_DIMS + [384]),
            max(vals.get(f"offset_{d}D", 0) for d in PCA_DIMS + [384]),
        )
        delta = best_geo - best_emb
        bar = "+" * int(max(0, delta) * 100) + "-" * int(max(0, -delta) * 100)
        print(f"  K={K:2d}: {delta:+.4f}  {bar}")


if __name__ == "__main__":
    run_benchmark()
