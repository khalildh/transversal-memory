#!/usr/bin/env python3
"""
exp_triadic_gram.py — Test: can triadic memory store and recall Gram matrices?

Minimal test:
1. Generate synthetic Gram matrices from different "topics"
2. Encode (context, layer, gram) as SDRs via random projection
3. Store in triadic memory
4. Query with new context from same topic → does it recall the right Gram?

Success = recalled Gram is more similar to same-topic Grams than cross-topic.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Volumes/PRO-G40/Code/TDGA/src")
from tdga.torch_indicator_memory import TorchIndicatorMemory

# ── SDR encoding via random projection ──────────────────────────────────

N = 5000   # SDR dimensionality
P = 50     # active bits per SDR


class RandomProjectionSDR:
    """Encode continuous vectors as SDRs via random projection."""

    def __init__(self, input_dim, n=N, p=P, seed=42):
        self.n = n
        self.p = p
        rng = np.random.default_rng(seed)
        self.R = rng.standard_normal((input_dim, n)).astype(np.float32)
        # Normalize columns for stability
        norms = np.linalg.norm(self.R, axis=0, keepdims=True) + 1e-12
        self.R = self.R / norms
        # Pseudoinverse for decoding
        self.R_pinv = np.linalg.pinv(self.R)  # (n, input_dim)

    def encode(self, vec):
        """Continuous vector → SDR (top-P indices)."""
        projected = vec @ self.R  # (n,)
        indices = np.argsort(-projected)[:self.p]
        return np.sort(indices).astype(np.uint32)

    def decode(self, sdr_indices):
        """SDR indices → approximate continuous vector."""
        binary = np.zeros(self.n, dtype=np.float32)
        binary[sdr_indices] = 1.0
        return binary @ self.R_pinv  # (input_dim,)


# ── Synthetic Gram generation ───────────────────────────────────────────

def make_topic_grams(n_topics=5, n_samples=10, n_heads=6, noise=0.1, seed=0):
    """
    Generate synthetic Gram matrices from distinct "topics."

    Each topic has a characteristic Gram structure (eigenvector pattern).
    Samples within a topic are noisy versions of the template.
    """
    rng = np.random.default_rng(seed)
    topics = {}

    for t in range(n_topics):
        # Random PSD template Gram per head
        template = []
        for h in range(n_heads):
            A = rng.standard_normal((6, 3))  # low-rank structure
            M = A @ A.T + 0.1 * np.eye(6)
            M = M / np.linalg.norm(M)  # normalize
            template.append(M)

        # Fixed topic direction for context (consistent across samples)
        topic_dir = rng.standard_normal(192)
        topic_dir = topic_dir / (np.linalg.norm(topic_dir) + 1e-12)

        # Generate noisy samples
        samples = []
        contexts = []
        for s in range(n_samples):
            grams = []
            for h in range(n_heads):
                noise_mat = rng.standard_normal((6, 6)) * noise
                noise_mat = (noise_mat + noise_mat.T) / 2
                M = template[h] + noise_mat
                grams.append(M)
            # Flatten all heads: 6 * 36 = 216
            flat = np.concatenate([g.flatten() for g in grams])
            samples.append(flat)

            # Context = topic direction (strong) + noise (weak)
            ctx = topic_dir * 3.0 + rng.standard_normal(192) * 0.3
            contexts.append(ctx)

        topics[t] = {
            "template": template,
            "grams": np.array(samples),
            "contexts": np.array(contexts),
        }

    return topics


# ── Main test ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Triadic Gram Memory Test")
    print("Store (context, layer, gram) → recall gram from context+layer")
    print("=" * 60)

    # Generate data
    n_topics = 5
    n_samples = 10
    n_heads = 6
    gram_dim = n_heads * 36  # 216
    ctx_dim = 192
    n_layers = 4

    topics = make_topic_grams(n_topics, n_samples, n_heads)

    # Create SDR encoders
    ctx_encoder = RandomProjectionSDR(ctx_dim, seed=100)
    gram_encoder = RandomProjectionSDR(gram_dim, seed=200)

    # Fixed layer SDRs (deterministic, orthogonal)
    layer_sdrs = []
    rng = np.random.default_rng(300)
    for l in range(n_layers):
        # Each layer gets a unique random SDR
        perm = rng.permutation(N)
        layer_sdrs.append(np.sort(perm[:P]).astype(np.uint32))

    # Verify layer SDRs are distinct
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            overlap = len(np.intersect1d(layer_sdrs[i], layer_sdrs[j]))
            print(f"  Layer {i}↔{j} SDR overlap: {overlap} bits (expected ~0.5)")

    # ── Test 1: SDR encoding quality ──
    print(f"\n{'─'*60}")
    print("Test 1: SDR overlap reflects Gram similarity")
    print(f"{'─'*60}")

    # Same-topic samples should have higher SDR overlap than cross-topic
    same_overlaps = []
    cross_overlaps = []

    for t in range(n_topics):
        grams = topics[t]["grams"]
        sdrs = [gram_encoder.encode(g) for g in grams]

        for i in range(len(sdrs)):
            for j in range(i + 1, len(sdrs)):
                overlap = len(np.intersect1d(sdrs[i], sdrs[j]))
                same_overlaps.append(overlap)

    for t1 in range(n_topics):
        for t2 in range(t1 + 1, n_topics):
            for i in range(min(3, n_samples)):
                s1 = gram_encoder.encode(topics[t1]["grams"][i])
                s2 = gram_encoder.encode(topics[t2]["grams"][i])
                overlap = len(np.intersect1d(s1, s2))
                cross_overlaps.append(overlap)

    print(f"  Same-topic SDR overlap:  {np.mean(same_overlaps):.1f} ± {np.std(same_overlaps):.1f}")
    print(f"  Cross-topic SDR overlap: {np.mean(cross_overlaps):.1f} ± {np.std(cross_overlaps):.1f}")
    gap = np.mean(same_overlaps) - np.mean(cross_overlaps)
    print(f"  Gap: {gap:.1f} bits {'✓' if gap > 5 else '✗ too small'}")

    # ── Test 2: Gram reconstruction quality ──
    print(f"\n{'─'*60}")
    print("Test 2: SDR → Gram reconstruction quality")
    print(f"{'─'*60}")

    recon_cosines = []
    for t in range(n_topics):
        for s in range(n_samples):
            orig = topics[t]["grams"][s]
            sdr = gram_encoder.encode(orig)
            recon = gram_encoder.decode(sdr)
            cos = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-12)
            recon_cosines.append(cos)

    print(f"  Reconstruction cosine: {np.mean(recon_cosines):.4f} ± {np.std(recon_cosines):.4f}")

    # ── Test 3: Triadic store + recall ──
    print(f"\n{'─'*60}")
    print("Test 3: Triadic memory store and recall")
    print(f"{'─'*60}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    triadic = TorchIndicatorMemory(N=N, P=P, device=device, exact=True)

    # Store: use first 8 samples per topic as training, last 2 as test
    n_train = 8
    n_test = 2
    layer_idx = 0  # test with layer 0

    stored = 0
    for t in range(n_topics):
        for s in range(n_train):
            ctx_sdr = ctx_encoder.encode(topics[t]["contexts"][s])
            gram_sdr = gram_encoder.encode(topics[t]["grams"][s])
            triadic.store(ctx_sdr, layer_sdrs[layer_idx], gram_sdr)
            stored += 1

    print(f"  Stored {stored} (context, layer, gram) triples")

    # Recall: use test samples
    correct = 0
    total = 0
    recall_same_cos = []
    recall_cross_cos = []

    for t in range(n_topics):
        for s in range(n_train, n_train + n_test):
            # Query with test context + layer → recall gram
            ctx_sdr = ctx_encoder.encode(topics[t]["contexts"][s])
            recalled_sdr = triadic.query(ctx_sdr, layer_sdrs[layer_idx], None)

            if recalled_sdr is None or len(recalled_sdr) == 0:
                print(f"  Topic {t} sample {s}: no recall")
                continue

            # Convert recalled SDR to numpy
            if isinstance(recalled_sdr, torch.Tensor):
                recalled_sdr = recalled_sdr.cpu().numpy()

            # Decode recalled Gram
            recalled_gram = gram_encoder.decode(recalled_sdr)

            # Compare to same-topic template (mean of training grams)
            same_topic_mean = topics[t]["grams"][:n_train].mean(axis=0)
            cos_same = np.dot(recalled_gram, same_topic_mean) / (
                np.linalg.norm(recalled_gram) * np.linalg.norm(same_topic_mean) + 1e-12)
            recall_same_cos.append(cos_same)

            # Compare to other-topic templates
            for t2 in range(n_topics):
                if t2 == t:
                    continue
                other_mean = topics[t2]["grams"][:n_train].mean(axis=0)
                cos_other = np.dot(recalled_gram, other_mean) / (
                    np.linalg.norm(recalled_gram) * np.linalg.norm(other_mean) + 1e-12)
                recall_cross_cos.append(cos_other)

            # Is same-topic the best match?
            all_cos = []
            for t2 in range(n_topics):
                t_mean = topics[t2]["grams"][:n_train].mean(axis=0)
                c = np.dot(recalled_gram, t_mean) / (
                    np.linalg.norm(recalled_gram) * np.linalg.norm(t_mean) + 1e-12)
                all_cos.append((t2, c))
            all_cos.sort(key=lambda x: -x[1])
            if all_cos[0][0] == t:
                correct += 1
            total += 1

    print(f"\n  Recall → same-topic cosine:  {np.mean(recall_same_cos):.4f} ± {np.std(recall_same_cos):.4f}")
    print(f"  Recall → cross-topic cosine: {np.mean(recall_cross_cos):.4f} ± {np.std(recall_cross_cos):.4f}")
    print(f"  Topic identification: {correct}/{total} correct ({100*correct/max(total,1):.0f}%)")

    # ── Test 4: Cross-layer recall ──
    print(f"\n{'─'*60}")
    print("Test 4: Layer-specific recall (store layer 0, query layer 1)")
    print(f"{'─'*60}")

    # Query with wrong layer → should get poor recall
    cross_layer_cos = []
    for t in range(n_topics):
        for s in range(n_train, n_train + n_test):
            ctx_sdr = ctx_encoder.encode(topics[t]["contexts"][s])
            # Query with WRONG layer
            recalled_sdr = triadic.query(ctx_sdr, layer_sdrs[1], None)
            if recalled_sdr is None or len(recalled_sdr) == 0:
                continue
            if isinstance(recalled_sdr, torch.Tensor):
                recalled_sdr = recalled_sdr.cpu().numpy()
            recalled_gram = gram_encoder.decode(recalled_sdr)
            same_mean = topics[t]["grams"][:n_train].mean(axis=0)
            cos = np.dot(recalled_gram, same_mean) / (
                np.linalg.norm(recalled_gram) * np.linalg.norm(same_mean) + 1e-12)
            cross_layer_cos.append(cos)

    if cross_layer_cos:
        print(f"  Wrong-layer recall cosine: {np.mean(cross_layer_cos):.4f}")
        print(f"  (vs correct-layer:         {np.mean(recall_same_cos):.4f})")
    else:
        print(f"  Wrong-layer: no recall (good — layer SDR acts as filter)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  SDR encoding preserves similarity:  gap = {gap:.1f} bits")
    print(f"  Reconstruction quality:             cos = {np.mean(recon_cosines):.4f}")
    print(f"  Triadic recall accuracy:            {correct}/{total} ({100*correct/max(total,1):.0f}%)")
    if recall_same_cos and recall_cross_cos:
        print(f"  Same vs cross topic recall:         {np.mean(recall_same_cos):.3f} vs {np.mean(recall_cross_cos):.3f}")

    verdict = correct / max(total, 1) > 0.5 and gap > 3
    print(f"\n  {'✓ WORKS' if verdict else '✗ NEEDS WORK'} — triadic Gram memory is {'viable' if verdict else 'not yet viable'}")


if __name__ == "__main__":
    main()
