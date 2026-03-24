"""
exp_arc_gram_hist_solve.py — ARC solver: Gram decoder → transversal scoring.

Pipeline:
  1. Gram transport: predict output Gram from input Gram (instant)
  2. Gram decoder: generate N candidate grids from predicted Gram (neural)
  3. Histogram+color transversal: score candidates against training pairs
  4. Rank 1 = answer

No brute force needed — works with any number of colors.

Usage:
  uv run python exp_arc_gram_hist_solve.py --task 0d3d703e
"""

import json
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transversal_memory import P3Memory
from transversal_memory.plucker import line_from_points

N_COLORS = 10
J6 = np.array([
    [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -1, 0], [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0], [0, -1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
], dtype=float)


# ── Spatial Gram ─────────────────────────────────────────────────────────────

def spatial_lines_3d(grid, z=1.0):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    lines = []
    for r in range(H):
        for c in range(W - 1):
            a = np.array([1., c/(max(W-1, 1)), r/(max(H-1, 1)), z + grid[r, c]/9.])
            b = np.array([1., (c+1)/(max(W-1, 1)), r/(max(H-1, 1)), z + grid[r, c+1]/9.])
            lines.append(line_from_points(a, b))
    for r in range(H - 1):
        for c in range(W):
            a = np.array([1., c/(max(W-1, 1)), r/(max(H-1, 1)), z + grid[r, c]/9.])
            b = np.array([1., c/(max(W-1, 1)), (r+1)/(max(H-1, 1)), z + grid[r+1, c]/9.])
            lines.append(line_from_points(a, b))
    return np.array(lines)


def gram_vec(grid, z=1.0):
    L = spatial_lines_3d(np.array(grid, dtype=int), z)
    M = L.T @ L
    return M[np.triu_indices(6)]


# ── Gram Decoder ─────────────────────────────────────────────────────────────

class GramDecoder(nn.Module):
    def __init__(self, H, W, n_colors=10, d_hidden=128, n_refine=3):
        super().__init__()
        self.H, self.W, self.n_colors = H, W, n_colors
        self.gram_enc = nn.Sequential(
            nn.Linear(21, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden))
        self.pos_enc = nn.Sequential(
            nn.Linear(2, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden))
        self.refine_blocks = nn.ModuleList()
        for _ in range(n_refine):
            self.refine_blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(d_hidden),
                'attn': nn.MultiheadAttention(d_hidden, num_heads=4, batch_first=True),
                'ln2': nn.LayerNorm(d_hidden),
                'ffn': nn.Sequential(nn.Linear(d_hidden, 4*d_hidden), nn.GELU(),
                                     nn.Linear(4*d_hidden, d_hidden)),
            }))
        self.head = nn.Linear(d_hidden, n_colors)
        positions = [[r/max(H-1, 1), c/max(W-1, 1)] for r in range(H) for c in range(W)]
        self.register_buffer('positions', torch.tensor(positions, dtype=torch.float32))

    def forward(self, gram):
        B = gram.shape[0]
        N = self.H * self.W
        g = self.gram_enc(gram).unsqueeze(1).expand(-1, N, -1)
        p = self.pos_enc(self.positions).unsqueeze(0).expand(B, -1, -1)
        x = g + p
        for block in self.refine_blocks:
            x_n = block['ln1'](x)
            a, _ = block['attn'](x_n, x_n, x_n)
            x = x + a
            x = x + block['ffn'](block['ln2'](x))
        return self.head(x)


def train_decoder(H, W, n_colors=10, n_steps=30000, device='cpu'):
    """Train Gram decoder on random grids."""
    model = GramDecoder(H, W, n_colors).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"  Training decoder ({sum(p.numel() for p in model.parameters()):,} params)...")
    t0 = time.time()

    for step in range(1, n_steps + 1):
        grids = np.random.randint(0, n_colors, (64, H, W))
        grams = np.array([gram_vec(g) for g in grids])
        grams_t = torch.tensor(grams, dtype=torch.float32, device=device)
        grids_t = torch.tensor(grids.reshape(64, H*W), dtype=torch.long, device=device)

        logits = model(grams_t)
        loss = F.cross_entropy(logits.reshape(-1, n_colors), grids_t.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 5000 == 0:
            model.eval()
            with torch.no_grad():
                g_e = np.random.randint(0, n_colors, (256, H, W))
                gv_e = np.array([gram_vec(g) for g in g_e])
                pred = model(torch.tensor(gv_e, dtype=torch.float32, device=device)).argmax(-1).cpu().numpy()
                cell = np.mean(pred == g_e.reshape(256, H*W))
                grid = np.mean([np.array_equal(pred[i], g_e[i].flatten()) for i in range(256)])
            print(f"    step {step}: loss={loss.item():.3f} cell={cell:.3f} grid={grid:.3f} ({time.time()-t0:.0f}s)")
            model.train()

    return model


def generate_candidates(model, gram_vec_pred, n_candidates=200, temperature=1.0, device='cpu'):
    """Generate diverse candidate grids from predicted Gram via sampling."""
    model.eval()
    gram_t = torch.tensor(gram_vec_pred, dtype=torch.float32, device=device).unsqueeze(0)

    candidates = set()
    with torch.no_grad():
        # Greedy decode
        logits = model(gram_t)
        greedy = logits.argmax(-1)[0].cpu().numpy()
        candidates.add(tuple(greedy.tolist()))

        # Temperature sampling
        for temp in [0.3, 0.5, 0.8, 1.0, 1.5]:
            for _ in range(n_candidates // 5):
                probs = F.softmax(logits[0] / temp, dim=-1)
                sample = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()
                candidates.add(tuple(sample.tolist()))

    return [np.array(list(c)).reshape(model.H, model.W) for c in candidates]


# ── Histogram+Color Transversal Scoring ──────────────────────────────────────

def make_line_dual(sv, tv, W1, W2):
    combined = np.concatenate([sv, tv])
    p1 = W1 @ combined
    p2 = W2 @ combined
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    L = np.array([p1[i]*p2[j] - p1[j]*p2[i] for i, j in pairs])
    n = np.linalg.norm(L)
    return L / n if n > 1e-10 else None


def hist_color_embedding(r, c, in_c, out_c, inp_grid, out_grid, H, W):
    in_oh = np.zeros(N_COLORS); in_oh[in_c] = 1.0
    out_oh = np.zeros(N_COLORS); out_oh[out_c] = 1.0
    ih = np.array([np.sum(inp_grid == c) for c in range(N_COLORS)], dtype=float)
    oh = np.array([np.sum(out_grid == c) for c in range(N_COLORS)], dtype=float)
    diff = (oh - ih) / max(inp_grid.size, 1)
    return np.concatenate([in_oh, out_oh, diff])


def grid_pair_to_lines(inp, out, W1, W2, H, W):
    inp, out = np.array(inp), np.array(out)
    lines = []
    for r in range(H):
        for c in range(W):
            ea = hist_color_embedding(r, c, inp[r, c], out[r, c], inp, out, H, W)
            for dr, dc in [(0, 1), (1, 0)]:
                r2, c2 = r + dr, c + dc
                if r2 < H and c2 < W:
                    eb = hist_color_embedding(r2, c2, inp[r2, c2], out[r2, c2], inp, out, H, W)
                    L = make_line_dual(ea, eb, W1, W2)
                    if L is not None:
                        lines.append(L)
    return lines


def compute_transversals(lines, n_trans=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if len(lines) < 4:
        return []
    trans = []
    att = 0
    while len(trans) < n_trans and att < n_trans * 10:
        att += 1
        idx = rng.choice(len(lines), size=4, replace=False)
        mem = P3Memory()
        mem.store([lines[idx[i]] for i in range(3)])
        for T, res in mem.query_generative(lines[idx[3]]):
            n = np.linalg.norm(T)
            if n > 1e-10 and res < 1e-6:
                trans.append(T / n)
    return trans


def score_candidate(inp, cand, per_pair_trans, W1t, W2t, H, W):
    """Score a single candidate against training transversals."""
    cl = grid_pair_to_lines(inp, cand, W1t, W2t, H, W)
    if not cl:
        return 0.0
    Lm = np.stack(cl)
    pair_scores = []
    for trans in per_pair_trans:
        if trans:
            Tm = np.stack(trans)
            inner = Lm @ J6 @ Tm.T
            pair_scores.append(np.sum(np.log(np.abs(inner) + 1e-10)))
    return min(pair_scores) if pair_scores else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="0d3d703e")
    parser.add_argument("--n-candidates", type=int, default=500)
    parser.add_argument("--decoder-steps", type=int, default=30000)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(f"data/ARC-AGI/data/training/{args.task}.json") as f:
        task = json.load(f)

    H = len(task['train'][0]['input'])
    W = len(task['train'][0]['input'][0])
    test_inp = np.array(task['test'][0]['input'])
    test_out = np.array(task['test'][0]['output'])
    used_colors = sorted(set(
        c for p in task['train'] + task['test']
        for g in [p['input'], p['output']]
        for row in g for c in row
    ))

    print(f"Task: {args.task}, {H}x{W}, {len(used_colors)} colors")
    print(f"  Input:  {test_inp.flatten().tolist()}")
    print(f"  Output: {test_out.flatten().tolist()}")

    # Step 1: Gram transport
    print(f"\n=== Step 1: Gram transport ===")
    X = np.array([gram_vec(np.array(p['input']), z=1.0) for p in task['train']])
    Y = np.array([gram_vec(np.array(p['output']), z=3.0) for p in task['train']])
    Wt = np.linalg.lstsq(X, Y, rcond=None)[0]
    pred_gram = gram_vec(test_inp, z=1.0) @ Wt
    actual_gram = gram_vec(test_out, z=3.0)
    gram_err = np.linalg.norm(pred_gram - actual_gram) / np.linalg.norm(actual_gram)
    print(f"  Predicted Gram error: {gram_err:.4f}")

    # Step 2: Train decoder and generate candidates
    print(f"\n=== Step 2: Gram decoder → {args.n_candidates} candidates ===")
    decoder = train_decoder(H, W, n_colors=N_COLORS, n_steps=args.decoder_steps, device=device)
    candidates = generate_candidates(decoder, pred_gram, args.n_candidates, device=device)
    print(f"  Generated {len(candidates)} unique candidates")

    # Check if correct answer is among candidates
    correct_in_candidates = any(np.array_equal(c, test_out) for c in candidates)
    print(f"  Correct answer in candidates: {correct_in_candidates}")

    # Step 3: Transversal scoring
    print(f"\n=== Step 3: Histogram+color transversal scoring ===")
    emb_dim = 3 * N_COLORS
    rng_proj = np.random.RandomState(88)
    W1t = rng_proj.randn(4, 2 * emb_dim) * 0.1
    W2t = rng_proj.randn(4, 2 * emb_dim) * 0.1

    per_pair_trans = []
    for i, pair in enumerate(task['train']):
        lines = grid_pair_to_lines(pair['input'], pair['output'], W1t, W2t, H, W)
        trans = compute_transversals(lines, 20, np.random.default_rng(42 + i))
        per_pair_trans.append(trans)

    # Score all candidates
    scored = []
    for cand in candidates:
        s = score_candidate(test_inp, cand, per_pair_trans, W1t, W2t, H, W)
        scored.append((s, cand))

    scored.sort(key=lambda x: x[0])

    # Results
    print(f"\n=== Results ===")
    best = scored[0][1]
    match = np.array_equal(best, test_out)
    print(f"  Best candidate: {best.flatten().tolist()}")
    print(f"  Actual output:  {test_out.flatten().tolist()}")
    print(f"  Match: {'SOLVED ✓' if match else 'NO'}")

    if correct_in_candidates:
        for i, (s, c) in enumerate(scored):
            if np.array_equal(c, test_out):
                print(f"  Correct answer rank: {i+1}/{len(scored)}")
                break
    else:
        print(f"  Correct answer not in decoder candidates")

    print(f"\n  Top 5:")
    for i in range(min(5, len(scored))):
        s, c = scored[i]
        m = '✓' if np.array_equal(c, test_out) else ''
        print(f"    {i+1}: {c.flatten().tolist()} {m}")


if __name__ == "__main__":
    main()
