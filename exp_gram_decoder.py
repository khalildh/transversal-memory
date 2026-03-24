"""
exp_gram_decoder.py — Train a small model to reconstruct grids from Plücker Grams.

The 6×6 Gram (21 upper-triangle values) captures spatial structure of a grid.
This trains a decoder: Gram + (H, W) → grid cells.

Training data is unlimited — generate random grids, compute their Grams.
Test whether the Gram is invertible in practice via learned decoding.

Usage:
  uv run python exp_gram_decoder.py              # train and eval
  uv run python exp_gram_decoder.py --grid-size 5 # 5x5 grids
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
from transversal_memory.plucker import line_from_points

# ── Plücker Gram computation ────────────────────────────────────────────────

def spatial_lines_3d(grid, z=1.0):
    H, W = grid.shape
    lines = []
    for r in range(H):
        for c in range(W - 1):
            a = np.array([1.0, c/(max(W-1,1)), r/(max(H-1,1)), z + grid[r,c]/9.0])
            b = np.array([1.0, (c+1)/(max(W-1,1)), r/(max(H-1,1)), z + grid[r,c+1]/9.0])
            lines.append(line_from_points(a, b))
    for r in range(H - 1):
        for c in range(W):
            a = np.array([1.0, c/(max(W-1,1)), r/(max(H-1,1)), z + grid[r,c]/9.0])
            b = np.array([1.0, c/(max(W-1,1)), (r+1)/(max(H-1,1)), z + grid[r+1,c]/9.0])
            lines.append(line_from_points(a, b))
    return np.array(lines)


def gram_vec(grid, z=1.0):
    """Compute 21-dim upper triangle of 6×6 Gram."""
    L = spatial_lines_3d(np.array(grid), z)
    M = L.T @ L
    return M[np.triu_indices(6)]


# ── Data generation ──────────────────────────────────────────────────────────

def make_batch(batch_size, H, W, n_colors=10, z=1.0):
    """Generate random grids and their Grams."""
    grids = np.random.randint(0, n_colors, (batch_size, H, W))
    grams = np.array([gram_vec(g, z) for g in grids])
    return (
        torch.tensor(grams, dtype=torch.float32),
        torch.tensor(grids, dtype=torch.long),
    )


# ── Decoder model ────────────────────────────────────────────────────────────

class GramDecoder(nn.Module):
    """Decode a 21-dim Gram vector into an H×W grid of colors.

    Architecture: Gram → per-cell features → self-attention (cells communicate)
    → MLP → color logits. Cells see each other to resolve ambiguity.
    """
    def __init__(self, H, W, n_colors=10, d_hidden=128, n_refine=3):
        super().__init__()
        self.H = H
        self.W = W
        self.n_colors = n_colors

        # Gram encoder: 21 → d_hidden
        self.gram_enc = nn.Sequential(
            nn.Linear(21, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Position encoding: (row_norm, col_norm) → d_hidden
        self.pos_enc = nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Self-attention refinement blocks (cells communicate)
        self.refine_blocks = nn.ModuleList()
        for _ in range(n_refine):
            self.refine_blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(d_hidden),
                'attn': nn.MultiheadAttention(d_hidden, num_heads=4, batch_first=True),
                'ln2': nn.LayerNorm(d_hidden),
                'ffn': nn.Sequential(
                    nn.Linear(d_hidden, 4 * d_hidden),
                    nn.GELU(),
                    nn.Linear(4 * d_hidden, d_hidden),
                ),
            }))

        # Output head
        self.head = nn.Linear(d_hidden, n_colors)

        # Precompute normalized positions
        positions = []
        for r in range(H):
            for c in range(W):
                positions.append([r / max(H-1, 1), c / max(W-1, 1)])
        self.register_buffer('positions', torch.tensor(positions, dtype=torch.float32))

    def forward(self, gram):
        """gram: (B, 21) → logits: (B, H*W, n_colors)"""
        B = gram.shape[0]
        N = self.H * self.W

        # Encode Gram → broadcast to all cells
        g_feat = self.gram_enc(gram)  # (B, d_hidden)
        g_expanded = g_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, d_hidden)

        # Encode positions
        pos_feat = self.pos_enc(self.positions)  # (N, d_hidden)
        pos_expanded = pos_feat.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_hidden)

        # Initial cell features = Gram + position
        x = g_expanded + pos_expanded  # (B, N, d_hidden)

        # Self-attention refinement: cells communicate
        for block in self.refine_blocks:
            x_norm = block['ln1'](x)
            attn_out, _ = block['attn'](x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + block['ffn'](block['ln2'](x))

        return self.head(x)  # (B, N, n_colors)


# ── Training ─────────────────────────────────────────────────────────────────

def train(H, W, n_colors=10, n_steps=5000, batch_size=64, lr=3e-4, device='cpu'):
    model = GramDecoder(H, W, n_colors).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  GramDecoder: {params:,} params (H={H}, W={W}, colors={n_colors})")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()

    for step in range(1, n_steps + 1):
        model.train()
        grams, grids = make_batch(batch_size, H, W, n_colors)
        grams = grams.to(device)
        grids = grids.to(device)

        logits = model(grams)  # (B, H*W, n_colors)
        targets = grids.reshape(-1, H * W)  # (B, H*W)
        loss = F.cross_entropy(logits.reshape(-1, n_colors), targets.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == 1:
            # Eval on fresh batch
            model.eval()
            with torch.no_grad():
                g_eval, grid_eval = make_batch(256, H, W, n_colors)
                g_eval, grid_eval = g_eval.to(device), grid_eval.to(device)
                logits_eval = model(g_eval)
                preds = logits_eval.argmax(dim=-1)  # (B, H*W)
                targets_eval = grid_eval.reshape(-1, H * W)

                cell_acc = (preds == targets_eval).float().mean().item()
                grid_acc = (preds == targets_eval).all(dim=1).float().mean().item()

            elapsed = time.time() - t0
            print(f"    step {step:5d}: loss={loss.item():.3f}  cell={cell_acc:.3f}  grid={grid_acc:.3f}  ({elapsed:.1f}s)")

    # Final eval
    model.eval()
    with torch.no_grad():
        g_eval, grid_eval = make_batch(1000, H, W, n_colors)
        g_eval, grid_eval = g_eval.to(device), grid_eval.to(device)
        logits_eval = model(g_eval)
        preds = logits_eval.argmax(dim=-1)
        targets_eval = grid_eval.reshape(-1, H * W)
        cell_acc = (preds == targets_eval).float().mean().item()
        grid_acc = (preds == targets_eval).all(dim=1).float().mean().item()

    print(f"\n  Final (1000 grids): cell_acc={cell_acc:.3f}, grid_acc={grid_acc:.3f}")

    # Show some examples
    print(f"\n  Examples:")
    for i in range(3):
        orig = grid_eval[i].cpu().numpy()
        pred = preds[i].reshape(H, W).cpu().numpy()
        match = np.array_equal(orig, pred)
        print(f"    Original:      {orig.flatten().tolist()}")
        print(f"    Reconstructed: {pred.flatten().tolist()}")
        print(f"    Match: {match}\n")

    return model


# ── ARC full pipeline ────────────────────────────────────────────────────────

def test_arc_pipeline(model, device, H=3, W=3):
    """Full pipeline: Gram transport (from demos) → decoder → predicted grid."""
    import json
    import os

    arc_dir = "data/ARC-AGI/data/training"
    if not os.path.exists(arc_dir):
        print("  No ARC data found, skipping pipeline test")
        return

    # Find 3x3 → 3x3 tasks
    tasks_3x3 = []
    for fname in sorted(os.listdir(arc_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(arc_dir, fname)) as f:
            task = json.load(f)
        all_match = all(
            len(p['input']) == H and len(p['input'][0]) == W and
            len(p['output']) == H and len(p['output'][0]) == W
            for p in task['train'] + task['test']
        )
        if all_match and len(task['train']) >= 2:
            tasks_3x3.append((fname.replace('.json', ''), task))

    if not tasks_3x3:
        print(f"  No {H}x{W} → {H}x{W} tasks found")
        return

    print(f"\n{'='*60}")
    print(f"  FULL PIPELINE: Gram transport → decoder ({len(tasks_3x3)} tasks)")
    print(f"{'='*60}")

    model.eval()
    total_tasks = 0
    correct_grids = 0
    total_cells = 0
    correct_cells = 0

    with torch.no_grad():
        for tname, task in tasks_3x3:
            # Step 1: Learn Gram transport from demos
            X, Y = [], []
            for pair in task['train']:
                X.append(gram_vec(pair['input'], z=1.0))
                Y.append(gram_vec(pair['output'], z=3.0))
            W = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0]

            for test in task['test']:
                inp = np.array(test['input'])
                out_actual = np.array(test['output'])

                # Step 2: Predict output Gram
                v_pred = gram_vec(inp, z=1.0) @ W

                # Step 3: Decode
                gram_t = torch.tensor(v_pred, dtype=torch.float32, device=device).unsqueeze(0)
                pred_grid = model(gram_t).argmax(-1)[0].cpu().numpy().reshape(H, W)

                match = np.array_equal(pred_grid, out_actual)
                n_correct = np.sum(pred_grid == out_actual)
                total_tasks += 1
                if match:
                    correct_grids += 1
                total_cells += H * W
                correct_cells += n_correct

                status = "CORRECT" if match else f"{n_correct}/{H*W} cells"
                print(f"  {tname}: {status}")
                print(f"    Input:     {inp.flatten().tolist()}")
                print(f"    Predicted: {pred_grid.flatten().tolist()}")
                print(f"    Actual:    {out_actual.flatten().tolist()}")
                print()

    print(f"  TOTAL: {correct_grids}/{total_tasks} grids ({100*correct_grids/max(total_tasks,1):.0f}%)")
    print(f"         {correct_cells}/{total_cells} cells ({100*correct_cells/max(total_cells,1):.0f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=3, help="Grid H=W size")
    parser.add_argument("--colors", type=int, default=10, help="Number of colors")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--arc", action="store_true", help="Run ARC pipeline after training")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    size = args.grid_size
    print(f"\n{'='*60}")
    print(f"  Grid size: {size}x{size}, {args.colors} colors")
    print(f"  Possible grids: {args.colors}^{size*size} = {args.colors**(size*size):.1e}")
    print(f"  Gram has 21 features")
    print(f"{'='*60}")

    model = train(size, size, args.colors, args.steps, args.batch, device=device)

    if args.arc:
        test_arc_pipeline(model, device, size, size)


if __name__ == "__main__":
    main()
