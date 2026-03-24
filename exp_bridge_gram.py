"""
exp_bridge_gram.py — Test pencil bridge Grams as ARC rule encoding.

For each pair of input/output spatial lines, find valid Plücker lines in
their pencil (T = t*L_in + L_out satisfying the Plücker relation).
These "bridges" connect input and output geometric structure.
Their Gram should be consistent across demo pairs if it encodes the rule.

Usage:
  uv run python exp_bridge_gram.py
"""

import json
import os
import numpy as np
from transversal_memory.plucker import line_from_points
from transversal_memory.solver import solve_p3


def spatial_lines_3d(grid, z=1.0):
    grid = np.array(grid)
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


def bridge_gram_vec(inp, out):
    """Compute Gram of pencil bridges between input and output lines."""
    L_in = spatial_lines_3d(np.array(inp), z=1.0)
    L_out = spatial_lines_3d(np.array(out), z=3.0)
    bridges = []
    for i in range(len(L_in)):
        for j in range(len(L_out)):
            results = solve_p3(L_in[i], L_out[j])
            for T, residual in results:
                n = np.linalg.norm(T)
                if n > 1e-10 and residual < 1e-6:
                    bridges.append(T / n)
    if not bridges:
        return None
    B = np.array(bridges)
    M = B.T @ B
    return M[np.triu_indices(6)]


def main():
    arc_dir = "data/ARC-AGI/data/training"
    tested = 0
    results = []

    for fname in sorted(os.listdir(arc_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(arc_dir, fname)) as f:
            task = json.load(f)
        if len(task['train']) < 2:
            continue

        # Same-size only
        all_same = all(
            len(p['input']) == len(p['output']) and
            len(p['input'][0]) == len(p['output'][0])
            for p in task['train'] + task['test']
        )
        if not all_same:
            continue
        sizes = set((len(p['input']), len(p['input'][0]))
                    for p in task['train'] + task['test'])
        if len(sizes) > 1:
            continue

        try:
            demo_bgs = []
            for pair in task['train']:
                bg = bridge_gram_vec(pair['input'], pair['output'])
                if bg is not None:
                    demo_bgs.append(bg)

            if len(demo_bgs) < 2:
                continue

            mean_bg = np.mean(demo_bgs, axis=0)
            deviations = [np.linalg.norm(bg - mean_bg) / (np.linalg.norm(mean_bg) + 1e-10)
                          for bg in demo_bgs]
            consistency = np.mean(deviations)

            for test in task['test']:
                test_bg = bridge_gram_vec(test['input'], test['output'])
                if test_bg is None:
                    continue
                tested += 1

                err = np.linalg.norm(mean_bg - test_bg) / (np.linalg.norm(test_bg) + 1e-10)
                results.append((fname.replace('.json', ''), err, consistency))

                print(f"  {fname.replace('.json','')}: err={err:.4f}, consistency={consistency:.4f}")
        except Exception as e:
            print(f"  {fname}: FAILED ({e})")
            continue

    print(f"\n{'='*60}")
    print(f"Tested {tested} task-test pairs")

    if results:
        errs = [r[1] for r in results]
        print(f"\nBridge Gram prediction (mean of demos vs test):")
        print(f"  mean={np.mean(errs):.4f}, median={np.median(errs):.4f}")
        print(f"  <5%:  {sum(1 for e in errs if e<0.05)}/{len(errs)}")
        print(f"  <10%: {sum(1 for e in errs if e<0.1)}/{len(errs)}")
        print(f"  <20%: {sum(1 for e in errs if e<0.2)}/{len(errs)}")

        results.sort(key=lambda x: x[1])
        print(f"\nBest 5:")
        for name, err, cons in results[:5]:
            print(f"  {name}: err={err:.4f}, consistency={cons:.4f}")
        print(f"Worst 5:")
        for name, err, cons in results[-5:]:
            print(f"  {name}: err={err:.4f}, consistency={cons:.4f}")


if __name__ == "__main__":
    main()
