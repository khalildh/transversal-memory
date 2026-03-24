"""
test_c_vs_py_emb.py — Compare C and Python embedding implementations.

For each of the 8 embedding types, computes embeddings for all cells
of task 25ff71a9 using both Python and C (via arc_tables.so), then
reports the maximum absolute difference.
"""

import json
import numpy as np
import sys
import os

# ── Import Python embedding functions from exp_arc_fast_solve ────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_arc_fast_solve import (
    emb_hist_color, emb_color_only, emb_pos_color, emb_all,
    emb_row_features, emb_col_features, emb_color_count, emb_diagonal,
    N_COLORS, EMBEDDINGS,
)
from arc_tables_wrapper import precompute_cell_embeddings_c

# ── Load task ────────────────────────────────────────────────────────────────

TASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data/ARC-AGI/data/training/25ff71a9.json")

with open(TASK_PATH) as f:
    task = json.load(f)

# Use test input as both inp and out (matching the solver's test-time behavior)
test_inp = np.array(task['test'][0]['input'])
H, W = test_inp.shape

# Compute used_colors (same logic as FastArcSolver)
used_colors = sorted(set(
    c for p in task['train']
    for g in [p['input'], p['output']]
    for row in g for c in row
) | set(
    c for row in task['test'][0]['input'] for c in row
))
nc = len(used_colors)

print(f"Task: 25ff71a9")
print(f"Grid: {H}x{W}, used_colors={used_colors} (nc={nc})")
print()

# ── Python embedding functions (ordered to match C EMB_FUNCS) ────────────────

PY_EMB_FUNCS = [
    emb_hist_color,
    emb_color_only,
    emb_pos_color,
    emb_all,
    emb_row_features,
    emb_col_features,
    emb_color_count,
    emb_diagonal,
]

EMB_NAMES = [
    'hist_color', 'color_only', 'pos_color', 'all',
    'row_feat', 'col_feat', 'color_count', 'diagonal',
]

EMB_DIMS = [30, 20, 22, 42, 44, 42, 24, 26]

# ── Compare each embedding type ─────────────────────────────────────────────

all_pass = True

for emb_idx in range(8):
    name = EMB_NAMES[emb_idx]
    dim = EMB_DIMS[emb_idx]
    py_fn = PY_EMB_FUNCS[emb_idx]

    # Python: compute embeddings for all (pos, candidate_color) combos
    py_embs = np.zeros((H * W, nc, dim), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            in_c = int(test_inp[r, c])
            for ci_idx in range(nc):
                out_c = used_colors[ci_idx]
                emb = py_fn(r, c, in_c, out_c, test_inp, test_inp, H, W)
                py_embs[r * W + c, ci_idx, :] = emb

    # C: compute embeddings via arc_tables.so
    c_embs = precompute_cell_embeddings_c(emb_idx, test_inp, used_colors, H, W)

    # Compare
    max_diff = np.max(np.abs(py_embs - c_embs))
    mean_diff = np.mean(np.abs(py_embs - c_embs))

    status = "PASS" if max_diff < 1e-6 else "FAIL"
    if status == "FAIL":
        all_pass = False

    print(f"[{status}] {name:15s}  dim={dim:2d}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")

    if max_diff >= 1e-6:
        # Find and report first divergence
        for pos in range(H * W):
            for ci_idx in range(nc):
                diff = np.abs(py_embs[pos, ci_idx] - c_embs[pos, ci_idx])
                if np.max(diff) >= 1e-6:
                    r, c = divmod(pos, W)
                    in_c = int(test_inp[r, c])
                    out_c = used_colors[ci_idx]
                    print(f"  First divergence at r={r}, c={c}, in_c={in_c}, out_c={out_c}")
                    for d in range(dim):
                        if abs(py_embs[pos, ci_idx, d] - c_embs[pos, ci_idx, d]) >= 1e-7:
                            print(f"    dim[{d}]: py={py_embs[pos,ci_idx,d]:.8f}  "
                                  f"c={c_embs[pos,ci_idx,d]:.8f}  "
                                  f"diff={abs(py_embs[pos,ci_idx,d]-c_embs[pos,ci_idx,d]):.2e}")
                    break
            else:
                continue
            break

print()
if all_pass:
    print("ALL PASS: C and Python embeddings match exactly (within 1e-6).")
else:
    print("SOME FAILURES: See details above.")

# ── Also test with a non-square, larger grid from a different task ───────────

print("\n" + "="*70)
print("Testing with additional tasks for edge-case coverage...")
print("="*70 + "\n")

extra_tasks = []
train_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data/ARC-AGI/data/training")
# Pick a few tasks with varied grid sizes
for fname in sorted(os.listdir(train_dir))[:20]:
    if fname.endswith('.json'):
        with open(os.path.join(train_dir, fname)) as f:
            t = json.load(f)
        ti = np.array(t['test'][0]['input'])
        if ti.shape[0] != ti.shape[1] or ti.size > 9:  # non-square or larger
            extra_tasks.append((fname[:-5], t))
            if len(extra_tasks) >= 3:
                break

for task_id, task in extra_tasks:
    test_inp2 = np.array(task['test'][0]['input'])
    H2, W2 = test_inp2.shape
    uc2 = sorted(set(
        c for p in task['train']
        for g in [p['input'], p['output']]
        for row in g for c in row
    ) | set(c for row in task['test'][0]['input'] for c in row))
    nc2 = len(uc2)
    print(f"Task: {task_id}, Grid: {H2}x{W2}, nc={nc2}")

    task_pass = True
    for emb_idx in range(8):
        name = EMB_NAMES[emb_idx]
        dim = EMB_DIMS[emb_idx]
        py_fn = PY_EMB_FUNCS[emb_idx]

        py_embs2 = np.zeros((H2 * W2, nc2, dim), dtype=np.float32)
        for r in range(H2):
            for c_pos in range(W2):
                in_c = int(test_inp2[r, c_pos])
                for ci_idx in range(nc2):
                    out_c = uc2[ci_idx]
                    emb = py_fn(r, c_pos, in_c, out_c, test_inp2, test_inp2, H2, W2)
                    py_embs2[r * W2 + c_pos, ci_idx, :] = emb

        c_embs2 = precompute_cell_embeddings_c(emb_idx, test_inp2, uc2, H2, W2)
        max_diff = np.max(np.abs(py_embs2 - c_embs2))
        status = "PASS" if max_diff < 1e-6 else "FAIL"
        if status == "FAIL":
            task_pass = False
            all_pass = False
        print(f"  [{status}] {name:15s}  max_diff={max_diff:.2e}")

        if max_diff >= 1e-6:
            for pos in range(H2 * W2):
                for ci_idx in range(nc2):
                    diff = np.abs(py_embs2[pos, ci_idx] - c_embs2[pos, ci_idx])
                    if np.max(diff) >= 1e-6:
                        r, c_pos = divmod(pos, W2)
                        in_c = int(test_inp2[r, c_pos])
                        out_c = uc2[ci_idx]
                        print(f"    First divergence at r={r}, c={c_pos}, in_c={in_c}, out_c={out_c}")
                        for d in range(dim):
                            if abs(py_embs2[pos, ci_idx, d] - c_embs2[pos, ci_idx, d]) >= 1e-7:
                                print(f"      dim[{d}]: py={py_embs2[pos,ci_idx,d]:.8f}  "
                                      f"c={c_embs2[pos,ci_idx,d]:.8f}")
                        break
                else:
                    continue
                break

print()
if all_pass:
    print("ALL PASS across all tasks.")
else:
    print("SOME FAILURES found.")
