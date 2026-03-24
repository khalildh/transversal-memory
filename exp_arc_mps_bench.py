"""
exp_arc_mps_bench.py — Benchmark MPS vs CPU for multi-transversal scoring.
"""

import json
import numpy as np
import torch
import time
from transversal_memory import P3Memory

N_COLORS = 10
J6 = np.array([[0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
                [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0]], dtype=np.float32)


def emb_color_only(in_c, out_c):
    v = np.zeros(20, dtype=np.float32)
    v[in_c] = 1.0
    v[10 + out_c] = 1.0
    return v


def compute_trans(lines, n_trans=200, rng=None):
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


def main():
    with open("data/ARC-AGI/data/training/0d3d703e.json") as f:
        task = json.load(f)

    H, W = 3, 3
    ti = np.array(task['test'][0]['input'])
    to = np.array(task['test'][0]['output'])
    uc = np.array(sorted(set(
        c for p in task['train'] + task['test']
        for g in [p['input'], p['output']] for row in g for c in row)))
    nc = len(uc)
    color_to_idx = {c: i for i, c in enumerate(uc)}

    adj_pairs = [(r, c, r+dr, c+dc)
                 for r in range(H) for c in range(W)
                 for dr, dc in [(0, 1), (1, 0)]
                 if r+dr < H and c+dc < W]

    rng = np.random.RandomState(hash('color_only') % 2**31)
    W1 = rng.randn(4, 40).astype(np.float32) * 0.1
    W2 = rng.randn(4, 40).astype(np.float32) * 0.1

    # Precompute line tables
    line_tables = []
    for r, c, r2, c2 in adj_pairs:
        table = np.zeros((nc, nc, 6), dtype=np.float32)
        for ia in range(nc):
            for ib in range(nc):
                combined = np.concatenate([
                    emb_color_only(ti[r, c], uc[ia]),
                    emb_color_only(ti[r2, c2], uc[ib])])
                p1 = W1 @ combined
                p2 = W2 @ combined
                L = np.array([p1[i]*p2[j] - p1[j]*p2[i]
                              for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],
                             dtype=np.float32)
                n = np.linalg.norm(L)
                if n > 1e-10:
                    table[ia, ib] = L / n
        line_tables.append(table)

    # Transversals
    def pair_lines(inp, out):
        lines = []
        for r, c, r2, c2 in adj_pairs:
            combined = np.concatenate([
                emb_color_only(inp[r, c], out[r, c]),
                emb_color_only(inp[r2, c2], out[r2, c2])])
            p1 = W1 @ combined
            p2 = W2 @ combined
            L = np.array([p1[i]*p2[j] - p1[j]*p2[i]
                          for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],
                         dtype=np.float32)
            n = np.linalg.norm(L)
            if n > 1e-10:
                lines.append(L / n)
        return lines

    trans = []
    for i, pair in enumerate(task['train']):
        trans.extend(compute_trans(
            pair_lines(np.array(pair['input']), np.array(pair['output'])),
            200, np.random.default_rng(42 + i)))

    JTm_np = (J6 @ np.stack(trans).T).astype(np.float32)  # (6, n_trans)
    correct_colors = np.array([[color_to_idx[to[r, c]] for c in range(W)]
                                for r in range(H)])

    N = 100000
    rng2 = np.random.RandomState(0)
    cands = rng2.randint(0, nc, size=(N, H, W))

    print(f"Task: 0d3d703e, {nc} colors, {len(trans)} transversals")
    print(f"Candidates: {N:,}")
    print()

    # === CPU (numpy) ===
    t0 = time.time()
    scores_cpu = np.zeros(N, dtype=np.float64)
    for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
        L = line_tables[ap_idx][cands[:, r, c], cands[:, r2, c2]]
        inner = L @ JTm_np
        scores_cpu += np.sum(np.log(np.abs(inner) + 1e-10), axis=1)
    t_cpu = time.time() - t0

    cs_cpu = 0.0
    for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
        L = line_tables[ap_idx][correct_colors[r, c], correct_colors[r2, c2]].reshape(1, 6)
        cs_cpu += np.sum(np.log(np.abs(L @ JTm_np) + 1e-10))
    b_cpu = int(np.sum(scores_cpu < cs_cpu))

    print(f"CPU:  {t_cpu:.2f}s ({N/t_cpu:,.0f}/s) — {b_cpu} beat correct")

    # === MPS (torch) ===
    if torch.backends.mps.is_available():
        device = torch.device("mps")

        # Move tables and transversal matrix to MPS
        lt_torch = [torch.tensor(t, device=device) for t in line_tables]
        JTm_t = torch.tensor(JTm_np, device=device)
        cands_t = torch.tensor(cands, dtype=torch.long, device=device)

        # Warmup
        _ = lt_torch[0][cands_t[:100, 0, 0], cands_t[:100, 0, 1]] @ JTm_t
        torch.mps.synchronize()

        t0 = time.time()
        scores_mps = torch.zeros(N, device=device, dtype=torch.float32)
        for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
            L = lt_torch[ap_idx][cands_t[:, r, c], cands_t[:, r2, c2]]
            inner = L @ JTm_t
            scores_mps += torch.sum(torch.log(torch.abs(inner) + 1e-10), dim=1)
        torch.mps.synchronize()
        t_mps = time.time() - t0

        cs_t = torch.tensor(correct_colors, dtype=torch.long, device=device)
        cs_mps = 0.0
        for ap_idx, (r, c, r2, c2) in enumerate(adj_pairs):
            L = lt_torch[ap_idx][cs_t[r, c], cs_t[r2, c2]].unsqueeze(0)
            cs_mps += torch.sum(torch.log(torch.abs(L @ JTm_t) + 1e-10)).item()
        b_mps = int((scores_mps < cs_mps).sum().item())

        print(f"MPS:  {t_mps:.2f}s ({N/t_mps:,.0f}/s) — {b_mps} beat correct")
        print(f"Speedup: {t_cpu/t_mps:.1f}x")
        print(f"\nAt MPS speed, 134M = {134217728/N*t_mps:.0f}s = {134217728/N*t_mps/60:.1f}min")
    else:
        print("MPS not available")

    print(f"At CPU speed, 134M = {134217728/N*t_cpu:.0f}s = {134217728/N*t_cpu/60:.1f}min")


if __name__ == "__main__":
    main()
