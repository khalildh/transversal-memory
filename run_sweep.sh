#!/bin/bash
cd /Volumes/PRO-G40/Code/transversal-memory
echo "===== DECAY SWEEP (fast 2-layer model) ====="
echo "--- standard baseline ---"
uv run python -u exp_fast.py standard --fast --from-scratch --epochs 7 2>&1 | grep --line-buffered -E "best PPL|ep "
echo
echo "--- decay=0.95 ---"
uv run python -u exp_fast.py online_mem --fast --from-scratch --decay 0.95 --epochs 7 2>&1 | grep --line-buffered -E "best PPL|ep "
echo
echo "--- decay=0.99 ---"
uv run python -u exp_fast.py online_mem --fast --from-scratch --decay 0.99 --epochs 7 2>&1 | grep --line-buffered -E "best PPL|ep "
echo
echo "--- decay=0.995 ---"
uv run python -u exp_fast.py online_mem --fast --from-scratch --decay 0.995 --epochs 7 2>&1 | grep --line-buffered -E "best PPL|ep "
echo
echo "--- decay=1.0 (no forgetting) ---"
uv run python -u exp_fast.py online_mem --fast --from-scratch --decay 1.0 --epochs 7 2>&1 | grep --line-buffered -E "best PPL|ep "
echo
echo "===== DONE ====="
