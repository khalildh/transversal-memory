#!/bin/bash
# Run C solver on all same-size ARC tasks, saving each result to results/<task>.txt
# Usage: bash run_all_tasks.sh [parallel_jobs]

set -e
JOBS=${1:-8}
RESDIR="results"
SOLVER="./arc_solver_st"
TASKS="/tmp/arc_tasks_list.txt"
TIMEOUT=120

# Build single-threaded solver if needed
if [ ! -f "$SOLVER" ]; then
    echo "Building single-threaded solver..."
    cc -O3 -march=native -framework Accelerate -o "$SOLVER" arc_solver.c -lm
fi

# Generate task list
mkdir -p "$RESDIR"
uv run python -c "
import json, os
d = 'data/ARC-AGI/data/training'
for f in sorted(os.listdir(d)):
    if not f.endswith('.json'): continue
    with open(os.path.join(d, f)) as fh: t = json.load(fh)
    if len(t['train']) < 2: continue
    ok = all(len(p['input'])==len(p['output']) and len(p['input'][0])==len(p['output'][0])
             for p in t['train'] + t['test'])
    if ok: print(os.path.join(d, f))
" > "$TASKS"

TOTAL=$(wc -l < "$TASKS")
echo "Running $TOTAL tasks with $JOBS parallel workers (${TIMEOUT}s timeout each)"
echo "Results: $RESDIR/<task>.txt"
echo ""

run_task() {
    local f="$1"
    local name=$(basename "$f" .json)
    local out="$RESDIR/$name.txt"
    # macOS doesn't have timeout; use perl alarm instead
    perl -e 'alarm shift; exec @ARGV' "$TIMEOUT" "$SOLVER" "$f" > "$out" 2>&1
    local rc=$?
    if [ $rc -eq 142 ]; then
        echo "TIMEOUT" >> "$out"
        echo "  $name: TIMEOUT"
    else
        local rank=$(grep "rank [0-9]" "$out" | head -1 | sed 's/.*rank /rank /' | sed 's/ *$//')
        echo "  $name: $rank"
    fi
}
export -f run_task
export SOLVER RESDIR TIMEOUT

cat "$TASKS" | xargs -P "$JOBS" -I {} bash -c 'run_task "$@"' _ {}

echo ""
echo "=== SUMMARY ==="
echo "Total: $(ls "$RESDIR"/*.txt | wc -l) tasks"
echo "Rank 1: $(grep -l 'rank 1/' "$RESDIR"/*.txt 2>/dev/null | wc -l)"
echo "Solved: $(grep -l 'SOLVED' "$RESDIR"/*.txt 2>/dev/null | wc -l)"
echo "Timeout: $(grep -l 'TIMEOUT' "$RESDIR"/*.txt 2>/dev/null | wc -l)"
