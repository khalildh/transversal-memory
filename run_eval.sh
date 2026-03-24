#!/bin/bash
DIR="data/ARC-AGI/data/evaluation"
RESDIR="results_eval"
SOLVER="./arc_solver_st"
TIMEOUT=120

mkdir -p "$RESDIR"

# Get same-size eval tasks
uv run python -c "
import json, os
d = 'data/ARC-AGI/data/evaluation'
for f in sorted(os.listdir(d)):
    if not f.endswith('.json'): continue
    with open(os.path.join(d, f)) as fh: t = json.load(fh)
    if len(t['train']) < 2: continue
    ok = all(len(p['input'])==len(p['output']) and len(p['input'][0])==len(p['output'][0])
             for p in t['train'] + t['test'])
    if ok: print(os.path.join(d, f))
" > /tmp/eval_tasks.txt

total=$(wc -l < /tmp/eval_tasks.txt)
echo "Running $total eval tasks with 8 parallel workers"

run_task() {
    local f="$1"
    local name=$(basename "$f" .json)
    local out="results_eval/$name.txt"
    perl -e 'alarm shift; exec @ARGV' 120 ./arc_solver_st "$f" > "$out" 2>&1
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

cat /tmp/eval_tasks.txt | xargs -P 8 -I {} bash -c 'run_task "$@"' _ {}

echo ""
echo "=== EVAL RESULTS ==="
echo "Total: $(ls results_eval/*.txt 2>/dev/null | wc -l)"
echo "Rank 1: $(grep -rl 'rank 1/' results_eval/*.txt 2>/dev/null | wc -l)"
echo "Solved: $(grep -rl 'SOLVED' results_eval/*.txt 2>/dev/null | wc -l)"
echo "Timeout: $(grep -rl 'TIMEOUT' results_eval/*.txt 2>/dev/null | wc -l)"
