# autoresearch — transversal memory

This is an experiment to have the LLM do its own research on geometric
associative memory.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context, mathematical background.
   - `evaluate.py` — fixed evaluation harness. Do not modify.
   - `experiment.py` — the file you modify. Encoding, transversals, ranking.
   - `transversal_memory/plucker.py` — Plücker geometry primitives (read-only reference).
   - `transversal_memory/higher_grass.py` — higher Grassmannian ops (read-only reference).
   - `transversal_memory/memory.py` — P3Memory, GramMemory (read-only reference).
4. **Verify data exists**: Check that `data/cache/` contains `associations.pkl` and `embeddings_dim32.pkl`. If not, tell the human.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The task

You have a geometric memory system that encodes word associations as lines
in projective space (Plücker coordinates). Given a word's known associates,
it computes "transversal" lines — lines that intersect all stored lines —
and uses these to rank the full 67K vocabulary.

**The goal: maximise `mean_p@10`** — the average precision at 10, measuring
how many of the top-10 ranked words are actual held-out associates.

The evaluation holds out 25% of associates, uses the remaining 75% to build
the geometric structure, and measures whether held-out words rank highly.
The cosine-NN baseline (just ranking by embedding similarity) is provided
for comparison.

## Experimentation

Each experiment runs in seconds (vocabulary ranking is fully vectorised).
You launch it simply as: `python3 experiment.py`

**What you CAN do:**
- Modify `experiment.py` — this is the only file you edit. Everything is fair game: Grassmannian dimension, number of transversals, scoring function, encoding strategy, projection matrices, hybrid approaches combining geometry with embeddings, etc.
- You can use any function from `transversal_memory/` as a building block.

**What you CANNOT do:**
- Modify `evaluate.py`. It is read-only.
- Modify anything in `transversal_memory/`. Those are fixed building blocks.
- Install new packages or add dependencies.
- Modify the evaluation harness or metrics.

**Simplicity criterion**: All else being equal, simpler is better. A small
improvement that adds ugly complexity is not worth it. Removing something
and getting equal or better results is a great outcome.

## What to try

The search space is rich. Here are starting directions:

**Geometric parameters:**
- Grassmannian dimension: N_PROJ=3 (6D), 4 (10D), 5 (15D), 6 (21D), ...
- Number of transversals: 5, 10, 20, 50, 100, 200
- Scoring method: sum_log, mean, max, or custom aggregations
- Multiple projection seeds / ensembles

**Encoding strategies:**
- Different projection matrix constructions (orthogonal, sparse, etc.)
- Mixing source and target embeddings differently
- Using singular values as weights

**Hybrid approaches:**
- Combine geometric score with cosine similarity
- Use geometry as a filter, then re-rank by cosine
- Use cosine to pre-filter candidates, then score geometrically
- Weighted combination: α * geometric + (1-α) * cosine

**Structural ideas:**
- GramMemory scores as features alongside transversals
- Multiple Grassmannians simultaneously (ensemble across dimensions)
- Adaptive transversal count based on associate count
- Smarter transversal sampling (e.g. diverse subsets)

**Mathematical ideas:**
- Different Hodge dual formulations
- Alternative constraint aggregation (geometric mean, harmonic mean)
- Spectral methods on the Gram matrix
- Using Plücker relation residuals as a quality signal

## Output format

The script prints a summary table:

```
============================================================
Metric               Experiment    Cosine NN     Lift
------------------------------------------------------------
  p@10                  0.0450       0.0300      1.5x
  p@50                  0.0280       0.0200      1.4x
  ...
============================================================
```

Extract the key metric: `mean_p@10` (the primary optimisation target).

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 5 columns:

```
commit	mean_p@10	baseline_p@10	status	description
```

1. git commit hash (short, 7 chars)
2. mean_p@10 achieved (e.g. 0.045000)
3. baseline_p@10 from cosine NN (should be stable ~same each run)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar12`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit
2. Modify `experiment.py` with an experimental idea
3. git commit
4. Run the experiment: `python3 experiment.py > run.log 2>&1`
5. Read out the results: `grep "mean_p@10\|Primary metric" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` to debug
7. Record results in the TSV (do not commit results.tsv)
8. If mean_p@10 improved, keep the commit (advance the branch)
9. If mean_p@10 is equal or worse, `git reset --hard HEAD~1`

**Each experiment takes ~10-30 seconds** (200 words × vectorised ranking).
You can run ~200 experiments per hour.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human.
The human might be asleep. You are autonomous. If you run out of ideas,
think harder — re-read the math, try combining previous near-misses,
try more radical approaches. The loop runs until interrupted.
