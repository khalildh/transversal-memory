# CLAUDE.md

## Project overview

Content-addressable memory via projective geometry and Schubert calculus.
Maps item pairs to Plücker lines in P³ (6-vectors), stores them in Gram
matrices, retrieves via transversal computation or energy scoring.

## Running experiments

```bash
uv run python <script>         # use uv for dependency management
uv run python experiment.py    # word association ranking (8-signal RRF)
uv run python evaluate.py      # cosine baseline only
uv run python exp_lm_variants.py kernel|bigram|hybrid|all
```

- Python: use `uv run python` (not `python` directly)
- PyTorch: Apple Silicon MPS available (`torch.device("mps")`)
- MPS can hang when running from external drive — run from internal SSD if needed
- `evaluate.py` is the ground-truth evaluation harness — **DO NOT MODIFY**
- Buffered stdout: use `PYTHONUNBUFFERED=1` for background tasks

## Key files

- `transversal_memory/` — core library (plucker.py, solver.py, memory.py, etc.)
- `experiment.py` — best word association ranker (8-signal RRF, p@10=0.128)
- `evaluate.py` — fixed evaluation harness (200 words, 25% holdout, DO NOT MODIFY)
- `exp_lm.py` — original Plücker attention LM (broken, PPL 2063)
- `exp_lm_variants.py` — three fixes: kernel (252), bigram (215), hybrid (207)
- `exp_lm_v2.py`, `exp_lm_v3.py`, `exp_lm_v4.py` — standalone variant scripts
- `exp_mem_attn.py` — memory-augmented attention (static + online Gram memory)
- `exp_fast.py` — fast iteration harness (cached data, checkpoint init, early stopping)
- `ideas.md` — active ideas, experiment results, and resolved questions
- `data/cache/` — pickled PPMI+SVD embeddings + cached token tensors (auto-built)
- `data/wikitext/` — WikiText-2 parquet files (gitignored)
- `checkpoints/` — saved model weights (gitignored)

## Architecture knowledge

### Plücker geometry
- Lines in P³ → 6-vectors via exterior product a∧b
- Plücker inner product p·(★q) = 0 iff lines meet (incidence)
- Gram matrix M = Σ pᵢ⊗pᵢ stores relational structure
- Transversal: given 4 lines, find the (exactly 2) lines meeting all four
- J6 matrix: Hodge dual, maps between minor and dual Plücker formats

### Plücker attention (exp_lm_variants.py)
- Kernel: separate W1/W2 projections for Q and K lines, signed inner product
- Bigram: query lines from token pairs [x_i; x_{i-1}], degree-4 interactions
- Hybrid: standard Q·K + additive Plücker bias with learned per-head scale
- All use `exterior()` for Plücker embedding and J6 for inner product

### Word association ranking (experiment.py)
- 8-signal RRF ensemble over PPMI+SVD embeddings (dim=32)
- Signals: cos_src, cos_tgt, max_sim, mean_sim, recip_NN, Mahalanobis_cent, whitened_cos, Mahalanobis_src
- Mahalanobis-to-centroid is the strongest single signal

### Online Gram memory (exp_mem_attn.py) ← current best
- Standard Q·K attention + causal Gram accumulation as scalar gate
- M_t = Σ_{s<t} decay^{t-s} · p_s⊗p_s (6×6 relational summary)
- PPL 206 vs standard 209 — first geometry win
- Memory-efficient: computes (B,H,T,T) incidence matrix, not (B,T,H,6,6) Grams
- Geometry should AUGMENT attention (scalar gate), not COMPETE (dual pathway fails)

### Key results
- Word association: p@10=0.128 (73% over cosine baseline)
- Plücker geometry adds zero to embedding ensemble for discriminative ranking
- Geometry beats embeddings at K≥10 seed associates (few-shot regime)
- Online Gram memory: PPL 206 vs standard 209 (first geometry win in LM)
- Hybrid attention ties standard (PPL 207 vs 208)
- Dual-pathway incidence attention: PPL 372 (fails — geometry can't replace Q·K routing)

### Fast iteration (exp_fast.py)
- Cached tokenized data, no baseline re-training, batch_size=128, cosine LR
- Checkpoint init from standard model + 3 epoch fine-tune (~5 min per experiment)
- `--fast` flag for 2-layer screening model (~1 min per experiment)
- Early stopping with patience=2

## Conventions

- **Always update `ideas.md` after testing an idea** — record result, mark status, note lessons learned. This is the project's experimental log.
- Checkpoints: `checkpoints/lm_<variant>.pt` with dict keys: model, type, losses, ppls, vocab
- Config class for LM hyperparameters (d_model=192, n_heads=6, n_layers=4, seq_len=128)
- Tokenizer: tiktoken gpt2 encoding
- Training: AdamW, lr=3e-4, gradient clipping at 1.0, 10 epochs
