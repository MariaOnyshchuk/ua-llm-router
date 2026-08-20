# Specialist bake-off — decide router combinations from metrics

Do **not** wire a new router combo until single-model bake-offs show a clear
per-bucket winner. Combinations are a *consequence* of the matrix, not a starting point.

## Systems to measure (alone, same suite)

| ID | Model | Port | Priority |
|----|-------|------|----------|
| S0 | Mamay-4B-IT | 8003 | done on `mixed_ua_v3` |
| S2 | Mamay-12B-IT | 8002 | done |
| Q | Qwen2.5-Coder-3B | 8004 | **needed** (alone on full suite) |
| L | Lapa-12B-Instruct | 8001 | **needed** (esp. translate) |
| — | Aya-101 / Mistral-7B / Llama-8B | — | later / footnote only |

Suite: `benchmarks/mixed_ua_v3.jsonl` (182). Decoding: `T=0`, `seed=42`, ideally 3×.

Fair VRAM: prefer **one live model at a time** when collecting `peak_vram_gb`.
Quality bake-off can run with several GPUs occupied — quality is the decision driver.

## Metrics (always report)

| Metric | Why |
|--------|-----|
| Quality per bucket | Who wins the bucket → who may receive that route |
| Macro overall | Equal-bucket view (don’t let translate/alignment dominate) |
| Micro overall | Sample-weighted; secondary |
| Latency p50 per bucket | Cost of picking that specialist |
| Peak VRAM (isolated) | Efficiency claim |
| GPU-seconds / prompt | Same-run resource log |
| Δ vs Mamay-4B | Route only if gain ≥ margin (default **0.02**) |
| mean ± sd over 3× | Variance pin |

## Decision rule

For each bucket `B`:

```
if best(B) - mamay4(B) >= 0.02:
    route B → best(B)
else:
    route B → mamay4
```

Build a multi-model router **only if** ≥1 bucket gets a non-default route.
If every bucket stays on Mamay-4B → thesis result is “no useful specialist pool yet.”

## Commands

```bash
# After each single-model run:
python scripts/score_results.py results/<model>_mixed_ua_v3_*.jsonl \
  --out results/scores_<model>_v3.json

# Matrix + suggested routes:
python scripts/build_specialist_matrix.py \
  --scores results/scores_mean_sd_v3.json results/scores_qwen_v3.json results/scores_lapa_v3.json \
  --default mamay4 --margin 0.02 \
  --out results/specialist_matrix_v3.json

# Efficiency (optional, needs *.meta.json):
python scripts/build_efficiency_table.py \
  --scores results/scores_....json \
  --meta-glob 'results/*_mixed_ua_v3_*.meta.json' \
  --out results/efficiency_table_v3.json
```

## After the matrix

Only then pick combinations (cascade, Lapa-translate swap, drop Qwen, …).
Candidates like Aya-101 / Mistral-7B stay in “considered, not prioritized”
unless B–D underperform.
