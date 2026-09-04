# Week 9 — the two missing reference points

*Plan, not results. Fills S2 and S3 from [`docs/thesis_scope.md`](thesis_scope.md).*

Everything so far compares the router against models **inside its own pool**, which a reviewer can call circular. These two runs give the comparisons a decision-maker actually weighs: *scale up one open model* or *call a hosted frontier model*.

Both use the product protocol: `mixed_ua_v4_balanced` (192), `T=0`, `seed=42`, `max_tokens=256`, `--align-prompt fewshot`, 3 repeats.

| Slot | System | Where | Cost |
|------|--------|-------|------|
| S2 | Mamay-12B (one larger open model) | cluster, 1 GPU | ~40 min wall |
| S3 | Hosted frontier API | laptop | ~$1–4 total |

## S2 — Mamay-12B on v4

The runner already has the mode and the sbatch file exists. Serve on `:8002` at the **same** `gpu_memory_utilization=0.90` as every other single-model run, so the VRAM figure stays comparable.

```bash
ssh ucu-lab-2240
cd ~/Diploma
sbatch cluster/serve_mamay12.sbatch
squeue -u "$USER"                      # wait for R
curl -s localhost:8002/v1/models       # health

python scripts/run_small_router_cluster.py --mode mamay12 \
  --bench benchmarks/mixed_ua_v4_balanced.jsonl \
  --align-prompt fewshot \
  --repeats 3 \
  --out-dir results/week_9_baselines
```

Interpretation guide before the numbers land:

- **Mamay-12B < 0.848** — composition beats scaling on this workload. The strongest version of the thesis.
- **Mamay-12B ≈ 0.848** — the honest claim becomes latency and per-bucket robustness, not headline quality. Router still wins p50 (633 ms vs ~1.2 s on v3) and does not have Lapa's translate hole.
- **Mamay-12B > 0.848** — report it. The fallback claim is that three small specialists on commodity cards approach a 12B, and the router is the cheaper deployment per query.

Any of the three is publishable. Only *not running it* is a problem.

## S3 — hosted frontier API on v4

`scripts/run_api_baseline.py` writes the same row shape as the cluster runner, so `score_results.py` and `aggregate_repeat_scores.py` work unchanged. Requests are sequential so p50 stays comparable.

```bash
export OPENAI_API_KEY=sk-...

# smoke first — 6 items, cents
python scripts/run_api_baseline.py \
  --system gpt --model gpt-4o-2024-11-20 \
  --bucket alignment --limit 6 \
  --out-dir results/week_9_baselines

# full run
python scripts/run_api_baseline.py \
  --system gpt --model gpt-4o-2024-11-20 \
  --align-prompt fewshot --repeats 3 \
  --price-in 2.5 --price-out 10 \
  --out-dir results/week_9_baselines
```

Works with any OpenAI-compatible endpoint via `--base-url` (OpenRouter, Azure, a local gateway), so a second frontier model is one flag away.

**Rules for citing this system:** it has no VRAM or GPU-seconds figure — its latency is network wall-clock. Keep it out of hardware columns and give it a **USD per 1000 queries** column instead. Expect it to win on quality; that is the point. The thesis argument is that this option is *unavailable* under a data-residency constraint, so the relevant question is how much quality the on-prem composite gives up.

## Scoring both

`score_results.py` de-duplicates on `(system, id)`, so score **each repeat separately** and only then aggregate — passing all reps at once silently keeps whichever file was read last.

```bash
cd results/week_9_baselines
for r in 1 2 3; do
  python ../../scripts/score_results.py *_rep${r}_*.jsonl --out scores_rep${r}.json
done

python ../../scripts/aggregate_repeat_scores.py scores_rep*.json \
  --out scores_mean_sd_baselines.json
```

## The table this produces

| System | Hardware | Overall | p50 | Data leaves perimeter |
|--------|----------|--------:|----:|-----------------------|
| Frontier API | none (hosted) | ? | ? | **yes** |
| Rules v2 + few-shot | 3 GPUs resident | **0.848** | 633 ms | no |
| Mamay-12B | 1 GPU | ? | ? | no |
| Aya-8B (best single) | 1 GPU | 0.785 | 867 ms | no |
| Mamay-4B | 1 GPU | 0.762 | 1125 ms | no |
| Lapa-12B | 1 GPU | 0.732 | 1681 ms | no |

That last column is the thesis. Everything above the router line is unavailable to the target user; everything below it is worse.
