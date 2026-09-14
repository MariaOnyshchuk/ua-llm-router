# mixed_ua_v6 methods

Validate the **measuring instrument** before choosing a router architecture (tinyBenchmarks / IRT-inspired curation). Historical suites `mixed_ua_v3`–`v5` stay frozen. **Do not quote screening-pool means as thesis scores** — those items were drawn to *find* a suite, not to report quality.

## Why a smaller suite is still valid

tinyBenchmarks / IRT-style evaluation keeps items that **discriminate** between systems (high between-model score variance) and then **balances difficulty**, instead of scoring thousands of near-ties. The headline target is the binomial sample size

\[
N \ge \bigl(z \cdot \sqrt{p(1-p)} / \varepsilon\bigr)^2
\]

with \(p=0.8\), \(\varepsilon=0.03\), \(z=1.96\) → **n\* = 683** per headline bucket. That is the CI claim, not “bigger corpus = better paper.” Screening used up to 1200 items/bucket; curation drops the lowest-variance fraction (`drop_frac=0.43` so a 1200-pool still yields 683 after the drop) and samples equal terciles of mean specialist score.

**Instruct** cannot hit n\*: the warehouse only has 573 IFEval-UA items (327 kept). Report a wider CI there; do not pad with English IFEval.

Frozen file: `benchmarks/mixed_ua_v6.jsonl` (seed 42). Meta: `benchmarks/mixed_ua_v6_meta.json`.

## Headline vs appendix

| Role | Buckets | Why |
|------|---------|-----|
| Headline | knowledge, translate, alignment, instruct | Public UA corpora (ZNO, Belebele, MMLU-UA, ARC-UA, FLORES, WMT-22, UAlign, IFEval-UA). Target n\* = 683 where the pool allows. |
| Appendix | chat, code | No large scored UA chat dump; UA-Code needs an Eolymp judge. Keep small; **do not** use for CI claims. Chat scoring remains a UA-ratio heuristic (low discriminability). |

Fairness sets (StereoSet-UK, WinoBias-UK, BBQ-UK) are **not** mixed into UAlign quality: they use different metrics.

Variance for curation uses **Mamay-4B, Lapa, Aya only**. The Mamay-12B screening pass is incomplete (HTTP 0 / empty generations on knowledge, alignment, and code). Including those zeros would inflate “discriminability” with a serving failure. 12B remains an optional S2 bake-off, not a router member.

## Router choice vs RouteLLM / GraphRouter

This thesis is **multi-LLM routing over independently trained Ukrainian specialists**, not MoE.

| Method | When / what / how | Role here |
|--------|-------------------|-----------|
| **Rules v2** (product default) | Pre-generation; task/domain regex; one hop | Deterministic, no extra encoder, matches the lab API |
| **Embedding kNN / logistic clf** | Pre-generation; prompt embedding (multilingual E5); sklearn | LLMRouterBench-style learned alternative. Train labels = argmax specialist on screening items **held out of v6**. |
| RouteLLM | Often preference/Bradley–Terry over strong/weak pairs | Different problem (cheap vs expensive *same family*). Cite, do not reimplement. |
| GraphRouter | Task–model graph / trained scorer | Heavier machinery than three specialists justify until the bake-off beats rules. |

A learned router that does **not** beat BestSingle is still a reportable result. Default profile stays **v2** until v6 says otherwise.

## Cascade note (FrugalGPT)

The v4 micro-cascade escalate rate ≈1.56% with Δ≈0 is expected: cascade gain scales with the **share of queries that need escalation** (FrugalGPT). Do not treat that as a method failure on a bucket the first hop already solves. Re-evaluate cascade only after v6 quality numbers exist — not from screening.

## Pipeline

```bash
# 1. Warehouse (HF; skip keys that fail)
python scripts/extract_full_corpora.py --only ifeval,belebele,mmlu,arc,wmt22

# 2. Screening draw (≤1200/bucket or the whole pool)
python scripts/sample_screening_suite.py --per-bucket 1200 --seed 42 \
  --out benchmarks/mixed_ua_v6_screen.jsonl

# 3. Cluster: one pass per model (not 3×)
bash scripts/week_v6_eval_pipeline.sh screen

# 4. Curate (exclude incomplete mamay12 screening)
PYTHONPATH=. python scripts/curate_discriminative_suite.py \
  --suite benchmarks/mixed_ua_v6_screen.jsonl \
  --scores results/v6_screen/*_detail.jsonl \
  --systems mamay4 lapa aya \
  --out benchmarks/mixed_ua_v6.jsonl --keep-appendix

# 5. Train live profiles (hold out v6 ids)
PYTHONPATH=. python scripts/train_learned_router.py \
  --scores results/v6_screen/*_detail.jsonl \
  --suite benchmarks/mixed_ua_v6_screen.jsonl \
  --holdout benchmarks/mixed_ua_v6.jsonl \
  --encoder e5

# 6. 3× bake-off on v6 (T=0, seed=42, max_tokens=1024)
# Serve jobs: 48h wall (cluster/serve_{mamay4,lapa,aya}.sbatch).
# Split solos if needed: v6_solo_mamay4 | v6_solo_lapa | v6_solo_aya
bash scripts/week_v6_eval_pipeline.sh v6_solos
bash scripts/week_v6_eval_pipeline.sh v6_rules
bash scripts/week_v6_eval_pipeline.sh v6_learned
```

Screening/v6 decoding: `T=0`, `seed=42`, **`max_tokens=1024`** (IFEval). Do not mix 256-token v4 numbers with this suite.

`v6_learned` writes `results/v6_eval/router_comparison.json` (solos, Rules v2, kNN, clf, Oracle, BestSingle, leave-one-bucket-out). Headline buckets only.

## VRAM / GPU-seconds (optional, efficiency)

Not required to freeze quality numbers. Protocol: **one live model at a time** for solos; `gpu_memory_utilization=0.90` still reports KV reservation on nvidia-smi, not bare weights. Use `gpu_seconds` from run metadata for the cost axis.

## Systems

Rules **v2** remains the default API profile. `knn` / `clf` are selectable profiles in `router/intent_rules.py`. Leave-one-bucket-out accuracy is written to `router/artifacts/train_report.json` at train time and copied into the comparison payload.
