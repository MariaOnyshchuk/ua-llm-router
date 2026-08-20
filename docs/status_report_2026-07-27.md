# Thesis Status Report — Routing Between Ukrainian Open-Source LLMs

**Date:** 2026-07-27
**Author:** M. Onyshchuk
**Compute:** UCU lab node `ucu-lab-2240`, 4× NVIDIA RTX 6000 Ada (49 140 MB each), vLLM via SLURM
**Scope of this document:** everything built and measured to date, consolidated. Supersedes the ad-hoc numbers in `scores_s0_s1.json`, `scores_s0_s1_s2.json`, `scores_dual.json` and `scores_small_vs_large.json`.

---

## 1. Executive summary

The experimental platform is complete and now runs on real public Ukrainian benchmarks (ZNO-Eval, FLORES-200, UAlign) instead of only hand-written prompts. Sample size grew from 40 to 160 prompts per system, and four systems were evaluated instead of three.

**The central result is negative and worth stating plainly:** on the 160-item suite the rules router (S1) scores **0.6918**, below the fixed Mamay-4B baseline (S0) at **0.7281** and well below Mamay-12B (S2) at **0.7744**. The paired 95% confidence interval for S1 − S0 is **[−0.073, −0.004]**, so the router is genuinely, if narrowly, worse rather than merely noisy.

**The diagnostic explanation is the most useful finding of this phase.** The current router sends **152 of 160** prompts to the same backend that S0 uses. Only the 8 code prompts are routed differently — to Qwen-Coder-3B, which scores 0.875 against Mamay-4B's 1.000 on that bucket. So the router is close to a no-op that loses exactly where it acts. Everything else in the gap is sampling nondeterminism: on the 152 identical-backend prompts the two systems still produce different scores on **37 items**, and different text on **30 of 96** items in a directly comparable run.

Two secondary findings matter for the thesis argument:

- **Router overhead is negligible.** Measured on 96 same-backend prompts, the router adds **−3.2 ms** on average (median −0.9 ms). Earlier claims that "the router is 13% slower" were an artifact of comparing runs, not of routing cost.
- **The 4B model is statistically indistinguishable from the 12B model.** S0 − S2 is −0.0463 with a 95% CI of **[−0.099, +0.005]**, which crosses zero. The larger model is directionally better but not provably so at n=160.

---

## 2. What is built

### 2.1 Serving and routing stack

| Layer | Location | Status |
|---|---|---|
| Rules router (FastAPI, port 4010) | `router/app.py` | Working; OpenAI-compatible, streaming supported, exposes `/v1/route/preview` |
| Intent rules | `router/intent_rules.py` | Working; priority-ordered regex, first match wins |
| LiteLLM proxy (port 4000) | `litellm/config.yaml` | Aliases `mamay4`, `qwen`, `mamay12` with fallbacks |
| vLLM backends | `cluster/serve_*.sbatch` | Mamay-4B :8003, Qwen-Coder-3B :8004, Mamay-12B :8002, Lapa :8001, Mamay-27B (4-bit) |

The router exposes routing metadata both in the response body and as `x-router-model` / `x-router-intent` / `x-router-reason` headers, so every benchmark row records which backend served it.

### 2.2 Evaluation tooling

| Script | Purpose |
|---|---|
| `scripts/run_small_router_cluster.py` | Cluster-side runner for S0/S1/S2 plus a standalone Qwen system; records latency, tokens, routing metadata, and an `nvidia-smi` VRAM snapshot per run |
| `scripts/score_results.py` | Bucket-specific scoring: chrF-like for translation, executed unit tests for code, format checks for instruction following, ZNO letter matching, UAlign label matching |
| `scripts/sample_zno.py`, `sample_flores.py`, `sample_ualign.py`, `sample_uacode.py` | Reproducible sampling from public datasets |
| `scripts/merge_benchmarks.py` | Builds the v1 and v2 suites from v0 plus samples |
| `scripts/analyze_routing.py` | Intent accuracy and backend-oracle accuracy against gold bucket labels |
| `scripts/summarize_results.py` | Raw latency and routing summary from JSONL |
| `scripts/verify_report_stats.py` | Recomputes every headline number in this report from the raw runs (bootstrap CIs, paired differences, routing distribution, nondeterminism, overhead, per-dataset accuracy, tokens, run health) |

Decoding is fixed across all systems: `max_tokens=256`, `temperature=0.2`, 180 s timeout.

### 2.3 Benchmark suites

| Suite | n | Composition |
|---|---|---|
| `mixed_ua_v0` | 40 | Hand-written: 8 each of chat, translate, instruct, knowledge, code |
| `mixed_ua_v1` | 64 | v0 plus 24 ZNO-Eval knowledge items |
| `mixed_ua_v2` | 160 | v1 plus 48 FLORES-200 translation items plus 48 UAlign alignment items |
| `external_ua_v2` | 96 | FLORES plus UAlign only, for incremental runs |

Bucket composition of v2: alignment 48, translate 56, knowledge 32, chat 8, code 8, instruct 8.

Public sources now integrated:

- **ZNO-Eval** (`NLPForUA/ZNO`, arXiv:2501.06715) — 24 single-answer Ukrainian school exam items across Ukrainian language, history, geography, and mathematics. Scored by exact letter match (А–Д).
- **FLORES-200** (`facebook/flores`) — 24 source segments evaluated in both directions (48 items). Scored with a chrF-like character n-gram F-score against the official reference.
- **UAlign** (`Stereotypes-in-LLMs/UAlign`) — 24 ETHICS plus 24 Social Chemistry 101 items. Scored by exact class match on a single-digit label.
- **UA-Code-Bench** — access requested, not yet sampled. The code bucket is still the 8 hand-written items.

---

## 3. Experiment inventory

All runs below returned **HTTP 200 on every request with no empty responses**.

### 3.1 Current phase (2026-07-27)

| Run file | System | Suite | n | Mean latency |
|---|---|---|---|---|
| `mamay4_mixed_ua_v1_20260727T070125Z.jsonl` | S0 | v1 | 64 | 2 669 ms |
| `router_small_mixed_ua_v1_20260727T070416Z.jsonl` | S1 | v1 | 64 | 3 029 ms |
| `mamay12_mixed_ua_v1_20260727T070730Z.jsonl` | S2 | v1 | 64 | 1 966 ms |
| `qwen_mixed_ua_v1_20260727T070936Z.jsonl` | Qwen alone | v1 | 64 | 1 212 ms |
| `mamay4_external_ua_v2_20260727T082941Z.jsonl` | S0 | external v2 | 96 | 813 ms |
| `router_small_external_ua_v2_20260727T083059Z.jsonl` | S1 | external v2 | 96 | 810 ms |
| `mamay12_external_ua_v2_20260727T083217Z.jsonl` | S2 | external v2 | 96 | 1 135 ms |
| `qwen_external_ua_v2_20260727T083406Z.jsonl` | Qwen alone | external v2 | 96 | 412 ms |

Scored into `results/scores_mixed_ua_v1.json` (n=64) and `results/scores_mixed_ua_v2.json` (n=160).

### 3.2 Earlier phases (2026-07-25, retained as history)

| Experiment | Systems | Outcome | Status |
|---|---|---|---|
| Lapa wiring demo | Lapa vs router-to-Lapa | 0.8286 vs 0.8280 — router was a no-op | Superseded |
| Dual large router | Lapa + Mamay-12B | 0.9310 vs Lapa 0.8286 and Mamay-12B 0.9233 | Superseded; violates the "specialists smaller than S2" rule |
| Small vs large, v0 | Mamay-4B, router, Mamay-12B | 0.8805 / 0.8568 / 0.8769 on 40 items | Superseded by v1 and v2 |

**Do not quote the v0 numbers as thesis results.** They rest on 8 translation items and are dominated by scoring artifacts; see §7.

---

## 4. Results on the 160-item suite (v2)

### 4.1 Overall

| System | Backends | Quality | 95% CI | Mean latency | p50 | p95 | Tokens/prompt |
|---|---|---|---|---|---|---|---|
| **S2 · Mamay-12B** | 12B | **0.7744** | [0.7181, 0.8276] | 1 467 ms | 922 ms | 4 240 ms | 111.5 |
| **S0 · Mamay-4B** | 4B | **0.7281** | [0.6673, 0.7866] | 1 556 ms | 917 ms | 7 046 ms | 123.1 |
| **S1 · Rules router** | 4B + Qwen-3B | **0.6918** | [0.6280, 0.7527] | 1 698 ms | 967 ms | 7 446 ms | 133.5 |
| Qwen-Coder-3B alone | 3B | 0.6659 | [0.6032, 0.7255] | **732 ms** | **305 ms** | 3 868 ms | 184.0 |

Confidence intervals are non-parametric bootstrap over items (20 000 resamples).

### 4.2 By bucket

| Bucket | n | Mamay-4B | Qwen-3B | Router | Mamay-12B |
|---|---|---|---|---|---|
| chat | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| code | 8 | **1.000** | 0.875 | 0.875 | **1.000** |
| instruct | 8 | 0.875 | 0.875 | 0.875 | 0.875 |
| translate (all) | 56 | 0.7767 | 0.6792 | 0.7801 | **0.8198** |
| alignment (UAlign) | 48 | 0.6875 | 0.6667 | 0.6458 | **0.7500** |
| knowledge (incl. ZNO) | 32 | 0.5312 | 0.4531 | 0.4375 | **0.5938** |

### 4.3 Public-dataset detail

**ZNO-Eval, 24 multiple-choice items** — correct answers:

| System | Correct | Accuracy |
|---|---|---|
| Mamay-12B | 12/24 | 0.500 |
| Mamay-4B | 10/24 | 0.417 |
| Qwen-3B | 10/24 | 0.417 |
| Router | 7/24 | 0.292 |

All systems clear the ~20–25% random-guess floor, but not by much. This is now the hardest bucket and the one that discriminates best between models.

**FLORES-200, 48 items, chrF-like score:**

| System | en→uk (24) | uk→en (24) | Both |
|---|---|---|---|
| Mamay-12B | 0.7807 | **0.8306** | **0.8056** |
| Router | 0.7765 | 0.8213 | 0.7989 |
| Mamay-4B | **0.7826** | 0.8121 | 0.7974 |
| Qwen-3B | 0.5798 | 0.7762 | 0.6780 |

Both Mamay models translate into English slightly better than into Ukrainian. Qwen is clearly unsuited to en→uk, which is consistent with it being a code model.

**UAlign, 48 items, exact label match:**

| System | Correct | Accuracy |
|---|---|---|
| Mamay-12B | 36/48 | 0.750 |
| Mamay-4B | 33/48 | 0.688 |
| Qwen-3B | 32/48 | 0.667 |
| Router | 31/48 | 0.646 |

Responses here are 2 completion tokens, so alignment latency is 54–190 ms and contributes almost nothing to the mean latency figures.

### 4.4 Statistical comparisons (paired bootstrap, n=160)

| Comparison | Mean difference | 95% CI | Verdict |
|---|---|---|---|
| Router − Mamay-4B | −0.0363 | [−0.0730, −0.0041] | **Significant**: the router is worse |
| Router − Mamay-12B | −0.0826 | [−0.1378, −0.0309] | **Significant**: the router is worse |
| Mamay-4B − Mamay-12B | −0.0463 | [−0.0989, +0.0051] | Not significant |
| Qwen − Mamay-4B | −0.0622 | [−0.1300, +0.0049] | Not significant |

Item-level win/loss against Mamay-12B: Mamay-4B 26 wins / 37 losses / 97 ties; router 27 / 40 / 93; Qwen 13 / 67 / 80.

### 4.5 Results on the 64-item suite (v1), for continuity

| System | Quality | Mean latency |
|---|---|---|
| Mamay-12B | 0.7693 | 1 966 ms |
| Mamay-4B | 0.7066 | 2 669 ms |
| Qwen-3B | 0.6561 | 1 212 ms |
| Router | 0.6459 | 3 029 ms |

The ordering is identical to v2, which is reassuring: adding FLORES and UAlign changed absolute scores but not the ranking.

---

## 5. Why the router loses: diagnosis

### 5.1 The router barely routes

On the 160-item suite the router sends **152 prompts to Mamay-4B and 8 to Qwen**. Its only substantive decision is the code bucket, and that decision is wrong for this model pair: Mamay-4B scores 1.000 on code while Qwen scores 0.875, failing `code-006` (Fibonacci) with 0 of 2 unit tests passing.

So the router's design currently guarantees it cannot beat S0. It can only match S0 (where it forwards to the same backend) or lose (where it diverts to a weaker specialist).

### 5.2 A large share of the measured gap is nondeterminism, not routing

Because 152 of 160 prompts hit the identical backend with the identical prompt, S0 and S1 should score identically on those items. They do not:

- Scores differ on **37 of 152** identical-backend items, with a mean difference of **−0.0316**.
- In the directly comparable `external_ua_v2` pair of runs, the two systems produced **different text on 30 of 96** items despite identical backend, prompt, and decoding settings.
- Whole items flip between 0 and 1: `ualign-social-011`, `ualign-social-012`, `ualign-social-022`, `zno-geo-30-018`, `zno-mat-00-010`, `zno-mat-04-001` all flipped against the router, while `ualign-social-013` flipped in its favour.

The cause is sampling at `temperature=0.2` combined with binary scoring metrics. On multiple-choice and label-classification items a single token change moves the score by a full point.

**Implication:** without repeated runs, any difference of roughly ±0.03 in overall quality is indistinguishable from run-to-run variance. This applies retroactively to the v0 "4B beats 12B" result and to the current router deficit, part of which is real (the code regression) and part of which is noise.

### 5.3 Routing overhead is not the problem

Measured on the 96 `external_ua_v2` prompts, where the router forwards everything to Mamay-4B:

| Metric | Value |
|---|---|
| Mean latency, direct Mamay-4B | 813.0 ms |
| Mean latency, via router | 809.8 ms |
| Mean per-item difference | **−3.2 ms** |
| Median per-item difference | −0.9 ms |

Regex classification plus an extra HTTP hop costs nothing measurable relative to generation time. The router's higher aggregate latency in the summary tables comes from *which model it selects* (Qwen was slower than Mamay-4B on the code bucket in these runs, at 3 149 ms versus 2 000 ms) and from generation variance, not from routing itself.

### 5.4 Routing accuracy degrades as the suite broadens

| Suite | Intent-label accuracy | Backend-oracle accuracy |
|---|---|---|
| v0 (40) | 32/40 = 0.800 | 40/40 = 1.000 |
| v1 (64) | 37/64 = 0.578 | 64/64 = 1.000 |
| v2 (160) | 85/160 = 0.531 | 160/160 = 1.000 |

Per-bucket intent accuracy on v2: translate 56/56, code 8/8, chat 7/8, instruct 7/8, knowledge 7/32, **alignment 0/48**.

The rules have no concept of ZNO-style exam questions or of UAlign moral-judgement items, so both fall through to the default `chat` intent. Backend-oracle accuracy stays at 100% only because the oracle is trivial with two backends — everything except code should go to Mamay-4B anyway. That metric will stop being flattering the moment a third specialist is added.

---

## 6. Resource measurements

Concurrent VRAM snapshot recorded with every run (4× RTX 6000 Ada, 49 140 MB each):

| GPU | Used | Implied occupant | Evidence |
|---|---|---|---|
| 0 | 42 248 MB | Mamay-12B | `util 0.85` → 41 769 MB implied, +479 MB overhead |
| 1 | 17 970 MB | Mamay-4B | `util 0.35` → 17 199 MB implied, +771 MB overhead |
| 2 | 15 358 MB | Qwen-Coder-3B | `util 0.30` → 14 742 MB implied, +616 MB overhead |
| 3 | 13 086 MB | other workload | — |

**The GPU-to-model mapping is inferred, not recorded.** The sbatch files request
`--gpus=1` and let SLURM choose the device, and the runner stores only an unlabelled
`nvidia-smi` snapshot, so nothing in the artifacts ties an index to a model. The
attribution above comes from matching each observed figure to its configured
`gpu-memory-utilization`; the three ratios separate cleanly, which is why it is
reliable. An earlier version of this table swapped GPUs 1 and 2. Future runs should log
`CUDA_VISIBLE_DEVICES` alongside the snapshot so this is recorded rather than deduced.

**This does not yet support an efficiency claim.** All models were resident simultaneously and `gpu_memory_util` differs per sbatch file (0.85 for the 12B, 0.35 and 0.30 for the small models), so the numbers reflect the vLLM allocator's reservation, not the model's actual requirement. A fair comparison needs one model loaded at a time with identical utilisation settings.

Token accounting over 160 prompts. Verbosity must be read off **completion** tokens; `total_tokens` mixes in the prompt, and the prompt cost differs per model because the chat templates and tokenizers differ.

| System | Prompt | **Completion** | Total |
|---|---|---|---|
| Mamay-12B | 88.7 | **22.8** | 111.5 |
| Mamay-4B | 88.7 | **34.4** | 123.1 |
| Qwen-3B | 141.8 | **42.2** | 184.0 |
| Router | 90.2 | **43.4** | 133.5 |

The most verbose generator is the router (43.4 completion tokens), marginally ahead of Qwen (42.2); Mamay-12B is by far the most concise at 22.8. Qwen's large `total` is mostly template overhead — it spends 142 prompt tokens where Mamay spends 89 — so it should not be described as the most verbose model.

Naive quality-per-second of mean latency: Qwen 0.910, Mamay-12B 0.528, Mamay-4B 0.468, router 0.408. This favours Qwen purely because its 2-token alignment answers are extremely fast, which shows the metric needs to be computed per bucket rather than aggregated.

No monetary cost data exists; inference is self-hosted.

---

## 7. Measurement issues found and open

| Issue | Detail | Consequence |
|---|---|---|
| Run-to-run nondeterminism | `temperature=0.2`, single pass per system, binary metrics | ±0.03 overall quality is noise; needs repeated runs |
| No fixed seed | vLLM sampling seed not pinned in the sbatch files or the runner | Runs are not reproducible bit-for-bit |
| Three different Mamay-12B runs disagree | Hand-written translation bucket scored 0.8663, 0.6344, and 0.9048 across three runs of the same model | Only the 2026-07-27 run should be cited |
| Single-reference chrF | A correct paraphrase can score below a lucky near-copy; chatty prefixes such as "Звичайно! Ось переклад:" collapse the score | Translation numbers understate real quality |
| Saturated buckets | chat is 1.000 and instruct is 0.875 for every system | These buckets carry no discriminative information |
| Fragile ZNO extraction | The final fallback in `score_knowledge` matches any standalone letter А–Д in the response | Risk of crediting or penalising the wrong answer; needs an audit |
| Alignment scoring | The first digit 0–2 anywhere in the response is taken as the label | Fine while completions are 2 tokens; fragile if models start explaining |
| Code bucket still tiny | 8 hand-written items; UA-Code-Bench not yet accessible | The one decision the router actually makes rests on 8 items |
| Instruct bucket is home-made | No Ukrainian public instruction-following set used | Must be declared as a limitation |
| Repository hygiene | No git repository; `README.md` and `DESIGN.md` still describe the four-specialist Lapa/Gemma design; `run_local_stack.sh` and `sync_cluster.sh` still target Lapa only; `.env` holds a Hugging Face token | Reproducibility and disclosure risk |

---

## 8. Next steps, in priority order

1. **Make runs reproducible and quantify variance.** Pin the sampling seed, or set `temperature=0` for all scored buckets, and run each system three times. Report mean and standard deviation. Without this, no difference under 0.05 can be defended. This is the single highest-value task.
2. **Redesign the routing policy around measured strengths.** The evidence says Mamay-4B should currently handle every bucket, including code. Either find a specialist that actually beats Mamay-4B on Ukrainian code prompts, or change the thesis claim from "specialist routing improves quality" to "routing must be justified per bucket, and here is the measurement framework that decides it."
3. **Fix the rules for the new buckets.** Add intent patterns for exam-style multiple choice and for moral-judgement classification, so intent accuracy stops falling as the suite grows. Then re-measure.
4. **Measure resources honestly.** Load one model at a time with identical `gpu_memory_util`, record steady-state VRAM, tokens per second, and GPU-seconds per prompt. Only then compute quality per gigabyte and quality per second, and compute them per bucket.
5. **Strengthen scoring where it is weakest.** Add multiple references or an LLM judge for translation, audit the ZNO extraction regex against all 24 items by hand, and either drop the saturated chat bucket from the headline average or replace it with something discriminative.
6. **Close the code-bucket gap.** Pursue UA-Code-Bench access; failing that, expand the hand-written code set well beyond 8 items so the router's only real decision rests on a defensible sample.
7. **Clean up the repository.** Initialise git, remove the token from `.env` and document it as an environment variable, update `README.md` and `DESIGN.md` to the current three-model design, and mark the July 25 result files as historical.

---

## 9. Talking points for the supervisor meeting

1. "The platform is finished and now runs on real public Ukrainian benchmarks: ZNO-Eval, FLORES-200, and UAlign. Sample size went from 40 to 160 prompts per system, and I now evaluate four systems with bootstrap confidence intervals."
2. "The headline result is negative. The rules router scores 0.692 against 0.728 for Mamay-4B alone and 0.774 for Mamay-12B, and the router deficit against Mamay-4B is statistically significant."
3. "I can explain exactly why. The router sends 152 of 160 prompts to the same model as the baseline, so its only real decision is routing code to Qwen — and Qwen is worse on code than Mamay-4B. The router is structurally unable to win as currently configured."
4. "I also found that a meaningful part of the measured differences is sampling noise. On prompts where the router and the baseline use the identical backend, scores still differ on 37 items. That means any gap below roughly 0.03 is not interpretable, and it invalidates my earlier 40-item result where the 4B model appeared to beat the 12B model."
5. "One clearly positive finding: routing overhead is essentially zero, at −3.2 ms mean across 96 matched prompts. Whatever the router costs, it is not latency."
6. "Mamay-4B and Mamay-12B are statistically indistinguishable at this sample size, with a confidence interval of [−0.099, +0.005]. That is itself an interesting efficiency result worth pursuing properly."
7. "My proposed next phase is: pin determinism and run repeats, rebuild the routing policy from measured per-bucket strengths, and measure VRAM and throughput with one model loaded at a time. The framing shifts from 'routing wins' to 'here is a measurement framework that decides when routing is justified' — which I believe is the more honest and more defensible thesis."

---

## 10. Artifact index

**Authoritative results:** `results/scores_mixed_ua_v2.json` and `results/scores_mixed_ua_v2_detail.jsonl` (n=160), `results/scores_mixed_ua_v1.json` and `_detail.jsonl` (n=64).

**Raw runs, 2026-07-27:** the eight `*_mixed_ua_v1_*.jsonl` and `*_external_ua_v2_*.jsonl` files in `results/`. Each row carries its own `vram_snapshot_mb` field; the per-run `.meta.json` files written by the runner were left on the cluster and still need to be copied back.

**Benchmarks:** `benchmarks/mixed_ua_v0.jsonl`, `mixed_ua_v1.jsonl`, `mixed_ua_v2.jsonl`, `external_ua_v2.jsonl`, and the per-source samples in `benchmarks/samples/`.

**Historical, do not cite as results:** all `results/*_20260725T*.jsonl`, plus `scores_s0_s1.json`, `scores_s0_s1_s2.json`, `scores_dual.json`, `scores_with_latency.json`, `scores_small_vs_large.json`.

**Related documents:** `docs/thesis_scope.md` (research question, systems, metrics), `docs/benchmarks.md` (dataset strategy and sampling commands).
