# Supervisor Briefing — Routing Between Ukrainian Open-Source LLMs

**Date:** 2026-07-27 · **Author:** M. Onyshchuk
**Compute:** UCU lab node `ucu-lab-2240`, 4× RTX 6000 Ada (49 140 MB each), vLLM under SLURM

This is the speaking version of `docs/status_report_2026-07-27.md`. Every number below was
re-derived from the raw run files by `scripts/verify_report_stats.py`, so it can be
regenerated live if asked.

---

## 1. The one-paragraph version

The experimental platform is finished and now measures on real public Ukrainian
benchmarks instead of only my own hand-written prompts. I grew the suite from 40 to
**160 prompts** per system and evaluated **four systems** with bootstrap confidence
intervals. **The headline result is negative:** the rules router scores **0.6918**,
below the single small model at **0.7281** and the large baseline at **0.7744**, and the
deficit against the small model is statistically significant. I can explain precisely
why, and the explanation is the actual contribution of this phase: the router sends
**152 of 160** prompts to the same model as the baseline, so its only real decision is
routing code to Qwen — and Qwen is *worse* at code than the baseline. The router is
structurally incapable of winning as currently configured. I also established that a
meaningful share of all my measured differences is sampling noise, which retroactively
invalidates an earlier result I was ready to believe.

---

## 2. What is built and working

**Serving and routing stack.** A FastAPI rules router (port 4010) that is
OpenAI-compatible, supports streaming, and exposes `/v1/route/preview`. It emits its
decision in both the response body and `x-router-model` / `x-router-intent` /
`x-router-reason` headers, so **every benchmark row records which backend served it** —
that is what made the diagnosis in §5 possible. Behind it, a LiteLLM proxy (port 4000)
with aliases and fallbacks, and vLLM backends under SLURM: Mamay-4B, Qwen-Coder-3B,
Mamay-12B, plus Lapa and a 4-bit Mamay-27B.

**Evaluation tooling.** A cluster-side runner that records latency, token usage, routing
metadata and an `nvidia-smi` VRAM snapshot per request; a bucket-aware scorer;
reproducible samplers for each public dataset; suite merging; routing diagnostics; and a
verification script that recomputes the whole report from raw artifacts.

**Decoding is held fixed across all systems:** `max_tokens=256`, `temperature=0.2`,
180 s timeout. All 640 requests across the eight runs returned **HTTP 200 with zero
empty responses**.

### What changed this cycle: the test cases

This is the main build increment, and it is where the sample-size growth comes from.

| Suite | n | Composition |
|---|---|---|
| `mixed_ua_v0` | 40 | Hand-written, 8 each of chat / translate / instruct / knowledge / code |
| `mixed_ua_v1` | 64 | v0 + **24 ZNO-Eval** knowledge items |
| `mixed_ua_v2` | **160** | v1 + **48 FLORES-200** translation + **48 UAlign** alignment |
| `external_ua_v2` | 96 | FLORES + UAlign only, for incremental runs |

Bucket composition of v2: alignment 48, translate 56, knowledge 32, chat 8, code 8,
instruct 8. Three public Ukrainian sources are now integrated and scored:

- **ZNO-Eval** (`NLPForUA/ZNO`, arXiv:2501.06715) — 24 single-answer school-exam items
  across Ukrainian language, history, geography, mathematics. Exact letter match (А–Д).
  This is the same dataset family INSAIT used to report MamayLM scores, so it doubles as
  a sanity check on my large baseline.
- **FLORES-200** (`facebook/flores`) — 24 source segments in both directions (48 items),
  scored with a chrF-like character n-gram F-score against the official reference.
- **UAlign** (`Stereotypes-in-LLMs/UAlign`) — 24 ETHICS + 24 Social Chemistry 101 items,
  exact class match on a single-digit label.
- **UA-Code-Bench** — access requested, **not yet available**. The code bucket is still
  8 hand-written items, which matters a lot (see §7).

---

## 3. The systems compared

| ID | System | Backends |
|---|---|---|
| S0 | Single small model | Mamay-4B on every prompt |
| S1 | Rules router | Mamay-4B + Qwen-Coder-3B |
| S2 | Large baseline | Mamay-12B — the quality ceiling |
| — | Qwen-Coder-3B alone | Added as a control this cycle |

The specialists are deliberately **smaller than S2**; that is the efficiency premise of
the thesis. (An earlier Lapa-12B vs Mamay-12B experiment was a wiring demo, not an
efficiency experiment, and is retired.)

---

## 4. Metrics

### 4.1 Overall, 160 items

| System | Quality | 95% CI | Mean latency | p50 | p95 |
|---|---|---|---|---|---|
| **S2 · Mamay-12B** | **0.7744** | [0.7184, 0.8271] | 1 467 ms | 922 ms | 4 240 ms |
| **S0 · Mamay-4B** | **0.7281** | [0.6675, 0.7849] | 1 556 ms | 917 ms | 7 046 ms |
| **S1 · Rules router** | **0.6918** | [0.6285, 0.7522] | 1 698 ms | 967 ms | 7 446 ms |
| Qwen-Coder-3B alone | 0.6659 | [0.6029, 0.7256] | **732 ms** | **305 ms** | 3 868 ms |

Non-parametric bootstrap over items, 20 000 resamples.

### 4.2 By bucket

| Bucket | n | Mamay-4B | Qwen-3B | Router | Mamay-12B |
|---|---|---|---|---|---|
| chat | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| code | 8 | **1.000** | 0.875 | 0.875 | **1.000** |
| instruct | 8 | 0.875 | 0.875 | 0.875 | 0.875 |
| translate | 56 | 0.7767 | 0.6792 | 0.7801 | **0.8198** |
| alignment | 48 | 0.6875 | 0.6667 | 0.6458 | **0.7500** |
| knowledge | 32 | 0.5312 | 0.4531 | 0.4375 | **0.5938** |

### 4.3 The public datasets individually

| System | ZNO-Eval (24) | UAlign (48) | FLORES en→uk | FLORES uk→en | FLORES both |
|---|---|---|---|---|---|
| Mamay-12B | **12/24 = 0.500** | **36/48 = 0.750** | 0.7807 | **0.8306** | **0.8056** |
| Mamay-4B | 10/24 = 0.417 | 33/48 = 0.688 | **0.7826** | 0.8121 | 0.7974 |
| Qwen-3B | 10/24 = 0.417 | 32/48 = 0.667 | 0.5798 | 0.7762 | 0.6780 |
| Router | 7/24 = 0.292 | 31/48 = 0.646 | 0.7765 | 0.8213 | 0.7989 |

Three things to say about this table:

1. **ZNO is now my most discriminative bucket** and also brutally hard — the best system
   gets half the questions right, against a ~20–25% guessing floor.
2. **Both Mamay models translate into English better than into Ukrainian**, which is
   worth a sentence in the thesis.
3. **Qwen collapses on en→uk (0.5798)**, exactly as expected for a code model, which is
   good evidence that the buckets are measuring something real.

### 4.4 Statistical comparisons (paired bootstrap, n=160)

| Comparison | Mean difference | 95% CI | Verdict |
|---|---|---|---|
| Router − Mamay-4B | −0.0363 | [−0.0729, −0.0041] | **Significant — the router is worse** |
| Router − Mamay-12B | −0.0826 | [−0.1370, −0.0298] | **Significant — the router is worse** |
| Mamay-4B − Mamay-12B | −0.0463 | [−0.0994, +0.0049] | Not significant |
| Qwen − Mamay-4B | −0.0622 | [−0.1301, +0.0059] | Not significant |

Item-level record against Mamay-12B: Mamay-4B 26 W / 37 L / 97 T; router 27 / 40 / 93;
Qwen 13 / 67 / 80.

**The 4B model is statistically indistinguishable from the 12B model at this sample
size.** That interval crosses zero. It is a genuinely interesting efficiency finding and
I want to pursue it properly rather than over-claim it now.

### 4.5 Continuity check on the 64-item suite

| System | Quality (v1, n=64) |
|---|---|
| Mamay-12B | 0.7693 |
| Mamay-4B | 0.7066 |
| Qwen-3B | 0.6561 |
| Router | 0.6459 |

The ranking is identical to v2. Adding FLORES and UAlign moved absolute scores but not
the ordering, which is reassuring about suite stability.

---

## 5. Why the router loses — the diagnosis

This is the part I would most like to discuss.

### 5.1 The router barely routes

On 160 prompts the router sends **152 to Mamay-4B and 8 to Qwen**. Its only substantive
decision is the code bucket, and for this model pair that decision is simply wrong:
Mamay-4B scores **1.000** on code, Qwen **0.875**, failing `code-006` (Fibonacci) with
0 of 2 unit tests passing.

So by construction the router can only **match** S0 (where it forwards to the same
backend) or **lose** (where it diverts to a weaker model). It has no path to a win. That
is a design flaw I can now demonstrate with numbers rather than suspect.

### 5.2 A large share of the measured gap is noise, not routing

Since 152 of 160 prompts hit an identical backend with an identical prompt, S0 and S1
*should* be identical on those items. They are not:

- Scores differ on **37 of 152** identical-backend items, mean difference −0.0316.
- On the 96 matched `external_ua_v2` prompts the two systems produced **different text
  on 30 of 96** items, despite identical backend, prompt and decoding settings.
- Whole items flip between 0 and 1: several `ualign-social-*` and `zno-*` items flip
  against the router, one flips in its favour.

Cause: sampling at `temperature=0.2` combined with binary metrics. On multiple-choice
and label items, one changed token moves the score a full point.

**Consequence, stated plainly: any overall difference below roughly ±0.03 is currently
uninterpretable.** That applies to part of the router deficit — and it invalidates my
earlier 40-item result where the 4B model appeared to beat the 12B model. I would rather
report this than quietly keep the flattering number.

### 5.3 One clean positive result: routing overhead is essentially zero

Measured on the 96 prompts where the router forwards everything to Mamay-4B:

| Metric | Value |
|---|---|
| Mean latency, direct | 813.0 ms |
| Mean latency, via router | 809.8 ms |
| Mean per-item difference | **−3.2 ms** |
| Median per-item difference | −1.0 ms |

Regex classification plus an extra HTTP hop costs nothing measurable next to generation.
The router's worse aggregate latency comes from *which model it picks* and from
generation variance, not from routing. This also retires an earlier claim of mine that
"the router is 13% slower" — that was an artifact of comparing whole runs.

### 5.4 Routing accuracy degrades as the suite broadens

| Suite | Intent accuracy | Backend-oracle accuracy |
|---|---|---|
| v0 (40) | 32/40 = 0.800 | 40/40 = 1.000 |
| v1 (64) | 37/64 = 0.578 | 64/64 = 1.000 |
| v2 (160) | **85/160 = 0.531** | 160/160 = 1.000 |

Per-bucket on v2: translate 56/56, code 8/8, chat 7/8, instruct 7/8, knowledge 7/32,
**alignment 0/48**.

The rules have no concept of exam-style questions or moral-judgement items, so both fall
through to the default `chat` intent. And **backend-oracle accuracy of 100% is a
flattering metric, not a real one** — with only two backends the oracle is trivial, since
everything except code should go to Mamay-4B anyway. It will stop looking good the moment
a third specialist exists.

---

## 6. Resources — measured, but not yet a claim

Concurrent VRAM snapshot: GPU0 42 248 MB, GPU1 17 970 MB, GPU2 15 358 MB, GPU3
13 086 MB (other workload). Matching each figure to its configured
`gpu-memory-utilization` identifies GPU0 as Mamay-12B (0.85), GPU1 as Mamay-4B (0.35)
and GPU2 as Qwen-3B (0.30). Note this mapping is **deduced, not recorded** — the sbatch
files request `--gpus=1` and let SLURM pick the device, so future runs should log
`CUDA_VISIBLE_DEVICES` next to the snapshot.

**I do not want to present this as an efficiency result yet.** All models were resident
simultaneously and `gpu_memory_util` differs per sbatch file, so each figure is within
~800 MB of `utilisation × 49 140 MB` — these are the vLLM allocator's **reservations,
not the models' actual requirements**. Serving also runs with `--enforce-eager`, which
disables CUDA graphs and depresses throughput across the board. A fair comparison needs
one model loaded at a time at identical utilisation.

Taken at face value the reservations do give the shape of the intended efficiency story
— the router's two small backends need 33.3 GB against the 12B's 41.3 GB, about 79% —
but I would not defend that number until it is re-measured properly.

Generation length (completion tokens per prompt): Mamay-12B **22.8**, Mamay-4B **34.4**,
Qwen-3B **42.2**, router **43.4**. Note that Qwen's much larger *total* token count (184)
is mostly chat-template overhead — 142 prompt tokens versus Mamay's 89 — so Qwen is not
the most verbose generator even though it looks it in the raw totals.

Naive quality-per-second: Qwen 0.910, Mamay-12B 0.528, Mamay-4B 0.468, router 0.408.
This flatters Qwen purely because its 2-token alignment answers are extremely fast, which
tells me the efficiency metric has to be computed **per bucket**, not aggregated.

No cost figures — inference is self-hosted.

---

## 7. Weaknesses I want to name before you find them

| Issue | Consequence |
|---|---|
| Nondeterminism: `temperature=0.2`, one pass per system, binary metrics | ±0.03 is noise; needs repeated runs |
| Sampling seed not pinned in the runner or sbatch files | Runs not bit-for-bit reproducible |
| Three Mamay-12B runs disagree on the hand-written translation bucket (0.8663 / 0.6344 / 0.9048) | Only the 2026-07-27 run is citable |
| Single-reference chrF | A correct paraphrase can score below a lucky near-copy; a chatty prefix collapses the score, so translation numbers understate real quality |
| Saturated buckets: chat 1.000 and instruct 0.875 for **every** system | These carry no discriminative information |
| Fragile ZNO extraction — last-resort regex matches any standalone letter А–Д | Could credit or penalise wrongly; needs a manual audit of all 24 items |
| Alignment scoring takes the first digit 0–2 anywhere in the response | Fine at 2-token completions, fragile if models start explaining |
| **Code bucket is 8 hand-written items**; UA-Code-Bench unavailable | The router's *only* real decision rests on 8 items |
| Instruct bucket is home-made — no public UA instruction-following set | Must be declared a limitation |
| Repo hygiene: no git; `README`/`DESIGN` still describe the retired 4-specialist design; HF token sitting in `.env` | Reproducibility and disclosure risk |

---

## 8. Next steps, in priority order

1. **Make runs reproducible and quantify variance.** Pin the sampling seed or set
   `temperature=0` for all scored buckets, then run each system three times and report
   mean ± sd. **Highest-value task** — without it no difference under 0.05 is defensible,
   including my own headline result.
2. **Redesign the routing policy around measured strengths.** The evidence says Mamay-4B
   should currently handle every bucket, code included. Either find a specialist that
   actually beats it on Ukrainian code, or reframe the claim (see §9).
3. **Fix the rules for the new buckets.** Add intent patterns for exam-style MCQ and for
   moral-judgement classification so intent accuracy stops falling as the suite grows,
   then re-measure. Also replace the flattering backend-oracle metric.
4. **Measure resources honestly.** One model at a time, identical `gpu_memory_util`,
   steady-state VRAM, tokens/s, GPU-seconds per prompt. Then compute quality per GB and
   per second — **per bucket**.
5. **Strengthen the weakest scoring.** Multiple references or an LLM judge for
   translation; hand-audit the ZNO extraction against all 24 items; drop or replace the
   saturated chat bucket in the headline average.
6. **Close the code-bucket gap.** Chase UA-Code-Bench access; failing that, expand the
   hand-written set well beyond 8 items so the router's one real decision rests on a
   defensible sample.
7. **Clean up the repository.** Initialise git, move the HF token to an environment
   variable, update `README.md` and `DESIGN.md` to the current three-model design, mark
   the July 25 results as historical.

---

## 9. The decision I need from you

My original claim was *"routing across small Ukrainian specialists improves quality
per unit of compute."* On current evidence I cannot support it: with only two backends
that overlap almost completely in skill, the router is a near-no-op that loses precisely
where it acts.

I see two honest ways forward and would like your view:

- **(A) Keep the original claim and earn it.** Add genuinely complementary specialists —
  a real code model that beats Mamay-4B on Ukrainian code, and probably a third domain —
  so routing has something to exploit. Higher risk: it may still come out negative, and
  it depends on finding models that are actually better in their niche.
- **(B) Reframe the thesis as a measurement framework.** The contribution becomes
  *"here is a reproducible methodology that decides when specialist routing is justified,
  applied to Ukrainian open models — and here is the negative result it produces."* This
  is well supported by what I already have: the diagnosis, the noise floor, the overhead
  measurement, and the per-bucket comparison. Lower risk, and I think more honest.

My inclination is **B as the backbone, with A pursued opportunistically** — the framework
is defensible regardless of which way the routing result falls, and it makes the negative
result a finding rather than a failure.

Two further points I would value guidance on: whether the ZNO / FLORES / UAlign sample
sizes (24–48 each) are adequate for a diploma, and whether the statistically
indistinguishable 4B-vs-12B result is worth elevating into a claim of its own with
proper repeated runs.

---

## 10. Where everything lives

**Authoritative results:** `results/scores_mixed_ua_v2.json` + `_detail.jsonl` (n=160);
`results/scores_mixed_ua_v1.json` + `_detail.jsonl` (n=64).

**Raw runs (2026-07-27):** eight `*_mixed_ua_v1_*.jsonl` and `*_external_ua_v2_*.jsonl`
files in `results/`, each row carrying its own `vram_snapshot_mb`.

**Reproduce every number in this brief:** `python3 scripts/verify_report_stats.py`.

**Do not cite as results:** anything `results/*_20260725T*.jsonl`, plus
`scores_s0_s1*.json`, `scores_dual.json`, `scores_with_latency.json`,
`scores_small_vs_large.json`.

**Full detail:** `docs/status_report_2026-07-27.md`. Scope and metric definitions:
`docs/thesis_scope.md`. Dataset strategy: `docs/benchmarks.md`.
