# Thesis Scope — A Composite Ukrainian Assistant from Open Specialists

*Reframed 31 Aug 2026. Supersedes the original "less VRAM than one big model" framing, which our own week-8 measurements do not support.*

## 1. Problem

An organisation that needs a competent Ukrainian-language assistant **on its own hardware** — because the data cannot leave the perimeter — has no single open model that covers the workload. Every candidate is a specialist with a hole:

| Model | Wins | Collapses on |
|-------|------|--------------|
| Aya Expanse 8B | translate 0.821, chat 1.000 | knowledge 0.562 |
| Mamay-4B | code 0.984, instruct 0.781 | knowledge 0.469 |
| Lapa-12B | knowledge 0.750, alignment 0.750 | translate 0.436 |

Picking any one of them means accepting a bucket where it is close to unusable. The practical question is therefore not "which model is best" but **"can several open specialists be composed into one system that behaves like a single competent model?"**

## 2. Research question

Can a lightweight router over independently trained open Ukrainian specialists deliver a **single assistant** whose quality across both a heterogeneous workload and dependent multi-skill requests exceeds any of its members, approaches a hosted frontier model, and stays deployable on lab-scale hardware?

Sub-questions:

1. **Does composition beat selection?** Router vs each specialist alone, same suite.
2. **Does composition beat scaling?** Router (3 small specialists) vs one larger open model (Mamay-12B) on the same suite.
3. **How far from the unavailable option?** Router vs a hosted frontier API — the thing a sovereignty constraint rules out.
4. **How much machinery is justified?** One hop vs cascade vs ensemble vs quantized serving.
5. **Can the system compose skills within one request?** LLM-generated workflow vs stored workflow oracle vs direct single-model answers.

## 3. Hypothesis

Complementary specialists behind a deterministic router approximate a single stronger model's *coverage* on a mixed Ukrainian workload. **"Разом ≈ великий модель" means an effective capability ensemble, not summed parameters** — state this explicitly in the diploma. This is multi-LLM routing across independently trained models, not MoE inside one network (survey: arXiv:2603.04445).

## 4. Systems compared

| ID | System | Status |
|----|--------|--------|
| S0 | Each specialist alone (Mamay-4B, Lapa-12B, Aya-8B) | **Done** — `week_6_v4_solos`, `week_5_aya_qwen7` |
| S1 | **Rules router v2 + few-shot** (the product) | **Done** — 0.848 / p50 633 ms |
| S2 | One larger open model — Mamay-12B | **Missing on v4** (only v3: 0.793) |
| S3 | Hosted frontier API | **Missing** — the sovereignty-constrained alternative |
| S4 | Cascade / ensemble / 4-bit variants | **Done** — none beat S1 |
| S5 | Oracle composite pipeline | **Done** — 0.882, diagnostic ceiling on composite v1 |
| S6 | Hybrid planner + specialist executor | **Done** — 0.768; valid 0.991, exact workflow 0.806 |

S2 and S3 are the two reference points that turn "router beats its own pool" (near-tautological) into "composition is the right choice under these constraints."

## 5. Evaluation

**Single-skill next suite:** `benchmarks/mixed_ua_v6.jsonl` is frozen (2414 items; headline 683 knowledge / 683 translate / 683 alignment / 327 instruct; chat+code appendix). Methods: [`docs/mixed_ua_v6.md`](mixed_ua_v6.md). Published claims stay on `mixed_ua_v4_balanced.jsonl` (192×3) until the 3× v6 bake-off is scored — **do not quote screening means**. Decoding for v6: `T=0`, `seed=42`, `max_tokens=1024`.

Sources: ZNO-Eval (knowledge), FLORES-200 (translate), UAlign (alignment), UA-Code + HumanEval (code). Chat and instruct are hand-written — open UA data barely exists for them, and that is a stated limitation.

**Composite suite:** `benchmarks/mixed_ua_composite_v1.jsonl` — 36 deterministic items, 12 each for translate→code, knowledge→explain, and translate→knowledge→write. The stored workflow is hidden from the LLM planner. Final success uses executable tests, exact JSON/labels, and constrained evidence checks.

**Metrics:** quality per bucket/family, final task success, all-stages-pass, plan exact match, intent F1, invalid-plan rate, calls, latency mean / p50 / p95, GPU-seconds per prompt for self-hosted systems, USD per 1000 queries for the API baseline.

## 6. What we claim about resources — and what we do not

**Claim:** the router lowers **per-query latency** (p50 633 ms vs 867–1681 ms for singles) and costs roughly **1.5 GPU-seconds per prompt**, with routing overhead itself immeasurable (−3.2 ms).

**Do not claim:** lower resident VRAM. An always-on three-model deployment occupies three GPUs of KV-reserved memory at `gpu_memory_utilization=0.90`. Week-8 4-bit did not change this and is **not** a packing result. Colocating the pool on one card was never measured.

## 7. Out of scope

- Training or fine-tuning the **specialist** LLMs
- Production SLA, autoscaling, uptime
- Non-Ukrainian workloads
- Mixing fairness-preference datasets (StereoSet-UK, BBQ-UK, …) into the UAlign quality mean

Learned **routing** (embedding kNN / logistic regression profiles `knn` and `clf`) is in scope as a RouterBench-style comparison against Rules v2, Oracle, and BestSingle. The product default remains Rules v2 until that bake-off wins.

## 8. Composite-task phase

The first phase (`mixed_ua_v4`) exercises one skill and one hop. The second phase adds requests where intermediate output is consumed by a later specialist. It compares:

- direct single-model response;
- current one-hop rules router;
- stored oracle workflow executed by specialists;
- Mamay-4B JSON plan executed by the same specialist router.

This supports claims only over the three tested workflow families. It is not evidence that arbitrary real-world tasks can be decomposed reliably.

**Composite finding:** the oracle pipeline establishes a 0.070 quality
opportunity over rules-v2 direct (0.882 vs 0.812), but the prompted planner
scores 0.768 at 3.5 calls and 13.5 s p50. Therefore workflow composition is
potentially useful, but general planning is not yet the product router.

## 9. Compute note

UCU lab node `ucu-lab-2240`, 4× RTX 6000 Ada (49 140 MB each), vLLM under SLURM.

_(Add grant wording — ELEKS / Talents for Ukraine — before submission.)_
