# Week 10 — composite router

*Completed 1 Sep 2026. 36 items × 3 deterministic repeats, T=0, seed=42,
max_tokens=256. Authoritative summary:
`results/week_10_composite/scores.json`.*

## Question

Can the specialist pool solve one request that requires dependent skills, rather
than merely route different independent requests to different models?

Composite v1 has 36 controlled items:

| Family | n | Oracle workflow | Final check |
|--------|--:|-----------------|-------------|
| translate→code | 12 | Aya → Mamay-4B | executable Python cases |
| knowledge→explain | 12 | Lapa → Mamay-4B | correct label + exact two-line evidence format |
| translate→knowledge→write | 12 | Aya → Lapa → Mamay-4B | exact JSON fields |

The model planner receives only the original user request. The benchmark's
`oracle_plan` is used by the oracle system and scorer, never included in the
planner prompt.

## Systems

1. Each specialist answers the full request directly.
2. Rules v2 answers it in one hop (the pre-week-10 router).
3. Oracle workflow executes benchmark-defined stages through the same
   specialist map.
4. Hybrid workflow asks Mamay-4B for a bounded JSON plan, validates it, then
   executes each dependent stage through the specialist map.

This separates two failure sources:

- **workflow error**: invalid or wrong decomposition;
- **execution error**: the workflow is right but a specialist output fails.

## Metrics

- Final score / exact final success
- All stages pass and mean stage score
- Valid-plan rate, exact workflow match, intent F1
- Calls per item, end-to-end latency, GPU-seconds per item

Direct systems have no observable intermediate stages, so compare them on
**final score**. All-stages-pass is a diagnostic for oracle/hybrid workflows,
not a penalty applied to direct models.

## Results

| System | Final score | SD | Exact success | Calls | p50 |
|--------|------------:|---:|--------------:|------:|----:|
| **Oracle workflow** | **0.882** | 0.000 | **0.667** | 2.33 | 3711 ms |
| **Rules v2 direct** | **0.812** | 0.000 | 0.583 | **1.00** | 1725 ms |
| Mamay-4B direct | 0.794 | 0.004 | 0.593 | 1.00 | 1718 ms |
| Qwen-7B direct | 0.778 | 0.000 | 0.528 | 1.00 | **666 ms** |
| **Hybrid planner v2** | **0.768** | 0.032 | 0.435 | 3.50 | 13 533 ms |
| Aya-8B direct | 0.755 | 0.016 | 0.500 | 1.00 | 1517 ms |
| Lapa-12B direct | 0.736 | 0.000 | 0.417 | 1.00 | 2183 ms |

The oracle proves **composition headroom**: a correct workflow improves 0.070
over the current one-hop router and 0.088 over the best direct model. The
prompted planner does not realize that headroom: it is 0.044 below one-hop,
uses 3.5 calls, and is about 7.8× slower at p50.

### Family breakdown

| Family | Oracle | Rules direct | Mamay-4B | Hybrid v2 |
|--------|-------:|-------------:|---------:|----------:|
| translate→code | 0.958 | 0.958 | **1.000** | 0.931 |
| knowledge→explain | 0.917 | 0.917 | **0.924** | 0.694 |
| translate→knowledge→write | **0.771** | 0.563 | 0.458 | 0.681 |

Only the three-stage source→JSON family benefits clearly from composition.
Simple translation→code and knowledge→explain tasks are already solved better
by one direct model; forcing decomposition adds failure surfaces.

### Planner audit

- Valid plan: **0.991**
- Exact workflow match: **0.806**
- Intent F1: **0.954**
- translate→code: 36/36 exact workflows
- translate→knowledge→write: 36/36 exact workflows
- knowledge→explain: 15/36 exact; 18/36 inserted an unnecessary translation
  step, two stopped after that step, and one invalid plan fell back.

The first planner prompt scored 0.609 and exact-match 0.333. Definitions,
three workflow examples, and a deterministic final-format guard improved it to
0.768 / 0.806. No further tuning was done on the full test set.

## Implementation

- Benchmark builder: `scripts/build_composite_benchmark.py`
- Benchmark: `benchmarks/mixed_ua_composite_v1.jsonl`
- Planner: `router/planner.py`
- Executor: `router/orchestrator.py`
- Shared backend call: `router/client.py`
- Runner: `scripts/run_composite_router.py`
- Scorer: `scripts/score_composite_results.py`
- CPU/fake-backend tests: `tests/test_composite_router.py`

## Verification completed

```text
36 items, 12 per family: valid
all 36 oracle workflows × 3 fixture repeats: final + all stages pass
malformed JSON: repair and fallback paths pass
dependency and cycle checks: pass
orchestrator HTTP endpoint with fake backends: pass
legacy leakage regression: 11/11 hard cases pass
live six-item balanced smoke: audited before full run
live 36×3: 756 system-item rows, all HTTP executions completed
```

The CPU fixture validates plumbing; the 36×3 artifacts above are the model
quality result.

## Live commands

```bash
# First inspect six traces.
python scripts/run_composite_router.py \
  --systems mamay4,router_direct,oracle,hybrid \
  --limit 6 --out-dir results/week_10_composite_smoke

python scripts/score_composite_results.py \
  results/week_10_composite_smoke/*.jsonl \
  --out results/week_10_composite_smoke/scores.json

# Only after the traces look correct:
python scripts/run_composite_router.py \
  --systems mamay4,lapa,aya,qwen7,router_direct,oracle,hybrid \
  --repeats 3 --out-dir results/week_10_composite

python scripts/score_composite_results.py \
  results/week_10_composite/*.jsonl \
  --out results/week_10_composite/scores.json
```

## Claim boundary

Do not call the three families “general complex reasoning.” They test bounded
workflow composition under deterministic rubrics. Generalisation to arbitrary
tasks remains future work.
