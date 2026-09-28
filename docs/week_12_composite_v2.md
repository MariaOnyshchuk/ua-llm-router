# Week 12 — composite v2 + constrained workflows

*Planning held-out completed 28 Sep 2026. Live specialist execution bake-off is
ready on the lab (`cluster/eval_composite_v2_bakeoff.sbatch`) but not yet run
from this laptop (SSH to `ucu-lab-2240` unavailable in-session).*

## Question

Can a larger, leakage-controlled composite suite and a constrained workflow
selector close the planning gap that free-form Hybrid v3 left on v1?

## Benchmark (`mixed_ua_composite_v2`)

| | |
|--|--|
| Path | `benchmarks/mixed_ua_composite_v2.jsonl` |
| Size | **200** = 5 families × (16 train + 12 dev + 12 test) |
| Builder | `python scripts/build_composite_benchmark.py --version v2` |
| Fingerprint | `7805aa17885b5f81` |

Families (depth 2–3, deterministic rubrics):

| Family | Workflow | n |
|--------|----------|--:|
| translate_code | translate → code | 40 |
| knowledge_explain | knowledge → instruct | 40 |
| translate_knowledge_write | translate → knowledge → instruct | 40 |
| extract_classify_write | knowledge → knowledge → instruct | 40 |
| translate_summarize_format | translate → instruct → instruct | 40 |

Leakage controls: disjoint `source_key` pools across splits; stable IDs;
`oracle_plan` never enters the user prompt. **Do not tune prompts/templates on
`test`.** v1 stays the historical 36-item suite — never put v1 and v2 scores in
one “did we improve?” cell.

## Systems

| System | Role |
|--------|------|
| `router_direct` | One-hop rules v2 |
| `oracle` | Stored workflow through specialists |
| `hybrid` / `hybrid_fewshot` / `hybrid_minimal` / `hybrid_mamay12` | Free-form JSON planner profiles |
| **`template`** | Constrained selector → fixed stage prompts ([`router/workflows.py`](../router/workflows.py)) |

Trace fields now include `split`, `selector_mode`, `selected_template`,
`fallback_reason`, `workflow_signature`, superfluous/missing step counts.

## Dev ablation (offline)

Command: `python scripts/ablate_composite_planner_dev.py --split dev`

| Profile | Exact workflow | Template-id acc | Fallback | n |
|---------|---------------:|----------------:|---------:|--:|
| **template** | **1.000** | **1.000** | 0.000 | 60 |
| hybrid (live) | — | — | — | not run (needs GPU) |

Context (different suite): Hybrid v3 on v1 had exact workflow **0.639**.

**Freeze:** `template`. Distillation **not** warranted while the bounded family
prompts stay in-distribution for the regex/template registry.

## Held-out test — planning

`python scripts/eval_composite_selector.py --split test`

| Split | Exact workflow | Template-id | Fallback | n |
|-------|---------------:|------------:|---------:|--:|
| test | **1.000** | **1.000** | 0.000 | 60 |

All five families match. Artifact:
`results/week_12_composite_v2_test_bakeoff/planning_heldout.json`,
`freeze_and_heldout.json`.

## Held-out test — execution

| Kind | Status | Artifact |
|------|--------|----------|
| Fixture specialists (plumbing) | Done — oracle & template final **1.0** on 60×3 | `results/week_12_composite_v2_test_bakeoff/fixture/scores.json` |
| Live specialists | **Pending GPU** | `sbatch cluster/eval_composite_v2_bakeoff.sbatch` |

Fixture scores validate rubrics and orchestration only. Cite live
`scores.json` from the sbatch for specialist quality. Systems in the live job:
`router_direct,oracle,hybrid,template`, T=0, seed=42, max_tokens=256, 3 repeats,
`--split test`.

## Distillation decision

**No.** Template already realises the oracle workflow signature on v2 train/dev/test.
A distilled planner would only help if natural-language requests leave the
template surface. Revisit after freer prompts or a failed live hybrid comparison.

## Commands

```bash
# Rebuild / check v2
python scripts/build_composite_benchmark.py --version v2
python scripts/build_composite_benchmark.py --version v2 --check

# Offline selector (dev then test)
python scripts/ablate_composite_planner_dev.py --split dev
python scripts/eval_composite_selector.py --split test \
  --out results/week_12_composite_v2_test_bakeoff/planning_heldout.json

# Live bake-off on the lab
sbatch cluster/eval_composite_v2_bakeoff.sbatch

# Optional live LLM planner ablation on DEV only
python scripts/ablate_composite_planner_dev.py --split dev --live \
  --profiles hybrid,hybrid_fewshot
```

## Claim boundary

Planning claims on v2 use the offline selector metrics above. Do not claim
specialist final-score improvements until the live test bake-off lands. Do not
compare v2 finals to v1 0.812 / 0.882 / 0.8125 in one cell.
