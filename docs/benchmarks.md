# Benchmark strategy

Keep `benchmarks/mixed_ua_v0.jsonl` (40 hand-written items) as the cheap smoke / routing suite.
Extend **selectively** with Ukrainian-native public benchmarks for statistical weight on code and knowledge.

## Two categories

| Category | Examples | Role for this thesis |
|----------|----------|----------------------|
| General routing benchmarks | RouterBench, RouterEval, MixInstruct, LLMRouterBench | **Cite only** — English methodology templates; not prompt sources |
| Ukrainian task-native | ZNO-Eval, UA-Code-Bench, (+ FLORES-200 for MT) | **Extend** `knowledge` / `code` (and optionally `translate`) |

## Bucket map

| Bucket | Source | Plan |
|--------|--------|------|
| `knowledge` | [ZNO-Eval](https://github.com/NLPForUA/ZNO) (arXiv:2501.06715) | Sample ~20–30 single-answer, no-photo items across subjects. Same family INSAIT used for MamayLM ZNO scores → sanity-check S2. |
| `code` | [UA-Code-Bench](https://huggingface.co/datasets/NLPForUA/ua-code-bench) (arXiv:2511.05040) | Sample ~20–30 from difficulty bands 1–2 only (3–4B specialists). **License/ops note:** statements are © Eolymp and often URL-referenced — do not republish; score via local tests or Eolymp judge where available. |
| `alignment` | [UAlign](https://huggingface.co/datasets/Stereotypes-in-LLMs/UAlign) | 24 balanced ETHICS + 24 balanced Social Chemistry 101 items |
| `chat` | Hand-written | Keep custom; UAlign is reported separately, not mixed into casual chat |
| `translate` | Hand-written + [FLORES-200](https://huggingface.co/datasets/facebook/flores) | 24 source pairs in both directions (48 items) |
| `instruct` | Hand-written | Keep custom — under-covered in UA lit; note in limitations |

Out of scope for now: UA-Legal-Bench (domain escalation story only if needed later).

## File layout

```
benchmarks/
  mixed_ua_v0.jsonl          # original 40 — do not replace
  mixed_ua_v1.jsonl          # v0 ∪ 24 sampled ZNO items
  mixed_ua_v2.jsonl          # v1 ∪ 48 FLORES ∪ 48 UAlign (160)
  mixed_ua_v3.jsonl          # v2 ∪ 22 hand-written code items (182; code=30) — unbalanced
  mixed_ua_v4_balanced.jsonl # **32 per bucket × 6 = 192** (preferred next suite)
  external_ua_v2.jsonl       # FLORES ∪ UAlign only (incremental run)
  samples/
    zno_knowledge_v1.jsonl
    flores_translate_v1.jsonl
    ualign_v1.jsonl
    code_extra_v1.jsonl / code_extra_v2.jsonl
    chat_extra_v1.jsonl / instruct_extra_v1.jsonl
    uacode_easy_v1.jsonl     # UA-Code-Bench bands 1–2 (imported; score deferred)
```

## Sampling scripts

```bash
# ZNO (needs git clone of NLPForUA/ZNO or path to tests/*.json)
python scripts/sample_zno.py --zno-dir /path/to/ZNO --n 24 --out benchmarks/samples/zno_knowledge_v1.jsonl

# UA-Code-Bench (HF; bands 1–2) — inventory only until Eolymp judge
python scripts/sample_uacode.py --n 32 --bands 1,2 --out benchmarks/samples/uacode_easy_v1.jsonl

# FLORES-200 (24 pairs × 2 directions)
python scripts/sample_flores.py --n-per-direction 24

# UAlign (24 items from each config)
python scripts/sample_ualign.py --n-per-config 24

# Balanced suite (32/bucket)
python scripts/build_balanced_suite.py --per-bucket 32 --out benchmarks/mixed_ua_v4_balanced.jsonl
```

## Scoring notes

- **ZNO:** exact match on letter (А/Б/В/Г/Д) after normalizing answer prefix.
- **FLORES:** chrF-like score against the official reference translation.
- **UAlign:** exact class accuracy (ETHICS: 0/1; Social Chemistry: 0/1/2).
- **UA-Code:** prefer Eolymp judge; local unit tests only for hand-written `code-*`.
- Always report suite version explicitly (`v3` vs `v4_balanced`) so scores stay comparable.

## Current execution status (2026-08-07)

- ZNO-Eval: sampled ✓
- FLORES-200: sampled ✓
- UAlign: sampled ✓
- UA-Code-Bench: **imported** (32 easy) — not in scored v4 until judge
- Hand-written chat/instruct/code expanded to fill **32/bucket**
- **`mixed_ua_v4_balanced.jsonl`**: 192 = 32×6

## Composite v1 (September phase)

`benchmarks/mixed_ua_composite_v1.jsonl` tests **dependent skills inside one
request**, rather than another mixture of single-skill rows:

| Family | n | Stored workflow | Final deterministic check |
|--------|--:|-----------------|---------------------------|
| `translate_code` | 12 | translate → code | Python function cases |
| `knowledge_explain` | 12 | knowledge → instruct | answer label, two-line format, required evidence |
| `translate_knowledge_write` | 12 | translate → knowledge → instruct | exact JSON fields and keys |

The top-level `prompt` is visible to every system. `oracle_plan.steps[]` contains
the hidden workflow, prompt templates, dependencies, and rubrics. The hybrid
planner sees only the top-level prompt.

Build and validate:

```bash
python scripts/build_composite_benchmark.py
python scripts/build_composite_benchmark.py --check
```

Run a six-item smoke and then the deterministic evaluation:

```bash
python scripts/run_composite_router.py \
  --systems mamay4,lapa,aya,router_direct,oracle,hybrid \
  --limit 6 --out-dir results/week_10_composite_smoke

python scripts/run_composite_router.py \
  --systems mamay4,lapa,aya,qwen7,router_direct,oracle,hybrid \
  --repeats 3 --out-dir results/week_10_composite

python scripts/score_composite_results.py \
  results/week_10_composite/*.jsonl \
  --out results/week_10_composite/scores.json
```

This suite is deliberately small and controlled. Its handcrafted prompts are
inspired by HumanEval-, ZNO-, and FLORES-style tasks but must not be described
as new samples from those public datasets.
