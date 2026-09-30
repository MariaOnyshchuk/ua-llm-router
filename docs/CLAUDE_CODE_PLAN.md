# Plan for Claude Code: test the router, improve it, build a debug UI

Read `AGENTS.md` first (evidence rules, scoring, bookkeeping). This file says what to do and in which order. Do the phases in order. After each phase, stop and report in a table what was done and what the user must run. Do not start a phase if the previous one has open items.

## Goal

Find out how good the router (Mamay-4B, Lapa-12B, Aya-8B behind rules) is compared with one large model (Mamay-12B, Mamay-27B), find exactly where and why it is worse, and improve it. Keep everything reproducible and small enough for a diploma.

## Ground rules

- Use only existing benchmarks. Do not create synthetic benchmarks. Single skills: `benchmarks/mixed_ua_v6.jsonl` (2414). Composite tasks: AgentCoMa-UK, `benchmarks/agentcoma_uk_200_*.jsonl` (50 dev + 150 test, local only, never commit or publish prompts). `mixed_ua_v4_balanced` (192) is for wiring checks only.
- One run per experiment is fine (T=0, seed=42). Every claimed difference has a paired bootstrap 95% interval (`scripts/analysis/bootstrap_compare.py`). If the interval contains 0, write "not distinguishable".
- Tune on dev only. Freeze the routing table, prompts and few-shot examples, write the commit hash in the ledger, then run test once.
- Metrics always go in tables. Plain direct language. Answer the user in Ukrainian or English, never Russian.
- Do not write raw generation rows to git or HF. Scores and item ids only.
- Any scorer change means rescoring every cited run. Say so and list the commands.
- Work in the repo. Do not leave one-off scripts in `scripts/` root; put them in the group folder described in `scripts/README.md`.

## Phase 0: clean base (before any new run)

1. Confirm the scorer fix in `scripts/score_results.py` is committed and tests pass (`pytest -q`).
2. List which cited runs still have old scores (week_6, week_7, week_9, week_11, v6 solos) and give the user the rescoring commands. Do not proceed to comparisons until they are rescored.
3. Rebuild `results/progress_ledger.csv`; check that the FP8 pack run has its own label.

Done when: one table of run, scorer version, rescored yes/no.

## Phase 1: baseline matrix

Systems: mamay4, lapa, aya, mamay12, mamay27 (FP8), router rules v2. On AgentCoMa also `oracle` (gold plan) and the frozen template selector.

| Benchmark | Split | Systems | Runner |
|---|---|---|---|
| v6 | all 2414 | all six | `cluster/eval_v6_solo.sbatch`, `cluster/eval_v6_router.sbatch` |
| AgentCoMa-UK | dev 50 first, test 150 once | all six plus oracle and template | `cluster/eval_agentcoma_200.sbatch` (add the solo systems; they are missing) |

Also run the AgentCoMa step questions alone (`question_commonsense_uk`, `question_math_uk` in `agentcoma_uk_200_merged.jsonl`) for every model, so the compositional gap can be computed: share of tasks where both steps are right alone minus share of composite answers that are right.

Done when: one table per benchmark with score, 95% interval, p50 and p95 latency, GPU-seconds per prompt, for every system, and the reference row (largest single model) marked.

## Phase 2: why is the router worse than the big model

The final score is not enough. Produce these tables from the existing per-item files, no new model runs unless a table cannot be built:

| Analysis | Question it answers | Source |
|---|---|---|
| Per-bucket gap to the big model, with intervals | where does the router lose | v6 scored details |
| Routing accuracy: share of items sent to the best solo model per bucket, and the score if every item went to its best model (route oracle) | is the loss from routing or from the specialists | router route field + solo scores |
| Parse and format failures vs wrong answers | is the loss real quality or scorer/format | `method` fields in scorer details |
| Latency and GPU cost per point of score | is the router worth it | ledger, `plot_tradeoff.py` |
| Composite: plan valid, plan match, stage scores, final score, by category and operation | where does a chain break | composite scorer details |
| Composite: compositional gap per model | is the problem composition or single steps | Phase 1 step runs |
| Composite error attribution: wrong plan, wrong specialist output at a step, wrong final compute | which component to fix | stage_scores |

Done when: a short ranked list of the three biggest causes of loss, each with a number and interval.

## Phase 3: prompt and route experiments (dev only)

For each experiment: one hypothesis, one change, one dev run, paired bootstrap against baseline, one ledger row. Start with the largest cause from Phase 2. Candidates already prepared:

- per-bucket few-shot (`--bucket-fewshot`, check no overlap with `bucket_prompt_variants.py --check`)
- alignment prompt variants
- route table changes evaluated first by `scripts/analysis/estimate_route_table.py`, then run (for example Mamay-12B on instruct and alignment)
- planner or template prompts for AgentCoMa

Stop rule: two experiments in a row with intervals containing 0 on the same cause means change the cause, not the prompt.

Done when: the winning config is frozen, its commit hash is in the ledger, and test is run once for it and for the baseline.

## Phase 4: debug UI

Requirements only; choose the smallest stack that works (the repo already has `router/` and `litellm/`; check what streaming support exists before adding dependencies). It must run locally and reach the cluster models through the SSH tunnel or through mock mode.

| Need | Detail |
|---|---|
| Chat with streaming | tokens appear as they are generated |
| Routing panel | detected intent, chosen model, rule that fired, alternatives |
| Composite view | plan steps, each step's prompt, model, output, latency, dependencies |
| Compare mode | same prompt to the router and to one chosen single model, side by side |
| Benchmark browser | load an item from v6 or AgentCoMa, run it, show gold answer and score next to the output |
| Raw view | exact prompt sent, sampling settings, finish reason, token counts |
| Session log | save runs as local JSONL, never to git |

Done when: the user can pick a failing v6 item from the Phase 2 lists, run it through the router and the big model, and see where it went wrong in one screen.

## Phase 5: write-up support

Update `docs/` (thesis scope, conclusion, key takeaways), the HF dataset card and the tracker from the frozen tables only. Fix the corpus README count (24,839). Keep a table of claims, each with its evidence file and interval.

## Report format after each phase

| Item | Status | Evidence file | User action needed |
|---|---|---|---|
