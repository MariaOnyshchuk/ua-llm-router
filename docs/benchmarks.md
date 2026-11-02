# Benchmark strategy

Keep `benchmarks/archive/mixed_ua_v0.jsonl` (40 hand-written items) as the cheap smoke / routing suite.
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
  tasks_mixed_192.jsonl # **32 per bucket × 6 = 192** (historical claims)
  mixed_ua_v6_screen.jsonl   # warehouse screening draw
  mixed_ua_v6.jsonl          # withdrawn 12 Oct 2026: selection was inverted (docs/mixed_ua_v6.md)
  tasks_public_4401.jsonl          # claim suite: whole pool, dev 1487 / test 2750
  mixed_ua_composite_v1.jsonl  # 36 dependent workflows (historical)
  mixed_ua_composite_v2.jsonl  # 200 items, 5 families, train/dev/test
  agentcoma_uk_50*.jsonl       # AgentCoMa-UK 50 (source dev split) — local only, gitignored
  graphs_agentcoma_150_test_source.jsonl  # +150 from source test split — local only
  graphs_agentcoma_200_source.jsonl / graphs_agentcoma_200.jsonl  # 50 dev + 150 test — local only
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
python scripts/benchmarks/sample_zno.py --zno-dir /path/to/ZNO --n 24 --out benchmarks/samples/zno_knowledge_v1.jsonl

# UA-Code-Bench (HF; bands 1–2) — inventory only until Eolymp judge
python scripts/benchmarks/sample_uacode.py --n 32 --bands 1,2 --out benchmarks/samples/uacode_easy_v1.jsonl

# FLORES-200 (24 pairs × 2 directions)
python scripts/benchmarks/sample_flores.py --n-per-direction 24

# UAlign (24 items from each config)
python scripts/benchmarks/sample_ualign.py --n-per-config 24

# Balanced suite (32/bucket)
python scripts/benchmarks/build_balanced_suite.py --per-bucket 32 --out benchmarks/tasks_mixed_192.jsonl
```

## Scoring notes

- **ZNO / Belebele / MMLU-UA / ARC-UA:** letter match (A–E / А–Д).
- **FLORES / WMT-22:** chrF-like score against the official reference translation.
- **UAlign:** exact class accuracy (ETHICS: 0/1; Social Chemistry: 0/1/2).
- **IFEval-UA:** fraction of instruction constraints satisfied (`scripts/ifeval_check.py`).
- **UA-Code:** prefer Eolymp judge; local unit tests only for hand-written `code-*`.
- Always report suite version explicitly (`v3` vs `v4_balanced`) so scores stay comparable.

## Current execution status (2026-08-07)

- ZNO-Eval: sampled ✓
- FLORES-200: sampled ✓
- UAlign: sampled ✓
- UA-Code-Bench: **imported** (32 easy) — not in scored v4 until judge
- Hand-written chat/instruct/code expanded to fill **32/bucket**
- **`tasks_mixed_192.jsonl`**: 192 = 32×6

## Composite v1 (September phase)

`benchmarks/archive/mixed_ua_composite_v1.jsonl` tests **dependent skills inside one
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
python scripts/benchmarks/build_composite_benchmark.py
python scripts/benchmarks/build_composite_benchmark.py --check
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

## Composite graphs_mixed_100_dev (28 Oct 2026)

`benchmarks/graphs_mixed_100_dev.jsonl`, built by `scripts/benchmarks/build_graphs_mixed_100_dev.py` in the
graph schema of the vault notes (classes, merge rule, value-dependency edges, gold for every node). Replaces
probe-100 for composite work: probe-100 had only linear chains from the five composite v2 templates, and the
template selector and the gold plan scored the same on 86 of its 100 items. Second build of the day, after
review: fewer simple items, more task types, gold kinds.

| Class | n | Shapes |
|---|---:|---|
| simple | 10 | one per task type: time, ZNO, search in notices, summary, classification, translate, code, instruct, math, alignment |
| chain | 30 | 18 with two nodes, 12 with three |
| graph | 30 | 13 joins, 10 forks, 7 four-node fork-then-join; 3 one-skill math graphs are shape controls |
| parallel | 30 | 20 with two nodes, 10 with three; the gold plan ends with a step that only joins the parts |

- Task types across items: time (17: durations with seconds, schedules, time zones, an invalid date, seconds to
  days/hours, ordering times in mixed formats), ZNO (12 items: Ukrainian, history, geography), search in notices or
  logs given in the request, summaries (Belebele passages), classification of items, word problems, extraction,
  translation, code, alignment.
- Sources: 19 items use GSM8K or ToT Arithmetic problems from STaD (ibm-research/STaD, Apache-2.0), translated with
  the numbers kept; every STaD gold was checked against the item. ZNO items and Belebele passages are read by id
  from `benchmarks/corpus/`, none of them in `tasks_public_4401`.
- Gold kinds per node: exact 108, accepted 81, example 61. For `example` (open writing) the gold is only an
  example; the check covers form and anchors, and `criteria` are written for the LLM judge (not calibrated yet).
- Split: all 100 items are a development set (decided 28 Oct 2026). Claims will come from a separate composite
  test set, built later with the same prompts and frozen before its first run.
- Build-time checks: every node gold passes its own rubric, every final gold passes the final check, code golds pass
  their tests, math is computed by Python, shapes match the class, prompts reference exactly their parents, and
  all 100 gold plans pass `validate_plan`. New rubric types in `score_composite_results.py`: `has_number`,
  `sentence_count`, `regex`, `any`.
- Not done: review by a person, the G2–G4 teacher checks, judge calibration. Math (and time) nodes
  run as `chat` (Aya) in the gold plan; who should get math is still open.
- Viewer: `python scripts/benchmarks/render_composite_viewer.py benchmarks/graphs_mixed_100_dev.jsonl`.

### First run (29 Oct 2026, JUPITER job 2574111, dev only)

JSC JUPITER, one GH200 node, Mamay-4B, Lapa and Aya in bf16 on one card each, vLLM 0.25.1, T=0, seed=42,
max_tokens=1024, one pass, all 100 items, HTTP 200 on all 500 rows. Dataset sha256 prefix 3f564ca75c3bd00c.
Scored by the automatic rubrics (no LLM judge yet). Numbers from JUPITER are compared only with JUPITER runs.

| System | All (100) | simple (10) | chain (30) | graph (30) | parallel (30) | Calls | p50, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| One call (`router_direct`) | 0.625 | 0.750 | 0.461 | 0.639 | 0.733 | 1.00 | 3.8 |
| Gated planner, Mamay-4B | 0.627 | 0.750 | 0.478 | 0.661 | 0.700 | 1.61 | 6.4 |
| Gated planner, Lapa | 0.632 | 0.750 | 0.522 | 0.633 | 0.700 | 1.61 | 6.5 |
| Gated planner, Aya | 0.647 | 0.750 | 0.561 | 0.628 | 0.717 | 1.53 | 6.1 |
| Gold graph (`oracle`) | 0.712 | 0.750 | 0.683 | 0.736 | 0.703 | 2.97 | 7.4 |

Paired bootstrap, 95%:
- Gold graph − one call: +0.087 [+0.020, +0.153]; on chain +0.222 [+0.094, +0.361]; on graph +0.097
  [−0.039, +0.231] and on parallel −0.031 [−0.131, +0.061], not distinguishable.
- Every gated planner − one call: not distinguishable (Mamay-4B +0.002, Lapa +0.007, Aya +0.022). Planner Aya on
  chain: +0.100 [+0.006, +0.211].
- Gold graph − planner Aya: +0.065 [−0.007, +0.138], not distinguishable.
- Node scores under the gold graph by skill: translate 0.907, code 0.852, knowledge 0.833, math as chat 0.779,
  instruct 0.706, alignment 0.529.

Reading: decomposition pays off on chains. The planners split far fewer items than the gold graph (about 1.6
calls against 3.0), so most of the gap to the gold graph is items the gate sends as one call.

Rule gates removed (2 Nov 2026). In these runs the gated profiles had three regex rules in front of the planner
LLM: a one-symbol answer, a numeric word problem (which also got a reasoning suffix) and "no sequence word or
two joined task verbs" each sent the request to one call without asking the planner. On graphs_mixed_100_dev they
decided 75 of the 100 requests (65 one action, 10 word problems), so the planner was asked on 25 only. Plan
repair ran on 2 requests with the Mamay-4B planner (both failed and fell back to one call) and on 0–1 with Aya.
The rules are gone from `router/planner.py`: the planner LLM now gets every request and decides one task or
several itself; an invalid plan after one repair still falls back to one call. The numbers above are with the
rules (code at commit 961f476) and are not comparable with runs after the change.

### Learned routers and single models (30 Oct 2026, JUPITER job 2596743, dev only)

Same node, settings and dataset as job 2574111; HTTP 200 and a non-empty answer on all 900 rows. All 14
systems were rescored together. kNN and MLP are RouterBench-style predictive routers
(`scripts/train_routerbench_router.py`): e5-large embedding of the prompt, a regressor predicts the score of
Mamay-4B, Lapa and Aya, the highest prediction wins. Trained on all of `tasks_public_4401` dev (1423 items, no math,
no code, no composite items), so every graphs_mixed_100_dev prompt is outside their training distribution. kNN k=40,
MLP one hidden layer of 128, alpha 0.1, both chosen by 5-fold CV. On plan steps they route by the step prompt
and ignore the step's intent label. `+knn` / `+mlp` means the same plan with learned instead of rules v2 step
routing.

| System | All (100) | simple (10) | chain (30) | graph (30) | parallel (30) | Calls |
|---|---:|---:|---:|---:|---:|---:|
| One call, rules v2 (`router_direct`) | 0.625 | 0.750 | 0.461 | 0.639 | 0.733 | 1.00 |
| One call, kNN (`router_knn`) | 0.676 | 0.850 | 0.622 | 0.692 | 0.656 | 1.00 |
| One call, MLP (`router_mlp`) | 0.673 | 0.850 | 0.644 | 0.639 | 0.678 | 1.00 |
| Mamay-4B only | 0.635 | 0.750 | 0.589 | 0.589 | 0.689 | 1.00 |
| Lapa only | 0.643 | 0.725 | 0.594 | 0.658 | 0.650 | 1.00 |
| Aya only | 0.628 | 0.750 | 0.472 | 0.594 | 0.778 | 1.00 |
| Planner Aya, rules v2 steps | 0.647 | 0.750 | 0.561 | 0.628 | 0.717 | 1.53 |
| Planner Aya, kNN steps | 0.607 | 0.850 | 0.600 | 0.511 | 0.628 | 1.53 |
| Planner Aya, MLP steps | 0.650 | 0.850 | 0.622 | 0.606 | 0.656 | 1.54 |
| Gold graph, rules v2 steps | 0.712 | 0.750 | 0.683 | 0.736 | 0.703 | 2.97 |
| Gold graph, kNN steps | 0.649 | 0.850 | 0.617 | 0.675 | 0.589 | 2.97 |
| Gold graph, MLP steps | 0.657 | 0.850 | 0.528 | 0.672 | 0.706 | 2.97 |

Model choice on the 100 one-call requests: rules v2 Aya 63 / Lapa 20 / Mamay-4B 17; kNN Mamay-4B 50 / Lapa 33 /
Aya 17; MLP Mamay-4B 66 / Aya 20 / Lapa 14.

Paired bootstrap, 95% (10 000 resamples, seed 42):
- One call, kNN − rules v2: +0.051 [−0.021, +0.123]; MLP − rules v2: +0.048 [−0.010, +0.108]. Not
  distinguishable overall. On chain: kNN +0.161 [+0.006, +0.317], MLP +0.183 [+0.056, +0.317].
- kNN − MLP: +0.003 [−0.042, +0.046], not distinguishable.
- Rules v2 against each single model: not distinguishable (Mamay-4B +0.010, Lapa +0.018, Aya +0.003 over rules).
  kNN against the best single model (Lapa): +0.033 [−0.031, +0.094]. MLP − Mamay-4B: +0.038 [+0.000, +0.082],
  at the edge.
- Gold graph, kNN steps − rules v2 steps: −0.062 [−0.127, +0.000]; MLP steps: −0.055 [−0.125, +0.012]. Not
  distinguishable overall; MLP on chain −0.156 [−0.311, −0.006].
- Planner Aya, kNN steps − rules v2 steps: −0.040 [−0.095, +0.015]; worse on graph −0.117 [−0.228, −0.011] and
  parallel −0.089 [−0.189, −0.011]. MLP steps: +0.003, not distinguishable.
- Gold graph with rules v2 steps − one call with kNN: +0.036 [−0.039, +0.109], not distinguishable.
- The +0.100 on simple is one item out of ten.

Node scores under the gold graph by skill (rules v2 / kNN / MLP): alignment 0.529 / 0.765 / 0.765 (17 nodes;
rules send these to Lapa, both learned routers to Aya), math as chat 0.779 / 0.602 / 0.607 (61; rules Aya, learned
mostly Mamay-4B or Lapa), instruct 0.706 / 0.642 / 0.658 (110), knowledge 0.833 / 0.792 / 0.758 (60), translate
0.907 / 0.861 / 0.937 (33), code 0.852 / 0.852 / 0.840 (16).

Reading:
- On a whole composite request, rules v2 does no better than any one model, because it sends 63 of 100 requests
  to Aya, the weakest model on chains. The learned routers send most requests to Mamay-4B or Lapa and gain on
  chains. Overall the gain is not distinguishable.
- On single-skill plan steps the rules with the intent label do better than the learned routers. The learned
  routers lose most on math steps (v7 has no math, so they have no signal there) and on instruct steps. They
  win on alignment steps, where Aya beats the Lapa choice in the rules table.
- So learned routing helps where the rules table has no fitting row (a mixed request) and hurts where it has one
  (a labelled single-skill step). These are dev numbers on 100 items; nothing here was tuned on graphs_mixed_100_test.

Cost–quality view: `docs/figures/routerbench_panels_graphs_mixed_100_dev.png` (`scripts/analysis/plot_routerbench_panels.py`).
It sweeps the kNN and MLP trade-off lambda * P_j − c_j over the one-call answers of the three models (c_j = mean
latency on v7 dev). AIQ over the single-model latency range: Zero router 0.636, kNN 0.668, MLP 0.675. At lambda =
inf the simulation picks the same model as the live runs on all 100 requests. Scores differ on 3 requests for
kNN and 1 for MLP, because vLLM at T=0 is not bit-exact across runs.

## Dataset file names (1 Nov 2026)

Files are named `<unit>_<origin>_<number of tasks>[_<split>].jsonl`. Unit: `tasks` (one request, one skill), `graphs` (a request made of several nodes) or `nodes` (one step). Origin: `synthetic` (written by us), `public` (from existing datasets) or `mixed`. Each file starts with a `# why:` line kept in `scripts/benchmarks/dataset_headers.py`. Item ids and contents did not change, so every fingerprint is the same (graphs_mixed_100_test stays frozen at `e60df145df92437c`). Result files written before this date keep the old names in their file names and `*.meta.json`.

| Old name | New name | Why |
|---|---|---|
| `mixed_ua_composite_graph100` (later `mixed_ua_compose_dev`) | `graphs_mixed_100_dev` | 100 graph tasks (250 nodes), all dev; written by us + STaD, ZNO, Belebele |
| `mixed_ua_composite_test100` (later `mixed_ua_compose_test`) | `graphs_mixed_100_test` | same build, held out, frozen |
| `mixed_ua_nodes_train` | `nodes_mixed_600_train` | 545 written + 40 ZNO + 15 UAlign; training labels for the learned routers |
| — | `nodes_mixed_600_test` | planned: same composition, held out |
| `mixed_ua_v7` | `tasks_public_4401` | claim suite, dev 1542 / test 2859; all public except 32 chat items |
| `mixed_ua_v4_balanced` | `tasks_mixed_192` | development suite for wiring checks; hand-written + ZNO, FLORES, UAlign |
| `intent_natural_v1` | `intent_requests_72` | diagnostic for intent detection |
| `agentcoma_uk_200_composite`, `agentcoma_uk_50_composite` | `graphs_agentcoma_200`, `graphs_agentcoma_50_dev` | AgentCoMa-UK in composite form; gated, local only |
| `agentcoma_uk_200_merged`, `agentcoma_uk_50items_merged`, `agentcoma_uk_150_test_translated` | `graphs_agentcoma_200_source`, `graphs_agentcoma_50_dev_source`, `graphs_agentcoma_150_test_source` | translated source items; gated, local only |
| `mixed_ua_v0`–`v3`, `v5`, `v6`, `v6_screen`, `external_ua_v2`, `mixed_ua_composite_v1`, `v2`, `probe100` | same names in `benchmarks/archive/` | historical or withdrawn; names kept so old results still point at them |

Builders renamed with them: `build_graphs_mixed_100_dev.py`, `build_graphs_mixed_100_test.py`, `build_nodes_mixed_600.py`, `build_tasks_public_4401.py`. The `pool_sha1` in `tasks_public_4401_meta.json` leaves out the `# why:` line of the pool, so it matches the value recorded on 12 Oct.

## Node training set for the learned routers (31 Oct 2026)

`benchmarks/nodes_mixed_600_train.jsonl`, built by `scripts/benchmarks/build_nodes_mixed_600.py` from item
definitions in `scripts/benchmarks/nodes_train/`. 600 single-skill items written in the form of graphs_mixed_100_dev step
prompts: 100 each for math, translate, instruct, knowledge, alignment and code. Only for training and
cross-validating the kNN and MLP routers; graphs_mixed_100_dev stays the dev evaluation and graphs_mixed_100_test the test.

- Why: the routers trained on `tasks_public_4401` dev had seen only four item types. 116 of the 250 graphs_mixed_100_dev step
  prompts are less similar to v7 dev than 95% of v7 dev items are to each other (e5-large cosine to the nearest
  item, threshold 0.865). With this set added, 23 are. By skill: math 44 → 5, knowledge 42 → 12, instruct 23 → 6,
  code 5 → 0.
- 218 items (36%) already contain a value an earlier step would produce (a chosen list, a number, an action to
  judge, a translated sentence). v7 has none of these.
- Sources: drafted by Claude following `scripts/benchmarks/prompts/node_train_generation.md`; 40 real ZNO items
  (Ukrainian language, literature, history) and 15 real UAlign social items drawn at random (seed 42) from
  `benchmarks/corpus/`. No MMLU, ARC or STaD.
- Each item stores `skill`, `label` (the intent the planner gives such a step; math → chat), `subtype`, `form`,
  `gold`, a `wrong` answer and a `check` in the rubric types of `scripts/score_composite_results.py`.
- Build checks: every gold scores 1.0 on its check and every wrong answer below 1.0; math golds are computed
  from the arithmetic; code golds pass their unit tests; no 10-word overlap with v7, graphs_mixed_100_dev or graphs_mixed_100_test after
  removing shared instruction wording (the alignment scale, answer formats); no duplicate prompts.
- Items are never selected using model scores. Fingerprint `4fffe04dc24736e6` (four code items that returned
  tuples were changed to return lists on 31 Oct, because JSONL turns the expected tuples into lists and no answer
  could pass). A teacher check (N2) and a person's review of the full set are not done yet.
- Labels: Mamay-4B, Lapa and Aya each answered all 600 items once on JUPITER (job 2611271, T=0, seed 42,
  max_tokens 1024), scored in `results/nodes_mixed_600_train/`: Mamay-4B 0.778, Lapa 0.785, Aya 0.793, best per
  item 0.935; all three tie on 335 items. These are training labels, not results.

### Test split: nodes_mixed_600_test (1 Nov 2026)

`benchmarks/nodes_mixed_600_test.jsonl`, `build_nodes_mixed_600.py --split test`, items in
`scripts/benchmarks/nodes_test/`. Held out for one comparison of the learned routers with rules v2, run once after
the routers are frozen. Fingerprint `2dd49822418791a0`.

- Same composition as train: 100 per skill, the same subtype × form counts, 40 ZNO and 15 UAlign social items
  drawn at random (seed 42) from items not used in train, `tasks_public_4401` or the graph sets. Written alignment
  ratings: 28 × 0, 28 × 1, 29 × 2.
- Written fresh and worded differently from train: inside a skill no two written items share their first three
  words, and no four-word phrase appears in more than three of them (the 0/1/2 scale and answer markers excepted).
  Train repeats openers such as «Переклади англійською:» and «Напиши Python-функцію»; the test does not, so wording
  alone does not route an item, for the rules or for kNN.
- Extra build checks: no 10-word overlap with or copy of a train prompt; no written item closer than 0.95 (e5
  cosine) to a train item of its skill. While writing, 22 drafts were replaced because they repeated a train task
  in new words (for example the same problem with the same numbers, or `max_value` as `biggest`).
- Decision rule fixed before the test: switch to the learned router if the lower end of the 95% paired bootstrap
  interval of (learned − rules v2) is above −0.02. Rules v2 is scored twice: from the prompt text (its regexes)
  and with the item's intent label, as on plan steps.

#### Result (1 Nov 2026, JUPITER job 2623903, run once)

Mamay-4B, Lapa and Aya each answered all 600 items once (T=0, seed 42, max_tokens 1024; 1800 of 1800 complete,
none empty). Routers were scored on these stored answers by `scripts/analysis/node_test_routers.py`, frozen with the
routers and rules v2 at commit 4548da0; the script was committed before the first score (bf97061, f6672fa,
02c3741). Output: `results/nodes_mixed_600_test/router_comparison.json`.

| System | All | math | translate | instruct | knowledge | alignment | code |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mamay-4B | 0.773 | 0.740 | 0.927 | 0.892 | 0.605 | 0.610 | 0.865 |
| Lapa | 0.745 | 0.830 | 0.569 | 0.809 | 0.684 | 0.780 | 0.800 |
| Aya | 0.781 | 0.710 | 0.924 | 0.821 | 0.693 | 0.745 | 0.796 |
| rules v2, from text | 0.793 | 0.780 | 0.901 | 0.810 | 0.693 | 0.780 | 0.791 |
| rules v2, with intent label | 0.809 | 0.710 | 0.924 | 0.892 | 0.684 | 0.780 | 0.865 |
| kNN (v7 dev) | 0.804 | 0.790 | 0.920 | 0.852 | 0.664 | 0.790 | 0.810 |
| MLP (v7 dev) | 0.789 | 0.750 | 0.913 | 0.854 | 0.625 | 0.740 | 0.855 |
| kNN + nodes | **0.834** | 0.810 | 0.939 | 0.925 | 0.697 | 0.770 | 0.865 |
| MLP + nodes | 0.814 | 0.810 | 0.925 | 0.890 | 0.674 | 0.780 | 0.806 |
| oracle (best per item) | 0.933 | 0.930 | 0.995 | 0.978 | 0.875 | 0.880 | 0.940 |

| Difference (paired bootstrap, 10 000, seed 42) | vs rules v2 from text | vs rules v2 with intent label | Passes the rule |
|---|---|---|---|
| kNN (v7 dev) | +0.012 [−0.015, +0.039] | −0.005 [−0.031, +0.021] | no |
| MLP (v7 dev) | −0.003 [−0.030, +0.024] | −0.020 [−0.046, +0.007] | no |
| kNN + nodes | **+0.042 [+0.014, +0.069]** | **+0.025 [+0.003, +0.049]** | yes |
| MLP + nodes | +0.021 [−0.004, +0.048] | +0.005 [−0.019, +0.029] | yes |

- By the rule fixed before the run (lower bound above −0.02 against both rules variants), kNN + nodes and MLP +
  nodes pass; the routers trained on v7 dev alone do not. kNN + nodes is also better than both rules variants
  (both intervals above 0); MLP + nodes is not distinguishable from them.
- Undecided items (out of 600). Rules v2 from text matches no rule on 319 (53%) and sends them to Aya: math 74,
  translate 59, instruct 80, knowledge 45, alignment 0, code 61. On those items Aya scores 0.803, Mamay-4B would
  score 0.841. With the label, only the 100 math items reach the default, by design. The learned routers never
  fall back (0 hard fallbacks). kNN has exact ties, where the choice comes from list order (Mamay-4B): 25 (v7) and
  40 (+ nodes); MLP has none. Top two predictions less than 0.02 apart: kNN 158, kNN + nodes 129, MLP 144,
  MLP + nodes 96.
- Caveats. This test is single steps written by the same author and prompt as the training nodes, so it is in
  distribution for the +nodes routers; rules v2 was tuned on `tasks_mixed_192`. On `graphs_mixed_100_dev` plan
  steps (live, 30–31 Oct) kNN + nodes was worse than the rules (oracle plan: −0.060 [−0.120, −0.001]) while MLP +
  nodes was not distinguishable (−0.014 [−0.076, +0.049]). The two sets disagree about kNN; they agree that
  MLP + nodes is at least as good as the rules. `graphs_mixed_100_test` (frozen, not run) is the end-to-end check.

#### Follow-up: routers trained on the node set alone (2 Nov 2026, not part of the frozen comparison)

`scripts/train_routerbench_router.py --only-extra` trains kNN and MLP on the 600 train nodes without v7 dev;
settings chosen by 5-fold CV on the train nodes (kNN k=10, CV MSE 0.1178; MLP one layer of 128, alpha 0.001,
0.1234). Scored on the same stored test answers by `scripts/analysis/node_test_nodes_only.py`
(`results/nodes_mixed_600_test/router_comparison_nodes_only.json`). This run comes after the node test, so it
is reported as a follow-up and not used for the switch decision.

| Router | Trained on | Test score | − rules v2 text | − rules v2 intent | − same kind with v7 dev + nodes |
|---|---|---:|---|---|---|
| kNN nodes only | 600 nodes | 0.830 | +0.038 [+0.012, +0.063] | +0.021 [−0.002, +0.045] | −0.004 [−0.022, +0.014] |
| MLP nodes only | 600 nodes | 0.823 | +0.030 [+0.004, +0.056] | +0.013 [−0.012, +0.038] | +0.009 [−0.013, +0.030] |

- Adding v7 dev to the nodes does not change the node-test score (both differences contain 0). On this test the
  1423 v7 dev items add nothing; whether they help on planner-written steps is not measured here.
- The setting matters little. Refit on all train nodes, the test score is 0.830–0.836 for k = 5…80 and
  0.811–0.827 over the four MLP settings (for information only, not used to choose).
- Choice of k by 5-fold CV MSE on each router's own training set (from `router/artifacts/*/config.json`):

  | kNN trained on | k=5 | k=10 | k=20 | k=40 | k=80 | chosen |
  |---|---:|---:|---:|---:|---:|---:|
  | v7 dev (1423) | 0.1876 | 0.1685 | 0.1640 | 0.1631 | 0.1654 | 40 |
  | v7 dev + nodes (2023) | 0.1710 | 0.1550 | 0.1528 | 0.1531 | 0.1562 | 20 |
  | nodes only (600) | 0.1235 | 0.1178 | 0.1217 | 0.1276 | 0.1335 | 10 |

  Next to the best k the error changes by less than 0.005; the MSE values are not comparable across rows
  (different training sets). k does not change the routing result in any way these tests can show.
- Ties. kNN with k=10 averages ten 0/1 scores, so predictions come in steps of 0.1: 124 of 600 items (21%) are
  exact ties decided by list order. MLP nodes only has 53 exact ties, because its predictions are clipped at 1.0
  before the argmax.

## Composite graphs_mixed_100_test (28 Oct 2026)

`benchmarks/graphs_mixed_100_test.jsonl`, built by `scripts/benchmarks/build_graphs_mixed_100_test.py` with the
same prompts, helpers and build-time checks as graphs_mixed_100_dev. Held-out test set for composite claims.

| Class | n | Shapes |
|---|---:|---|
| simple | 10 | one per task type |
| chain | 30 | 20 with two nodes, 10 with three |
| graph | 30 | 13 joins, 10 forks, 7 four-node fork-then-join; 3 one-skill math graphs are shape controls |
| parallel | 30 | 18 with two nodes, 12 with three |

- No overlap with graphs_mixed_100_dev or `tasks_public_4401` (requests, STaD problems, ZNO items, Belebele passages), checked at
  build time. 34 STaD problems (16 GSM8K, 18 ToT Arithmetic; every gold checked against STaD), 20 ZNO items,
  6 Belebele passages, the rest new.
- Gold kinds per node: exact 113, accepted 76, example 60.
- Written on 28 Oct 2026 before any graphs_mixed_100_dev result was seen. First build fingerprint `e60df145df92437c`.
- Freeze rule: frozen at the commit that adds it; run once, after the router and planner are frozen; never
  edited after a run. Not reviewed by a person yet.

## AgentCoMa-UK 200 (compositional commonsense + arithmetic)

`graphs_agentcoma_200_source.jsonl` is a Ukrainian translation of 200 items from
[AgentCoMa](https://huggingface.co/datasets/LisaAlaz/AgentCoMa) (gated; no
reusable dataset license, so **every `agentcoma_uk_*` file stays local and is
gitignored**; publish only prompt-free metrics). Each item chains one
commonsense step (which objects, places, or people qualify) with one arithmetic
step over the numbers in the text.

| Split | n | Source split | Translation | Ids |
|-------|--:|--------------|-------------|-----|
| `dev` | 50 | AgentCoMa dev | `claude_manual_v1` (earlier) | `eval_*` |
| `test` | 150 | AgentCoMa test | `claude_manual_v2` (28 Sep) | `test_*` |

Per category (`house_working`, `web_shopping`, `science_experiments`,
`smart_assistant`, `travel_agent`): 40 items = 10 dev + 30 test; operations
addition / subtraction / multiplication / division split 53 / 52 / 48 / 47
overall (7–8 per category × operation among the test items). The 150 test items
are the lowest-numbered ids per category × operation, so the selection is
deterministic and disjoint from the existing 50. All rows have
`needs_human_review: true`. Numbers, units, and currency symbols are kept
exactly as in the source; brand names, Latin species names, and the source's
own inconsistencies (e.g. `test_TA_add_3` quotes £ in one field and $ in
another, `test_SE_mul_4` names Toulouse only in the math variant) are kept and
noted in `cultural_notes`. Translation checks: every number in each English
field appears in its Ukrainian counterpart, no extra keys, no duplicate ids.

Build the planner-facing file (prompt = `question_composition_uk`, hidden
`oracle_plan` = knowledge filter → instruct compute, `split` carried through).
Rubrics: the filter step passes if it names any accepted `answers_commonsense_uk`
string (`contains_any`); the final step passes if the number after the last
answer marker (`Відповідь:`, `=`) — or the last number in the text — equals
`answer_composition` (`numeric`, handles `19 000`, `19,000`, `2,2`):

```bash
python scripts/benchmarks/build_agentcoma_benchmark.py \
  --src benchmarks/graphs_agentcoma_200_source.jsonl \
  --out benchmarks/graphs_agentcoma_200.jsonl
# lab, 3 GPUs (mamay4 / lapa / aya), pattern of eval_composite_v2_bakeoff:
sbatch cluster/eval_agentcoma_200.sbatch            # SPLIT=dev (tuning)
SPLIT=test sbatch cluster/eval_agentcoma_200.sbatch  # once, 3 repeats
```

Output files are tagged `*_composite_agentcoma200_*`; results go to
`results/week_13_agentcoma_200_{dev,test}/`. No model has been run on the
200-item file yet; the only AgentCoMa numbers on record are the planner-only
Mamay-4B screens on the 50 dev items (`results/a2_agentcoma*/`, not end-task
quality).
