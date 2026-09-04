# Weeks 7–10 takeaways

*Written 1 Sep 2026. Authoritative single-skill numbers are on* `mixed_ua_v4_balanced` *unless a table names another suite. Decode protocol:* `T=0`*,* `seed=42`*,* `max_tokens=256`*,* `--align-prompt fewshot`*, 3 repeats when marked* `×3`*.*

This note is the glossary + metrics pack for the second half of the experiments. The live product claim is still: **rules v2 + few-shot = 0.848 / p50 633 ms**, matching the gold-bucket oracle. Extra hops and 4-bit do not beat it. Week 9 numbers are **not scored yet**. Week 10 has plumbing and a 6-item smoke; **do not cite smoke as quality**.

Per-week lab notes: `[week_7_router_best.md](week_7_router_best.md)`, `[week_7_rules_v2.md](week_7_rules_v2.md)`, `[week_7_ensemble.md](week_7_ensemble.md)`, `[week_8_quant.md](week_8_quant.md)`, `[week_9_baselines.md](week_9_baselines.md)`, `[week_10_composite.md](week_10_composite.md)`. Scope: `[thesis_scope.md](thesis_scope.md)`.

---



## 1. Glossary



### Specialist

An independently trained open model served with vLLM. The product pool is three models on three GPUs:


| Alias    | Model                  | Port | Role in rules v2        |
| -------- | ---------------------- | ---- | ----------------------- |
| `mamay4` | MamayLM Gemma-3 **4B** | 8003 | instruct, UA code       |
| `lapa`   | Lapa **12B**           | 8001 | knowledge, alignment    |
| `aya`    | Aya Expanse **8B**     | 8005 | translate, default chat |


`qwen7` (Qwen2.5-Coder-7B, `:8004`) is still in the serve graph but **not used** by rules v2. `mamay12` (Mamay **12B**, `:8002`) is the week-9 scaling baseline, not a router member.

### Bucket (intent)

One skill class on the mixed suite. v4 has six buckets of 32 items: **chat, code, translate, instruct, knowledge, alignment**. The scorer and the gold-bucket oracle use the item’s stored `bucket` field. The **rules router never sees that field** — it guesses intent from the prompt with regex.

### Router (rules / one-hop)

`router/intent_rules.py`: first matching regex → one specialist → one HTTP call. This is **multi-LLM routing** (separate models), not MoE inside one network.

- **Rules v1** (week 6 product): code → Qwen-7B, default chat → Mamay-4B.
- **Rules v2** (week 7 product): code → Mamay-4B, default chat → Aya. Qwen unused.



### Oracle (two different things — do not mix)


| Name                                             | When                    | What it knows                | What it measures                                                                      |
| ------------------------------------------------ | ----------------------- | ---------------------------- | ------------------------------------------------------------------------------------- |
| **Gold-bucket oracle** (weeks 7–9, single-skill) | `mixed_ua_v4` / v5      | the item’s gold `bucket`     | ceiling of “always send this bucket to its best specialist.” One hop.                 |
| **Workflow oracle** (week 10, composite)         | `mixed_ua_composite_v1` | stored `oracle_plan.steps[]` | ceiling of “run the right multi-step pipeline on the same specialists.” Several hops. |


Neither oracle is a bigger model. Both are **diagnostic ceilings over the same pool**. The product system must **not** read gold `bucket` or `oracle_plan`.

### ++Few++-shot (alignment prompt)

Social UAlign items collapse at baseline (`1` → `2`). A short few-shot prefix on the **prompt**, not a model swap, recovers alignment **0.594 → 0.750**. All week-7+ product numbers include `--align-prompt fewshot`.

### Cascade vs ensemble vs hybrid


| System                     | Extra machinery                                           | Verdict                            |
| -------------------------- | --------------------------------------------------------- | ---------------------------------- |
| Micro-cascade (week 6)     | if social answer is `2`, retry Mamay-4B                   | Δquality ≈ 0; skip                 |
| Discrete ensemble (week 7) | majority vote of Lapa + Mamay-4B + Aya on alignment + ZNO | 0.843 vs 0.848; skip               |
| Hybrid planner (week 10)   | Mamay-4B emits a JSON workflow; specialists execute steps | implemented; full GPU eval pending |




### Overall / p50 / GPU-s

- **Overall** — micro quality on the suite (mean of item scores).
- **p50** — median end-to-end latency (ms). Router p50 ≪ mean because short buckets dominate the median; chat is slow.
- **GPU-s** — active GPU-seconds for the job, not resident VRAM. Always-on serving still occupies one card per live vLLM at `gpu_memory_utilization=0.90`.

---



## 2. Benchmark versions

**Never put v3, v4, v5, and composite in one “did we get better?” cell.** Different items, sometimes different `max_tokens`.


| Suite           | Path                          | n              | What it is                                            | Use for claims                                 |
| --------------- | ----------------------------- | -------------- | ----------------------------------------------------- | ---------------------------------------------- |
| v0              | `mixed_ua_v0.jsonl`           | 40             | hand-written smoke                                    | routing checks only                            |
| v3              | `mixed_ua_v3.jsonl`           | ~182           | unbalanced; week-4 bake-off                           | specialist ranking **only**                    |
| **v4 balanced** | `mixed_ua_v4_balanced.jsonl`  | **192 = 32×6** | chat, code, translate, instruct, knowledge, alignment | **authoritative single-skill**                 |
| v5              | `mixed_ua_v5.jsonl`           | 224            | v4 + 32 English HumanEval                             | harder code mix; optional                      |
| Composite v1    | `mixed_ua_composite_v1.jsonl` | 36             | 12+12+12 dependent workflows                          | skill chaining, not another mixed-bucket score |


Warehouse behind the samples: `benchmarks/corpus/` (~14k). Full-corpus eval was deferred (time, imbalance, UA-Code without Eolymp judge). Chat and instruct at scale barely exist in open UA data; those 32+32 stay mostly hand-written.

**v4 sources (32 each):** ZNO-Eval (knowledge), FLORES-200 (translate), UAlign (alignment), UA-Code-style / hand-written executable Python (code), hand-written chat and instruct.

**v5 extra:** official HumanEval (`openai/openai_humaneval`, MIT) → 32 English `humaneval-`* items. Rules treat them as **code** → Mamay-4B.

**Composite v1 families** (planner sees only the user prompt; `oracle_plan` is hidden):


| Family                    | n   | Stored workflow       | Final check                                    |
| ------------------------- | --- | --------------------- | ---------------------------------------------- |
| translate→code            | 12  | Aya → Mamay-4B        | executable Python cases                        |
| knowledge→explain         | 12  | Lapa → Mamay-4B       | correct label + exact two-line evidence format |
| translate→knowledge→write | 12  | Aya → Lapa → Mamay-4B | exact JSON fields                              |


---



## 3. Router structures



### 3.1 Rules v2 (product, one hop)

```
User prompt
    │
    ▼
regex table (first match wins)     ← never sees gold bucket
    │
    ├─ translate  → aya :8005
    ├─ alignment  → lapa :8001
    ├─ instruct   → mamay4 :8003   (strong format markers only; not bare «тільки»)
    ├─ knowledge  → lapa :8001
    ├─ code       → mamay4 :8003   (v1 was qwen7)
    └─ default    → aya :8005      (chat; v1 was mamay4)
    │
    ▼
one vLLM completion  +  few-shot prefix on alignment items
```

| Intent | Rules v1 | Rules v2 | Gold-bucket oracle |
|--------|----------|----------|--------------------|
| code | Qwen-Coder-7B | **Mamay-4B** | **Mamay-4B** |
| default chat | Mamay-4B | **Aya-8B** | **Aya-8B** |
| translate | Aya-8B | Aya-8B | Aya-8B |
| knowledge | Lapa-12B | Lapa-12B | Lapa-12B |
| alignment | Lapa-12B | Lapa-12B | Lapa-12B |
| instruct | Mamay-4B | Mamay-4B | Mamay-4B |

v2 and the oracle share the **same specialist map**. The difference is the **intent source**: regex over the prompt vs the item’s gold `bucket` field. Qwen is unused in both v2 and the oracle.

Traffic on v4 (192): **aya 63 · mamay4 64 · lapa 65**. Soft leftover: `chat-017` still regexes to knowledge (oracle would send it to Aya).

### 3.2 Gold-bucket oracle (single-skill ceiling)

Same specialist **map** as rules v2, but the intent comes from the benchmark label, not regex. On v4 it scores **0.848 / 633 ms** — rules v2 matches this, so the regex table has essentially no remaining headroom on this suite.

### 3.3 Ensemble (week 7, not product)

```
rules v2 first hop
    │
    ├─ chat / code / translate / open instruct  → keep first hop
    └─ alignment (32) + ZNO ids (24)            → call Lapa, Mamay-4B, Aya
                                                    majority vote; tie = first hop
```

56/192 voted, 3 flipped, net **−1** item. p50 doubles because those items pay three sequential HTTP calls.

### 3.4 Composite systems (week 10)

```
                    ┌─ specialist alone ────────── one model answers the full prompt
User prompt ───────├─ router_direct (rules v2) ── one hop, same as weeks 7–8
                    ├─ workflow oracle ─────────── stored steps → specialist map
                    └─ hybrid ─┬─ Mamay-4B JSON plan (2–4 steps)
                               ├─ validate (intents, deps, cycles, placeholders)
                               ├─ one repair if invalid
                               └─ execute steps; {{step.content}} into later prompts
                                  invalid plan → single-hop fallback
```

Intent of each **step** still maps through the same rules-v2 specialist table. Two error sources: **workflow** (wrong/invalid plan) vs **execution** (right plan, specialist fails the rubric). Direct systems have no observable stages — compare them on **final score** only.

---



## 4. Headline table (same suite: v4, 192)


| System                               | Overall   | p50 ms  | chat      | code      | translate | instruct | knowledge | alignment | Hardware note                 |
| ------------------------------------ | --------- | ------- | --------- | --------- | --------- | -------- | --------- | --------- | ----------------------------- |
| **Rules v2 + few-shot** (`×3`)       | **0.848** | **633** | **1.000** | **0.984** | 0.820     | 0.781    | **0.750** | **0.750** | 3 GPUs, bf16                  |
| Gold-bucket oracle + few-shot (`×3`) | **0.848** | 633     | **1.000** | **0.984** | 0.821     | 0.781    | **0.750** | **0.750** | 3 GPUs                        |
| Ensemble vote (`1×`)                 | 0.843     | 1201    | 1.000     | 0.984     | 0.822     | 0.781    | 0.688     | **0.781** | 3 calls on 56 items           |
| Rules v2 **4-bit bnb** (`1×`)        | 0.830     | 1115    | 1.000     | **1.000** | 0.795     | 0.719    | 0.719     | 0.750     | 3 processes, still ~44 GB/GPU |
| Rules v1 + few-shot (code→Qwen)      | 0.842     | **441** | 0.984     | 0.969     | 0.821     | 0.781    | 0.750     | 0.750     | 4 GPUs incl. Qwen             |
| Aya-8B alone                         | 0.785     | 867     | **1.000** | 0.953     | **0.821** | 0.781    | 0.562     | 0.594     | 1 GPU                         |
| Mamay-4B alone                       | 0.762     | 1125    | 0.984     | **0.984** | 0.758     | 0.781    | 0.469     | 0.594     | 1 GPU                         |
| Lapa-12B alone                       | 0.732     | 1681    | 0.984     | 0.938     | 0.436     | 0.688    | **0.750** | 0.594     | 1 GPU                         |
| Mamay-12B on **v4**                  | —         | —       |           |           |           |          |           |           | **week 9, not run**           |
| Hosted frontier API                  | —         | —       |           |           |           |          |           |           | **week 9, not run**           |


Rules v1 is faster at p50 because code went to Qwen. Rules v2 pays Mamay-4B on code to pick up **+0.016 chat** and **+0.016 code** and to match the oracle.

HTTP: 192/192 on these cluster runs.

---



## 5. Week 7 — lock the product map, then stop adding hops



### 5.1 Put the oracle map into regex

The gold-bucket run (`results/week_7_router_best/`) showed the two v1 mistakes: chat should be Aya, code should be Mamay-4B. Wiring that into `intent_rules.py` produced **rules v2 = oracle** on v4 (0.8475 vs 0.8477).

### 5.2 Ensemble — measured and rejected


|          | Overall   | p50     | knowledge | alignment | extra GPU-s |
| -------- | --------- | ------- | --------- | --------- | ----------- |
| Rules v2 | **0.848** | **633** | **0.750** | 0.750     | —           |
| Vote     | 0.843     | 1201    | 0.688     | **0.781** | +59         |


Flips: two correct Lapa ZNO answers overwritten by Mamay+Aya; one social item improved. Net −1. **Skip for product.**

### 5.3 v5 is a different suite

Rules v2 on v5 (3×224, **max_tokens=512** global):


|          | Overall | p50  | chat  | code (64) | UA code (32) | HumanEval (32)    |
| -------- | ------- | ---- | ----- | --------- | ------------ | ----------------- |
| Rules v2 | 0.796   | 1043 | 0.891 | 0.789     | **0.984**    | **19/32 = 0.594** |


Do **not** read 0.796 vs 0.848 as a router regression. HumanEval is harder; global 512 tokens also pushed some chat answers over the scorer’s 1200-character cap (chat 1.000 → 0.891). Traffic: mamay4 96 (64 code + 32 instruct) · lapa 65 · aya 63. All 32 HumanEval routed as code.

---



## 6. Week 8 — 4-bit bitsandbytes is a regression here

Same product router, same v4 192, 1 repeat. Serve flags: `--quantization bitsandbytes --load-format bitsandbytes`, `gpu_memory_utilization=0.90`, `max-model-len=8192` (matched to bf16).


|                      | Overall   | p50     | GPU-s | instruct  | knowledge | translate |
| -------------------- | --------- | ------- | ----- | --------- | --------- | --------- |
| bf16 rules v2 (`×3`) | **0.848** | **633** | ~295  | **0.781** | **0.750** | 0.820     |
| 4-bit bnb (`1×`)     | 0.830     | 1115    | 473   | 0.719     | 0.719     | 0.795     |


Latency got **worse**: bitsandbytes on vLLM 0.25.1 / Ada is a slower matmul than fused bf16. nvidia-smi still ~44 GB/GPU because KV cache fills the reservation. **This is not a packing / “fits on one GPU” result.** Colocation was not measured. Keep bf16.

---



## 7. Week 9 — the two missing reference points (plan)

Everything above compares the router to models **inside its pool**. A reviewer can call that circular. Week 9 adds the comparisons a decision-maker actually weighs:


| Slot | System                                                        | Question                                                         | Status                                                      |
| ---- | ------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| S2   | Mamay-12B on v4, 1 GPU, same util 0.90                        | Does composition beat **scaling** one open model?                | Script + sbatch ready; **no v4 score yet** (only v3: 0.793) |
| S3   | Hosted frontier API (e.g. gpt-4o), sequential, USD/1k queries | How far from the option a **data-residency** constraint forbids? | `scripts/run_api_baseline.py` ready; **not run**            |


Interpretation if S2 lands:

- Mamay-12B **< 0.848** — strongest thesis line (composition beats scale on this mix).
- **≈ 0.848** — claim latency and per-bucket robustness, not headline quality.
- **> 0.848** — report it; fallback is “three small specialists approach a 12B, cheaper per query.”

S3 is expected to win quality. Keep it **out of GPU columns**; cite USD and “data leaves perimeter = yes.” The thesis question is how much the on-prem composite gives up, not whether GPT is better in the abstract.

Draft table (router row filled; S2/S3 empty until scored):


| System              | Hardware        | Overall   | p50     | Data leaves perimeter |
| ------------------- | --------------- | --------- | ------- | --------------------- |
| Frontier API        | hosted          | ?         | ?       | **yes**               |
| Rules v2 + few-shot | 3 GPUs resident | **0.848** | 633 ms  | no                    |
| Mamay-12B           | 1 GPU           | ?         | ?       | no                    |
| Aya-8B              | 1 GPU           | 0.785     | 867 ms  | no                    |
| Mamay-4B            | 1 GPU           | 0.762     | 1125 ms | no                    |
| Lapa-12B            | 1 GPU           | 0.732     | 1681 ms | no                    |


---



## 8. Week 10 — can the pool chain skills in one request?

Single-skill v4 only asks “who should answer this isolated bucket?” Composite v1 asks whether **intermediate output must feed the next specialist**.

**Implemented:** benchmark builder, `router/planner.py`, `router/orchestrator.py`, runner, scorer, CPU tests (oracle fixtures, malformed JSON, cycles, HTTP with fake backends, 11/11 leakage cases).

**Verified without claiming model quality:** 36 items well-formed; oracle workflows × fixture repeats pass; hybrid repair/fallback paths pass.

**Live smoke (6** `translate_code` **items only, 1 repeat):** plumbing works (plans valid, hybrid exact-match 6/6 on that slice). Final scores of 1.0 on n=6 are **not** a thesis number. Full command is 7 systems × 36 × 3 in `results/week_10_composite/`.

Metrics to report after the full run: final score / exact success; all-stages-pass and mean stage score (oracle/hybrid only); valid-plan rate, exact workflow match, intent F1; calls per item; e2e latency; GPU-s per item.

**Claim boundary:** three hand-built workflow families with deterministic rubrics. Not “general complex reasoning.”

---



## 9. What is closed vs next


| Item                       | Result                                                  |
| -------------------------- | ------------------------------------------------------- |
| Alignment hole             | Prompt bug; few-shot 0.750; do not swap Lapa → Mamay-4B |
| Leakage                    | Fixed in regex; leftover `chat-017` → knowledge         |
| Rules v2                   | Matches gold-bucket oracle on v4                        |
| Cascade / ensemble / 4-bit | None beat 0.848; skip product                           |
| One-GPU packing            | **Not measured**                                        |
| Mamay-12B on v4 (S2)       | **Next**                                                |
| Frontier API on v4 (S3)    | **Next**                                                |
| Composite 36×3 (S5/S6)     | Code done; **GPU eval next**                            |


**Honest resource line:** the router cuts **median latency** and **active GPU-seconds** (~1.5 GPU-s / single-skill prompt). It does **not** use less resident VRAM than “one model on one GPU.” Three hot specialists ≈ three cards of KV-reserved memory.

**Product stop (single-skill):** rules v2 + few-shot, bf16, one hop, three specialists. Write the diploma around that, then add S2/S3 and composite when the JSONL exists.