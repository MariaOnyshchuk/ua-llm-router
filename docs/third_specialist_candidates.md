# Third router specialist — investigation (beyond UA leaderboard)

Target trio: **Mamay-4B** + **Lapa** + **one non-Gemma / non-Mamay lineage model**.

Important lineage note: Lapa is Gemma-3-12B fine-tuned for UA; Mamay is Gemma-3 fine-tuned for UA.
So “different” means **not another Gemma/Mamay clone** (exclude Mamay-12B / Tower+ / plain Gemma as the *third* slot).

Sources used (not only lang-uk leaderboard):
- Cohere Labs Aya Expanse model card + arXiv:2412.04261
- EuroLLM-9B tech report arXiv:2506.04079 + eurollm.io
- RANLP 2025 Ukrainian benchmark study (UCU) — base vs instruct on FLORES / Belebele / MMLU-UA
- Unbabel Tower+ arXiv:2506.17080 (translation specialist; base is Gemma → **not** a lineage diversifier)
- Qwen2.5 official blog / model cards (code, math, multilingual)
- Our own bake-off: Qwen2.5-Coder-**3B** lost code to Mamay-4B

---

## Shortlist (fits ~1× RTX 6000 Ada, open weights)

| Rank | Model | Lineage | Size | License | Why it could earn a bucket | Risks |
|------|-------|---------|------|---------|----------------------------|-------|
| **1** | **CohereLabs/aya-expanse-8b** | Cohere Command (not Gemma) | 8B | CC-BY-NC | Explicitly trained for **23 languages incl. Ukrainian**; paper claims strong multilingual Arena wins vs Llama/Qwen/Gemma class | Non-commercial license; gated HF; not UA-specialized like Lapa/Mamay |
| **2** | **utter-project/EuroLLM-9B-Instruct** | EuroLLM (from-scratch EU) | 9B | Apache-2.0 | Built for **all EU langs + Ukrainian**; heavy **parallel MT data** in pretrain; natural **translate** candidate | RANLP study: instruct variant weaker than base on several UA tasks (0-shot IT vs 3-shot base) — must bake-off carefully |
| **3** | **Qwen/Qwen2.5-7B-Instruct** | Qwen (Alibaba) | 7B | Apache-2.0 | Strong **code + math + IFEval** on English cards; genuine different stack; upgrades the failed Qwen-Coder-3B story | Tower+ paper notes Qwen2.5 weaker on MT vs Gemma-family; may RU-contaminate UA like 3B coder |
| **4** | **Qwen/Qwen2.5-Coder-7B-Instruct** | Qwen | 7B | Apache-2.0 | Purpose-built **code** specialist larger than the 3B that lost | Still not UA-tuned; only worth it if it beats Mamay-4B on our 32 code items |
| **5** | **mistralai/Mistral-Nemo-Instruct-2407** | Mistral | 12B | Apache-2.0 | Different EU vendor; general instruct; same size class as Lapa | No UA specialization claim; 12B competes with Lapa on VRAM |
| — | Llama-3.1-8B-Instruct | Meta | 8B | Llama | Easy negative control | RANLP + UA leaderboard: weak on UA-specific tasks |
| — | Mistral-7B-Instruct-v0.3 | Mistral | 7B | Apache | Cheap ablation | RANLP: poor UA FLORES / Belebele vs peers |
| ❌ | Unbabel/Tower-Plus-9B | **Gemma-2** | 9B | CC-BY-NC | Excellent **translate** paper results, Ukrainian listed | **Same family problem** — does not diversify lineage vs Lapa/Mamay |
| ❌ | Mamay-12B | Gemma-3 | 12B | — | Wins translate on our bake-off | Same family; user asked to leave this out of the “different” slot |

---

## Suggested router shape (after bake-off of the third model)

Keep fixed until measured:

| Bucket | Likely model | Rationale |
|--------|--------------|-----------|
| chat / instruct / code (default) | **Mamay-4B** | Already wins code locally; small default |
| knowledge / alignment | **Lapa** | Already wins those buckets on our matrix |
| **translate** (or multilingual chat) | **Aya-8B or EuroLLM-9B** | Third lineage; both advertise UA + MT |

If Aya/EuroLLM lose translate to Mamay-4B on `mixed_ua_v4`, try **Qwen2.5-7B** on **code+instruct** instead and keep translate on Mamay-4B or Lapa.

---

## What each source actually says (one line)

1. **Aya Expanse (Cohere):** multilingual preference training + merging; 23 langs including Ukrainian; open weights for research (CC-BY-NC).
2. **EuroLLM-9B:** EU-funded, trained from scratch on 4T tokens incl. parallel en↔xx; Instruct tuned with MT focus; Apache-2.0.
3. **RANLP UA study (2025):** for generic multilingual models, **base+few-shot often beats instruct+0-shot** on UA FLORES/Belebele — so don’t trust English instruct scores; bake-off on our suite.
4. **Tower+:** great translate story but **Gemma-2 base** → wrong for lineage diversity.
5. **Qwen2.5 cards:** dominate HumanEval/MATH/IF relative to same-size Llama/Gemma-IT; multilingual training claimed, but not UA-first.

---

## Recommendation

**Download and bake-off #1 first: `CohereLabs/aya-expanse-8b`**  
(or Apache-friendly **`EuroLLM-9B-Instruct`** if CC-BY-NC is a thesis constraint).

Do **not** put Mamay-12B or Tower+ in the “different model” slot. Use them only as optional same-family ceilings.

Next concrete step: serve Aya (or EuroLLM) alone on `mixed_ua_v4_balanced`, compare per-bucket to Mamay-4B and Lapa, then rewrite routes to:

`mamay4 + lapa + <winner third>`.
