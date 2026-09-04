# Week 7 — rules v2 + mixed_ua_v5

*21 Aug 2026.*

## 1. Put the v4-best map into the **rules** (not gold labels)

`router/intent_rules.py`: **code → Mamay-4B**, **default chat → Aya**. Qwen-7B unused. Few-shot alignment kept.

| Intent | Rules v1 | Rules v2 | Gold-bucket oracle |
|--------|----------|----------|--------------------|
| code | Qwen-Coder-7B | **Mamay-4B** | **Mamay-4B** |
| default chat | Mamay-4B | **Aya-8B** | **Aya-8B** |
| translate | Aya-8B | Aya-8B | Aya-8B |
| knowledge | Lapa-12B | Lapa-12B | Lapa-12B |
| alignment | Lapa-12B | Lapa-12B | Lapa-12B |
| instruct | Mamay-4B | Mamay-4B | Mamay-4B |

v2 copies the oracle **map**. The oracle still reads gold `bucket`; v2 guesses intent with regex.

Same suite as the oracle (`mixed_ua_v4` 192×3, T=0, max_tokens=256):

| System | Overall | p50 | chat | code | GPUs |
|--------|--------:|----:|-----:|-----:|-----:|
| Rules v1 + few-shot (week 6) | 0.842 | **441** | 0.984 | 0.969 | 4 (incl. Qwen) |
| Oracle best + few-shot | 0.848 | 633 | 1.000 | 0.984 | 3 |
| **Rules v2 + few-shot** | **0.848** | 633 | **1.000** | **0.984** | **3** |

Rules v2 **matches the oracle** on v4 (0.8475 vs 0.8477). Traffic: aya 63 · mamay4 64 · lapa 65. Soft leftover still `chat-017` → Lapa.

p50 still 633 ms: you pay Mamay-4B on code to pick up +0.016 code / +0.016 chat.

## 2. Bigger suite: `mixed_ua_v5` (224)

Not a web scrape: official **HumanEval** (`openai/openai_humaneval`, MIT) → `benchmarks/corpus/humaneval_code_en.jsonl` (164). Suite = **v4 192 + 32 HumanEval** (seed 42).

Chat and instruct are still **32** (hand-written). Those, not code, are now the balanced-size cap. Knowledge/translate/alignment already have thousands in `corpus/`.

| | n | code | other buckets |
|--|--:|-----:|----------------|
| v4 | 192 | 32 UA exec | 32×5 |
| v5 | 224 | 32 UA + 32 EN | same 32×5 |

Rules v2 on v5 (3×224, **max_tokens=512** for HumanEval):

| | Overall | p50 | chat | code (64) | UA code (32) | HumanEval (32) |
|--|--------:|----:|-----:|----------:|-------------:|---------------:|
| Rules v2 | 0.796 | 1043 | 0.891 | 0.789 | **0.984** | **19/32 = 0.594** |

**Do not compare 0.796 to 0.848 as a router regression.** Two confounds:

1. HumanEval is harder (pass@1 19/32 on Mamay-4B). Combined code 0.789 is the mix, not a drop on UA `code-*`.
2. **max_tokens=512** was global, so some **chat** answers exceeded the scorer’s 1200-character cap → chat 0.891 vs 1.000 on v4@256.

Traffic v5: mamay4 96 (64 code + 32 instruct) · lapa 65 · aya 63. All 32 HumanEval routed as **code**.

## Closed

- S3 ensemble: **done** — [`docs/week_7_ensemble.md`](week_7_ensemble.md). 0.843 vs 0.848; skip product.
- Quantization (bitsandbytes 4-bit): **done** — [`docs/week_8_quant.md`](week_8_quant.md). 0.830 / slower; keep bf16. One-GPU packing **not** measured.
- Social prompt 1b: **skipped** — few-shot already 0.750; leftover is 2→1, not 1→2. See [`docs/conclusion.md`](conclusion.md).

Fair v5 number (256 on chat/instruct, 512 only on `humaneval-*`) and a larger balanced chat/instruct set remain optional, not product blockers.

Artifacts: `results/week_7_rules_v2/`, `results/week_7_v5/`.
