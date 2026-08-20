# Full corpora warehouse (`benchmarks/corpus/`)

**This is the thousands layer** — not the tiny `samples/` used in early weeks.

Regenerate:
```bash
python scripts/extract_full_corpora.py
```

| File | Rows | Notes |
|------|-----:|-------|
| `zno_knowledge_full.jsonl` | **4 776** | Eligible single-letter MCQ, no photo (of 7 380 raw ZNO tasks) |
| `flores_translate_full.jsonl` | **4 018** | FLORES en↔uk, both directions, dev+devtest |
| `ualign_ethics_full.jsonl` | **1 700** | UAlign ETHICS test |
| `ualign_social_full.jsonl` | **3 682** | UAlign Social Chemistry 101 test |
| `uacode_problems_full.jsonl` | **468** | Unique Eolymp problems (HF has 7 499 model-runs; deduped; media skipped) |
| **Total** | **≈14 644** | |

`INVENTORY.json` — machine-readable counts.

### What is still small (no public UA corpus of thousands)

| Bucket | Reality |
|--------|---------|
| `chat` | Hand-written only (~32) — no large open UA chat-eval dump in our stack |
| `instruct` | Hand-written only (~32) — IFEval-UA not wired as full dump yet |
| `code` scored | Hand-written exec-tested (32) — UA-Code full is imported but needs Eolymp judge |

### Balance from the warehouse

```bash
python scripts/build_balanced_suite.py --per-bucket 32 \
  --out benchmarks/mixed_ua_v4_balanced.jsonl
# optional: pull code from UA-Code corpus (unscored locally)
python scripts/build_balanced_suite.py --per-bucket 32 --include-uacode \
  --out benchmarks/mixed_ua_v4_with_uacode.jsonl
```
