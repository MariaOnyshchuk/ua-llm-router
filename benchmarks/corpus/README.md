# Full corpora warehouse (`benchmarks/corpus/`)

**This is the thousands layer** — not the tiny `samples/` used in early weeks.

Regenerate:
```bash
python scripts/extract_full_corpora.py
python scripts/extract_full_corpora.py --only ifeval,belebele,mmlu,arc,wmt22
```

| File | Rows | Notes |
|------|-----:|-------|
| `zno_knowledge_full.jsonl` | **4 776** | Eligible single-letter MCQ, no photo (of 7 380 raw ZNO tasks) |
| `flores_translate_full.jsonl` | **4 018** | FLORES en↔uk, both directions, dev+devtest |
| `ualign_ethics_full.jsonl` | **1 700** | UAlign ETHICS test |
| `ualign_social_full.jsonl` | **3 682** | UAlign Social Chemistry 101 test |
| `uacode_problems_full.jsonl` | **468** | Unique Eolymp problems (HF has 7 499 model-runs; deduped; media skipped) |
| `ifeval_ukr_instruct.jsonl` | ~541 | IFEval-UA (`INSAIT-Institute/ifeval_ukr`) |
| `belebele_uk_knowledge.jsonl` | ~900 | Belebele `ukr_Cyrl` |
| `mmlu_ukr_knowledge.jsonl` | ≤2000 | Stratified cap of `INSAIT-Institute/mmlu_ukr` |
| `arc_challenge_ukr_knowledge.jsonl` | ~2.5k | `INSAIT-Institute/arc-challenge_ukr` |
| `wmt22_translate.jsonl` | varies | WMT-22 en↔uk (skipped if HF load fails) |

`INVENTORY.json` — machine-readable counts.

### What is still small (no public UA corpus of thousands)

| Bucket | Reality |
|--------|---------|
| `chat` | Hand-written only (~32) — no large open UA chat-eval dump in our stack |
| `instruct` | `ifeval_ukr_instruct.jsonl` (~541) — official IFEval-UA; hand-written `if-*` only in historical suites |
| `code` | Hand-written exec-tested (~32) — UA-Code needs Eolymp judge |

### Balance from the warehouse

```bash
python scripts/build_balanced_suite.py --per-bucket 32 \
  --out benchmarks/mixed_ua_v4_balanced.jsonl
# optional: pull code from UA-Code corpus (unscored locally)
python scripts/build_balanced_suite.py --per-bucket 32 --include-uacode \
  --out benchmarks/mixed_ua_v4_with_uacode.jsonl
```
