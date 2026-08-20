# Thesis Scope — Router over Ukrainian Open-Source LLM Specialists

## 1. Research question

Can a lightweight router that dispatches tasks across several small Ukrainian
open-source specialist LLMs (e.g. Lapa, Mamay, Gemma, Qwen) match — or come
close to — the quality of a single large monolithic LLM on a mixed Ukrainian
workload, while using meaningfully less GPU memory, latency, and/or cost?

## 2. Hypothesis

A rules-based router over **small** open specialists (~3–7B, optionally one
mid-size UA model) achieves quality within an acceptable margin of a **larger
baseline** (e.g. Mamay 12B / 27B), while using **lower peak VRAM** and/or
**lower latency** when only one small model is hot (or when the concurrent
small footprint ≪ the large model).

## 3. Systems compared

| ID | System | Description |
|----|--------|-------------|
| S0 | Single small | One small specialist alone (e.g. Mamay-4B) on every prompt |
| S1 | Rules router | Rules → small specialists (Mamay-4B, Qwen-Coder-3B/7B, …) |
| S2 | Large baseline | **MamayLM 12B v2** (or 27B when servable) — quality / cost ceiling |

**Size story (important):** specialists must be **smaller than S2**. Lapa-12B vs Mamay-12B was a useful wiring demo, not the efficiency experiment.

### Target model pool (next)

| Role | Candidate (HF) | Size |
|------|----------------|------|
| UA general / translate / IF | `INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0` | **4B** |
| Code | `Qwen/Qwen2.5-Coder-3B-Instruct` (or 7B) | **3B–7B** |
| Large baseline S2 | `INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0` | **12B** |

## 4. Evaluation suites

| Suite | Contents | Role |
|-------|----------|------|
| `mixed_ua_v0` | 40 hand-written prompts (8×5 buckets) | Cheap smoke + routing accuracy |
| `mixed_ua_v1` | v0 ∪ ZNO-Eval sample ∪ (planned) UA-Code-Bench easy | Statistical weight on knowledge/code |

See `docs/benchmarks.md`. Cite English router suites (RouterBench, etc.) as **methodology only** — do not use as UA prompt sources.

## 5. Metrics

**Quality** (per bucket):
- `chat` — heuristic / later LLM-as-judge
- `translate` — chrF-like vs reference (+ optional FLORES-200 uk-Cyrl)
- `instruct` — format / exact constraints (custom; UA gap → limitations)
- `knowledge` — letter match on ZNO-Eval MCQ; keyword/ref on custom items
- `code` — exec unit tests (custom); Eolymp judge for UA-Code-Bench when wired

**Resources (always report):**
- **Latency:** mean, p50, p95 (ms) overall and per bucket
- **Peak VRAM (GB)** per live backend / per system
- Optional: tokens/s

**Efficiency:**
- Quality / peak VRAM
- Quality / mean latency

Always report **v0 and v1 separately** so early S0/S1/S2 numbers stay comparable.

## 6. Out of scope

- Training or fine-tuning a 120B model
- Building a full production-grade MoE / learned router
- Production SLA, autoscaling, uptime guarantees
- Non-Ukrainian-language workloads
- Growing the deployment prototype beyond the current thin rules → LiteLLM → Lapa stack
- UA-Legal-Bench / domain-specific legal escalation (unless thesis scope expands)

## 7. Compute note

_(Fill in grant/compute wording here if using ELEKS / Talents for Ukraine
resources, e.g. GPU allocation IDs 2240 / 2232.)_