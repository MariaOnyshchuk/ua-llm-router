# Specialist Router for Ukrainian LLMs

## Thesis claim

Instead of hosting one ~120B model, route each query to a **specialist** (~12–30B). Complementary skills approximate a larger model’s coverage. This is **multi-LLM routing** (across independently trained models), not MoE inside one network — see [Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey](https://arxiv.org/html/2603.04445v2) (arxiv:2603.04445).

“Разом = 120B” means **effective capability ensemble**, not summed parameters. State that explicitly in the diploma.

## Specialist map (hypotheses)

| Alias    | Model (placeholder HF / API)              | Strength (hypothesis)     | Route when                                      |
|----------|-------------------------------------------|---------------------------|-------------------------------------------------|
| `lapa`   | `lapa-llm/lapa-v0.1.2-instruct`           | UA personality, translation | UA chat, style, en↔uk                         |
| `mamay`  | `INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0` | Instruction following | Multi-step, system prompts, IF-style tasks      |
| `gemma`  | `google/gemma-3-12b-it` (or Gemma 4 when pinned) | Fresher knowledge     | Facts, “що нового”, general world knowledge     |
| `qwen`   | `Qwen/Qwen2.5-Coder-14B-Instruct`         | Coding                    | Code, debug, technical implementation           |

Fill real endpoints in `litellm/config.yaml` once vLLM (or APIs) are up.

## System sketch

```
Client (OpenAI SDK)
        │  model: "auto"  |  or explicit alias
        ▼
 Intent router (rules-first)     ← this repo: router/
        │  rewrites model → lapa|mamay|gemma|qwen
        ▼
 LiteLLM proxy                   ← litellm/config.yaml
        │
   ┌────┼────┬────┐
 vLLM  vLLM API  API             (lab GPUs via SLURM; or remote)
 lapa  mamay gemma qwen
```

## Routing strategy (phased)

1. **Rules-first (MVP)** — keyword / regex intent → one specialist. Deterministic, easy to ablate.
2. **Cascade (next)** — try small/cheap specialist; escalate on low confidence or failed checks (survey: cascading paradigm).
3. **Learned router (stretch)** — classifier or preference model over the same aliases.

Paper axes for Methods: **when** (pre-generation), **what** (task/domain signals), **how** (rules → later classifier).

## Evaluation ladder

1. Per-skill baselines (each specialist alone on its niche).
2. Mixed suite (UA chat/MT + IF + knowledge + code).
3. Compare: best single model vs rules router vs (later) cascade / learned router vs optional ~70–120B reference.

## Lab constraints

- GPU work only via **SLURM**.
- `ucu-lab-2232`: 2× RTX 3090 — typically 1–2 quantized backends live; rest API or swap.
- `ucu-lab-2240`: 4× A6000 — better for concurrent specialists.
- Acknowledge compute grants in papers/demos (Talents for Ukraine / ELEKS as per lab docs).

## Repo layout

```
Diploma/
  DESIGN.md                 ← this file
  litellm/config.yaml       ← aliases → backends
  router/intent_rules.py    ← rules
  router/app.py             ← tiny OpenAI-compatible shim
  scripts/smoke_test.sh     ← curl checks
```
