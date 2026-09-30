# Working rules for this repository

Rules for any AI agent (and any human) working in this repo. Cursor loads this
file automatically. Keep it short; put long explanations in `docs/`.

## Git

- Commit only as **Maria Onyshchuk `<MariaOnyshchuk@users.noreply.github.com>`**.
  The repo-local `git config` is already set to that. Never use the global
  `monys@softserveinc.com` identity here. Before every commit run
  `git config user.email` and stop if it is not the noreply address.
- Remote is `github.com/MariaOnyshchuk/ua-llm-router`. Do not add other remotes.
- Commit messages: one imperative sentence describing the change
  (e.g. `Record Mamay-12B S2 baseline on mixed_ua_v4`). No prefixes, no emoji.
- Never commit `.env`, API keys, `logs/`, `.venv/`, or anything already in
  `.gitignore`.
- Logs and per-iteration generation files — JSONL rows that contain the model
  answer (`prompt` / `content`) — do not go in this git repo. Publish them on
  the Hugging Face eval dataset. Git keeps code, docs, score summaries,
  `*.meta.json`, `scores_detail` (ids and scores only), tables, and
  `progress_ledger.csv`.
- `scripts/publish_hf_eval.py` uploads those summaries. It skips raw
  `prompt`/`content` rows on purpose: they reprint ZNO-Eval, FLORES-200,
  UAlign, HumanEval, and UA-Code text. AgentCoMa prompts
  (`benchmarks/agentcoma_uk_50*`, `results/a2_*`) stay local; that source is
  gated and has no reusable license.
- Do not force-push or rewrite history on `main`.

## Evaluation protocol

- **Development suite (not for claims):** `benchmarks/mixed_ua_v4_balanced.jsonl`
  (192 = 32×6), `T=0`, `seed=42`, `max_tokens=256`, `--align-prompt fewshot`,
  **3 repeats**. Every number quoted next to the router's 0.848 must use this.
- `mixed_ua_v6` uses `max_tokens=1024` and different items. Never place v3, v4,
  v5, v6 or composite numbers in the same "did we improve?" cell.
- Screening means (`results/tables/v6_screen_*`, `results/v6_screen/`) are not
  for claims. Cite only 3× bake-off results.
- Single-model runs: **one model live on the GPU at a time**, bf16,
  `gpu_memory_utilization=0.90`, so VRAM figures stay comparable.
- Hosted API baselines (S3): no VRAM or GPU-seconds column. Report USD per 1000
  queries and "data leaves perimeter = yes". Keep `--concurrency 1`.
- Historical artifacts listed as "do not cite" in
  `docs/status_report_2026-07-27.md` stay uncited.

## Evidence rules (added 29 Sep 2026)

One item on v4 moves the overall score by 0.0052 and a bucket by 0.031. The 95%
interval for a router-vs-model difference on v4 is about ±0.04, so most v4
differences are noise. The rules below follow from that.

- **Claims come from `benchmarks/mixed_ua_v6.jsonl`** (2414 items: knowledge 683,
  translate 683, alignment 683, instruct 327; chat and code have 19 each and are
  appendix only). v4 is for development and diagnostics.
- **Every claimed difference carries a paired bootstrap 95% interval**
  (`python scripts/analysis/bootstrap_compare.py`). If the interval contains 0,
  write "not distinguishable", not "better" or "worse".
- **Freeze before you test.** Routing table, prompts and few-shot examples are
  chosen on v4 (or a dev half) and recorded with the git commit hash before the
  first v6 run. No tuning on v6. Any later change starts a new, labelled run.
- **One pass on v6 is enough** (n is large; T=0, seed=42, max_tokens=1024). Do not
  compare v6 numbers with v4 numbers.
- **Item-level scores are kept for every cited run** (`scored/*scores_detail.jsonl`),
  otherwise no interval can be computed.
- **After any change to `score_results.py`, rescore every run that is cited** and
  rebuild the ledger. On 29 Sep 2026 a bug from 4 Sep (numeric answers such as 1991
  read as multiple-choice indexes) had understated every model by 0.0104 on v4.
- **Say what v6 is:** it was built by dropping low-variance items using the
  Mamay-4B, Lapa and Aya screening runs, and its knowledge bucket is 99% multiple
  choice. State both limits next to any v6 headline.

## Scoring and bookkeeping

- `score_results.py` de-duplicates on `(system, id)`: score **each repeat
  separately**, then `aggregate_repeat_scores.py` (or `score_repeat_dir.py`).
  Never pass all reps to one scoring call.
- Check every JSONL is HTTP 200 on all 192 items before scoring. Runs with
  `HTTP 0` / empty generations are serving failures, not model scores.
- After a run lands: rebuild `results/progress_ledger.csv` with
  `python scripts/build_progress_ledger.py`; do not hand-edit it. Then update
  the status rows in `docs/thesis_scope.md`, `docs/key_takeaways.md`,
  `docs/weeks_7_10_takeaways.md`, and `results/README.md`.
- Every new `results/<group>/` folder gets a one-line entry in
  `results/README.md`.

## Cluster (`ucu-lab-2240`)

- Slurm lives on the lab, not the laptop. Submit with `sbatch cluster/*.sbatch`.
- The lab `~/Diploma` is **not a git checkout**. Copy changed files with `scp`
  and verify with a checksum diff before running.
- Prefer co-located serve + eval in one sbatch (pattern:
  `cluster/eval_s2_v4.sbatch`, `cluster/eval_v6_solo.sbatch`) so the job
  survives a dropped SSH session and nvidia-smi sees only its GPU.
- Check `squeue -u $USER` and `nvidia-smi` before submitting; do not start a
  fair single-model run while another model is resident on the same GPU.
- Pull results back with `rsync` into the matching `results/<group>/` folder.

## Editing conventions

- Ask before destructive actions, deleting result files, or spending money
  (API runs). Reversible edits that follow from the request: just do them.
- Temporary scripts and scratch checks go in `/tmp`, never in the repo.
- Do not create new docs unless asked; extend the existing `docs/*.md`.
- Prose in docs: plain, no marketing language, state negative results plainly.
