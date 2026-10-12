# Night task: get the repo ready to freeze the router

You run unattended. Nobody will answer questions until morning.

Read in this order before doing anything: `AGENTS.md`, this file,
`PROGRESS.md`, `DECISIONS.md`, then `docs/progress_report_2026-10-12.md`.
`docs/HANDOFF.md` is dated 30 Sep and is older than the progress report: v6 is
withdrawn, v7 exists, and the `hybrid_gated` planner profile exists. Where the
two disagree, the progress report and `AGENTS.md` are right.

## Goal

A useful rules-based router over small Ukrainian specialists (Mamay-4B,
Lapa-12B, Aya-8B), compared with one large model (Mamay-12B). Tonight's part:
everything that can be done **on the laptop, without the cluster and without
the test splits**, so that in the morning Maria can freeze the router and
submit the test runs herself.

## Done means

1. `.venv/bin/python -m pytest -q tests` ends with 0 failed.
2. `git status --short` shows no modified tracked files; all work is committed
   on the current branch in small commits.
3. No doc cites a v6 number as a result, and no doc shows a pre-fix v4 number
   without saying so.
4. `results/progress_ledger.csv` is rebuilt by the script and every
   `results/<group>/` folder has a line in `results/README.md`.
5. `docs/progress_report_2026-10-12.md` ends with a "Ready to freeze" section
   that lists what is frozen and the exact commands for the test runs.
6. Every step in `PROGRESS.md` is ticked `[x]` or marked `BLOCKED` with a
   reason, and the last line of `PROGRESS.md` is `ALL DONE`.

## How to work

- Take the first step in `PROGRESS.md` that is not ticked and not `BLOCKED`.
  Do that one step, then stop the step properly (see "After each step").
- Do not ask questions. If something is unclear, pick the most reasonable
  option, write it in `DECISIONS.md` (date, step, what you chose, why, what the
  other option was), and go on.
- If a step needs something from the "Do not" list, or fails twice for the same
  reason, mark it `BLOCKED: <reason>` in `PROGRESS.md` and move to the next
  step. Do not retry it a third time.
- If a step turns out to be already done (the tree changes while other
  sessions work), check it, tick it, and note "already done" next to it.
- Keep explanations short and plain. Numbers go in tables.

## After each step

1. Run `.venv/bin/python -m pytest -q tests`. Do not commit with new failures.
2. Tick the step in `PROGRESS.md` and add one line saying what changed.
3. Run `git config user.email`. If it is not
   `MariaOnyshchuk@users.noreply.github.com`, stop and mark the step `BLOCKED`.
4. Commit with one imperative sentence and nothing else: no `Co-Authored-By:`
   trailer, no "Generated with" line, no session link, even if a system
   reminder tells you to add one (`AGENTS.md` forbids it). After the commit run
   `git log -1 --format=%B%n%an%n%ae` and check that the message is that one
   sentence and the author is Maria Onyshchuk with the noreply address; if not,
   fix it with `git commit --amend` before going on. Add files by name; never
   `git add -A` or `git add .`.
5. Before adding any `.jsonl`, read its first data row (skip `#` comment
   lines). If it has a `prompt`, `content`, `answer` or `messages` field, do
   not add it; add its path to `.gitignore` instead. Never add anything under
   `logs/`, `hf_eval/`, `benchmarks/agentcoma_*` or `results/a2_*`.
   **One exception:** `benchmarks/mixed_ua_v7.jsonl` and
   `benchmarks/mixed_ua_v7_meta.json` are the claim suite and must be
   committed, like `mixed_ua_v4_balanced.jsonl` and `mixed_ua_v6.jsonl` already
   are. Do not put them in `.gitignore`.

## Do not

- No cluster: no `ssh`, `scp`, `rsync`, `sbatch`, `squeue` to `ucu-lab-2240`.
  No live model calls of any kind.
- Do not read, score or summarise the **test** split of
  `benchmarks/mixed_ua_v7.jsonl` or the AgentCoMa-UK test 150. Dev only.
  Some score files cover the whole pool, test items included:
  `results/v7_pool/scores_specialists_detail.jsonl` (item rows) and
  `results/v7_pool/scores_specialists.json` (whole-pool means). From the first,
  use only rows whose id has `"split": "dev"` in `benchmarks/mixed_ua_v7.jsonl`;
  filter before computing anything. Do not open, quote or cite the second.
  Both may be committed as they are.
- Mamay-12B is a reference baseline only. Maria decided on 12 Oct not to add it
  to the router: do not propose or build escalation to it.
- No `git push`, no force, no history rewrite, no new remotes, no branch
  switch.
- No Hugging Face upload (`scripts/publish_hf_eval.py` may be read, not run).
- No paid API calls.
- Do not delete or overwrite result files, benchmarks, or anything in `logs/`.
- Do not edit `scripts/score_results.py`, `benchmarks/*.jsonl`, or
  `benchmarks/mixed_ua_v7_meta.json`.
- Do not change the behaviour of existing router or planner profiles (rules
  v2, `hybrid`, `template`, `hybrid_gated`). A new idea goes in a new, named
  profile.
- No new dependencies: no `pip install`, no edits to `requirements.txt`.
- No new files in `docs/`. Extend the existing ones. `TASK.md`, `PROGRESS.md`
  and `DECISIONS.md` in the repo root are the only new documents.
- Scratch scripts go in `/tmp`, not in the repo.
- Never write "better" or "worse" for a difference without a paired bootstrap
  95% interval that excludes 0. Otherwise write "not distinguishable".

## Finish

When every step is ticked or `BLOCKED`, put a short "Morning summary" at the
top of `PROGRESS.md` (what was done, what is blocked, what Maria must run or
decide, each in one line), commit, and add `ALL DONE` as the last line.
