# Progress

## Morning summary (13 Oct 2026)

**Done**

- All work from 10–12 Oct is committed in small commits, authored by Maria only, no trailers.
- Tests: 69 passed, 1 skipped (the kNN test, `scikit-learn` not installed). 0 failed.
- `v3_instruct` routing profile tried offline on v7 dev: +0.006 [−0.030, +0.041] on instruct, not distinguishable. Not recommended.
- Docs: Mamay-12B on v4 is 0.884 and the FP8 pack 0.852 (rescored); week 5–8 numbers are marked "not rescored"; v6 is marked withdrawn everywhere and the docs point to v7.
- Ledger rebuilt (567 rows): v6 runs `superseded`, two v7 dev runs `provisional`. Every results folder has a README line.
- The progress report ends with "Ready to freeze".

**Blocked**

- Nothing is blocked.

**Maria must run or decide**

- Push the branch: `git push --force-with-lease origin 2026-08-31-323p` (history was rewritten to drop 17 `Co-Authored-By` trailers; backup branch `backup/before-trailer-rewrite-20261013`).
- The test jobs are not ready: the v6 cluster jobs are pinned to the v6 file and the AgentCoMa jobs to `max_tokens 256`. See "What is missing before the test runs can be submitted" in the report. A v7 job has to be written and tried on dev first.
- Decide whether v7 test runs the planner (`hybrid_gated`) or only one-hop rules v2.
- `docs/HANDOFF.md` is updated but still untracked, as you asked earlier. Commit it if you want it in the repo.
- The `v6_rules` ledger note still names the old hash `f3bc8ff` (now `34ab9e4`).
- Read `DECISIONS.md` for the choices made without asking.

Rules are in `TASK.md`. Take the first step that is not ticked and not
`BLOCKED`. One line of notes under each step when it is finished. The finish
line described in `TASK.md` goes at the very end of this file, and only when
every step is ticked or blocked.

## Steps

- [x] **1. Commit the work that is already in the tree.**
  Done 13 Oct: seven commits by topic (`3dae3f1` … `584adae`). `benchmarks/mixed_ua_v7.jsonl` committed under the
  exception in `TASK.md`. `docs/HANDOFF.md` left untracked on purpose, see `DECISIONS.md`.
  Run the tests first. Group the modified and untracked files into a few
  commits by topic (v7 build and v6 withdrawal; planner `hybrid_gated` and its
  tests; debug UI; AgentCoMa oracle fix; v7 dev results and analysis; progress
  report and figures). Check every `.jsonl` as `TASK.md` says. Leave
  `benchmarks/mixed_ua_v7.jsonl` out only if it fails that check; record the
  decision.

- [x] **2. Make the test suite green without new dependencies.**
  Done 13 Oct: the kNN test skips when `scikit-learn` is missing; 68 passed, 1 skipped. `hybrid_gated` tests already existed.
  `tests/test_v6_pipeline.py::LearnedRouterTests::test_knn_profile_with_hash_artifacts`
  fails because `scikit-learn` is not in the local venv. Make it skip when the
  module is missing. `hybrid_gated` already has its tests in
  `tests/test_composite_router.py` (`GatedPlannerTests`, six cases); do not add
  more and do not change them. Note: only the **last** step of a multi-step
  plan carries the original request. That is intended (an earlier step that saw
  the whole request did the whole task), and a test asserts it.

- [x] **3. Try the instruct routing fix, offline, on v7 dev only.**
  Done 13 Oct: profile `v3_instruct` added; instruct +0.006 [−0.030, +0.041], macro +0.001 [−0.008, +0.010]:
  not distinguishable, not recommended. `results/analysis/v7_dev_instruct_profile.json`.
  On v7 dev, 122 of 212 instruct items are detected as chat and go to Aya
  (0.539) while Mamay-4B scores 0.583. Add a new rules profile (rules v2 stays
  as it is) that keeps these on Mamay-4B. Estimate it from the stored scores
  in `results/v7_pool/scores_specialists_detail.jsonl`, **dev rows only**: that
  file also holds test items, so first keep the ids with `"split": "dev"` in
  `benchmarks/mixed_ua_v7.jsonl`. Give a paired bootstrap interval against
  rules v2, for instruct and for the macro of four. This is a simulation from
  baseline-prompt scores, not a live run; say so. For reference, the live
  rules v2 router scored 0.553 on dev instruct
  (`results/analysis/v7_dev_router_vs_mamay12.json`).
  Save the result in `results/analysis/` and add a short subsection to the
  progress report. If the interval contains 0, say "not distinguishable" and do
  not recommend the new profile.

- [x] **4. Fix the old v4 numbers in the docs.**
  Done 13 Oct: Mamay-12B 0.874 → 0.884 and FP8 pack 0.842 → 0.852 (both rescored, change is all in knowledge);
  a scorer note at the top of the four docs marks week 5–7 numbers as not rescored; "router wins knowledge" removed.
  `docs/key_takeaways.md`, `docs/thesis_scope.md`,
  `docs/weeks_7_10_takeaways.md` and `docs/conclusion.md` still show scores
  from before the 29 Sep scorer fix (the values 0.874 and 0.842 appear among
  them; check in each doc which system and suite a number belongs to, do not
  assume). Take the current numbers from `results/progress_ledger.csv` and the
  scored summaries, not from memory. Week 6 and week 7 runs were not rescored:
  keep their numbers and mark them "before the scorer fix, not rescored".

- [x] **5. Remove v6 as evidence from the docs.**
  Done 13 Oct: `CLAUDE_CODE_PLAN.md`, `thesis_scope.md`, `benchmarks.md`, `hf_dataset_card.md`, `results/README.md`
  now name v7 and mark v6 withdrawn; `README.md` had no v6 number; `HANDOFF.md` is handled in step 6.
  Search all of `docs/`, `README.md`, `results/README.md` and
  `docs/CLAUDE_CODE_PLAN.md` for v6 numbers and statements ("not
  distinguishable from the specialists", route oracle 0.660, −0.027). Mark each
  as withdrawn with a pointer to `docs/mixed_ua_v6.md`, or replace it with the
  v7 dev number labelled **dev**. Update the ground rules in
  `docs/CLAUDE_CODE_PLAN.md` so they name v7 (dev for tuning, test for claims).

- [x] **6. Bookkeeping.**
  Done 13 Oct: ledger rebuilt (567 rows; v6 runs now `superseded`, two v7 dev runs added as `provisional`);
  `results/README.md` has a line for every results folder; HF card text and `HANDOFF.md` STATE/NEXT updated. Nothing uploaded.
  Rebuild the ledger with `python scripts/build_progress_ledger.py`. Add a
  one-line entry in `results/README.md` for every results folder that has none
  (`v7_pool`, `v7_dev_router`, `v7_dev_router_bucketfs`,
  `v7_dev_mamay12_fewshot`, `v7_confidence`, and any other). Update the text of
  `docs/hf_dataset_card.md` to the current state; do not upload. Rewrite the
  STATE and NEXT lines of `docs/HANDOFF.md` so they match 12 Oct.

- [x] **7. Write the "Ready to freeze" section.**
  Done 13 Oct, with one finding: the v6 jobs are pinned to the v6 file and the AgentCoMa jobs to `max_tokens 256`,
  so no ready `sbatch` line exists for v7 test. The section lists what is missing instead of inventing commands.
  At the end of `docs/progress_report_2026-10-12.md` list: the routing profile,
  planner profile, prompts and few-shot files proposed for freezing, with the
  commit hash from this branch; the open decisions for Maria (use the step 3
  profile or not; escalation to Mamay-12B is already decided: no); and the exact
  `sbatch` commands, one per line, for v7 test (router, three specialists,
  Mamay-12B), AgentCoMa-UK test 150 with **3 repeats** (two runs of the same
  config at T=0 gave different text on 20 of 50 dev items and a different
  score on 3), and Mamay-27B FP8. Note that the lab `~/Diploma` is not a git
  checkout and has only `benchmarks/mixed_ua_v7_dev.jsonl`: list the files
  Maria must `scp` first. Check that each named
  `cluster/*.sbatch` file exists and accepts the v7 test split; if one does
  not, say what is missing instead of inventing a command. Run nothing.

- [x] **8. Close.**
  Done 13 Oct: 69 passed, 1 skipped; no modified tracked files; summary written.
  Run the full test suite, check every "Done means" item in `TASK.md`, write
  the morning summary at the top of this file, commit, and add the finish line.
