# scripts/ map

Run everything from the repo root. `python scripts/<name>.py --help` for options.

| Folder | What is there |
|---|---|
| `scripts/` (top level) | Live pipeline: run models and routers (`run_small_router_cluster.py`, `run_composite_router.py`), score (`score_results.py`, `score_repeat_dir.py`, `aggregate_repeat_scores.py`, `score_composite_results.py`), v6 suite build and evaluation (`sample_screening_suite.py`, `curate_discriminative_suite.py`, `extract_full_corpora.py`, `extract_ua_leaderboard.py`, `week_v6_eval_pipeline.sh`, `build_router_comparison.py`, `train_learned_router.py`), ledger and publishing (`build_progress_ledger.py`, `publish_hf_eval.py`), AgentCoMa step questions alone (`run_agentcoma_steps.py`), debug UI without GPUs (`mock_backends.py`; then `uvicorn router.app:app --port 4010` and open `/debug`), prompt variants (`alignment_prompt_variants.py`, `bucket_prompt_variants.py`), helpers (`mcq_format.py`, `ifeval_check.py`, `sample_size_ci.py`) |
| `scripts/analysis/` | Statistics and plots for the text: `bootstrap_compare.py` (paired bootstrap intervals, v4 only), `v6_compare.py` (v6 paired bootstrap per bucket, route oracle, parse failures), `estimate_route_table.py`, `plot_tradeoff.py` |
| `scripts/benchmarks/` | Build of the small v0 to v4 suites, run once: `sample_*.py`, `build_balanced_suite.py`, `merge_benchmarks.py`, `import_english_code.py`, `check_router_leakage.py` |
| `scripts/archive/` | Superseded or one-off scripts from weeks 1 to 9. Kept for history, not part of the pipeline. Paths inside them may be stale |

Cluster jobs are in `cluster/`. The claim suite is v6; see `AGENTS.md`, section Evidence rules.

