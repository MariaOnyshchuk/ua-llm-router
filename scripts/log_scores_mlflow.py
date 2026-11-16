#!/usr/bin/env python3
"""Attach scores to the MLflow run that scripts/run_composite_router.py --mlflow made for a log file.

    python scripts/log_scores_mlflow.py --mlflow logs/<group>/mlflow \
        --log logs/<group>/<system>_composite_..._rep1_<stamp>.jsonl \
        --scores results/<group>/scores_<system>.json

Logs every number in the system's summary (final_score_mean, plan_exact_match_rate, ...), the final score
per family, and the summary and `*_detail.jsonl` (ids and scores) as artifacts. The run is found by its
name, which is the log file stem. Running it again after a rescore adds a new metric step; the UI shows the latest.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlflow_tracking import connect


def flat_metrics(summary: dict) -> dict[str, float]:
    out = {k: float(v) for k, v in summary.items()
           if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in {"n", "repeats"}}
    for family, values in (summary.get("by_family") or {}).items():
        if isinstance(values.get("final_score"), (int, float)):
            out[f"family.{family}.final_score"] = float(values["final_score"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mlflow", required=True, metavar="DIR", help="store given to run_composite_router --mlflow")
    parser.add_argument("--log", required=True, help="generation log (.jsonl) of the run")
    parser.add_argument("--scores", required=True, help="scores_<system>.json from score_composite_results.py")
    parser.add_argument("--experiment", default="", help="default: any experiment in the store")
    args = parser.parse_args()

    mlflow = connect(Path(args.mlflow))
    client = mlflow.MlflowClient()
    run_name = Path(args.log).stem
    experiments = ([client.get_experiment_by_name(args.experiment).experiment_id] if args.experiment
                   else [e.experiment_id for e in client.search_experiments()])
    runs = client.search_runs(experiments, filter_string=f"tags.log_file = '{run_name}'")
    if len(runs) != 1:
        raise SystemExit(f"expected one MLflow run for {run_name}, found {len(runs)}")
    run_id = runs[0].info.run_id
    system = runs[0].data.params["system"]
    scores_path = Path(args.scores)
    systems = json.loads(scores_path.read_text(encoding="utf-8"))["systems"]
    if system not in systems:
        raise SystemExit(f"{scores_path} has no system {system!r} (has {sorted(systems)})")

    from mlflow.entities import Metric  # noqa: PLC0415

    now = int(time.time() * 1000)
    step = len(client.get_metric_history(run_id, "final_score_mean"))  # a rescore adds a step
    client.log_batch(run_id, metrics=[Metric(k, v, now, step) for k, v in flat_metrics(systems[system]).items()])
    client.log_artifact(run_id, str(scores_path), artifact_path="scores")
    detail = scores_path.with_name(scores_path.stem + "_detail.jsonl")
    if detail.exists():
        client.log_artifact(run_id, str(detail), artifact_path="scores")
    print(f"{run_name}: final_score_mean={systems[system].get('final_score_mean')} logged")


if __name__ == "__main__":
    main()
