"""Optional MLflow logging for scripts/run_composite_router.py and scripts/log_scores_mlflow.py.

The store is a local folder (SQLite file + artifact folder), e.g. logs/<group>/mlflow, so it travels back
from the lab with the logs. It holds prompts and answers, so it stays out of git, like logs/.

One MLflow run per log file (run name = log file stem). One trace per item: the request at the root,
then the planner call(s) and every step (or the single direct call) as child spans with their real
start time and latency, the model, the route reason, the prompt and the answer.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _mlflow():
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        raise SystemExit("--mlflow needs the mlflow package: pip install -r requirements.txt") from None
    return mlflow


def connect(store: Path):
    """Point MLflow at a local store (SQLite file + artifact folder); returns the mlflow module."""
    mlflow = _mlflow()
    store = store.resolve()
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
    return mlflow


def open_store(store: Path, experiment: str):
    """Connect and select (or create) the experiment; returns the mlflow module."""
    mlflow = connect(store)
    store = store.resolve()
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


def git_commit() -> str:
    """Commit hash of the code that ran (the lab copy is not a checkout: then 'unknown')."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, check=True)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _span(mlflow, parent, name: str, call: dict[str, Any], cursor: int, inputs: dict[str, Any],
          attributes: dict[str, Any]) -> int:
    """One LLM call as a child span; returns its end time. Calls logged before 16 Nov 2026 have no
    `started_ns`: they are laid end to end from `cursor`."""
    from mlflow.entities import SpanType  # noqa: PLC0415

    start = int(call.get("started_ns") or cursor)
    end = start + int(float(call.get("latency_ms") or 0) * 1e6)
    span = mlflow.start_span_no_context(name, span_type=SpanType.LLM, parent_span=parent, inputs=inputs,
                                        start_time_ns=start,
                                        attributes={k: v for k, v in attributes.items() if v is not None})
    span.set_outputs({"answer": call.get("content")})
    if not call.get("ok", True):
        span.set_attribute("error", str(call.get("error")))
    span.end(end_time_ns=end)
    return end


def trace_item(mlflow, row: dict[str, Any], start_ns: int, end_ns: int, run_id: str) -> None:
    """Write one item of a run as a trace."""
    from mlflow.entities import SpanType  # noqa: PLC0415

    root = mlflow.start_span_no_context(
        str(row["id"]), span_type=SpanType.CHAIN, inputs={"request": row["prompt"]}, start_time_ns=start_ns,
        metadata={"mlflow.sourceRun": run_id},
        tags={"system": row["system"], "family": str(row.get("family")), "item": str(row["id"]),
              "plan_match": str(row.get("plan_match")), "complete": str(row.get("complete"))},
        attributes={"calls": row.get("calls"), "superfluous_steps": row.get("superfluous_steps"),
                    "missing_steps": row.get("missing_steps")})
    cursor = start_ns
    for i, attempt in enumerate((row.get("planning") or {}).get("attempts") or []):
        cursor = _span(mlflow, root, "planner" if i == 0 else f"planner retry {i}", attempt, cursor,
                       {"model": attempt.get("alias")},
                       {"model": attempt.get("alias"), "usage": attempt.get("usage"),
                        "http_status": attempt.get("http_status")})
    if row.get("plan"):
        root.set_attribute("plan", row["plan"])
    calls = row.get("steps") or ([row["direct"]] if row.get("direct") else [])
    for step in calls:
        route = step.get("route") or {}
        name = f"{step.get('id', 'direct')} · {step.get('intent') or route.get('detected_intent') or '-'} → {step.get('alias')}"
        cursor = _span(mlflow, root, name, step, cursor, {"prompt": step.get("prompt") or row["prompt"]},
                       {"model": step.get("alias"), "depends_on": step.get("depends_on"),
                        "route_reason": route.get("reason"), "usage": step.get("usage"),
                        "http_status": step.get("http_status")})
    root.set_outputs({"final": row.get("content")})
    root.end(end_time_ns=max(end_ns, cursor))


class RunLogger:
    """Run + traces for one log file. Logging errors are printed and never stop the evaluation."""

    def __init__(self, store: Path, experiment: str, run_name: str, params: dict[str, Any]):
        self.mlflow = open_store(store, experiment)
        self.run = self.mlflow.start_run(run_name=run_name, tags={"log_file": run_name})
        self.mlflow.log_params({**params, "git_commit": git_commit()})
        self.failed = 0

    def item(self, row: dict[str, Any], start_ns: int, end_ns: int) -> None:
        try:
            trace_item(self.mlflow, row, start_ns, end_ns, self.run.info.run_id)
        except Exception as exc:  # noqa: BLE001
            self.failed += 1
            if self.failed <= 3:
                print(f"mlflow: trace for {row.get('id')} not written: {type(exc).__name__}: {exc}", file=sys.stderr)

    def finish(self, metrics: dict[str, float]) -> None:
        try:
            self.mlflow.log_metrics(metrics)
        finally:
            self.mlflow.end_run()
