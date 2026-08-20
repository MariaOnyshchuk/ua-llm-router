#!/usr/bin/env python3
"""Flatten every scored run into one long-format progress ledger (CSV).

One row per (run, system, bucket, metric). Import the CSV as a Notion database
and every progress chart becomes a filtered view over it.

Two columns exist to stop invalid comparisons:
  * ``comparable_key`` = suite + scorer_epoch. Numbers are only comparable
    within one key; across keys either the items or the scorer changed.
  * ``status`` = confirmed / provisional / superseded / missing. ``missing``
    rows are emitted on purpose so planned-but-unrun cells show up as gaps.

Example:
  python scripts/build_progress_ledger.py --out results/progress_ledger.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUCKETS = ("chat", "instruct", "translate", "knowledge", "code", "alignment")

# How each bucket is scored. Matters for plots: a "full credit" count is
# meaningless for chrF, and ordinal/continuous buckets need CIs, not sd.
SCORE_KIND = {
    "chat": "ordinal_quarter",
    "instruct": "binary_format",
    "translate": "continuous_chrf",
    "knowledge": "binary_label",
    "code": "binary_exec",
    "alignment": "binary_label",
}

# Scorer changed on 2026-08-08 (aya was rescored: code 0.8906 -> 0.9531,
# instruct 0.5625 -> 0.7812). Runs scored before that are not comparable to
# runs scored after it, even on an identical suite. Inferred from artifacts:
# the repo has no git history, so this cannot be read off a commit.
SCORER_PRE = "pre_2026_08_08"
SCORER_POST = "post_2026_08_08"
SCORER_UNKNOWN = "unknown_early"


@dataclass(frozen=True)
class Run:
    """One score artifact, with the context the artifact itself does not record."""

    run_group: str
    date: str
    path: str
    suite: str
    n_repeats: int
    scorer_epoch: str
    status: str
    notes: str = ""
    detail: str | None = None
    systems: tuple[str, ...] = ()  # empty = all systems in the file
    buckets: tuple[str, ...] = ()  # empty = all buckets in the file


# Suite and repeat count are not stored inside the score files, so they are
# declared here and validated against the bucket signature where possible.
REGISTRY: tuple[Run, ...] = (
    Run(
        run_group="week_1_wiring_smoke",
        date="2026-07-25",
        path="results/week_1_wiring_smoke/scores_small_vs_large.json",
        suite="mixed_ua_v0",
        n_repeats=1,
        scorer_epoch=SCORER_UNKNOWN,
        status="superseded",
        notes="Wiring smoke test, unpinned decoding.",
        detail="results/week_1_wiring_smoke/scores_small_vs_large_detail.jsonl",
    ),
    Run(
        run_group="week_1_wiring_smoke",
        date="2026-07-25",
        path="results/week_1_wiring_smoke/scores_with_latency.json",
        suite="mixed_ua_v0",
        n_repeats=1,
        scorer_epoch=SCORER_UNKNOWN,
        status="superseded",
        notes="Wiring smoke test; only artifact covering lapa/router_dual.",
        detail="results/week_1_wiring_smoke/scores_with_latency_detail.jsonl",
    ),
    Run(
        run_group="week_2_v1_and_external",
        date="2026-07-27",
        path="results/week_2_v1_and_external/scores_mixed_ua_v1.json",
        suite="mixed_ua_v1",
        n_repeats=1,
        scorer_epoch=SCORER_UNKNOWN,
        status="superseded",
        notes="Unpinned decoding.",
        detail="results/week_2_v1_and_external/scores_mixed_ua_v1_detail.jsonl",
    ),
    Run(
        run_group="week_3_v2_and_pinned_v3",
        date="2026-07-28",
        path="results/week_3_v2_and_pinned_v3/scores_mixed_ua_v2.json",
        suite="mixed_ua_v2",
        n_repeats=1,
        scorer_epoch=SCORER_UNKNOWN,
        status="superseded",
        notes="Unpinned decoding.",
        detail="results/week_3_v2_and_pinned_v3/scores_mixed_ua_v2_detail.jsonl",
    ),
    Run(
        run_group="week_3_v2_and_pinned_v3",
        date="2026-07-29",
        path="results/week_3_v2_and_pinned_v3/scores_mamay4_t0.json",
        suite="mixed_ua_v2",
        n_repeats=1,
        scorer_epoch=SCORER_UNKNOWN,
        status="superseded",
        notes="First pinned run (T=0, seed=42); single system.",
        detail="results/week_3_v2_and_pinned_v3/scores_mamay4_t0_detail.jsonl",
    ),
    Run(
        run_group="week_4_specialist_bakeoff",
        date="2026-08-03",
        path="results/week_4_specialist_bakeoff/scores_mean_sd_bakeoff.json",
        suite="mixed_ua_v3",
        n_repeats=3,
        scorer_epoch=SCORER_PRE,
        status="confirmed",
        notes=(
            "Authoritative v3 bake-off. Superset of scores_mean_sd_v3.json. "
            "No per-item detail kept, so no bootstrap CI is possible."
        ),
    ),
    Run(
        run_group="week_5_aya_qwen7",
        date="2026-08-07",
        path="results/week_5_aya_qwen7/scores_mean_sd_aya_v4.json",
        suite="mixed_ua_v4_balanced",
        n_repeats=3,
        scorer_epoch=SCORER_PRE,
        status="superseded",
        notes="Superseded by scores_mean_sd_aya_v4_rescored.json.",
    ),
    Run(
        run_group="week_5_aya_qwen7",
        date="2026-08-07",
        path="results/week_5_aya_qwen7/scores_mean_sd_qwen7_code_v4.json",
        suite="mixed_ua_v4_balanced",
        n_repeats=3,
        scorer_epoch=SCORER_PRE,
        status="superseded",
        notes=(
            "Code bucket only. Same generations as the router code bucket "
            "(identical p50) but old scorer: 0.9062 vs 0.9688. Needs rescoring."
        ),
        buckets=("code",),
    ),
    Run(
        run_group="week_5_router_v4",
        date="2026-08-08",
        path="results/week_5_router_v4/scores_mean_sd_aya_v4_rescored.json",
        suite="mixed_ua_v4_balanced",
        n_repeats=3,
        scorer_epoch=SCORER_POST,
        status="confirmed",
        notes="Rescored baseline; the one comparable to the router run.",
    ),
    Run(
        run_group="week_5_router_v4",
        date="2026-08-08",
        path="results/week_5_router_v4/scores_mean_sd_router_v4.json",
        suite="mixed_ua_v4_balanced",
        n_repeats=3,
        scorer_epoch=SCORER_POST,
        status="provisional",
        notes="Run before the intent_rules.py leakage fix; needs a re-run.",
        detail=(
            "results/week_5_router_v4/scored/"
            "router_matrix_mixed_ua_v4_balanced_t0_s42_rep1_20260807T121444Z"
            ".scores_detail.jsonl"
        ),
    ),
    Run(
        run_group="week_5_alignment_bakeoff",
        date="2026-08-08",
        path="results/week_5_alignment_bakeoff/scores_alignment_bakeoff.json",
        suite="mixed_ua_v4_balanced",
        n_repeats=1,
        scorer_epoch=SCORER_POST,
        status="superseded",
        notes=(
            "Baseline prompts only. Superseded by scores_alignment_variants_flat.json, "
            "which also covers the clarified/few-shot variants."
        ),
        detail="results/week_5_alignment_bakeoff/scores_alignment_bakeoff_detail.jsonl",
        buckets=("alignment",),
    ),
    Run(
        run_group="week_5_alignment_bakeoff",
        date="2026-08-08",
        path="results/week_5_alignment_bakeoff/scores_alignment_variants_flat.json",
        suite="mixed_ua_v4_balanced",
        n_repeats=2,
        scorer_epoch=SCORER_POST,
        status="confirmed",
        notes=(
            "Alignment subset; system name carries the prompt variant. Identical "
            "across both repeats, ethics control unchanged. The social reference=1 "
            "collapse is fixed (McNemar p<0.01), but the bucket-wide gain is not "
            "significant at n=32 because the variants add a 2->1 error. "
            "See paired_vs_baseline in scores_alignment_variants.json."
        ),
        detail="results/week_5_alignment_bakeoff/scores_alignment_variants_detail.jsonl",
        buckets=("alignment",),
    ),
)

# Artifacts deliberately left out, recorded so the omission is auditable.
SKIPPED: tuple[tuple[str, str], ...] = (
    ("results/week_1_wiring_smoke/scores_dual.json", "duplicate v0 numbers"),
    ("results/week_1_wiring_smoke/scores_s0_s1.json", "duplicate v0 numbers"),
    ("results/week_1_wiring_smoke/scores_s0_s1_s2.json", "duplicate v0 numbers"),
    (
        "results/week_3_v2_and_pinned_v3/scores_mean_sd_v3.json",
        "subset of week_4_specialist_bakeoff, identical values",
    ),
    (
        "results/week_3_v2_and_pinned_v3/efficiency_table_v2.json",
        "gpu_seconds_per_prompt equals latency_avg_ms/1000 and peak_vram_gb is null, "
        "so it adds no independent resource axis",
    ),
    (
        "results/week_4_specialist_bakeoff/specialist_matrix_bakeoff.json",
        "routing advice derived from the bake-off scores, not measurements",
    ),
    (
        "results/week_3_v2_and_pinned_v3/specialist_matrix_v3.json",
        "routing advice derived from scores, not measurements",
    ),
)

# Runs the thesis needs but which do not exist yet. Emitted as status=missing
# rows so a Notion coverage view renders them as visible holes.
PLANNED_MISSING: tuple[tuple[str, str, str], ...] = (
    ("mamay4", "mixed_ua_v4_balanced", "Needed for router-vs-default on one suite."),
    ("lapa", "mixed_ua_v4_balanced", "Needed for router-vs-best-specialist on one suite."),
    ("mamay12", "mixed_ua_v4_balanced", "Needed for router-vs-large-model on one suite."),
)


def load_suites(bench_dir: Path) -> dict[str, dict[str, int]]:
    """Bucket -> item count for every suite, used to validate the registry."""
    suites: dict[str, dict[str, int]] = {}
    for path in sorted(bench_dir.glob("*.jsonl")):
        counts: dict[str, int] = defaultdict(int)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            counts[json.loads(line).get("bucket", "?")] += 1
        suites[path.stem] = dict(counts)
    return suites


def bootstrap_ci(
    scores: list[float], n_boot: int = 10_000, seed: int = 42
) -> tuple[float | None, float | None]:
    """Percentile bootstrap over items — the only honest interval here.

    sd across repeats is ~0 because decoding is pinned, so it measures
    determinism, not uncertainty about the mean.
    """
    if len(scores) < 2:
        return None, None
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_boot):
        means.append(sum(scores[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return round(lo, 4), round(hi, 4)


def read_detail(path: Path) -> dict[tuple[str, str], list[float]]:
    """(system, bucket) -> per-item scores, plus (system, 'overall') pools."""
    pools: dict[tuple[str, str], list[float]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        system, bucket = row.get("system"), row.get("bucket")
        score = row.get("score")
        if system is None or bucket is None or score is None:
            continue
        pools[(system, bucket)].append(float(score))
        pools[(system, "overall_micro")].append(float(score))
    return pools


def parse_flat(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Single-run schema: values are plain numbers."""
    out: dict[str, dict[str, Any]] = {}
    for system, s in (payload.get("systems") or {}).items():
        metrics: dict[str, Any] = {
            "overall_micro": (s.get("overall_mean"), None, s.get("n")),
            "latency_p50_ms": (s.get("latency_p50_ms"), None, s.get("n")),
            "latency_avg_ms": (s.get("latency_avg_ms"), None, s.get("n")),
        }
        if s.get("overall_mean_macro") is not None:
            metrics["overall_macro"] = (s["overall_mean_macro"], None, s.get("n"))
        if s.get("gpu_seconds_per_prompt") is not None:
            metrics["gpu_seconds_per_prompt"] = (
                s["gpu_seconds_per_prompt"],
                None,
                s.get("n"),
            )
        buckets: dict[str, dict[str, Any]] = {}
        for bucket, b in (s.get("by_bucket") or {}).items():
            if isinstance(b, dict):
                buckets[bucket] = {
                    "quality": (b.get("quality_mean"), None, b.get("n")),
                    "latency_p50_ms": (b.get("latency_p50_ms"), None, b.get("n")),
                }
            else:  # earliest week-1 files stored a bare quality float
                buckets[bucket] = {"quality": (b, None, None)}
        out[system] = {"metrics": metrics, "buckets": buckets}
    return out


def parse_mean_sd(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Aggregated schema: {metric: {mean, sd, n, values}}."""
    out: dict[str, dict[str, Any]] = {}
    for system, s in (payload.get("systems") or {}).items():

        def cell(key: str) -> tuple[Any, Any, None] | None:
            entry = s.get(key)
            if not isinstance(entry, dict):
                return None
            return (entry.get("mean"), entry.get("sd"), None)

        metrics = {
            name: value
            for name, key in (
                ("overall_micro", "overall_mean"),
                ("overall_macro", "overall_mean_macro"),
                ("latency_p50_ms", "latency_p50_ms"),
                ("latency_avg_ms", "latency_avg_ms"),
            )
            if (value := cell(key)) is not None
        }
        buckets: dict[str, dict[str, Any]] = {}
        for key in s:
            if not key.startswith("bucket."):
                continue
            _, bucket, metric = key.split(".", 2)
            name = "latency_p50_ms" if metric == "latency_p50" else metric
            if (value := cell(key)) is not None:
                buckets.setdefault(bucket, {})[name] = value
        out[system] = {"metrics": metrics, "buckets": buckets}
    return out


def build_rows(root: Path, suites: dict[str, dict[str, int]]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for run in REGISTRY:
        path = root / run.path
        if not path.exists():
            warnings.append(f"missing artifact: {run.path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        parsed = (
            parse_mean_sd(payload) if "n_repeats" in payload else parse_flat(payload)
        )
        pools = read_detail(root / run.detail) if run.detail else {}
        suite_sig = suites.get(run.suite, {})
        suite_items = sum(suite_sig.values())

        for system, data in parsed.items():
            if run.systems and system not in run.systems:
                continue
            covered = run.buckets or tuple(data["buckets"]) or ()
            for bucket in (*covered, "overall" if not run.buckets else ""):
                if bucket:
                    seen.add((system, run.suite, bucket))

            def emit(
                bucket: str,
                metric: str,
                cell: tuple[Any, Any, Any],
                pool_key: str | None = None,
            ) -> None:
                value, sd, n_items = cell
                if value is None:
                    return
                ci_low = ci_high = None
                ci_method = ""
                pool = pools.get((system, pool_key)) if pool_key else None
                if pool:
                    ci_low, ci_high = bootstrap_ci(pool)
                    ci_method = "bootstrap_items_percentile_95"
                    n_items = n_items or len(pool)
                if n_items is None and bucket in suite_sig:
                    n_items = suite_sig[bucket]
                elif n_items is None and bucket.startswith("overall"):
                    n_items = suite_items or None
                rows.append(
                    {
                        "run_group": run.run_group,
                        "run_date": run.date,
                        "system": system,
                        "suite": run.suite,
                        "suite_items": suite_items or "",
                        "n_repeats": run.n_repeats,
                        "scorer_epoch": run.scorer_epoch,
                        "comparable_key": f"{run.suite}|{run.scorer_epoch}",
                        "scope": "bucket_subset" if run.buckets else "full_suite",
                        "status": run.status,
                        "bucket": bucket,
                        "metric": metric,
                        "score_kind": SCORE_KIND.get(bucket, ""),
                        "value": value,
                        "sd_across_repeats": "" if sd is None else sd,
                        "ci_low": "" if ci_low is None else ci_low,
                        "ci_high": "" if ci_high is None else ci_high,
                        "ci_method": ci_method,
                        "n_items": n_items or "",
                        "artifact": run.path,
                        "notes": run.notes,
                    }
                )

            # A single-bucket run has no meaningful "overall".
            if not run.buckets:
                for metric, cell in data["metrics"].items():
                    pool_key = "overall_micro" if metric == "overall_micro" else None
                    emit("overall", metric, cell, pool_key)

            for bucket, metrics in sorted(data["buckets"].items()):
                if run.buckets and bucket not in run.buckets:
                    continue
                if suite_sig and bucket not in suite_sig:
                    warnings.append(
                        f"{run.path}: bucket {bucket!r} not in suite {run.suite}"
                    )
                for metric, cell in metrics.items():
                    emit(bucket, metric, cell, bucket if metric == "quality" else None)

            observed = {
                b: m.get("quality", (None, None, None))[2]
                for b, m in data["buckets"].items()
            }
            if all(v is not None for v in observed.values()) and observed:
                declared = {b: suite_sig.get(b) for b in observed}
                if observed != declared and not run.buckets:
                    warnings.append(
                        f"{run.path} [{system}]: bucket sizes {observed} do not match "
                        f"declared suite {run.suite} {declared}"
                    )

    for system, suite, note in PLANNED_MISSING:
        suite_sig = suites.get(suite, {})
        for bucket in ("overall", *BUCKETS):
            if (system, suite, bucket) in seen:
                continue
            rows.append(
                {
                    "run_group": "",
                    "run_date": "",
                    "system": system,
                    "suite": suite,
                    "suite_items": sum(suite_sig.values()) or "",
                    "n_repeats": 0,
                    "scorer_epoch": SCORER_POST,
                    "comparable_key": f"{suite}|{SCORER_POST}",
                    "scope": "full_suite",
                    "status": "missing",
                    "bucket": bucket,
                    "metric": "quality" if bucket != "overall" else "overall_macro",
                    "score_kind": SCORE_KIND.get(bucket, ""),
                    "value": "",
                    "sd_across_repeats": "",
                    "ci_low": "",
                    "ci_high": "",
                    "ci_method": "",
                    "n_items": suite_sig.get(bucket, "") or "",
                    "artifact": "",
                    "notes": note,
                }
            )
    return rows, warnings


def add_success_counts(root: Path, rows: list[dict]) -> None:
    """Per-bucket full-credit / zero counts — the clearest week-over-week tracker."""
    path = root / "results/week_5_router_v4/success_stats_rep1.json"
    if not path.exists():
        return
    stats = json.loads(path.read_text(encoding="utf-8"))
    for bucket, s in (stats.get("by_bucket") or {}).items():
        kind = SCORE_KIND.get(bucket, "")
        for metric, key in (("full_credit_count", "full"), ("zero_score_count", "zero")):
            note = "Counts are meaningless for chrF-scored buckets."
            rows.append(
                {
                    "run_group": "week_5_router_v4",
                    "run_date": "2026-08-08",
                    "system": "router_matrix",
                    "suite": "mixed_ua_v4_balanced",
                    "suite_items": 192,
                    "n_repeats": 1,
                    "scorer_epoch": SCORER_POST,
                    "comparable_key": f"mixed_ua_v4_balanced|{SCORER_POST}",
                    "scope": "full_suite",
                    "status": "provisional" if kind != "continuous_chrf" else "invalid",
                    "bucket": bucket,
                    "metric": metric,
                    "score_kind": kind,
                    "value": s.get(key),
                    "sd_across_repeats": "",
                    "ci_low": "",
                    "ci_high": "",
                    "ci_method": "",
                    "n_items": s.get("n", ""),
                    "artifact": "results/week_5_router_v4/success_stats_rep1.json",
                    "notes": note if kind == "continuous_chrf" else "rep1 only.",
                }
            )


FIELDS = (
    "run_group",
    "run_date",
    "system",
    "suite",
    "suite_items",
    "n_repeats",
    "scorer_epoch",
    "comparable_key",
    "scope",
    "status",
    "bucket",
    "metric",
    "score_kind",
    "value",
    "sd_across_repeats",
    "ci_low",
    "ci_high",
    "ci_method",
    "n_items",
    "artifact",
    "notes",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--out", type=Path, default=Path("results/progress_ledger.csv"))
    args = p.parse_args()

    suites = load_suites(args.root / "benchmarks")
    rows, warnings = build_rows(args.root, suites)
    add_success_counts(args.root, rows)

    out = args.out if args.out.is_absolute() else args.root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    coverage: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row["metric"] == "quality" and row["status"] not in ("superseded", "missing"):
            coverage[(row["comparable_key"], row["scope"], row["system"])].add(row["bucket"])
    print(f"{'comparable_key':38} {'system':16} buckets scored")
    for (key, scope, system), buckets in sorted(coverage.items()):
        gaps = [b for b in BUCKETS if b not in buckets]
        if scope == "bucket_subset":
            tail = f"  (ablation cell, {'/'.join(sorted(buckets))} only)"
        else:
            tail = f"  MISSING: {', '.join(gaps)}" if gaps else ""
        print(f"{key:38} {system:16} {len(buckets)}/6{tail}")
    print()
    for path, reason in SKIPPED:
        print(f"skipped  {path}  ({reason})")
    for warning in warnings:
        print(f"WARNING  {warning}")
    print(json.dumps({"wrote": str(out), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
