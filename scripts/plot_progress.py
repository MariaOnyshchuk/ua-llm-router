#!/usr/bin/env python3
"""Render the progress charts from results/progress_ledger.csv.

Every chart filters the ledger to a single ``comparable_key`` (suite + scorer
version), because rows from different keys are not comparable. Error bars are
drawn only where the ledger carries a bootstrap CI; ``sd_across_repeats`` is
never used as an error bar since pinned decoding makes it ~0.

Example:
  .venv/bin/python scripts/plot_progress.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BUCKETS = ("chat", "instruct", "translate", "knowledge", "code", "alignment")
V3 = "mixed_ua_v3|pre_2026_08_08"
V4 = "mixed_ua_v4_balanced|post_2026_08_08"
LIVE = ("confirmed", "provisional")

PALETTE = ["#4C6EF5", "#F76707", "#37B24D", "#AE3EC9", "#F59F00", "#1098AD", "#E03131"]
STATUS_COLORS = {
    "confirmed": "#37B24D",
    "provisional": "#F59F00",
    "missing": "#E9ECEF",
    "superseded": "#CED4DA",
    "invalid": "#E03131",
}

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pick(
    rows: list[dict[str, str]], live_only: bool = False, **filters: Any
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if live_only and row["status"] not in LIVE:
            continue
        if all(row.get(k) == v for k, v in filters.items()):
            out.append(row)
    return out


def legend_below(ax: plt.Axes, ncol: int) -> None:
    ax.legend(
        fontsize=8,
        ncol=ncol,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
    )


def annotate_source(fig: plt.Figure, text: str) -> None:
    fig.text(0.005, 0.005, text, fontsize=6, color="#868E96", va="bottom")


def plot_bakeoff(rows: list[dict[str, str]], out: Path) -> None:
    """v3 bake-off: the only fully populated comparison in the ledger."""
    data = pick(rows, live_only=True, comparable_key=V3, metric="quality", scope="full_suite")
    systems = sorted({r["system"] for r in data})
    values = {(r["system"], r["bucket"]): num(r["value"]) for r in data}

    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.8 / len(systems)
    for i, system in enumerate(systems):
        offsets = [b + i * width - 0.4 + width / 2 for b in range(len(BUCKETS))]
        ax.bar(
            offsets,
            [values.get((system, b), 0) or 0 for b in BUCKETS],
            width,
            label=system,
            color=PALETTE[i % len(PALETTE)],
        )
    ax.set_xticks(range(len(BUCKETS)), BUCKETS)
    ax.set_ylabel("quality")
    ax.set_ylim(0, 1.08)
    ax.set_title("Specialist bake-off per bucket — mixed_ua_v3, 3 pinned repeats")
    legend_below(ax, ncol=6)
    annotate_source(
        fig,
        "No CIs: per-item scores were not kept for v3, so bars are point estimates only.",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_router_vs_aya(rows: list[dict[str, str]], out: Path) -> None:
    """The only same-suite, same-scorer comparison available."""
    data = pick(rows, live_only=True, comparable_key=V4, metric="quality", scope="full_suite")
    systems = sorted({r["system"] for r in data})
    by = {(r["system"], r["bucket"]): r for r in data}

    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.8 / len(systems)
    for i, system in enumerate(systems):
        offsets = [b + i * width - 0.4 + width / 2 for b in range(len(BUCKETS))]
        heights, lo, hi = [], [], []
        for bucket in BUCKETS:
            row = by.get((system, bucket))
            value = num(row["value"]) if row else 0.0
            heights.append(value or 0.0)
            low, high = (num(row["ci_low"]), num(row["ci_high"])) if row else (None, None)
            lo.append((value - low) if (low is not None and value is not None) else 0.0)
            hi.append((high - value) if (high is not None and value is not None) else 0.0)
        ax.bar(
            offsets,
            heights,
            width,
            label=system,
            color=PALETTE[i % len(PALETTE)],
            yerr=[lo, hi] if any(lo) or any(hi) else None,
            capsize=2.5,
            error_kw={"lw": 0.9, "ecolor": "#343A40"},
        )
    ax.set_xticks(range(len(BUCKETS)), BUCKETS)
    ax.set_ylabel("quality")
    ax.set_ylim(0, 1.15)
    ax.set_title("Router vs Aya — mixed_ua_v4_balanced, 32 items/bucket")
    legend_below(ax, ncol=2)
    annotate_source(
        fig,
        "Error bars = 95% bootstrap CI over items (router only; Aya per-item scores "
        "were not saved). At n=32 a bucket CI spans ~±0.15.",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(rows: list[dict[str, str]], out: Path) -> None:
    """Alignment prompt ablation — the only chart with CIs and a control."""
    data = pick(rows, scope="bucket_subset", metric="quality", bucket="alignment")
    data = [r for r in data if r["status"] in LIVE]
    order = [
        "lapa@baseline",
        "lapa@clarified",
        "lapa@fewshot",
        "mamay4@baseline",
        "mamay4@fewshot",
    ]
    by = {r["system"]: r for r in data}
    cells = [s for s in order if s in by]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    values = [num(by[c]["value"]) or 0 for c in cells]
    lo = [(num(by[c]["value"]) or 0) - (num(by[c]["ci_low"]) or 0) for c in cells]
    hi = [(num(by[c]["ci_high"]) or 0) - (num(by[c]["value"]) or 0) for c in cells]
    colors = ["#ADB5BD" if c.endswith("baseline") else "#4C6EF5" for c in cells]
    labels = [c.replace("@", "\n") for c in cells]
    ax.bar(
        labels,
        values,
        color=colors,
        yerr=[lo, hi],
        capsize=3,
        error_kw={"lw": 0.9, "ecolor": "#343A40"},
    )
    for x, value in enumerate(values):
        ax.text(x, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("alignment quality")
    ax.set_ylim(0, 1.05)
    ax.set_title("Alignment prompt ablation — mixed_ua_v4_balanced, 32 items")
    ax.legend(
        handles=[
            Patch(color="#ADB5BD", label="baseline prompt"),
            Patch(color="#4C6EF5", label="rewritten social prompt"),
        ],
        fontsize=8,
        frameon=False,
    )
    annotate_source(
        fig,
        "Ethics control unchanged at 0.800 in all five cells; both repeats identical. "
        "Bucket-wide deltas are not significant at n=32, but the social reference=1 "
        "collapse is fixed (McNemar p<0.01).",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_frontier(rows: list[dict[str, str]], out: Path) -> None:
    """Quality vs median latency. Stands in for the missing cost axis."""
    fig, ax = plt.subplots(figsize=(7, 4)) 
    for key, marker, label in ((V3, "o", "mixed_ua_v3"), (V4, "s", "mixed_ua_v4_balanced")):
        quality = {
            r["system"]: num(r["value"])
            for r in pick(rows, comparable_key=key, metric="overall_macro", scope="full_suite")
            if r["status"] in LIVE
        }
        latency = {
            r["system"]: num(r["value"])
            for r in pick(
                rows,
                comparable_key=key,
                metric="latency_p50_ms",
                bucket="overall",
                scope="full_suite",
            )
            if r["status"] in LIVE
        }
        shared = sorted(set(quality) & set(latency))
        if not shared:
            continue
        ax.scatter(
            [latency[s] for s in shared],
            [quality[s] for s in shared],
            s=70,
            marker=marker,
            label=label,
            zorder=3,
        )
        # Systems can land on the same point (router_small and router_cascade are
        # identical on v3); label them once as a shared point rather than overprint.
        collapsed: dict[tuple[int, float], list[str]] = defaultdict(list)
        for system in shared:
            collapsed[(round(latency[system] / 20), round(quality[system], 3))].append(system)
        for names in collapsed.values():
            ax.annotate(
                " / ".join(sorted(names)),
                (latency[names[0]], quality[names[0]]),
                textcoords="offset points",
                xytext=(9, -3),
                fontsize=7.5,
            )
    ax.margins(x=0.22)  # room for right-hand point labels
    ax.set_xlabel("median latency p50 (ms) — lower is better")
    ax.set_ylabel("macro quality — higher is better")
    ax.set_title("Quality vs latency (NOT a cost frontier: no GPU-s or VRAM measured)")
    ax.legend(fontsize=8, frameon=False, title="separate suites, do not compare across")
    annotate_source(
        fig,
        "The two suites differ in items AND scorer version, so the two marker groups "
        "are independent pictures plotted on shared axes for layout only.",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_failure_profile(rows: list[dict[str, str]], out: Path) -> None:
    """Full-credit / partial / zero split per bucket — the week-over-week tracker."""
    full = {r["bucket"]: r for r in pick(rows, metric="full_credit_count")}
    zero = {r["bucket"]: r for r in pick(rows, metric="zero_score_count")}
    buckets = [b for b in BUCKETS if b in full and full[b]["status"] in LIVE]
    excluded = [b for b in BUCKETS if b in full and full[b]["status"] == "invalid"]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    n = [num(full[b]["n_items"]) or 0 for b in buckets]
    fulls = [num(full[b]["value"]) or 0 for b in buckets]
    zeros = [num(zero[b]["value"]) or 0 for b in buckets]
    partial = [t - f - z for t, f, z in zip(n, fulls, zeros)]

    ax.bar(buckets, fulls, color="#37B24D", label="full credit")
    ax.bar(buckets, partial, bottom=fulls, color="#F59F00", label="partial")
    ax.bar(
        buckets,
        zeros,
        bottom=[f + p for f, p in zip(fulls, partial)],
        color="#E03131",
        label="zero",
    )
    for x, z in enumerate(zeros):
        if z:
            ax.text(x, n[x] + 0.6, f"{int(z)} zeros", ha="center", fontsize=8, color="#E03131")
    ax.set_ylabel("items (out of 32)")
    ax.set_ylim(0, 37)
    ax.set_title("Router failure profile per bucket — mixed_ua_v4_balanced, rep1")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="lower right")
    annotate_source(
        fig,
        f"Excluded: {', '.join(excluded) or 'none'} — chrF-scored, so 'full credit' is "
        "not a meaningful count. Zero counts are integers on a fixed suite, which makes "
        "them the cleanest thing to track week over week.",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(rows: list[dict[str, str]], out: Path) -> None:
    """What exists vs what the thesis still needs, on the v4 suite."""
    grid: dict[str, dict[str, str]] = defaultdict(dict)
    for row in pick(rows, comparable_key=V4, scope="full_suite"):
        if row["metric"] not in ("quality", "overall_macro"):
            continue
        bucket = row["bucket"]
        current = grid[row["system"]].get(bucket)
        rank = {"confirmed": 3, "provisional": 2, "invalid": 1, "missing": 0}
        if current is None or rank.get(row["status"], 0) > rank.get(current, 0):
            grid[row["system"]][bucket] = row["status"]

    systems = sorted(grid)
    columns = ("overall", *BUCKETS)
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(systems) + 1.9))
    for y, system in enumerate(systems):
        for x, column in enumerate(columns):
            status = grid[system].get(column, "missing")
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=STATUS_COLORS.get(status, "#E9ECEF"),
                    edgecolor="white",
                    lw=2,
                )
            )
            if status == "missing":
                ax.text(x + 0.5, y + 0.5, "—", ha="center", va="center", color="#868E96")
    ax.set_xlim(0, len(columns))
    ax.set_ylim(0, len(systems))
    ax.set_xticks([x + 0.5 for x in range(len(columns))], columns, fontsize=8)
    ax.set_yticks([y + 0.5 for y in range(len(systems))], systems, fontsize=8)
    ax.grid(False)
    ax.set_title("Coverage on mixed_ua_v4_balanced — the actual progress tracker")
    ax.legend(
        handles=[
            Patch(color=STATUS_COLORS["confirmed"], label="confirmed"),
            Patch(color=STATUS_COLORS["provisional"], label="provisional"),
            Patch(color=STATUS_COLORS["missing"], label="not run"),
        ],
        fontsize=8,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    annotate_source(
        fig,
        "Green cells turning up week over week is the honest progress metric; a quality "
        "line chart is not possible while suites and scorers keep changing.",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", type=Path, default=Path("results/progress_ledger.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    args = p.parse_args()

    rows = load(args.ledger)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    charts = (
        ("01_bakeoff_v3_per_bucket.png", plot_bakeoff),
        ("02_router_vs_aya_v4.png", plot_router_vs_aya),
        ("03_alignment_ablation.png", plot_ablation),
        ("04_quality_vs_latency.png", plot_frontier),
        ("05_failure_profile.png", plot_failure_profile),
        ("06_coverage_v4.png", plot_coverage),
    )
    for name, fn in charts:
        fn(rows, args.out_dir / name)
        print(f"wrote {args.out_dir / name}")


if __name__ == "__main__":
    main()
