#!/usr/bin/env python3
"""Figures for the RouterBench-style evaluation and rules v2 on v7 dev.

Reads results/routerbench_v7dev/{summary.json,curves.csv} and
results/analysis/v7_dev_router_vs_mamay12.json; writes PNGs to docs/figures/.

    .venv/bin/python scripts/analysis/plot_routerbench_dev.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs/figures"

# Reference palette, first three categorical slots (pass all-pairs CVD checks).
MODEL_COLOR = {"mamay4": "#2a78d6", "lapa": "#eb6834", "aya": "#1baf7a"}
MODEL_NAME = {"mamay4": "Mamay-4B", "lapa": "Lapa", "aya": "Aya"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BUCKETS = ("knowledge", "translate", "alignment", "instruct")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def ndch(points):
    import importlib.util

    spec = importlib.util.spec_from_file_location("routerbench_dev", ROOT / "scripts/analysis/routerbench_dev.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ndch(points)


def cost_quality(summary: dict, curves: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    singles = [(summary["points"][m]["cost_s"], summary["points"][m]["macro"]) for m in MODEL_COLOR]
    zero = ndch(singles)
    c_max = summary["cost_range"][1]
    zx = [p[0] for p in zero] + [c_max]
    zy = [p[1] for p in zero] + [zero[-1][1]]
    ax.plot(zx, zy, color=MUTED, lw=2, ls=(0, (4, 3)), label=f"Zero router (AIQ {summary['aiq']['zero_router']:.3f})")
    for kind, style, name in (("mlp", "-", "MLP"), ("knn", (0, (1, 1.5)), "kNN")):
        pts = sorted({(float(r["cost_s"]), float(r["macro"])) for r in curves if r["system"] == kind})
        hull = ndch(pts)
        ax.plot([p[0] for p in hull], [p[1] for p in hull], color=INK, lw=2, ls=style,
                label=f"{name} router (AIQ {summary['aiq'][kind]:.3f})")
    for m, (c, q) in zip(MODEL_COLOR, singles):
        ax.scatter(c, q, s=70, color=MODEL_COLOR[m], edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.annotate(MODEL_NAME[m], (c, q), xytext=(6, -12), textcoords="offset points", color=INK2)
    for key, name, dy in (("rules_v2_sim", "rules v2", 8), ("bucket_oracle", "best per bucket", 8),
                          ("item_oracle", "oracle per item", -14)):
        c, q = summary["points"][key]["cost_s"], summary["points"][key]["macro"]
        ax.scatter(c, q, s=60, marker="D" if key == "rules_v2_sim" else "s", color=SURFACE, edgecolor=INK,
                   linewidth=1.6, zorder=5)
        ax.annotate(f"{name} {q:.3f}", (c, q), xytext=(6, dy), textcoords="offset points", color=INK2)
    ax.set_xlabel("mean latency of the chosen model, s per request")
    ax.set_ylabel("macro quality (4 buckets)")
    ax.set_title("Cost–quality on v7 dev (1423 items, 5-fold CV)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "routerbench_v7dev_cost_quality.png", dpi=160)
    plt.close(fig)


def rules_vs_mamay12(diff: dict) -> None:
    rows = [("macro", diff["rows"]["macro4"])] + [(b, diff["rows"][b]) for b in BUCKETS]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for y, (name, r) in enumerate(reversed(rows)):
        lo, hi = r["ci95"]
        ax.plot([lo, hi], [y, y], color=INK2, lw=2, solid_capstyle="round")
        ax.scatter(r["diff"], y, s=60, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{r['diff']:+.3f}  [{lo:+.3f}, {hi:+.3f}]", (hi, y), xytext=(8, -3),
                    textcoords="offset points", color=INK2, fontsize=9)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [f"{n} (n={r['n']})" for n, r in reversed(rows)])
    ax.set_xlim(-0.26, 0.16)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("rules v2 − Mamay-12B, with 95% paired bootstrap interval")
    ax.set_title("Rules v2 against Mamay-12B on v7 dev: the gap is instruct", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "rules_v2_vs_mamay12_v7dev.png", dpi=160)
    plt.close(fig)


def rules_routing() -> None:
    from router.intent_rules import route_intent

    rows = [json.loads(line) for line in (ROOT / "benchmarks/tasks_public_4401.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    order = ("knowledge", "translate", "alignment", "instruct", "code", "chat")
    count = Counter((r["bucket"], route_intent(r["prompt"], profile="v2").model) for r in rows if r["split"] == "dev")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for y, b in enumerate(reversed(order)):
        total = sum(count[(b, m)] for m in MODEL_COLOR)
        left = 0.0
        for m, color in MODEL_COLOR.items():
            share = count[(b, m)] / total
            if share:
                ax.barh(y, share, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.62)
                if share >= 0.12:
                    ax.text(left + share / 2, y, str(count[(b, m)]), ha="center", va="center", color="#ffffff",
                            fontsize=9, fontweight="bold")
            left += share
    ax.set_yticks(range(len(order)), [f"{b} (n={sum(count[(b, m)] for m in MODEL_COLOR)})" for b in reversed(order)])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODEL_COLOR.values()]
    ax.legend(handles, [MODEL_NAME[m] for m in MODEL_COLOR], frameon=False, ncol=3, loc="upper left",
              bbox_to_anchor=(0, 1.13), fontsize=9)
    ax.set_xlabel("share of the gold bucket's items")
    ax.set_title("Where rules v2 send v7 dev items", loc="left", fontsize=11, pad=24)
    fig.tight_layout()
    fig.savefig(OUT / "rules_v2_routing_v7dev.png", dpi=160)
    plt.close(fig)


def per_bucket(summary: dict) -> None:
    systems = (("rules_v2_sim", "rules v2", "#c3c2b7"), ("mlp_max_quality", "MLP router", "#52514e"),
               ("item_oracle", "oracle per item", "#0b0b0b"))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.26
    for j, (key, name, color) in enumerate(systems):
        vals = [summary["per_bucket"][key][b] for b in BUCKETS]
        xs = [i + (j - 1) * w for i in range(len(BUCKETS))]
        ax.bar(xs, vals, width=w - 0.03, color=color, label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", color=INK2, fontsize=8)
    ax.set_xticks(range(len(BUCKETS)), BUCKETS)
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("mean score")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    ax.set_title("Per bucket on v7 dev: routers stay far below the item oracle", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "routerbench_v7dev_per_bucket.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / "results/routerbench_v7dev/summary.json").read_text(encoding="utf-8"))
    with (ROOT / "results/routerbench_v7dev/curves.csv").open(encoding="utf-8") as fh:
        curves = list(csv.DictReader(fh))
    diff = json.loads((ROOT / "results/analysis/v7_dev_router_vs_mamay12.json").read_text(encoding="utf-8"))
    cost_quality(summary, curves)
    per_bucket(summary)
    rules_vs_mamay12(diff)
    rules_routing()
    print("wrote", *sorted(p.name for p in OUT.glob("*v7dev*.png")))


if __name__ == "__main__":
    main()
