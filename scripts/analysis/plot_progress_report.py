"""Figures for docs/progress_report_2026-10-12.md.

Reads score summaries only (no prompts, no generations):
  results/analysis/v6_selection_check.json
  results/analysis/v6_router_vs_mamay12.json
  results/analysis/agentcoma_dev_20261012_summary.json

Usage: python scripts/analysis/plot_progress_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a8a79f"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / "analysis" / name).read_text())


def fig_selection() -> None:
    """Specialist scores on the items v6 kept vs the items it dropped."""
    data = load("v6_selection_check.json")["buckets"]
    models = [("mamay4", "Mamay-4B", BLUE), ("lapa", "Lapa-12B", ORANGE), ("aya", "Aya-8B", AQUA)]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, bucket in zip(axes, ("knowledge", "translate", "alignment", "instruct")):
        for gi, part in enumerate(("kept_in_v6", "dropped")):
            for mi, (key, label, color) in enumerate(models):
                value = data[bucket][part][key]
                x = gi + (mi - 1) * 0.26
                ax.bar(x, value, width=0.22, color=color, label=label if gi == 0 else None)
                ax.text(x, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"kept in v6\nn={data[bucket]['kept_in_v6']['n']}",
                            f"dropped\nn={data[bucket]['dropped']['n']}"])
        ax.set_title(bucket)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean score (screening run)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=3, fontsize=9)
    fig.suptitle("v6 kept the items where the three specialists score the same", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "v6_selection_effect.png", dpi=160)
    plt.close(fig)


def fig_v6_gap() -> None:
    """Router minus Mamay-12B on v6, paired bootstrap 95% interval."""
    rows = [r for r in load("v6_router_vs_mamay12.json")["rows"]
            if r["scope"] in ("knowledge", "translate", "alignment", "instruct") or r["scope"].startswith("macro")]
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for y, r in enumerate(rows):
        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.plot(r["diff"], y, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(r["ci_low"] - 0.004, y, f"{r['diff']:+.3f}", va="center", ha="right", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['scope']}  (n={r['n']})" for r in rows])
    ax.set_xlabel("router rules v2 − Mamay-12B (score difference, 95% interval)")
    ax.set_xlim(-0.11, 0.07)
    ax.grid(axis="y", visible=False)
    fig.suptitle("On v6 the router is below Mamay-12B in three of four buckets", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "v6_router_vs_mamay12.png", dpi=160)
    plt.close(fig)


def fig_agentcoma() -> None:
    """AgentCoMa-UK dev accuracy by system, bootstrap 95% interval."""
    systems = load("agentcoma_dev_20261012_summary.json")["systems"]
    order = [
        ("oracle", "Oracle plan before the fix (step 1 lacks the task)", GREY),
        ("hybrid", "Router + LLM planner (hybrid), before", BLUE),
        ("router_direct", "Router rules v2, one call, before", BLUE),
        ("direct_lapa", "Lapa-12B alone (best specialist)", GREY),
        ("think_aya", "Aya-8B + reasoning suffix", GREY),
        ("hybrid_gated_v2", "Router + gate + reasoning suffix, after", BLUE),
        ("oracle_commonsense", "Router on the gold two-step plan (oracle, fixed)", BLUE),
        ("think_mamay12", "Mamay-12B + reasoning suffix", ORANGE),
        ("direct_mamay12", "Mamay-12B alone (reference)", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    for y, (key, label, color) in enumerate(order):
        s = systems[key]
        ax.barh(y, s["acc"], height=0.5, color=color)
        ax.plot([s["lo"], s["hi"]], [y, y], color=INK, linewidth=1.2)
        ax.text(s["hi"] + 0.012, y, f"{s['acc']:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o[1] for o in order])
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("share of correct final answers (n=50, dev split, one run)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREY)]
    ax.legend(handles, ["router", "Mamay-12B", "other variants"], frameon=False,
              loc="lower right", fontsize=8)
    fig.suptitle("AgentCoMa-UK dev: the gate removes the planner loss, Mamay-12B still leads", x=0.01,
                 ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "agentcoma_dev_accuracy.png", dpi=160)
    plt.close(fig)


def fig_cost() -> None:
    """Calls and latency for the three router variants (same runner, sequential)."""
    rows = [("LLM planner\n(hybrid), before", 4.29, 43.6), ("rules v2,\none call", 1.0, 4.4),
            ("gate + suffix\n(hybrid_gated)", 1.28, 5.5)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, idx, title, unit in ((axes[0], 1, "Model calls per request", ""), (axes[1], 2, "Latency p50, seconds", " s")):
        for x, row in enumerate(rows):
            ax.bar(x, row[idx], width=0.5, color=BLUE)
            ax.text(x, row[idx], f"{row[idx]:g}{unit}", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(r[idx] for r in rows) * 1.18)
    fig.tight_layout()
    fig.savefig(OUT / "agentcoma_dev_cost.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_selection()
    fig_v6_gap()
    fig_agentcoma()
    fig_cost()
    print(f"wrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
