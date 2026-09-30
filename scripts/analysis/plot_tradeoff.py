#!/usr/bin/env python3
"""Quality vs latency and quality vs GPU-seconds/prompt on mixed_ua_v4_balanced.

Quality and p50 latency come from results/tables/v4_model_bucket_quality.csv
(rows marked confirmed) plus the Mamay-12B and FP8-pack score files. GPU-seconds
per prompt come from the runs' *.meta.json where they exist; systems without a
meta file are omitted from the second panel, not estimated.

Usage: python scripts/analysis/plot_tradeoff.py
Outputs: results/figures/tradeoff_v4.png, results/analysis/tradeoff_points.csv
"""

from __future__ import annotations

import csv
import json
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# system_id in the CSV -> (label, resident GPUs at gpu_memory_utilization=0.90, meta glob)
CSV_SYSTEMS = {
    "mamay4": ("Mamay-4B solo", 1, "results/week_6_v4_solos/mamay4_*.meta.json"),
    "lapa": ("Lapa-12B solo", 1, "results/week_6_v4_solos/lapa_*.meta.json"),
    "aya": ("Aya-8B solo", 1, None),
    "router_v1_fewshot": ("Router v1 + few-shot", 3, "results/week_6_router_fewshot/*.meta.json"),
    "router_v2_fewshot": ("Router v2 + few-shot", 3, None),
}


# label -> text offset in points, to keep neighbouring labels apart
OFFSETS = {
    "Router v1 + few-shot": (6, -16),
    "Router v2 + few-shot": (6, 8),
    "Router v2 FP8, 1 card": (10, -28),
    "Lapa-12B solo": (-8, 8),
}


def mean_gpu_s(pattern: str | None):
    if not pattern:
        return None
    vals = []
    for f in glob(str(ROOT / pattern)):
        m = json.loads(Path(f).read_text(encoding="utf-8"))
        if m.get("gpu_seconds_per_prompt") is not None:
            vals.append(float(m["gpu_seconds_per_prompt"]))
    return float(np.mean(vals)) if vals else None


def load_points():
    pts = []
    with (ROOT / "results/tables/v4_model_bucket_quality.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sid = r["system_id"]
            if sid in CSV_SYSTEMS:
                label, gpus, meta = CSV_SYSTEMS[sid]
                pts.append((label, float(r["overall"]), float(r["p50_ms"]), mean_gpu_s(meta), gpus))

    s2 = json.loads((ROOT / "results/week_9_baselines/scores_mean_sd_s2.json").read_text())
    s2 = s2.get("systems", s2)["mamay12"]
    pts.append(("Mamay-12B solo (S2)", float(s2["overall_mean"]["mean"]) if isinstance(s2["overall_mean"], dict) else float(s2["overall_mean"]),
                float(s2["latency_p50_ms"]["mean"]) if isinstance(s2["latency_p50_ms"], dict) else float(s2["latency_p50_ms"]),
                mean_gpu_s("results/week_9_baselines/*.meta.json"), 1))

    pk = json.loads((ROOT / "results/week_11_pack_fp8/scores_mean_sd_pack.json").read_text())
    pk = pk.get("systems", pk)["router_matrix_v2_fewshot"]
    g = lambda v: float(v["mean"]) if isinstance(v, dict) else float(v)
    pts.append(("Router v2 FP8, 1 card", g(pk["overall_mean"]), g(pk["latency_p50_ms"]),
                mean_gpu_s("results/week_11_pack_fp8/*.meta.json"), 1))
    return pts


def main() -> None:
    pts = load_points()
    out_csv = ROOT / "results/analysis/tradeoff_points.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "overall_quality", "p50_latency_ms", "gpu_seconds_per_prompt", "resident_gpus"])
        for p in pts:
            w.writerow([p[0], f"{p[1]:.4f}", f"{p[2]:.1f}", "" if p[3] is None else f"{p[3]:.3f}", p[4]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {1: "#2a6f97", 3: "#c1440e"}
    for ax, (xi, xlabel) in zip(axes, [(2, "p50 latency, ms (lower is better)"),
                                       (3, "GPU-seconds per prompt (lower is better)")]):
        for label, q, p50, gs, gpus in pts:
            x = p50 if xi == 2 else gs
            if x is None:
                continue
            ax.scatter(x, q, s=90, color=colors[gpus], zorder=3)
            off = OFFSETS.get(label, (6, 5))
            ax.annotate(label, (x, q), textcoords="offset points", xytext=off, fontsize=8,
                        ha="right" if off[0] < 0 else "left")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Overall quality on mixed_ua_v4_balanced")
    axes[1].set_title("Systems without a meta file are omitted here", fontsize=8, loc="right")
    axes[0].set_title("Quality vs latency", fontsize=10, loc="left")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=colors[k], label=f"{k} resident GPU(s)") for k in (1, 3)]
    axes[0].legend(handles=handles, loc="center right", fontsize=8)
    fig.tight_layout()
    out = ROOT / "results/figures/tradeoff_v4.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} and {out_csv}")
    for p in pts:
        print(p)


if __name__ == "__main__":
    main()
