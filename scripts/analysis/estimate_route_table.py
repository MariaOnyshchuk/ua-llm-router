#!/usr/bin/env python3
"""Estimate what a different bucket->model routing table would score on v4.

This is arithmetic over ALREADY MEASURED solo per-bucket scores, not a new run.
Buckets are balanced (32 each), so the overall score is the mean of the six
bucket scores. Picking the best model per bucket on the same suite is
optimistic (selection on the test data): confirm any winner on a different
suite (v5/v6) or on a held-out half before quoting it.

Usage: python scripts/analysis/estimate_route_table.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUCKETS = ["chat", "code", "translate", "instruct", "knowledge", "alignment"]


def load_solos() -> dict[str, dict[str, float]]:
    solos: dict[str, dict[str, float]] = {}
    with (ROOT / "results/tables/v4_model_bucket_quality.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["system_id"] in {"mamay4", "lapa", "aya"}:
                solos[r["system_id"]] = {b: float(r[b]) for b in BUCKETS}
    s2 = json.loads((ROOT / "results/week_9_baselines/scores_mean_sd_s2.json").read_text())
    m = s2["systems"]["mamay12"] if "systems" in s2 else s2["mamay12"]
    solos["mamay12"] = {b: float(m[f"bucket.{b}.quality"]["mean"]) for b in BUCKETS}
    # The solo table uses the baseline alignment prompt; the router and the
    # Mamay-12B S2 run use the few-shot alignment prompt. Lapa with few-shot
    # scores 0.75 on alignment (router_v2_fewshot row). Aya and Mamay-4B were not
    # re-measured with few-shot in a full 3-rep run, so they keep the baseline
    # value (conservative: they are not chosen for alignment on that basis).
    solos["lapa"]["alignment"] = 0.75
    return solos


def best_table(solos: dict[str, dict[str, float]], pool: list[str]):
    table = {}
    for b in BUCKETS:
        winner = max(pool, key=lambda s: solos[s][b])
        table[b] = (winner, solos[winner][b])
    return table


def main() -> None:
    solos = load_solos()
    current = {  # rules v2 product routing, from docs/key_takeaways.md
        "chat": "aya", "translate": "aya", "knowledge": "lapa",
        "alignment": "lapa", "code": "mamay4", "instruct": "mamay4",
    }
    rows = []
    cur_scores = {b: solos[current[b]][b] for b in BUCKETS}
    rows.append(("current rules v2 (solo-score estimate)", current, cur_scores))
    for label, pool in [
        ("best per bucket: aya, lapa, mamay4 (no 12B)", ["aya", "lapa", "mamay4"]),
        ("best per bucket: aya, lapa, mamay4, mamay12", ["aya", "lapa", "mamay4", "mamay12"]),
        ("best per bucket: aya, lapa, mamay12 (drop 4B)", ["aya", "lapa", "mamay12"]),
    ]:
        t = best_table(solos, pool)
        rows.append((label, {b: t[b][0] for b in BUCKETS}, {b: t[b][1] for b in BUCKETS}))

    out = ROOT / "results/analysis/route_table_estimate.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["table", *[f"{b}_model" for b in BUCKETS], *BUCKETS, "overall_estimate"])
        for label, route, sc in rows:
            w.writerow([label, *[route[b] for b in BUCKETS], *[f"{sc[b]:.4f}" for b in BUCKETS],
                        f"{sum(sc.values()) / len(BUCKETS):.4f}"])
    for label, route, sc in rows:
        print(f"{label}: {sum(sc.values()) / len(BUCKETS):.4f}  " + ", ".join(f"{b}->{route[b]}" for b in BUCKETS))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
