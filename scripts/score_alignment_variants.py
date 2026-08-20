#!/usr/bin/env python3
"""Score the UAlign alignment ablation, keyed by prompt variant.

The generations already exist on disk, so this needs no GPU.

``score_results.py`` de-duplicates on ``(system, id)``, but every variant run
reuses the same system and item ids. Passing all variant files to it therefore
keeps only whichever file was read last. This script keys on
``(system, variant, id)`` instead, and reports the two things the baseline
number cannot show: the social confusion matrix, and whether the ethics
subset (which no variant rewrites) stayed fixed as a control.

Example:
  python scripts/score_alignment_variants.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_results import score_row  # noqa: E402

LABEL_RE = re.compile(r"got=(\S+) ref=(\S+)")
DEFAULT_DIR = Path("results/week_5_alignment_bakeoff")


def family(item_id: str) -> str:
    return "social" if "social" in item_id else "ethics"


def variant_of(row: dict[str, Any]) -> str:
    return (row.get("alignment_variant") or "baseline").lower()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def collect(paths: list[Path]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    """(system, variant) -> id -> scored row, tracking each repeat separately."""
    cells: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    repeats: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in sorted(paths):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("system")), variant_of(row))
            repeats[key].add(path.name)
            scored = score_row(row)
            item_id = str(row.get("id"))
            match = LABEL_RE.search(scored.get("detail") or "")
            record = {
                "id": item_id,
                "family": family(item_id),
                "score": float(scored["score"]),
                "ref": match.group(2) if match else "",
                "got": match.group(1) if match else "",
                "latency_ms": row.get("latency_ms"),
                "source": path.name,
            }
            prior = cells[key].get(item_id)
            if prior is not None and prior["score"] != record["score"]:
                record["nondeterministic"] = True
            cells[key][item_id] = record
    for key in cells:
        cells[key]["__repeats__"] = {"files": sorted(repeats[key])}  # type: ignore[assignment]
    return cells


def summarize(cells: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (system, variant), items in sorted(cells.items()):
        meta = items.pop("__repeats__")  # type: ignore[arg-type]
        rows = list(items.values())
        social = [r for r in rows if r["family"] == "social"]
        ref1 = [r for r in social if r["ref"] == "1"]

        # Split by family: ethics references are also 0/1, so a merged
        # confusion matrix would hide which scale actually moved.
        confusion: dict[str, dict[str, int]] = {}
        for name, group in (("social", social), ("ethics", [r for r in rows if r["family"] == "ethics"])):
            counts: dict[str, int] = defaultdict(int)
            for row in group:
                counts[f"{row['ref']}->{row['got'] or '?'}"] += 1
            confusion[name] = dict(sorted(counts.items()))

        out[f"{system}@{variant}"] = {
            "system": system,
            "variant": variant,
            "n": len(rows),
            "n_repeats": len(meta["files"]),
            "deterministic": not any(r.get("nondeterministic") for r in rows),
            "quality": round(mean([r["score"] for r in rows]), 4),
            "quality_ethics": round(
                mean([r["score"] for r in rows if r["family"] == "ethics"]), 4
            ),
            "quality_social": round(mean([r["score"] for r in social]), 4),
            "social_ref1_n": len(ref1),
            "social_ref1_correct": sum(int(r["score"] == 1.0) for r in ref1),
            "social_ref1_answered_2": sum(int(r["got"] == "2") for r in ref1),
            "latency_avg_ms": round(
                mean([float(r["latency_ms"]) for r in rows if r["latency_ms"] is not None]), 1
            ),
            "n_social": len(social),
            "confusion": confusion,
            "files": meta["files"],
        }
    return out


def exact_mcnemar_p(fixed: int, broken: int) -> float | None:
    """Two-sided exact binomial test on the discordant pairs."""
    n = fixed + broken
    if n == 0:
        return None
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return round(min(1.0, 2 * tail), 6)


def paired_deltas(
    cells: dict[tuple[str, str], dict[str, dict[str, Any]]],
    n_boot: int = 10_000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Baseline vs variant on identical items, so pair per item.

    Reported twice: over the whole bucket, and over the social reference=1
    items that were the diagnosed failure mode. The bucket-wide delta is
    diluted by the variants introducing a milder 2->1 error, so the subset is
    what actually tests the diagnosis.
    """
    subsets = {
        "all_alignment": lambda r: True,
        "social_ref1": lambda r: r["family"] == "social" and r["ref"] == "1",
    }
    out: list[dict[str, Any]] = []
    for system in sorted({system for system, _ in cells}):
        base = cells.get((system, "baseline"))
        if not base:
            continue
        for (other_system, variant), items in sorted(cells.items()):
            if other_system != system or variant == "baseline":
                continue
            for subset, keep in subsets.items():
                shared = sorted(i for i in set(base) & set(items) if keep(base[i]))
                if not shared:
                    continue
                diffs = [items[i]["score"] - base[i]["score"] for i in shared]
                rng = random.Random(seed)
                n = len(diffs)
                means = sorted(
                    sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                    for _ in range(n_boot)
                )
                fixed = sum(1 for d in diffs if d > 0)
                broken = sum(1 for d in diffs if d < 0)
                out.append(
                    {
                        "system": system,
                        "variant": variant,
                        "subset": subset,
                        "n_items": n,
                        "delta": round(sum(diffs) / n, 4),
                        "ci_low": round(means[int(0.025 * n_boot)], 4),
                        "ci_high": round(means[min(n_boot - 1, int(0.975 * n_boot))], 4),
                        "fixed": fixed,
                        "broken": broken,
                        "mcnemar_p": exact_mcnemar_p(fixed, broken),
                    }
                )
    return out


def to_scores_schema(summary: dict[str, Any]) -> dict[str, Any]:
    """Emit the repo's standard scores schema so the ledger can ingest it."""
    systems: dict[str, Any] = {}
    for key, cell in summary.items():
        systems[key] = {
            "n": cell["n"],
            "overall_mean": cell["quality"],
            "overall_mean_macro": cell["quality"],
            "latency_avg_ms": cell["latency_avg_ms"],
            "latency_p50_ms": cell["latency_avg_ms"],
            "by_bucket": {
                "alignment": {
                    "quality_mean": cell["quality"],
                    "latency_avg_ms": cell["latency_avg_ms"],
                    "latency_p50_ms": cell["latency_avg_ms"],
                    "n": cell["n"],
                }
            },
        }
    return {"systems": systems}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    run_files = [
        path
        for path in sorted(args.dir.glob("*.jsonl"))
        if not path.name.startswith("scores_")
    ]
    if not run_files:
        raise SystemExit(f"no run files in {args.dir}")

    cells = collect(run_files)
    summary = summarize(cells)

    header = (
        f"{'cell':22} {'n':>3} {'reps':>4} {'qual':>6} {'ethics':>7} {'social':>7} "
        f"{'ref=1 ok':>9} {'ref=1 ->2':>10}"
    )
    print(header)
    print("-" * len(header))
    for key, c in summary.items():
        print(
            f"{key:22} {c['n']:3d} {c['n_repeats']:4d} {c['quality']:6.4f} "
            f"{c['quality_ethics']:7.4f} {c['quality_social']:7.4f} "
            f"{c['social_ref1_correct']:>4}/{c['social_ref1_n']:<4} "
            f"{c['social_ref1_answered_2']:>4}/{c['social_ref1_n']:<5}"
        )

    print("\nsocial confusion, reference -> predicted (social items only):")
    for key, c in summary.items():
        print(f"  {key:22} n={c['n_social']:<3} {c['confusion']['social']}")

    pairs = paired_deltas(cells)
    print("\npaired vs baseline on identical items (95% bootstrap on the delta):")
    for row in pairs:
        cell = f"{row['system']}@{row['variant']}"
        print(
            f"  {cell:20} {row['subset']:13} n={row['n_items']:<3} "
            f"delta {row['delta']:+.4f} CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  "
            f"fixed {row['fixed']:2d} / broke {row['broken']:2d}  "
            f"McNemar p={row['mcnemar_p']}"
        )

    ethics = {c["quality_ethics"] for c in summary.values()}
    print(
        f"\ncontrol — ethics subset is never rewritten by a variant: "
        f"{'unchanged' if len(ethics) == 1 else f'DIVERGED {sorted(ethics)}'}"
    )
    nondet = [k for k, c in summary.items() if not c["deterministic"]]
    print(f"determinism across repeats: {'all identical' if not nondet else nondet}")

    out = args.out or args.dir / "scores_alignment_variants.json"
    out.write_text(
        json.dumps(
            {
                "suite": "mixed_ua_v4_balanced",
                "bucket": "alignment",
                "cells": summary,
                "paired_vs_baseline": pairs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    flat = args.dir / "scores_alignment_variants_flat.json"
    flat.write_text(
        json.dumps(to_scores_schema(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    detail = args.dir / "scores_alignment_variants_detail.jsonl"
    with detail.open("w", encoding="utf-8") as fh:
        for (system, variant), items in sorted(cells.items()):
            for row in items.values():
                fh.write(
                    json.dumps(
                        {
                            "system": f"{system}@{variant}",
                            "bucket": "alignment",
                            **row,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(json.dumps({"wrote": [str(out), str(flat), str(detail)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
