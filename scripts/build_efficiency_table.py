#!/usr/bin/env python3
"""Build efficiency table from score JSON + optional *.meta.json VRAM stats.

Example:
  python scripts/build_efficiency_table.py \\
    --scores results/scores_mixed_ua_v2.json \\
    --meta-glob 'results/*_mixed_ua_v1_*.meta.json' \\
    --out results/efficiency_table.json
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any


def load_meta(paths: list[str]) -> dict[str, dict[str, Any]]:
    """Keep latest meta per system (by filename stamp / mtime)."""
    by_sys: dict[str, tuple[float, dict]] = {}
    for path in paths:
        p = Path(path)
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sys = str(meta.get("system") or "")
        if not sys:
            continue
        mtime = p.stat().st_mtime
        prev = by_sys.get(sys)
        if prev is None or mtime >= prev[0]:
            by_sys[sys] = (mtime, meta)
    return {k: v[1] for k, v in by_sys.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--meta-glob", action="append", default=[])
    p.add_argument("--out", type=Path, default=Path("results/efficiency_table.json"))
    args = p.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    meta_paths: list[str] = []
    for pattern in args.meta_glob:
        meta_paths.extend(glob(pattern))
    metas = load_meta(meta_paths)

    rows = []
    for sys, s in sorted(scores.get("systems", {}).items()):
        meta = metas.get(sys, {})
        vram = meta.get("vram") or {}
        peak_gb = vram.get("peak_single_gpu_gb") or vram.get("peak_total_vram_gb")
        # Fallback: max used from vram_before snapshot if present
        if peak_gb is None and meta.get("vram_before"):
            used = [
                g.get("memory_used_mb", 0.0)
                for g in meta["vram_before"]
                if isinstance(g, dict) and "memory_used_mb" in g
            ]
            if used:
                peak_gb = round(max(used) / 1024.0, 3)

        quality = s.get("overall_mean_macro", s.get("overall_mean", 0.0))
        quality_micro = s.get("overall_mean", 0.0)
        gpu_s = s.get("gpu_seconds_per_prompt") or meta.get("gpu_seconds_per_prompt")
        row = {
            "system": sys,
            "n": s.get("n"),
            "quality_macro": quality,
            "quality_micro": quality_micro,
            "latency_avg_ms": s.get("latency_avg_ms"),
            "latency_p50_ms": s.get("latency_p50_ms"),
            "latency_p95_ms": s.get("latency_p95_ms"),
            "gpu_seconds_per_prompt": gpu_s,
            "peak_vram_gb": peak_gb,
            "quality_per_gb": round(quality / peak_gb, 4) if peak_gb else None,
            "decoding": meta.get("decoding"),
        }
        rows.append(row)

    out = {"rows": rows, "scores": str(args.scores), "n_meta": len(metas)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'system':12} {'q_macro':>7} {'q_micro':>7} {'p50_ms':>8} {'vram_GB':>8} {'q/GB':>8}")
    for r in rows:
        print(
            f"{r['system']:12} {r['quality_macro']:7.3f} {r['quality_micro']:7.3f} "
            f"{(r['latency_p50_ms'] or 0):8.1f} "
            f"{(r['peak_vram_gb'] or 0):8.3f} "
            f"{(r['quality_per_gb'] or 0):8.3f}"
        )
    print(json.dumps({"wrote": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
