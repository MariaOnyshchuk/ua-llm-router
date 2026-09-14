#!/usr/bin/env python3
"""Wide CSVs of scored runs for browsing (Excel / Numbers).

  PYTHONPATH=. python scripts/export_results_tables.py
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"
BUCKETS = ("chat", "code", "translate", "instruct", "knowledge", "alignment")
HEADLINE = ("knowledge", "translate", "alignment", "instruct")
APPENDIX = ("chat", "code")
V6_SCREEN_SYSTEMS = ("mamay4", "lapa", "aya", "mamay12")
V6_ROUTER_SYSTEMS = ("mamay4", "lapa", "aya")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd_quality(blob: dict, system: str, bucket: str) -> tuple[float | None, float | None]:
    sys = blob.get("systems", {}).get(system) or {}
    rec = sys.get(f"bucket.{bucket}.quality")
    if isinstance(rec, dict) and "mean" in rec:
        return rec["mean"], rec.get("sd")
    return None, None


def mean_sd_overall(blob: dict, system: str) -> tuple[float | None, float | None, float | None]:
    sys = blob.get("systems", {}).get(system) or {}
    ov = sys.get("overall_mean") or {}
    p50 = sys.get("latency_p50_ms") or {}
    return ov.get("mean"), ov.get("sd"), p50.get("mean")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def v4_same_suite() -> None:
    """Authoritative mixed_ua_v4_balanced (192, T=0, seed=42, 3 repeats)."""
    solos = load(ROOT / "results/week_6_v4_solos/scores_mean_sd_v4_solos.json")
    aya = load(ROOT / "results/week_5_router_v4/scores_mean_sd_aya_v4_rescored.json")
    r_base = load(ROOT / "results/week_5_router_v4/scores_mean_sd_router_v4.json")
    r_fs = load(ROOT / "results/week_6_router_fewshot/scores_mean_sd_router_fewshot.json")
    cascade = load(ROOT / "results/week_6_cascade_micro/scores_mean_sd_cascade_micro.json")
    oracle = load(ROOT / "results/week_7_router_best/scores_mean_sd_router_best.json")
    v2 = load(ROOT / "results/week_7_rules_v2/scores_mean_sd_rules_v2.json")
    ens = load(ROOT / "results/week_7_ensemble/scores.json")
    quant = load(ROOT / "results/week_8_quant/scores.json")

    specs = [
        ("mamay4", "Mamay-4B solo", "baseline", solos, "mamay4", "confirmed", "week_6_v4_solos"),
        ("lapa", "Lapa-12B solo", "baseline", solos, "lapa", "confirmed", "week_6_v4_solos"),
        ("aya", "Aya-8B solo", "baseline", aya, "aya", "confirmed", "week_5_router_v4 rescored"),
        (
            "router_v1_baseline",
            "Rules v1 (code→Qwen)",
            "baseline",
            r_base,
            "router_matrix",
            "provisional",
            "week_5_router_v4 pre-leakage-fix",
        ),
        (
            "router_v1_fewshot",
            "Rules v1 + social few-shot",
            "fewshot",
            r_fs,
            "router_matrix_fewshot",
            "confirmed",
            "week_6_router_fewshot",
        ),
        (
            "cascade_micro",
            "Rules v1 few-shot + micro-cascade",
            "fewshot",
            cascade,
            "router_cascade_micro_fewshot",
            "confirmed",
            "week_6_cascade_micro",
        ),
        (
            "oracle_gold_bucket",
            "Gold-bucket oracle + few-shot",
            "fewshot",
            oracle,
            "router_best_fewshot",
            "confirmed",
            "week_7_router_best",
        ),
        (
            "router_v2_fewshot",
            "Rules v2 + few-shot (product)",
            "fewshot",
            v2,
            "router_matrix_v2_fewshot",
            "confirmed",
            "week_7_rules_v2",
        ),
    ]

    rows = []
    for sid, label, prompt, blob, key, status, src in specs:
        q, sd, p50 = mean_sd_overall(blob, key)
        row = {
            "system_id": sid,
            "label": label,
            "prompt": prompt,
            "suite": "mixed_ua_v4_balanced",
            "n_items": 192,
            "n_repeats": 3,
            "status": status,
            "overall": q,
            "overall_sd": sd,
            "p50_ms": p50,
            "source": src,
            "cite_as_thesis": "yes" if status == "confirmed" else "no",
        }
        for b in BUCKETS:
            qb, _ = mean_sd_quality(blob, key, b)
            row[b] = qb
        rows.append(row)

    # Ensemble / 4-bit summaries are flat scores.json (not mean_sd nested).
    for sid, label, path, syskey, src in (
        (
            "ensemble_vote",
            "Ensemble vote (align+ZNO)",
            ens,
            "router_ensemble_vote_fewshot",
            "week_7_ensemble",
        ),
        (
            "rules_v2_4bit",
            "Rules v2 4-bit bnb",
            quant,
            "router_matrix_v2_fewshot",
            "week_8_quant",
        ),
    ):
        sys = path.get("systems", {}).get(syskey) or {}
        by = sys.get("by_bucket") or {}
        row = {
            "system_id": sid,
            "label": label,
            "prompt": "fewshot",
            "suite": "mixed_ua_v4_balanced",
            "n_items": 192,
            "n_repeats": 1,
            "status": "confirmed",
            "overall": sys.get("overall_mean"),
            "overall_sd": None,
            "p50_ms": sys.get("latency_p50_ms"),
            "source": src,
            "cite_as_thesis": "yes",
        }
        for b in BUCKETS:
            cell = by.get(b) or {}
            row[b] = cell.get("quality_mean")
        rows.append(row)

    fields = [
        "system_id",
        "label",
        "prompt",
        "suite",
        "n_items",
        "n_repeats",
        "status",
        "cite_as_thesis",
        "overall",
        "overall_sd",
        "p50_ms",
        *BUCKETS,
        "source",
    ]
    write_csv(OUT / "v4_model_bucket_quality.csv", rows, fields)


def alignment_prompts() -> None:
    blob = load(ROOT / "results/week_5_alignment_bakeoff/scores_alignment_variants_flat.json")
    rows = []
    for key, sys in sorted(blob.get("systems", {}).items()):
        model, _, variant = key.partition("@")
        rows.append(
            {
                "model": model,
                "prompt_variant": variant,
                "n_items": sys.get("n"),
                "alignment_quality": sys.get("overall_mean"),
                "p50_ms": sys.get("latency_p50_ms"),
                "note": "32-item alignment slice only; Aya/Qwen not in this grid",
            }
        )
    write_csv(
        OUT / "v4_alignment_prompt_grid.csv",
        rows,
        ["model", "prompt_variant", "n_items", "alignment_quality", "p50_ms", "note"],
    )


def composite() -> None:
    blob = load(ROOT / "results/week_10_composite/scores.json")
    rows = []
    for name, sys in sorted(blob.get("systems", {}).items()):
        row = {
            "system": name,
            "planner_model": "mamay4" if name == "hybrid" else "",
            "final_score_mean": sys.get("final_score_mean"),
            "final_success_rate": sys.get("final_success_rate"),
            "plan_exact_match_rate": sys.get("plan_exact_match_rate"),
            "calls_per_item": sys.get("calls_per_item"),
            "latency_p50_ms": sys.get("latency_p50_ms"),
        }
        for fam, cell in (sys.get("by_family") or {}).items():
            row[f"{fam}_final"] = cell.get("final_score")
        rows.append(row)
    fields = [
        "system",
        "planner_model",
        "final_score_mean",
        "final_success_rate",
        "plan_exact_match_rate",
        "calls_per_item",
        "latency_p50_ms",
        "knowledge_explain_final",
        "translate_code_final",
        "translate_knowledge_write_final",
    ]
    write_csv(OUT / "composite_v1.csv", rows, fields)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def source_from_notes(notes: str) -> str:
    for part in str(notes or "").split():
        if part.startswith("source="):
            return part.split("=", 1)[1]
    return ""


def load_screen_detail() -> dict[str, dict[str, dict]]:
    """item_id -> system -> {score, latency_ms}."""
    path = ROOT / "results/v6_screen/scores_screen_detail.jsonl"
    by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in load_jsonl(path):
        iid = str(row.get("id") or "")
        sys = str(row.get("system") or "")
        if not iid or not sys:
            continue
        lat = row.get("latency_ms")
        by_item[iid][sys] = {
            "score": float(row.get("score") or 0.0),
            "latency_ms": float(lat) if lat is not None else None,
        }
    return by_item


def _mean(vals: list[float]) -> float | None:
    return round(statistics.fmean(vals), 4) if vals else None


def _p50(vals: list[float]) -> float | None:
    if not vals:
        return None
    return round(statistics.median(vals), 1)


def _summarize_slice(
    items: list[dict],
    detail: dict[str, dict[str, dict]],
    systems: tuple[str, ...],
    suite: str,
    cite: str,
    note: str,
) -> tuple[list[dict], list[dict]]:
    """Wide model rows + long model×bucket rows."""
    wide: list[dict] = []
    long: list[dict] = []
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_bucket[str(it.get("bucket") or "")].append(it)

    for sys in systems:
        bucket_means: dict[str, float | None] = {}
        overall_scores: list[float] = []
        overall_lats: list[float] = []
        n_scored = 0
        for b in BUCKETS:
            scores: list[float] = []
            lats: list[float] = []
            for it in by_bucket.get(b, []):
                cell = detail.get(str(it["id"]), {}).get(sys)
                if cell is None:
                    continue
                scores.append(cell["score"])
                if cell.get("latency_ms") is not None:
                    lats.append(cell["latency_ms"])
            bucket_means[b] = _mean(scores)
            n_scored += len(scores)
            overall_scores.extend(scores)
            overall_lats.extend(lats)
            long.append(
                {
                    "system": sys,
                    "suite": suite,
                    "bucket": b,
                    "role": "headline" if b in HEADLINE else "appendix",
                    "n": len(scores),
                    "quality_mean": _mean(scores),
                    "latency_p50_ms": _p50(lats),
                    "latency_avg_ms": round(statistics.fmean(lats), 1) if lats else None,
                    "cite_as_thesis": cite,
                    "note": note if sys == "mamay12" else "",
                }
            )
        macros = [v for v in bucket_means.values() if v is not None]
        row = {
            "system": sys,
            "suite": suite,
            "n": n_scored,
            "cite_as_thesis": cite,
            "overall_mean": _mean(overall_scores),
            "overall_mean_macro": _mean(macros) if macros else None,
            "latency_p50_ms": _p50(overall_lats),
            "latency_avg_ms": round(statistics.fmean(overall_lats), 1) if overall_lats else None,
            "note": note,
        }
        for b in BUCKETS:
            row[b] = bucket_means[b]
            row[f"{b}_n"] = sum(
                1 for it in by_bucket.get(b, []) if sys in detail.get(str(it["id"]), {})
            )
        wide.append(row)
    return wide, long


def v6_screen() -> None:
    blob = load(ROOT / "results/v6_screen/scores_screen.json")
    cite_screen = "NO — screening pool, not the frozen v6 suite"
    note12 = "mamay12 knowledge/alignment/code are failed empty generations"
    rows = []
    for name, sys in sorted(blob.get("systems", {}).items()):
        row = {
            "system": name,
            "suite": "mixed_ua_v6_screen",
            "n": sys.get("n"),
            "overall_mean": sys.get("overall_mean"),
            "overall_mean_macro": sys.get("overall_mean_macro"),
            "latency_p50_ms": sys.get("latency_p50_ms"),
            "latency_avg_ms": sys.get("latency_avg_ms"),
            "cite_as_thesis": cite_screen,
            "note": note12,
        }
        for b in BUCKETS:
            cell = (sys.get("by_bucket") or {}).get(b) or {}
            row[b] = cell.get("quality_mean")
            row[f"{b}_n"] = cell.get("n")
            row[f"{b}_p50_ms"] = cell.get("latency_p50_ms")
        rows.append(row)
    fields = [
        "system",
        "suite",
        "n",
        "cite_as_thesis",
        "overall_mean",
        "overall_mean_macro",
        "latency_p50_ms",
        "latency_avg_ms",
        *BUCKETS,
        *[f"{b}_n" for b in BUCKETS],
        *[f"{b}_p50_ms" for b in BUCKETS],
        "note",
    ]
    write_csv(OUT / "v6_screen_not_for_claims.csv", rows, fields)

    screen_items = load_jsonl(ROOT / "benchmarks/mixed_ua_v6_screen.jsonl")
    v6_items = load_jsonl(ROOT / "benchmarks/mixed_ua_v6.jsonl")
    detail = load_screen_detail()
    cite_preview = (
        "NO — screening pass sliced to frozen mixed_ua_v6 ids; official 3× bake-off not scored yet"
    )

    _, screen_long = _summarize_slice(
        screen_items, detail, V6_SCREEN_SYSTEMS, "mixed_ua_v6_screen", cite_screen, note12
    )
    preview_wide, preview_long = _summarize_slice(
        v6_items, detail, V6_SCREEN_SYSTEMS, "mixed_ua_v6", cite_preview, note12
    )
    write_csv(
        OUT / "v6_screen_model_bucket.csv",
        screen_long,
        [
            "system",
            "suite",
            "bucket",
            "role",
            "n",
            "quality_mean",
            "latency_p50_ms",
            "latency_avg_ms",
            "cite_as_thesis",
            "note",
        ],
    )
    write_csv(
        OUT / "v6_preview_from_screen.csv",
        preview_wide,
        [
            "system",
            "suite",
            "n",
            "cite_as_thesis",
            "overall_mean",
            "overall_mean_macro",
            "latency_p50_ms",
            "latency_avg_ms",
            *BUCKETS,
            *[f"{b}_n" for b in BUCKETS],
            "note",
        ],
    )
    write_csv(
        OUT / "v6_preview_model_bucket.csv",
        preview_long,
        [
            "system",
            "suite",
            "bucket",
            "role",
            "n",
            "quality_mean",
            "latency_p50_ms",
            "latency_avg_ms",
            "cite_as_thesis",
            "note",
        ],
    )

    item_rows = []
    for it in v6_items:
        iid = str(it.get("id") or "")
        bucket = str(it.get("bucket") or "")
        cells = detail.get(iid, {})
        scores_3 = []
        winners = []
        row = {
            "id": iid,
            "bucket": bucket,
            "role": "headline" if bucket in HEADLINE else "appendix",
            "source": source_from_notes(it.get("notes") or ""),
            "cite_as_thesis": cite_preview,
        }
        for sys in V6_SCREEN_SYSTEMS:
            cell = cells.get(sys) or {}
            score = cell.get("score")
            row[f"{sys}_score"] = score
            row[f"{sys}_latency_ms"] = cell.get("latency_ms")
            if sys in V6_ROUTER_SYSTEMS and score is not None:
                scores_3.append(float(score))
        if scores_3:
            best = max(float(cells[s]["score"]) for s in V6_ROUTER_SYSTEMS if s in cells)
            winners = [s for s in V6_ROUTER_SYSTEMS if s in cells and float(cells[s]["score"]) == best]
            row["best_of_mamay4_lapa_aya"] = "+".join(winners)
            row["score_var_3"] = round(statistics.pvariance(scores_3), 6) if len(scores_3) > 1 else 0.0
        else:
            row["best_of_mamay4_lapa_aya"] = ""
            row["score_var_3"] = None
        item_rows.append(row)
    write_csv(
        OUT / "v6_preview_items.csv",
        item_rows,
        [
            "id",
            "bucket",
            "role",
            "source",
            "mamay4_score",
            "lapa_score",
            "aya_score",
            "mamay12_score",
            "mamay4_latency_ms",
            "lapa_latency_ms",
            "aya_latency_ms",
            "mamay12_latency_ms",
            "best_of_mamay4_lapa_aya",
            "score_var_3",
            "cite_as_thesis",
        ],
    )


def v6_solos_one_pass() -> None:
    """Dedicated 1× frozen-v6 solos (Mamay-12B, Qwen-7B). Not 3-repeat mean±sd."""
    path = ROOT / "results/v6_solos/scores.json"
    if not path.is_file():
        return
    blob = load(path)
    cite = "YES — frozen mixed_ua_v6, 1 pass, T=0, seed=42, max_tokens=1024"
    rows = []
    long = []
    for name, sys in sorted(blob.get("systems", {}).items()):
        row = {
            "system": name,
            "suite": "mixed_ua_v6",
            "n": sys.get("n"),
            "repeats": 1,
            "cite_as_thesis": cite,
            "overall_mean": sys.get("overall_mean"),
            "overall_mean_macro": sys.get("overall_mean_macro"),
            "latency_p50_ms": sys.get("latency_p50_ms"),
            "latency_avg_ms": sys.get("latency_avg_ms"),
            "note": "dedicated solo; mamay4/lapa/aya dedicated 1× not in this file",
        }
        for b in BUCKETS:
            cell = (sys.get("by_bucket") or {}).get(b) or {}
            row[b] = cell.get("quality_mean")
            row[f"{b}_n"] = cell.get("n")
            row[f"{b}_p50_ms"] = cell.get("latency_p50_ms")
            long.append(
                {
                    "system": name,
                    "suite": "mixed_ua_v6",
                    "bucket": b,
                    "role": "headline" if b in HEADLINE else "appendix",
                    "n": cell.get("n"),
                    "quality_mean": cell.get("quality_mean"),
                    "latency_p50_ms": cell.get("latency_p50_ms"),
                    "latency_avg_ms": cell.get("latency_avg_ms"),
                    "cite_as_thesis": cite,
                    "note": "",
                }
            )
        rows.append(row)
    write_csv(
        OUT / "v6_solos_1pass.csv",
        rows,
        [
            "system",
            "suite",
            "n",
            "repeats",
            "cite_as_thesis",
            "overall_mean",
            "overall_mean_macro",
            "latency_p50_ms",
            "latency_avg_ms",
            *BUCKETS,
            *[f"{b}_n" for b in BUCKETS],
            *[f"{b}_p50_ms" for b in BUCKETS],
            "note",
        ],
    )
    write_csv(
        OUT / "v6_solos_1pass_model_bucket.csv",
        long,
        [
            "system",
            "suite",
            "bucket",
            "role",
            "n",
            "quality_mean",
            "latency_p50_ms",
            "latency_avg_ms",
            "cite_as_thesis",
            "note",
        ],
    )


def v3_bakeoff() -> None:
    blob = load(ROOT / "results/week_4_specialist_bakeoff/specialist_matrix_bakeoff.json")
    by = blob.get("quality_by_bucket") or {}
    models = blob.get("models") or []
    rows = []
    for m in models:
        row = {
            "system": m,
            "suite": "mixed_ua_v3",
            "overall_micro": (blob.get("overall") or {}).get(m, {}).get("micro"),
        }
        for b in BUCKETS:
            row[b] = (by.get(b) or {}).get(m)
        rows.append(row)
    write_csv(
        OUT / "v3_specialist_bakeoff.csv",
        rows,
        ["system", "suite", "overall_micro", *BUCKETS],
    )


def main() -> None:
    v4_same_suite()
    alignment_prompts()
    composite()
    v6_screen()
    v6_solos_one_pass()
    v3_bakeoff()
    print(f"wrote {OUT}")
    for p in sorted(OUT.glob("*.csv")):
        print(p.relative_to(ROOT), p.stat().st_size)


if __name__ == "__main__":
    main()
