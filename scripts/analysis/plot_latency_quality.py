#!/usr/bin/env python3
"""Latency vs. score for the routers, RouterBench style (Hu et al. 2024, fig. "cost vs. performance").

Every router is scored from stored answers of one machine (the UCU lab run of 2 Nov 2026,
cluster/eval_mamay27_reference_lab.sbatch): on each item the router picks a model and gets that model's
score and that model's measured latency on the item. Router overhead (rules: none; kNN/MLP: one e5
embedding) is not included.

  - single models: one point each; Mamay-27B is the reference (two cards, tensor parallel 2);
  - zero router: the upper hull of the three small models (a random mix of two models lies on the line
    between them);
  - kNN / MLP curves: pick argmax_m (lambda * predicted score_m - mean latency_m), lambda swept from 0
    to infinity; mean latency per model comes from the router's training items, not from the test set;
  - Mamay-4B classifier + table (routing profile `llm`): Mamay-4B names the intent, the v2 table picks the
    model; its latency includes the classifier call (results/<bench>_lab/intent_llm_mamay4.jsonl);
  - oracle: the best model on each item, the fastest one among ties.

AIQ is the area under each curve over the latency range of the three small models, divided by that
range (higher is better). Points, not intervals: paired intervals for the score differences are in
router_comparison*.json.

  .venv/bin/python scripts/analysis/plot_latency_quality.py --bench nodes_mixed_600_test
  .venv/bin/python scripts/analysis/plot_latency_quality.py --bench graphs_mixed_100_dev
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from router.intent_rules import route_intent  # noqa: E402

spec = importlib.util.spec_from_file_location("routerbench_dev", ROOT / "scripts/analysis/routerbench_dev.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

MODELS = ("mamay4", "lapa", "aya")
REFERENCE = "mamay27"
CURVES = ("knn_nodes_only", "mlp_nodes_only")
POINTS = ("knn", "mlp", "knn_nodes", "mlp_nodes")
NAMES = {"mamay4": "Mamay-4B", "lapa": "Lapa-12B", "aya": "Aya-8B", "mamay27": "Mamay-27B (reference)",
         "knn": "kNN (v7 dev)", "mlp": "MLP (v7 dev)", "knn_nodes": "kNN (v7 dev + nodes)",
         "mlp_nodes": "MLP (v7 dev + nodes)", "knn_nodes_only": "kNN (nodes only)",
         "mlp_nodes_only": "MLP (nodes only)", "rules_v2_text": "Rules v2 (text)",
         "rules_v2_intent": "Rules v2 (gold intent label)",
         "llm_mamay4": "Mamay-4B classifier + table", "oracle": "Oracle", "zero": "Zero router (model mix)"}
TRAIN_LOGS = {"knn_nodes_only": ["nodes_mixed_600_train"], "mlp_nodes_only": ["nodes_mixed_600_train"]}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]


def newest(folder: Path, system: str) -> Path | None:
    files = sorted(folder.glob(f"{system}_composite_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_system(bench: str, system: str, tag: str) -> tuple[dict[str, float], dict[str, float]] | None:
    detail = ROOT / "results" / f"{bench}{tag}" / f"scores_{system}_detail.jsonl"
    raw = newest(ROOT / "logs" / f"{bench}{tag}", system)
    if not detail.exists() or raw is None:
        return None
    score = {r["id"]: float(r["final_score"] or 0.0) for r in load_jsonl(detail) if r["system"] == system}
    rows = load_jsonl(raw)
    bad = [r["id"] for r in rows if r.get("http_status") != 200]
    if bad:
        raise SystemExit(f"{raw.name}: {len(bad)} rows without HTTP 200, e.g. {bad[:3]}")
    return score, {r["id"]: float(r["latency_ms"]) / 1000 for r in rows}


def train_latency(kind: str) -> np.ndarray:
    """Mean latency per model on the router's training items (seconds)."""
    out = []
    for m in MODELS:
        vals = []
        for bench in TRAIN_LOGS[kind]:
            path = newest(ROOT / "logs" / bench, m)
            vals += [float(r["latency_ms"]) / 1000 for r in load_jsonl(path)]
        out.append(np.mean(vals))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--tag", default="_lab", help="suffix of the logs/ and results/ folders")
    args = ap.parse_args()

    items = load_jsonl(ROOT / "benchmarks" / f"{args.bench}.jsonl")
    ids = [it["id"] for it in items]
    data = {m: load_system(args.bench, m, args.tag) for m in (*MODELS, REFERENCE)}
    for m in MODELS:
        if data[m] is None:
            raise SystemExit(f"missing lab answers for {m} on {args.bench}")
        missing = set(ids) - set(data[m][0]) | set(ids) - set(data[m][1])
        if missing:
            raise SystemExit(f"{m}: {len(missing)} items without score or latency")
    Q = np.array([[data[m][0][i] for m in MODELS] for i in ids])
    C = np.array([[data[m][1][i] for m in MODELS] for i in ids])
    rows = np.arange(len(ids))
    point = lambda pick: (float(C[rows, pick].mean()), float(Q[rows, pick].mean()))
    idx = lambda names: np.array([MODELS.index(n if n in MODELS else "aya") for n in names])

    points: dict[str, tuple[float, float]] = {m: point(np.full(len(ids), j)) for j, m in enumerate(MODELS)}
    if data[REFERENCE] is not None and set(ids) <= set(data[REFERENCE][0]) and set(ids) <= set(data[REFERENCE][1]):
        points[REFERENCE] = (float(np.mean([data[REFERENCE][1][i] for i in ids])),
                             float(np.mean([data[REFERENCE][0][i] for i in ids])))
    points["rules_v2_text"] = point(idx([route_intent(it["prompt"], profile="v2").model for it in items]))
    if all("label" in it for it in items):
        points["rules_v2_intent"] = point(idx([route_intent(it["prompt"], profile="v2", intent_hint=it["label"]).model
                                               for it in items]))
    llm_file = ROOT / "results" / f"{args.bench}{args.tag}" / "intent_llm_mamay4.jsonl"
    if llm_file.exists():
        from router.llm_intent import INTENT_MODEL
        cls = {r["id"]: r for r in load_jsonl(llm_file)}
        if set(ids) <= set(cls):
            picks, extra = [], []
            for it in items:
                r = cls[it["id"]]
                # Profile `llm`: the v2 rules decide only when the classifier call fails.
                picks.append(INTENT_MODEL[r["intent"]] if r["intent"] else route_intent(it["prompt"], profile="v2").model)
                extra.append((r["latency_ms"] or 0.0) / 1000)
            c, q = point(idx(picks))
            points["llm_mamay4"] = (c + float(np.mean(extra)), q)
    best = np.where(Q == Q.max(axis=1, keepdims=True), C, np.inf).argmin(axis=1)
    points["oracle"] = point(best)

    X = rb.embed([it["prompt"] for it in items], "intfloat/multilingual-e5-large", ROOT / ".cache/routerbench")
    preds = {k: pickle.loads((ROOT / "router/artifacts" / k / "model.pkl").read_bytes()) for k in (*CURVES, *POINTS)}
    for k in POINTS:
        points[k] = point(idx([str(m) for m in preds[k].predict(X)]))

    lambdas = np.concatenate([[0.0], np.logspace(-2, 3, 120), [np.inf]])
    curves: dict[str, list[tuple[float, float]]] = {}
    for k in CURVES:
        P = np.clip(preds[k].predict_quality(X), 0, 1)
        cost = train_latency(k)
        curves[k] = [point((P if np.isinf(lam) else lam * P - cost).argmax(axis=1)) for lam in lambdas]
        points[k] = curves[k][-1]
    curves["zero"] = rb.ndch([points[m] for m in MODELS])

    c_min = min(points[m][0] for m in MODELS)
    c_max = max(points[m][0] for m in MODELS)
    aiq = {"zero": rb.aiq([points[m] for m in MODELS], c_min, c_max)}
    for k in CURVES:
        aiq[k] = rb.aiq(curves[k], c_min, c_max)

    out_dir = ROOT / "results" / f"{args.bench}{args.tag}"
    summary = {
        "bench": args.bench, "n_items": len(ids), "machine": "UCU lab ml-compute-1, RTX 6000 Ada",
        "x": "mean latency of the chosen model per item, seconds (router overhead not included)",
        "y": "mean final_score", "latency_range_for_aiq": [round(c_min, 3), round(c_max, 3)],
        "points": {k: {"latency_s": round(c, 3), "score": round(q, 4)} for k, (c, q) in points.items()},
        "aiq": {k: round(v, 4) for k, v in aiq.items()},
        "curves": {k: [[round(c, 3), round(q, 4)] for c, q in rb.ndch(v)] for k, v in curves.items()},
    }
    (out_dir / "latency_quality.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    colors = {"zero": "#7a7a7a", "knn_nodes_only": "#2a6fdb", "mlp_nodes_only": "#e07b00"}
    for k, style in (("zero", "s-"), ("knn_nodes_only", "D-"), ("mlp_nodes_only", "^-")):
        hull = rb.ndch(curves[k])
        label = f"{NAMES[k]}, AIQ {aiq[k]:.3f}"
        ax.plot([p[0] for p in hull], [p[1] for p in hull], style, color=colors[k], lw=2, ms=5, label=label)
    marks = {"mamay4": ("o", "#3a9d5d"), "lapa": ("o", "#c0392b"), "aya": ("o", "#8e44ad"),
             "mamay27": ("P", "#111111"), "rules_v2_text": ("X", "#555555"), "rules_v2_intent": ("x", "#555555"),
             "llm_mamay4": ("h", "#1f8a70"),
             "knn": ("d", "#9cbcf0"), "knn_nodes": ("d", "#5b8fe0"), "mlp": ("v", "#f5c27a"),
             "mlp_nodes": ("v", "#eb9a33"), "oracle": ("*", "#2ecc71")}
    for k, (m, col) in marks.items():
        if k in points:
            ax.scatter(*points[k], marker=m, s=220 if k == "oracle" else 70, color=col, zorder=3,
                       edgecolor="white", linewidth=1, label=f"{NAMES[k]}  {points[k][1]:.3f}")
    ax.set_xlabel("Mean latency per item, s (model answer only)")
    ax.set_ylabel("Mean score")
    ax.set_title(f"Latency vs. score: {args.bench} (n={len(ids)}, lab answers, T=0)")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_quality.png", bbox_inches="tight")

    for k, (c, q) in sorted(points.items(), key=lambda kv: kv[1][0]):
        print(f"{NAMES[k]:28} latency {c:6.2f}s  score {q:.3f}")
    print("AIQ:", {k: round(v, 4) for k, v in aiq.items()})


if __name__ == "__main__":
    main()
