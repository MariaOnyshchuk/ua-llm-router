#!/usr/bin/env python3
"""Cost-quality panels in the style of RouterBench Figure 3 (Hu et al., 2024), from existing outputs only.

Two figures, one panel per bucket or family:

- v7 dev (1423 items, 4 headline buckets): out-of-fold kNN / MLP predictions, 5-fold CV, as in
  scripts/analysis/routerbench_dev.py. The first panel uses one router for all buckets; each bucket panel
  uses a router trained and cross-validated inside that bucket (RouterBench trains one per eval); scores and latencies from results/v7_pool (lab screening runs).
- graphs_mixed_100_dev (100 dev items, 4 families): the saved live routers (router/artifacts/{knn,mlp},
  trained on all v7 dev) predict quality for each request; scores and latencies of Mamay-4B, Lapa and Aya
  come from their one-call runs on JUPITER (results/graphs_mixed_100_dev/scores_all_detail.jsonl). A routed
  request gets the single-model answer of the chosen model: same prompt, T=0, same machine.

Router: pick argmax_j (lambda * P_j - c_j), with c_j the mean latency of model j on v7 dev (the training
data); lambda is swept from 0 to inf. Zero router = convex hull of the single models. AIQ = mean of the
non-decreasing convex hull over the cost range of the single models in that panel. Panels show the mean
latency of the chosen model (s per request) against the mean score. Lambda is swept once on all items;
on graphs_mixed_100_dev a family panel shows the same choices restricted to that family (there is no composite
training data). Live router runs can differ from the simulation on a few items: vLLM at T=0 is not
bit-exact across runs (3 of 100 graphs_mixed_100_dev items for kNN, 1 for MLP; the routing itself matches).

    .venv/bin/python scripts/analysis/plot_routerbench_panels.py

Writes docs/figures/routerbench_panels_{v7dev,graphs_mixed_100_dev}.png and results/routerbench_panels/summary.json.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("routerbench_dev", ROOT / "scripts/analysis/routerbench_dev.py")
rb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rb)

from router.intent_rules import route_intent  # noqa: E402

MODELS = rb.MODELS  # ("mamay4", "lapa", "aya")
MODEL_COLOR = {"mamay4": "#2a78d6", "lapa": "#eb6834", "aya": "#1baf7a"}
MODEL_NAME = {"mamay4": "Mamay-4B", "lapa": "Lapa", "aya": "Aya"}
KNN_COLOR, MLP_COLOR = "#0b0b0b", "#8a5cc2"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
LAMBDAS = np.concatenate([[0.0], np.logspace(-2, 3, 80), [np.inf]])
OUT_FIG = ROOT / "docs/figures"
OUT_RES = ROOT / "results/routerbench_panels"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def route(P: np.ndarray, cost_est: np.ndarray, lam: float) -> np.ndarray:
    util = P if np.isinf(lam) else lam * P - cost_est
    return util.argmax(axis=1)


def panel_data(Q, C, P, cost_est, fixed: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    """Points, router curves and AIQ for the items in `mask`."""
    idx = np.where(mask)[0]
    n = len(Q)

    def point(choice):
        return float(C[idx, choice[idx]].mean()), float(Q[idx, choice[idx]].mean())

    singles = {m: point(np.full(n, j)) for j, m in enumerate(MODELS)}
    c_min = min(c for c, _ in singles.values())
    c_max = max(c for c, _ in singles.values())
    curves = {k: [point(route(P[k], cost_est, lam)) for lam in LAMBDAS] for k in P}
    aiq = {"zero": rb.aiq(list(singles.values()), c_min, c_max)}
    aiq.update({k: rb.aiq(curves[k], c_min, c_max) for k in curves})
    return {"n": int(mask.sum()), "singles": singles, "curves": curves, "aiq": aiq,
            "fixed": {k: point(v) for k, v in fixed.items()}, "cost_range": [c_min, c_max]}


def draw(ax, d: dict, title: str, fixed_style: dict) -> None:
    zero = rb.ndch(list(d["singles"].values()))
    ax.plot([p[0] for p in zero], [p[1] for p in zero], color=MUTED, lw=1.6, marker="s", ms=4, zorder=2)
    for kind, color, marker in (("knn", KNN_COLOR, "d"), ("mlp", MLP_COLOR, "^")):
        pts = sorted(set(d["curves"][kind]))
        hull = rb.ndch(pts)
        ax.plot([p[0] for p in hull], [p[1] for p in hull], color=color, lw=1.6, zorder=3)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", marker=marker, ms=3.5, color=color,
                alpha=0.55, zorder=3)
    for m, (c, q) in d["singles"].items():
        ax.scatter(c, q, s=46, color=MODEL_COLOR[m], edgecolor=SURFACE, linewidth=1.5, zorder=5)
    for key, (c, q) in d["fixed"].items():
        st = fixed_style[key]
        ax.scatter(c, q, s=st["s"], marker=st["marker"], color=st["face"], edgecolor=st["edge"], linewidth=1.4,
                   zorder=6)
    a = d["aiq"]
    ax.set_title(f"{title} (n={d['n']})\nAIQ: zero {a['zero']:.3f} · kNN {a['knn']:.3f} · MLP {a['mlp']:.3f}",
                 fontsize=9.5, linespacing=1.5)
    ax.set_xlabel("latency of the chosen model, s", fontsize=8.5)


def figure(panels: list[tuple[str, dict]], fixed_style: dict, suptitle: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    flat = axes.ravel()
    for ax, (title, d) in zip(flat, panels):
        draw(ax, d, title, fixed_style)
    for ax in flat[::3]:
        ax.set_ylabel("mean score")
    leg_ax = flat[len(panels)]
    leg_ax.axis("off")
    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=MODEL_COLOR[m], label=MODEL_NAME[m]) for m in MODELS]
    handles += [Line2D([], [], color=MUTED, lw=1.6, marker="s", ms=4, label="Zero router (single-model hull)"),
                Line2D([], [], color=KNN_COLOR, lw=1.6, marker="d", ms=4, label="kNN router (hull, λ points)"),
                Line2D([], [], color=MLP_COLOR, lw=1.6, marker="^", ms=4, label="MLP router (hull, λ points)")]
    handles += [Line2D([], [], ls="none", marker=st["marker"], ms=8, markerfacecolor=st["face"],
                       markeredgecolor=st["edge"], label=st["label"]) for st in fixed_style.values()]
    leg_ax.legend(handles=handles, loc="center", frameon=False, fontsize=9)
    fig.suptitle(suptitle, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def out_of_fold(Q, C, X, buckets):
    """Out-of-fold predicted quality (5-fold CV, inner CV for hyper-parameters) and training-fold mean cost."""
    folds = rb.stratified_folds(buckets, 5, 42)
    P = {k: np.zeros_like(Q) for k in rb.GRIDS}
    cost_est = np.zeros_like(C)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(Q)), te)
        cost_est[te] = C[tr].mean(axis=0)
        for kind in rb.GRIDS:
            P[kind][te], _ = rb.fit_predict(kind, X, Q, buckets, tr, te, 42)
    return P, cost_est


def v7_dev() -> tuple[list[tuple[str, dict]], np.ndarray]:
    rows, Q, C = rb.load_dev(ROOT / "benchmarks/tasks_public_4401.jsonl",
                             ROOT / "results/v7_pool/scores_specialists_detail.jsonl", "dev")
    buckets = np.array([r["bucket"] for r in rows])
    X = rb.embed([r["prompt"] for r in rows], "intfloat/multilingual-e5-large", ROOT / ".cache/routerbench")
    rules = np.array([MODELS.index(route_intent(r["prompt"], profile="v2").model) for r in rows])
    best_q = Q.max(axis=1, keepdims=True)
    oracle = np.where(Q == best_q, -C, -np.inf).argmax(axis=1)
    fixed = {"rules_v2": rules, "item_oracle": oracle}
    # Macro panel: one router over all buckets. Bucket panels: a router trained and cross-validated inside
    # the bucket, with that bucket's costs, as RouterBench trains one router per eval.
    P, cost_est = out_of_fold(Q, C, X, buckets)
    panels = [("all 4 buckets, one router", panel_data(Q, C, P, cost_est, fixed, np.ones(len(rows), bool)))]
    for b in rb.BUCKETS:
        m = buckets == b
        Pb, cb = out_of_fold(Q[m], C[m], X[m], buckets[m])
        panels.append((f"{b}, own router", panel_data(Q[m], C[m], Pb, cb, {k: v[m] for k, v in fixed.items()},
                                                       np.ones(int(m.sum()), bool))))
    return panels, C.mean(axis=0)


def graphs_mixed_100_dev(v7_cost: np.ndarray) -> list[tuple[str, dict]]:
    items = [json.loads(line) for line in (ROOT / "benchmarks/graphs_mixed_100_dev.jsonl")
             .read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    detail: dict[tuple[str, str], dict] = {}
    for line in (ROOT / "results/graphs_mixed_100_dev/scores_all_detail.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        detail[(r["system"], r["id"])] = r
    ids = [it["id"] for it in items]
    Q = np.array([[detail[(m, i)]["final_score"] for m in MODELS] for i in ids], dtype=float)
    C = np.array([[detail[(m, i)]["latency_ms"] / 1000.0 for m in MODELS] for i in ids], dtype=float)
    X = rb.embed([it["prompt"] for it in items], "intfloat/multilingual-e5-large", ROOT / ".cache/routerbench")
    P = {}
    for kind in ("knn", "mlp"):
        router = pickle.loads((ROOT / f"router/artifacts/{kind}/model.pkl").read_bytes())
        P[kind] = router.predict_quality(X)
    cost_est = np.tile(v7_cost, (len(ids), 1))
    best_q = Q.max(axis=1, keepdims=True)
    fixed = {"item_oracle": np.where(Q == best_q, -C, -np.inf).argmax(axis=1)}
    families = np.array([it["family"] for it in items])
    panels = []
    for title, mask in [("all families", np.ones(len(ids), bool))] + [(f, families == f) for f in
                                                                       ("simple", "chain", "graph", "parallel")]:
        d = panel_data(Q, C, P, cost_est, fixed, mask)
        idx = np.where(mask)[0]
        # Measured runs (not simulated): rules v2 one call and the gold graph with rules v2 steps.
        for system, key in (("router_direct", "rules_v2"), ("oracle", "gold_graph")):
            d["fixed"][key] = (float(np.mean([detail[(system, ids[i])]["latency_ms"] / 1000.0 for i in idx])),
                               float(np.mean([detail[(system, ids[i])]["final_score"] for i in idx])))
        panels.append((title, d))
    # Check: the lambda = inf end must equal the live router_knn / router_mlp runs.
    for kind in ("knn", "mlp"):
        sim = Q[np.arange(len(ids)), route(P[kind], cost_est, np.inf)].mean()
        live = np.mean([detail[(f"router_{kind}", i)]["final_score"] for i in ids])
        print(f"graphs_mixed_100_dev {kind}: simulated max-quality end {sim:.3f}, live run {live:.3f}", file=sys.stderr)
    return panels


def to_json(panels):
    return {t: {"n": d["n"], "aiq": {k: round(v, 4) for k, v in d["aiq"].items()},
                "cost_range": [round(x, 3) for x in d["cost_range"]],
                "points": {k: [round(c, 3), round(q, 4)] for k, (c, q) in {**d["singles"], **d["fixed"]}.items()},
                "max_quality_end": {k: [round(d["curves"][k][-1][0], 3), round(d["curves"][k][-1][1], 4)]
                                    for k in d["curves"]}}
            for t, d in panels}


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    star = {"marker": "*", "s": 170, "face": "#9fd89a", "edge": "#9fd89a", "label": "Oracle (best model per item)"}
    rules = {"marker": "D", "s": 44, "face": SURFACE, "edge": INK, "label": "Rules v2"}
    v7_panels, v7_cost = v7_dev()
    figure(v7_panels, {"rules_v2": {**rules, "label": "Rules v2 (simulated)"}, "item_oracle": star},
           "Cost vs. quality on mixed_ua_v7 dev (5-fold CV, lab latencies)", OUT_FIG / "routerbench_panels_v7dev.png")
    g_panels = graphs_mixed_100_dev(v7_cost)
    figure(g_panels, {"item_oracle": star, "rules_v2": {**rules, "label": "Rules v2, one call (run)"},
                      "gold_graph": {"marker": "P", "s": 70, "face": INK, "edge": INK,
                                     "label": "Gold graph, rules v2 steps (run)"}},
           "Cost vs. quality on graphs_mixed_100_dev (dev, JUPITER latencies; routers trained on v7 dev)",
           OUT_FIG / "routerbench_panels_graphs_mixed_100_dev.png")
    summary = {"what": "RouterBench-style cost-quality panels; see the script docstring",
               "git_commit": rb.git_commit(), "cost": "mean latency of the chosen model, s per request",
               "v7_dev": to_json(v7_panels), "graphs_mixed_100_dev": to_json(g_panels)}
    (OUT_RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {t: v["aiq"] for t, v in summary[k].items()} for k in ("v7_dev", "graphs_mixed_100_dev")}, indent=1))


if __name__ == "__main__":
    main()
