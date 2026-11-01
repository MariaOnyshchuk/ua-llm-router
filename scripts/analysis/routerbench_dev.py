#!/usr/bin/env python3
"""Predictive routers as in RouterBench (Hu et al., 2024, arXiv 2403.12031), offline on v7 dev.

For every request the router predicts the quality P_j of each specialist from the prompt
embedding (kNN regression: mean score of the k nearest training prompts; MLP regression),
then picks the model with the largest  lambda * P_j - cost_j.  Sweeping lambda (the
"willingness to pay") gives a cost-quality curve instead of one point.

Cost is latency in seconds: the router uses each model's mean latency on the training
fold; the reported cost is the measured latency of the chosen model on the item
(screening runs of 4 Sep 2026, one model per GPU, bf16).

Baselines on the same items and scores:
- each specialist alone;
- Zero router: random mixtures of single models, i.e. their non-decreasing convex hull;
- rules v2 (simulated: `route_intent(prompt, "v2")` picks the model, pool scores are used);
- best model per gold bucket (knows the bucket, so it is an oracle of the bucket);
- item oracle (the best model on each item, cheapest among ties).

Each curve is turned into its non-decreasing convex hull and summarised by AIQ, the mean
quality over the shared cost range [cost of the cheapest model, cost of the dearest].
Below its cheapest point a hull is drawn to quality 0 at the low end of the range, as in
RouterBench Figure 2.

Quality is the macro mean over the four headline buckets (knowledge, translate,
alignment, instruct). Only the dev split is read; the test split is never loaded.
Hyper-parameters are chosen by an inner 3-fold CV on each outer training fold.

    .venv/bin/python scripts/analysis/routerbench_dev.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODELS = ("mamay4", "lapa", "aya")
BUCKETS = ("knowledge", "translate", "alignment", "instruct")


# ---------------------------------------------------------------- data
def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def load_dev(suite: Path, scores: Path, split: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Items of `split` in the headline buckets that all three specialists answered."""
    items = {r["id"]: r for r in load_jsonl(suite) if r.get("split") == split and r.get("bucket") in BUCKETS}
    score: dict[str, dict[str, float]] = defaultdict(dict)
    lat: dict[str, dict[str, float]] = defaultdict(dict)
    for r in load_jsonl(scores):
        if r.get("id") in items and r.get("system") in MODELS:
            score[r["id"]][r["system"]] = float(r.get("score") or 0.0)
            lat[r["id"]][r["system"]] = float(r.get("latency_ms") or 0.0) / 1000.0
    ids = sorted(i for i in items if len(score[i]) == len(MODELS) and len(lat[i]) == len(MODELS))
    rows = [items[i] for i in ids]
    Q = np.array([[score[i][m] for m in MODELS] for i in ids])
    C = np.array([[lat[i][m] for m in MODELS] for i in ids])
    return rows, Q, C


def embed(prompts: list[str], model_name: str, cache_dir: Path) -> np.ndarray:
    key = hashlib.sha256(("\n\x00".join(prompts) + model_name).encode("utf-8")).hexdigest()[:16]
    path = cache_dir / f"emb_{model_name.replace('/', '_')}_{key}.npy"
    if path.is_file():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [f"query: {p}" if "e5" in model_name.lower() else p for p in prompts]
    X = np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True),
                   dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    return X


# ---------------------------------------------------------------- metrics
def macro(q_item: np.ndarray, buckets: np.ndarray) -> float:
    return float(np.mean([q_item[buckets == b].mean() for b in BUCKETS]))


def ndch(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-decreasing upper convex hull of (cost, quality) points, sorted by cost."""
    pts = sorted(set(points))
    hull: list[tuple[float, float]] = []
    for p in pts:  # upper hull (monotone chain)
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    out: list[tuple[float, float]] = []
    for p in hull:  # keep the rising part only
        if not out or p[1] > out[-1][1]:
            out.append(p)
    return out


def hull_value(hull: list[tuple[float, float]], c: float, c_low: float) -> float:
    xs = [p[0] for p in hull]
    ys = [p[1] for p in hull]
    if c >= xs[-1]:
        return ys[-1]
    if c < xs[0]:
        if c_low >= xs[0]:
            return ys[0]
        return float(np.interp(c, [c_low, xs[0]], [0.0, ys[0]]))
    return float(np.interp(c, xs, ys))


def aiq(points: list[tuple[float, float]], c_min: float, c_max: float, n: int = 2001) -> float:
    hull = ndch(points)
    grid = np.linspace(c_min, c_max, n)
    vals = np.array([hull_value(hull, c, c_min) for c in grid])
    return float(np.trapezoid(vals, grid) / (c_max - c_min))


def bootstrap_macro(a: np.ndarray, b: np.ndarray, buckets: np.ndarray, n_boot: int, seed: int) -> dict:
    """Paired bootstrap of macro(a) - macro(b), items resampled inside each bucket."""
    rng = np.random.default_rng(seed)
    groups = [np.where(buckets == bk)[0] for bk in BUCKETS]
    d = a - b
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        diffs[k] = np.mean([d[rng.choice(g, size=len(g), replace=True)].mean() for g in groups])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(macro(a, buckets) - macro(b, buckets), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "distinguishable": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------- predictors
def stratified_folds(buckets: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for b in BUCKETS:
        idx = np.where(buckets == b)[0]
        rng.shuffle(idx)
        for n, i in enumerate(idx):
            folds[n % k].append(int(i))
    return [np.array(sorted(f)) for f in folds]


def make_model(kind: str, hp: dict):
    if kind == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor(n_neighbors=hp["k"], weights="uniform")
    from sklearn.neural_network import MLPRegressor

    return MLPRegressor(hidden_layer_sizes=hp["hidden"], alpha=hp["alpha"], early_stopping=True,
                        max_iter=500, random_state=42)


GRIDS = {
    "knn": [{"k": k} for k in (5, 10, 20, 40, 80)],
    "mlp": [{"hidden": h, "alpha": a} for h in ((128,), (256, 64)) for a in (1e-3, 1e-1)],
}


def fit_predict(kind: str, X: np.ndarray, Q: np.ndarray, buckets: np.ndarray, tr: np.ndarray, te: np.ndarray,
                seed: int) -> tuple[np.ndarray, dict]:
    """Choose hyper-parameters by inner 3-fold MSE on `tr`, refit on `tr`, predict `te`."""
    inner = stratified_folds(buckets[tr], 3, seed + 1)
    best, best_mse = None, np.inf
    for hp in GRIDS[kind]:
        errs = []
        for f in inner:
            itr = np.setdiff1d(np.arange(len(tr)), f)
            m = make_model(kind, hp).fit(X[tr][itr], Q[tr][itr])
            errs.append(float(np.mean((np.clip(m.predict(X[tr][f]), 0, 1) - Q[tr][f]) ** 2)))
        if np.mean(errs) < best_mse:
            best, best_mse = hp, float(np.mean(errs))
    m = make_model(kind, best).fit(X[tr], Q[tr])
    return np.clip(m.predict(X[te]), 0, 1), {**best, "inner_mse": round(best_mse, 4)}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--suite", type=Path, default=ROOT / "benchmarks/tasks_public_4401.jsonl")
    p.add_argument("--scores", type=Path, default=ROOT / "results/v7_pool/scores_specialists_detail.jsonl")
    p.add_argument("--split", default="dev", choices=["dev"])  # test stays closed until the router is frozen
    p.add_argument("--encoder", default="intfloat/multilingual-e5-large")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--cache", type=Path, default=ROOT / ".cache/routerbench")
    p.add_argument("--out", type=Path, default=ROOT / "results/routerbench_v7dev")
    args = p.parse_args()

    rows, Q, C = load_dev(args.suite, args.scores, args.split)
    buckets = np.array([r["bucket"] for r in rows])
    n = len(rows)
    X = embed([r["prompt"] for r in rows], args.encoder, args.cache)
    print(f"{n} dev items, embeddings {X.shape}", file=sys.stderr)

    # Out-of-fold predicted quality for every item and model, and the fold's mean cost per model.
    folds = stratified_folds(buckets, args.folds, args.seed)
    P = {k: np.zeros_like(Q) for k in GRIDS}
    cost_est = np.zeros_like(C)
    chosen_hp: dict[str, list[dict]] = {k: [] for k in GRIDS}
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        cost_est[te] = C[tr].mean(axis=0)
        for kind in GRIDS:
            P[kind][te], hp = fit_predict(kind, X, Q, buckets, tr, te, args.seed)
            chosen_hp[kind].append(hp)

    rng_items = np.arange(n)
    lambdas = np.concatenate([[0.0], np.logspace(-2, 3, 80), [np.inf]])

    def route(Pk: np.ndarray, lam: float) -> np.ndarray:
        util = Pk if np.isinf(lam) else lam * Pk - cost_est
        return util.argmax(axis=1)

    def point(choice: np.ndarray) -> tuple[float, float]:
        return float(C[rng_items, choice].mean()), macro(Q[rng_items, choice], buckets)

    singles = {m: point(np.full(n, j)) for j, m in enumerate(MODELS)}
    c_min = min(c for c, _ in singles.values())
    c_max = max(c for c, _ in singles.values())

    curves: dict[str, list[tuple[float, float, float]]] = {}
    for kind in GRIDS:
        curves[kind] = [(float(lam), *point(route(P[kind], lam))) for lam in lambdas]

    from router.intent_rules import route_intent

    rules_choice = np.array([MODELS.index(route_intent(r["prompt"], profile="v2").model)
                             if route_intent(r["prompt"], profile="v2").model in MODELS else MODELS.index("aya")
                             for r in rows])
    bucket_best = {b: int(np.argmax(Q[buckets == b].mean(axis=0))) for b in BUCKETS}
    bucket_choice = np.array([bucket_best[b] for b in buckets])
    best_q = Q.max(axis=1, keepdims=True)
    oracle_choice = np.where(Q == best_q, -C, -np.inf).argmax(axis=1)

    fixed = {"rules_v2_sim": rules_choice, "bucket_oracle": bucket_choice, "item_oracle": oracle_choice}
    points = {**{m: singles[m] for m in MODELS}, **{k: point(v) for k, v in fixed.items()}}

    zero_pts = list(singles.values())
    aiq_rows = {"zero_router": aiq(zero_pts, c_min, c_max)}
    for kind in GRIDS:
        aiq_rows[kind] = aiq([(c, q) for _, c, q in curves[kind]], c_min, c_max)
    aiq_rows["rules_v2_sim + singles"] = aiq(zero_pts + [points["rules_v2_sim"]], c_min, c_max)
    aiq_rows["bucket_oracle + singles"] = aiq(zero_pts + [points["bucket_oracle"]], c_min, c_max)

    # Quality-only end of each curve (lambda = inf) against the strongest baselines.
    best_single = max(MODELS, key=lambda m: singles[m][1])
    q_best_single = Q[:, MODELS.index(best_single)]
    comparisons = {}
    for kind in GRIDS:
        q_router = Q[rng_items, route(P[kind], np.inf)]
        comparisons[f"{kind}_max_quality - {best_single}"] = bootstrap_macro(q_router, q_best_single, buckets,
                                                                             args.n_boot, args.seed)
        comparisons[f"{kind}_max_quality - rules_v2_sim"] = bootstrap_macro(q_router, Q[rng_items, rules_choice],
                                                                            buckets, args.n_boot, args.seed)
        comparisons[f"{kind}_max_quality - bucket_oracle"] = bootstrap_macro(q_router, Q[rng_items, bucket_choice],
                                                                             buckets, args.n_boot, args.seed)
    comparisons[f"rules_v2_sim - {best_single}"] = bootstrap_macro(Q[rng_items, rules_choice], q_best_single,
                                                                   buckets, args.n_boot, args.seed)

    per_bucket = {}
    for name, choice in {**{m: np.full(n, j) for j, m in enumerate(MODELS)}, **fixed,
                         **{f"{k}_max_quality": route(P[k], np.inf) for k in GRIDS}}.items():
        per_bucket[name] = {b: round(float(Q[rng_items, choice][buckets == b].mean()), 4) for b in BUCKETS}
        per_bucket[name]["share"] = {m: round(float((choice == j).mean()), 3) for j, m in enumerate(MODELS)}

    mse = {k: {m: round(float(np.mean((P[k][:, j] - Q[:, j]) ** 2)), 4) for j, m in enumerate(MODELS)} for k in GRIDS}
    mse["baseline_predict_mean"] = {m: round(float(np.var(Q[:, j])), 4) for j, m in enumerate(MODELS)}

    summary = {
        "what": "RouterBench-style predictive routers (kNN, MLP) on mixed_ua_v7 dev, offline from pool scores",
        "split": args.split, "n_items": n, "n_per_bucket": {b: int((buckets == b).sum()) for b in BUCKETS},
        "encoder": args.encoder, "folds": args.folds, "seed": args.seed, "git_commit": git_commit(),
        "cost": "latency of the chosen model, seconds per request (screening runs, 4 Sep 2026)",
        "quality": "macro mean over knowledge, translate, alignment, instruct",
        "cost_range": [round(c_min, 4), round(c_max, 4)],
        "points": {k: {"cost_s": round(c, 4), "macro": round(q, 4)} for k, (c, q) in points.items()},
        "aiq": {k: round(v, 4) for k, v in aiq_rows.items()},
        "max_quality_end": {k: {"cost_s": round(curves[k][-1][1], 4), "macro": round(curves[k][-1][2], 4)}
                            for k in GRIDS},
        "comparisons_macro": comparisons,
        "per_bucket": per_bucket,
        "prediction_mse": mse,
        "hyper_parameters_per_fold": chosen_hp,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "curves.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "lambda", "cost_s", "macro"])
        for kind in GRIDS:
            for lam, c, q in curves[kind]:
                w.writerow([kind, lam, round(c, 4), round(q, 4)])
        for name, (c, q) in points.items():
            w.writerow([name, "", round(c, 4), round(q, 4)])
    print(json.dumps({k: summary[k] for k in ("n_items", "points", "aiq", "max_quality_end", "comparisons_macro")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
