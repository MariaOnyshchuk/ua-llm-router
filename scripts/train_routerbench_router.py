#!/usr/bin/env python3
"""Train the RouterBench-style kNN and MLP routers on mixed_ua_v7 dev and save them for live use.

Same data, encoder and hyper-parameter grid as scripts/analysis/routerbench_dev.py: e5-large
embeddings of the v7 dev prompts, per-model scores from results/v7_pool. Hyper-parameters are
chosen by 5-fold MSE on all dev items, then each regressor is refit on all dev items. The saved
router picks the model with the highest predicted score (max-quality end). Only the dev split is
read; v7 test stays closed.

  .venv/bin/python scripts/train_routerbench_router.py

Writes router/artifacts/{knn,mlp}/model.pkl and config.json (with the git commit, the chosen
hyper-parameters and the training-set fingerprint).

With --extra-suite / --extra-scores the node training set is added (benchmarks/nodes_mixed_600_train.jsonl,
scores from score_composite_results.py, `final_score` per model), and --suffix names the output folders:

  .venv/bin/python scripts/train_routerbench_router.py \
    --extra-suite benchmarks/nodes_mixed_600_train.jsonl \
    --extra-scores results/nodes_mixed_600_train/scores_all_detail.jsonl --suffix _nodes

With --only-extra the v7 dev items are left out and the routers learn from the node set alone:

  .venv/bin/python scripts/train_routerbench_router.py --only-extra \
    --extra-suite benchmarks/nodes_mixed_600_train.jsonl \
    --extra-scores results/nodes_mixed_600_train/scores_all_detail.jsonl --suffix _nodes_only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router.learned import ARTIFACT_ROOT, QualityArgmaxRouter  # noqa: E402

_spec = importlib.util.spec_from_file_location("routerbench_dev", ROOT / "scripts/analysis/routerbench_dev.py")
rb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rb)


def cv_mse(kind: str, hp: dict, X: np.ndarray, Q: np.ndarray, folds: list[np.ndarray]) -> float:
    errs = []
    for te in folds:
        tr = np.setdiff1d(np.arange(len(X)), te)
        m = rb.make_model(kind, hp).fit(X[tr], Q[tr])
        errs.append(float(np.mean((np.clip(m.predict(X[te]), 0, 1) - Q[te]) ** 2)))
    return float(np.mean(errs))


def load_extra(suite: Path, scores: Path) -> tuple[list[dict], np.ndarray]:
    """Node items that all three models answered; their bucket is the skill (math, code, ...)."""
    items = {r["id"]: r for r in rb.load_jsonl(suite)}
    score: dict[str, dict[str, float]] = {}
    for r in rb.load_jsonl(scores):
        if r.get("id") in items and r.get("system") in rb.MODELS:
            if not r.get("complete", True):
                raise SystemExit(f"{scores}: {r['system']} {r['id']} did not complete")
            score.setdefault(r["id"], {})[r["system"]] = float(r.get("final_score") or 0.0)
    ids = sorted(i for i in items if len(score.get(i, {})) == len(rb.MODELS))
    if len(ids) != len(items):
        raise SystemExit(f"{scores}: {len(items) - len(ids)} items lack a score from every model")
    return [items[i] for i in ids], np.array([[score[i][m] for m in rb.MODELS] for i in ids])


def folds_by_group(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Folds stratified by bucket (v7) or skill (node items). v7 buckets come first, in the order of
    routerbench_dev.stratified_folds, so a v7-only run gives the same folds as before."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    order = [b for b in rb.BUCKETS if b in set(groups)] + sorted(set(groups) - set(rb.BUCKETS))
    for g in order:
        idx = np.where(groups == g)[0]
        rng.shuffle(idx)
        for n, i in enumerate(idx):
            folds[n % k].append(int(i))
    return [np.array(sorted(f)) for f in folds]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--suite", type=Path, default=ROOT / "benchmarks/tasks_public_4401.jsonl")
    p.add_argument("--scores", type=Path, default=ROOT / "results/v7_pool/scores_specialists_detail.jsonl")
    p.add_argument("--encoder", default="intfloat/multilingual-e5-large")
    p.add_argument("--encoder-revision", default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache", type=Path, default=ROOT / ".cache/routerbench")
    p.add_argument("--kinds", default="knn,mlp")
    p.add_argument("--extra-suite", type=Path, default=None)
    p.add_argument("--extra-scores", type=Path, default=None)
    p.add_argument("--suffix", default="", help="output folder router/artifacts/<kind><suffix>")
    p.add_argument("--only-extra", action="store_true", help="train on the extra suite alone, without v7 dev")
    args = p.parse_args()
    if args.only_extra and not args.extra_suite:
        p.error("--only-extra needs --extra-suite and --extra-scores")
    args.suite, args.scores = args.suite.resolve(), args.scores.resolve()
    if args.extra_suite:
        args.extra_suite, args.extra_scores = args.extra_suite.resolve(), args.extra_scores.resolve()

    if args.only_extra:
        rows, Q = [], np.zeros((0, len(rb.MODELS)))
    else:
        rows, Q, _C = rb.load_dev(args.suite, args.scores, "dev")
    n_v7 = len(rows)
    if args.extra_suite:
        extra_rows, extra_Q = load_extra(args.extra_suite, args.extra_scores)
        rows, Q = rows + extra_rows, np.vstack([Q, extra_Q])
    buckets = np.array([r["bucket"] for r in rows])
    X = rb.embed([r["prompt"] for r in rows], args.encoder, args.cache)
    ids_hash = hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()[:16]
    folds = folds_by_group(buckets, args.folds, args.seed)
    print(f"{len(rows)} items ({n_v7} v7 dev), embeddings {X.shape}", file=sys.stderr)

    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        scored = [(cv_mse(kind, hp, X, Q, folds), hp) for hp in rb.GRIDS[kind]]
        best_mse, best = min(scored, key=lambda t: t[0])
        router = QualityArgmaxRouter(rb.make_model(kind, best).fit(X, Q), rb.MODELS)
        out = ARTIFACT_ROOT / f"{kind}{args.suffix}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.pkl").write_bytes(pickle.dumps(router))
        import sklearn

        config = {
            "kind": f"{kind}{args.suffix}",
            "router": "QualityArgmaxRouter (max-quality end, lambda = inf)",
            "encoder": "e5",
            "encoder_name": args.encoder,
            "encoder_revision": args.encoder_revision,
            "models": list(rb.MODELS),
            "default": "aya",
            "hyper_parameters": best,
            "cv_mse": round(best_mse, 4),
            "grid": [{"hp": hp, "cv_mse": round(m, 4)} for m, hp in scored],
            "train": {"suite": None if args.only_extra else str(args.suite.relative_to(ROOT)),
                      "split": None if args.only_extra else "dev", "n_items": len(rows),
                      "ids_sha256_16": ids_hash,
                      "scores": None if args.only_extra else str(args.scores.relative_to(ROOT)),
                      "n_v7_dev": n_v7,
                      "extra_suite": str(args.extra_suite.relative_to(ROOT)) if args.extra_suite else None,
                      "extra_scores": str(args.extra_scores.relative_to(ROOT)) if args.extra_scores else None},
            "sklearn": sklearn.__version__,
            "git_commit": rb.git_commit(),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        share = {m: round(float((router.predict(X) == m).mean()), 3) for m in rb.MODELS}
        print(f"{kind}: {best} cv_mse={best_mse:.4f} train share={share} -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
