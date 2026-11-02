#!/usr/bin/env python3
"""Routers trained on the node set alone, scored on nodes_mixed_600_test from the stored answers.

Added on 2 Nov 2026, after the node test of 1 Nov (scripts/analysis/node_test_routers.py), so this is a
labelled follow-up run, not part of that frozen comparison. knn_nodes_only and mlp_nodes_only were trained with

    .venv/bin/python scripts/train_routerbench_router.py --only-extra \\
      --extra-suite benchmarks/nodes_mixed_600_train.jsonl \\
      --extra-scores results/nodes_mixed_600_train/scores_all_detail.jsonl --suffix _nodes_only

and their hyper-parameters were chosen by 5-fold CV on the train nodes only. The script also refits every
grid setting on all train nodes and reports its test score. That table shows how much the setting matters;
it is not used to choose one (choosing on test would be tuning on test).

    .venv/bin/python scripts/analysis/node_test_nodes_only.py
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from router.intent_rules import route_intent  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load("node_test_routers", ROOT / "scripts/analysis/node_test_routers.py")
rb = _load("routerbench_dev", ROOT / "scripts/analysis/routerbench_dev.py")
MODELS, SKILLS = base.MODELS, base.SKILLS
TRAIN = ROOT / "benchmarks/nodes_mixed_600_train.jsonl"
TRAIN_SCORES = ROOT / "results/nodes_mixed_600_train/scores_all_detail.jsonl"
OUT = ROOT / "results/nodes_mixed_600_test/router_comparison_nodes_only.json"
LEARNED = ("knn", "mlp", "knn_nodes", "mlp_nodes", "knn_nodes_only", "mlp_nodes_only")


def scores_by_id(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in base.load_jsonl(path):
        if r["system"] in MODELS:
            out.setdefault(r["id"], {})[r["system"]] = float(r["final_score"] or 0.0)
    return out


def main() -> None:
    items = base.load_jsonl(base.BENCH)
    S = scores_by_id(base.DETAIL)
    Q = np.array([[S[it["id"]][m] for m in MODELS] for it in items])
    skills = np.array([it["skill"] for it in items])
    embed = lambda ps: rb.embed(ps, "intfloat/multilingual-e5-large", ROOT / ".cache/routerbench")
    X = embed([it["prompt"] for it in items])
    pick_score = lambda picks: np.array([Q[i, MODELS.index(m if m in MODELS else "aya")] for i, m in enumerate(picks)])

    per_item = {m: Q[:, j] for j, m in enumerate(MODELS)}
    text = [route_intent(it["prompt"], profile="v2") for it in items]
    hint = [route_intent(it["prompt"], profile="v2", intent_hint=it["label"]) for it in items]
    per_item["rules_v2_text"] = pick_score([d.model for d in text])
    per_item["rules_v2_intent"] = pick_score([d.model for d in hint])
    ties, shares, configs = {}, {}, {}
    for kind in LEARNED:
        router = pickle.loads((ROOT / "router/artifacts" / kind / "model.pkl").read_bytes())
        picks = [str(m) for m in router.predict(X)]
        per_item[kind] = pick_score(picks)
        P = np.sort(router.predict_quality(X), axis=1)
        gap = P[:, -1] - P[:, -2]
        ties[kind] = {"exact_tie": int((gap <= 1e-9).sum()), "gap_under_0.02": int((gap < 0.02).sum())}
        shares[kind] = {m: round(picks.count(m) / len(picks), 3) for m in MODELS}
        cfg = json.loads((ROOT / "router/artifacts" / kind / "config.json").read_text())
        configs[kind] = {"hp": cfg["hyper_parameters"], "cv_mse": cfg["cv_mse"], "n_train": cfg["train"]["n_items"]}
    per_item["oracle"] = Q.max(axis=1)

    summary = {n: {"all": round(float(v.mean()), 4), **{s: round(float(v[skills == s].mean()), 4) for s in SKILLS}}
               for n, v in per_item.items()}
    comparisons = {}
    for a in ("knn_nodes_only", "mlp_nodes_only"):
        for b in ("rules_v2_text", "rules_v2_intent", "knn_nodes", "mlp_nodes", "knn", "mlp"):
            d = per_item[a] - per_item[b]
            lo, hi = base.bootstrap(d)
            comparisons[f"{a} - {b}"] = {"diff": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}

    # Every grid setting refit on all train nodes; for information only.
    train = sorted(base.load_jsonl(TRAIN), key=lambda r: r["id"])
    St = scores_by_id(TRAIN_SCORES)
    Xt = embed([r["prompt"] for r in train])
    Qt = np.array([[St[r["id"]][m] for m in MODELS] for r in train])
    sweep = {}
    for kind in ("knn", "mlp"):
        for hp in rb.GRIDS[kind]:
            reg = rb.make_model(kind, hp).fit(Xt, Qt)
            picks = [MODELS[j] for j in reg.predict(X).argmax(axis=1)]
            sweep[f"{kind} {json.dumps(hp)}"] = round(float(pick_score(picks).mean()), 4)

    out = {"bench": str(base.BENCH.relative_to(ROOT)), "n_items": len(items), "note": "follow-up after the 1 Nov node test",
           "configs": configs, "scores": summary, "model_share": shares, "ties": ties,
           "comparisons": comparisons, "test_score_per_setting_info_only": sweep}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'system':18}" + "".join(f"{c:>10}" for c in ("all",) + SKILLS))
    for name, row in summary.items():
        print(f"{name:18}" + "".join(f"{row[c]:10.3f}" for c in ("all",) + SKILLS))
    print()
    for k, v in comparisons.items():
        print(f"{k:34} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print("\nconfigs:", configs, "\nties:", ties, "\nshares:", shares)
    print("\ntest score per setting (info only):")
    for k, v in sweep.items():
        print(f"  {k:40} {v:.3f}")


if __name__ == "__main__":
    main()
