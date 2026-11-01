#!/usr/bin/env python3
"""Compare routers on nodes_mixed_600_test using the stored answers of the three models.

Each model answered every test item once (cluster/jupiter_nodes_mixed_600.sbatch, SPLIT=test). A router only picks
a model per item, so its score on an item is that model's score: every router is judged on the same answers and
no new generation is needed. One pass, no tuning: the routers and rules v2 are frozen at commit 4548da0.

Systems: the three single models; rules v2 from the prompt text (its regexes) and with the item's intent label
(as on plan steps); kNN and MLP trained on tasks_public_4401 dev, and the same with nodes_mixed_600_train added;
the per-item oracle. Differences are paired bootstrap 95% intervals (10 000 resamples over items, seed 42).

Decision rule, fixed before the run: switch to a learned router if the lower end of the interval of
(learned − rules v2) is above −0.02, against both rules variants.

    .venv/bin/python scripts/analysis/node_test_routers.py
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

MODELS = ("mamay4", "lapa", "aya")
SKILLS = ("math", "translate", "instruct", "knowledge", "alignment", "code")
BENCH = ROOT / "benchmarks" / "nodes_mixed_600_test.jsonl"
DETAIL = ROOT / "results" / "nodes_mixed_600_test" / "scores_all_detail.jsonl"
OUT = ROOT / "results" / "nodes_mixed_600_test" / "router_comparison.json"
MARGIN = -0.02


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]


def bootstrap(d: np.ndarray, n: int = 10_000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    items = load_jsonl(BENCH)
    score: dict[str, dict[str, float]] = {}
    for r in load_jsonl(DETAIL):
        if r["system"] in MODELS:
            if not r.get("complete", True):
                raise SystemExit(f"{r['system']} {r['id']}: did not complete")
            score.setdefault(r["id"], {})[r["system"]] = float(r["final_score"])
    missing = [it["id"] for it in items if len(score.get(it["id"], {})) != len(MODELS)]
    if missing:
        raise SystemExit(f"{len(missing)} items lack a score from every model, e.g. {missing[:3]}")
    Q = np.array([[score[it["id"]][m] for m in MODELS] for it in items])
    skills = np.array([it["skill"] for it in items])

    picks: dict[str, list[str]] = {
        "rules_v2_text": [route_intent(it["prompt"], profile="v2").model for it in items],
        "rules_v2_intent": [route_intent(it["prompt"], profile="v2", intent_hint=it["label"]).model for it in items],
    }
    spec = importlib.util.spec_from_file_location("routerbench_dev", ROOT / "scripts/analysis/routerbench_dev.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    X = rb.embed([it["prompt"] for it in items], "intfloat/multilingual-e5-large", ROOT / ".cache/routerbench")
    for kind in ("knn", "mlp", "knn_nodes", "mlp_nodes"):
        router = pickle.loads((ROOT / "router/artifacts" / kind / "model.pkl").read_bytes())
        picks[kind] = list(router.predict(X))

    per_item = {m: Q[:, j] for j, m in enumerate(MODELS)}
    for name, chosen in picks.items():
        per_item[name] = np.array([Q[i, MODELS.index(m)] if m in MODELS else Q[i, MODELS.index("aya")]
                                   for i, m in enumerate(chosen)])
    per_item["oracle"] = Q.max(axis=1)

    summary = {
        name: {"all": round(float(v.mean()), 4), **{s: round(float(v[skills == s].mean()), 4) for s in SKILLS}}
        for name, v in per_item.items()
    }
    shares = {name: {m: round(sum(c == m for c in chosen) / len(chosen), 3) for m in MODELS} for name, chosen in picks.items()}
    comparisons = {}
    for learned in ("knn", "mlp", "knn_nodes", "mlp_nodes"):
        for rules in ("rules_v2_text", "rules_v2_intent"):
            d = per_item[learned] - per_item[rules]
            lo, hi = bootstrap(d)
            comparisons[f"{learned} - {rules}"] = {
                "diff": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "not_worse_by_more_than_0.02": lo > MARGIN,
                "per_skill": {s: round(float(d[skills == s].mean()), 4) for s in SKILLS},
            }
    decision = {
        learned: all(comparisons[f"{learned} - {r}"]["not_worse_by_more_than_0.02"] for r in ("rules_v2_text", "rules_v2_intent"))
        for learned in ("knn", "mlp", "knn_nodes", "mlp_nodes")
    }
    out = {
        "bench": str(BENCH.relative_to(ROOT)), "n_items": len(items), "freeze_commit": "4548da0",
        "decision_rule": "switch if the lower end of the 95% paired bootstrap interval of (learned - rules v2) > -0.02, against both rules variants",
        "scores": summary, "model_share": shares, "comparisons": comparisons, "passes_rule": decision,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'system':18}" + "".join(f"{c:>10}" for c in ("all",) + SKILLS))
    for name, row in summary.items():
        print(f"{name:18}" + "".join(f"{row[c]:10.3f}" for c in ("all",) + SKILLS))
    print()
    for k, v in comparisons.items():
        print(f"{k:32} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  passes={v['not_worse_by_more_than_0.02']}")
    print("\npasses the rule against both rules variants:", decision)


if __name__ == "__main__":
    main()
