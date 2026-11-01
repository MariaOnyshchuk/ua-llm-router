#!/usr/bin/env python3
"""Compare routers on nodes_mixed_600_test using the stored answers of the three models.

Each model answered every test item once (cluster/jupiter_nodes_mixed_600.sbatch, SPLIT=test). A router only picks
a model per item, so its score on an item is that model's score: every router is judged on the same answers and
no new generation is needed. One pass, no tuning: the routers and rules v2 are frozen at commit 4548da0.

Systems: the three single models; rules v2 from the prompt text (its regexes) and with the item's intent label
(as on plan steps); kNN and MLP trained on tasks_public_4401 dev, and the same with nodes_mixed_600_train added;
the per-item oracle. Differences are paired bootstrap 95% intervals (10 000 resamples over items, seed 42).

Fallbacks are recorded for every router, per skill:
- rules v2 from text: no rule matched, so the request went to the default model (Aya);
- rules v2 with the intent label: the label maps to the default ("chat", which math steps carry);
- learned routers: hard fallbacks (predicted model outside the pool, which sends the request to Aya in
  router/learned.py), exact ties between the top predicted scores (argmax then takes the first model in
  list order, Mamay-4B, so the choice comes from list order, not from the prediction) and near ties (top two
  predicted scores less than 0.01, 0.02 or 0.05 apart). These are the learned routers' counterpart of the
  rules' "no rule matched": items where the router has no real winner.

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

    text_decisions = [route_intent(it["prompt"], profile="v2") for it in items]
    hint_decisions = [route_intent(it["prompt"], profile="v2", intent_hint=it["label"]) for it in items]
    picks: dict[str, list[str]] = {
        "rules_v2_text": [d.model for d in text_decisions],
        "rules_v2_intent": [d.model for d in hint_decisions],
    }
    fallback_flags: dict[str, dict[str, np.ndarray]] = {
        "rules_v2_text": {"no_rule_matched_default_aya": np.array([d.intent == "chat" for d in text_decisions])},
        "rules_v2_intent": {"label_maps_to_default_aya": np.array([d.intent == "chat" for d in hint_decisions])},
    }
    spec = importlib.util.spec_from_file_location("routerbench_dev", ROOT / "scripts/analysis/routerbench_dev.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    X = rb.embed([it["prompt"] for it in items], "intfloat/multilingual-e5-large", ROOT / ".cache/routerbench")
    for kind in ("knn", "mlp", "knn_nodes", "mlp_nodes"):
        router = pickle.loads((ROOT / "router/artifacts" / kind / "model.pkl").read_bytes())
        picks[kind] = list(router.predict(X))
        P = np.sort(router.predict_quality(X), axis=1)
        gap = P[:, -1] - P[:, -2]
        fallback_flags[kind] = {
            "hard_fallback_default_aya": np.array([str(m) not in MODELS for m in picks[kind]]),
            "exact_tie_first_in_list": gap <= 1e-9,
            "gap_under_0.01": gap < 0.01,
            "gap_under_0.02": gap < 0.02,
            "gap_under_0.05": gap < 0.05,
        }

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
    fallbacks = {
        name: {kind: {"n": int(flag.sum()), "share": round(float(flag.mean()), 3),
                      "per_skill": {s: int(flag[skills == s].sum()) for s in SKILLS},
                      "score_on_these_items": round(float(per_item[name][flag].mean()), 4) if flag.any() else None}
               for kind, flag in flags.items()}
        for name, flags in fallback_flags.items()
    }
    out = {
        "bench": str(BENCH.relative_to(ROOT)), "n_items": len(items), "freeze_commit": "4548da0",
        "decision_rule": "switch if the lower end of the 95% paired bootstrap interval of (learned - rules v2) > -0.02, against both rules variants",
        "scores": summary, "model_share": shares, "fallbacks": fallbacks, "comparisons": comparisons, "passes_rule": decision,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'system':18}" + "".join(f"{c:>10}" for c in ("all",) + SKILLS))
    for name, row in summary.items():
        print(f"{name:18}" + "".join(f"{row[c]:10.3f}" for c in ("all",) + SKILLS))
    print()
    for k, v in comparisons.items():
        print(f"{k:32} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  passes={v['not_worse_by_more_than_0.02']}")
    print("\nfallbacks (count of 600; per skill m/t/i/k/a/c):")
    for name, kinds in fallbacks.items():
        for kind, v in kinds.items():
            print(f"  {name:16} {kind:30} {v['n']:4}  " + "/".join(str(v['per_skill'][s]) for s in SKILLS))
    print("\npasses the rule against both rules variants:", decision)


if __name__ == "__main__":
    main()
