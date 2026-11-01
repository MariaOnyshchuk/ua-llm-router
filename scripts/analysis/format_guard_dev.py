#!/usr/bin/env python3
"""Offline estimate of router/format_guard.py on the instruct bucket of mixed_ua_v7 dev.

Applies the guard to stored answers and rescores them. No model is called. Rows are
filtered to the dev split before anything is computed.

  PYTHONPATH=. python scripts/analysis/format_guard_dev.py
"""

from __future__ import annotations

import ast
import glob
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from router.format_guard import apply, detect  # noqa: E402
from scripts.ifeval_check import score_ifeval  # noqa: E402

RAW = {
    "router": "results/v7_dev_router/router_matrix_v2_fewshot_*Z.jsonl",
    "mamay4": "results/v6_screen/mamay4_*Z.jsonl",
    "aya": "results/v6_screen/aya_*Z.jsonl",
    "lapa": "results/v6_screen/lapa_*Z.jsonl",
    "mamay12": "results/v7_pool/mamay12_*Z.jsonl",
}
STORED = {
    "router": ("results/v7_dev_router/scores_router_dev_detail.jsonl", "router_matrix_v2_fewshot"),
    "mamay12": ("results/v7_dev_router/scores_mamay12_dev_detail.jsonl", "mamay12"),
    "mamay4": ("results/v7_pool/scores_specialists_detail.jsonl", "mamay4"),
    "aya": ("results/v7_pool/scores_specialists_detail.jsonl", "aya"),
    "lapa": ("results/v7_pool/scores_specialists_detail.jsonl", "lapa"),
}
LABELS = {"no_comma": "punctuation:no_comma", "quotes": "startend:quotation",
          "lower": "change_case:english_lowercase", "upper": "change_case:english_capital"}


def rows(path: str):
    for line in open(path, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            yield json.loads(line)


def instruction_ids(row: dict) -> list[str]:
    ids = row.get("ifeval_instruction_id_list")
    if not ids:
        return []
    return list(ast.literal_eval(ids)) if isinstance(ids, str) else list(ids)


def item_score(row: dict, text: str) -> float | None:
    if not instruction_ids(row):
        return None  # hand-written items without IFEval metadata keep their stored score
    return score_ifeval(text, row.get("ifeval_instruction_id_list"), row.get("ifeval_kwargs"),
                        row.get("prompt", ""))["score"]


def main() -> None:
    bench = {r["id"]: r for r in rows(str(ROOT / "benchmarks/tasks_public_4401.jsonl"))}
    dev_all = {i for i, r in bench.items() if r.get("split") == "dev"}
    dev = sorted(i for i in dev_all if bench[i]["bucket"] == "instruct")
    keep = set(dev)

    raw: dict[str, dict[str, dict]] = {}
    for name, pattern in RAW.items():
        path = glob.glob(str(ROOT / pattern))
        if not path:
            raise SystemExit(f"raw answers not found on this machine: {pattern} (they are local only)")
        raw[name] = {r["id"]: r for r in rows(path[0]) if r["id"] in keep}
    stored: dict[tuple[str, str], float] = {}
    for name, (path, system) in STORED.items():
        for r in rows(str(ROOT / path)):
            if r["id"] in keep and r["system"] == system:
                stored[(name, r["id"])] = float(r["score"])

    detection = {}
    for guard_name, label in LABELS.items():
        labelled = {i for i in dev if label in instruction_ids(raw["mamay12"][i])}
        found = {i for i in dev if guard_name in detect(bench[i]["prompt"])}
        detection[guard_name] = {"labelled": len(labelled), "found": len(labelled & found),
                                 "missed": len(labelled - found), "unlabelled_hits": len(found - labelled)}
    other = Counter(bench[i]["bucket"] for i in dev_all if bench[i]["bucket"] != "instruct" and detect(bench[i]["prompt"]))

    scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    changed: Counter = Counter()
    for name in RAW:
        for i in dev:
            row = raw[name][i]
            text = row.get("content") or ""
            guarded, made = apply(row["prompt"], text)
            changed[name] += bool(made)
            for tag, value in (("raw", text), ("guard", guarded)):
                s = item_score(row, value)
                scores[(name, tag)][i] = stored[(name, i)] if s is None else s
    for name in RAW:
        assert all(abs(scores[(name, "raw")][i] - stored[(name, i)]) < 1e-9 for i in dev), f"{name}: rescoring drifted"

    rng = random.Random(42)

    def paired(a: tuple[str, str], b: tuple[str, str], n: int = 5000) -> dict:
        d = [scores[a][i] - scores[b][i] for i in dev]
        means = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(n))
        ci = [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)]
        return {"a": round(st.mean(scores[a].values()), 4), "b": round(st.mean(scores[b].values()), 4),
                "diff": round(st.mean(d), 4), "ci95": ci,
                "verdict": "A higher" if ci[0] > 0 else "A lower" if ci[1] < 0 else "not distinguishable"}

    out = {
        "suite": "mixed_ua_v7", "split": "dev", "bucket": "instruct", "n": len(dev),
        "note": "Offline: the guard is applied to stored dev answers, which are then rescored. Constraints are "
                "detected from the request text only. The patterns were written against these same dev items.",
        "detection_vs_ifeval_labels": detection,
        "guard_hits_in_other_dev_buckets": dict(other),
        "per_system": {name: {**paired((name, "guard"), (name, "raw")), "answers_changed": changed[name]} for name in RAW},
        "comparisons": {
            "router raw vs Mamay-12B raw": paired(("router", "raw"), ("mamay12", "raw")),
            "router + guard vs Mamay-12B raw": paired(("router", "guard"), ("mamay12", "raw")),
            "router + guard vs Mamay-12B + guard": paired(("router", "guard"), ("mamay12", "guard")),
        },
    }
    path = ROOT / "results/analysis/v7_dev_format_guard.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("detection:", detection)
    print("guard hits outside instruct (dev):", dict(other))
    print("\n| system | raw | + guard | difference | 95% interval | answers changed |\n|---|---:|---:|---:|---|---:|")
    for name, r in out["per_system"].items():
        print(f"| {name} | {r['b']:.3f} | {r['a']:.3f} | {r['diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['answers_changed']} |")
    for title, r in out["comparisons"].items():
        print(f"{title}: {r['a']:.3f} vs {r['b']:.3f}, {r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] {r['verdict']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
