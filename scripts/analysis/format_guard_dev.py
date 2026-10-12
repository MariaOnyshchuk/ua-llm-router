#!/usr/bin/env python3
"""Offline estimate of router/format_guard.py on the instruct bucket of mixed_ua_v7 dev.

Applies the guard to stored answers and rescores them. No model is called. Rows are
filtered to the dev split before anything is computed.

Also reports a what-if with three checks in scripts/ifeval_check.py repaired for this
analysis only (the scorer itself is not changed):
  - detectable_format:number_bullet_lists is registered under detectable_content, so it never passes;
  - number_placeholders matches Latin capitals only, so "[місто]" is not counted;
  - repeat_prompt compares with the whole request instead of `prompt_to_repeat`.

  PYTHONPATH=. python scripts/analysis/format_guard_dev.py
"""

from __future__ import annotations

import ast
import glob
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from router.format_guard import apply, detect  # noqa: E402
from scripts.ifeval_check import BULLET_RE, _kwargs_list, _relation_ok, check_instruction, score_ifeval  # noqa: E402

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


def item_score(row: dict, text: str, repaired_checks: bool) -> float | None:
    ids = instruction_ids(row)
    if not ids:
        return None  # hand-written items without IFEval metadata keep their stored score
    if not repaired_checks:
        return score_ifeval(text, row.get("ifeval_instruction_id_list"), row.get("ifeval_kwargs"),
                            row.get("prompt", ""))["score"]
    flags = []
    for iid, kw in zip(ids, _kwargs_list(row.get("ifeval_kwargs"), len(ids))):
        if iid == "detectable_format:number_bullet_lists":
            ok = len(BULLET_RE.findall(text)) >= int(kw.get("num_bullets") or 1)
        elif iid == "detectable_content:number_placeholders":
            ok = _relation_ok(len(re.findall(r"\[[^\[\]\n]{1,40}\]", text)), int(kw.get("num_placeholders") or 1),
                              str(kw.get("relation") or "at least"))
        elif iid == "combination:repeat_prompt":
            target = str(kw.get("prompt_to_repeat") or "").strip()
            ok = bool(target) and text.strip().startswith(target)
        else:
            ok = check_instruction(iid, text, kw, row.get("prompt", ""))
        flags.append(bool(ok))
    return sum(flags) / len(flags)


def main() -> None:
    bench = {r["id"]: r for r in rows(str(ROOT / "benchmarks/mixed_ua_v7.jsonl"))}
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
            for tag, value, repaired in (("raw", text, False), ("guard", guarded, False),
                                         ("raw_fixed", text, True), ("guard_fixed", guarded, True)):
                s = item_score(row, value, repaired)
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
        "repaired_checks_what_if": {
            "per_system": {name: {"raw": round(st.mean(scores[(name, "raw_fixed")].values()), 4),
                                  "guard": round(st.mean(scores[(name, "guard_fixed")].values()), 4)} for name in RAW},
            "router raw vs Mamay-12B raw": paired(("router", "raw_fixed"), ("mamay12", "raw_fixed")),
            "router + guard vs Mamay-12B + guard": paired(("router", "guard_fixed"), ("mamay12", "guard_fixed")),
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
    w = out["repaired_checks_what_if"]
    print("what-if, three checks repaired:", w["per_system"])
    for title in ("router raw vs Mamay-12B raw", "router + guard vs Mamay-12B + guard"):
        r = w[title]
        print(f"  {title}: {r['a']:.3f} vs {r['b']:.3f}, {r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] {r['verdict']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
