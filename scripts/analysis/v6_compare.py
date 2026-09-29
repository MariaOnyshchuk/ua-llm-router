#!/usr/bin/env python3
"""Paired bootstrap comparisons and routing diagnostics on mixed_ua_v6.

Reads per-item ``scores_detail.jsonl`` files (fields: system, id, bucket, score,
optional router.model and detail). One pass on v6, so no repeat averaging.

Claim buckets are knowledge, translate, alignment, instruct. chat and code have 19
items each and are reported as appendix rows only. Intervals are paired percentile
bootstraps over items, stratified by bucket (10000 resamples, seed 42). If an
interval contains 0 the verdict is "not distinguishable".

  compare  A - B per bucket, macro over the 4 claim buckets, and micro
  route    for a router run: routing accuracy against the best solo model per
           item, the score of a route oracle, and the loss decomposition
  parse    unparsed answers and extractor mix (alignment) per system

Examples:
  python scripts/analysis/v6_compare.py compare --detail results/v6_solos/scores_detail.jsonl \\
      --a mamay4 --b mamay12
  python scripts/analysis/v6_compare.py route --detail results/v6_solos/scores_detail.jsonl \\
      results/v6_rules/scores_detail.jsonl --router router_matrix_v2_fewshot \\
      --solos mamay4 lapa aya --reference mamay12
  python scripts/analysis/v6_compare.py parse --detail results/v6_solos/scores_detail.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLAIM = ("knowledge", "translate", "alignment", "instruct")
APPENDIX = ("chat", "code")


def load(paths: list[str]) -> dict[str, dict[str, dict]]:
    """system -> id -> row. Refuses duplicate (system, id) so a repeat is never averaged silently."""
    data: dict[str, dict[str, dict]] = defaultdict(dict)
    for name in paths:
        p = Path(name)
        p = p if p.is_absolute() else ROOT / p
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "score" not in row:
                continue
            if row["id"] in data[row["system"]]:
                raise SystemExit(f"duplicate ({row['system']}, {row['id']}) in {p}")
            data[row["system"]][row["id"]] = row
    return data


def paired(data, a: str, b: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    for s in (a, b):
        if s not in data:
            raise SystemExit(f"system {s!r} not found; have {sorted(data)}")
    ids = sorted(set(data[a]) & set(data[b]))
    if len(ids) < min(len(data[a]), len(data[b])):
        print(f"warning: {len(ids)} shared ids of {len(data[a])}/{len(data[b])}", file=sys.stderr)
    out: dict[str, list] = defaultdict(lambda: ([], []))
    for i in ids:
        bucket = data[a][i]["bucket"]
        out[bucket][0].append(data[a][i]["score"])
        out[bucket][1].append(data[b][i]["score"])
    return {k: (np.array(v[0], float), np.array(v[1], float)) for k, v in out.items()}


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "A higher"
    if hi < 0:
        return "A lower"
    return "not distinguishable"


def compare_rows(data, a: str, b: str, n_boot: int, seed: int) -> list[dict]:
    pairs = paired(data, a, b)
    rng = np.random.default_rng(seed)
    idx = {k: rng.integers(0, len(v[0]), size=(n_boot, len(v[0]))) for k, v in pairs.items()}
    boot_diff = {k: (pairs[k][0] - pairs[k][1])[idx[k]].mean(axis=1) for k in pairs}
    rows = []

    def add(label, point_a, point_b, boots, n, note=""):
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"scope": label, "n": n, "a": round(point_a, 4), "b": round(point_b, 4),
                     "diff": round(point_a - point_b, 4), "ci_low": round(float(lo), 4),
                     "ci_high": round(float(hi), 4), "verdict": verdict(lo, hi), "note": note})

    for k in CLAIM + APPENDIX:
        if k in pairs:
            x, y = pairs[k]
            add(k, x.mean(), y.mean(), boot_diff[k], len(x), "appendix only" if k in APPENDIX else "")
    have = [k for k in CLAIM if k in pairs]
    add("macro (4 claim buckets)", np.mean([pairs[k][0].mean() for k in have]),
        np.mean([pairs[k][1].mean() for k in have]), np.mean([boot_diff[k] for k in have], axis=0),
        sum(len(pairs[k][0]) for k in have))
    # micro over claim items: weight buckets by size, resample within bucket
    n_tot = sum(len(pairs[k][0]) for k in have)
    micro_boot = sum(boot_diff[k] * len(pairs[k][0]) for k in have) / n_tot
    add("micro (claim items)", sum(pairs[k][0].sum() for k in have) / n_tot,
        sum(pairs[k][1].sum() for k in have) / n_tot, micro_boot, n_tot)
    return rows


def table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def cmd_compare(args) -> None:
    data = load(args.detail)
    rows = compare_rows(data, args.a, args.b, args.n_boot, args.seed)
    print(f"A = {args.a}, B = {args.b}; diff = A - B; paired bootstrap 95%, n_boot={args.n_boot}\n")
    print(table(rows, ["scope", "n", "a", "b", "diff", "ci_low", "ci_high", "verdict", "note"]))
    if args.out:
        Path(args.out).write_text(json.dumps({"a": args.a, "b": args.b, "rows": rows}, indent=2), encoding="utf-8")


def cmd_route(args) -> None:
    data = load(args.detail)
    r = args.router
    if r not in data:
        raise SystemExit(f"router system {r!r} not found; have {sorted(data)}")
    solos = args.solos
    ids = sorted(set(data[r]).intersection(*[set(data[s]) for s in solos]))
    rows = []
    for bucket in CLAIM + APPENDIX:
        bids = [i for i in ids if data[r][i]["bucket"] == bucket]
        if not bids:
            continue
        rs = np.array([data[r][i]["score"] for i in bids])
        mat = np.array([[data[s][i]["score"] for s in solos] for i in bids])
        best = mat.max(axis=1)
        routed = [data[r][i].get("router", {}).get("model") for i in bids]
        # routed model's own solo score on the item (same decode settings), when it is a listed solo
        own = np.array([mat[k, solos.index(m)] if m in solos else np.nan for k, m in enumerate(routed)])
        acc = float(np.nanmean(own >= best - 1e-9)) if not np.all(np.isnan(own)) else float("nan")
        share = Counter(routed)
        fixed = {s: float(mat[:, j].mean()) for j, s in enumerate(solos)}
        best_fixed = max(fixed, key=fixed.get)
        row = {"scope": bucket, "n": len(bids), "router": round(float(rs.mean()), 4),
               "route_oracle": round(float(best.mean()), 4),
               "best_fixed_specialist": f"{best_fixed} {fixed[best_fixed]:.4f}",
               "routing_accuracy": round(acc, 4),
               "routed_to": ", ".join(f"{m}:{c}" for m, c in share.most_common()),
               "note": "appendix only" if bucket in APPENDIX else ""}
        if args.reference and args.reference in data:
            row["reference"] = round(float(np.mean([data[args.reference][i]["score"] for i in bids])), 4)
        rows.append(row)
    print(f"router = {r}; specialists = {solos}; route oracle = per-item best specialist\n")
    cols = ["scope", "n", "router", "route_oracle", "best_fixed_specialist", "routing_accuracy", "routed_to"]
    if args.reference:
        cols.insert(3, "reference")
    print(table(rows, cols))
    have = [x for x in rows if x["scope"] in CLAIM]
    if have:
        print("\nmacro over claim buckets: router {:.4f}, route oracle {:.4f}".format(
            np.mean([x["router"] for x in have]), np.mean([x["route_oracle"] for x in have])))
    print("\nrouting_accuracy = share of items where the routed model's solo score equals the best specialist score "
          "(ties count as correct); it needs the routed model to be one of --solos.")
    if args.out:
        Path(args.out).write_text(json.dumps({"router": r, "solos": solos, "rows": rows}, indent=2), encoding="utf-8")


def cmd_parse(args) -> None:
    data = load(args.detail)
    rows = []
    for system, items in sorted(data.items()):
        for bucket in ("knowledge", "alignment"):
            sub = [x for x in items.values() if x["bucket"] == bucket]
            if not sub:
                continue
            got = [re.search(r"got=(\S*)", x.get("detail") or "") for x in sub]
            unparsed = sum(1 for m in got if m and m.group(1) in ("?", "", "none"))
            anchored = sum(1 for x in sub if "method=anchored" in (x.get("detail") or ""))
            last_digit = sum(1 for x in sub if "method=last_digit" in (x.get("detail") or ""))
            rows.append({"system": system, "bucket": bucket, "n": len(sub), "unparsed": unparsed,
                         "anchored": anchored, "last_digit": last_digit})
    print(table(rows, ["system", "bucket", "n", "unparsed", "anchored", "last_digit"]))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("compare", cmd_compare), ("route", cmd_route), ("parse", cmd_parse)):
        s = sub.add_parser(name)
        s.add_argument("--detail", nargs="+", required=True, help="scores_detail.jsonl files")
        s.add_argument("--out", default="")
        s.set_defaults(fn=fn)
        if name == "compare":
            s.add_argument("--a", required=True)
            s.add_argument("--b", required=True)
            s.add_argument("--n-boot", type=int, default=10000)
            s.add_argument("--seed", type=int, default=42)
        if name == "route":
            s.add_argument("--router", required=True)
            s.add_argument("--solos", nargs="+", required=True)
            s.add_argument("--reference", default="")
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
