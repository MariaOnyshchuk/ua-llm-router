#!/usr/bin/env python3
"""Development-set planner ablation for composite v2.

Offline profiles (no GPU):
  - template  — constrained workflow selector

Live profiles (require serving backends; optional --live):
  - hybrid, hybrid_minimal, hybrid_fewshot, hybrid_mamay12

Always writes a freeze recommendation from offline + any live scores present.
Tuning must use --split dev only; never open test for prompt edits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "results" / "week_12_composite_v2_dev_ablation"


def run_offline(split: str, out_dir: Path) -> dict[str, Any]:
    out = out_dir / f"selector_{split}.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_composite_selector.py"),
            "--bench",
            str(ROOT / "benchmarks" / "mixed_ua_composite_v2.jsonl"),
            "--split",
            split,
            "--out",
            str(out),
        ],
        check=True,
    )
    return json.loads(out.read_text(encoding="utf-8"))


def run_live(profile: str, split: str, out_dir: Path, repeats: int) -> Path:
    dest = out_dir / f"live_{profile}_{split}"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_composite_router.py"),
            "--bench",
            "benchmarks/mixed_ua_composite_v2.jsonl",
            "--systems",
            profile,
            "--split",
            split,
            "--repeats",
            str(repeats),
            "--out-dir",
            str(dest),
        ],
        check=True,
        cwd=str(ROOT),
    )
    score_out = dest / "scores.json"
    files = sorted(dest.glob(f"{profile}_*.jsonl"))
    if not files:
        raise SystemExit(f"no live traces for {profile}")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "score_composite_results.py"),
            *[str(p) for p in files],
            "--out",
            str(score_out),
        ],
        check=True,
    )
    return score_out


def recommend(offline_dev: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    template_exact = offline_dev["overall"]["plan_exact_match_rate"]
    # Historical v1 hybrid v3 exact workflow match for context.
    hybrid_v3_exact_v1 = 0.639
    winner = "template"
    reasons = [
        f"template exact workflow match on v2/dev = {template_exact}",
        f"v1 hybrid v3 exact workflow match was {hybrid_v3_exact_v1} (different suite; planning headroom)",
    ]
    if live:
        best_live = max(
            live.items(),
            key=lambda kv: (
                (kv[1].get("systems") or {}).get(kv[0], {}).get("plan_exact_match_rate") or 0,
                (kv[1].get("systems") or {}).get(kv[0], {}).get("final_score_mean") or 0,
            ),
        )
        reasons.append(f"best live profile on dev: {best_live[0]}")
        live_exact = (
            (best_live[1].get("systems") or {}).get(best_live[0], {}).get("plan_exact_match_rate")
        )
        if live_exact is not None and live_exact >= template_exact:
            winner = best_live[0]
            reasons.append("live planner matched or beat template exact-match on dev")
        else:
            reasons.append("template remains ahead of live planners on exact workflow match")
    else:
        reasons.append("no live LLM planner scores; freeze template from offline ablation")
    return {
        "frozen_profile": winner,
        "open_test": False,
        "distillation_warranted": False,
        "distillation_reason": (
            "Template already reaches perfect workflow match on the bounded v2 families. "
            "Distill only if freer natural-language prompts break the regex selector."
        ),
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", help="Must stay off test while tuning")
    parser.add_argument("--live", action="store_true", help="Also run LLM planner profiles")
    parser.add_argument(
        "--profiles",
        default="hybrid,hybrid_fewshot,hybrid_minimal",
        help="Live profiles when --live is set",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if "test" in {x.strip() for x in args.split.split(",")}:
        raise SystemExit("refusing to ablate on test; use --split dev (or train)")

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    offline = run_offline(args.split, out_dir)
    live_scores: dict[str, Any] = {}
    if args.live:
        for profile in [p.strip() for p in args.profiles.split(",") if p.strip()]:
            score_path = run_live(profile, args.split, out_dir, args.repeats)
            live_scores[profile] = json.loads(score_path.read_text(encoding="utf-8"))

    decision = recommend(offline, live_scores)
    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": args.split,
        "offline_template": offline,
        "live": {k: v.get("systems", {}).get(k) for k, v in live_scores.items()},
        "decision": decision,
        "historical_reference": {
            "suite": "mixed_ua_composite_v1",
            "hybrid_v3_plan_exact_match": 0.639,
            "hybrid_v3_final": 0.8125,
            "oracle_final": 0.882,
            "rules_direct_final": 0.812,
        },
    }
    path = out_dir / "ablation_summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
