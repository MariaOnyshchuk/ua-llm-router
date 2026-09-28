#!/usr/bin/env python3
"""Wrap a locally acquired AgentComA-UK translation as planner-screening rows.

The planner sees only ``prompt`` (the composition question). ``oracle_plan`` is
diagnostic: filter with knowledge, then compute a number with instruct.
It is not shown to the model.

AgentCoMa is gated and does not publish a reusable dataset license. Keep the
source and generated prompt JSONL local; publish only prompt-free metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "benchmarks" / "agentcoma_uk_50items_merged.jsonl"
DEFAULT_OUT = ROOT / "benchmarks" / "agentcoma_uk_50_composite.jsonl"

# Known local source files -> expected item count and provenance tag.
# ``agentcoma_uk_200_merged`` = 50 dev items (claude_manual_v1) + 150 test
# items (claude_manual_v2); each row carries ``split`` in {dev, test}.
KNOWN_SOURCES: dict[str, tuple[int, str]] = {
    "agentcoma_uk_50items_merged.jsonl": (50, "agentcoma_uk_50_claude_manual_v1"),
    "agentcoma_uk_150_test_translated.jsonl": (150, "agentcoma_uk_150_test_claude_manual_v2"),
    "agentcoma_uk_200_merged.jsonl": (200, "agentcoma_uk_200_claude_manual_v1v2"),
}

ORACLE_FILTER = (
    "Визнач, які об'єкти, факти чи категорії з запиту потрібні для відповіді. "
    "Не виконуй арифметику. Коротко перелічи лише релевантне."
)
ORACLE_COMPUTE = (
    "За запитом користувача та попереднім кроком обчисли числову відповідь.\n"
    "Запит:\n{{user}}\n\nФільтр:\n{{filter.content}}\n\n"
    "Поверни лише число, без пояснень."
)


def wrap_row(
    row: dict[str, Any], provenance: str = "agentcoma_uk_50_claude_manual_v1"
) -> dict[str, Any]:
    prompt = str(row["question_composition_uk"]).strip()
    compute = ORACLE_COMPUTE.replace("{{user}}", prompt)
    # Rubrics let score_composite_results.py grade these rows: the filter step
    # must name at least one accepted commonsense answer; the final step must
    # end with the gold number (AgentCoMa's own check is number containment).
    filter_rubric: dict[str, Any] = {
        "type": "contains_any",
        "values": [str(x) for x in (row.get("answers_commonsense_uk") or []) if str(x).strip()],
    }
    compute_rubric: dict[str, Any] = {"type": "numeric", "value": row.get("answer_composition")}
    return {
        "id": row["id"],
        "bucket": "agentcoma",
        "family": row.get("category") or "",
        "operation_type": row.get("operation_type") or "",
        # dev = original 50 items (source dev split); test = 150 items from the
        # source test split. Older 50-item files have no split field.
        "split": row.get("split") or "dev",
        "prompt": prompt,
        "provenance": provenance,
        "answer_composition": row.get("answer_composition"),
        "oracle_plan": {
            "steps": [
                {
                    "id": "filter",
                    "intent": "knowledge",
                    "depends_on": [],
                    "prompt": ORACLE_FILTER,
                    "rubric": filter_rubric,
                },
                {
                    "id": "compute",
                    "intent": "instruct",
                    "depends_on": ["filter"],
                    "prompt": compute,
                    "rubric": compute_rubric,
                },
            ]
        },
        "final_step": "compute",
    }


def load_source(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DEFAULT_SRC))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--expect",
        type=int,
        default=0,
        help="expected item count (default: inferred from a known source filename)",
    )
    parser.add_argument(
        "--provenance",
        default="",
        help="provenance tag (default: inferred from a known source filename)",
    )
    args = parser.parse_args()
    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    known = KNOWN_SOURCES.get(src.name)
    expect = args.expect or (known[0] if known else 0)
    provenance = args.provenance or (known[1] if known else f"agentcoma_uk_{src.stem}")
    rows = [wrap_row(x, provenance=provenance) for x in load_source(src)]
    if expect and len(rows) != expect:
        raise SystemExit(f"expected {expect} AgentComA items, got {len(rows)} from {src}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate ids in {src}")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    splits = sorted({r["split"] for r in rows})
    counts = {s: sum(1 for r in rows if r["split"] == s) for s in splits}
    print(f"wrote {len(rows)} items to {out} (split counts: {counts})")


if __name__ == "__main__":
    main()
