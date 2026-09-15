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

ORACLE_FILTER = (
    "Визнач, які об'єкти, факти чи категорії з запиту потрібні для відповіді. "
    "Не виконуй арифметику. Коротко перелічи лише релевантне."
)
ORACLE_COMPUTE = (
    "За запитом користувача та попереднім кроком обчисли числову відповідь.\n"
    "Запит:\n{{user}}\n\nФільтр:\n{{filter.content}}\n\n"
    "Поверни лише число, без пояснень."
)


def wrap_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row["question_composition_uk"]).strip()
    compute = ORACLE_COMPUTE.replace("{{user}}", prompt)
    return {
        "id": row["id"],
        "bucket": "agentcoma",
        "family": row.get("category") or "",
        "operation_type": row.get("operation_type") or "",
        "prompt": prompt,
        "provenance": "agentcoma_uk_50_claude_manual_v1",
        "answer_composition": row.get("answer_composition"),
        "oracle_plan": {
            "steps": [
                {
                    "id": "filter",
                    "intent": "knowledge",
                    "depends_on": [],
                    "prompt": ORACLE_FILTER,
                },
                {
                    "id": "compute",
                    "intent": "instruct",
                    "depends_on": ["filter"],
                    "prompt": compute,
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
    args = parser.parse_args()
    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    rows = [wrap_row(x) for x in load_source(src)]
    if len(rows) != 50:
        raise SystemExit(f"expected 50 AgentComA items, got {len(rows)} from {src}")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} items to {out}")


if __name__ == "__main__":
    main()
