#!/usr/bin/env python3
"""Sample ZNO-Eval single-answer questions into our knowledge JSONL schema.

Usage:
  python scripts/sample_zno.py --zno-dir vendor/ZNO --n 24 --seed 42 \\
    --out benchmarks/samples/zno_knowledge_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SUBJECT_FILES = {
    "ukrainian": "ukrainian_raw.json",
    "history": "history_raw.json",
    "math": "math_raw.json",
    "geography": "geography_raw.json",
}


def iter_tasks(zno_dir: Path, subjects: list[str]):
    for subject in subjects:
        path = zno_dir / "tests" / SUBJECT_FILES[subject]
        tests = json.loads(path.read_text(encoding="utf-8"))
        for ti, test in enumerate(tests):
            for task in test.get("tasks") or []:
                yield subject, ti, test.get("link", ""), task


def is_eligible(task: dict) -> bool:
    """Prefer single-letter MCQ without photos / matching grids."""
    if task.get("with_photo"):
        return False
    if task.get("answer_hheader"):
        return False  # matching / sequence
    correct = task.get("correct_answer") or []
    if len(correct) != 1:
        return False
    answers = task.get("answers") or []
    if len(answers) < 2:
        return False
    q = (task.get("question") or "").strip()
    if len(q) < 20:
        return False
    return True


def format_prompt(task: dict) -> str:
    lines = [
        "Дай відповідь на тестове завдання ЗНО. Відповідь — лише літера варіанту (А, Б, В, Г або Д).",
        "",
        task["question"].strip(),
        "",
        "Варіанти:",
    ]
    for a in task.get("answers") or []:
        letter = a.get("answer", "")
        text = (a.get("text") or "").strip()
        lines.append(f"{letter}) {text}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zno-dir", type=Path, default=Path("vendor/ZNO"))
    p.add_argument("--n", type=int, default=24, help="Total items to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--subjects",
        default="ukrainian,history,math,geography",
        help="Comma-separated subjects",
    )
    p.add_argument("--per-subject", type=int, default=0, help="If >0, sample this many per subject")
    p.add_argument("--out", type=Path, default=Path("benchmarks/samples/zno_knowledge_v1.jsonl"))
    args = p.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    pool: dict[str, list[dict]] = {s: [] for s in subjects}

    for subject, ti, link, task in iter_tasks(args.zno_dir, subjects):
        if not is_eligible(task):
            continue
        pool[subject].append(
            {
                "id": f"zno-{subject[:3]}-{ti:02d}-{int(task.get('task_id', 0)):03d}",
                "bucket": "knowledge",
                "prompt": format_prompt(task),
                "reference": (task.get("correct_answer") or [""])[0],
                "notes": f"source=zno-eval subject={subject} link={link} comment={(task.get('comment') or '')[:120]}",
            }
        )

    rng = random.Random(args.seed)
    selected: list[dict] = []

    if args.per_subject > 0:
        for subject in subjects:
            items = pool[subject][:]
            rng.shuffle(items)
            selected.extend(items[: args.per_subject])
    else:
        # Balanced: round-robin until n
        bags = {s: pool[s][:] for s in subjects}
        for s in bags:
            rng.shuffle(bags[s])
        while len(selected) < args.n and any(bags.values()):
            for s in subjects:
                if len(selected) >= args.n:
                    break
                if bags[s]:
                    selected.append(bags[s].pop())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {s: sum(1 for r in selected if f"subject={s}" in r["notes"]) for s in subjects}
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "n": len(selected),
                "eligible_pool": {s: len(pool[s]) for s in subjects},
                "sampled": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
