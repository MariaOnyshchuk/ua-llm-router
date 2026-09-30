"""Few-shot prompt variants for the non-alignment buckets of mixed_ua_v4.

Alignment already has its own few-shot rewrite in alignment_prompt_variants.py.
Here each selected bucket gets two worked examples put in front of the original
prompt. The original prompt is kept verbatim at the end, so scoring is unchanged.

The examples are hand-written and must not be items from any benchmark file.
`check_no_overlap` verifies that; run `python scripts/bucket_prompt_variants.py`
to check every file under benchmarks/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BUCKETS = ("chat", "code", "translate", "instruct", "knowledge")

# bucket -> list of (task, answer) pairs shown as worked examples
EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "translate": [
        (
            "Переклади з англійської українською. Надай лише переклад, без пояснень:\n"
            "The library closes at nine in the evening on weekdays.",
            "Бібліотека зачиняється о дев'ятій вечора у будні.",
        ),
        (
            "Переклади з української англійською. Надай лише переклад, без пояснень:\n"
            "Ми зустрілися біля вокзалу й пішли пішки до центру міста.",
            "We met near the station and walked to the city centre.",
        ),
    ],
    "instruct": [
        (
            "Поверни ТІЛЬКИ валідний JSON з полями city (рядок) і temp (число). "
            "city=Lviv, temp=12.",
            '{"city":"Lviv","temp":12}',
        ),
        (
            "Дай відповідь рівно двома пунктами списку, без вступу та висновку: "
            "чому корисно робити перерви під час роботи.",
            "- Перерви знижують втому очей і спини.\n- Після відпочинку легше зосередитися на задачі.",
        ),
    ],
    "knowledge": [
        (
            "Хто написав збірку «Кобзар»? Відповідь одним коротким реченням.",
            "Збірку «Кобзар» написав Тарас Шевченко.",
        ),
        (
            "Яка столиця Польщі?\nА) Прага\nБ) Варшава\nВ) Відень\nГ) Будапешт\n"
            "Відповідай лише літерою правильного варіанта.",
            "Б",
        ),
    ],
    "code": [
        (
            "Напиши функцію Python square(x), що повертає квадрат числа. Тільки код, без пояснень.",
            "```python\ndef square(x):\n    return x * x\n```",
        ),
        (
            "Напиши функцію Python last_item(items), що повертає останній елемент списку. Тільки код.",
            "```python\ndef last_item(items):\n    return items[-1]\n```",
        ),
    ],
    "chat": [
        (
            "Привіт! Порадь коротко, як швидко заснути.",
            "Провітри кімнату, вимкни екрани за годину до сну й дихай повільно: вдих на чотири, видих на шість.",
        ),
        (
            "Розкажи коротко, чим цікавий Київ восени.",
            "Восени Київ золотий від каштанів: гарно гуляти Ботанічним садом і пити каву на Подолі.",
        ),
    ],
}

_INTRO = "Ось приклади виконання схожих завдань.\n\n"
_OUTRO = "Тепер виконай справжнє завдання, дотримуючись його вимог до формату.\n\n"


def render_examples(bucket: str) -> str:
    parts = [f"Завдання: {task}\nВідповідь: {answer}" for task, answer in EXAMPLES[bucket]]
    return _INTRO + "\n\n".join(parts) + "\n\n" + _OUTRO


def parse_buckets(spec: str) -> frozenset[str]:
    """'' -> none, 'all' -> every bucket here, otherwise a comma list."""
    spec = (spec or "").strip().lower()
    if not spec or spec == "none":
        return frozenset()
    if spec == "all":
        return frozenset(BUCKETS)
    chosen = {x.strip() for x in spec.split(",") if x.strip()}
    unknown = chosen - set(BUCKETS)
    if unknown:
        raise ValueError(f"unknown buckets for few-shot: {sorted(unknown)}; allowed: {list(BUCKETS)}")
    return frozenset(chosen)


def apply_bucket_fewshot(item: dict, buckets: frozenset[str]) -> dict:
    """Return a copy of the item with examples put before the original prompt."""
    out = dict(item)
    bucket = str(item.get("bucket") or "")
    if bucket not in buckets or bucket not in EXAMPLES:
        return out
    out["prompt"] = render_examples(bucket) + str(item.get("prompt") or "")
    out["notes"] = (str(item.get("notes") or "") + f" | prompt=fewshot_{bucket}").strip(" |")
    return out


def _benchmark_prompts(paths: list[Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                texts.append(str(row.get("prompt") or ""))
                texts.append(str(row.get("reference") or ""))
    return texts


# Instruction lines that legitimately repeat across many items; not a leak by themselves.
_GENERIC_PREFIXES = ("Переклади з", "Відповідай лише", "Тільки код", "Поверни ТІЛЬКИ")


def _needles(task: str, answer: str) -> list[str]:
    needles = [task]
    for line in task.split("\n") + answer.split("\n"):
        line = line.strip()
        if len(line) >= 15 and not line.startswith(_GENERIC_PREFIXES):
            needles.append(line)
    return needles


def check_no_overlap(paths: list[Path]) -> list[tuple[str, str, str]]:
    """Return (bucket, example text, file) for every example found inside a benchmark."""
    problems: list[tuple[str, str, str]] = []
    for path in paths:
        texts = _benchmark_prompts([path])
        for bucket, pairs in EXAMPLES.items():
            for task, answer in pairs:
                for needle in _needles(task, answer):
                    if any(needle in text for text in texts):
                        problems.append((bucket, needle[:80], path.name))
    return problems


def main() -> int:
    files = sorted((ROOT / "benchmarks").rglob("*.jsonl"))
    problems = check_no_overlap(files)
    if problems:
        for bucket, needle, name in problems:
            print(f"OVERLAP bucket={bucket} file={name}: {needle}")
        return 1
    print(f"ok: no few-shot example appears in {len(files)} benchmark files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
