#!/usr/bin/env python3
"""Build and validate the 36-item composite-task benchmark.

The benchmark deliberately uses small, deterministic tasks. Its purpose is to
measure orchestration (plan -> specialist stages -> final artifact), not to add
another broad knowledge benchmark. The top-level prompt is visible to every
system; ``oracle_plan`` and rubrics are evaluation-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "benchmarks" / "mixed_ua_composite_v1.jsonl"


CODE_TASKS = [
    ("sum_even", "Return the sum of all even integers in a list.", "сум", "парн", [[[1, 2, 3, 4]], 6], [[[]], 0]),
    ("count_vowels", "Count English vowels in a string, ignoring case.", "голосн", "регістр", [["Diploma"], 3], [["xyz"], 0]),
    ("reverse_words", "Reverse the order of whitespace-separated words.", "зворотн", "слів", [["one two three"], "three two one"], [["hello"], "hello"]),
    ("is_palindrome", "Return whether a string is a palindrome after lowercasing and removing spaces.", "паліндром", "пробіл", [["Never odd or even"], True], [["router"], False]),
    ("clamp", "Clamp a number x to the inclusive interval [low, high].", "обмеж", "діапазон", [[8, 0, 5], 5], [[-2, 0, 5], 0]),
    ("unique_sorted", "Return the sorted unique integers from a list.", "унікальн", "сорту", [[[3, 1, 3, 2]], [1, 2, 3]], [[[]], []]),
    ("word_lengths", "Map every whitespace-separated word to its length.", "довжин", "слов", [["ua models work"], [2, 6, 4]], [[""], []]),
    ("fizzbuzz_value", "For integer n return 'FizzBuzz' if divisible by 15, 'Fizz' by 3, 'Buzz' by 5, otherwise str(n).", "кратн", "FizzBuzz", [[30], "FizzBuzz"], [[7], "7"]),
    ("safe_divide", "Return a / b, but return None when b is zero.", "ділен", "нуль", [[8, 2], 4.0], [[1, 0], None]),
    ("flatten_once", "Flatten a list of lists by exactly one level.", "спис", "рів", [[[[1, 2], [], [3]]], [1, 2, 3]], [[[["a"], ["b", "c"]]], ["a", "b", "c"]]),
    ("running_total", "Return cumulative sums of the input integers.", "накопич", "сум", [[[1, 2, 3]], [1, 3, 6]], [[[-1, 1]], [-1, 0]]),
    ("initials", "Return uppercase initials for non-empty whitespace-separated words.", "ініціал", "велик", [["Ada Lovelace"], "AL"], [["  Ivan   Franko "], "IF"]),
]


KNOWLEDGE_TASKS = [
    ("Автором «Кобзаря» є:", ["А Леся Українка", "Б Тарас Шевченко", "В Іван Франко"], "Б", ["Тарас", "Шевчен"]),
    ("Столиця України:", ["А Львів", "Б Харків", "В Київ"], "В", ["Київ", "столиц"]),
    ("Конституцію України ухвалено у:", ["А 1991", "Б 1996", "В 2004"], "Б", ["1996", "Конституц"]),
    ("Найбільша за площею область України:", ["А Одеська", "Б Київська", "В Львівська"], "А", ["Одеськ", "площ"]),
    ("Річка, на якій стоїть Київ:", ["А Дністер", "Б Дніпро", "В Південний Буг"], "Б", ["Дніпр", "Київ"]),
    ("Автор роману «Тигролови»:", ["А Іван Багряний", "Б Валер'ян Підмогильний", "В Микола Хвильовий"], "А", ["Багрян", "Тигролов"]),
    ("Незалежність України проголошено:", ["А 24 серпня 1991", "Б 28 червня 1996", "В 1 грудня 1990"], "А", ["24 серпня", "1991"]),
    ("Найвища вершина України:", ["А Говерла", "Б Петрос", "В Піп Іван"], "А", ["Говерл", "2061"]),
    ("Хімічний символ заліза:", ["А Fe", "Б Zn", "В Ag"], "А", ["Fe", "заліз"]),
    ("Планета, відома як Червона планета:", ["А Венера", "Б Марс", "В Юпітер"], "Б", ["Марс", "червон"]),
    ("Кількість областей в Україні:", ["А 22", "Б 24", "В 27"], "Б", ["24", "област"]),
    ("Мову Python створив:", ["А Джеймс Гослінг", "Б Гвідо ван Россум", "В Б'ярне Страуструп"], "Б", ["Гвідо", "Россум"]),
]


PASSAGE_TASKS = [
    ("The Dnipro is 2,201 km long. Kyiv stands on the Dnipro.", {"subject": "Дніпро", "number": 2201, "place": "Київ"}),
    ("Hoverla is 2,061 metres high. It is located in the Ukrainian Carpathians.", {"subject": "Говерла", "number": 2061, "place": "Українські Карпати"}),
    ("The Kyiv Metro opened in 1960. Its first line had five stations.", {"subject": "Київський метрополітен", "number": 1960, "place": "Київ"}),
    ("Lviv was first mentioned in 1256. The city is in western Ukraine.", {"subject": "Львів", "number": 1256, "place": "західна Україна"}),
    ("The Antonov An-225 first flew in 1988. It was designed in Kyiv.", {"subject": "Ан-225", "number": 1988, "place": "Київ"}),
    ("The Constitution of Ukraine was adopted in 1996. It was adopted by the Verkhovna Rada.", {"subject": "Конституція України", "number": 1996, "place": "Верховна Рада"}),
    ("The Odesa Opera House opened in 1887. It stands in Odesa.", {"subject": "Одеський оперний театр", "number": 1887, "place": "Одеса"}),
    ("The first Ukrainian book printed by Ivan Fedorov appeared in Lviv in 1574.", {"subject": "Апостол Івана Федорова", "number": 1574, "place": "Львів"}),
    ("The Paton Bridge opened in 1953. It crosses the Dnipro in Kyiv.", {"subject": "Міст Патона", "number": 1953, "place": "Київ"}),
    ("The Askania-Nova reserve was founded in 1898. It is in Kherson region.", {"subject": "Асканія-Нова", "number": 1898, "place": "Херсонська область"}),
    ("The Lviv tram began electric service in 1894. It operates in Lviv.", {"subject": "Львівський трамвай", "number": 1894, "place": "Львів"}),
    ("The Ukrainian Academy of Sciences was founded in 1918. Its first president was Volodymyr Vernadsky.", {"subject": "Українська академія наук", "number": 1918, "place": "Володимир Вернадський"}),
]


def code_item(i: int, data: tuple[Any, ...]) -> dict[str, Any]:
    fn, spec, term1, term2, case1, case2 = data
    return {
        "id": f"comp-tc-{i:03d}",
        "bucket": "composite",
        "family": "translate_code",
        "prompt": (
            "Виконай складене завдання. Спочатку переклади й нормалізуй англомовну "
            f"специфікацію українською, потім реалізуй Python-функцію `{fn}`. "
            "Фінальна відповідь — лише блок ```python без пояснень.\n\n"
            f"Specification: {spec}"
        ),
        "provenance": "handcrafted_from_humaneval_style",
        "oracle_plan": {
            "steps": [
                {
                    "id": "translate",
                    "intent": "translate",
                    "depends_on": [],
                    "prompt": f"Переклади українською цю специфікацію без додавання вимог:\n{spec}",
                    "rubric": {"type": "contains_all", "values": [term1, term2]},
                },
                {
                    "id": "implement",
                    "intent": "code",
                    "depends_on": ["translate"],
                    "prompt": (
                        f"Реалізуй Python-функцію `{fn}` за специфікацією нижче. "
                        "Поверни лише блок ```python.\n\n{{translate.content}}"
                    ),
                    "rubric": {
                        "type": "python_function",
                        "entry_point": fn,
                        "only_code": True,
                        "cases": [
                            {"args": case1[0], "expected": case1[1]},
                            {"args": case2[0], "expected": case2[1]},
                        ],
                    },
                },
            ]
        },
        "final_step": "implement",
    }


def knowledge_item(i: int, data: tuple[Any, ...]) -> dict[str, Any]:
    question, options, answer, evidence = data
    options_text = "\n".join(options)
    return {
        "id": f"comp-ke-{i:03d}",
        "bucket": "composite",
        "family": "knowledge_explain",
        "prompt": (
            "Розв'яжи тестове питання, а потім коротко обґрунтуй відповідь. "
            "Фінальний формат — рівно два рядки: `Відповідь: <літера>` та "
            "`Пояснення: <одне речення>`.\n\n"
            f"{question}\n{options_text}"
        ),
        "provenance": "handcrafted_zno_style",
        "oracle_plan": {
            "steps": [
                {
                    "id": "answer",
                    "intent": "knowledge",
                    "depends_on": [],
                    "prompt": f"{question}\n{options_text}\nВідповідай лише літерою А, Б або В.",
                    "rubric": {"type": "label", "value": answer},
                },
                {
                    "id": "explain",
                    "intent": "instruct",
                    "depends_on": ["answer"],
                    "prompt": (
                        "Дай фінальну відповідь рівно у двох рядках:\n"
                        "Відповідь: <літера>\nПояснення: <одне речення>\n"
                        f"Питання: {question}\n{options_text}\n"
                        "Попередня відповідь: {{answer.content}}"
                    ),
                    "rubric": {
                        "type": "formatted_evidence",
                        "label": answer,
                        "required": evidence,
                        "line_count": 2,
                    },
                },
            ]
        },
        "final_step": "explain",
    }


def passage_item(i: int, data: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    passage, fields = data
    return {
        "id": f"comp-tkw-{i:03d}",
        "bucket": "composite",
        "family": "translate_knowledge_write",
        "prompt": (
            "Опрацюй англомовне джерело: переклади його українською, витягни три факти "
            "і поверни фінально лише валідний JSON з ключами `subject`, `number`, `place`. "
            "Не використовуй зовнішні відомості.\n\n"
            f"Source: {passage}"
        ),
        "provenance": "handcrafted_flores_style",
        "oracle_plan": {
            "steps": [
                {
                    "id": "translate",
                    "intent": "translate",
                    "depends_on": [],
                    "prompt": f"Точно переклади джерело українською:\n{passage}",
                    "rubric": {
                        "type": "contains_all",
                        "values": [str(fields["number"]), str(fields["subject"]).split()[0]],
                    },
                },
                {
                    "id": "extract",
                    "intent": "knowledge",
                    "depends_on": ["translate"],
                    "prompt": (
                        "Витягни з тексту subject, number і place. Поверни лише JSON.\n"
                        "{{translate.content}}"
                    ),
                    "rubric": {"type": "json_fields", "fields": fields},
                },
                {
                    "id": "write",
                    "intent": "instruct",
                    "depends_on": ["extract"],
                    "prompt": (
                        "Нормалізуй дані у валідний JSON. Рівно три ключі: subject, number, "
                        "place. Без markdown і пояснень.\n{{extract.content}}"
                    ),
                    "rubric": {"type": "json_fields", "fields": fields, "exact_keys": True},
                },
            ]
        },
        "final_step": "write",
    }


def build_items() -> list[dict[str, Any]]:
    return (
        [code_item(i, task) for i, task in enumerate(CODE_TASKS, 1)]
        + [knowledge_item(i, task) for i, task in enumerate(KNOWLEDGE_TASKS, 1)]
        + [passage_item(i, task) for i, task in enumerate(PASSAGE_TASKS, 1)]
    )


def validate_items(items: list[dict[str, Any]]) -> None:
    ids = [str(item.get("id")) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate composite ids")
    families: dict[str, int] = {}
    for item in items:
        if item.get("bucket") != "composite":
            raise ValueError(f"{item.get('id')}: bucket must be composite")
        family = str(item.get("family"))
        families[family] = families.get(family, 0) + 1
        steps = (item.get("oracle_plan") or {}).get("steps") or []
        if not 2 <= len(steps) <= 4:
            raise ValueError(f"{item['id']}: expected 2-4 steps")
        seen: set[str] = set()
        for step in steps:
            sid = str(step.get("id"))
            if not sid or sid in seen:
                raise ValueError(f"{item['id']}: invalid/duplicate step id {sid!r}")
            deps = step.get("depends_on") or []
            if any(dep not in seen for dep in deps):
                raise ValueError(f"{item['id']}:{sid}: dependency must precede step")
            if step.get("intent") not in {
                "translate",
                "knowledge",
                "instruct",
                "code",
                "alignment",
                "chat",
            }:
                raise ValueError(f"{item['id']}:{sid}: invalid intent")
            if not step.get("prompt") or not step.get("rubric"):
                raise ValueError(f"{item['id']}:{sid}: prompt/rubric required")
            seen.add(sid)
        if item.get("final_step") not in seen:
            raise ValueError(f"{item['id']}: final_step missing")
    expected = {
        "translate_code": 12,
        "knowledge_explain": 12,
        "translate_knowledge_write": 12,
    }
    if families != expected:
        raise ValueError(f"family balance mismatch: {families}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="Validate existing output only")
    args = parser.parse_args()

    if args.check:
        items = [
            json.loads(line)
            for line in args.out.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        items = build_items()
    validate_items(items)

    if not args.check:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# composite v1 — 12×3 families; generated, do not hand-edit"]
        lines.extend(json.dumps(item, ensure_ascii=False) for item in items)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(items)} items → {args.out}")
    print("validation OK: 36 items, 12 per family")


if __name__ == "__main__":
    main()
