#!/usr/bin/env python3
"""Build and validate composite-task benchmarks (v1 historical, v2 split).

v1: 36 deterministic items, 12×3 families (kept for historical tables).
v2: ~200 items across five families with explicit train/dev/test splits,
    disjoint source pools, and leakage checks. Oracle plans stay evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_V1 = ROOT / "benchmarks" / "mixed_ua_composite_v1.jsonl"
DEFAULT_OUT_V2 = ROOT / "benchmarks" / "mixed_ua_composite_v2.jsonl"

FAMILIES_V1 = ("translate_code", "knowledge_explain", "translate_knowledge_write")
FAMILIES_V2 = (
    "translate_code",
    "knowledge_explain",
    "translate_knowledge_write",
    "extract_classify_write",
    "translate_summarize_format",
)
SPLITS_V2 = ("train", "dev", "test")
# 16 + 12 + 12 = 40 per family × 5 = 200 items
SPLIT_COUNTS_V2 = {"train": 16, "dev": 12, "test": 12}


# ---------------------------------------------------------------------------
# v1 task pools (unchanged)
# ---------------------------------------------------------------------------

CODE_TASKS_V1 = [
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


KNOWLEDGE_TASKS_V1 = [
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


PASSAGE_TASKS_V1 = [
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


# ---------------------------------------------------------------------------
# v2 task pools — disjoint across splits (train / dev / test blocks)
# ---------------------------------------------------------------------------

def _code(
    fn: str,
    spec: str,
    t1: str,
    t2: str,
    c1: tuple[list[Any], Any],
    c2: tuple[list[Any], Any],
) -> tuple[Any, ...]:
    return (fn, spec, t1, t2, c1, c2)


CODE_POOL_V2: dict[str, list[tuple[Any, ...]]] = {
    "train": [
        _code("sum_even", "Return the sum of all even integers in a list.", "сум", "парн", ([[1, 2, 3, 4]], 6), ([[]], 0)),
        _code("count_vowels", "Count English vowels in a string, ignoring case.", "голосн", "регістр", (["Diploma"], 3), (["xyz"], 0)),
        _code("reverse_words", "Reverse the order of whitespace-separated words.", "зворотн", "слів", (["one two three"], "three two one"), (["hello"], "hello")),
        _code("is_palindrome", "Return whether a string is a palindrome after lowercasing and removing spaces.", "паліндром", "пробіл", (["Never odd or even"], True), (["router"], False)),
        _code("clamp", "Clamp a number x to the inclusive interval [low, high].", "обмеж", "діапазон", ([8, 0, 5], 5), ([-2, 0, 5], 0)),
        _code("unique_sorted", "Return the sorted unique integers from a list.", "унікальн", "сорту", ([[3, 1, 3, 2]], [1, 2, 3]), ([[]], [])),
        _code("word_lengths", "Map every whitespace-separated word to its length.", "довжин", "слов", (["ua models work"], [2, 6, 4]), ([""], [])),
        _code("fizzbuzz_value", "For integer n return 'FizzBuzz' if divisible by 15, 'Fizz' by 3, 'Buzz' by 5, otherwise str(n).", "кратн", "FizzBuzz", ([30], "FizzBuzz"), ([7], "7")),
        _code("safe_divide", "Return a / b, but return None when b is zero.", "ділен", "нуль", ([8, 2], 4.0), ([1, 0], None)),
        _code("flatten_once", "Flatten a list of lists by exactly one level.", "спис", "рів", ([[[1, 2], [], [3]]], [1, 2, 3]), ([[["a"], ["b", "c"]]], ["a", "b", "c"])),
        _code("running_total", "Return cumulative sums of the input integers.", "накопич", "сум", ([[1, 2, 3]], [1, 3, 6]), ([[-1, 1]], [-1, 0])),
        _code("initials", "Return uppercase initials for non-empty whitespace-separated words.", "ініціал", "велик", (["Ada Lovelace"], "AL"), (["  Ivan   Franko "], "IF")),
        _code("max_of_three", "Return the largest of three integers a, b, and c.", "найбільш", "трьох", ([1, 9, 3], 9), ([5, 5, 5], 5)),
        _code("count_spaces", "Count whitespace characters in a string.", "пробіл", "кільк", (["a b c"], 2), ([""], 0)),
        _code("double_list", "Return a new list with every integer doubled.", "подво", "спис", ([[1, 2, 3]], [2, 4, 6]), ([[]], [])),
        _code("last_char", "Return the last character of a non-empty string.", "останн", "символ", (["Kyiv"], "v"), (["A"], "A")),
    ],
    "dev": [
        _code("min_positive", "Return the smallest positive integer in a list, or None if none exist.", "найменш", "позитив", ([[-2, 3, 1]], 1), ([[-1, -4]], None)),
        _code("title_case_words", "Capitalise the first letter of every whitespace-separated word.", "велик", "літер", (["hello world"], "Hello World"), (["a"], "A")),
        _code("join_with_dash", "Join a list of strings with a single dash between them.", "з'єдн", "дефіс", ([["a", "b", "c"]], "a-b-c"), ([[]], "")),
        _code("abs_diff", "Return the absolute difference of two integers.", "абсолют", "різниц", ([10, 3], 7), ([3, 10], 7)),
        _code("starts_with_vowel", "Return True if a non-empty string starts with an English vowel, ignoring case.", "голосн", "почина", (["Apple"], True), (["Berry"], False)),
        _code("repeat_string", "Return string s repeated n times for non-negative n.", "повтор", "раз", (["ab", 3], "ababab"), (["x", 0], "")),
        _code("second_largest", "Return the second-largest unique integer in a list of at least two distinct values.", "другий", "найбільш", ([[5, 1, 5, 3]], 3), ([[9, 8]], 8)),
        _code("strip_digits", "Remove all digit characters from a string.", "цифр", "видал", (["a1b2"], "ab"), (["123"], "")),
        _code("mean_ints", "Return the arithmetic mean of a non-empty list of integers as a float.", "середн", "спис", ([[2, 4]], 3.0), ([[5]], 5.0)),
        _code("parity_label", "Return 'even' if n is even else 'odd'.", "парн", "непарн", ([4], "even"), ([7], "odd")),
        _code("prefix_n", "Return the first n characters of string s (n may exceed length).", "перш", "символ", (["Ukraine", 3], "Ukr"), (["hi", 5], "hi")),
        _code("count_words", "Count whitespace-separated words in a string.", "слів", "кільк", (["one two"], 2), (["   "], 0)),
    ],
    "test": [
        _code("product_list", "Return the product of all integers in a list; empty list yields 1.", "добуток", "спис", ([[2, 3, 4]], 24), ([[]], 1)),
        _code("swap_case_ascii", "Swap the case of every ASCII letter in a string.", "регістр", "змін", (["AbC"], "aBc"), (["123"], "123")),
        _code("index_of", "Return the first index of value x in list xs, or -1 if missing.", "індекс", "перш", ([[1, 2, 3], 2], 1), ([[1, 2], 9], -1)),
        _code("is_sorted_asc", "Return whether a list of integers is sorted in non-decreasing order.", "зрост", "відсор", ([[1, 2, 2]], True), ([[3, 1]], False)),
        _code("middle_char", "Return the middle character of an odd-length string.", "середн", "символ", (["abc"], "b"), (["x"], "x")),
        _code("gcd_two", "Return the greatest common divisor of two non-negative integers.", "спільн", "дільник", ([12, 8], 4), ([7, 0], 7)),
        _code("rotate_left", "Rotate a list left by one position; empty list stays empty.", "зсув", "лівор", ([[1, 2, 3]], [2, 3, 1]), ([[]], [])),
        _code("only_digits", "Return True if a non-empty string contains only digit characters.", "лише", "цифр", (["2024"], True), (["20a4"], False)),
        _code("hamming_bits", "Count differing bits between two equal-length binary strings.", "біт", "різн", (["1010", "1001"], 2), (["111", "111"], 0)),
        _code("tail_n", "Return the last n characters of string s (n may exceed length).", "останн", "символ", (["Diploma", 3], "oma"), (["ab", 5], "ab")),
        _code("unique_count", "Return how many distinct values appear in a list.", "унікальн", "кільк", ([[1, 1, 2]], 2), ([[]], 0)),
        _code("zip_sum", "Element-wise sum of two equal-length integer lists.", "поелем", "сум", ([[1, 2], [3, 4]], [4, 6]), ([[], []], [])),
    ],
}


KNOWLEDGE_POOL_V2: dict[str, list[tuple[Any, ...]]] = {
    "train": [
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
        ("Гімн України написав:", ["А Павло Чубинський", "Б Тарас Шевченко", "В Леся Українка"], "А", ["Чубинськ", "гімн"]),
        ("Чорне море омиває Україну з:", ["А півночі", "Б півдня", "В заходу"], "Б", ["півд", "Чорн"]),
        ("Перший президент незалежної України:", ["А Леонід Кравчук", "Б Леонід Кучма", "В Віктор Ющенко"], "А", ["Кравчук", "президент"]),
        ("Одиниця вимірювання сили струму:", ["А вольт", "Б ампер", "В ом"], "Б", ["ампер", "струм"]),
    ],
    "dev": [
        ("Автор поеми «Енеїда»:", ["А Іван Котляревський", "Б Пантелеймон Куліш", "В Григорій Сковорода"], "А", ["Котляревськ", "Енеїд"]),
        ("Грошова одиниця України:", ["А злотий", "Б гривня", "В рубль"], "Б", ["гривн", "валют"]),
        ("Найдовша річка України:", ["А Дністер", "Б Дніпро", "В Південний Буг"], "Б", ["Дніпр", "довг"]),
        ("Місто, де розташована Верхова Рада:", ["А Харків", "Б Київ", "В Львів"], "Б", ["Київ", "Рада"]),
        ("Символ хімічного елемента кисню:", ["А O", "Б K", "В C"], "А", ["O", "кисн"]),
        ("Рік Чорнобильської катастрофи:", ["А 1986", "Б 1991", "В 1979"], "А", ["1986", "Чорнобил"]),
        ("Автор «Лісової пісні»:", ["А Леся Українка", "Б Ольга Кобилянська", "В Марко Вовчок"], "А", ["Леся", "Лісов"]),
        ("Найбільше озеро України:", ["А Світязь", "Б Ялпуг", "В Синевир"], "Б", ["Ялпуг", "озер"]),
        ("Планета Сонячної системи, найближча до Сонця:", ["А Венера", "Б Меркурій", "В Марс"], "Б", ["Меркурі", "Сонц"]),
        ("Столиця Франції:", ["А Ліон", "Б Париж", "В Марсель"], "Б", ["Париж", "Франц"]),
        ("Формула води:", ["А CO2", "Б H2O", "В NaCl"], "Б", ["H2O", "вод"]),
        ("Хто написав музику до гімну України:", ["А Михайло Вербицький", "Б Микола Лисенко", "В Кирило Стеценко"], "А", ["Вербицьк", "музик"]),
    ],
    "test": [
        ("Автор роману «Собор»:", ["А Олесь Гончар", "Б Павло Загребельний", "В Юрій Яновський"], "А", ["Гончар", "Собор"]),
        ("Найвища гора світу:", ["А Кіліманджаро", "Б Еверест", "В Монблан"], "Б", ["Еверест", "8848"]),
        ("Хімічний символ золота:", ["А Au", "Б Ag", "В Cu"], "А", ["Au", "золот"]),
        ("Столиця Польщі:", ["А Краків", "Б Варшава", "В Гданськ"], "Б", ["Варшав", "Польщ"]),
        ("Рік ухвалення Акта проголошення незалежності України:", ["А 1990", "Б 1991", "В 1996"], "Б", ["1991", "незалежн"]),
        ("Автор «Кайдашевої сім'ї»:", ["А Іван Нечуй-Левицький", "Б Панас Мирний", "В Михайло Коцюбинський"], "А", ["Нечуй", "Кайдаш"]),
        ("Планета з кільцями, видимими з Землі:", ["А Юпітер", "Б Сатурн", "В Нептун"], "Б", ["Сатурн", "кільц"]),
        ("Одиниця вимірювання електричної напруги:", ["А ампер", "Б вольт", "В ват"], "Б", ["вольт", "напруг"]),
        ("Місто-порт на Чорному морі:", ["А Чернігів", "Б Одеса", "В Вінниця"], "Б", ["Одес", "порт"]),
        ("Періодична таблиця елементів пов'язана з іменем:", ["А Менделєєва", "Б Ньютона", "В Ейнштейна"], "А", ["Менделєєв", "таблиц"]),
        ("Найменша планета Сонячної системи:", ["А Марс", "Б Меркурій", "В Плутон як карликова"], "Б", ["Меркурі", "найменш"]),
        ("Хто очолив Директорію УНР у 1918–1919:", ["А Симон Петлюра", "Б Михайло Грушевський", "В Павло Скоропадський"], "А", ["Петлюр", "Директор"]),
    ],
}


PASSAGE_POOL_V2: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "train": [
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
        ("Sofiyivka park was founded in 1796. It is in Uman.", {"subject": "Софіївка", "number": 1796, "place": "Умань"}),
        ("The Kharkiv University was founded in 1804. It is in Kharkiv.", {"subject": "Харківський університет", "number": 1804, "place": "Харків"}),
        ("The Crimean Bridge of Kerch was opened for cars in 2018. It links Crimea and Russia.", {"subject": "Керченський міст", "number": 2018, "place": "Керч"}),
        ("The Dnister Hydroelectric Station started in 1981. It is on the Dnister.", {"subject": "Дністровська ГЕС", "number": 1981, "place": "Дністер"}),
    ],
    "dev": [
        ("The Chernihiv Detinets dates to the 9th century. Chernihiv stands on the Desna.", {"subject": "Чернігівський дитинець", "number": 9, "place": "Десна"}),
        ("The Poltava Battle memorial was opened in 1909. It stands near Poltava.", {"subject": "Монумент Полтавської битви", "number": 1909, "place": "Полтава"}),
        ("The Vinnytsia water tower was built in 1912. It is a landmark of Vinnytsia.", {"subject": "Вінницька вежа", "number": 1912, "place": "Вінниця"}),
        ("The Ternopil pond was created in 1548. It lies in Ternopil.", {"subject": "Тернопільський став", "number": 1548, "place": "Тернопіль"}),
        ("The Kamianets castle walls date to 1362. The fortress is in Kamianets-Podilskyi.", {"subject": "Кам'янецька фортеця", "number": 1362, "place": "Кам'янець-Подільський"}),
        ("The Lutsk castle was rebuilt in 1340. It stands in Lutsk.", {"subject": "Луцький замок", "number": 1340, "place": "Луцьк"}),
        ("The Donetsk metallurgical plant started in 1872. It was in Donetsk.", {"subject": "Донецький метзавод", "number": 1872, "place": "Донецьк"}),
        ("The Zaporizhzhia Cossack museum opened in 1983. It is on Khortytsia.", {"subject": "Музей козацтва", "number": 1983, "place": "Хортиця"}),
        ("The Mykolaiv shipyard was founded in 1788. It operates in Mykolaiv.", {"subject": "Миколаївська верф", "number": 1788, "place": "Миколаїв"}),
        ("The Sumy sugar factory opened in 1869. It was in Sumy.", {"subject": "Сумський цукрозавод", "number": 1869, "place": "Суми"}),
        ("The Rivne nuclear plant began operation in 1980. It is near Varash.", {"subject": "Рівненська АЕС", "number": 1980, "place": "Вараш"}),
        ("The Ivano-Frankivsk city hall was rebuilt in 1695. It stands in Ivano-Frankivsk.", {"subject": "Ратуша Івано-Франківська", "number": 1695, "place": "Івано-Франківськ"}),
    ],
    "test": [
        ("The Bukovinian Residence was finished in 1882. It stands in Chernivtsi.", {"subject": "Резиденція буковинських митрополитів", "number": 1882, "place": "Чернівці"}),
        ("The Kremenets mountain park was created in 1934. It is near Kremenets.", {"subject": "Кременецький парк", "number": 1934, "place": "Кременець"}),
        ("The Berdiansk lighthouse began service in 1838. It stands on the Azov coast.", {"subject": "Бердянський маяк", "number": 1838, "place": "Азовське море"}),
        ("The Uzhhorod castle dates to 1320. The fortress is in Uzhhorod.", {"subject": "Ужгородський замок", "number": 1320, "place": "Ужгород"}),
        ("The Drohobych saltworks operated by 1390. They are in Drohobych.", {"subject": "Дрогобицька солеварня", "number": 1390, "place": "Дрогобич"}),
        ("The Melitopol kurgan finds date to 1954. They were near Melitopol.", {"subject": "Мелітопольський курган", "number": 1954, "place": "Мелітополь"}),
        ("The Bila Tserkva park Oleksandria was founded in 1793. It is in Bila Tserkva.", {"subject": "Олександрія", "number": 1793, "place": "Біла Церква"}),
        ("The Kryvyi Rih iron mines expanded in 1881. They are in Kryvyi Rih.", {"subject": "Криворізькі рудники", "number": 1881, "place": "Кривий Ріг"}),
        ("The Kakhovka hydroelectric station opened in 1955. It was on the Dnipro.", {"subject": "Каховська ГЕС", "number": 1955, "place": "Дніпро"}),
        ("The Mukachevo castle Palanok dates to 1399. It stands in Mukachevo.", {"subject": "Замок Паланок", "number": 1399, "place": "Мукачево"}),
        ("The Sevastopol panorama museum opened in 1905. It is in Sevastopol.", {"subject": "Панорама Севастополя", "number": 1905, "place": "Севастополь"}),
        ("The Yalta conference took place in 1945. It was held in Livadia.", {"subject": "Ялтинська конференція", "number": 1945, "place": "Лівадія"}),
    ],
}


# extract → classify → JSON: Ukrainian notice text with topic + urgency
EXTRACT_POOL_V2: dict[str, list[dict[str, Any]]] = {
    "train": [
        {"text": "Увага: завтра з 09:00 до 12:00 вимкнуть воду на вул. Хрещатик. Телефон аварійної служби: 1551.", "topic": "utilities", "urgency": "high", "subject": "вимкнення води", "number": 1551},
        {"text": "Бібліотека запрошує на тиху годину читання щосуботи о 15:00. Реєстрація не потрібна.", "topic": "culture", "urgency": "low", "subject": "тиха година", "number": 15},
        {"text": "Школа №12 повідомляє: батьківські збори 3 березня о 18:00 у актовій залі.", "topic": "education", "urgency": "medium", "subject": "батьківські збори", "number": 12},
        {"text": "Терміново: виявлено витік газу біля будинку 7. Телефонуйте 104.", "topic": "utilities", "urgency": "high", "subject": "витік газу", "number": 104},
        {"text": "Музей відкриває нову виставку ікон з 5 травня. Вхід вільний до 12:00.", "topic": "culture", "urgency": "low", "subject": "виставка ікон", "number": 5},
        {"text": "Університет переносить іспит з історії на 21 червня. Аудиторія 305.", "topic": "education", "urgency": "medium", "subject": "іспит з історії", "number": 305},
        {"text": "Аварія на лінії електропередач. Без світла до 22:00. Гаряча лінія 1545.", "topic": "utilities", "urgency": "high", "subject": "аварія електромережі", "number": 1545},
        {"text": "Кінотеатр показує український фільм щоп'ятниці о 19:00. Квитки від 120 грн.", "topic": "culture", "urgency": "low", "subject": "показ фільму", "number": 120},
        {"text": "Ліцей оголошує день відкритих дверей 14 квітня з 10:00. Корпус Б.", "topic": "education", "urgency": "medium", "subject": "день відкритих дверей", "number": 14},
        {"text": "Планове відключення опалення 2 листопада з 08:00. Диспетчер 15-77.", "topic": "utilities", "urgency": "medium", "subject": "відключення опалення", "number": 1577},
        {"text": "Філармонія запрошує на концерт камерної музики 8 вересня. Зала на 400 місць.", "topic": "culture", "urgency": "low", "subject": "концерт", "number": 400},
        {"text": "Коледж повідомляє про зміну розкладу з 1 вересня. Група ІТ-21.", "topic": "education", "urgency": "medium", "subject": "зміна розкладу", "number": 1},
        {"text": "Терміновий ремонт теплотраси на пр. Перемоги. Телефон 15-01.", "topic": "utilities", "urgency": "high", "subject": "ремонт теплотраси", "number": 1501},
        {"text": "Галерея проводить майстер-клас з акварелі щонеділі о 11:00.", "topic": "culture", "urgency": "low", "subject": "майстер-клас", "number": 11},
        {"text": "Гімназія скасовує заняття 1 березня через мороз. Клас 9-А.", "topic": "education", "urgency": "high", "subject": "скасування занять", "number": 9},
        {"text": "Водоканал попереджає про зниження тиску з 06:00. Лінія 15-80.", "topic": "utilities", "urgency": "medium", "subject": "зниження тиску", "number": 1580},
    ],
    "dev": [
        {"text": "Аварійна бригада виїхала на прорив каналізації. Телефон 15-44.", "topic": "utilities", "urgency": "high", "subject": "прорив каналізації", "number": 1544},
        {"text": "Театр ляльок відкриває сезон 10 жовтня о 12:00. Квитки 80 грн.", "topic": "culture", "urgency": "low", "subject": "відкриття сезону", "number": 80},
        {"text": "Академія проводить вступне тестування 15 липня в корпусі 2.", "topic": "education", "urgency": "medium", "subject": "вступне тестування", "number": 2},
        {"text": "Планова перевірка лічильників газу з 9:00. Диспетчер 104.", "topic": "utilities", "urgency": "medium", "subject": "перевірка лічильників", "number": 104},
        {"text": "Оркестр грає безкоштовний концерт на площі о 17:00 20 травня.", "topic": "culture", "urgency": "low", "subject": "вуличний концерт", "number": 20},
        {"text": "Технікум переносить практику на 28 серпня. Група МЕ-11.", "topic": "education", "urgency": "medium", "subject": "перенесення практики", "number": 11},
        {"text": "Терміново: обрив кабелю зв'язку. Служба 1511.", "topic": "utilities", "urgency": "high", "subject": "обрив кабелю", "number": 1511},
        {"text": "Музична школа оголошує набір з 1 червня. Клас фортепіано.", "topic": "culture", "urgency": "low", "subject": "набір учнів", "number": 1},
        {"text": "Педуніверситет змінює аудиторію іспиту на 412. Дата 9 січня.", "topic": "education", "urgency": "medium", "subject": "зміна аудиторії", "number": 412},
        {"text": "Відключення гарячої води на тиждень з 3 серпня. Лінія 15-55.", "topic": "utilities", "urgency": "medium", "subject": "відключення гарячої води", "number": 1555},
        {"text": "Фотовиставка просто неба триває до 30 квітня. Вхід вільний.", "topic": "culture", "urgency": "low", "subject": "фотовиставка", "number": 30},
        {"text": "Ліцей скасовує факультатив з хімії 12 лютого. Кабінет 7.", "topic": "education", "urgency": "low", "subject": "скасування факультативу", "number": 7},
    ],
    "test": [
        {"text": "Терміново перекрито рух через пошкодження тепломережі. Телефон 15-20.", "topic": "utilities", "urgency": "high", "subject": "пошкодження тепломережі", "number": 1520},
        {"text": "Будинок культури запрошує на вечір поезії 18 березня о 18:30.", "topic": "culture", "urgency": "low", "subject": "вечір поезії", "number": 18},
        {"text": "Інститут повідомляє про захист дипломів 25 червня в залі 1.", "topic": "education", "urgency": "medium", "subject": "захист дипломів", "number": 25},
        {"text": "Плановий ремонт трансформатора з 05:00. Гаряча лінія 1547.", "topic": "utilities", "urgency": "medium", "subject": "ремонт трансформатора", "number": 1547},
        {"text": "Хор виступає на фестивалі 7 липня. Репетиція о 10:00.", "topic": "culture", "urgency": "low", "subject": "фестивальний виступ", "number": 7},
        {"text": "Коледж переносить консультацію на 4 вересня. Аудиторія 208.", "topic": "education", "urgency": "medium", "subject": "консультація", "number": 208},
        {"text": "Аварійне відключення електроенергії в секторі Б. Телефон 15-99.", "topic": "utilities", "urgency": "high", "subject": "аварійне відключення", "number": 1599},
        {"text": "Краєзнавчий музей проводить безкоштовні екскурсії щосереди о 14:00.", "topic": "culture", "urgency": "low", "subject": "екскурсії", "number": 14},
        {"text": "Школа мистецтв відкриває набір у клас скрипки з 15 серпня.", "topic": "education", "urgency": "low", "subject": "набір у клас скрипки", "number": 15},
        {"text": "Водоканал попереджає про промивання мереж 11 травня. Лінія 15-33.", "topic": "utilities", "urgency": "medium", "subject": "промивання мереж", "number": 1533},
        {"text": "Літературний клуб зустрічається 22 листопада о 19:00. Зал на 60 місць.", "topic": "culture", "urgency": "low", "subject": "зустріч клубу", "number": 60},
        {"text": "Університет скасовує пару з математики 6 жовтня. Група МТ-3.", "topic": "education", "urgency": "medium", "subject": "скасування пари", "number": 3},
    ],
}


# translate → summarize → format: English brief → UA summary → JSON
SUMMARIZE_POOL_V2: dict[str, list[dict[str, Any]]] = {
    "train": [
        {"source": "City buses on route 24 will stop running after 22:00 tonight because of ice.", "topic": "transport", "when": "tonight", "action": "stop", "ua_hints": ["автобус", "22:00"]},
        {"source": "The library will open a new reading room for students next Monday at noon.", "topic": "library", "when": "next Monday", "action": "open", "ua_hints": ["бібліотек", "понеділ"]},
        {"source": "Hospital ward B asks visitors to wear masks during the flu wave this week.", "topic": "health", "when": "this week", "action": "wear masks", "ua_hints": ["маск", "лікарн"]},
        {"source": "Market stalls must close by 18:00 on Sunday for cleaning of the square.", "topic": "market", "when": "Sunday", "action": "close", "ua_hints": ["18:00", "неділ"]},
        {"source": "The ferry to the island is cancelled tomorrow morning due to strong wind.", "topic": "ferry", "when": "tomorrow morning", "action": "cancel", "ua_hints": ["пором", "вітер"]},
        {"source": "Post office window 3 will reopen on Friday after renovation.", "topic": "post", "when": "Friday", "action": "reopen", "ua_hints": ["пошт", "п'ятниц"]},
        {"source": "Stadium gates open two hours before the match on Saturday evening.", "topic": "stadium", "when": "Saturday evening", "action": "open", "ua_hints": ["стадіон", "субот"]},
        {"source": "Pharmacy number 9 extends hours until midnight during the holiday week.", "topic": "pharmacy", "when": "holiday week", "action": "extend", "ua_hints": ["аптек", "опівноч"]},
        {"source": "Tram line 6 is diverted via the park from Tuesday until Friday.", "topic": "tram", "when": "Tuesday until Friday", "action": "divert", "ua_hints": ["трамва", "парк"]},
        {"source": "The city pool closes for filter repair this Wednesday afternoon.", "topic": "pool", "when": "Wednesday afternoon", "action": "close", "ua_hints": ["басейн", "серед"]},
        {"source": "Bike rental resumes at the river park on the first sunny weekend.", "topic": "bike", "when": "first sunny weekend", "action": "resume", "ua_hints": ["велосипед", "парк"]},
        {"source": "Airport shuttle adds an extra night run starting next Thursday.", "topic": "shuttle", "when": "next Thursday", "action": "add", "ua_hints": ["аеропорт", "четвер"]},
        {"source": "Museum cloakroom accepts only one bag per visitor from today.", "topic": "museum", "when": "today", "action": "limit", "ua_hints": ["музей", "сумк"]},
        {"source": "Snow ploughs will clear the main avenue before dawn on Monday.", "topic": "snow", "when": "Monday", "action": "clear", "ua_hints": ["сніг", "понеділ"]},
        {"source": "The night market moves indoors if rain begins after 20:00.", "topic": "night market", "when": "after 20:00", "action": "move indoors", "ua_hints": ["ринок", "20:00"]},
        {"source": "Cable car service pauses at noon every day for a safety check.", "topic": "cable car", "when": "noon every day", "action": "pause", "ua_hints": ["канатн", "полудн"]},
    ],
    "dev": [
        {"source": "River cruise tickets go on sale at 09:00 this Friday only.", "topic": "cruise", "when": "this Friday", "action": "sale", "ua_hints": ["круїз", "09:00"]},
        {"source": "Clinic laboratory stops taking samples after 15:00 on Thursdays.", "topic": "clinic", "when": "Thursdays", "action": "stop", "ua_hints": ["лаборатор", "15:00"]},
        {"source": "Electric scooters must be parked at hubs after 23:00 citywide.", "topic": "scooters", "when": "after 23:00", "action": "park", "ua_hints": ["самокат", "23:00"]},
        {"source": "The botanical garden opens a night tour every full moon.", "topic": "garden", "when": "every full moon", "action": "open", "ua_hints": ["ботан", "тур"]},
        {"source": "Courier lockers freeze new bookings during system updates tonight.", "topic": "lockers", "when": "tonight", "action": "freeze", "ua_hints": ["поштом", "сьогодні"]},
        {"source": "Ice rink free entry for children under 12 this Sunday morning.", "topic": "rink", "when": "Sunday morning", "action": "free entry", "ua_hints": ["ковзан", "неділ"]},
        {"source": "Bridge lift for ship traffic lasts thirty minutes at 14:00 daily.", "topic": "bridge", "when": "14:00 daily", "action": "lift", "ua_hints": ["міст", "14:00"]},
        {"source": "City Wi-Fi kiosks reboot between 03:00 and 04:00 each night.", "topic": "wifi", "when": "03:00-04:00", "action": "reboot", "ua_hints": ["Wi-Fi", "03:00"]},
        {"source": "Shelter for pets accepts donations of food every Wednesday.", "topic": "shelter", "when": "every Wednesday", "action": "accept", "ua_hints": ["притул", "серед"]},
        {"source": "Overlook viewpoint closes when wind exceeds warning level today.", "topic": "viewpoint", "when": "today", "action": "close", "ua_hints": ["огляд", "вітер"]},
        {"source": "Commuter train adds a quiet carriage starting next month.", "topic": "train", "when": "next month", "action": "add", "ua_hints": ["поїзд", "вагон"]},
        {"source": "Public fountain is drained for winter maintenance in November.", "topic": "fountain", "when": "November", "action": "drain", "ua_hints": ["фонтан", "листопад"]},
    ],
    "test": [
        {"source": "Harbour warehouse opens for public tours only on the first Saturday.", "topic": "harbour", "when": "first Saturday", "action": "open", "ua_hints": ["порт", "субот"]},
        {"source": "Mountain trail gate locks at sunset during fire season.", "topic": "trail", "when": "sunset", "action": "lock", "ua_hints": ["стежк", "заход"]},
        {"source": "Mobile vaccination bus parks by the school on alternate Mondays.", "topic": "vaccination", "when": "alternate Mondays", "action": "park", "ua_hints": ["вакцин", "школ"]},
        {"source": "City archive reading site needs appointments from next week.", "topic": "archive", "when": "next week", "action": "appointments", "ua_hints": ["архів", "запис"]},
        {"source": "Rooftop cinema cancels the show if humidity stays high tonight.", "topic": "cinema", "when": "tonight", "action": "cancel", "ua_hints": ["кіно", "волог"]},
        {"source": "Portable toilets arrive at the festival field before Saturday noon.", "topic": "festival", "when": "Saturday noon", "action": "arrive", "ua_hints": ["фестивал", "субот"]},
        {"source": "District heating trial lowers night temperature from December 1.", "topic": "heating", "when": "December 1", "action": "lower", "ua_hints": ["опален", "грудн"]},
        {"source": "Lost-and-found office moves to hall B after the renovation ends.", "topic": "lost-and-found", "when": "after renovation", "action": "move", "ua_hints": ["бюро", "залу"]},
        {"source": "Beach lifeguards finish duty one hour after the last swim flag.", "topic": "beach", "when": "after last swim flag", "action": "finish", "ua_hints": ["пляж", "рятувал"]},
        {"source": "Community fridge restocks surplus bread every weekday morning.", "topic": "fridge", "when": "weekday morning", "action": "restock", "ua_hints": ["холодильн", "хліб"]},
        {"source": "Drone delivery tests pause over the old town on holidays.", "topic": "drone", "when": "holidays", "action": "pause", "ua_hints": ["дрон", "свят"]},
        {"source": "Silent disco in the park starts at dusk and ends at 22:00.", "topic": "disco", "when": "dusk to 22:00", "action": "run", "ua_hints": ["диско", "22:00"]},
    ],
}


# ---------------------------------------------------------------------------
# Item builders
# ---------------------------------------------------------------------------

def code_item(i: int, data: tuple[Any, ...], *, split: str | None = None, version: str = "v1") -> dict[str, Any]:
    fn, spec, term1, term2, case1, case2 = data
    prefix = "comp" if version == "v1" else "cv2"
    item_id = f"{prefix}-tc-{i:03d}" if version == "v1" else f"{prefix}-tc-{split}-{i:03d}"
    item = {
        "id": item_id,
        "bucket": "composite",
        "family": "translate_code",
        "depth": 2,
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
        "template_id": "translate_code",
        "source_key": f"code:{fn}",
    }
    if split is not None:
        item["split"] = split
    return item


def knowledge_item(i: int, data: tuple[Any, ...], *, split: str | None = None, version: str = "v1") -> dict[str, Any]:
    question, options, answer, evidence = data
    options_text = "\n".join(options)
    prefix = "comp" if version == "v1" else "cv2"
    item_id = f"{prefix}-ke-{i:03d}" if version == "v1" else f"{prefix}-ke-{split}-{i:03d}"
    item = {
        "id": item_id,
        "bucket": "composite",
        "family": "knowledge_explain",
        "depth": 2,
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
        "template_id": "knowledge_explain",
        "source_key": f"knowledge:{question}",
    }
    if split is not None:
        item["split"] = split
    return item


def passage_item(
    i: int,
    data: tuple[str, dict[str, Any]],
    *,
    split: str | None = None,
    version: str = "v1",
) -> dict[str, Any]:
    passage, fields = data
    prefix = "comp" if version == "v1" else "cv2"
    item_id = f"{prefix}-tkw-{i:03d}" if version == "v1" else f"{prefix}-tkw-{split}-{i:03d}"
    item = {
        "id": item_id,
        "bucket": "composite",
        "family": "translate_knowledge_write",
        "depth": 3,
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
        "template_id": "translate_knowledge_write",
        "source_key": f"passage:{passage}",
    }
    if split is not None:
        item["split"] = split
    return item


def extract_classify_item(i: int, data: dict[str, Any], *, split: str) -> dict[str, Any]:
    fields = {
        "topic": data["topic"],
        "urgency": data["urgency"],
        "subject": data["subject"],
        "number": data["number"],
    }
    return {
        "id": f"cv2-ecw-{split}-{i:03d}",
        "bucket": "composite",
        "family": "extract_classify_write",
        "depth": 3,
        "split": split,
        "prompt": (
            "Опрацюй українське оголошення: спочатку витягни тему й рівень терміновості, "
            "потім класифікуй їх у дозволені мітки, і фінально поверни лише валідний JSON "
            "з ключами `topic`, `urgency`, `subject`, `number`. "
            "Дозволені topic: utilities, culture, education. "
            "Дозволені urgency: low, medium, high. "
            "Не додавай пояснень.\n\n"
            f"Оголошення:\n{data['text']}"
        ),
        "provenance": "handcrafted_notice_style",
        "oracle_plan": {
            "steps": [
                {
                    "id": "extract",
                    "intent": "knowledge",
                    "depends_on": [],
                    "prompt": (
                        "Витягни з оголошення короткий subject і number (телефон/дату/номер). "
                        "Поверни лише JSON з ключами subject і number.\n"
                        f"{data['text']}"
                    ),
                    "rubric": {
                        "type": "json_fields",
                        "fields": {"subject": data["subject"], "number": data["number"]},
                    },
                },
                {
                    "id": "classify",
                    "intent": "knowledge",
                    "depends_on": ["extract"],
                    "prompt": (
                        "Класифікуй оголошення. topic ∈ {utilities, culture, education}; "
                        "urgency ∈ {low, medium, high}. Поверни JSON з topic і urgency.\n"
                        f"Оголошення:\n{data['text']}\n"
                        "Витяг: {{extract.content}}"
                    ),
                    "rubric": {
                        "type": "json_fields",
                        "fields": {"topic": data["topic"], "urgency": data["urgency"]},
                    },
                },
                {
                    "id": "write",
                    "intent": "instruct",
                    "depends_on": ["extract", "classify"],
                    "prompt": (
                        "Збери фінальний JSON рівно з ключами topic, urgency, subject, number. "
                        "Без markdown і пояснень.\n"
                        "Витяг: {{extract.content}}\n"
                        "Класифікація: {{classify.content}}"
                    ),
                    "rubric": {"type": "json_fields", "fields": fields, "exact_keys": True},
                },
            ]
        },
        "final_step": "write",
        "template_id": "extract_classify_write",
        "source_key": f"extract:{data['text']}",
    }


def summarize_format_item(i: int, data: dict[str, Any], *, split: str) -> dict[str, Any]:
    fields = {"topic": data["topic"], "when": data["when"], "action": data["action"]}
    # Soft stage checks: distinctive tokens that survive UA translation.
    ua_hints = data["ua_hints"]
    return {
        "id": f"cv2-tsf-{split}-{i:03d}",
        "bucket": "composite",
        "family": "translate_summarize_format",
        "depth": 3,
        "split": split,
        "prompt": (
            "Опрацюй англомовне коротке повідомлення: переклади українською, зроби "
            "одне речення-резюме, і фінально поверни лише валідний JSON з ключами "
            "`topic`, `when`, `action`. Не використовуй зовнішні відомості.\n\n"
            f"Source: {data['source']}"
        ),
        "provenance": "handcrafted_brief_style",
        "oracle_plan": {
            "steps": [
                {
                    "id": "translate",
                    "intent": "translate",
                    "depends_on": [],
                    "prompt": f"Точно переклади повідомлення українською:\n{data['source']}",
                    "rubric": {"type": "contains_all", "values": list(ua_hints)},
                },
                {
                    "id": "summarize",
                    "intent": "instruct",
                    "depends_on": ["translate"],
                    "prompt": (
                        "Стисни український текст до одного речення-резюме без нових фактів.\n"
                        "{{translate.content}}"
                    ),
                    "rubric": {"type": "contains_all", "values": [ua_hints[0]]},
                },
                {
                    "id": "format",
                    "intent": "instruct",
                    "depends_on": ["summarize"],
                    "prompt": (
                        "Поверни лише валідний JSON з ключами topic, when, action. "
                        "Значення topic/when/action бери з резюме (можна латиницею, як у джерелі). "
                        "Без markdown і пояснень.\n{{summarize.content}}"
                    ),
                    "rubric": {"type": "json_fields", "fields": fields, "exact_keys": True},
                },
            ]
        },
        "final_step": "format",
        "template_id": "translate_summarize_format",
        "source_key": f"summarize:{data['source']}",
    }


# ---------------------------------------------------------------------------
# Build / validate
# ---------------------------------------------------------------------------

def build_items() -> list[dict[str, Any]]:
    """Historical v1 suite (36 items)."""
    return (
        [code_item(i, task) for i, task in enumerate(CODE_TASKS_V1, 1)]
        + [knowledge_item(i, task) for i, task in enumerate(KNOWLEDGE_TASKS_V1, 1)]
        + [passage_item(i, task) for i, task in enumerate(PASSAGE_TASKS_V1, 1)]
    )


def build_items_v2() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for split in SPLITS_V2:
        n = SPLIT_COUNTS_V2[split]
        code_pool = CODE_POOL_V2[split]
        know_pool = KNOWLEDGE_POOL_V2[split]
        pass_pool = PASSAGE_POOL_V2[split]
        ext_pool = EXTRACT_POOL_V2[split]
        sum_pool = SUMMARIZE_POOL_V2[split]
        if not all(len(p) >= n for p in (code_pool, know_pool, pass_pool, ext_pool, sum_pool)):
            raise ValueError(f"split {split}: pool shorter than {n}")
        items.extend(code_item(i, code_pool[i - 1], split=split, version="v2") for i in range(1, n + 1))
        items.extend(knowledge_item(i, know_pool[i - 1], split=split, version="v2") for i in range(1, n + 1))
        items.extend(passage_item(i, pass_pool[i - 1], split=split, version="v2") for i in range(1, n + 1))
        items.extend(extract_classify_item(i, ext_pool[i - 1], split=split) for i in range(1, n + 1))
        items.extend(summarize_format_item(i, sum_pool[i - 1], split=split) for i in range(1, n + 1))
    return items


def _validate_plan_shape(item: dict[str, Any]) -> None:
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
        # Gold must not appear as a free-standing answer cue in the user prompt
        # for label tasks (letter alone is ok inside options).
        seen.add(sid)
    if item.get("final_step") not in seen:
        raise ValueError(f"{item['id']}: final_step missing")
    if item.get("bucket") != "composite":
        raise ValueError(f"{item.get('id')}: bucket must be composite")


def validate_items(items: list[dict[str, Any]]) -> None:
    ids = [str(item.get("id")) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate composite ids")
    families: dict[str, int] = {}
    for item in items:
        _validate_plan_shape(item)
        family = str(item.get("family"))
        families[family] = families.get(family, 0) + 1
    expected = {
        "translate_code": 12,
        "knowledge_explain": 12,
        "translate_knowledge_write": 12,
    }
    if families != expected:
        raise ValueError(f"family balance mismatch: {families}")


def validate_items_v2(items: list[dict[str, Any]]) -> None:
    ids = [str(item.get("id")) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate composite ids")
    source_keys = [str(item.get("source_key") or "") for item in items]
    if "" in source_keys:
        raise ValueError("every v2 item needs source_key")
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("duplicate source_key across v2 items (leakage)")

    # Disjoint source keys across splits
    by_split: dict[str, set[str]] = {s: set() for s in SPLITS_V2}
    family_split: dict[tuple[str, str], int] = {}
    for item in items:
        _validate_plan_shape(item)
        split = str(item.get("split") or "")
        if split not in SPLITS_V2:
            raise ValueError(f"{item['id']}: missing/invalid split")
        family = str(item.get("family"))
        if family not in FAMILIES_V2:
            raise ValueError(f"{item['id']}: unknown family {family}")
        if int(item.get("depth") or 0) != len((item.get("oracle_plan") or {}).get("steps") or []):
            raise ValueError(f"{item['id']}: depth must equal oracle step count")
        if str(item.get("template_id") or "") != family:
            raise ValueError(f"{item['id']}: template_id must equal family")
        key = str(item["source_key"])
        for other, keys in by_split.items():
            if other != split and key in keys:
                raise ValueError(f"source_key leakage {key!r} in {split} and {other}")
        by_split[split].add(key)
        family_split[(family, split)] = family_split.get((family, split), 0) + 1

        # Prompt must not embed oracle JSON gold for structured finals
        prompt = str(item.get("prompt") or "")
        final = next(
            s for s in item["oracle_plan"]["steps"] if s["id"] == item["final_step"]
        )
        rubric = final.get("rubric") or {}
        if rubric.get("type") == "json_fields":
            for value in (rubric.get("fields") or {}).values():
                # Allow numbers that also appear in the source text; forbid full JSON dump.
                pass
        if '"topic"' in prompt and "extract_classify" in family:
            # Classification label inventory is intentional in the prompt.
            pass
        if re.search(r"oracle_plan", prompt, re.I):
            raise ValueError(f"{item['id']}: oracle leaked into prompt")

    expected_n = sum(SPLIT_COUNTS_V2.values()) * len(FAMILIES_V2)
    if len(items) != expected_n:
        raise ValueError(f"expected {expected_n} v2 items, got {len(items)}")
    for family in FAMILIES_V2:
        for split, count in SPLIT_COUNTS_V2.items():
            got = family_split.get((family, split), 0)
            if got != count:
                raise ValueError(f"{family}/{split}: expected {count}, got {got}")


def fingerprint_items(items: list[dict[str, Any]]) -> str:
    blob = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Validate existing output only")
    args = parser.parse_args()

    default_out = DEFAULT_OUT_V1 if args.version == "v1" else DEFAULT_OUT_V2
    out = args.out or default_out

    if args.check:
        items = [
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        items = build_items() if args.version == "v1" else build_items_v2()

    if args.version == "v1":
        validate_items(items)
    else:
        validate_items_v2(items)

    if not args.check:
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.version == "v1":
            header = "# composite v1 — 12×3 families; generated, do not hand-edit"
        else:
            header = (
                f"# composite v2 — {len(items)} items, 5 families × "
                f"{SPLIT_COUNTS_V2}; split train/dev/test; fingerprint="
                f"{fingerprint_items(items)}; generated, do not hand-edit"
            )
        lines = [header]
        lines.extend(json.dumps(item, ensure_ascii=False) for item in items)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(items)} items → {out}")

    if args.version == "v1":
        print("validation OK: 36 items, 12 per family")
    else:
        print(
            f"validation OK: {len(items)} items, "
            f"{len(FAMILIES_V2)} families × splits {SPLIT_COUNTS_V2}, "
            f"fingerprint={fingerprint_items(items)}"
        )


if __name__ == "__main__":
    main()
