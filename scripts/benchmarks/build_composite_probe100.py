#!/usr/bin/env python3
"""Build composite probe-100: 100 new items in the composite v2 format.

Same five families, step structure and rubrics as composite v2
(scripts/benchmarks/build_composite_benchmark.py), new content (no source shared with v2),
20 items per family. Each family has four wordings of the user request, five
items each:

  A  the v2 wording, word for word
  B  a paraphrase with two task verbs ("переклади ... і напиши ...")
  C  a paraphrase with sequence words ("спершу ... потім ...")
  D  a paraphrase that names the parts as nouns, with no sequence word

The wordings were written before any model ran on these items. This set is a
probe of planning and step execution, not a claim suite: no number from it goes
next to v7 or composite v2 numbers.

    python scripts/benchmarks/build_composite_probe100.py
    python scripts/benchmarks/build_composite_probe100.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

from build_composite_benchmark import (  # noqa: E402
    FAMILIES_V2,
    _validate_plan_shape,
    code_item,
    extract_classify_item,
    knowledge_item,
    passage_item,
    summarize_format_item,
)

OUT = ROOT / "benchmarks" / "archive" / "mixed_ua_composite_probe100.jsonl"
V2 = ROOT / "benchmarks" / "archive" / "mixed_ua_composite_v2.jsonl"
PER_FAMILY = 20
WORDINGS = "ABCD"

# (function, English spec, two UA stems the translation must contain, case 1, case 2)
CODE_POOL = [
    ("count_upper", "Count uppercase ASCII letters in a string.", "велик", "літер", (["HeLLo"], 3), (["abc"], 0)),
    ("sum_digits", "Return the sum of the decimal digits of a non-negative integer.", "сум", "цифр", ([1234], 10), ([0], 0)),
    ("dedupe_keep_order", "Remove repeated items from a list, keeping the order of first occurrences.", "поряд", "спис", ([[3, 1, 3, 2, 1]], [3, 1, 2]), ([[]], [])),
    ("count_char", "Count how many times the character c occurs in the string s.", "символ", "рядк", (["banana", "a"], 3), (["abc", "z"], 0)),
    ("square_list", "Return a new list with the square of every integer.", "квадрат", "спис", ([[1, 2, 3]], [1, 4, 9]), ([[]], [])),
    ("is_even_length", "Return True if a string has an even number of characters.", "парн", "символ", (["ab"], True), (["abc"], False)),
    ("capitalize_first", "Return the string with its first character in uppercase and the rest unchanged; an empty string stays empty.", "перш", "велик", (["kyiv"], "Kyiv"), ([""], "")),
    ("count_negatives", "Count the negative integers in a list.", "ємн", "спис", ([[-1, 2, -3]], 2), ([[]], 0)),
    ("reverse_string", "Return the characters of a string in reverse order.", "символ", "поряд", (["abc"], "cba"), ([""], "")),
    ("max_index", "Return the index of the largest value in a non-empty list; if it is tied, return the first index.", "індекс", "найбільш", ([[1, 5, 5, 2]], 1), ([[7]], 0)),
    ("sum_odd", "Return the sum of all odd integers in a list.", "сум", "непарн", ([[1, 2, 3]], 4), ([[]], 0)),
    ("ends_with_dot", "Return True if a string ends with a full stop.", "крапк", "закінч", (["Hi."], True), (["Hi"], False)),
    ("long_words", "Return the words of a sentence that are longer than three characters, in their original order.", "довш", "слов", (["a big elephant ran"], ["elephant"]), ([""], [])),
    ("celsius_to_fahrenheit", "Convert a temperature from Celsius to Fahrenheit using F = C * 9 / 5 + 32.", "цельсі", "фаренгейт", ([100], 212.0), ([0], 32.0)),
    ("count_even", "Count the even integers in a list.", "парн", "спис", ([[1, 2, 4]], 2), ([[]], 0)),
    ("first_word", "Return the first whitespace-separated word of a string, or an empty string if there is none.", "перш", "слов", (["hello big world"], "hello"), (["   "], "")),
    ("range_of", "Return the difference between the largest and the smallest integer in a non-empty list.", "різниц", "найбільш", ([[3, 9, 1]], 8), ([[5]], 0)),
    ("pad_zeros", "Pad a non-negative integer with leading zeros to width n and return it as a string.", "нул", "ряд", ([7, 3], "007"), ([1234, 2], "1234")),
    ("has_digit", "Return True if a string contains at least one digit.", "цифр", "ряд", (["ab3"], True), (["abc"], False)),
    ("sum_squares", "Return the sum of the squares of the integers in a list.", "сум", "квадрат", ([[1, 2, 3]], 14), ([[]], 0)),
]

# (question, options, gold letter, stems the explanation must contain)
KNOWLEDGE_POOL = [
    ("Автор повісті «Захар Беркут»:", ["А Іван Франко", "Б Панас Мирний", "В Марко Вовчок"], "А", ["Франк", "Беркут"]),
    ("Столиця Японії:", ["А Осака", "Б Кіото", "В Токіо"], "В", ["Токіо", "Япон"]),
    ("Хімічний символ натрію:", ["А Na", "Б N", "В Ne"], "А", ["Na", "натр"]),
    ("Найбільший океан Землі:", ["А Атлантичний", "Б Тихий", "В Індійський"], "Б", ["Тих", "океан"]),
    ("Автор роману «Місто»:", ["А Валер'ян Підмогильний", "Б Микола Хвильовий", "В Іван Багряний"], "А", ["Підмогильн", "Місто"]),
    ("Рік хрещення Київської Русі за літописом:", ["А 862", "Б 988", "В 1054"], "Б", ["988", "хрещ"]),
    ("Одиниця вимірювання потужності:", ["А джоуль", "Б ват", "В ньютон"], "Б", ["ват", "потужн"]),
    ("Столиця Канади:", ["А Торонто", "Б Монреаль", "В Оттава"], "В", ["Оттав", "Канад"]),
    ("Автор новели «Intermezzo»:", ["А Михайло Коцюбинський", "Б Іван Франко", "В Василь Стефаник"], "А", ["Коцюбинськ", "Intermezzo"]),
    ("Найбільший материк Землі:", ["А Африка", "Б Євразія", "В Північна Америка"], "Б", ["Євразі", "материк"]),
    ("Хімічна формула кухонної солі:", ["А KCl", "Б CaCO3", "В NaCl"], "В", ["NaCl", "сол"]),
    ("Автор поеми «Мойсей»:", ["А Іван Франко", "Б Тарас Шевченко", "В Леся Українка"], "А", ["Франк", "Мойсей"]),
    ("Скільки градусів має прямий кут:", ["А 45", "Б 90", "В 180"], "Б", ["90", "прям"]),
    ("Столиця Німеччини:", ["А Мюнхен", "Б Гамбург", "В Берлін"], "В", ["Берлін", "Німеч"]),
    ("Автор повісті «Тіні забутих предків»:", ["А Ольга Кобилянська", "Б Михайло Коцюбинський", "В Василь Стефаник"], "Б", ["Коцюбинськ", "предк"]),
    ("Газ, якого найбільше в атмосфері Землі:", ["А кисень", "Б вуглекислий газ", "В азот"], "В", ["азот", "атмосфер"]),
    ("Конституція 1710 року пов'язана з іменем гетьмана:", ["А Пилипа Орлика", "Б Івана Мазепи", "В Богдана Хмельницького"], "А", ["Орлик", "1710"]),
    ("Скільки планет у Сонячній системі:", ["А 7", "Б 8", "В 9"], "Б", ["планет", "Сонячн"]),
    ("Авторка повісті «Земля»:", ["А Ольга Кобилянська", "Б Леся Українка", "В Марко Вовчок"], "А", ["Кобилянськ", "Земл"]),
    ("Одиниця вимірювання частоти:", ["А паскаль", "Б герц", "В кельвін"], "Б", ["герц", "частот"]),
]

# (English passage, gold fields in Ukrainian)
PASSAGE_POOL = [
    ("The Kyiv Pechersk Lavra was founded in 1051. It stands in Kyiv.", {"subject": "Києво-Печерська лавра", "number": 1051, "place": "Київ"}),
    ("The Khotyn Fortress was rebuilt in 1400. It stands on the Dniester.", {"subject": "Хотинська фортеця", "number": 1400, "place": "Дністер"}),
    ("The Lviv Opera House opened in 1900. It is in Lviv.", {"subject": "Львівська опера", "number": 1900, "place": "Львів"}),
    ("The Derzhprom building was completed in 1928. It stands in Kharkiv.", {"subject": "Держпром", "number": 1928, "place": "Харків"}),
    ("The Potemkin Stairs were completed in 1841. They are in Odesa.", {"subject": "Потьомкінські сходи", "number": 1841, "place": "Одеса"}),
    ("Chernivtsi University was founded in 1875. It is in Chernivtsi.", {"subject": "Чернівецький університет", "number": 1875, "place": "Чернівці"}),
    ("Shatsk National Park was created in 1983. It is in Volyn region.", {"subject": "Шацький національний парк", "number": 1983, "place": "Волинська область"}),
    ("The Kyiv Funicular opened in 1905. It links the Upper Town with Podil.", {"subject": "Київський фунікулер", "number": 1905, "place": "Поділ"}),
    ("Olesko Castle was first mentioned in 1327. It is in Lviv region.", {"subject": "Олеський замок", "number": 1327, "place": "Львівська область"}),
    ("The Dnipro Metro opened in 1995. It operates in Dnipro.", {"subject": "Дніпровський метрополітен", "number": 1995, "place": "Дніпро"}),
    ("The Kyiv Zoo was founded in 1908. It is in Kyiv.", {"subject": "Київський зоопарк", "number": 1908, "place": "Київ"}),
    ("The Pochaiv Lavra was first mentioned in 1240. It is in Ternopil region.", {"subject": "Почаївська лавра", "number": 1240, "place": "Тернопільська область"}),
    ("The Museum of Cosmonautics opened in 1970. It is in Zhytomyr.", {"subject": "Музей космонавтики", "number": 1970, "place": "Житомир"}),
    ("Synevyr National Park was created in 1989. It is in Zakarpattia.", {"subject": "Національний парк Синевир", "number": 1989, "place": "Закарпаття"}),
    ("Lviv Polytechnic was founded in 1844. It is in Lviv.", {"subject": "Львівська політехніка", "number": 1844, "place": "Львів"}),
    ("The Kyiv TV Tower was completed in 1973. It stands in Kyiv.", {"subject": "Київська телевежа", "number": 1973, "place": "Київ"}),
    ("The DniproHES dam started operation in 1932. It is in Zaporizhzhia.", {"subject": "ДніпроГЕС", "number": 1932, "place": "Запоріжжя"}),
    ("The Medzhybizh Fortress dates to 1516. It is in Khmelnytskyi region.", {"subject": "Меджибізька фортеця", "number": 1516, "place": "Хмельницька область"}),
    ("The Golden Gate of Kyiv was built in 1037. It stands in Kyiv.", {"subject": "Золоті ворота", "number": 1037, "place": "Київ"}),
    ("The Odesa Film Studio was founded in 1919. It is in Odesa.", {"subject": "Одеська кіностудія", "number": 1919, "place": "Одеса"}),
]

# Ukrainian notice with one number in it
EXTRACT_POOL = [
    {"text": "Терміново: прорив водогону на вул. Садовій. Аварійна служба 1563.", "topic": "utilities", "urgency": "high", "subject": "прорив водогону", "number": 1563},
    {"text": "Обласна бібліотека проводить книжковий ярмарок 12 квітня. Вхід вільний.", "topic": "culture", "urgency": "low", "subject": "книжковий ярмарок", "number": 12},
    {"text": "Гімназія нагадує про контрольну з алгебри наступного вівторка в кабінеті 21.", "topic": "education", "urgency": "medium", "subject": "контрольна з алгебри", "number": 21},
    {"text": "Планове відключення світла в четвер зранку. Довідки за номером 1550.", "topic": "utilities", "urgency": "medium", "subject": "відключення світла", "number": 1550},
    {"text": "Краєзнавчий клуб запрошує на лекцію про козацтво в залі 3.", "topic": "culture", "urgency": "low", "subject": "лекція про козацтво", "number": 3},
    {"text": "Університет призначає додаткову сесію для групи 17 через карантин.", "topic": "education", "urgency": "medium", "subject": "додаткова сесія", "number": 17},
    {"text": "Увага: витік аміаку на складі. Не виходьте з дому, телефон 101.", "topic": "utilities", "urgency": "high", "subject": "витік аміаку", "number": 101},
    {"text": "Театр імпровізації дає благодійну виставу, квитки по 150 грн.", "topic": "culture", "urgency": "low", "subject": "благодійна вистава", "number": 150},
    {"text": "Школа скасовує завтрашні уроки через аварію опалення. Корпус 4.", "topic": "education", "urgency": "high", "subject": "скасування уроків", "number": 4},
    {"text": "Водоканал проводить планову заміну труб у мікрорайоні 6.", "topic": "utilities", "urgency": "medium", "subject": "заміна труб", "number": 6},
    {"text": "Джаз-фестиваль у парку триває 3 дні, вхід безкоштовний.", "topic": "culture", "urgency": "low", "subject": "джаз-фестиваль", "number": 3},
    {"text": "Коледж нагадує: заяви на гуртожиток приймають у кімнаті 114.", "topic": "education", "urgency": "low", "subject": "заяви на гуртожиток", "number": 114},
    {"text": "Терміново: пошкоджено газопровід на вул. Липовій. Служба газу 104.", "topic": "utilities", "urgency": "high", "subject": "пошкодження газопроводу", "number": 104},
    {"text": "Музей історії міста відкриває зал сучасного мистецтва на 2 поверсі.", "topic": "culture", "urgency": "low", "subject": "зал сучасного мистецтва", "number": 2},
    {"text": "Ліцей переносить олімпіаду з фізики в аудиторію 12.", "topic": "education", "urgency": "medium", "subject": "олімпіада з фізики", "number": 12},
    {"text": "Енергетики попереджають про аварійні відключення сьогодні ввечері. Гаряча лінія 1588.", "topic": "utilities", "urgency": "high", "subject": "аварійні відключення", "number": 1588},
    {"text": "Хор ветеранів шукає нових учасників, репетиції в кімнаті 9.", "topic": "culture", "urgency": "low", "subject": "набір у хор", "number": 9},
    {"text": "Університет переносить дедлайн курсових робіт для групи 31.", "topic": "education", "urgency": "medium", "subject": "дедлайн курсових робіт", "number": 31},
    {"text": "Комунальники попереджають про ремонт ліфтів у будинку 45 цього тижня.", "topic": "utilities", "urgency": "medium", "subject": "ремонт ліфтів", "number": 45},
    {"text": "Дитяча школа мистецтв запрошує на звітний концерт у залу на 250 місць.", "topic": "culture", "urgency": "low", "subject": "звітний концерт", "number": 250},
]

# English brief, gold JSON fields in English, two UA stems the translation must contain
SUMMARIZE_POOL = [
    {"source": "The central library closes early at 16:00 this Saturday for inventory.", "topic": "library", "when": "this Saturday", "action": "close early", "ua_hints": ["бібліотек", "16:00"]},
    {"source": "Trolleybus route 12 will run every ten minutes from next Monday.", "topic": "trolleybus", "when": "next Monday", "action": "run", "ua_hints": ["тролейбус", "понеділ"]},
    {"source": "The zoo opens its new penguin house to visitors on Friday morning.", "topic": "zoo", "when": "Friday morning", "action": "open", "ua_hints": ["зоопарк", "пінгвін"]},
    {"source": "Water in the eastern district will be off from 10:00 to 14:00 tomorrow.", "topic": "water", "when": "tomorrow", "action": "off", "ua_hints": ["вод", "10:00"]},
    {"source": "The art museum offers free entry to students every Thursday.", "topic": "museum", "when": "every Thursday", "action": "free entry", "ua_hints": ["музе", "четвер"]},
    {"source": "Night buses will not run during the marathon on Sunday.", "topic": "buses", "when": "Sunday", "action": "not run", "ua_hints": ["автобус", "марафон"]},
    {"source": "The city park bans barbecues until the end of August because of drought.", "topic": "park", "when": "until the end of August", "action": "ban", "ua_hints": ["парк", "серп"]},
    {"source": "The swimming school starts a new course for adults on 1 October.", "topic": "swimming school", "when": "1 October", "action": "start", "ua_hints": ["плаван", "жовт"]},
    {"source": "Street lights on the bridge will be repaired tonight after 23:00.", "topic": "street lights", "when": "tonight", "action": "repair", "ua_hints": ["міст", "23:00"]},
    {"source": "The farmers' market moves to the stadium parking lot from May.", "topic": "market", "when": "from May", "action": "move", "ua_hints": ["ринок", "трав"]},
    {"source": "The planetarium shows a children's programme every Sunday at 11:00.", "topic": "planetarium", "when": "every Sunday", "action": "show", "ua_hints": ["планетар", "11:00"]},
    {"source": "Taxi stands near the station are closed for roadworks this week.", "topic": "taxi", "when": "this week", "action": "close", "ua_hints": ["таксі", "вокзал"]},
    {"source": "The theatre box office opens an hour earlier during the festival.", "topic": "box office", "when": "during the festival", "action": "open", "ua_hints": ["театр", "фестивал"]},
    {"source": "Parking on Central Square is free on public holidays.", "topic": "parking", "when": "public holidays", "action": "free", "ua_hints": ["паркув", "свят"]},
    {"source": "The ski lift stops when the wind is stronger than the safety limit.", "topic": "ski lift", "when": "strong wind", "action": "stop", "ua_hints": ["підйомник", "вітер"]},
    {"source": "The children's hospital opens a new playroom on Wednesday.", "topic": "hospital", "when": "Wednesday", "action": "open", "ua_hints": ["лікарн", "серед"]},
    {"source": "Recycling bins for glass will be emptied every Tuesday morning.", "topic": "recycling", "when": "every Tuesday morning", "action": "empty", "ua_hints": ["скл", "вівтор"]},
    {"source": "The river beach closes this afternoon after the storm warning.", "topic": "beach", "when": "this afternoon", "action": "close", "ua_hints": ["пляж", "штор"]},
    {"source": "The science fair accepts student projects until 20 March.", "topic": "science fair", "when": "until 20 March", "action": "accept", "ua_hints": ["студент", "берез"]},
    {"source": "The old tram depot hosts a craft fair on the last weekend of June.", "topic": "craft fair", "when": "last weekend of June", "action": "host", "ua_hints": ["трам", "черв"]},
]

# Wordings B, C, D per family. A keeps the prompt that the v2 builder writes.
WORDING_TEMPLATES: dict[str, dict[str, str]] = {
    "translate_code": {
        "B": (
            "Маю англійський опис функції. Переклади його українською і напиши за ним "
            "Python-функцію `{fn}`. У відповіді — тільки блок ```python, без пояснень.\n\n"
            "Опис: {spec}"
        ),
        "C": (
            "Спершу переклади цю специфікацію українською, далі на її основі реалізуй "
            "`{fn}` мовою Python. Віддай лише код у блоці ```python.\n\n{spec}"
        ),
        "D": (
            "Потрібні дві речі: український переклад специфікації та Python-функція "
            "`{fn}` за нею. Фінальна відповідь — лише блок ```python.\n\n"
            "Specification: {spec}"
        ),
    },
    "knowledge_explain": {
        "B": (
            "Обери правильний варіант і поясни, чому він правильний. Відповідь дай рівно "
            "у двох рядках: `Відповідь: <літера>` і `Пояснення: <одне речення>`.\n\n"
            "{question}\n{options}"
        ),
        "C": (
            "Ось питання з тесту. Спочатку визнач правильну відповідь, потім одним "
            "реченням обґрунтуй її. Формат — два рядки: `Відповідь: <літера>` та "
            "`Пояснення: <одне речення>`.\n\n{question}\n{options}"
        ),
        "D": (
            "Потрібна правильна відповідь на тестове питання разом із коротким "
            "поясненням. Формат — рівно два рядки: `Відповідь: <літера>` та "
            "`Пояснення: <одне речення>`.\n\n{question}\n{options}"
        ),
    },
    "translate_knowledge_write": {
        "B": (
            "Переклади англійський текст українською, знайди в ньому назву об'єкта, рік "
            "і місце та поверни лише валідний JSON з ключами `subject`, `number`, `place`. "
            "Користуйся тільки цим текстом.\n\nText: {passage}"
        ),
        "C": (
            "Спершу переклади джерело українською, потім витягни з перекладу три факти "
            "(що, коли, де). Наостанок оформи їх як валідний JSON з ключами `subject`, "
            "`number`, `place` — без пояснень і без зовнішніх знань.\n\nSource: {passage}"
        ),
        "D": (
            "З англомовного джерела нижче потрібен JSON з трьома фактами українською: "
            "`subject`, `number`, `place`. Лише відомості з цього тексту. Поверни тільки "
            "валідний JSON.\n\nSource: {passage}"
        ),
    },
    "extract_classify_write": {
        "B": (
            "Прочитай оголошення, витягни його предмет і число (телефон, номер чи "
            "кількість), визнач тему й терміновість і поверни лише валідний JSON з ключами "
            "`topic`, `urgency`, `subject`, `number`. topic: utilities, culture або "
            "education. urgency: low, medium або high. Без пояснень.\n\nОголошення:\n{text}"
        ),
        "C": (
            "Спочатку витягни з оголошення subject і number, потім класифікуй його за "
            "темою (utilities / culture / education) і терміновістю (low / medium / high). "
            "Фінально поверни лише валідний JSON з ключами `topic`, `urgency`, `subject`, "
            "`number`.\n\nОголошення:\n{text}"
        ),
        "D": (
            "Для оголошення нижче потрібен JSON з ключами `topic`, `urgency`, `subject`, "
            "`number`. Дозволені topic: utilities, culture, education; дозволені urgency: "
            "low, medium, high. Лише JSON, без пояснень.\n\n{text}"
        ),
    },
    "translate_summarize_format": {
        "B": (
            "Переклади це повідомлення українською, стисни його до одного речення й "
            "поверни лише валідний JSON з ключами `topic`, `when`, `action`. Нічого не "
            "додавай від себе.\n\nMessage: {source}"
        ),
        "C": (
            "Спершу переклади повідомлення українською. Потім зроби з перекладу резюме "
            "в одне речення. Насамкінець поверни лише валідний JSON з ключами `topic`, "
            "`when`, `action`, без зовнішніх відомостей.\n\nSource: {source}"
        ),
        "D": (
            "Для англійського повідомлення нижче потрібні український переклад, резюме "
            "в одне речення і, як фінальна відповідь, лише валідний JSON з ключами "
            "`topic`, `when`, `action`.\n\nSource: {source}"
        ),
    },
}

SHORT = {
    "translate_code": "tc",
    "knowledge_explain": "ke",
    "translate_knowledge_write": "tkw",
    "extract_classify_write": "ecw",
    "translate_summarize_format": "tsf",
}


def _fill(family: str, wording: str, data: Any) -> str:
    template = WORDING_TEMPLATES[family][wording]
    if family == "translate_code":
        return template.format(fn=data[0], spec=data[1])
    if family == "knowledge_explain":
        return template.format(question=data[0], options="\n".join(data[1]))
    if family == "translate_knowledge_write":
        return template.format(passage=data[0])
    if family == "extract_classify_write":
        return template.format(text=data["text"])
    return template.format(source=data["source"])


def build_items() -> list[dict[str, Any]]:
    makers = {
        "translate_code": (CODE_POOL, lambda i, d: code_item(i, d, split="probe", version="v2")),
        "knowledge_explain": (KNOWLEDGE_POOL, lambda i, d: knowledge_item(i, d, split="probe", version="v2")),
        "translate_knowledge_write": (PASSAGE_POOL, lambda i, d: passage_item(i, d, split="probe", version="v2")),
        "extract_classify_write": (EXTRACT_POOL, lambda i, d: extract_classify_item(i, d, split="probe")),
        "translate_summarize_format": (SUMMARIZE_POOL, lambda i, d: summarize_format_item(i, d, split="probe")),
    }
    items: list[dict[str, Any]] = []
    for family in FAMILIES_V2:
        pool, make = makers[family]
        for i, data in enumerate(pool, 1):
            item = make(i, data)
            wording = WORDINGS[(i - 1) % len(WORDINGS)]
            if wording != "A":
                item["prompt"] = _fill(family, wording, data)
            item["id"] = f"cp100-{SHORT[family]}-{i:03d}"
            item["wording"] = wording
            item["provenance"] = f"{item['provenance']}; probe100"
            items.append(item)
    return items


def validate(items: list[dict[str, Any]]) -> None:
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids")
    keys = [item["source_key"] for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate source_key")
    v2_keys = {
        json.loads(line)["source_key"]
        for line in V2.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    shared = set(keys) & v2_keys
    if shared:
        raise ValueError(f"sources shared with composite v2: {sorted(shared)[:3]}")
    counts: dict[tuple[str, str], int] = {}
    for item in items:
        _validate_plan_shape(item)
        if "oracle_plan" in item["prompt"]:
            raise ValueError(f"{item['id']}: oracle leaked into prompt")
        counts[(item["family"], item["wording"])] = counts.get((item["family"], item["wording"]), 0) + 1
    for family in FAMILIES_V2:
        for wording in WORDINGS:
            if counts.get((family, wording)) != PER_FAMILY // len(WORDINGS):
                raise ValueError(f"{family}/{wording}: {counts.get((family, wording))}")
    if len(items) != PER_FAMILY * len(FAMILIES_V2):
        raise ValueError(f"expected {PER_FAMILY * len(FAMILIES_V2)} items, got {len(items)}")


def fingerprint(items: list[dict[str, Any]]) -> str:
    blob = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true", help="Validate the existing file only")
    args = parser.parse_args()
    if args.check:
        items = [
            json.loads(line)
            for line in args.out.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if fingerprint(items) != fingerprint(build_items()):
            raise SystemExit("file differs from the builder output; rebuild it")
    else:
        items = build_items()
    validate(items)
    if not args.check:
        header = (
            f"# composite probe-100 — {len(items)} items, 5 families × 20, wordings A-D × 5; "
            f"probe only, not for claims; fingerprint={fingerprint(items)}; generated, do not hand-edit"
        )
        lines = [header, *(json.dumps(item, ensure_ascii=False) for item in items)]
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(items)} items → {args.out}")
    print(f"validation OK: {len(items)} items, fingerprint={fingerprint(items)}")


if __name__ == "__main__":
    main()
