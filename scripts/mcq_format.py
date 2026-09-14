"""Shared MCQ prompt / answer letter helpers for warehouse extractors."""

from __future__ import annotations

from typing import Any

LATIN = "ABCDEFGHI"
CYR = "АБВГДЕЄЖЗ"


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return [value]


def letter_from_answer(answer: Any, n_choices: int, *, one_indexed: bool = False) -> str:
    """Map index / letter / digit to a Latin A–I letter.

    Integer answers are 0-based by default (MMLU). Pass one_indexed=True for
    Belebele-style 1..n labels.
    """
    if n_choices <= 0:
        return "A"
    letters = LATIN[:n_choices]
    if answer is None:
        return letters[0]
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        idx = int(answer)
        if one_indexed:
            if 1 <= idx <= n_choices:
                return letters[idx - 1]
            if 0 <= idx < n_choices:
                return letters[idx]
            return letters[0]
        if 0 <= idx < n_choices:
            return letters[idx]
        if 1 <= idx <= n_choices:
            return letters[idx - 1]
        return letters[0]
    text = str(answer).strip()
    if not text:
        return letters[0]
    ch = text[0]
    upper = ch.upper()
    if upper in letters:
        return upper
    cyr_i = CYR.find(ch.upper() if ch.upper() in CYR else ch)
    if cyr_i >= 0 and cyr_i < n_choices:
        return letters[cyr_i]
    if text.isdigit():
        return letter_from_answer(int(text), n_choices, one_indexed=one_indexed)
    return letters[0]


def format_mcq_prompt(
    question: str,
    choices: list[str],
    *,
    passage: str = "",
    intro: str = "",
) -> str:
    letters = LATIN[: len(choices)]
    intro = intro or (
        "Дай відповідь на тестове завдання. "
        f"Відповідь — лише літера варіанту ({', '.join(letters)})."
    )
    lines = [intro, ""]
    if passage.strip():
        lines.extend([passage.strip(), ""])
    lines.extend([question.strip(), "", "Варіанти:"])
    for letter, choice in zip(letters, choices, strict=False):
        lines.append(f"{letter}) {str(choice).strip()}")
    return "\n".join(lines)


def choices_from_item(item: dict[str, Any]) -> list[str]:
    raw = item.get("choices")
    if raw is None and isinstance(item.get("options"), list):
        raw = item["options"]
    if isinstance(raw, dict):
        texts = raw.get("text") or raw.get("texts")
        if texts is not None:
            return [str(x).strip() for x in as_list(texts)]
    if raw is not None and not isinstance(raw, dict):
        seq = as_list(raw)
        if seq and not isinstance(seq[0], (dict, list)):
            return [str(x).strip() for x in seq]
    labeled = []
    for key in LATIN:
        if item.get(key) is not None:
            labeled.append(str(item[key]).strip())
        elif item.get(key.lower()) is not None:
            labeled.append(str(item[key.lower()]).strip())
    if labeled:
        return labeled
    numbered = []
    for i in range(1, 6):
        val = item.get(f"mc_answer{i}") or item.get(f"answer{i}")
        if val is None:
            break
        numbered.append(str(val).strip())
    return numbered
