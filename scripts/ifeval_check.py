"""IFEval-style constraint checker (prompt-level fraction of instructions).

Covers the instruction ids used by google-research IFEval / INSAIT ifeval_ukr.
Does not depend on the official package.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from scripts.mcq_format import as_list

PLACEHOLDER_RE = re.compile(r"\[[A-Z0-9_ -]{1,20}\]")
HIGHLIGHT_RE = re.compile(r"\*[^*]+\*")
TITLE_RE = re.compile(r"<<.+>>")
BULLET_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S+")
SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]+|[^.!?…]+$")
POSTSCRIPT_RE = re.compile(r"(?im)\b(p\.?\s*s\.?|постскриптум|н\.?\s*б\.?)\b")
UA_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")
LATIN_RE = re.compile(r"[A-Za-z]")


def _kwargs_list(raw: Any, n: int) -> list[dict[str, Any]]:
    items = as_list(raw)
    out: list[dict[str, Any]] = []
    for i in range(n):
        if i < len(items):
            val = items[i]
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    val = {}
            out.append(dict(val) if isinstance(val, dict) else {})
        else:
            out.append({})
    return out


def _words(text: str) -> list[str]:
    return re.findall(r"\w+(?:['’]\w+)?", text, flags=re.U)


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_RE.findall(text or "") if s.strip()]
    return parts


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def _relation_ok(value: int, target: int, relation: str) -> bool:
    rel = (relation or "at least").lower().replace("_", " ").strip()
    if rel in {"at least", "least", ">="}:
        return value >= target
    if rel in {"at most", "most", "<="}:
        return value <= target
    if rel in {"equal", "exactly", "=="}:
        return value == target
    return value >= target


def _lang_ok(text: str, lang: str) -> bool:
    code = (lang or "en").lower()
    ua = len(UA_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    total = max(1, ua + latin)
    if code in {"uk", "ukr", "ua", "ukrainian"}:
        return ua / total >= 0.6
    if code in {"en", "eng", "english"}:
        return latin / total >= 0.6
    return True


def check_instruction(instruction_id: str, text: str, kwargs: dict[str, Any], prompt: str = "") -> bool:
    iid = str(instruction_id or "").strip()
    kw = kwargs or {}
    checkers: dict[str, Callable[[], bool]] = {
        "keywords:existence": lambda: all(
            str(k).lower() in text.lower() for k in as_list(kw.get("keywords"))
        ),
        "keywords:frequency": lambda: _relation_ok(
            sum(text.lower().count(str(k).lower()) for k in as_list(kw.get("keywords") or kw.get("keyword"))),
            int(kw.get("frequency") or kw.get("num_keywords") or 1),
            str(kw.get("relation") or "at least"),
        ),
        "keywords:forbidden_words": lambda: all(
            str(k).lower() not in text.lower() for k in as_list(kw.get("forbidden_words"))
        ),
        "keywords:letter_frequency": lambda: _relation_ok(
            text.lower().count(str(kw.get("letter") or "a").lower()),
            int(kw.get("let_frequency") or kw.get("frequency") or 1),
            str(kw.get("let_relation") or kw.get("relation") or "at least"),
        ),
        "language:response_language": lambda: _lang_ok(text, str(kw.get("language") or "en")),
        "length_constraints:number_sentences": lambda: _relation_ok(
            len(_sentences(text)),
            int(kw.get("num_sentences") or 1),
            str(kw.get("relation") or "at least"),
        ),
        "length_constraints:number_paragraphs": lambda: _relation_ok(
            len(_paragraphs(text)),
            int(kw.get("num_paragraphs") or 1),
            str(kw.get("relation") or "exactly"),
        ),
        "length_constraints:number_words": lambda: _relation_ok(
            len(_words(text)),
            int(kw.get("num_words") or 1),
            str(kw.get("relation") or "at least"),
        ),
        "length_constraints:nth_paragraph_first_word": lambda: _nth_paragraph_first_word(text, kw),
        "detectable_content:number_placeholders": lambda: _relation_ok(
            len(PLACEHOLDER_RE.findall(text)),
            int(kw.get("num_placeholders") or 1),
            str(kw.get("relation") or "at least"),
        ),
        "detectable_content:postscript": lambda: bool(POSTSCRIPT_RE.search(text))
        and str(kw.get("postscript_marker") or "").lower() in text.lower()
        if kw.get("postscript_marker")
        else bool(POSTSCRIPT_RE.search(text)),
        "detectable_content:number_bullet_lists": lambda: _relation_ok(
            len(BULLET_RE.findall(text)),
            int(kw.get("num_bullets") or 1),
            "at least",
        ),
        "detectable_format:number_highlighted_sections": lambda: _relation_ok(
            len(HIGHLIGHT_RE.findall(text)),
            int(kw.get("num_highlights") or 1),
            "at least",
        ),
        "detectable_format:title": lambda: bool(TITLE_RE.search(text)),
        "detectable_format:json_format": lambda: _is_json(text),
        "detectable_format:number_json": lambda: _is_json(text),
        "detectable_format:multiple_sections": lambda: _multiple_sections(text, kw),
        "detectable_format:constrained_response": lambda: text.strip()
        in {
            "My answer is yes.",
            "My answer is no.",
            "Моя відповідь — так.",
            "Моя відповідь — ні.",
        },
        "combination:two_responses": lambda: text.count("******") >= 1
        and len([p for p in text.split("******") if p.strip()]) >= 2,
        "combination:repeat_prompt": lambda: (prompt or "") and text.strip().startswith((prompt or "").strip()),
        "startend:end_checker": lambda: text.rstrip().endswith(str(kw.get("end_phrase") or "")),
        "startend:quotation": lambda: text.strip().startswith('"') and text.strip().endswith('"'),
        "change_case:english_capital": lambda: bool(text) and text == text.upper(),
        "change_case:english_lowercase": lambda: bool(text) and text == text.lower(),
        "change_case:capital_word_frequency": lambda: _relation_ok(
            sum(1 for w in _words(text) if w.isupper()),
            int(kw.get("capital_frequency") or 1),
            str(kw.get("capital_relation") or "at least"),
        ),
        "punctuation:no_comma": lambda: "," not in text,
    }
    fn = checkers.get(iid)
    if fn is None:
        # Unknown id: do not fail the whole item; treat as unscored pass-through 0.
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _nth_paragraph_first_word(text: str, kw: dict[str, Any]) -> bool:
    paras = _paragraphs(text)
    n = int(kw.get("nth_paragraph") or 1) - 1
    want = str(kw.get("first_word") or "").strip().lower()
    if n < 0 or n >= len(paras) or not want:
        return False
    words = _words(paras[n])
    return bool(words) and words[0].lower() == want


def _is_json(text: str) -> bool:
    blob = text.strip()
    try:
        json.loads(blob)
        return True
    except Exception:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", blob)
        if not m:
            return False
        try:
            json.loads(m.group(1))
            return True
        except Exception:
            return False


def _multiple_sections(text: str, kw: dict[str, Any]) -> bool:
    splitter = str(kw.get("section_spliter") or kw.get("section_splitter") or "Section")
    return text.lower().count(splitter.lower()) >= int(kw.get("num_sections") or 2)


def score_ifeval(
    content: str,
    instruction_id_list: Any,
    kwargs: Any,
    prompt: str = "",
) -> dict[str, Any]:
    ids = [str(x) for x in as_list(instruction_id_list)]
    if not ids:
        return {"score": 0.0, "detail": "ifeval_missing_ids"}
    kw_list = _kwargs_list(kwargs, len(ids))
    flags = [check_instruction(iid, content or "", kw_list[i], prompt) for i, iid in enumerate(ids)]
    passed = sum(flags)
    return {
        "score": passed / len(ids),
        "detail": f"ifeval {passed}/{len(ids)} " + ",".join(f"{iid}={'1' if ok else '0'}" for iid, ok in zip(ids, flags)),
        "ifeval_passed": passed,
        "ifeval_n": len(ids),
    }
