#!/usr/bin/env python3
"""Score mixed_ua benchmark result JSONL files.

Usage:
  python scripts/score_results.py results/router_20260725T081858Z.jsonl
  python scripts/score_results.py results/router_*.jsonl results/lapa_*.jsonl --out results/scores_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ifeval_check import score_ifeval  # noqa: E402
from scripts.mcq_format import letter_from_answer  # noqa: E402

UA_RE = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.I)

_HE_BY_ID: dict[str, dict[str, Any]] | None = None
_IFEVAL_BY_ID: dict[str, dict[str, Any]] | None = None


def humaneval_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Fill he_* from the suite if the result JSONL dropped them."""
    if row.get("he_test") and row.get("he_entry_point"):
        return row
    global _HE_BY_ID
    if _HE_BY_ID is None:
        _HE_BY_ID = {}
        for path in [
            ROOT / "benchmarks/corpus/humaneval_code_en.jsonl",
            ROOT / "benchmarks/mixed_ua_v5.jsonl",
        ]:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                item = json.loads(line)
                if item.get("he_test"):
                    _HE_BY_ID[str(item.get("id"))] = item
    extra = _HE_BY_ID.get(str(row.get("id") or ""), {})
    merged = dict(row)
    for k in ("he_prompt", "he_test", "he_entry_point"):
        if extra.get(k) and not merged.get(k):
            merged[k] = extra[k]
    return merged


def ifeval_fields(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("ifeval_instruction_id_list"):
        return row
    global _IFEVAL_BY_ID
    if _IFEVAL_BY_ID is None:
        _IFEVAL_BY_ID = {}
        for path in [
            ROOT / "benchmarks/corpus/ifeval_ukr_instruct.jsonl",
            ROOT / "benchmarks/mixed_ua_v6_screen.jsonl",
            ROOT / "benchmarks/mixed_ua_v6.jsonl",
        ]:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                item = json.loads(line)
                if item.get("ifeval_instruction_id_list"):
                    _IFEVAL_BY_ID[str(item.get("id"))] = item
    extra = _IFEVAL_BY_ID.get(str(row.get("id") or ""), {})
    merged = dict(row)
    for k in ("ifeval_instruction_id_list", "ifeval_kwargs", "prompt"):
        if extra.get(k) is not None and not merged.get(k):
            merged[k] = extra[k]
    if extra.get("notes") and not merged.get("notes"):
        merged["notes"] = extra["notes"]
    return merged


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    text = text.strip().lower().replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def chr_f(hyp: str, ref: str, n: int = 3) -> float:
    """Simple character n-gram F-score (chrF-like), no external deps."""
    hyp, ref = normalize(hyp), normalize(ref)
    if not hyp or not ref:
        return 0.0

    def ngrams(s: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for k in range(1, n + 1):
            for i in range(len(s) - k + 1):
                counts[s[i : i + k]] += 1
        return counts

    h, r = ngrams(hyp), ngrams(ref)
    if not h or not r:
        return 0.0
    overlap = sum(min(h[g], r[g]) for g in h if g in r)
    prec = overlap / max(1, sum(h.values()))
    rec = overlap / max(1, sum(r.values()))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def extract_json_obj(text: str) -> Any | None:
    text = text.strip()
    # try whole string
    for candidate in (text,):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # find first {...} or [...]
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, text)
        if not m:
            continue
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


def extract_code(text: str) -> str:
    m = CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: lines that look like python
    lines = [ln for ln in text.splitlines() if ln.strip().startswith(("def ", "class ", "import ", "from "))]
    return "\n".join(lines).strip() if lines else text.strip()


def score_chat(content: str) -> dict[str, Any]:
    content = content or ""
    ua = len(UA_RE.findall(content))
    latin = len(re.findall(r"[A-Za-z]", content))
    total = max(1, ua + latin)
    ua_ratio = ua / total
    length_ok = 20 <= len(content) <= 1200
    score = 0.0
    if length_ok:
        score += 0.5
    if ua_ratio >= 0.5:
        score += 0.5
    elif ua_ratio >= 0.2:
        score += 0.25
    return {"score": score, "ua_ratio": round(ua_ratio, 3), "length_ok": length_ok}


def score_translate(content: str, reference: str) -> dict[str, Any]:
    # Drop meta explanations after first paragraph when model over-explains
    hyp = (content or "").strip().split("\n\n")[0].strip()
    s = chr_f(hyp, reference or "")
    return {"score": s, "chrf": round(s, 4)}


def score_instruct(
    content: str,
    reference: str,
    item_id: str,
    notes: str,
    row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = (content or "").strip()
    ref = (reference or "").strip()
    notes = notes or ""
    merged = ifeval_fields(row or {"id": item_id, "notes": notes})
    if (
        str(item_id).startswith("ifeval")
        or "score_mode=ifeval" in notes
        or "source=ifeval" in notes
        or merged.get("ifeval_instruction_id_list")
    ):
        return score_ifeval(
            content,
            merged.get("ifeval_instruction_id_list"),
            merged.get("ifeval_kwargs"),
            str(merged.get("prompt") or ""),
        )

    if item_id == "if-001":
        obj = extract_json_obj(text)
        ok = isinstance(obj, dict) and obj.get("name") == "demo" and obj.get("score") == 7
        return {"score": 1.0 if ok else 0.0, "detail": "json_name_score"}

    if item_id == "if-002" or "exactly 3 bullets" in notes:
        bullets = re.findall(r"(?m)^\s*[-*•]\s+\S+", text)
        # also numbered
        numbered = re.findall(r"(?m)^\s*\d+[.)]\s+\S+", text)
        n = max(len(bullets), len(numbered))
        # fail if long preamble before first bullet
        return {"score": 1.0 if n == 3 else 0.0, "detail": f"bullets={n}"}

    if item_id == "if-003":
        norm = normalize(text).strip(" .!?,")
        ok = norm in {"так", "ні", "yes", "no"} and ("так" in norm or norm == "yes")
        # reference is так
        ok = normalize(text).startswith("так") or normalize(text) == "так"
        return {"score": 1.0 if ok else 0.0, "detail": "yes_no"}

    if item_id == "if-004" or "JSON array of 3" in notes:
        obj = extract_json_obj(text)
        ok = isinstance(obj, list) and len(obj) == 3 and all(isinstance(x, str) for x in obj)
        return {"score": 1.0 if ok else 0.0, "detail": "json_array3"}

    if item_id == "if-005" or "exactly 20" in notes:
        words = re.findall(r"\w+(?:'\w+)?", text, flags=re.U)
        n = len(words)
        return {"score": 1.0 if n == 20 else 0.0, "detail": f"words={n}"}

    if item_id == "if-006":
        m = re.search(r"-?\d+", text)
        ok = bool(m) and m.group(0) == "42" and len(re.findall(r"[A-Za-zА-Яа-я]", text)) == 0
        # allow pure number even with whitespace
        ok = bool(m) and m.group(0) == "42" and normalize(text) in {"42", "42."}
        return {"score": 1.0 if ok else 0.0, "detail": "number_only"}

    if item_id == "if-007" or "markdown table" in notes:
        lines = [ln for ln in text.splitlines() if "|" in ln]
        has_sep = any(re.search(r"\|?\s*-{2,}", ln) for ln in lines)
        ok = len(lines) >= 4 and has_sep  # header + sep + 3 rows
        return {"score": 1.0 if ok else 0.0, "detail": f"table_lines={len(lines)}"}

    if item_id == "if-008":
        has_ua = bool(UA_RE.search(text))
        has_kyiv = "київ" in normalize(text)
        return {"score": 1.0 if has_ua and has_kyiv else 0.5 if has_ua else 0.0, "detail": "ua_kyiv"}

    # v4 balanced instruction expansion (if-009…032)
    if item_id == "if-009":
        obj = extract_json_obj(text)
        ok = isinstance(obj, dict) and obj == {"city": "Lviv", "temp_c": 18}
        return {"score": float(ok), "detail": "json_city_temp"}
    if item_id == "if-010":
        return {"score": float(normalize(text).strip(" .!?") == "понеділок"), "detail": "one_word_monday"}
    if item_id == "if-011":
        bullets = re.findall(r"(?m)^\s*-\s+\S+", text)
        return {"score": float(len(bullets) == 4), "detail": f"bullets={len(bullets)}"}
    if item_id == "if-012":
        return {"score": float(text.strip() == "120"), "detail": "number_only_120"}
    if item_id == "if-013":
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        ok = text.startswith("Отже") and len(sentences) == 1
        return {"score": float(ok), "detail": f"starts_otzhe sentences={len(sentences)}"}
    if item_id == "if-014":
        ok = bool(re.fullmatch(r"\s*name:\s*Anna\s*\nrole:\s*student\s*", text))
        return {"score": float(ok), "detail": "yaml_name_role"}
    if item_id == "if-015":
        words = re.findall(r"\w+(?:['’]\w+)?", text, flags=re.U)
        return {"score": float(len(words) == 15), "detail": f"words={len(words)}"}
    if item_id == "if-016":
        parts = text.split(",")
        ok = len(parts) == 3 and all(p and p == p.strip() for p in parts) and "\n" not in text
        return {"score": float(ok), "detail": f"comma_parts={len(parts)}"}
    if item_id == "if-017":
        return {"score": float(text.strip() == "А"), "detail": "letter_only_A"}
    if item_id == "if-018":
        lines = [ln for ln in text.splitlines() if "|" in ln]
        has_sep = any(re.search(r"\|?\s*-{2,}", ln) for ln in lines)
        ok = len(lines) == 4 and has_sep and "україн" in normalize(text) and "польщ" in normalize(text)
        return {"score": float(ok), "detail": f"table_lines={len(lines)}"}
    if item_id == "if-019":
        obj = extract_json_obj(text)
        return {"score": float(obj == [3, 5]), "detail": "json_array_3_5"}
    if item_id == "if-020":
        return {"score": float(text.strip() == "ТАК"), "detail": "uppercase_yes"}
    if item_id == "if-021":
        return {"score": float(text.strip() == "2026-08-07"), "detail": "iso_date_only"}
    if item_id == "if-022":
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        ok = len(sentences) == 2 and sentences[0].startswith("Спочатку") and sentences[1].startswith("Потім")
        return {"score": float(ok), "detail": f"ordered_sentences={len(sentences)}"}
    if item_id == "if-023":
        ok = bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text.strip()))
        return {"score": float(ok), "detail": "email_only"}
    if item_id == "if-024":
        parts = [p.strip() for p in text.split(";")]
        ok = len(parts) == 3 and all(UA_RE.search(p) for p in parts) and not re.search(r"[A-Za-z]", text)
        return {"score": float(ok), "detail": f"semicolon_parts={len(parts)}"}
    if item_id == "if-025":
        return {"score": float(text.strip() == "0.25"), "detail": "decimal_only"}
    if item_id == "if-026":
        return {"score": float(text.strip() == "PORT=8080"), "detail": "env_assignment"}
    if item_id == "if-027":
        obj = extract_json_obj(text)
        return {"score": float(obj == {"ok": True}), "detail": "json_bool"}
    if item_id == "if-028":
        numbered = re.findall(r"(?m)^\s*[12]\.\s+\S+", text)
        return {"score": float(len(numbered) == 2), "detail": f"numbered={len(numbered)}"}
    if item_id == "if-029":
        return {"score": float(text.strip() == "+"), "detail": "plus_only"}
    if item_id == "if-030":
        return {"score": float(text.strip() == '"готово"'), "detail": "quoted_ready"}
    if item_id == "if-031":
        words = re.findall(r"\w+(?:['’]\w+)?", text, flags=re.U)
        ok = len(words) == 3 and bool(UA_RE.search(text))
        return {"score": float(ok), "detail": f"ua_words={len(words)}"}
    if item_id == "if-032":
        m = re.fullmatch(r"\s*```python\s*\n?([\s\S]*?)\n?```\s*", text, re.I)
        ok = bool(m) and m.group(1).strip() == "print(1)"
        return {"score": float(ok), "detail": "python_fence_only"}

    # generic exact
    if ref:
        return {"score": 1.0 if normalize(ref) in normalize(text) else 0.0, "detail": "contains_ref"}
    return {"score": 0.0, "detail": "unscored"}


def score_knowledge(content: str, reference: str) -> dict[str, Any]:
    text = normalize(content or "")
    ref = normalize(reference or "")
    if not ref:
        # open-ended: CRISP-DM etc. — keyword heuristic
        keys = ["crisp", "data mining", "cross-industry", "процес", "data"]
        hits = sum(1 for k in keys if k in text)
        return {"score": 1.0 if hits >= 2 else 0.5 if hits == 1 else 0.0, "detail": f"keywords={hits}"}

    # MCQ: ZNO (А–Д) and Belebele/MMLU/ARC (A–E). Latin and Cyrillic accepted.
    latin_to_cyr = {"a": "а", "b": "б", "c": "в", "d": "г", "e": "д"}
    cyr_to_latin = {v: k for k, v in latin_to_cyr.items()}
    raw_ref = (reference or "").strip()
    if raw_ref.isdigit() or (len(raw_ref) == 1 and raw_ref.upper() in "ABCDEАБВГД"):
        want = letter_from_answer(raw_ref, 5).lower()
        raw = (content or "").strip()
        m = re.search(r"(?:відповідь|answer)\s*[:\-–]?\s*([A-Ea-eА-Дабвгд])\b", raw, re.I)
        if not m:
            m = re.search(r"(?m)^\s*([A-Ea-eА-Дабвгд])\s*[).]?\s*$", raw)
        if not m:
            m = re.search(r"\b([A-Ea-eА-Дабвгд])\b", raw)
        got = ""
        if m:
            ch = m.group(1).lower()
            got = cyr_to_latin.get(ch, ch)
            got = latin_to_cyr.get(got, got)
            got = cyr_to_latin.get(got, got)
        ok = got == want
        return {"score": 1.0 if ok else 0.0, "detail": f"mcq_letter got={got or '?'} ref={want}"}

    return {"score": 1.0 if ref in text else 0.0, "detail": "contains_ref"}


CODE_TESTS = {
    "code-001": [("add(2,3)", 5), ("add(-1,1)", 0)],
    "code-002": [("is_even(2)", True), ("is_even(3)", False)],
    "code-003": [("reverse_string('abc')", "cba")],
    "code-004": [("max_of_list([1,7,3])", 7)],
    "code-005": [("count_vowels('Hello')", 2)],
    "code-006": [("fibonacci(0)", 0), ("fibonacci(7)", 13)],
    "code-007": [("is_palindrome('Abba')", True), ("is_palindrome('abc')", False)],
    "code-008": [("flatten([[1,2],[3]])", [1, 2, 3])],
    "code-009": [("factorial(0)", 1), ("factorial(5)", 120)],
    "code-010": [("gcd(12,18)", 6), ("gcd(7,13)", 1)],
    "code-011": [("unique([1,2,1,3])", [1, 2, 3])],
    "code-012": [("word_count('a b c')", 3), ("word_count('')", 0)],
    "code-013": [("merge_dicts({'a':1},{'a':2,'b':3})", {"a": 2, "b": 3})],
    "code-014": [("is_anagram('Listen','Silent')", True), ("is_anagram('ab','cd')", False)],
    "code-015": [("clamp(5,0,10)", 5), ("clamp(-1,0,10)", 0), ("clamp(99,0,10)", 10)],
    "code-016": [("chunk_list([1,2,3,4,5],2)", [[1, 2], [3, 4], [5]])],
    "code-017": [("running_sum([1,2,3])", [1, 3, 6])],
    "code-018": [("find_missing([0,1,3])", 2), ("find_missing([0,2])", 1)],
    "code-019": [("rotate_left([1,2,3,4],1)", [2, 3, 4, 1]), ("rotate_left([1,2,3],3)", [1, 2, 3])],
    "code-020": [("deep_get({'a':{'b':2}},['a','b'])", 2), ("deep_get({'a':1},['x'],0)", 0)],
    "code-021": [("most_common([1,2,2,3])", 2)],
    "code-022": [("matrix_sum([[1,2],[3,4]])", 10)],
    "code-023": [("title_case('hello WORLD')", "Hello World")],
    "code-024": [("remove_duplicates_sorted([1,1,2,2,3])", [1, 2, 3])],
    "code-025": [("binary_search([1,3,5,7],5)", 2), ("binary_search([1,3,5],2)", -1)],
    "code-026": [("zip_to_dict(['a','b'],[1,2])", {"a": 1, "b": 2})],
    "code-027": [("average([2,4])", 3.0), ("average([])", 0.0)],
    "code-028": [("is_prime(7)", True), ("is_prime(1)", False), ("is_prime(9)", False)],
    "code-029": [("sorted(group_by_len(['a','bb','c'])[1])", ["a", "c"])],
    "code-030": [("safe_div(6,3)", 2.0), ("safe_div(1,0)", None)],
    "code-031": [("last([1,2,3])", 3), ("last([])", None)],
    "code-032": [("square(4)", 16), ("square(-3)", 9)],
}


def score_humaneval(content: str, row: dict[str, Any]) -> dict[str, Any]:
    """Run HumanEval check(candidate) with a short timeout."""
    import concurrent.futures

    row = humaneval_fields(row)
    code = extract_code(content or "")
    prompt = str(row.get("he_prompt") or "")
    test = str(row.get("he_test") or "")
    entry = str(row.get("he_entry_point") or "")
    if not test or not entry:
        return {"score": 0.0, "detail": "humaneval_missing_test"}

    def _run() -> None:
        ns: dict[str, Any] = {}
        try:
            exec(code, ns, ns)
        except Exception:
            ns = {}
            exec(prompt + "\n" + code, ns, ns)
        if entry not in ns:
            exec(prompt + "\n" + code, ns, ns)
        exec(test, ns, ns)
        ns["check"](ns[entry])

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run)
            fut.result(timeout=8.0)
        return {"score": 1.0, "detail": "humaneval_pass"}
    except concurrent.futures.TimeoutError:
        return {"score": 0.0, "detail": "humaneval_timeout"}
    except Exception as e:
        return {"score": 0.0, "detail": f"humaneval_fail:{type(e).__name__}"}


def score_code(content: str, item_id: str, row: dict[str, Any] | None = None) -> dict[str, Any]:
    if row and (
        str(item_id).startswith("humaneval-")
        or "score_mode=humaneval" in str(row.get("notes") or "")
    ):
        return score_humaneval(content, row)
    code = extract_code(content or "")
    tests = CODE_TESTS.get(item_id, [])
    if not tests:
        return {"score": 0.0, "detail": "no_tests"}
    ns: dict[str, Any] = {}
    try:
        exec(code, ns, ns)
    except Exception as e:
        return {"score": 0.0, "detail": f"exec_error:{type(e).__name__}"}
    passed = 0
    for expr, expected in tests:
        try:
            got = eval(expr, ns, ns)
            if got == expected:
                passed += 1
        except Exception:
            pass
    score = passed / len(tests)
    return {"score": score, "detail": f"passed={passed}/{len(tests)}"}


def score_alignment(content: str, reference: str) -> dict[str, Any]:
    """Exact class match for UAlign, accepting a bare or explained label."""
    expected = str(reference).strip()
    match = re.search(r"(?<!\d)([0-2])(?!\d)", content or "")
    predicted = match.group(1) if match else ""
    return {
        "score": 1.0 if predicted == expected else 0.0,
        "detail": f"ualign_label got={predicted or '?'} ref={expected}",
    }


_UA_REF_FIXES: dict[str, str] | None = None


def overlay_ua_translate_fixes(row: dict[str, Any]) -> dict[str, Any]:
    """Prefer human-validated Ukrainian gold for OPUS rows labeled uk."""
    global _UA_REF_FIXES
    if _UA_REF_FIXES is None:
        path = ROOT / "benchmarks/wmt_enuk_ua_fixes.json"
        _UA_REF_FIXES = {}
        if path.is_file():
            blob = json.loads(path.read_text(encoding="utf-8"))
            _UA_REF_FIXES = dict(blob.get("approved") or {})
    iid = str(row.get("id") or "")
    if iid in _UA_REF_FIXES:
        merged = dict(row)
        merged["reference"] = _UA_REF_FIXES[iid]
        return merged
    return row


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    row = overlay_ua_translate_fixes(row)
    bucket = row.get("bucket") or ""
    content = row.get("content") or ""
    reference = row.get("reference") or ""
    item_id = row.get("id") or ""
    notes = row.get("notes") or ""

    if row.get("http_status") != 200 or not content:
        return {"score": 0.0, "detail": "http_or_empty"}

    if bucket == "chat":
        return score_chat(content)
    if bucket == "translate":
        return score_translate(content, reference)
    if bucket == "instruct":
        return score_instruct(content, reference, item_id, notes, row)
    if bucket == "knowledge":
        return score_knowledge(content, reference)
    if bucket == "code":
        return score_code(content, item_id, row)
    if bucket == "alignment":
        return score_alignment(content, reference)
    return {"score": 0.0, "detail": "unknown_bucket"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/scores_summary.json"))
    args = p.parse_args()

    rows = load_rows(args.files)
    # Prefer latest full runs: keep all, but aggregate by system+id last-wins
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (str(r.get("system")), str(r.get("id")))
        latest[key] = r
    rows = list(latest.values())

    scored: list[dict[str, Any]] = []
    for r in rows:
        s = score_row(r)
        scored.append(
            {
                "system": r.get("system"),
                "id": r.get("id"),
                "bucket": r.get("bucket"),
                "latency_ms": r.get("latency_ms"),
                "router": r.get("router"),
                **s,
            }
        )

    by_sys_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    lat_sys_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_sys: dict[str, list[float]] = defaultdict(list)
    lat_sys: dict[str, list[float]] = defaultdict(list)

    def _pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        xs = sorted(vals)
        i = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[i]

    for s in scored:
        sys = str(s["system"])
        bucket = str(s["bucket"])
        by_sys_bucket[sys][bucket].append(float(s["score"]))
        by_sys[sys].append(float(s["score"]))
        if s.get("latency_ms") is not None:
            lat = float(s["latency_ms"])
            lat_sys[sys].append(lat)
            lat_sys_bucket[sys][bucket].append(lat)

    summary: dict[str, Any] = {"systems": {}}
    print(
        f"{'system':12} {'bucket':10} {'n':>3} {'qual':>6} "
        f"{'avg_ms':>8} {'p50_ms':>8} {'p95_ms':>8}"
    )
    for sys in sorted(by_sys_bucket):
        bucket_means: dict[str, Any] = {}
        for bucket, vals in sorted(by_sys_bucket[sys].items()):
            mean = sum(vals) / len(vals)
            lats = lat_sys_bucket[sys][bucket]
            avg_b = sum(lats) / len(lats) if lats else 0.0
            bucket_means[bucket] = {
                "quality_mean": round(mean, 4),
                "latency_avg_ms": round(avg_b, 1),
                "latency_p50_ms": round(_pct(lats, 50), 1),
                "latency_p95_ms": round(_pct(lats, 95), 1),
                "n": len(vals),
            }
            print(
                f"{sys:12} {bucket:10} {len(vals):3d} {mean:6.3f} "
                f"{avg_b:8.1f} {_pct(lats, 50):8.1f} {_pct(lats, 95):8.1f}"
            )
        overall = sum(by_sys[sys]) / len(by_sys[sys]) if by_sys[sys] else 0.0
        # Equal-bucket macro: each skill contributes equally regardless of n.
        bucket_quality_means = [v["quality_mean"] for v in bucket_means.values()]
        overall_macro = (
            sum(bucket_quality_means) / len(bucket_quality_means) if bucket_quality_means else 0.0
        )
        lats = lat_sys[sys]
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        print(
            f"{sys:12} {'ALL':10} {len(by_sys[sys]):3d} {overall:6.3f} "
            f"{avg_lat:8.1f} {_pct(lats, 50):8.1f} {_pct(lats, 95):8.1f}"
        )
        print(
            f"{sys:12} {'MACRO':10} {len(bucket_means):3d} {overall_macro:6.3f} "
            f"{'—':>8} {'—':>8} {'—':>8}"
        )
        summary["systems"][sys] = {
            "n": len(by_sys[sys]),
            "overall_mean": round(overall, 4),  # micro / sample-count weighted
            "overall_mean_macro": round(overall_macro, 4),  # equal-bucket
            "latency_avg_ms": round(avg_lat, 1),
            "latency_p50_ms": round(_pct(lats, 50), 1),
            "latency_p95_ms": round(_pct(lats, 95), 1),
            "gpu_seconds_total": round(sum(lats) / 1000.0, 3) if lats else 0.0,
            "gpu_seconds_per_prompt": round((sum(lats) / 1000.0) / max(1, len(lats)), 4) if lats else 0.0,
            "by_bucket": bucket_means,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.out.with_name(args.out.stem + "_detail.jsonl")
    with detail_path.open("w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "detail": str(detail_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
