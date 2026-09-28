#!/usr/bin/env python3
"""Deterministically score composite execution traces."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.I)
CYR_LABEL_RE = re.compile(r"(?<![А-ЯІЇЄҐ])([А-В])(?![А-ЯІЇЄҐ])", re.I)


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                row = json.loads(line)
                # Allow rerunning with a broad *.jsonl glob after a prior score:
                # detail rows are outputs, not raw composite traces.
                if not row.get("oracle_plan"):
                    continue
                row["_source"] = path.name
                rows.append(row)
    return rows


def parse_json_content(content: str) -> dict[str, Any] | None:
    raw = (content or "").strip()
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, re.I)
    if match:
        raw = match.group(1)
    else:
        start, finish = raw.find("{"), raw.rfind("}")
        if start >= 0 and finish > start:
            raw = raw[start : finish + 1]
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def score_python(content: str, rubric: dict[str, Any]) -> tuple[float, str]:
    match = CODE_FENCE_RE.search(content or "")
    code = match.group(1) if match else (content or "")
    only_code_ok = bool(
        match
        and re.fullmatch(
            r"\s*```(?:python)?\s*[\s\S]*?\s*```\s*",
            content or "",
            re.I,
        )
    )
    entry = str(rubric.get("entry_point") or "")
    cases = rubric.get("cases") or []
    harness = (
        "import json,sys\n"
        "ns={}\n"
        "code=sys.stdin.readline()\n"
        "exec(json.loads(code),ns,ns)\n"
        f"fn=ns[{entry!r}]\n"
        f"cases={cases!r}\n"
        "passed=0\n"
        "details=[]\n"
        "for case in cases:\n"
        "  try:\n"
        "    got=fn(*case['args'])\n"
        "    ok=got==case['expected'] or "
        "(isinstance(got,float) and isinstance(case['expected'],float) and abs(got-case['expected'])<1e-6)\n"
        "    passed+=int(ok); details.append({'ok':ok,'got':repr(got)})\n"
        "  except Exception as e: details.append({'ok':False,'error':type(e).__name__+': '+str(e)})\n"
        "print(json.dumps({'passed':passed,'n':len(cases),'details':details}))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", harness],
            input=json.dumps(code) + "\n",
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        # Generated code may contain top-level examples that print. The harness
        # emits its machine-readable result last, so ignore earlier stdout.
        output_lines = [line for line in proc.stdout.splitlines() if line.strip()]
        result = json.loads(output_lines[-1]) if output_lines else {}
        passed = int(result.get("passed") or 0)
        total = max(1, int(result.get("n") or len(cases)))
        test_score = passed / total
        if rubric.get("only_code"):
            return (test_score + float(only_code_ok)) / 2, (
                f"python {passed}/{total}; only_code={only_code_ok}"
            )
        return test_score, f"python {passed}/{total}"
    except (subprocess.TimeoutExpired, ValueError, TypeError) as exc:
        return 0.0, f"python error: {type(exc).__name__}"


def normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


NUMBER_RE = re.compile(r"-?\d+(?:[ \u00a0]\d{3})+(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?")
ANSWER_MARK_RE = re.compile(r"(?:відповідь|answer|разом|загалом|=)\s*[:\-–—]?\s*", re.I)


def parse_number(token: str) -> float | None:
    """Parse ``19 000``, ``10,000``, ``3.50``, ``2,2`` -> float."""
    raw = token.replace("\u00a0", " ").replace(" ", "")
    if "," in raw and "." not in raw:
        head, _, tail = raw.rpartition(",")
        # "10,000" / "1,234,567" are thousands groups; "2,2" / "12,5" are decimals.
        if len(tail) == 3 and re.fullmatch(r"-?\d{1,3}(?:,\d{3})*", head):
            raw = raw.replace(",", "")
        else:
            raw = head.replace(",", "") + "." + tail
    else:
        raw = raw.replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def extract_numbers(content: str) -> list[float]:
    values = [parse_number(m.group(0)) for m in NUMBER_RE.finditer(content or "")]
    return [v for v in values if v is not None]


def score_numeric(content: str, rubric: dict[str, Any]) -> tuple[float, str]:
    """Final-number match for AgentCoMa-style items.

    The answer is the number after the last answer marker (``Відповідь:``,
    ``=``, ...) when one exists, otherwise the last number in the text.
    """
    expected = float(rubric.get("value"))
    tol = float(rubric.get("tolerance", 1e-6))
    text = content or ""
    marks = list(ANSWER_MARK_RE.finditer(text))
    candidates: list[float] = []
    if marks:
        candidates = extract_numbers(text[marks[-1].end() :])
    if not candidates:
        candidates = extract_numbers(text)
    if not candidates:
        return 0.0, f"numeric got=? ref={expected:g}"
    got = candidates[0] if marks and candidates else candidates[-1]
    ok = math.isclose(got, expected, rel_tol=1e-9, abs_tol=tol)
    return float(ok), f"numeric got={got:g} ref={expected:g}"


def score_rubric(content: str, rubric: dict[str, Any]) -> tuple[float, str]:
    kind = rubric.get("type")
    if kind == "contains_all":
        values = [normalise(x) for x in rubric.get("values") or []]
        text = normalise(content)
        hits = sum(value in text for value in values)
        return hits / max(1, len(values)), f"contains {hits}/{len(values)}"
    if kind == "contains_any":
        values = [normalise(x) for x in rubric.get("values") or []]
        text = normalise(content)
        hit = any(value in text for value in values)
        return float(hit), f"contains_any {int(hit)}/1"
    if kind == "numeric":
        return score_numeric(content, rubric)
    if kind == "label":
        match = CYR_LABEL_RE.search(content or "")
        got = match.group(1).upper() if match else ""
        expected = str(rubric.get("value") or "").upper()
        return float(got == expected), f"label got={got or '?'} ref={expected}"
    if kind == "formatted_evidence":
        lines = [line.strip() for line in (content or "").strip().splitlines() if line.strip()]
        expected_lines = int(rubric.get("line_count") or 2)
        label = str(rubric.get("label") or "").upper()
        required = [normalise(x) for x in rubric.get("required") or []]
        text = normalise(content)
        checks = [
            len(lines) == expected_lines,
            bool(lines and re.search(rf"^відповідь:\s*{re.escape(label)}\b", lines[0], re.I)),
            bool(len(lines) > 1 and lines[1].casefold().startswith("пояснення:")),
            all(value in text for value in required),
        ]
        return sum(checks) / len(checks), f"format/evidence {sum(checks)}/{len(checks)}"
    if kind == "json_fields":
        value = parse_json_content(content)
        if value is None:
            return 0.0, "invalid json"
        fields = rubric.get("fields") or {}
        checks = [normalise(value.get(key, "")) == normalise(expected) for key, expected in fields.items()]
        if rubric.get("exact_keys"):
            checks.append(set(value) == set(fields))
        return sum(checks) / max(1, len(checks)), f"json {sum(checks)}/{len(checks)}"
    if kind == "python_function":
        return score_python(content, rubric)
    return 0.0, f"unknown rubric {kind}"


def intent_f1(actual: list[str], expected: list[str]) -> float:
    actual_counts, expected_counts = Counter(actual), Counter(expected)
    overlap = sum((actual_counts & expected_counts).values())
    if not actual and not expected:
        return 1.0
    precision = overlap / max(1, len(actual))
    recall = overlap / max(1, len(expected))
    return 2 * precision * recall / max(1e-12, precision + recall)


def workflow_error(row: dict[str, Any], plan_match: bool, expected_n: int, actual_n: int) -> str | None:
    """Classify planning error when the workflow is observable."""
    if row.get("plan") is None:
        return None
    planning = row.get("planning") or {}
    if planning.get("fallback"):
        return "fallback"
    if not row.get("plan_valid", True):
        return "invalid_plan"
    if plan_match:
        return None
    if actual_n > expected_n:
        return "superfluous_steps"
    if actual_n < expected_n:
        return "missing_steps"
    return "wrong_intents_or_deps"


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    oracle_steps = (row.get("oracle_plan") or {}).get("steps") or []
    trace = row.get("steps") or []
    plan_steps = (row.get("plan") or {}).get("steps") or []
    stage_scores: list[dict[str, Any]] = []
    for index, expected in enumerate(oracle_steps):
        if index >= len(trace):
            stage_scores.append(
                {"id": expected["id"], "intent": expected["intent"], "score": 0.0, "detail": "missing stage"}
            )
            continue
        score, detail = score_rubric(trace[index].get("content") or "", expected["rubric"])
        stage_scores.append(
            {
                "id": expected["id"],
                "intent": expected["intent"],
                "actual_intent": trace[index].get("intent"),
                "score": round(score, 4),
                "detail": detail,
            }
        )

    final_expected = next(
        step for step in oracle_steps if step["id"] == row.get("final_step_expected")
    )
    final_score, final_detail = score_rubric(row.get("content") or "", final_expected["rubric"])
    stage_mean = (
        sum(x["score"] for x in stage_scores) / len(stage_scores)
        if trace
        else final_score
    )
    all_stages = bool(trace) and len(trace) == len(oracle_steps) and all(x["score"] == 1.0 for x in stage_scores)
    expected_intents = [str(step["intent"]) for step in oracle_steps]
    actual_intents = [str(step.get("intent") or "") for step in trace]
    plan_match = bool(row.get("plan_match"))
    err = workflow_error(row, plan_match, len(oracle_steps), len(plan_steps))
    return {
        "system": row.get("system"),
        "id": row.get("id"),
        "family": row.get("family"),
        "split": row.get("split"),
        "selector_mode": row.get("selector_mode") or (row.get("planning") or {}).get("mode"),
        "selected_template": row.get("selected_template")
        or (row.get("planning") or {}).get("template_id"),
        "planner_profile": row.get("planner_profile") or row.get("system"),
        "repeat": row.get("repeat", 1),
        "final_score": round(final_score, 4),
        "final_success": final_score == 1.0,
        "final_detail": final_detail,
        "stage_mean": round(stage_mean, 4),
        "all_stages_pass": all_stages,
        "stage_scores": stage_scores,
        "plan_valid": row.get("plan_valid") if row.get("plan") else None,
        "plan_match": plan_match,
        "workflow_error": err,
        "superfluous_steps": int(row.get("superfluous_steps") or max(0, len(plan_steps) - len(oracle_steps))),
        "missing_steps": int(row.get("missing_steps") or max(0, len(oracle_steps) - len(plan_steps))),
        "intent_f1": round(intent_f1(actual_intents, expected_intents), 4) if trace else None,
        "complete": bool(row.get("complete")),
        "http_ok": int(row.get("http_status") or 0) == 200 or bool(row.get("complete")),
        "calls": int(row.get("calls") or 0),
        "latency_ms": float(row.get("latency_ms") or 0),
        "gpu_seconds": float(row.get("gpu_seconds") or 0),
        "source": row.get("_source"),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / (len(values) - 1))


def _slice_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    plan_rows = [row for row in rows if row["plan_valid"] is not None]
    intent_rows = [row for row in rows if row["intent_f1"] is not None]
    errors = Counter(row["workflow_error"] for row in plan_rows if row.get("workflow_error"))
    return {
        "n": len(rows),
        "final_score": round(mean([float(x["final_score"]) for x in rows]), 4),
        "final_success_rate": round(mean([float(x["final_success"]) for x in rows]), 4),
        "all_stages_pass_rate": round(mean([float(x["all_stages_pass"]) for x in rows]), 4),
        "plan_exact_match_rate": (
            round(mean([float(x["plan_match"]) for x in plan_rows]), 4) if plan_rows else None
        ),
        "intent_f1": (
            round(mean([float(x["intent_f1"]) for x in intent_rows]), 4) if intent_rows else None
        ),
        "fallback_rate": (
            round(mean([float(x["workflow_error"] == "fallback") for x in plan_rows]), 4)
            if plan_rows
            else None
        ),
        "workflow_errors": dict(errors),
        "http_ok_rate": round(mean([float(x["http_ok"]) for x in rows]), 4),
    }


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[str(row["system"])].append(row)
    systems: dict[str, Any] = {}
    for system, rows in sorted(grouped.items()):
        by_repeat: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_repeat[int(row["repeat"])].append(row)
        repeat_final = [mean([float(x["final_score"]) for x in group]) for group in by_repeat.values()]
        plan_rows = [row for row in rows if row["plan_valid"] is not None]
        intent_rows = [row for row in rows if row["intent_f1"] is not None]
        families: dict[str, Any] = {}
        for family in sorted({str(row["family"]) for row in rows}):
            families[family] = _slice_summary([row for row in rows if row["family"] == family])
        splits: dict[str, Any] = {}
        for split in sorted({str(row["split"]) for row in rows if row.get("split")}):
            splits[split] = _slice_summary([row for row in rows if row.get("split") == split])
        modes: dict[str, Any] = {}
        for mode in sorted({str(row["selector_mode"]) for row in rows if row.get("selector_mode")}):
            modes[mode] = _slice_summary([row for row in rows if row.get("selector_mode") == mode])
        errors = Counter(row["workflow_error"] for row in plan_rows if row.get("workflow_error"))
        systems[system] = {
            "n": len(rows),
            "repeats": len(by_repeat),
            "final_score_mean": round(mean(repeat_final), 4),
            "final_score_sd": round(sample_sd(repeat_final), 4),
            "final_success_rate": round(mean([float(x["final_success"]) for x in rows]), 4),
            "stage_mean": round(mean([float(x["stage_mean"]) for x in rows]), 4),
            "all_stages_pass_rate": round(mean([float(x["all_stages_pass"]) for x in rows]), 4),
            "plan_valid_rate": (
                round(mean([float(bool(x["plan_valid"])) for x in plan_rows]), 4)
                if plan_rows
                else None
            ),
            "plan_exact_match_rate": (
                round(mean([float(x["plan_match"]) for x in plan_rows]), 4)
                if plan_rows
                else None
            ),
            "intent_f1": (
                round(mean([float(x["intent_f1"]) for x in intent_rows]), 4)
                if intent_rows
                else None
            ),
            "fallback_rate": (
                round(mean([float(x["workflow_error"] == "fallback") for x in plan_rows]), 4)
                if plan_rows
                else None
            ),
            "workflow_errors": dict(errors),
            "http_ok_rate": round(mean([float(x["http_ok"]) for x in rows]), 4),
            "calls_per_item": round(mean([float(x["calls"]) for x in rows]), 3),
            "latency_p50_ms": round(sorted(float(x["latency_ms"]) for x in rows)[len(rows) // 2], 1),
            "latency_avg_ms": round(mean([float(x["latency_ms"]) for x in rows]), 1),
            "gpu_seconds_per_item": round(mean([float(x["gpu_seconds"]) for x in rows]), 4),
            "by_family": families,
            "by_split": splits,
            "by_selector_mode": modes,
        }
    return {"systems": systems}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/week_10_composite/scores.json"))
    args = parser.parse_args()
    scored = [score_row(row) for row in load_rows(args.files)]
    summary = summarize(scored)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    detail = args.out.with_name(f"{args.out.stem}_detail.jsonl")
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    detail.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored),
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(args.out), "detail": str(detail)}, ensure_ascii=False))
    for system, values in summary["systems"].items():
        print(
            f"{system:14} final={values['final_score_mean']:.3f} "
            f"all={values['all_stages_pass_rate']:.3f} "
            f"calls={values['calls_per_item']:.2f} p50={values['latency_p50_ms']:.0f}ms"
        )


if __name__ == "__main__":
    main()
