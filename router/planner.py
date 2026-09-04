"""Mamay-4B JSON planner for bounded composite workflows."""

from __future__ import annotations

import json
import re
from typing import Any

from router.intent_rules import route_intent
from router.orchestrator import Plan, PlanStep, validate_plan

PLANNER_MODEL = "mamay4"

SYSTEM_INSTRUCTION = """Ти планувальник складених запитів для українського асистента.
Розбий запит на 2-4 залежні кроки. Дозволені intent:
translate, knowledge, instruct, code, alignment, chat.

Точні значення intent:
- translate: ТІЛЬКИ переклад між мовами;
- knowledge: відповідь на факт/тест або витяг фактів із наданого джерела;
- instruct: пояснення, написання чи переформатування у JSON/Markdown;
- code: написання, виправлення або перевірка коду;
- alignment: ТІЛЬКИ моральна/соціальна оцінка 0/1/2;
- chat: звичайна розмова.
Слово «літера» у тесті НЕ означає translate. JSON НЕ означає alignment.

Поверни ЛИШЕ JSON:
{"steps":[{"id":"step1","intent":"translate","depends_on":[],"prompt":"..."},
{"id":"step2","intent":"code","depends_on":["step1"],
"prompt":"... {{step1.content}} ... Поверни лише блок ```python без пояснень."}]}

Правила:
- кожен крок має одну чітку навичку;
- залежність використовуй у prompt як {{step_id.content}};
- фінальний крок повинен дослівно повторити всі вимоги користувача до формату
  фінальної відповіді (наприклад: лише JSON, рівно два рядки, лише блок коду,
  без пояснень);
- не вигадуй факти, яких немає у запиті або результатах попередніх кроків;
- не додавай ключів поза steps/id/intent/depends_on/prompt.

Приклади структури:
1) Англійська специфікація → реалізація:
translate → code.
2) Тестове питання → відповідь із поясненням у двох рядках:
knowledge → instruct.
3) Англійське джерело → витяг фактів → фінальний JSON:
translate → knowledge → instruct.
"""


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw, re.I)
    if fenced:
        raw = fenced.group(1)
    else:
        start, finish = raw.find("{"), raw.rfind("}")
        if start >= 0 and finish > start:
            raw = raw[start : finish + 1]
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("planner output must be a JSON object")
    return value


def fallback_plan(prompt: str) -> Plan:
    decision = route_intent(prompt)
    return Plan(
        steps=(
            PlanStep(
                id="fallback",
                intent=decision.intent,
                depends_on=(),
                prompt=prompt,
            ),
        ),
        source="single_hop_fallback",
    )


def enforce_final_constraint(plan: Plan, user_prompt: str) -> Plan:
    """Rules gate: preserve explicit final-format constraints the planner drops."""
    first_block = (user_prompt or "").split("\n\n", 1)[0]
    match = re.search(r"(?i)(фінальн\w*[\s\S]*|фінально[\s\S]*)", first_block)
    if not match or not plan.steps:
        return plan
    constraint = match.group(1).strip()
    final = plan.steps[-1]
    if constraint.casefold() in final.prompt.casefold():
        return plan
    guarded = PlanStep(
        id=final.id,
        intent=final.intent,
        depends_on=final.depends_on,
        prompt=(
            f"{final.prompt}\n\nОБОВ'ЯЗКОВА ВИМОГА ДО ФІНАЛЬНОЇ ВІДПОВІДІ: "
            f"{constraint}"
        ),
    )
    return Plan(steps=(*plan.steps[:-1], guarded), source=plan.source)


async def generate_plan(
    user_prompt: str,
    caller,
    *,
    repair: bool = True,
) -> tuple[Plan, dict[str, Any]]:
    planner_prompt = f"{SYSTEM_INSTRUCTION}\n\nЗАПИТ КОРИСТУВАЧА:\n{user_prompt}"
    attempts: list[dict[str, Any]] = []
    hop = await caller(PLANNER_MODEL, planner_prompt)
    attempts.append(hop)

    try:
        plan = Plan.from_dict(extract_json_object(hop.get("content") or ""), source="llm")
        plan = enforce_final_constraint(plan, user_prompt)
        validate_plan(plan)
        return plan, {"valid": True, "repaired": False, "attempts": attempts}
    except (ValueError, json.JSONDecodeError) as exc:
        first_error = str(exc)

    if repair:
        repair_prompt = (
            f"{SYSTEM_INSTRUCTION}\n\nПопередня відповідь невалідна: {first_error}\n"
            "Виправ її. Поверни тільки валідний JSON.\n\n"
            f"ЗАПИТ:\n{user_prompt}\n\nПОПЕРЕДНЯ ВІДПОВІДЬ:\n{hop.get('content') or ''}"
        )
        repaired = await caller(PLANNER_MODEL, repair_prompt)
        attempts.append(repaired)
        try:
            plan = Plan.from_dict(
                extract_json_object(repaired.get("content") or ""),
                source="llm_repaired",
            )
            plan = enforce_final_constraint(plan, user_prompt)
            validate_plan(plan)
            return plan, {"valid": True, "repaired": True, "attempts": attempts}
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    else:
        last_error = first_error

    return fallback_plan(user_prompt), {
        "valid": False,
        "repaired": repair and len(attempts) == 2,
        "error": last_error,
        "attempts": attempts,
        "fallback": True,
    }
