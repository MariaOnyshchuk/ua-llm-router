"""Mamay JSON planner for bounded composite workflows."""

from __future__ import annotations

import json
import re
from typing import Any

from router.intent_rules import route_intent
from router.orchestrator import Plan, PlanStep, ensure_dependency_placeholders, validate_plan
from router.workflows import select_plan as select_template_plan

PLANNER_MODEL = "mamay4"

# Named planner profiles used by run_composite_router / ablations.
PLANNER_PROFILES = (
    "hybrid",           # current free-form SYSTEM_INSTRUCTION (default)
    "hybrid_minimal",   # SYSTEM_INSTRUCTION_V3_MINIMAL
    "hybrid_fewshot",   # SYSTEM_INSTRUCTION_FEWSHOT
    "template",         # constrained workflow selector (no LLM plan)
    "hybrid_mamay12",   # free-form planner on Mamay-12B
)

SYSTEM_INSTRUCTION_V3_MINIMAL = """Ти плануєш виконання складеного запиту для українського асистента,
розбиваючи його на 2-4 залежні кроки. Кожен крок має один з intent:
translate, knowledge, instruct, code, alignment, chat — обери сам,
який підходить найкраще для того, що конкретно потрібно зробити на цьому кроці.

Поверни ЛИШЕ JSON у форматі:
{"steps":[{"id":"step1","intent":"...","depends_on":[],"prompt":"..."}, ...]}

Залежність між кроками познач у prompt як {{step_id.content}}.
Не вигадуй фактів, яких немає в запиті чи в результатах попередніх кроків.
Фінальний крок має дослівно відтворити всі вимоги користувача щодо формату
відповіді (наприклад: лише JSON, рівно два рядки, лише блок коду тощо).
"""

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
1) Тестове питання → відповідь із поясненням у двох рядках:
knowledge → instruct.
2) Англійське джерело → витяг фактів → фінальний JSON:
translate → knowledge → instruct.
"""

SYSTEM_INSTRUCTION_FEWSHOT = """Ти планувальник складених запитів для українського асистента.
Розбий запит на 2-4 залежні кроки. Дозволені intent:
translate, knowledge, instruct, code, alignment, chat.

Точні значення intent:
- translate: ТІЛЬКИ переклад між мовами;
- knowledge: відповідь на факт/тест або витяг/класифікація з наданого джерела;
- instruct: пояснення, резюме чи переформатування у JSON/Markdown;
- code: написання, виправлення або перевірка коду;
- alignment: ТІЛЬКИ моральна/соціальна оцінка 0/1/2;
- chat: звичайна розмова.
Слово «літера» у тесті НЕ означає translate. JSON НЕ означає alignment.
НЕ додавай зайвий крок translate, якщо весь запит уже українською.

Поверни ЛИШЕ JSON:
{"steps":[{"id":"step1","intent":"...","depends_on":[],"prompt":"..."}, ...]}

Правила:
- кожен крок має одну чітку навичку;
- залежність використовуй у prompt як {{step_id.content}};
- фінальний крок повинен дослівно повторити всі вимоги користувача до формату;
- не вигадуй факти; не додавай ключів поза steps/id/intent/depends_on/prompt.

Few-shot (структура, не копіюй текст запиту):
A) UA тест → два рядки Відповідь/Пояснення:
{"steps":[
 {"id":"answer","intent":"knowledge","depends_on":[],"prompt":"... питання + лише літера"},
 {"id":"explain","intent":"instruct","depends_on":["answer"],
  "prompt":"Рівно два рядки Відповідь/Пояснення. Попередня: {{answer.content}}"}]}
B) EN spec → UA → Python:
{"steps":[
 {"id":"translate","intent":"translate","depends_on":[],"prompt":"Переклади специфікацію..."},
 {"id":"implement","intent":"code","depends_on":["translate"],
  "prompt":"Реалізуй функцію. Лише ```python. {{translate.content}}"}]}
C) EN source → факти → JSON subject/number/place:
translate → knowledge → instruct
D) UA оголошення → extract → classify → JSON topic/urgency/subject/number:
knowledge → knowledge → instruct (без translate)
E) EN brief → translate → summarize → JSON topic/when/action:
translate → instruct → instruct
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


def resolve_profile(profile: str) -> tuple[str, str | None]:
    """Return (mode, planner_model_override). mode is template|llm."""
    if profile == "template":
        return "template", None
    if profile == "hybrid_mamay12":
        return "llm", "mamay12"
    if profile in {"hybrid", "hybrid_minimal", "hybrid_fewshot"}:
        return "llm", PLANNER_MODEL
    raise ValueError(f"unknown planner profile: {profile}")


def instruction_for_profile(profile: str) -> str:
    if profile == "hybrid_minimal":
        return SYSTEM_INSTRUCTION_V3_MINIMAL
    if profile == "hybrid_fewshot":
        return SYSTEM_INSTRUCTION_FEWSHOT
    if profile in {"hybrid", "hybrid_mamay12"}:
        return SYSTEM_INSTRUCTION
    raise ValueError(f"profile {profile} has no LLM instruction")


async def generate_plan(
    user_prompt: str,
    caller,
    *,
    repair: bool = True,
    instruction: str | None = None,
    model: str | None = None,
    profile: str | None = None,
) -> tuple[Plan, dict[str, Any]]:
    if profile == "template":
        plan, meta = select_template_plan(user_prompt)
        return plan, meta

    system = SYSTEM_INSTRUCTION if instruction is None else instruction
    if profile is not None and instruction is None:
        system = instruction_for_profile(profile)
    planner_model = PLANNER_MODEL if model is None else model
    if profile is not None and model is None:
        _, planner_model = resolve_profile(profile)
        planner_model = planner_model or PLANNER_MODEL

    planner_prompt = f"{system}\n\nЗАПИТ КОРИСТУВАЧА:\n{user_prompt}"
    attempts: list[dict[str, Any]] = []
    hop = await caller(planner_model, planner_prompt)
    attempts.append(hop)

    def _meta(valid: bool, **extra: Any) -> dict[str, Any]:
        return {
            "mode": "llm",
            "profile": profile or "hybrid",
            "template_id": None,
            "valid": valid,
            "attempts": attempts,
            **extra,
        }

    try:
        plan = Plan.from_dict(extract_json_object(hop.get("content") or ""), source="llm")
        plan = enforce_final_constraint(plan, user_prompt)
        plan = ensure_dependency_placeholders(plan)
        validate_plan(plan)
        return plan, _meta(True, repaired=False, fallback=False)
    except (ValueError, json.JSONDecodeError) as exc:
        first_error = str(exc)

    if repair:
        repair_prompt = (
            f"{system}\n\nПопередня відповідь невалідна: {first_error}\n"
            "Виправ її. Поверни тільки валідний JSON.\n\n"
            f"ЗАПИТ:\n{user_prompt}\n\nПОПЕРЕДНЯ ВІДПОВІДЬ:\n{hop.get('content') or ''}"
        )
        repaired = await caller(planner_model, repair_prompt)
        attempts.append(repaired)
        try:
            plan = Plan.from_dict(
                extract_json_object(repaired.get("content") or ""),
                source="llm_repaired",
            )
            plan = enforce_final_constraint(plan, user_prompt)
            plan = ensure_dependency_placeholders(plan)
            validate_plan(plan)
            return plan, _meta(True, repaired=True, fallback=False)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    else:
        last_error = first_error

    return fallback_plan(user_prompt), _meta(
        False,
        repaired=repair and len(attempts) == 2,
        error=last_error,
        fallback=True,
        fallback_reason="llm_plan_invalid",
    )
