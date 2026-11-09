"""Mamay JSON planner for bounded composite workflows."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from router.intent_rules import route_intent
from router.orchestrator import Plan, PlanStep, ensure_dependency_placeholders, validate_plan
from router.workflows import select_plan as select_template_plan

PLANNER_MODEL = "mamay4"

# Named planner profiles used by run_composite_router / ablations.
PLANNER_PROFILES = (
    "hybrid",           # current free-form SYSTEM_INSTRUCTION (default)
    "hybrid_minimal",   # SYSTEM_INSTRUCTION_MINIMAL
    "hybrid_fewshot",   # SYSTEM_INSTRUCTION_FEWSHOT
    "template",         # constrained workflow selector (no LLM plan)
    "hybrid_mamay12",   # free-form planner on Mamay-12B
    "hybrid_gated",     # decide "one task or several" first; one task = one call
    # The same gate and prompt with another model writing the plan (dev comparison only).
    "hybrid_gated_lapa",
    "hybrid_gated_aya",
    "hybrid_gated_mamay12",
    "hybrid_gated_mamay27",
)

# Gated profiles and the model that writes the plan (None = PLANNER_MODEL).
GATED_PLANNERS: dict[str, str | None] = {
    "hybrid_gated": None,
    "hybrid_gated_lapa": "lapa",
    "hybrid_gated_aya": "aya",
    "hybrid_gated_mamay12": "mamay12",
    "hybrid_gated_mamay27": "mamay27",
}

SYSTEM_INSTRUCTION_MINIMAL = """Ти - планувальник, який розподіляє частини запиту від користувача між найпідходящими моделями. Кількість питань при розбитті вирішуєш ти. Кожен крок має один з intent:
translate, knowledge, instruct, code, alignment, math, chat.

Поверни ЛИШЕ JSON у форматі:
{"steps":[{"id":"step1","intent":"...","depends_on":[],"prompt":"..."}, ...]}

Залежність між кроками познач у prompt як {{step_id.content}}.
Не вигадуй фактів, яких немає в запиті чи в результатах попередніх кроків.
Останній крок мусить сформулювати фінальну відповідь, яку побачить користувач. Будь ласка, зважай на вказівки стосовно формату відповіді.
"""

SYSTEM_INSTRUCTION_GATED = """Ти - планувальник, який розподіляє частини запиту від користувача між найпідходящими моделями. Кількість кроків при розбитті вирішуєш ти (не більше 4). Кожен крок має один з intent:
translate, knowledge, instruct, code, alignment, math, chat.

Виріши, чи запит — ОДИН крок.
Один крок — це один виклик однієї навички (одного intent нижче):
- вимога до форми відповіді (JSON, регістр, кількість рядків чи слів) — це частина кроку, а не окремий крок;
- один intent — один крок (текстова задача з кількома обчисленнями — один крок, intent - math);
- новий крок потрібен лише тоді, коли частина запиту потребує іншого intent спеціаліста;

Якщо запит — один крок, поверни рівно: {"steps":[]}
Інакше поверни кроки у форматі:
{"steps":[{"id":"step1","intent":"...","depends_on":[],"prompt":"..."},
{"id":"step2","intent":"...","depends_on":["step1"],"prompt":"... {{step1.content}}"}]}

Дозволені intent (лише ці сім слів, інших не вигадуй):
- translate: ТІЛЬКИ переклад між мовами;
- knowledge: відповідь на факт чи тест, витяг або класифікація фактів із наданого тексту;
- instruct: пояснення, написання, резюме чи переформатування (JSON, Markdown,
  задана кількість рядків чи речень);
- code: написання, виправлення або перевірка коду;
- alignment: ТІЛЬКИ моральна чи соціальна оцінка цифрою 0/1/2;
- math: задача, де треба порахувати (гроші, відсотки, час, дати, одиниці виміру);
- chat: звичайна розмова, порада, привітання.
Останній крок, що формує відповідь у заданому форматі, — instruct (або code,
якщо результат — код).
Правила для кроків:
- не додавай дій, яких користувач не просив;
- не перекладай текст, який уже українською;
- у prompt кожного кроку збережи всі деталі й обмеження із запиту;
- результат попереднього кроку встав як {{step_id.content}}.
Поверни ЛИШЕ JSON.
"""

SYSTEM_INSTRUCTION = """Ти - планувальник, який розподіляє частини запиту від користувача між найпідходящими моделями. Кількість кроків при розбитті вирішуєш ти (від 1 до 4). Кожен крок має один з intent:
translate, knowledge, instruct, code, alignment, math, chat.

Дозволені intent (лише ці сім слів, інших не вигадуй):
- translate: ТІЛЬКИ переклад між мовами;
- knowledge: відповідь на факт чи тест, витяг або класифікація фактів із наданого тексту;
- instruct: пояснення, написання, резюме чи переформатування (JSON, Markdown,
  задана кількість рядків чи речень);
- code: написання, виправлення або перевірка коду;
- alignment: ТІЛЬКИ моральна чи соціальна оцінка цифрою 0/1/2;
- math: задача, де треба порахувати (гроші, відсотки, час, дати, одиниці виміру);
- chat: звичайна розмова, порада, привітання.

Поверни ЛИШЕ JSON у форматі:
{"steps":[{"id":"step1","intent":"...","depends_on":[],"prompt":"..."},
{"id":"step2","intent":"...","depends_on":["step1"],"prompt":"... {{step1.content}}"}]}

Правила для кроків:
- не додавай дій, яких користувач не просив;
- не перекладай текст, який уже українською;
- у prompt кожного кроку збережи всі деталі й обмеження із запиту;
- залежність між кроками познач у prompt як {{step_id.content}};
- не вигадуй фактів, яких немає в запиті чи в результатах попередніх кроків.
Останній крок мусить сформулювати фінальну відповідь, яку побачить користувач. Будь ласка, зважай на вказівки стосовно формату відповіді.
"""

SYSTEM_INSTRUCTION_FEWSHOT = """Ти - планувальник, який розподіляє частини запиту від користувача між найпідходящими моделями. Кількість кроків при розбитті вирішуєш ти (від 1 до 4). Кожен крок має один з intent:
translate, knowledge, instruct, code, alignment, math, chat.

Дозволені intent (лише ці сім слів, інших не вигадуй):
- translate: ТІЛЬКИ переклад між мовами;
- knowledge: відповідь на факт чи тест, витяг або класифікація фактів із наданого тексту;
- instruct: пояснення, написання, резюме чи переформатування (JSON, Markdown,
  задана кількість рядків чи речень);
- code: написання, виправлення або перевірка коду;
- alignment: ТІЛЬКИ моральна чи соціальна оцінка цифрою 0/1/2;
- math: задача, де треба порахувати (гроші, відсотки, час, дати, одиниці виміру);
- chat: звичайна розмова, порада, привітання.

Поверни ЛИШЕ JSON у форматі:
{"steps":[{"id":"step1","intent":"...","depends_on":[],"prompt":"..."},
{"id":"step2","intent":"...","depends_on":["step1"],"prompt":"... {{step1.content}}"}]}

Правила для кроків:
- не додавай дій, яких користувач не просив;
- не перекладай текст, який уже українською;
- у prompt кожного кроку збережи всі деталі й обмеження із запиту;
- залежність між кроками познач у prompt як {{step_id.content}};
- не вигадуй фактів, яких немає в запиті чи в результатах попередніх кроків.
Останній крок мусить сформулювати фінальну відповідь, яку побачить користувач. Будь ласка, зважай на вказівки стосовно формату відповіді.

Приклади структури (не копіюй текст запиту):
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


def single_call_plan(prompt: str, router_profile: str = "v2") -> Plan:
    """One call on the user's own words, used when the planner LLM returns one task or no valid plan.

    The intent comes from the selected router table, so the single call goes where the
    plain router would send the same request.
    """
    decision = route_intent(prompt, profile=router_profile)
    return Plan(
        steps=(PlanStep(id="single", intent=decision.intent, depends_on=(), prompt=prompt),),
        source="single_call",
    )


# Mamay-4B often names a step "classification" or "instruction". The plan is fine; only
# the label is outside the allowed intents, and validation used to reject the whole plan.
_INTENT_ALIASES = {
    "translate": ("translat", "переклад"),
    "code": ("code", "coding", "program", "код"),
    "alignment": ("alignment", "moral", "ethic", "мораль", "етич"),
    "knowledge": ("knowledge", "extract", "classif", "fact", "retriev", "analy", "answer", "qa",
                  "знан", "витяг", "класиф", "аналіз"),
    "instruct": ("instruct", "format", "summar", "writ", "explain", "rewrite", "generat", "compos",
                 "інструк", "формат", "резюм", "поясн", "напис"),
    "math": ("math", "calc", "arithm", "comput", "матем", "обчисл", "розрах", "порах"),
    "chat": ("chat", "convers", "розмов"),
}


def normalize_intents(raw: dict[str, Any]) -> dict[str, Any]:
    """Map near-miss intent labels onto the allowed seven; unknown labels become "chat"."""
    steps = raw.get("steps")
    if not isinstance(steps, list):
        return raw
    fixed = []
    for step in steps:
        if isinstance(step, dict):
            label = str(step.get("intent") or "").strip().lower()
            if label not in _INTENT_ALIASES:
                label = next((name for name, stems in _INTENT_ALIASES.items()
                              if any(label.startswith(stem) for stem in stems)), "chat")
            step = {**step, "intent": label}
        fixed.append(step)
    return {**raw, "steps": fixed}


_OPTION_LINE = re.compile(r"^\s*[А-ДA-E][\s.)]\s*\S", re.M)


def request_data(user_prompt: str) -> str | None:
    """The data the request works on: the text after the first blank line, else the answer options.

    Requests in the composite suites put the instruction first and the data (source text,
    notice, test question with options, specification) after a blank line.
    """
    text = (user_prompt or "").strip()
    if "\n\n" in text:
        data = text.split("\n\n", 1)[1].strip()
        return data or None
    options = [line for line in text.splitlines() if _OPTION_LINE.match(line)]
    return "\n".join(options) if len(options) >= 2 else None


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


_LINE_LABEL = re.compile(r"^\s*[\w ]{1,20}:\s+(?=\S)")


def has_request_data(prompt: str, data: str) -> bool:
    """Every line of the data is in the prompt, ignoring a leading label such as «Source:».

    Line by line, so options the planner wrote in one line («А 45, Б 90, В 180») still count.
    """
    text = _squash(prompt)
    lines = [_squash(_LINE_LABEL.sub("", line)) for line in data.splitlines()]
    return all(line in text for line in lines if line)


def with_request_data(prompt: str, data: str | None) -> str:
    """Append the request's data to a first step that lost it.

    The planner sometimes writes «Витягни з оголошення subject і number.» without the
    notice, or keeps a test question without its options; the specialist then answers
    something else (probe-100, 26 Oct). PLANNER_REQUEST_DATA=0 turns this off, for the
    before/after comparison on composite v2 dev.
    """
    if not data or has_request_data(prompt, data) or os.environ.get("PLANNER_REQUEST_DATA") == "0":
        return prompt
    return f"{prompt}\n\nДані із запиту:\n{data}"


def with_user_request(plan: Plan, user_prompt: str) -> Plan:
    """Tell each step its place in the plan and make the last step answer the whole request.

    The executor returns only the last step's output. A request such as "write a function
    and explain it" therefore lost the function. The last step now sees every earlier
    result and the request and must give the complete answer; earlier steps get only their own
    action, so they do not do the rest of the request too.
    """
    plan = ensure_dependency_placeholders(plan)
    data = request_data(user_prompt)
    steps: list[PlanStep] = []
    last = len(plan.steps) - 1
    for index, step in enumerate(plan.steps):
        if index < last:
            # No copy of the whole request here: with it, a specialist did the whole task
            # at step 1 and the next step repeated the work. A step with no inputs gets the
            # request's data, though, since it has nothing else to work on.
            prompt = step.prompt if step.depends_on else with_request_data(step.prompt, data)
            prompt = f"{prompt}\n\nВиконай лише цю одну дію і поверни лише її результат, без пояснень."
            steps.append(PlanStep(id=step.id, intent=step.intent, depends_on=step.depends_on, prompt=prompt))
            continue
        earlier = tuple(s.id for s in plan.steps[:last])
        # The last step gets the request and every earlier result with a label, and a fixed
        # instruction. The planner's own text for this step is not used: it sometimes held the
        # planner's answer (a wrong test option), which the model then preferred to the
        # specialist's result, and it dropped parts of the request.
        results = "\n".join(
            f"Крок {n} ({s.intent}): {{{{{s.id}.content}}}}" for n, s in enumerate(plan.steps[:last], start=1)
        )
        prompt = (
            f"Запит користувача:\n{user_prompt}\n\n"
            f"Результати попередніх кроків (їх дали спеціалісти; вважай їх правильними і не змінюй):\n{results}\n\n"
            "Напиши остаточну відповідь на запит користувача, спираючись на ці результати. "
            "Якщо результат кроку — це відповідь на питання чи тест (літера, число, назва), "
            "твоя відповідь має збігатися з нею; не обирай інший варіант. "
            "Відповідь має містити все, що просив користувач: якщо він просив переклад, код чи "
            "відповідь на питання і ще щось до них, наведи і те, і те. Дотримайся всіх вимог "
            "запиту до формату. Не додавай того, чого користувач не просив."
        )
        steps.append(PlanStep(id=step.id, intent=step.intent, depends_on=earlier, prompt=prompt))
    return Plan(steps=tuple(steps), source=plan.source)


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
    if profile in GATED_PLANNERS:
        return "llm", GATED_PLANNERS[profile] or PLANNER_MODEL
    if profile in {"hybrid", "hybrid_minimal", "hybrid_fewshot"}:
        return "llm", PLANNER_MODEL
    raise ValueError(f"unknown planner profile: {profile}")


def instruction_for_profile(profile: str) -> str:
    if profile == "hybrid_minimal":
        return SYSTEM_INSTRUCTION_MINIMAL
    if profile == "hybrid_fewshot":
        return SYSTEM_INSTRUCTION_FEWSHOT
    if profile in {"hybrid", "hybrid_mamay12"}:
        return SYSTEM_INSTRUCTION
    if profile in GATED_PLANNERS:
        return SYSTEM_INSTRUCTION_GATED
    raise ValueError(f"profile {profile} has no LLM instruction")


async def generate_plan(
    user_prompt: str,
    caller,
    *,
    repair: bool = True,
    instruction: str | None = None,
    model: str | None = None,
    profile: str | None = None,
    router_profile: str = "v2",
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

    # The planner LLM decides whether the request is one task or several; no rule decides
    # before it (rule gates removed 2 Nov 2026, last used at commit 961f476).
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

    gated = profile in GATED_PLANNERS
    if gated:
        try:
            raw_steps = extract_json_object(hop.get("content") or "").get("steps")
        except (ValueError, json.JSONDecodeError):
            raw_steps = None
        # No steps, or one step, means "one task": answer the user's own words in one call.
        if isinstance(raw_steps, list) and len(raw_steps) <= 1:
            return single_call_plan(user_prompt, router_profile), _meta(True, repaired=False, fallback=False, gate="single")

    try:
        parsed = extract_json_object(hop.get("content") or "")
        plan = Plan.from_dict(normalize_intents(parsed) if gated else parsed, source="llm")
        if gated:
            plan = with_user_request(plan, user_prompt)
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
            parsed = extract_json_object(repaired.get("content") or "")
            plan = Plan.from_dict(normalize_intents(parsed) if gated else parsed, source="llm_repaired")
            if gated:
                plan = with_user_request(plan, user_prompt)
            plan = enforce_final_constraint(plan, user_prompt)
            plan = ensure_dependency_placeholders(plan)
            validate_plan(plan)
            return plan, _meta(True, repaired=True, fallback=False)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    else:
        last_error = first_error

    if gated:
        return single_call_plan(user_prompt, router_profile), _meta(
            False, repaired=repair and len(attempts) == 2, error=last_error,
            fallback=True, fallback_reason="llm_plan_invalid", gate="single",
        )
    return fallback_plan(user_prompt), _meta(
        False,
        repaired=repair and len(attempts) == 2,
        error=last_error,
        fallback=True,
        fallback_reason="llm_plan_invalid",
    )
