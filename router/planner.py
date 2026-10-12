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
    "hybrid_gated",     # decide "one task or several" first; one task = one call
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

SYSTEM_INSTRUCTION_GATED = """Ти диспетчер запитів для українського асистента.

Крок 1. Виріши, чи запит — ОДНА задача.
Одна задача — це: питання, порада чи список ідей, один переклад, один текст,
один фрагмент коду (навіть із поясненням), одна текстова задача з числами.
Таких запитів більшість. Це також ОДНА задача:
- тестове питання з варіантами, де треба дати лише літеру чи цифру
  (не розбивай на «відповісти» і «оформити»);
- один текст із вимогами до форми: кількість слів чи речень, регістр, лапки,
  обов'язкові чи заборонені слова, мова відповіді (не розбивай на «написати»
  і «відформатувати»).
Якщо задача одна, поверни рівно: {"steps":[]}

Крок 2. Лише якщо користувач ЯВНО просить кілька різних дій, де результат
однієї потрібен для наступної, поверни 2-3 кроки:
{"steps":[{"id":"step1","intent":"...","depends_on":[],"prompt":"..."},
{"id":"step2","intent":"...","depends_on":["step1"],"prompt":"... {{step1.content}}"}]}

Дозволені intent (лише ці шість слів, інших не вигадуй):
- translate: ТІЛЬКИ переклад між мовами;
- knowledge: відповідь на факт чи тест, витяг або класифікація фактів із наданого тексту;
- instruct: пояснення, написання, резюме чи переформатування (JSON, Markdown,
  задана кількість рядків чи речень);
- code: написання, виправлення або перевірка коду;
- alignment: ТІЛЬКИ моральна чи соціальна оцінка цифрою 0/1/2;
- chat: звичайна розмова.
Останній крок, що формує відповідь у заданому форматі, — instruct (або code,
якщо результат — код). JSON НЕ означає alignment.
Правила для кроків:
- не додавай дій, яких користувач не просив: жодного перекладу, JSON, коду чи
  вибору «найкращого», якщо цього немає в запиті;
- не перекладай текст, який уже українською;
- у prompt кожного кроку збережи всі деталі й обмеження із запиту;
- результат попереднього кроку встав як {{step_id.content}}.
Поверни ЛИШЕ JSON.
"""

# Appended to a single-call numeric word problem. Without it the specialists
# compute at once and skip the commonsense condition (AgentCoMa dev, 12 Oct 2026).
THINK_SUFFIX = (
    "\n\nСпершу випиши кожен предмет чи число з умови і для кожного коротко виріши, "
    "чи він справді підходить під умову задачі (подумай, як це буває в реальному житті). "
    "Потім порахуй. В останньому рядку напиши: Відповідь: <число>"
)

# A word problem asks "how many / how much", or names a quantity ("яка площа", "якою
# буде температура"). A bare "який ... 2020 ...?" is not one: an earlier, wider rule
# fired on "Який фільм 2020 року порадиш?".
_QUANTITY = (
    r"сум|площ|вартіст|кількіст|цін|температур|відстан|швидкіст|тривал|популяці|"
    r"мас|ваг|об['’ʼ]?єм|довжин|висот|ширин|відсот|частк|вік|час|решт|залиш|бюджет|витрат|прибут"
)
_WORD_PROBLEM_ASK = re.compile(
    r"(?i)\b(скільки|обчисли|порахуй|how many|how much)\b"
    r"|\bяк(а|ий|е|і|у|ою|им|ої|ого)\b[^?.!]{0,60}?\b(" + _QUANTITY + r")\w*"
)
_NUMBER = re.compile(
    r"(?i)\d+(?:[.,]\d+)?|\b(один|одна|одну|два|дві|три|чотири|п['’ʼ]?ять|шість|сім|вісім|"
    r"дев['’ʼ]?ять|десять|двадцять|тридцять|сорок|сто|тисяч\w*|половин\w*|пів|десятиліт\w*|удвічі|втричі)\b"
)
# Multiple-choice and fixed-format prompts must keep their own answer format.
_FIXED_FORMAT = re.compile(r"(?i)(лише|тільки|only|рівно)\s+(літер|цифр|число|json|код|\d)|варіант(у|ів|и)\b|\(A, B")


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


# "Answer with the letter only" / "only a digit": the whole answer is one symbol, so one task.
_SINGLE_SYMBOL = re.compile(
    r"(?i)(відповід\w*|answer)\s*(—|-|:)?\s*(дай\s+)?(лише|тільки|only)\s+(однією\s+)?(літер|цифр|буквою|a letter|a digit)"
)


def is_word_problem(prompt: str) -> bool:
    """A question that asks for a quantity and states at least two numbers."""
    text = prompt or ""
    if _FIXED_FORMAT.search(text) or not text.rstrip().endswith("?"):
        return False
    return bool(_WORD_PROBLEM_ASK.search(text)) and len(_NUMBER.findall(text)) >= 2


def single_call_plan(prompt: str) -> Plan:
    """One call on the user's own words; word problems get the reasoning suffix."""
    decision = route_intent(prompt)
    text = prompt + THINK_SUFFIX if is_word_problem(prompt) else prompt
    return Plan(
        steps=(PlanStep(id="single", intent=decision.intent, depends_on=(), prompt=text),),
        source="single_call_gate",
    )


# Mamay-4B often names a step "classification" or "instruction". The plan is fine; only
# the label is outside the six allowed intents, and validation used to reject the whole plan.
_INTENT_ALIASES = {
    "translate": ("translat", "переклад"),
    "code": ("code", "coding", "program", "код"),
    "alignment": ("alignment", "moral", "ethic", "мораль", "етич"),
    "knowledge": ("knowledge", "extract", "classif", "fact", "retriev", "analy", "answer", "qa",
                  "знан", "витяг", "класиф", "аналіз"),
    "instruct": ("instruct", "format", "summar", "writ", "explain", "rewrite", "generat", "compos",
                 "інструк", "формат", "резюм", "поясн", "напис"),
    "chat": ("chat", "convers", "розмов"),
}


def normalize_intents(raw: dict[str, Any]) -> dict[str, Any]:
    """Map near-miss intent labels onto the allowed six; unknown labels become "chat"."""
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


def with_user_request(plan: Plan, user_prompt: str) -> Plan:
    """Tell each step its place in the plan and make the last step answer the whole request.

    The executor returns only the last step's output. A request such as "write a function
    and explain it" therefore lost the function. The last step now sees every earlier
    result and the request and must give the complete answer; earlier steps get only their own
    action, so they do not do the rest of the request too.
    """
    plan = ensure_dependency_placeholders(plan)
    steps: list[PlanStep] = []
    last = len(plan.steps) - 1
    for index, step in enumerate(plan.steps):
        if index < last:
            # No copy of the whole request here: with it, a specialist did the whole task
            # at step 1 and the next step repeated the work.
            prompt = f"{step.prompt}\n\nВиконай лише цю одну дію і поверни лише її результат, без пояснень."
            steps.append(PlanStep(id=step.id, intent=step.intent, depends_on=step.depends_on, prompt=prompt))
            continue
        earlier = tuple(s.id for s in plan.steps[:last])
        missing = [sid for sid in earlier if f"{{{{{sid}.content}}}}" not in step.prompt]
        results = "".join(f"\n\nРезультат кроку {sid}:\n{{{{{sid}.content}}}}" for sid in missing)
        prompt = (
            f"Запит користувача:\n{user_prompt}{results}\n\n"
            f"Твоя дія на останньому кроці:\n{step.prompt}\n\n"
            "Виконай цю дію і дай остаточну відповідь на запит користувача. Якщо користувач просив і "
            "результат попереднього кроку (наприклад сам переклад чи сам код), додай його до відповіді; "
            "якщо попередній крок був лише підготовчим, не повторюй його. Нічого зайвого не додавай."
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
    if profile in {"hybrid", "hybrid_minimal", "hybrid_fewshot", "hybrid_gated"}:
        return "llm", PLANNER_MODEL
    raise ValueError(f"unknown planner profile: {profile}")


def instruction_for_profile(profile: str) -> str:
    if profile == "hybrid_minimal":
        return SYSTEM_INSTRUCTION_V3_MINIMAL
    if profile == "hybrid_fewshot":
        return SYSTEM_INSTRUCTION_FEWSHOT
    if profile in {"hybrid", "hybrid_mamay12"}:
        return SYSTEM_INSTRUCTION
    if profile == "hybrid_gated":
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

    if profile == "hybrid_gated" and _SINGLE_SYMBOL.search(user_prompt or ""):
        return single_call_plan(user_prompt), {
            "mode": "rule", "profile": profile, "template_id": None, "valid": True,
            "attempts": [], "repaired": False, "fallback": False, "gate": "single_symbol_answer",
        }
    if profile == "hybrid_gated" and is_word_problem(user_prompt):
        # Mamay-4B splits word problems into steps; a rule is more reliable here and saves the call.
        return single_call_plan(user_prompt), {
            "mode": "rule", "profile": profile, "template_id": None, "valid": True,
            "attempts": [], "repaired": False, "fallback": False, "gate": "single_word_problem",
        }

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

    gated = profile == "hybrid_gated"
    if gated:
        try:
            raw_steps = extract_json_object(hop.get("content") or "").get("steps")
        except (ValueError, json.JSONDecodeError):
            raw_steps = None
        # No steps, or one step, means "one task": answer the user's own words in one call.
        if isinstance(raw_steps, list) and len(raw_steps) <= 1:
            return single_call_plan(user_prompt), _meta(True, repaired=False, fallback=False, gate="single")

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
        return single_call_plan(user_prompt), _meta(
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
