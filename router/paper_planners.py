"""Two planners recreated from papers, with our six intents in place of tools and our routers in place of
the papers' model choice.

HuggingGPT (Shen et al., 2023, arXiv 2303.17580), four stages:
  1. task planning: the LLM returns [{"task", "id", "dep", "args": {"text"}}]; "<resource>-k" in args stands
     for the output of task k, "dep": [-1] means no dependency; hand-written demonstrations in the prompt;
  2. model selection: our router (table, MLP, ...) on each task; the paper lets the LLM pick from model
     descriptions;
  3. task execution: tasks run as soon as their dependencies are done (independent tasks in parallel);
  4. response generation: the planner LLM gets the request and all task results and writes the answer.
     The paper also asks for a description of the workflow; here it must write only the answer, because
     the answer is scored.

LLMCompiler (Kim et al., 2023, arXiv 2312.04511):
  1. planner: numbered actions "N. intent("text with $k")", the last one join(), then <END_OF_PLAN>; the
     prompt asks for as much parallelism as possible;
  2. task fetching unit: an action is sent as soon as the actions it references are done; "$k" is replaced by
     the output of action k;
  3. executor: each action goes to the model our router picks;
  4. joiner: the planner LLM sees the request and all results and either finishes with the answer or asks
     to replan once (the paper allows several rounds; here at most one replan).

Both planners use the same demonstrations: graph-dev items chosen by shape only (two per class, seed 42),
listed in DEMO_IDS; those items are left out when these planners are evaluated.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from router.intent_rules import ROUTER_MODELS, route_intent
from router.orchestrator import ALLOWED_INTENTS, Plan, PlanStep, render_prompt, validate_plan
from router.planner import PLANNER_MODEL, extract_json_object

AsyncCaller = Callable[[str, str], Awaitable[dict[str, Any]]]

PAPER_PLANNERS = ("hgpt", "compiler", "hgpt_mamay27", "compiler_mamay27", "planner_v2", "planner_v2_mamay27")
# Model that plans (and writes the final answer: HuggingGPT response generation, LLMCompiler joiner,
# planner v2 final call).
PAPER_PLANNER_MODELS: dict[str, str | None] = {"hgpt": None, "compiler": None,
                                               "hgpt_mamay27": "mamay27", "compiler_mamay27": "mamay27",
                                               "planner_v2": None, "planner_v2_mamay27": "mamay27"}
MAX_STEPS = 6  # neither paper caps the plan at 4; 6 keeps runaway plans out

# Picked with random.Random(42).sample over the sorted ids of each class of graphs_mixed_100_dev, 8 Nov 2026.
DEMO_IDS = ("cg100-s-01", "cg100-s-02", "cg100-c-09", "cg100-c-24",
            "cg100-g-05", "cg100-g-08", "cg100-p-04", "cg100-p-24")

INTENT_LIST = """- translate: переклад між мовами;
- knowledge: відповідь на факт чи тест, витяг або класифікація фактів із наданого тексту;
- instruct: написання, пояснення, резюме чи переформатування тексту (JSON, кількість рядків чи речень);
- code: написання або виправлення коду;
- alignment: моральна чи соціальна оцінка вчинку цифрою 0/1/2;
- math: задача, де треба порахувати (гроші, відсотки, час, дати, одиниці виміру);
- chat: звичайна розмова, порада, привітання."""


# --------------------------------------------------------------------------------------------- demos

def _demo_steps(item: dict[str, Any]) -> list[dict[str, Any]]:
    return list(item["oracle_plan"]["steps"])


def hgpt_demo(item: dict[str, Any]) -> str:
    steps = _demo_steps(item)
    index = {s["id"]: i for i, s in enumerate(steps)}
    out = []
    for i, s in enumerate(steps):
        text = s["prompt"]
        for dep, j in index.items():
            text = text.replace("{{" + dep + ".content}}", f"<resource>-{j}")
        out.append({"task": s["intent"], "id": i,
                    "dep": [index[d] for d in s.get("depends_on") or []] or [-1], "args": {"text": text}})
    if len(out) == 1:
        out = []  # one task: answered by one call
    return f"Запит: {item['prompt']}\nЗадачі: {json.dumps(out, ensure_ascii=False)}"


def compiler_demo(item: dict[str, Any]) -> str:
    steps = _demo_steps(item)
    index = {s["id"]: i + 1 for i, s in enumerate(steps)}
    lines = []
    for i, s in enumerate(steps, 1):
        text = s["prompt"]
        for dep, j in index.items():
            text = text.replace("{{" + dep + ".content}}", f"${j}")
        lines.append(f"{i}. {s['intent']}({json.dumps(text, ensure_ascii=False)})")
    lines.append(f"{len(steps) + 1}. join()")
    return f"Запит: {item['prompt']}\nПлан:\n" + "\n".join(lines) + "\n<END_OF_PLAN>"


# ---------------------------------------------------------------------------------------- prompts

HGPT_PLANNING = """Ти — асистент, який розбирає запит користувача на задачі. Формат:
[{{"task": задача, "id": номер, "dep": номери_задач_від_яких_залежить, "args": {{"text": текст}}}}]
Поле "dep" — номери попередніх задач, результат яких потрібен цій задачі; [-1], якщо таких немає.
Спеціальний тег "<resource>-номер" у "args" означає результат задачі з цим номером.
Задача МУСИТЬ бути однією з: translate, knowledge, instruct, code, alignment, math, chat.
{intents}
У "text" кожної задачі дай усе, що потрібно для її виконання: дані із запиту та вимоги до форми.
Подумай крок за кроком, які задачі потрібні, щоб виконати запит. Виділи якомога менше задач, але так,
щоб запит був виконаний. Зважай на залежності й порядок задач. Якщо запит — одна задача, поверни [].
Поверни лише JSON.

Приклади:
{demos}

Запит: {request}
Задачі:"""

HGPT_RESPONSE = """Запит користувача:
{request}

Результати виконаних задач:
{results}

Використай ці результати і дай відповідь на запит користувача. Дотримуйся всіх вимог запиту до форми
відповіді. Пиши лише саму відповідь, без опису процесу."""

COMPILER_PLANNING = """Для запиту користувача склади план, у якому якомога більше дій можна виконати паралельно.
Кожна дія — одна з таких типів:
{intents}
Кожна дія записується як intent("текст завдання").
- join(): збирає результати попередніх дій. Після нього модель або дає фінальну відповідь, або
просить новий план. join завжди остання дія плану.

Правила:
- Кожна дія має унікальний номер, номери зростають.
- Вхід дії — або текст, або результати попередніх дій; результат дії з номером k позначай $k.
- Завжди закінчуй план дією join() і після неї напиши <END_OF_PLAN>.
- План має давати якомога більше паралельності.
- Використовуй лише ці типи дій, не вигадуй нових.
- Якщо запит — одна задача, план має одну дію і join().

Приклади:
{demos}

Запит: {request}
План:"""

COMPILER_REPLAN = """Попередній план не дав відповіді.
Попередній план і результати:
{results}
Причина: {reason}
Склади новий план за тими самими правилами.

Запит: {request}
План:"""

COMPILER_JOIN = """Запит користувача:
{request}

Результати дій:
{results}

Виріши, чи цих результатів досить для відповіді.
Якщо досить, напиши в першому рядку: ДІЯ: ЗАВЕРШИТИ
а далі з нового рядка: ВІДПОВІДЬ: <відповідь на запит з дотриманням усіх вимог до форми, без опису процесу>
Якщо не досить, напиши в першому рядку: ДІЯ: ПЕРЕПЛАНУВАТИ
а далі: ПРИЧИНА: <чого бракує>"""

COMPILER_JOIN_FINAL = """Запит користувача:
{request}

Результати дій:
{results}

Дай відповідь на запит з дотриманням усіх вимог до форми, без опису процесу. Почни з рядка
ВІДПОВІДЬ:"""


# Planner v2 (approved 9 Nov 2026): the HuggingGPT task format without demos, the math intent, rules against
# the problems found on graph-dev (chains for independent parts, extra steps, answers written into step
# prompts, lost request data, lost parts of the answer). The final answer is written by a separate call.
PLANNER_V2 = """Ти — планувальник. Розбий запит користувача на задачі для моделей-спеціалістів.

Типи задач (лише ці сім):
- translate: переклад між мовами;
- knowledge: факт, відповідь на тест, пошук або класифікація в наданому тексті;
- math: треба порахувати (гроші, відсотки, час, дати, одиниці виміру);
- instruct: написати або переформатувати текст (лист, оголошення, JSON, задана кількість рядків чи речень);
- code: написати або виправити код;
- alignment: оцінка вчинку цифрою 0/1/2;
- chat: звичайна розмова, порада, привітання.

Формат — лише JSON-список:
[{{"task": тип, "id": номер, "dep": [...], "args": {{"text": текст задачі}}}}]
- "id" — номер задачі, від 0.
- Якщо задачі потрібен результат іншої задачі, напиши в її "text" мітку "<resource>-k" у тому місці,
  куди треба підставити цей результат (k — номер тієї задачі), і додай k у "dep".
- Якщо задачі не потрібні результати інших задач, її "dep": [-1].
- Перед виконанням кожну мітку "<resource>-k" буде замінено на результат задачі k.

Правила:
1. Якщо запит потребує лише одного типу задачі, поверни [] — його виконає одна модель.
2. Виділи якомога менше задач: нова задача — лише коли частина запиту потребує іншого типу.
3. Задачі, що не залежать одна від одної, виконуються паралельно, тому не пов'язуй їх без потреби.
4. У "text" кожної задачі дай усі дані із запиту, які їй потрібні (числа, назви, цитати, варіанти відповіді), і вимоги до форми її результату.
5. Пиши в "text" лише завдання. Не розв'язуй його і не підставляй відповідь.
6. Не додавай дій, яких користувач не просив; не перекладай текст, що вже українською.
7. Не додавай задачу, що збирає фінальну відповідь: її складе окремий крок із результатів усіх задач.

Запит: {request}
Задачі:"""

PLANNER_V2_FIX = """{planning}

Попередня відповідь невалідна: {error}
Виправ її за тими самими правилами. Поверни лише JSON-список.

ПОПЕРЕДНЯ ВІДПОВІДЬ:
{previous}"""


def build_hgpt_prompt(request: str, demos: list[dict[str, Any]]) -> str:
    return HGPT_PLANNING.format(intents=INTENT_LIST, demos="\n\n".join(hgpt_demo(d) for d in demos),
                                request=request)


def build_compiler_prompt(request: str, demos: list[dict[str, Any]]) -> str:
    return COMPILER_PLANNING.format(intents=INTENT_LIST, demos="\n\n".join(compiler_demo(d) for d in demos),
                                    request=request)


# ---------------------------------------------------------------------------------------- parsing

def parse_hgpt(text: str) -> Plan:
    """HuggingGPT task list -> Plan with {{tK.content}} placeholders. An empty list means one call."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON list in the planning output")
    tasks = json.loads(text[start:end + 1])
    if not isinstance(tasks, list):
        raise ValueError("planning output is not a list")
    if not tasks:
        return Plan(steps=(), source="hgpt")
    ids = [int(t["id"]) for t in tasks]
    steps = []
    for t in tasks:
        intent = str(t.get("task") or "").strip().lower()
        args = t.get("args") or {}
        prompt = str(args.get("text") if isinstance(args, dict) else args or "")
        deps = {f"t{int(d)}" for d in (t.get("dep") or []) if int(d) >= 0 and int(d) in ids}
        for j in sorted(ids, reverse=True):
            if f"<resource>-{j}" in prompt:
                prompt = prompt.replace(f"<resource>-{j}", "{{t%d.content}}" % j)
                deps.add(f"t{j}")
        deps = tuple(sorted(deps - {f"t{int(t['id'])}"}))
        steps.append(PlanStep(id=f"t{int(t['id'])}", intent=intent, prompt=prompt, depends_on=deps))
    return _ordered(Plan(steps=tuple(steps), source="hgpt"))


def parse_compiler(text: str) -> Plan:
    """LLMCompiler plan lines -> Plan with {{sK.content}} placeholders; stops at join() or <END_OF_PLAN>."""
    body = text.split("<END_OF_PLAN>", 1)[0]
    entries: list[tuple[int, str, str]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        head, dot, rest = line.partition(".")
        if not dot or not head.strip().isdigit():
            if entries and line:  # a long argument continued on the next line
                n, name, arg = entries[-1]
                entries[-1] = (n, name, arg + "\n" + line)
            continue
        rest = rest.strip()
        name, paren, arg = rest.partition("(")
        name = name.strip().lower()
        if not paren:
            continue
        if name == "join":
            break
        entries.append((int(head), name, arg))
    steps = []
    numbers = [n for n, _, _ in entries]
    for n, name, arg in entries:
        arg = arg.strip()
        if arg.endswith(")"):
            arg = arg[:-1].strip()
        if len(arg) >= 2 and arg[0] == arg[-1] == '"':
            try:
                arg = json.loads(arg)
            except json.JSONDecodeError:
                arg = arg[1:-1]
        deps = []
        for k in sorted(numbers, reverse=True):  # replace $12 before $1
            if f"${k}" in arg:
                arg = arg.replace(f"${k}", "{{s%d.content}}" % k)
                deps.append(f"s{k}")
        steps.append(PlanStep(id=f"s{n}", intent=name, prompt=arg, depends_on=tuple(sorted(deps))))
    return _ordered(Plan(steps=tuple(steps), source="compiler"))


def _ordered(plan: Plan) -> Plan:
    """Put steps in an order where every dependency comes first (both papers allow any order of ids)."""
    done: list[PlanStep] = []
    left = list(plan.steps)
    while left:
        ready = [s for s in left if all(d in {x.id for x in done} for d in s.depends_on)]
        if not ready:
            raise ValueError("plan has a dependency cycle or an unknown dependency")
        done.extend(ready)
        left = [s for s in left if s not in ready]
    return Plan(steps=tuple(done), source=plan.source)


# --------------------------------------------------------------------------------------- execution

async def execute_parallel(plan: Plan, caller: AsyncCaller, *, profile: str) -> dict[str, Any]:
    """Run every step as soon as its dependencies are done. Latency is the critical path of the step
    latencies (independent steps overlap), not their sum."""
    validate_plan(plan, max_steps=MAX_STEPS)
    for step in plan.steps:
        if step.intent not in ALLOWED_INTENTS:
            raise ValueError(f"invalid intent {step.intent}")
    results: dict[str, dict[str, Any]] = {}
    finish: dict[str, float] = {}
    pending = list(plan.steps)
    gpu = 0.0
    while pending:
        ready = [s for s in pending if all(d in results for d in s.depends_on)]

        async def run(step: PlanStep) -> dict[str, Any]:
            prompt = render_prompt(step, results)
            decision = route_intent(prompt, profile=profile, intent_hint=step.intent)
            if decision.model not in ROUTER_MODELS:
                raise ValueError(f"router selected unsupported model: {decision.model}")
            hop = await caller(decision.model, prompt)
            return {**hop, "id": step.id, "intent": step.intent, "depends_on": list(step.depends_on),
                    "prompt": prompt, "route": {"model": decision.model, "detected_intent": decision.intent,
                                                "declared_intent": step.intent, "reason": decision.reason}}

        outs = await asyncio.gather(*(run(s) for s in ready))
        for step, out in zip(ready, outs):
            results[step.id] = out
            start = max((finish[d] for d in step.depends_on), default=0.0)
            finish[step.id] = start + float(out.get("latency_ms") or 0)
            gpu += float(out.get("gpu_seconds") or 0)
        pending = [s for s in pending if s.id not in results]
    trace = [results[s.id] for s in plan.steps]
    return {"steps": trace, "complete": all(t.get("ok") for t in trace), "calls": len(trace),
            "latency_ms": max(finish.values(), default=0.0), "gpu_seconds": gpu}


def results_block(trace: list[dict[str, Any]], numbering: str) -> str:
    """One line per step: [number] intent: output (HuggingGPT and LLMCompiler both pass all results on)."""
    return "\n".join(f"[{t['id'][1:]}] {t['intent']}: {(t.get('content') or '').strip()}" for t in trace)


def _answer_after(text: str, marker: str) -> str:
    pos = text.find(marker)
    return text[pos + len(marker):].strip() if pos >= 0 else text.strip()


async def _single_call(request: str, caller: AsyncCaller, profile: str) -> dict[str, Any]:
    decision = route_intent(request, profile=profile)
    hop = await caller(decision.model, request)
    return {"steps": [{**hop, "id": "single", "intent": decision.intent, "depends_on": [], "prompt": request,
                       "route": {"model": decision.model, "detected_intent": decision.intent,
                                 "reason": decision.reason}}],
            "complete": bool(hop.get("ok")), "calls": 1, "latency_ms": float(hop.get("latency_ms") or 0),
            "gpu_seconds": float(hop.get("gpu_seconds") or 0), "final_output": hop.get("content") or ""}


# ----------------------------------------------------------------------------------------- systems

async def run_hgpt(request: str, caller: AsyncCaller, *, profile: str, demos: list[dict[str, Any]],
                   planner_model: str = PLANNER_MODEL) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    hop = await caller(planner_model, build_hgpt_prompt(request, demos))
    attempts.append(hop)
    meta: dict[str, Any] = {"mode": "hgpt", "profile": "hgpt", "attempts": attempts, "valid": True,
                            "fallback": False, "repaired": False}
    try:
        plan = parse_hgpt(hop.get("content") or "")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        plan, meta = None, {**meta, "valid": False, "fallback": True, "fallback_reason": "llm_plan_invalid",
                            "error": str(exc)}
    if plan is not None and not plan.steps:
        meta["gate"] = "single"
    if plan is not None and len(plan.steps) == 1:
        meta["gate"] = "single"
    if plan is None or len(plan.steps) <= 1:
        execution = await _single_call(request, caller, profile)
        return {"plan": plan, "planning": meta, "execution": execution}
    try:
        execution = await execute_parallel(plan, caller, profile=profile)
    except ValueError as exc:
        meta = {**meta, "valid": False, "fallback": True, "fallback_reason": "llm_plan_invalid", "error": str(exc)}
        execution = await _single_call(request, caller, profile)
        return {"plan": plan, "planning": meta, "execution": execution}
    answer = await caller(planner_model, HGPT_RESPONSE.format(
        request=request, results=results_block(execution["steps"], "hgpt")))
    execution["steps"].append({**answer, "id": "response", "intent": "response", "depends_on": [s.id for s in plan.steps],
                               "route": {"model": planner_model, "reason": "HuggingGPT response generation"}})
    execution["calls"] += 1
    execution["latency_ms"] += float(answer.get("latency_ms") or 0)
    execution["gpu_seconds"] += float(answer.get("gpu_seconds") or 0)
    execution["complete"] = execution["complete"] and bool(answer.get("ok"))
    execution["final_output"] = (answer.get("content") or "").strip()
    return {"plan": plan, "planning": meta, "execution": execution}


async def run_compiler(request: str, caller: AsyncCaller, *, profile: str, demos: list[dict[str, Any]],
                       planner_model: str = PLANNER_MODEL, max_replans: int = 1) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"mode": "compiler", "profile": "compiler", "attempts": attempts, "valid": True,
                            "fallback": False, "repaired": False, "replans": 0}
    planning_prompt = build_compiler_prompt(request, demos)
    all_steps: list[dict[str, Any]] = []
    latency = gpu = 0.0
    calls = 0
    plan: Plan | None = None
    for round_no in range(max_replans + 1):
        hop = await caller(planner_model, planning_prompt)
        attempts.append(hop)
        try:
            plan = parse_compiler(hop.get("content") or "")
            if len(plan.steps) <= 1:
                meta["gate"] = "single" if round_no == 0 else meta.get("gate")
                if round_no == 0:
                    execution = await _single_call(request, caller, profile)
                    return {"plan": plan, "planning": meta, "execution": execution}
            execution = await execute_parallel(plan, caller, profile=profile)
        except (ValueError, json.JSONDecodeError) as exc:
            meta = {**meta, "valid": False, "fallback": True, "fallback_reason": "llm_plan_invalid", "error": str(exc)}
            execution = await _single_call(request, caller, profile)
            execution["latency_ms"] += latency
            execution["calls"] += calls
            return {"plan": plan, "planning": meta, "execution": execution}
        all_steps += execution["steps"]
        latency += execution["latency_ms"]
        gpu += execution["gpu_seconds"]
        calls += execution["calls"]
        block = results_block(execution["steps"], "compiler")
        last_round = round_no == max_replans
        join_prompt = (COMPILER_JOIN_FINAL if last_round else COMPILER_JOIN).format(request=request, results=block)
        join = await caller(planner_model, join_prompt)
        all_steps.append({**join, "id": f"join{round_no}", "intent": "join", "depends_on": [],
                          "route": {"model": planner_model, "reason": "LLMCompiler joiner"}})
        latency += float(join.get("latency_ms") or 0)
        gpu += float(join.get("gpu_seconds") or 0)
        calls += 1
        text = join.get("content") or ""
        if last_round or "ПЕРЕПЛАНУВАТИ" not in text.split("\n", 1)[0].upper():
            return {"plan": plan, "planning": meta,
                    "execution": {"steps": all_steps, "complete": all(s.get("ok") for s in all_steps),
                                  "calls": calls, "latency_ms": latency, "gpu_seconds": gpu,
                                  "final_output": _answer_after(text, "ВІДПОВІДЬ:")}}
        meta["replans"] += 1
        planning_prompt = build_compiler_prompt(request, demos).rsplit("Запит:", 1)[0] + COMPILER_REPLAN.format(
            results=block, reason=_answer_after(text, "ПРИЧИНА:"), request=request)
    raise AssertionError("unreachable")


async def run_planner_v2(request: str, caller: AsyncCaller, *, profile: str,
                         planner_model: str = PLANNER_MODEL) -> dict[str, Any]:
    """Plan with PLANNER_V2, run the tasks in parallel, then the planner model writes the answer.

    An output that cannot be parsed or run (bad JSON, unknown intent, broken dependency) is sent back once
    with the error; if it still fails, the request is answered in one call. [] or one task = one call.
    """
    planning = PLANNER_V2.format(request=request)
    attempts: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"mode": "planner_v2", "profile": "planner_v2", "attempts": attempts, "valid": True,
                            "fallback": False, "repaired": False}
    hop = await caller(planner_model, planning)
    attempts.append(hop)
    plan: Plan | None = None
    execution: dict[str, Any] | None = None
    error = ""
    for attempt in range(2):
        try:
            plan = parse_hgpt(hop.get("content") or "")
            if len(plan.steps) <= 1:
                meta["gate"] = "single"
                execution = await _single_call(request, caller, profile)
                return {"plan": plan, "planning": meta, "execution": execution}
            execution = await execute_parallel(plan, caller, profile=profile)
            break
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            error = str(exc)
            if attempt == 0:
                hop = await caller(planner_model, PLANNER_V2_FIX.format(
                    planning=planning, error=error, previous=hop.get("content") or ""))
                attempts.append(hop)
                meta["repaired"] = True
    if execution is None:
        meta = {**meta, "valid": False, "fallback": True, "fallback_reason": "llm_plan_invalid", "error": error}
        execution = await _single_call(request, caller, profile)
        return {"plan": plan, "planning": meta, "execution": execution}
    answer = await caller(planner_model, HGPT_RESPONSE.format(
        request=request, results=results_block(execution["steps"], "hgpt")))
    execution["steps"].append({**answer, "id": "response", "intent": "response",
                               "depends_on": [s.id for s in plan.steps],
                               "route": {"model": planner_model, "reason": "planner v2 final answer"}})
    execution["calls"] += 1
    execution["latency_ms"] += float(answer.get("latency_ms") or 0)
    execution["gpu_seconds"] += float(answer.get("gpu_seconds") or 0)
    execution["complete"] = execution["complete"] and bool(answer.get("ok"))
    execution["final_output"] = (answer.get("content") or "").strip()
    return {"plan": plan, "planning": meta, "execution": execution}
