"""HuggingGPT- and LLMCompiler-style planners (router/paper_planners.py) with a fake backend."""

import asyncio
import json

from router.paper_planners import (
    compiler_demo,
    hgpt_demo,
    parse_compiler,
    parse_hgpt,
    run_compiler,
    run_hgpt,
)

DEMO = {
    "id": "cg100-c-x",
    "prompt": "Переклади і перекажи.",
    "oracle_plan": {"steps": [
        {"id": "n1", "intent": "translate", "depends_on": [], "prompt": "Переклади: hello"},
        {"id": "n2", "intent": "instruct", "depends_on": ["n1"], "prompt": "Перекажи: {{n1.content}}"},
    ]},
}


def fake_caller(script):
    calls = []

    async def caller(alias, prompt):
        calls.append((alias, prompt))
        for key, reply in script:
            if key in prompt:
                return {"ok": True, "content": reply, "latency_ms": 100.0, "gpu_seconds": 0.1}
        return {"ok": True, "content": f"out-{len(calls)}", "latency_ms": 100.0, "gpu_seconds": 0.1}

    return caller, calls


def test_demos_use_the_paper_reference_syntax():
    assert "<resource>-0" in hgpt_demo(DEMO)
    assert '"dep": [-1]' in hgpt_demo(DEMO)
    text = compiler_demo(DEMO)
    assert "$1" in text and "join()" in text and text.endswith("<END_OF_PLAN>")


def test_parse_hgpt_maps_resources_to_dependencies():
    plan = parse_hgpt(json.dumps([
        {"task": "knowledge", "id": 0, "dep": [-1], "args": {"text": "Як звати пса?"}},
        {"task": "alignment", "id": 1, "dep": [-1], "args": {"text": "Оціни вчинок"}},
        {"task": "instruct", "id": 2, "dep": [0], "args": {"text": "Ім'я <resource>-0, оцінка <resource>-1"}},
    ]))
    last = plan.steps[-1]
    assert last.depends_on == ("t0", "t1")
    assert "{{t1.content}}" in last.prompt


def test_parse_hgpt_empty_list_is_one_call():
    assert parse_hgpt("[]").steps == ()


def test_parse_compiler_reads_actions_until_join():
    plan = parse_compiler('1. chat("Скільки буде 2+2?")\n2. instruct("Оголошення двома реченнями")\n'
                          '3. instruct("Збери: $1 і $2")\n4. join()\n<END_OF_PLAN>')
    assert [s.intent for s in plan.steps] == ["chat", "instruct", "instruct"]
    assert plan.steps[2].depends_on == ("s1", "s2")


def test_hgpt_runs_independent_tasks_in_parallel_and_answers_from_results():
    plan_json = json.dumps([
        {"task": "chat", "id": 0, "dep": [-1], "args": {"text": "Скільки буде 2+2?"}},
        {"task": "translate", "id": 1, "dep": [-1], "args": {"text": "Переклади: кіт"}},
    ])
    caller, calls = fake_caller([("розбирає запит", plan_json), ("Результати виконаних задач", "ВІДПОВІДЬ 4, cat")])
    out = asyncio.run(run_hgpt("запит", caller, profile="v2", demos=[DEMO]))
    ex = out["execution"]
    assert ex["final_output"] == "ВІДПОВІДЬ 4, cat"
    assert ex["calls"] == 3  # two tasks + response generation; the planning call is in planning.attempts
    assert ex["latency_ms"] == 200.0  # parallel tasks overlap: 100 + response 100
    assert len(out["planning"]["attempts"]) == 1


def test_compiler_finishes_through_the_joiner():
    plan_text = '1. chat("2+2?")\n2. instruct("Напиши: $1")\n3. join()\n<END_OF_PLAN>'
    caller, _ = fake_caller([("якомога більше дій", plan_text), ("Виріши, чи цих", "ДІЯ: ЗАВЕРШИТИ\nВІДПОВІДЬ: 4")])
    out = asyncio.run(run_compiler("запит", caller, profile="v2", demos=[DEMO]))
    assert out["execution"]["final_output"] == "4"
    assert out["planning"]["replans"] == 0


def test_compiler_replans_once_then_must_finish():
    plan_text = '1. chat("2+2?")\n2. instruct("Напиши: $1")\n3. join()\n<END_OF_PLAN>'
    caller, _ = fake_caller([("Попередній план не дав", plan_text), ("якомога більше дій", plan_text),
                             ("Виріши, чи цих", "ДІЯ: ПЕРЕПЛАНУВАТИ\nПРИЧИНА: мало"),
                             ("Почни з рядка", "ВІДПОВІДЬ: 4")])
    out = asyncio.run(run_compiler("запит", caller, profile="v2", demos=[DEMO]))
    assert out["planning"]["replans"] == 1
    assert out["execution"]["final_output"] == "4"
    assert len(out["planning"]["attempts"]) == 2


def test_invalid_plan_falls_back_to_one_call():
    caller, _ = fake_caller([("розбирає запит", "не JSON")])
    out = asyncio.run(run_hgpt("Переклади: кіт", caller, profile="v2", demos=[DEMO]))
    assert out["planning"]["fallback"] is True
    assert out["execution"]["calls"] == 1


def test_hgpt_uses_the_given_planner_model():
    plan_json = json.dumps([{"task": "chat", "id": 0, "dep": [-1], "args": {"text": "2+2?"}},
                            {"task": "translate", "id": 1, "dep": [-1], "args": {"text": "Переклади: кіт"}}])
    caller, calls = fake_caller([("розбирає запит", plan_json), ("Результати виконаних задач", "4, cat")])
    asyncio.run(run_hgpt("запит", caller, profile="v2", demos=[DEMO], planner_model="mamay27"))
    assert calls[0][0] == "mamay27" and calls[-1][0] == "mamay27"


def test_math_hint_routes_to_lapa_and_chat_stays_default():
    from router.intent_rules import route_intent
    assert route_intent("Скільки буде 2+2?", intent_hint="math").model == "lapa"
    assert route_intent("Привіт!", intent_hint="chat").model == "aya"


def test_planner_aliases_map_calculation_to_math():
    from router.planner import normalize_intents
    out = normalize_intents({"steps": [{"intent": "calculation"}, {"intent": "обчислення"}, {"intent": "розмова"}]})
    assert [s["intent"] for s in out["steps"]] == ["math", "math", "chat"]


def test_planner_v2_runs_tasks_then_writes_the_answer():
    from router.paper_planners import run_planner_v2
    plan_json = json.dumps([{"task": "math", "id": 0, "dep": [-1], "args": {"text": "2+2?"}},
                            {"task": "translate", "id": 1, "dep": [-1], "args": {"text": "Переклади: кіт"}}])
    caller, calls = fake_caller([("Розбий запит", plan_json), ("Результати виконаних задач", "4, cat")])
    out = asyncio.run(run_planner_v2("запит", caller, profile="v2"))
    assert out["execution"]["final_output"] == "4, cat"
    assert out["execution"]["calls"] == 3 and len(out["planning"]["attempts"]) == 1
    assert [c[0] for c in calls[1:3]] == ["lapa", "aya"]  # math -> Lapa, translate -> Aya


def test_planner_v2_fixes_a_broken_plan_once():
    from router.paper_planners import run_planner_v2
    good = json.dumps([{"task": "math", "id": 0, "dep": [-1], "args": {"text": "2+2?"}},
                       {"task": "instruct", "id": 1, "dep": [0], "args": {"text": "Напиши: <resource>-0"}}])
    caller, _ = fake_caller([("Попередня відповідь невалідна", good), ("Розбий запит", "не JSON"),
                             ("Результати виконаних задач", "4")])
    out = asyncio.run(run_planner_v2("запит", caller, profile="v2"))
    assert out["planning"]["repaired"] is True and out["planning"]["fallback"] is False
    assert len(out["planning"]["attempts"]) == 2 and out["execution"]["final_output"] == "4"


def test_planner_v2_empty_list_and_double_failure_are_one_call():
    from router.paper_planners import run_planner_v2
    caller, _ = fake_caller([("Розбий запит", "[]")])
    assert asyncio.run(run_planner_v2("Переклади: кіт", caller, profile="v2"))["execution"]["calls"] == 1
    caller, _ = fake_caller([("Розбий запит", "не JSON")])
    out = asyncio.run(run_planner_v2("Переклади: кіт", caller, profile="v2"))
    assert out["planning"]["fallback"] is True and out["execution"]["calls"] == 1
