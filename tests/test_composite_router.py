from __future__ import annotations

import asyncio
import json
import unittest
from collections import Counter
from unittest.mock import patch

from fastapi.testclient import TestClient

from router.app import app
from router.orchestrator import (
    Plan,
    PlanStep,
    execute_plan,
    plan_signature,
    render_prompt,
    validate_plan,
)
from router.planner import enforce_final_constraint, generate_plan, SYSTEM_INSTRUCTION_V3_MINIMAL
from scripts.build_agentcoma_benchmark import wrap_row
from scripts.build_composite_benchmark import build_items, validate_items
from scripts.run_composite_router import balanced_limit
from scripts.score_composite_results import parse_json_content, score_row, score_rubric


class BenchmarkTests(unittest.TestCase):
    def test_suite_shape(self) -> None:
        items = build_items()
        validate_items(items)
        self.assertEqual(len(items), 36)
        self.assertEqual(
            {family: sum(x["family"] == family for x in items) for family in {x["family"] for x in items}},
            {
                "translate_code": 12,
                "knowledge_explain": 12,
                "translate_knowledge_write": 12,
            },
        )

    def test_forward_dependency_rejected(self) -> None:
        items = build_items()
        items[0]["oracle_plan"]["steps"][0]["depends_on"] = ["implement"]
        with self.assertRaisesRegex(ValueError, "dependency"):
            validate_items(items)

    def test_smoke_limit_is_family_balanced(self) -> None:
        selected = balanced_limit(build_items(), 6)
        self.assertEqual(Counter(x["family"] for x in selected), {
            "knowledge_explain": 2,
            "translate_code": 2,
            "translate_knowledge_write": 2,
        })

    def test_agentcoma_wrapper_hides_gold_from_prompt(self) -> None:
        wrapped = wrap_row(
            {
                "id": "eval_HW_add_1",
                "category": "house_working",
                "operation_type": "addition",
                "question_composition_uk": "Скільки предметів підуть у шафу?",
                "answer_composition": 5,
            }
        )
        self.assertEqual(wrapped["oracle_plan"]["steps"][0]["intent"], "knowledge")
        self.assertEqual(wrapped["oracle_plan"]["steps"][1]["intent"], "instruct")
        self.assertEqual(wrapped["prompt"], "Скільки предметів підуть у шафу?")
        self.assertNotIn("5", wrapped["prompt"])
        self.assertNotIn("5", wrapped["oracle_plan"]["steps"][0]["prompt"])


class PlanTests(unittest.TestCase):
    def test_cycle_or_forward_dependency_rejected(self) -> None:
        plan = Plan(
            steps=(
                PlanStep("first", "translate", "x {{second.content}}", ("second",)),
                PlanStep("second", "code", "y", ()),
            )
        )
        with self.assertRaisesRegex(ValueError, "dependencies"):
            validate_plan(plan)

    def test_substitution(self) -> None:
        step = PlanStep("write", "instruct", "Result: {{fact.content}}", ("fact",))
        self.assertEqual(render_prompt(step, {"fact": {"content": "42"}}), "Result: 42")

    def test_signature_ignores_generated_ids(self) -> None:
        left = Plan(
            (
                PlanStep("translate", "translate", "x"),
                PlanStep("code", "code", "{{translate.content}}", ("translate",)),
            )
        )
        right = Plan(
            (
                PlanStep("step1", "translate", "x"),
                PlanStep("step2", "code", "{{step1.content}}", ("step1",)),
            )
        )
        self.assertEqual(plan_signature(left), plan_signature(right))

    def test_malformed_plan_repairs(self) -> None:
        outputs = iter(
            [
                "not json",
                json.dumps(
                    {
                        "steps": [
                            {"id": "a", "intent": "knowledge", "depends_on": [], "prompt": "fact"},
                            {
                                "id": "b",
                                "intent": "instruct",
                                "depends_on": ["a"],
                                "prompt": "{{a.content}}",
                            },
                        ]
                    }
                ),
            ]
        )

        async def caller(alias: str, prompt: str) -> dict:
            return {"alias": alias, "ok": True, "content": next(outputs), "latency_ms": 1, "gpu_seconds": 0.001}

        plan, meta = asyncio.run(generate_plan("question", caller))
        self.assertTrue(meta["valid"])
        self.assertTrue(meta["repaired"])
        self.assertEqual(len(plan.steps), 2)

    def test_malformed_plan_falls_back(self) -> None:
        async def caller(alias: str, prompt: str) -> dict:
            return {"alias": alias, "ok": True, "content": "bad", "latency_ms": 1, "gpu_seconds": 0.001}

        plan, meta = asyncio.run(generate_plan("original", caller))
        self.assertFalse(meta["valid"])
        self.assertEqual(plan.source, "single_hop_fallback")
        self.assertEqual(plan.steps[0].prompt, "original")

    def test_instruction_and_model_overrides_are_forwarded(self) -> None:
        seen: list[tuple[str, str]] = []

        async def caller(alias: str, prompt: str) -> dict:
            seen.append((alias, prompt))
            return {
                "alias": alias,
                "ok": True,
                "content": json.dumps(
                    {
                        "steps": [
                            {
                                "id": "a",
                                "intent": "knowledge",
                                "depends_on": [],
                                "prompt": "fact",
                            }
                        ]
                    }
                ),
                "latency_ms": 1,
                "gpu_seconds": 0.001,
            }

        plan, meta = asyncio.run(
            generate_plan(
                "question",
                caller,
                instruction=SYSTEM_INSTRUCTION_V3_MINIMAL,
                model="mamay12",
            )
        )
        self.assertTrue(meta["valid"])
        self.assertEqual(plan.steps[0].intent, "knowledge")
        self.assertEqual(seen[0][0], "mamay12")
        self.assertTrue(seen[0][1].startswith(SYSTEM_INSTRUCTION_V3_MINIMAL))

    def test_final_constraint_is_enforced(self) -> None:
        plan = Plan(
            (
                PlanStep("translate", "translate", "translate"),
                PlanStep("code", "code", "implement", ("translate",)),
            ),
            source="llm",
        )
        guarded = enforce_final_constraint(
            plan,
            "Зроби два кроки. Фінальна відповідь — лише блок ```python без пояснень.\n\nSpec",
        )
        self.assertIn("лише блок ```python", guarded.steps[-1].prompt)


class ScoringAndExecutionTests(unittest.TestCase):
    def test_rubrics(self) -> None:
        score, _ = score_rubric("Сума парних чисел", {"type": "contains_all", "values": ["сума", "парн"]})
        self.assertEqual(score, 1.0)
        self.assertEqual(parse_json_content('```json\n{"number": 2}\n```'), {"number": 2})
        score, _ = score_rubric(
            '{"subject":"Дніпро","number":2201,"place":"Київ"}',
            {
                "type": "json_fields",
                "fields": {"subject": "Дніпро", "number": 2201, "place": "Київ"},
                "exact_keys": True,
            },
        )
        self.assertEqual(score, 1.0)
        score, _ = score_rubric(
            "```python\ndef add(a, b):\n    return a + b\n```",
            {
                "type": "python_function",
                "entry_point": "add",
                "cases": [{"args": [2, 3], "expected": 5}],
            },
        )
        self.assertEqual(score, 1.0)
        score, _ = score_rubric(
            "```python\ndef add(a, b):\n    return a + b\nprint(add(1, 2))\n```",
            {
                "type": "python_function",
                "entry_point": "add",
                "cases": [{"args": [2, 3], "expected": 5}],
            },
        )
        self.assertEqual(score, 1.0)

    def test_oracle_pipeline_end_to_end(self) -> None:
        item = build_items()[0]
        plan = Plan.from_dict(item["oracle_plan"], source="oracle")
        prompts: list[str] = []

        async def caller(alias: str, prompt: str) -> dict:
            prompts.append(prompt)
            content = (
                "Повернути суму всіх парних цілих чисел."
                if alias == "aya"
                else "```python\ndef sum_even(values):\n    return sum(x for x in values if x % 2 == 0)\n```"
            )
            return {
                "alias": alias,
                "ok": True,
                "content": content,
                "latency_ms": 10,
                "gpu_seconds": 0.01,
                "http_status": 200,
            }

        result = asyncio.run(execute_plan(plan, caller))
        self.assertTrue(result["complete"])
        self.assertEqual([x["alias"] for x in result["steps"]], ["aya", "mamay4"])
        self.assertIn("Повернути суму", prompts[1])
        row = {
            "system": "oracle",
            "id": item["id"],
            "family": item["family"],
            "oracle_plan": item["oracle_plan"],
            "final_step_expected": item["final_step"],
            "steps": result["steps"],
            "content": result["final_output"],
            "complete": True,
            "calls": 2,
            "latency_ms": 20,
            "gpu_seconds": 0.02,
            "plan_valid": True,
            "plan_match": True,
        }
        scored = score_row(row)
        self.assertTrue(scored["final_success"])
        self.assertTrue(scored["all_stages_pass"])

    def test_all_36_oracle_workflows_three_times(self) -> None:
        def perfect_content(rubric: dict) -> str:
            kind = rubric["type"]
            if kind == "contains_all":
                return " ".join(str(x) for x in rubric["values"])
            if kind == "label":
                return str(rubric["value"])
            if kind == "formatted_evidence":
                return (
                    f"Відповідь: {rubric['label']}\n"
                    f"Пояснення: {' '.join(str(x) for x in rubric['required'])}"
                )
            if kind == "json_fields":
                return json.dumps(rubric["fields"], ensure_ascii=False)
            if kind == "python_function":
                entry = rubric["entry_point"]
                return (
                    "```python\n"
                    f"CASES = {rubric['cases']!r}\n"
                    f"def {entry}(*args):\n"
                    "    for case in CASES:\n"
                    "        if list(args) == case['args']:\n"
                    "            return case['expected']\n"
                    "    raise ValueError('unknown fixture case')\n```"
                )
            raise AssertionError(kind)

        for _repeat in range(3):
            for item in build_items():
                oracle_steps = item["oracle_plan"]["steps"]
                contents = iter(perfect_content(step["rubric"]) for step in oracle_steps)

                async def caller(alias: str, prompt: str) -> dict:
                    return {
                        "alias": alias,
                        "ok": True,
                        "content": next(contents),
                        "latency_ms": 1,
                        "gpu_seconds": 0.001,
                        "http_status": 200,
                    }

                result = asyncio.run(
                    execute_plan(Plan.from_dict(item["oracle_plan"], source="oracle"), caller)
                )
                scored = score_row(
                    {
                        "system": "fixture",
                        "id": item["id"],
                        "family": item["family"],
                        "oracle_plan": item["oracle_plan"],
                        "final_step_expected": item["final_step"],
                        "steps": result["steps"],
                        "content": result["final_output"],
                        "complete": result["complete"],
                        "calls": result["calls"],
                        "latency_ms": result["latency_ms"],
                        "gpu_seconds": result["gpu_seconds"],
                        "plan_valid": True,
                        "plan_match": True,
                    }
                )
                self.assertTrue(scored["final_success"], item["id"])
                self.assertTrue(scored["all_stages_pass"], item["id"])

    def test_orchestrate_endpoint_with_fake_backends(self) -> None:
        plan_json = json.dumps(
            {
                "steps": [
                    {
                        "id": "translate",
                        "intent": "translate",
                        "depends_on": [],
                        "prompt": "Переклади specification",
                    },
                    {
                        "id": "implement",
                        "intent": "code",
                        "depends_on": ["translate"],
                        "prompt": "Напиши Python код: {{translate.content}}",
                    },
                ]
            }
        )
        calls = 0

        async def fake_chat(client, alias, prompt, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                content = plan_json
            elif alias == "aya":
                content = "Українська специфікація"
            else:
                content = "```python\ndef solution():\n    return 1\n```"
            return {
                "alias": alias,
                "ok": True,
                "content": content,
                "latency_ms": 1,
                "gpu_seconds": 0.001,
                "http_status": 200,
            }

        with patch("router.app.async_chat", side_effect=fake_chat):
            response = TestClient(app).post(
                "/v1/orchestrate",
                json={"prompt": "Translate a spec and implement it", "max_tokens": 64},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["complete"])
        self.assertEqual(body["calls"], 3)
        self.assertEqual([step["alias"] for step in body["steps"]], ["aya", "mamay4"])


if __name__ == "__main__":
    unittest.main()
