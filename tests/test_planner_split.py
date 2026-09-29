import asyncio
import json

from scripts.analysis.planner_split import plan_one, run, sample_items, summarise, wilson


def plan(*steps):
    return json.dumps({"steps": [
        {"id": f"s{i}", "intent": intent, "depends_on": [f"s{i-1}"] if i else [], "prompt": "Зроби це. " + (f"{{{{s{i-1}.content}}}}" if i else "")}
        for i, intent in enumerate(steps)]})


def caller_for(text):
    async def caller(alias, prompt):
        return {"ok": True, "content": text, "latency_ms": 5.0, "alias": alias, "finish_reason": "stop"}
    return caller


def item(bucket, i=0):
    return {"id": f"{bucket}-{i}", "bucket": bucket, "prompt": "Запит."}


def test_single_step_matching_bucket_is_correct():
    r = asyncio.run(plan_one(item("knowledge"), caller_for(plan("knowledge")), "hybrid"))
    assert r["correct"] and not r["split"] and r["n_steps"] == 1


def test_three_steps_counts_as_split():
    r = asyncio.run(plan_one(item("alignment"), caller_for(plan("chat", "knowledge", "instruct")), "hybrid"))
    assert r["split"] and not r["correct"] and r["n_steps"] == 3


def test_garbage_falls_back_and_is_not_split():
    r = asyncio.run(plan_one(item("translate"), caller_for("не json"), "hybrid"))
    assert r["fallback"] and not r["split"] and not r["correct"] and r["planner_calls"] == 2


def test_wrong_intent_single_step_is_not_correct():
    r = asyncio.run(plan_one(item("instruct"), caller_for(plan("chat")), "hybrid"))
    assert not r["correct"] and not r["split"]


def test_summary_and_wilson():
    rows = asyncio.run(run([item("knowledge", i) for i in range(4)] + [item("instruct")],
                           caller_for(plan("knowledge", "instruct")), "hybrid", 2))
    s = summarise(rows)
    assert s["overall"]["split"]["k"] == 5
    assert s["by_bucket"]["translate"]["n"] == 0 and s["by_bucket"]["translate"]["split"]["rate"] is None
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60


def test_sampling_is_stratified_and_seeded():
    items = [item(b, i) for b in ("knowledge", "translate", "alignment", "instruct", "chat") for i in range(30)]
    a = sample_items(items, 10, 42)
    assert a == sample_items(items, 10, 42)
    assert len(a) == 40 and all(x["bucket"] != "chat" for x in a)
