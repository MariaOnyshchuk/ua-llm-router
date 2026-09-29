"""Debug UI endpoints that need no model backend."""

from fastapi.testclient import TestClient

from router.app import app

client = TestClient(app)


def test_page_served():
    r = client.get("/debug")
    assert r.status_code == 200 and "Router debug" in r.text


def test_route_trace_shows_fired_rule_and_alternatives():
    r = client.post("/debug/api/route", json={"prompt": "Переклади з англійської: The river froze."}).json()
    assert r["intent"] == "translate" and r["model"] == "aya"
    assert any(x["matched"] for x in r["rules"]) and len(r["rules"]) > 3


def test_bench_item_hides_nothing_needed_and_lists_buckets():
    r = client.get("/debug/api/bench/items", params={"suite": "v6", "bucket": "knowledge", "limit": 3}).json()
    assert r["total"] > 0 and len(r["items"]) == 3
    item = client.get("/debug/api/bench/item", params={"suite": "v6", "id": r["items"][0]["id"]}).json()
    assert item["reference"] and item["kind"] == "single"


def test_score_endpoint_uses_the_eval_scorer():
    ids = client.get("/debug/api/bench/items", params={"suite": "v6", "bucket": "knowledge", "limit": 1}).json()["items"]
    item = client.get("/debug/api/bench/item", params={"suite": "v6", "id": ids[0]["id"]}).json()
    good = client.post("/debug/api/score", json={"suite": "v6", "item_id": item["id"],
                                                  "content": f"Відповідь: {item['reference']}"}).json()
    assert good["score"] == 1.0


def test_detail_path_must_stay_under_results():
    r = client.get("/debug/api/bench/items", params={"suite": "v6", "detail": "../etc/passwd", "system": "x"})
    assert r.status_code == 400
