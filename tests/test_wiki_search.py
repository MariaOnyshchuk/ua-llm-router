"""Wikipedia retrieval without network: the API is replaced by a fake transport."""

import json

import httpx
import pytest

from router.wiki_search import (
    WikiClient,
    WikiConfig,
    build_queries,
    chunk_page,
    needs_retrieval,
    split_question,
    wiki_augment,
)

ZNO = (
    "Дай відповідь на тестове завдання ЗНО. Відповідь — лише літера варіанту (А, Б, В, Г або Д).\n\n"
    "Що стало наслідком розгортання в СРСР стахановського руху?\n\n"
    "Варіанти:\nА) уведення кримінальної відповідальності\nБ) заборона змін\n"
    "В) підвищення норм виробітку продукції\nГ) семиденний тиждень"
)

PAGE = (
    "Стахановський рух — масовий рух у СРСР за підвищення продуктивності праці. "
    "Наслідком стахановського руху стало підвищення норм виробітку на підприємствах.\n\n"
    "== Посилання ==\nhttps://example.org"
)


def fake_client(tmp_path, *, offline=False):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        if params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": [{"title": "Стахановський рух"}]}})
        return httpx.Response(200, json={"query": {"pages": [{"title": params["titles"], "extract": PAGE}]}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = WikiConfig(cache_dir=tmp_path, offline=offline)
    return WikiClient(cfg, http=http), calls


def test_split_question_drops_the_format_line_and_reads_options():
    stem, options = split_question(ZNO)
    assert stem.startswith("Що стало наслідком") and len(options) == 4
    assert options[2] == "підвищення норм виробітку продукції"


def test_gate_skips_passages_calculations_and_other_intents():
    assert needs_retrieval(ZNO, "knowledge") == (True, "knowledge question")
    assert needs_retrieval(ZNO, "translate")[0] is False
    passage = "Прочитай уривок і дай відповідь.\n\n" + "Текст " * 200 + "\n\nВаріанти:\nA) так\nB) ні"
    assert needs_retrieval(passage, "knowledge") == (False, "passage given")
    assert needs_retrieval("Обчисліть: 2x + 3 = 7, x^2 = ?", "knowledge") == (False, "calculation")
    years = "Проти кого Мономах ходив у 1103 р., 1111 р. та 1116 р.?"
    assert needs_retrieval(years, "knowledge")[0] is True


def test_queries_are_short_noun_phrases():
    queries = build_queries(ZNO)
    assert any("стахановського руху" in q for q in queries)
    assert all(len(q.split()) <= 6 for q in queries)
    assert not any(q.startswith("Дай") for q in queries)


def test_chunks_stop_at_reference_sections():
    chunks = chunk_page(PAGE, 50)
    assert chunks and not any("example.org" in c for c in chunks)


def test_augment_puts_passages_before_the_unchanged_prompt_and_caches(tmp_path):
    client, calls = fake_client(tmp_path)
    prompt, meta = wiki_augment(ZNO, "knowledge", client)
    assert meta["used"] and meta["passages"][0]["title"] == "Стахановський рух"
    assert prompt.endswith(ZNO) and "Вікіпедії" in prompt
    first = len(calls)

    offline, offline_calls = fake_client(tmp_path, offline=True)
    again, _ = wiki_augment(ZNO, "knowledge", offline)
    assert again == prompt and offline_calls == [] and first > 0
    stored = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert stored["lang"] == "uk" and "response" in stored


def test_offline_without_cache_returns_the_prompt_unchanged(tmp_path):
    client, _ = fake_client(tmp_path, offline=True)
    prompt, meta = wiki_augment(ZNO, "knowledge", client)
    assert prompt == ZNO and not meta["used"] and meta["errors"]


def test_other_intents_never_call_the_api(tmp_path):
    client, calls = fake_client(tmp_path)
    prompt, meta = wiki_augment("Переклади англійською: добрий день", "translate", client)
    assert prompt.startswith("Переклади") and calls == [] and meta["gate"] == "intent translate"


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_prompt_is_safe(tmp_path, text):
    client, _ = fake_client(tmp_path)
    prompt, meta = wiki_augment(text, "knowledge", client)
    assert prompt == text and not meta["used"]
