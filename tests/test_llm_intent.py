from router.llm_intent import INTENT_MODEL, INTENTS, parse


def test_parse_takes_the_first_allowed_word():
    assert parse("translate") == "translate"
    assert parse("Тип: **Knowledge**.") == "knowledge"
    assert parse("це code, не chat") == "code"
    assert parse("не знаю") is None
    assert parse("") is None


def test_every_intent_has_a_model():
    assert set(INTENT_MODEL) == set(INTENTS)
