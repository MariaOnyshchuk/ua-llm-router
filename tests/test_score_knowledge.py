"""Regression tests for knowledge scoring of numeric-fact vs MCQ references."""

from scripts.score_results import score_knowledge


def test_numeric_fact_reference_is_matched_by_contains():
    assert score_knowledge("Україна проголосила незалежність 24 серпня 1991 року.", "1991")["score"] == 1.0
    assert score_knowledge("В Україні 24 області.", "24")["score"] == 1.0


def test_numeric_fact_wrong_answer_is_zero():
    assert score_knowledge("Незалежність проголосили у 1917 році.", "1991")["score"] == 0.0


def test_mcq_letter_reference_still_works():
    assert score_knowledge("Відповідь: Г", "Г")["score"] == 1.0
    assert score_knowledge("Б", "Г")["score"] == 0.0
    assert score_knowledge("B", "Б")["score"] == 1.0  # latin B accepted for Cyrillic Б
