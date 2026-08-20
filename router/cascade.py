"""Cascade helpers: Mamay-4B first, escalate to Mamay-12B on low confidence."""

from __future__ import annotations

import re
from dataclasses import dataclass


HEDGE_RE = re.compile(
    r"("
    r"не знаю|не певн|важко сказати|можливо|ймовірно|"
    r"i don't know|not sure|maybe|uncertain|cannot determine"
    r")",
    re.I | re.U,
)

ZNO_LETTER_RE = re.compile(r"(?:відповідь|answer)?\s*[:\-–]?\s*([А-ДA-Ea-eабвгд])\b", re.I)
ALIGN_DIGIT_RE = re.compile(r"(?<!\d)([0-2])(?!\d)")


@dataclass(frozen=True)
class ConfidenceDecision:
    escalate: bool
    confidence: float
    reason: str


def confidence_for_bucket(bucket: str, content: str) -> ConfidenceDecision:
    """Heuristic confidence — cheap, no extra LLM call.

    Escalate when the answer fails format expectations or hedges.
    """
    text = (content or "").strip()
    if not text:
        return ConfidenceDecision(True, 0.0, "empty answer")

    if HEDGE_RE.search(text):
        return ConfidenceDecision(True, 0.2, "hedge phrase")

    if bucket == "alignment":
        if ALIGN_DIGIT_RE.search(text):
            return ConfidenceDecision(False, 0.85, "alignment digit found")
        return ConfidenceDecision(True, 0.15, "no alignment digit 0-2")

    if bucket == "knowledge":
        # ZNO-style MCQ prompts ask for a letter; also accept short factual answers.
        if ZNO_LETTER_RE.search(text) or re.search(r"[А-Д]\b", text):
            return ConfidenceDecision(False, 0.8, "mcq letter found")
        # Very long rambling answers on exam items → escalate
        if len(text) > 600:
            return ConfidenceDecision(True, 0.35, "overlong knowledge answer")
        # Short concrete answers are OK for open knowledge
        if len(text) < 15:
            return ConfidenceDecision(True, 0.3, "too short knowledge answer")
        return ConfidenceDecision(False, 0.65, "knowledge answer present")

    return ConfidenceDecision(False, 0.7, "bucket not cascaded")


CASCADE_BUCKETS = frozenset({"knowledge", "alignment"})


def should_retry_mamay4_micro(item: dict, content: str) -> ConfidenceDecision:
    """S2 micro-cascade: only social UAlign 0/1/2.

    Retry Mamay-4B when the first model (usually Lapa) answers 2 or no digit.
    Ethics 0/1 items are never escalated. Does not use Mamay-12B.
    """
    bucket = str(item.get("bucket") or "")
    iid = str(item.get("id") or "")
    if bucket != "alignment" or not iid.startswith("ualign-social"):
        return ConfidenceDecision(False, 0.9, "not social alignment")

    text = (content or "").strip()
    if not text:
        return ConfidenceDecision(True, 0.0, "empty social answer")
    match = ALIGN_DIGIT_RE.search(text)
    if not match:
        return ConfidenceDecision(True, 0.15, "no alignment digit 0-2")
    digit = match.group(1)
    if digit == "2":
        return ConfidenceDecision(True, 0.4, "social answered 2 → retry mamay4")
    return ConfidenceDecision(False, 0.85, f"social digit {digit}")
