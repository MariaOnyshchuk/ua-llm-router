"""Tests for the per-bucket few-shot prompt variants (no GPU needed)."""

from pathlib import Path

from scripts.bucket_prompt_variants import (
    BUCKETS,
    apply_bucket_fewshot,
    check_no_overlap,
    parse_buckets,
)

ROOT = Path(__file__).resolve().parents[1]


def _item(bucket: str, prompt: str = "Тестове завдання.") -> dict:
    return {"id": f"{bucket}-x", "bucket": bucket, "prompt": prompt, "reference": "", "notes": ""}


def test_parse_buckets():
    assert parse_buckets("") == frozenset()
    assert parse_buckets("none") == frozenset()
    assert parse_buckets("all") == frozenset(BUCKETS)
    assert parse_buckets("knowledge, code") == frozenset({"knowledge", "code"})


def test_parse_buckets_rejects_unknown_and_alignment():
    for bad in ("alignment", "nonsense"):
        try:
            parse_buckets(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_original_prompt_kept_verbatim_at_the_end():
    for bucket in BUCKETS:
        item = _item(bucket)
        out = apply_bucket_fewshot(item, frozenset({bucket}))
        assert out["prompt"].endswith(item["prompt"])
        assert len(out["prompt"]) > len(item["prompt"])
        assert out["reference"] == item["reference"]
        assert out["id"] == item["id"]


def test_only_selected_buckets_change():
    item = _item("code")
    assert apply_bucket_fewshot(item, frozenset({"knowledge"})) == item
    assert apply_bucket_fewshot(item, frozenset()) == item
    assert apply_bucket_fewshot(_item("alignment"), frozenset(BUCKETS))["prompt"] == "Тестове завдання."


def test_input_item_not_mutated():
    item = _item("translate")
    before = dict(item)
    apply_bucket_fewshot(item, frozenset({"translate"}))
    assert item == before


def test_examples_do_not_appear_in_any_benchmark():
    files = sorted((ROOT / "benchmarks").rglob("*.jsonl"))
    assert files
    assert check_no_overlap(files) == []


def test_overlap_check_detects_a_leak(tmp_path):
    leaked = tmp_path / "leak.jsonl"
    leaked.write_text(
        '{"id":"a","bucket":"knowledge","prompt":"Яка столиця Польщі?\\nА) Прага\\nБ) Варшава",'
        '"reference":"Б"}\n',
        encoding="utf-8",
    )
    assert check_no_overlap([leaked])
