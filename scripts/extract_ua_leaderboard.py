"""Extract wave-1 UA leaderboard tasks into benchmarks/corpus/.

Datasets: INSAIT ifeval_ukr / mmlu_ukr / arc-challenge_ukr, facebook/belebele,
optional WMT-22 en↔uk.
"""

from __future__ import annotations

import json
from typing import Any

from scripts.mcq_format import choices_from_item, format_mcq_prompt, letter_from_answer


def write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _item_id(prefix: str, raw: Any, index: int) -> str:
    if raw is None or str(raw).strip() == "":
        return f"{prefix}-{index:05d}"
    safe = str(raw).replace("/", "-").replace(" ", "_")[:80]
    return f"{prefix}-{safe}"


def rows_from_ifeval(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i, item in enumerate(records):
        prompt = str(item.get("prompt") or item.get("instruction") or "").strip()
        if not prompt:
            continue
        ids = item.get("instruction_id_list") or item.get("instruction_ids") or []
        kwargs = item.get("kwargs") or item.get("kwarg") or []
        rows.append(
            {
                "id": _item_id("ifeval", item.get("key") or item.get("id"), i),
                "bucket": "instruct",
                "prompt": prompt,
                "reference": "",
                "notes": "source=ifeval_ukr score_mode=ifeval",
                "ifeval_instruction_id_list": [str(x) for x in list(ids)],
                "ifeval_kwargs": json.loads(json.dumps(list(kwargs) if not isinstance(kwargs, dict) else [kwargs], default=str)),
            }
        )
    return rows


def rows_from_belebele(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i, item in enumerate(records):
        question = str(item.get("question") or "").strip()
        choices = [
            str(item.get(f"mc_answer{k}") or "").strip()
            for k in range(1, 5)
            if str(item.get(f"mc_answer{k}") or "").strip()
        ]
        if not question or len(choices) < 2:
            continue
        letter = letter_from_answer(item.get("correct_answer_num") or item.get("correct_answer"), len(choices), one_indexed=True)
        passage = str(item.get("flores_passage") or item.get("passage") or "").strip()
        rows.append(
            {
                "id": _item_id("belebele", item.get("link") or item.get("id"), i),
                "bucket": "knowledge",
                "prompt": format_mcq_prompt(
                    question,
                    choices,
                    passage=passage,
                    intro=(
                        "Прочитай уривок і дай відповідь на питання. "
                        "Відповідь — лише літера варіанту (A, B, C або D)."
                    ),
                ),
                "reference": letter,
                "notes": "source=belebele lang=ukr_Cyrl",
            }
        )
    return rows


def rows_from_mmlu(records: list[dict[str, Any]], *, cap: int = 0, seed: int = 42) -> list[dict[str, Any]]:
    import random
    from collections import defaultdict

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, item in enumerate(records):
        question = str(item.get("question") or item.get("input") or "").strip()
        choices = choices_from_item(item)
        if not question or len(choices) < 2:
            continue
        answer = item.get("answer")
        if answer is None:
            answer = item.get("answerKey") or item.get("label")
        letter = letter_from_answer(answer, len(choices))
        subject = str(item.get("subject") or item.get("category") or "general")
        row = {
            "id": _item_id("mmlu", item.get("id") or f"{subject}-{i}", i),
            "bucket": "knowledge",
            "prompt": format_mcq_prompt(question, choices),
            "reference": letter,
            "notes": f"source=mmlu_ukr subject={subject}",
        }
        by_subject[subject].append(row)

    rng = random.Random(seed)
    if cap and cap > 0:
        subjects = sorted(by_subject)
        if not subjects:
            return []
        per = max(1, cap // len(subjects))
        chosen: list[dict[str, Any]] = []
        leftover: list[dict[str, Any]] = []
        for subj in subjects:
            chunk = by_subject[subj][:]
            rng.shuffle(chunk)
            chosen.extend(chunk[:per])
            leftover.extend(chunk[per:])
        rng.shuffle(leftover)
        rows = (chosen + leftover)[:cap]
        rows.sort(key=lambda r: str(r["id"]))
        return rows

    out = [row for chunk in by_subject.values() for row in chunk]
    out.sort(key=lambda r: str(r["id"]))
    return out


def rows_from_arc(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i, item in enumerate(records):
        question = str(item.get("question") or "").strip()
        choices = choices_from_item(item)
        if not question or len(choices) < 2:
            continue
        letter = letter_from_answer(item.get("answerKey") or item.get("answer"), len(choices))
        rows.append(
            {
                "id": _item_id("arc", item.get("id"), i),
                "bucket": "knowledge",
                "prompt": format_mcq_prompt(question, choices),
                "reference": letter,
                "notes": "source=arc-challenge_ukr",
            }
        )
    return rows


def rows_from_wmt(records: list[dict[str, Any]], *, split: str = "test") -> list[dict[str, Any]]:
    rows = []
    for i, item in enumerate(records):
        translation = item.get("translation") if isinstance(item.get("translation"), dict) else item
        en = str(
            (translation or {}).get("en")
            or item.get("en")
            or item.get("english")
            or ""
        ).strip()
        uk = str(
            (translation or {}).get("uk")
            or (translation or {}).get("ukr")
            or item.get("uk")
            or ""
        ).strip()
        if not en or not uk:
            continue
        sid = item.get("id", i)
        rows.append(
            {
                "id": f"wmt22-en-uk-{split}-{sid}",
                "bucket": "translate",
                "prompt": (
                    "Переклади з англійської українською. "
                    "Надай лише переклад, без пояснень:\n" + en
                ),
                "reference": uk,
                "notes": f"source=wmt22 split={split} direction=en-uk",
            }
        )
        rows.append(
            {
                "id": f"wmt22-uk-en-{split}-{sid}",
                "bucket": "translate",
                "prompt": (
                    "Переклади з української англійською. "
                    "Надай лише переклад, без пояснень:\n" + uk
                ),
                "reference": en,
                "notes": f"source=wmt22 split={split} direction=uk-en",
            }
        )
    return rows


def _as_records(ds: Any) -> list[dict[str, Any]]:
    """Materialize HF Dataset / DatasetDict rows (datasets 5 `list(ds)` yields column names)."""
    if ds is None:
        return []
    typename = type(ds).__name__
    if typename == "DatasetDict" or (
        hasattr(ds, "keys") and callable(ds.keys) and not hasattr(ds, "features")
    ):
        rows: list[dict[str, Any]] = []
        for key in ds:
            rows.extend(_as_records(ds[key]))
        return rows
    if hasattr(ds, "to_list"):
        data = ds.to_list()
        if data and isinstance(data[0], dict):
            return data
    if hasattr(ds, "column_names") and hasattr(ds, "__len__"):
        cols = list(ds.column_names)
        n = len(ds)
        return [{col: ds[col][i] for col in cols} for i in range(n)]
    raise TypeError(f"cannot read records from {type(ds)}")


def _hf_records(path: str, name: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if name:
        ds = load_dataset(path, name, split=split) if split else load_dataset(path, name)
    else:
        ds = load_dataset(path, split=split) if split else load_dataset(path)
    return _as_records(ds)


def extract_ifeval_ukr(out_dir) -> dict:
    records = _hf_records("INSAIT-Institute/ifeval_ukr")
    rows = rows_from_ifeval(records)
    path = out_dir / "ifeval_ukr_instruct.jsonl"
    write_jsonl(path, rows)
    return {"wrote": str(path), "n": len(rows)}


def extract_belebele_uk(out_dir) -> dict:
    records = _hf_records("facebook/belebele", "ukr_Cyrl")
    rows = rows_from_belebele(records)
    path = out_dir / "belebele_uk_knowledge.jsonl"
    write_jsonl(path, rows)
    return {"wrote": str(path), "n": len(rows)}


def extract_mmlu_ukr(out_dir, *, cap: int = 2000, seed: int = 42) -> dict:
    records = _hf_records("INSAIT-Institute/mmlu_ukr")
    rows = rows_from_mmlu(records, cap=cap, seed=seed)
    path = out_dir / "mmlu_ukr_knowledge.jsonl"
    write_jsonl(path, rows)
    return {"wrote": str(path), "n": len(rows), "cap": cap, "source_n": len(records)}


def extract_arc_ukr(out_dir) -> dict:
    records = _hf_records("INSAIT-Institute/arc-challenge_ukr")
    rows = rows_from_arc(records)
    path = out_dir / "arc_challenge_ukr_knowledge.jsonl"
    write_jsonl(path, rows)
    return {"wrote": str(path), "n": len(rows)}


def extract_wmt22(out_dir) -> dict:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    split_used = "test"
    for path, name in (
        ("wmt/wmt22", "uk-en"),
        ("wmt22", "uk-en"),
        ("Helsinki-NLP/opus-100", "en-uk"),
    ):
        try:
            records = _hf_records(path, name, split="test")
            break
        except Exception as exc:
            errors.append(f"{path}/{name}: {type(exc).__name__}: {exc}")
    if not records:
        return {"wrote": None, "n": 0, "skipped": True, "errors": errors[:5]}
    rows = rows_from_wmt(records, split=split_used)
    out_path = out_dir / "wmt22_translate.jsonl"
    write_jsonl(out_path, rows)
    return {"wrote": str(out_path), "n": len(rows), "pairs": len(rows) // 2}
