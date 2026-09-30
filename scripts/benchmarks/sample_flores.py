#!/usr/bin/env python3
"""Sample deterministic English↔Ukrainian FLORES-200 translation pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-direction", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/samples/flores_translate_v1.jsonl"),
    )
    args = parser.parse_args()

    dataset = load_dataset(
        "facebook/flores",
        "eng_Latn-ukr_Cyrl",
        split="devtest",
    )
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.n_per_direction]

    rows: list[dict] = []
    for index in indices:
        item = dataset[index]
        en = item["sentence_eng_Latn"].strip()
        uk = item["sentence_ukr_Cyrl"].strip()
        source_id = item["id"]
        common_notes = (
            f"source=flores-200 split=devtest source_id={source_id} "
            f"domain={item.get('domain', '')} url={item.get('URL', '')}"
        )
        rows.extend(
            [
                {
                    "id": f"flores-en-uk-{source_id}",
                    "bucket": "translate",
                    "prompt": (
                        "Переклади з англійської українською. "
                        "Надай лише переклад, без пояснень:\n" + en
                    ),
                    "reference": uk,
                    "notes": common_notes + " direction=en-uk",
                },
                {
                    "id": f"flores-uk-en-{source_id}",
                    "bucket": "translate",
                    "prompt": (
                        "Translate from Ukrainian into English. "
                        "Return only the translation, without explanation:\n" + uk
                    ),
                    "reference": en,
                    "notes": common_notes + " direction=uk-en",
                },
            ]
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"wrote": str(args.out), "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
