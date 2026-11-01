#!/usr/bin/env python3
"""Fill the Wikipedia cache for a benchmark split and measure retrieval without any model.

Run on the laptop (the lab nodes may have no internet), then copy .cache/wiki to the lab
and run the eval with --wiki-rag --wiki-offline.

  python scripts/build_wiki_cache.py --bench benchmarks/tasks_public_4401.jsonl --split dev

Retrieval check for multiple-choice items: each option gets a support score, the share of
its lemmas that occur in the retrieved passages; an option counts as "mentioned" at
support >= 0.5. Compare gold_mentioned_rate with distractor_mentioned_rate: passages that
point at the answer mention the correct option more often than the wrong ones. This needs
no model and is not a model score.

Writes results/wiki_rag/retrieval_<bench>_<split>.json (summary) and
..._detail.jsonl (ids and numbers only, no prompt text and no passage text).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router.intent_rules import route_intent  # noqa: E402
from router.wiki_search import (  # noqa: E402
    WikiClient,
    WikiConfig,
    augment_prompt,
    split_question,
    tokens,
    wiki_augment,
)

LETTERS = ("ABCDE", "АБВГД")


def source_family(item_id: str) -> str:
    parts = item_id.split("-")
    return f"{parts[0]}-{parts[1]}" if parts[0] == "zno" else parts[0]


def gold_index(reference: str) -> int | None:
    ref = (reference or "").strip()[:1].upper()
    for alphabet in LETTERS:
        if ref and ref in alphabet:
            return alphabet.index(ref)
    return None


def option_support(options: list[str], passage_text: str) -> list[float]:
    have = set(tokens(passage_text))
    out = []
    for opt in options:
        want = set(tokens(opt))
        out.append(len(want & have) / len(want) if want else 0.0)
    return out


def cache_digest(cache_dir: Path) -> str:
    h = hashlib.sha1()
    for path in sorted(cache_dir.glob("*.json")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="benchmarks/tasks_public_4401.jsonl")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--bucket", default="knowledge")
    ap.add_argument("--cache", default=".cache/wiki")
    ap.add_argument("--offline", action="store_true", help="only read the cache (check that it is complete)")
    ap.add_argument("--id-prefix", default="", help="comma-separated id prefixes, e.g. zno-his,zno-geo")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="results/wiki_rag")
    args = ap.parse_args()

    bench = ROOT / args.bench if not Path(args.bench).is_absolute() else Path(args.bench)
    items = [json.loads(line) for line in bench.open(encoding="utf-8") if line.strip() and not line.startswith("#")]
    items = [x for x in items if (not args.split or x.get("split") == args.split) and x.get("bucket") == args.bucket]
    if args.id_prefix:
        prefixes = tuple(x.strip() for x in args.id_prefix.split(",") if x.strip())
        items = [x for x in items if str(x["id"]).startswith(prefixes)]
    if args.limit:
        items = items[: args.limit]
    cache = Path(args.cache)
    cfg = WikiConfig(cache_dir=cache if cache.is_absolute() else ROOT / cache, offline=args.offline)
    client = WikiClient(cfg)

    detail = []
    for i, item in enumerate(items, 1):
        prompt = str(item.get("prompt") or "")
        intent = route_intent(prompt).intent
        _new_prompt, meta = wiki_augment(prompt, intent, client)
        passages = meta.get("passages") or []
        _stem, options = split_question(prompt)
        gold = gold_index(str(item.get("reference") or ""))
        row = {
            "id": item["id"],
            "source": source_family(str(item["id"])),
            "gate_intent": intent,
            "gate": meta["gate"],
            "used": bool(meta.get("used")),
            "n_queries": len(meta.get("queries") or []),
            "n_titles": len(meta.get("titles") or []),
            "n_passages": len(passages),
            "context_chars": len(augment_prompt("", passages)),
            "errors": len(meta.get("errors") or []),
            "gold_mentioned": None,
            "distractors_mentioned": None,
        }
        if passages and options and gold is not None and gold < len(options):
            support = option_support(options, " ".join(p["text"] for p in passages))
            row["gold_support"] = round(support[gold], 3)
            row["gold_mentioned"] = support[gold] >= 0.5
            wrong = [x for k, x in enumerate(support) if k != gold]
            row["distractors_mentioned"] = round(sum(x >= 0.5 for x in wrong) / max(1, len(wrong)), 3)
        detail.append(row)
        if i % 25 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] api={client.api_calls} cache={client.cache_hits}", flush=True)
    client.close()

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in detail:
        by_source[row["source"]].append(row)
    by_source["ALL"] = detail

    def summarize(rows: list[dict]) -> dict:
        gated = [r for r in rows if r["gate"] == "knowledge question"]
        used = [r for r in rows if r["used"]]
        judged = [r for r in used if r["gold_mentioned"] is not None]
        return {
            "n": len(rows),
            "gate_pass": len(gated),
            "used": len(used),
            "use_rate": round(len(used) / max(1, len(rows)), 3),
            "mcq_judged": len(judged),
            "gold_mentioned_rate": round(sum(r["gold_mentioned"] for r in judged) / max(1, len(judged)), 3),
            "distractor_mentioned_rate": round(
                sum(r["distractors_mentioned"] for r in judged) / max(1, len(judged)), 3
            ),
            "items_with_errors": sum(r["errors"] > 0 for r in rows),
        }

    summary = {
        "bench": bench.name,
        "split": args.split,
        "bucket": args.bucket,
        "config": {k: v for k, v in vars(cfg).items() if k != "cache_dir"},
        "cache_files": len(list(cfg.cache_dir.glob("*.json"))),
        "cache_sha1": cache_digest(cfg.cache_dir),
        "api_calls": client.api_calls,
        "by_source": {k: summarize(v) for k, v in sorted(by_source.items())},
    }
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"retrieval_{bench.stem}_{args.split or 'all'}"
    if args.id_prefix or args.limit:
        stem += "_subset"
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / f"{stem}_detail.jsonl").open("w", encoding="utf-8") as f:
        for row in detail:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary["by_source"], ensure_ascii=False, indent=1))
    print(f"wrote {out_dir / stem}.json  cache_sha1={summary['cache_sha1']}")


if __name__ == "__main__":
    main()
