#!/usr/bin/env python3
"""Package pinned eval JSON summaries and upload them to Hugging Face Datasets.

Skips raw generation JSONL (licensed prompts) and empty/smoke artifacts.

Usage:
  python3 -m venv .venv && source .venv/bin/activate
  pip install huggingface_hub
  huggingface-cli login
  python scripts/publish_hf_eval.py --repo YOUR_HF_USER/ua-llm-router-eval
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CARD = ROOT / "docs" / "hf_dataset_card.md"

SKIP_DIRS = {
    "_empty_failed_stubs",
    "week_10_composite_smoke",
    "week_10_composite_smoke_v3",
}

SKIP_NAMES = {".json", ".gitkeep"}


def should_copy(path: Path) -> bool:
    if path.name in SKIP_NAMES or path.name.startswith("."):
        return False
    if any(p in SKIP_DIRS for p in path.parts):
        return False
    if path.suffix == ".json":
        return True
    if path.suffix == ".csv" and (
        path.name == "progress_ledger.csv" or "tables" in path.parts
    ):
        return True
    if path.suffix == ".jsonl" and "score" in path.name.lower():
        return True
    return False


def stage(out_dir: Path) -> list[Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    copied: list[Path] = []
    for src in RESULTS.rglob("*"):
        if not src.is_file() or not should_copy(src):
            continue
        dest = out_dir / src.relative_to(RESULTS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(dest)
    readme = out_dir / "README.md"
    shutil.copy2(CARD, readme)
    copied.append(readme)
    return copied


def patch_readme(out_dir: Path, repo_id: str) -> None:
    text = (out_dir / "README.md").read_text(encoding="utf-8")
    (out_dir / "README.md").write_text(
        text.replace("USERNAME/ua-llm-router-eval", repo_id),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="e.g. MariaOnyshchuk/ua-llm-router-eval")
    p.add_argument("--stage-dir", type=Path, default=Path("/tmp/ua-llm-router-eval"))
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    files = stage(args.stage_dir)
    patch_readme(args.stage_dir, args.repo)
    print(f"Staged {len(files)} files in {args.stage_dir}")
    if args.dry_run:
        return

    from huggingface_hub import HfApi, whoami

    me = whoami()
    print(f"Logged in as {me.get('name')}")
    api = HfApi()
    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(args.stage_dir),
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Add pinned evaluation JSON summaries and progress ledger",
    )
    print(f"Published https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
