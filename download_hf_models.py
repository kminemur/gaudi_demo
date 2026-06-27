#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Iterable

from huggingface_hub import snapshot_download


DEFAULT_MODEL_IDS = [
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-235B-A22B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model snapshots into local cache."
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Model repo id to download (repeatable). If omitted, defaults are used.",
    )
    parser.add_argument(
        "--all-defaults",
        action="store_true",
        help="Download all default demo model IDs.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache dir for huggingface_hub (otherwise HF_HOME/HF cache is used).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision (branch/tag/commit) for all downloads.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Only download files matching this glob (repeatable).",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Skip files matching this glob (repeatable).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HF token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN if set.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of concurrent download workers per model.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files already exist in cache.",
    )
    parser.add_argument(
        "--list-defaults",
        action="store_true",
        help="Print default model IDs and exit.",
    )
    return parser.parse_args()


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def resolve_model_ids(args: argparse.Namespace) -> list[str]:
    if args.model_id:
        return unique_preserve_order(args.model_id)
    if args.all_defaults:
        return DEFAULT_MODEL_IDS.copy()
    return DEFAULT_MODEL_IDS.copy()


def main() -> int:
    args = parse_args()

    if args.list_defaults:
        print("Default model IDs:")
        for model_id in DEFAULT_MODEL_IDS:
            print(f"- {model_id}")
        return 0

    model_ids = resolve_model_ids(args)
    allow_patterns = args.allow_pattern or None
    ignore_patterns = args.ignore_pattern or None

    if args.cache_dir:
        print(f"cache_dir={args.cache_dir}")
    else:
        hf_home = os.environ.get("HF_HOME", "(huggingface_hub default)")
        print(f"HF_HOME={hf_home}")

    print(f"models={len(model_ids)}")
    failures = []

    for index, model_id in enumerate(model_ids, start=1):
        print(f"[{index}/{len(model_ids)}] downloading {model_id}", flush=True)
        try:
            local_path = snapshot_download(
                repo_id=model_id,
                repo_type="model",
                cache_dir=args.cache_dir,
                revision=args.revision,
                token=args.token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                max_workers=args.max_workers,
                force_download=args.force_download,
            )
            print(f"  done: {local_path}")
        except Exception as error:  # noqa: BLE001
            failures.append((model_id, str(error)))
            print(f"  failed: {error}", file=sys.stderr)

    if failures:
        print("\nFailed downloads:", file=sys.stderr)
        for model_id, error in failures:
            print(f"- {model_id}: {error}", file=sys.stderr)
        return 1

    print("All downloads completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())