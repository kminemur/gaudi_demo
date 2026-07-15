#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


DEFAULT_MODEL_IDS = [
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-235B-A22B",
]
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HF_HOME = PROJECT_ROOT / "hf_cache"


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
        help="Optional Hugging Face hub cache dir. Defaults to HF_HOME/hub or ./hf_cache/hub.",
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
        "--prepare",
        action="store_true",
        help=(
            "Prepare models for chat_server.py: remove stale incomplete files, "
            "download snapshots, and verify the local cache afterwards."
        ),
    )
    parser.add_argument(
        "--clean-incomplete",
        action="store_true",
        help="Remove *.incomplete blob files for the selected models before downloading.",
    )
    parser.add_argument(
        "--clean-locks",
        action="store_true",
        help="Remove Hugging Face lock files for the selected models before downloading.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; verify that selected model snapshots are already complete locally.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the local cache verification after downloading.",
    )
    parser.add_argument(
        "--etag-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for Hugging Face metadata requests.",
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


def model_cache_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def cache_root(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir).expanduser()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return DEFAULT_HF_HOME / "hub"


def model_cache_dir(args: argparse.Namespace, model_id: str) -> Path:
    return cache_root(args) / model_cache_name(model_id)


def snapshot_ref_path(args: argparse.Namespace, model_id: str) -> Path | None:
    model_dir = model_cache_dir(args, model_id)
    ref_name = args.revision or "main"
    ref_path = model_dir / "refs" / ref_name
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        if revision:
            return model_dir / "snapshots" / revision
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(snapshots_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    return snapshots[0] if snapshots else None


def find_incomplete_files(args: argparse.Namespace, model_id: str) -> list[Path]:
    blobs_dir = model_cache_dir(args, model_id) / "blobs"
    if not blobs_dir.exists():
        return []
    return sorted(blobs_dir.glob("*.incomplete"))


def find_lock_files(args: argparse.Namespace, model_id: str) -> list[Path]:
    locks_dir = cache_root(args) / ".locks" / model_cache_name(model_id)
    if not locks_dir.exists():
        return []
    return sorted(locks_dir.glob("*.lock"))


def remove_paths(paths: Iterable[Path], label: str) -> None:
    removed = 0
    for path in paths:
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    print(f"  removed {removed} {label}", flush=True)


def verify_snapshot(args: argparse.Namespace, model_id: str, local_path: str | None = None) -> tuple[bool, list[str]]:
    snapshot_path = Path(local_path) if local_path else snapshot_ref_path(args, model_id)
    problems: list[str] = []
    if snapshot_path is None or not snapshot_path.exists():
        return False, ["snapshot is missing"]

    files = [path for path in snapshot_path.rglob("*") if path.is_file() or path.is_symlink()]
    if not files:
        problems.append("snapshot contains no files")

    broken_links = [path for path in files if path.is_symlink() and not path.exists()]
    if broken_links:
        sample = ", ".join(str(path.relative_to(snapshot_path)) for path in broken_links[:5])
        problems.append(f"{len(broken_links)} broken snapshot links: {sample}")

    incomplete_files = find_incomplete_files(args, model_id)
    if incomplete_files:
        total_size = sum(path.stat().st_size for path in incomplete_files if path.exists())
        problems.append(
            f"{len(incomplete_files)} incomplete blob files remain ({total_size / (1024 ** 3):.2f} GiB)"
        )

    return not problems, problems


def main() -> int:
    args = parse_args()

    if args.list_defaults:
        print("Default model IDs:")
        for model_id in DEFAULT_MODEL_IDS:
            print(f"- {model_id}", flush=True)
        return 0

    model_ids = resolve_model_ids(args)
    allow_patterns = args.allow_pattern or None
    ignore_patterns = args.ignore_pattern or None
    should_clean_incomplete = args.prepare or args.clean_incomplete
    should_verify = args.prepare or not args.no_verify

    if args.cache_dir:
        print(f"cache_dir={args.cache_dir}", flush=True)
    else:
        hf_home = os.environ.get("HF_HOME", str(DEFAULT_HF_HOME))
        print(f"HF_HOME={hf_home}", flush=True)
    print(f"cache_root={cache_root(args)}", flush=True)

    print(f"models={len(model_ids)}", flush=True)
    failures = []

    for index, model_id in enumerate(model_ids, start=1):
        print(f"[{index}/{len(model_ids)}] preparing {model_id}", flush=True)
        try:
            incomplete_files = find_incomplete_files(args, model_id)
            lock_files = find_lock_files(args, model_id)
            print(f"  cache: {model_cache_dir(args, model_id)}", flush=True)
            print(f"  incomplete files: {len(incomplete_files)}", flush=True)
            print(f"  lock files: {len(lock_files)}", flush=True)
            if should_clean_incomplete and incomplete_files:
                remove_paths(incomplete_files, "incomplete files")
            if args.clean_locks and lock_files:
                remove_paths(lock_files, "lock files")

            if args.verify_only:
                local_path = None
                print("  verify-only: skipping download", flush=True)
            else:
                local_path = snapshot_download(
                    repo_id=model_id,
                    repo_type="model",
                    cache_dir=str(cache_root(args)),
                    revision=args.revision,
                    token=args.token,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    max_workers=args.max_workers,
                    force_download=args.force_download,
                    local_files_only=False,
                    etag_timeout=args.etag_timeout,
                    resume_download=True,
                )
                print(f"  downloaded: {local_path}", flush=True)

            if should_verify or args.verify_only:
                ok, problems = verify_snapshot(args, model_id, local_path)
                if not ok:
                    raise RuntimeError("; ".join(problems))
                print("  verified: local snapshot is complete", flush=True)
        except Exception as error:  # noqa: BLE001
            failures.append((model_id, str(error)))
            print(f"  failed: {error}", flush=True)

    if failures:
        print("\nFailed downloads:", flush=True)
        for model_id, error in failures:
            print(f"- {model_id}: {error}", flush=True)
        return 1

    print("All downloads completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
