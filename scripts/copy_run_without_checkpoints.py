#!/usr/bin/env python3
"""Copy a training run directory while excluding checkpoint artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
from pathlib import Path


DEFAULT_EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "checkpoints",
}
DEFAULT_EXCLUDE_FILE_PATTERNS = {
    "*.ckpt",
    "*.pt",
    "*.pth",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a run directory to a download folder while excluding "
            "checkpoints and common checkpoint files."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Source run directory, e.g. outputs/my_run_123456.",
    )
    parser.add_argument(
        "dest_dir",
        type=Path,
        help="Destination directory to create or overwrite.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete destination first if it already exists.",
    )
    parser.add_argument(
        "--archive",
        choices=("none", "zip", "gztar"),
        default="none",
        help="Optionally create an archive next to dest_dir after copying.",
    )
    return parser.parse_args()


def _should_skip_file(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in DEFAULT_EXCLUDE_FILE_PATTERNS)


def _copy_tree(src: Path, dst: Path) -> dict[str, object]:
    copied_files = 0
    copied_bytes = 0
    skipped_dirs: list[str] = []
    skipped_files: list[str] = []

    for current_src_str, dirnames, filenames in os.walk(src):
        current_src = Path(current_src_str)
        rel = current_src.relative_to(src)
        current_dst = dst / rel
        current_dst.mkdir(parents=True, exist_ok=True)

        kept_dirs: list[str] = []
        for dirname in dirnames:
            child_rel = rel / dirname
            if dirname in DEFAULT_EXCLUDE_DIRS:
                skipped_dirs.append(child_rel.as_posix())
            else:
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            source_file = current_src / filename
            rel_file = rel / filename
            if _should_skip_file(source_file):
                skipped_files.append(rel_file.as_posix())
                continue
            target_file = current_dst / filename
            shutil.copy2(source_file, target_file)
            copied_files += 1
            copied_bytes += target_file.stat().st_size

    return {
        "source": str(src),
        "destination": str(dst),
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "excluded_directories": sorted(skipped_dirs),
        "excluded_files": sorted(skipped_files),
    }


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    dest_dir = args.dest_dir.resolve()

    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if dest_dir == run_dir or run_dir in dest_dir.parents:
        raise SystemExit("Destination must not be inside the source run directory.")
    if dest_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Destination already exists; pass --overwrite: {dest_dir}")
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    summary = _copy_tree(run_dir, dest_dir)

    manifest_path = dest_dir / "copy_manifest_no_checkpoints.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    archive_path = None
    if args.archive != "none":
        archive_path = shutil.make_archive(
            base_name=str(dest_dir),
            format=args.archive,
            root_dir=dest_dir.parent,
            base_dir=dest_dir.name,
        )
        summary["archive"] = archive_path
        manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
