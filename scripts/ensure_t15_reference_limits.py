from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from build_t15_reference_limits import main as build_reference_limits

REQUIRED_KEYS = ("positive_ramp_mean_a_per_s", "negative_ramp_abs_mean_a_per_s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ensure processed T15 reference limits include robust ramp means.")
    ap.add_argument("--path", default="data/processed/t15_reference_limits.json")
    ap.add_argument("--lock-timeout-s", type=float, default=300.0)
    args = ap.parse_args(argv)

    path = Path(args.path)
    if _has_required_keys(path):
        print(f"{path} has robust ramp mean fields")
        return 0

    lock_dir = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + float(args.lock_timeout_s)
    owns_lock = False
    while time.monotonic() < deadline:
        try:
            lock_dir.mkdir(parents=True)
            owns_lock = True
            break
        except FileExistsError:
            if _has_required_keys(path):
                print(f"{path} was rebuilt by another task")
                return 0
            time.sleep(1.0)
    if not owns_lock:
        raise TimeoutError(f"timed out waiting for reference-limit build lock: {lock_dir}")

    try:
        if _has_required_keys(path):
            print(f"{path} was rebuilt by another task")
            return 0
        print(f"rebuilding stale reference limits: {path}")
        return int(build_reference_limits(["--out", str(path)]))
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def _has_required_keys(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(key in raw for key in REQUIRED_KEYS)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"failed to ensure T15 reference limits: {exc}", file=sys.stderr)
        raise
