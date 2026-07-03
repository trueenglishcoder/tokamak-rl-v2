#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assert that a generated JSON experiment config differs from a "
            "source config only at explicitly allowed dotted paths."
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        help="Allowed dotted path, e.g. training.output_dir. May be repeated.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc


def _path_join(parent: str, key: str) -> str:
    return key if not parent else f"{parent}.{key}"


def _diff(source: Any, generated: Any, *, path: str = "") -> list[str]:
    if type(source) is not type(generated):
        return [path or "<root>"]

    if isinstance(source, dict):
        paths: list[str] = []
        keys = set(source) | set(generated)
        for key in sorted(keys):
            child = _path_join(path, str(key))
            if key not in source or key not in generated:
                paths.append(child)
            else:
                paths.extend(_diff(source[key], generated[key], path=child))
        return paths

    if isinstance(source, list):
        if len(source) != len(generated):
            return [path or "<root>"]
        paths = []
        for index, (a, b) in enumerate(zip(source, generated, strict=True)):
            paths.extend(_diff(a, b, path=_path_join(path, str(index))))
        return paths

    if source != generated:
        return [path or "<root>"]
    return []


def main() -> int:
    args = _parse_args()
    source = _load_json(args.source)
    generated = _load_json(args.generated)
    allowed = set(args.allow)
    diffs = sorted(set(_diff(source, generated)))
    unexpected = [path for path in diffs if path not in allowed]

    if unexpected:
        print("unexpected config differences:", flush=True)
        for path in unexpected:
            print(f"  {path}", flush=True)
        print("allowed differences:", flush=True)
        for path in sorted(allowed):
            print(f"  {path}", flush=True)
        return 2

    print("config_diff_ok", "changed_paths=" + ",".join(diffs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
