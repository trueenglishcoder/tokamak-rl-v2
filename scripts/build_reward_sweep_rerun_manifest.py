#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BAD_STATUSES = {
    "missing",
    "missing_policy_validation",
    "sweep_failed_training",
    "sweep_failed_eval",
    "interrupted",
    "failed_initial_state_library",
    "failed_reference_limits",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _manifest_roots(root: Path) -> list[Path]:
    candidates = [
        root,
        root / "pass1_broad",
        root / "pass2_focused",
    ]
    return [path for path in candidates if (path / "variants.json").exists()]


def _folder_for_variant(variant: dict[str, Any]) -> str:
    folder = variant.get("folder")
    if folder:
        return str(folder)
    return f"v{int(variant.get('index', 0)):03d}_{variant.get('name', 'variant')}"


def _reason_for_run(run_dir: Path) -> str | None:
    if not run_dir.exists():
        return "missing_folder"
    validation_path = run_dir / "policy_validation.json"
    if not validation_path.exists():
        return "missing_policy_validation"
    validation = _read_json(validation_path)
    status = str(validation.get("status") or "")
    if status in BAD_STATUSES or status.startswith("failed_"):
        return status or "bad_status"
    actor_eval = validation.get("actor_eval")
    if not isinstance(actor_eval, dict) or not actor_eval:
        return "missing_actor_eval"
    return None


def build(root: Path) -> dict[str, Any]:
    variants: list[dict[str, Any]] = []
    total_expected = 0
    for manifest_root in _manifest_roots(root):
        manifest = _read_json(manifest_root / "variants.json")
        sweep_pass = str(manifest.get("sweep_pass") or manifest_root.name)
        manifest_variants = manifest.get("variants")
        if not isinstance(manifest_variants, list):
            continue
        for variant in manifest_variants:
            if not isinstance(variant, dict):
                continue
            total_expected += 1
            folder = _folder_for_variant(variant)
            run_dir = manifest_root / folder
            reason = _reason_for_run(run_dir)
            if reason is None:
                continue
            variants.append(
                {
                    "sweep_pass": sweep_pass,
                    "manifest": str(manifest_root / "variants.json"),
                    "root": str(manifest_root),
                    "variant_index": int(variant.get("index", -1)),
                    "folder": folder,
                    "run_dir": str(run_dir),
                    "reason": reason,
                }
            )
    return {
        "root": str(root),
        "total_expected": total_expected,
        "missing_or_failed_count": len(variants),
        "variants": variants,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a manifest of missing or failed reward-sweep variants.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.out)
    print(f"missing_or_failed_count={payload['missing_or_failed_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
