#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


PFC_DERIV_LIMIT = 5070038.4
SOL_DERIV_LIMIT = 20950244.0


def _replace_scalar(text: str, key: str, value: float) -> str:
    pattern = re.compile(rf"^({re.escape(key)}\s*=\s*)([-+0-9.eE]+)\s*$", re.MULTILINE)
    replacement = rf"\g<1>{float(value):.10g}"
    text, count = pattern.subn(replacement, text)
    if count != 1:
        raise ValueError(f"expected exactly one {key!r} entry, found {count}")
    return text


def apply_limits(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _replace_scalar(text, "pfc_deriv_limit", PFC_DERIV_LIMIT)
    text = _replace_scalar(text, "sol_deriv_limit", SOL_DERIV_LIMIT)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply production T15 actuator derivative limits to a tokamak-sim TOML config.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)

    apply_limits(args.config)
    print(f"applied_pfc_deriv_limit={PFC_DERIV_LIMIT:.10g}")
    print(f"applied_sol_deriv_limit={SOL_DERIV_LIMIT:.10g}")
    print(f"config={args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
