#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


SUCCESSFUL_SIGMA = "3548133.8923357604"
SUCCESSFUL_INDUCTANCE_L = "4.466835921509635e-07"

PHYSICS_OVERRIDES = {
    "sigma": SUCCESSFUL_SIGMA,
    "inductance_L": SUCCESSFUL_INDUCTANCE_L,
}

BOUNDARY_OVERRIDES = {
    "mode": '"legacy_contour_limited"',
    "base_mode": '"legacy_contour_limited"',
    "legacy_precision_index2": "1e-06",
    "level_smoothing_alpha": "1.0",
    "track_level": "false",
    "smooth_selected_level": "false",
    "soft_level_selection": "false",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write the canonical trim50 plain-GPU-1e-6 T15 machine config used by "
            "the successful replay-window oracle training path. The output is a "
            "machine config only: embedded initial conditions are removed."
        )
    )
    parser.add_argument("--source", type=Path, default=Path("../tokamak-sim/configs/T15MD_new_data.toml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    source = args.source.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"source machine config does not exist: {source}")
    text = source.read_text(encoding="utf-8")
    rendered = _rewrite_machine_config(text)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    _validate(args.out)
    print(args.out)
    return 0


def _rewrite_machine_config(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    section = ""
    boundary_written: set[str] = set()

    def flush_boundary_missing() -> None:
        nonlocal boundary_written
        if section != "boundary":
            return
        for key, value in BOUNDARY_OVERRIDES.items():
            if key not in boundary_written:
                out.append(f"{key} = {value}")
                boundary_written.add(key)

    for raw in lines:
        stripped = raw.strip()
        match_section = re.fullmatch(r"\[([^\]]+)\]", stripped)
        if match_section:
            flush_boundary_missing()
            section = match_section.group(1)
            out.append(raw)
            continue

        key_match = re.match(r"\s*([A-Za-z0-9_]+)\s*=", raw)
        key = key_match.group(1) if key_match else None

        if section == "physics" and key == "Ip0":
            continue
        if section == "physics" and key in PHYSICS_OVERRIDES:
            out.append(f"{key} = {PHYSICS_OVERRIDES[key]}")
            continue

        if section == "boundary" and key is not None:
            if key.startswith("soft_level_") and key not in BOUNDARY_OVERRIDES:
                continue
            if key in BOUNDARY_OVERRIDES:
                out.append(f"{key} = {BOUNDARY_OVERRIDES[key]}")
                boundary_written.add(key)
                continue

        out.append(raw)

    flush_boundary_missing()
    return "\n".join(out).rstrip() + "\n"


def _validate(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    physics = _section_key_values(text, "physics")
    boundary = _section_key_values(text, "boundary")
    if "Ip0" in physics:
        raise ValueError(f"{path} still contains physics.Ip0")
    if float(physics.get("sigma")) != float(SUCCESSFUL_SIGMA):
        raise ValueError(f"{path} sigma was not set to successful-run value")
    if float(physics.get("inductance_L")) != float(SUCCESSFUL_INDUCTANCE_L):
        raise ValueError(f"{path} inductance_L was not set to successful-run value")
    for key, expected in BOUNDARY_OVERRIDES.items():
        value = boundary.get(key)
        if expected in {"true", "false"}:
            wanted = expected == "true"
            actual = str(value).lower()
            if actual not in {"true", "false"} or (actual == "true") != wanted:
                raise ValueError(f"{path} boundary.{key}={value!r}, expected {wanted!r}")
        elif expected.startswith('"'):
            if str(value) != expected.strip('"'):
                raise ValueError(f"{path} boundary.{key}={value!r}, expected {expected}")
        else:
            if float(value) != float(expected):
                raise ValueError(f"{path} boundary.{key}={value!r}, expected {expected}")


def _section_key_values(text: str, section_name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    section = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        match_section = re.fullmatch(r"\[([^\]]+)\]", stripped)
        if match_section:
            section = match_section.group(1)
            continue
        if section != section_name:
            continue
        match_key = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$", raw)
        if match_key:
            values[match_key.group(1)] = match_key.group(2).strip().strip('"')
    return values


if __name__ == "__main__":
    raise SystemExit(main())
