#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SHAPE_REGIMES = [
    {"id": "s0", "shape_mean_weight": 5.0, "shape_max_weight": 1.5},
    {"id": "s1", "shape_mean_weight": 6.0, "shape_max_weight": 2.0},
    {"id": "s2", "shape_mean_weight": 8.0, "shape_max_weight": 2.5},
    {"id": "s3", "shape_mean_weight": 10.0, "shape_max_weight": 3.0},
]

IP_REGIMES = [
    {"id": "i0", "ip_weight": 2.5},
    {"id": "i1", "ip_weight": 3.0},
    {"id": "i2", "ip_weight": 3.5},
]

CURRENT_REGIMES = [
    {"id": "c0", "current_weight": 4.0, "current_soft_fraction": 0.90},
    {"id": "c1", "current_weight": 6.0, "current_soft_fraction": 0.90},
    {"id": "c2", "current_weight": 6.0, "current_soft_fraction": 0.85},
    {"id": "c3", "current_weight": 8.0, "current_soft_fraction": 0.85},
]

ACTUATOR_REGIMES = [
    {"id": "a0", "derivative_weight": 0.25, "action_weight": 0.01, "delta_action_weight": 0.025},
    {"id": "a1", "derivative_weight": 0.50, "action_weight": 0.02, "delta_action_weight": 0.050},
    {"id": "a2", "derivative_weight": 0.75, "action_weight": 0.04, "delta_action_weight": 0.075},
]

FIXED_REWARD = {
    "shape_mean_scale_m": 0.03,
    "shape_max_scale_m": 0.08,
    "ip_scale_a": 25000.0,
    "reward_scale": 1.0,
    "terminal_reward": -20.0,
}


def build_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    index = 0
    for shape in SHAPE_REGIMES:
        for ip in IP_REGIMES:
            for current in CURRENT_REGIMES:
                for actuator in ACTUATOR_REGIMES:
                    name = f"{shape['id']}_{ip['id']}_{current['id']}_{actuator['id']}"
                    reward = {
                        **FIXED_REWARD,
                        "shape_mean_weight": shape["shape_mean_weight"],
                        "shape_max_weight": shape["shape_max_weight"],
                        "ip_weight": ip["ip_weight"],
                        "current_weight": current["current_weight"],
                        "current_soft_fraction": current["current_soft_fraction"],
                        "derivative_weight": actuator["derivative_weight"],
                        "action_weight": actuator["action_weight"],
                        "delta_action_weight": actuator["delta_action_weight"],
                    }
                    variants.append(
                        {
                            "index": index,
                            "name": name,
                            "folder": f"v{index:03d}_{name}",
                            "shape_regime": shape["id"],
                            "ip_regime": ip["id"],
                            "current_regime": current["id"],
                            "actuator_regime": actuator["id"],
                            "reward": reward,
                        }
                    )
                    index += 1
    return variants


def build_manifest() -> dict[str, Any]:
    variants = build_variants()
    return {
        "description": "144-run T15 CSV segmented-profile reward direction sweep",
        "variant_count": len(variants),
        "runs_per_array_task": 3,
        "array_task_count": 48,
        "fixed_reward": FIXED_REWARD,
        "shape_regimes": SHAPE_REGIMES,
        "ip_regimes": IP_REGIMES,
        "current_regimes": CURRENT_REGIMES,
        "actuator_regimes": ACTUATOR_REGIMES,
        "variants": variants,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the deterministic 144-variant reward sweep manifest.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
