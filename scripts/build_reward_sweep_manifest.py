#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BROAD_SHAPE_REGIMES = [
    {"id": "s0", "shape_mean_weight": 3.0, "shape_max_weight": 0.75},
    {"id": "s1", "shape_mean_weight": 5.0, "shape_max_weight": 1.5},
    {"id": "s2", "shape_mean_weight": 8.0, "shape_max_weight": 2.5},
    {"id": "s3", "shape_mean_weight": 12.0, "shape_max_weight": 3.5},
]

BROAD_IP_REGIMES = [
    {"id": "i0", "ip_weight": 2.0},
    {"id": "i1", "ip_weight": 4.0},
    {"id": "i2", "ip_weight": 6.0},
]

BROAD_CURRENT_REGIMES = [
    {"id": "c0", "current_weight": 0.25},
    {"id": "c1", "current_weight": 1.0},
    {"id": "c2", "current_weight": 3.0},
    {"id": "c3", "current_weight": 6.0},
]

BROAD_DERIVATIVE_REGIMES = [
    {"id": "d0", "derivative_weight": 0.05},
    {"id": "d1", "derivative_weight": 0.5},
]

FOCUS_SHAPE_FACTORS = [
    {"id": "sf0", "factor": 0.75},
    {"id": "sf1", "factor": 0.90},
    {"id": "sf2", "factor": 1.10},
    {"id": "sf3", "factor": 1.30},
]

FOCUS_IP_FACTORS = [
    {"id": "if0", "factor": 0.75},
    {"id": "if1", "factor": 0.90},
    {"id": "if2", "factor": 1.10},
    {"id": "if3", "factor": 1.30},
]

FOCUS_CURRENT_FACTORS = [
    {"id": "cf0", "factor": 0.50},
    {"id": "cf1", "factor": 0.80},
    {"id": "cf2", "factor": 1.20},
    {"id": "cf3", "factor": 1.80},
]

FOCUS_DERIVATIVE_FACTORS = [
    {"id": "df0", "factor": 0.50},
    {"id": "df1", "factor": 1.00},
    {"id": "df2", "factor": 1.80},
]

FIXED_REWARD = {
    "shape_mean_scale_m": 0.03,
    "shape_max_scale_m": 0.08,
    "ip_scale_a": 25000.0,
    "reward_scale": 1.0,
    "terminal_reward": -20.0,
    "current_soft_fraction": 1.0,
    "current_bad_fraction": 1.4,
    "derivative_soft_fraction": 1.0,
    "derivative_bad_fraction": 1.4,
    "action_weight": 0.0,
    "delta_action_weight": 0.0,
}


def _rounded(value: float) -> float:
    return float(round(float(value), 8))


def _variant(
    *,
    index: int,
    prefix: str,
    name: str,
    shape_regime: str,
    ip_regime: str,
    current_regime: str,
    derivative_regime: str,
    reward: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "index": int(index),
        "name": name,
        "folder": f"{prefix}{index:03d}_{name}",
        "shape_regime": shape_regime,
        "ip_regime": ip_regime,
        "current_regime": current_regime,
        "derivative_regime": derivative_regime,
        "reward": reward,
    }
    if extra:
        payload.update(extra)
    return payload


def build_broad_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    index = 0
    for shape in BROAD_SHAPE_REGIMES:
        for ip in BROAD_IP_REGIMES:
            for current in BROAD_CURRENT_REGIMES:
                for derivative in BROAD_DERIVATIVE_REGIMES:
                    name = f"{shape['id']}_{ip['id']}_{current['id']}_{derivative['id']}"
                    reward = {
                        **FIXED_REWARD,
                        "shape_mean_weight": shape["shape_mean_weight"],
                        "shape_max_weight": shape["shape_max_weight"],
                        "ip_weight": ip["ip_weight"],
                        "current_weight": current["current_weight"],
                        "derivative_weight": derivative["derivative_weight"],
                    }
                    variants.append(
                        _variant(
                            index=index,
                            prefix="b",
                            name=name,
                            shape_regime=str(shape["id"]),
                            ip_regime=str(ip["id"]),
                            current_regime=str(current["id"]),
                            derivative_regime=str(derivative["id"]),
                            reward=reward,
                        )
                    )
                    index += 1
    return variants


def _find_reward_mapping(raw: Any) -> dict[str, Any]:
    required = {"shape_mean_weight", "shape_max_weight", "ip_weight", "current_weight", "derivative_weight"}
    if isinstance(raw, dict):
        if required.issubset(raw.keys()):
            return raw
        reward = raw.get("reward")
        if isinstance(reward, dict):
            return reward
        for key in ("best_candidate", "best_pareto_candidate", "recommended_reward", "candidate"):
            found = _find_reward_mapping(raw.get(key))
            if found:
                return found
        for value in raw.values():
            found = _find_reward_mapping(value)
            if found:
                return found
    elif isinstance(raw, list):
        for value in raw:
            found = _find_reward_mapping(value)
            if found:
                return found
    return {}


def load_center_reward(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    reward = _find_reward_mapping(raw)
    required = ("shape_mean_weight", "shape_max_weight", "ip_weight", "current_weight", "derivative_weight")
    missing = [key for key in required if key not in reward]
    if missing:
        raise ValueError(f"Center candidate at {path} is missing reward fields: {', '.join(missing)}")
    return {key: float(reward[key]) for key in required}


def build_focused_variants(center_reward: dict[str, float]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    index = 0
    for shape in FOCUS_SHAPE_FACTORS:
        for ip in FOCUS_IP_FACTORS:
            for current in FOCUS_CURRENT_FACTORS:
                for derivative in FOCUS_DERIVATIVE_FACTORS:
                    name = f"{shape['id']}_{ip['id']}_{current['id']}_{derivative['id']}"
                    reward = {
                        **FIXED_REWARD,
                        "shape_mean_weight": _rounded(center_reward["shape_mean_weight"] * float(shape["factor"])),
                        "shape_max_weight": _rounded(center_reward["shape_max_weight"] * float(shape["factor"])),
                        "ip_weight": _rounded(center_reward["ip_weight"] * float(ip["factor"])),
                        "current_weight": _rounded(center_reward["current_weight"] * float(current["factor"])),
                        "derivative_weight": _rounded(center_reward["derivative_weight"] * float(derivative["factor"])),
                    }
                    variants.append(
                        _variant(
                            index=index,
                            prefix="f",
                            name=name,
                            shape_regime=str(shape["id"]),
                            ip_regime=str(ip["id"]),
                            current_regime=str(current["id"]),
                            derivative_regime=str(derivative["id"]),
                            reward=reward,
                            extra={
                                "shape_factor": shape["factor"],
                                "ip_factor": ip["factor"],
                                "current_factor": current["factor"],
                                "derivative_factor": derivative["factor"],
                            },
                        )
                    )
                    index += 1
    return variants


def _select_evenly(variants: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("variant count must be positive")
    if count >= len(variants):
        return variants
    if count == 1:
        selected = [variants[0]]
    else:
        indices = [round(i * (len(variants) - 1) / (count - 1)) for i in range(count)]
        selected = [variants[index] for index in indices]
    reindexed: list[dict[str, Any]] = []
    for new_index, variant in enumerate(selected):
        item = dict(variant)
        item["source_index"] = int(variant["index"])
        item["index"] = int(new_index)
        old_folder = str(variant["folder"])
        prefix = old_folder[0]
        item["folder"] = f"{prefix}{new_index:03d}_{variant['name']}"
        reindexed.append(item)
    return reindexed


def build_variants(sweep_pass: str = "broad", center_reward: dict[str, float] | None = None, variant_budget: int | None = None) -> list[dict[str, Any]]:
    if sweep_pass == "broad":
        variants = build_broad_variants()
    elif sweep_pass == "focused":
        if center_reward is None:
            raise ValueError("focused sweep requires center_reward")
        variants = build_focused_variants(center_reward)
    else:
        raise ValueError(f"Unknown reward sweep pass: {sweep_pass}")
    return _select_evenly(variants, int(variant_budget)) if variant_budget is not None else variants


def build_manifest(
    sweep_pass: str = "broad",
    center_reward: dict[str, float] | None = None,
    *,
    variant_budget: int | None = None,
    runs_per_array_task: int | None = None,
    array_task_count: int | None = None,
) -> dict[str, Any]:
    variants = build_variants(sweep_pass=sweep_pass, center_reward=center_reward, variant_budget=variant_budget)
    runs_per_task = int(runs_per_array_task if runs_per_array_task is not None else (1 if sweep_pass == "broad" else 2))
    if runs_per_task <= 0:
        raise ValueError("runs_per_array_task must be positive")
    if array_task_count is None:
        if len(variants) % runs_per_task != 0:
            raise ValueError(f"{len(variants)} variants is not divisible by {runs_per_task} runs per task")
        array_task_count = len(variants) // runs_per_task
    array_task_count = int(array_task_count)
    expected = array_task_count * runs_per_task
    if len(variants) != expected:
        raise ValueError(f"{sweep_pass} manifest has {len(variants)} variants, expected {expected}")
    manifest: dict[str, Any] = {
        "description": f"{len(variants)}-run T15 CSV segmented-profile legal-actuator reward sweep ({sweep_pass})",
        "sweep_pass": sweep_pass,
        "variant_count": len(variants),
        "runs_per_array_task": runs_per_task,
        "array_task_count": array_task_count,
        "fixed_reward": FIXED_REWARD,
        "variants": variants,
    }
    if sweep_pass == "broad":
        manifest.update(
            {
                "shape_regimes": BROAD_SHAPE_REGIMES,
                "ip_regimes": BROAD_IP_REGIMES,
                "current_regimes": BROAD_CURRENT_REGIMES,
                "derivative_regimes": BROAD_DERIVATIVE_REGIMES,
            }
        )
    else:
        manifest.update(
            {
                "center_reward": center_reward,
                "shape_factors": FOCUS_SHAPE_FACTORS,
                "ip_factors": FOCUS_IP_FACTORS,
                "current_factors": FOCUS_CURRENT_FACTORS,
                "derivative_factors": FOCUS_DERIVATIVE_FACTORS,
            }
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write deterministic two-pass reward sweep manifests.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pass", dest="sweep_pass", choices=("broad", "focused"), default="broad")
    parser.add_argument("--center", type=Path, default=None, help="physical_best_candidate.json for focused pass")
    parser.add_argument("--variant-budget", type=int, default=None)
    parser.add_argument("--runs-per-array-task", type=int, default=None)
    parser.add_argument("--array-task-count", type=int, default=None)
    args = parser.parse_args(argv)
    center_reward = load_center_reward(args.center) if args.sweep_pass == "focused" and args.center is not None else None
    manifest = build_manifest(
        sweep_pass=args.sweep_pass,
        center_reward=center_reward,
        variant_budget=args.variant_budget,
        runs_per_array_task=args.runs_per_array_task,
        array_task_count=args.array_task_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
