#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_LEGAL = "legal"
PROFILE_CURRENT_CONSTRAINT = "current_constraint"
PROFILE_FIXED_HORIZON = "fixed_horizon"
PROFILE_SATURATION = "saturation"

BROAD_SHAPE_REGIMES = [
    {"id": "s0", "shape_mean_weight": 1.5, "shape_max_weight": 0.375},
    {"id": "s1", "shape_mean_weight": 2.25, "shape_max_weight": 0.5625},
    {"id": "s2", "shape_mean_weight": 3.5, "shape_max_weight": 0.9},
    {"id": "s3", "shape_mean_weight": 5.5, "shape_max_weight": 1.5},
]

BROAD_IP_REGIMES = [
    {"id": "i0", "ip_weight": 3.0},
    {"id": "i1", "ip_weight": 5.0},
    {"id": "i2", "ip_weight": 7.0},
]

BROAD_ACTUATOR_REGIMES = [
    {"id": "a0", "current_weight": 0.3, "derivative_weight": 0.3},
    {"id": "a1", "current_weight": 1.2, "derivative_weight": 0.9},
    {"id": "a2", "current_weight": 2.5, "derivative_weight": 1.5},
]

FOCUS_SHAPE_FACTORS = [
    {"id": "sf0", "factor": 0.75},
    {"id": "sf1", "factor": 1.00},
    {"id": "sf2", "factor": 1.25},
]

FOCUS_IP_FACTORS = [
    {"id": "if0", "factor": 0.80},
    {"id": "if1", "factor": 1.00},
    {"id": "if2", "factor": 1.20},
]

FOCUS_ACTUATOR_FACTORS = [
    {"id": "af0", "current_factor": 0.60, "derivative_factor": 0.70},
    {"id": "af1", "current_factor": 1.00, "derivative_factor": 1.00},
    {"id": "af2", "current_factor": 1.60, "derivative_factor": 1.40},
    {"id": "af3", "current_factor": 2.20, "derivative_factor": 1.80},
]

CURRENT_CONSTRAINT_BROAD_SHAPE_REGIMES = [
    {"id": "s0", "shape_mean_weight": 1.0, "shape_max_weight": 0.25},
    {"id": "s1", "shape_mean_weight": 2.0, "shape_max_weight": 0.50},
    {"id": "s2", "shape_mean_weight": 4.0, "shape_max_weight": 1.00},
]

CURRENT_CONSTRAINT_BROAD_IP_REGIMES = [
    {"id": "i0", "ip_weight": 0.75},
    {"id": "i1", "ip_weight": 1.50},
    {"id": "i2", "ip_weight": 3.00},
]

CURRENT_CONSTRAINT_BROAD_SAFETY_REGIMES = [
    {
        "id": "a0",
        "current_weight": 2.0,
        "current_soft_fraction": 0.90,
        "derivative_weight": 0.10,
        "derivative_soft_fraction": 0.90,
        "terminal_remaining_cost": 25000.0,
    },
    {
        "id": "a1",
        "current_weight": 4.0,
        "current_soft_fraction": 0.90,
        "derivative_weight": 0.25,
        "derivative_soft_fraction": 0.90,
        "terminal_remaining_cost": 50000.0,
    },
    {
        "id": "a2",
        "current_weight": 6.0,
        "current_soft_fraction": 0.85,
        "derivative_weight": 0.35,
        "derivative_soft_fraction": 0.85,
        "terminal_remaining_cost": 100000.0,
    },
    {
        "id": "a3",
        "current_weight": 8.0,
        "current_soft_fraction": 0.85,
        "derivative_weight": 0.50,
        "derivative_soft_fraction": 0.85,
        "terminal_remaining_cost": 200000.0,
    },
]

CURRENT_CONSTRAINT_FOCUS_SHAPE_FACTORS = [
    {"id": "sf0", "factor": 0.75},
    {"id": "sf1", "factor": 1.00},
    {"id": "sf2", "factor": 1.25},
]

CURRENT_CONSTRAINT_FOCUS_IP_FACTORS = [
    {"id": "if0", "factor": 0.70},
    {"id": "if1", "factor": 1.00},
    {"id": "if2", "factor": 1.30},
]

CURRENT_CONSTRAINT_FOCUS_SAFETY_FACTORS = [
    {
        "id": "af0",
        "current_factor": 0.75,
        "derivative_factor": 0.75,
        "terminal_remaining_factor": 0.75,
        "current_soft_fraction": 0.90,
        "derivative_soft_fraction": 0.90,
    },
    {
        "id": "af1",
        "current_factor": 1.00,
        "derivative_factor": 1.00,
        "terminal_remaining_factor": 1.00,
        "current_soft_fraction": "center",
        "derivative_soft_fraction": "center",
    },
    {
        "id": "af2",
        "current_factor": 1.35,
        "derivative_factor": 1.35,
        "terminal_remaining_factor": 1.50,
        "current_soft_fraction": 0.85,
        "derivative_soft_fraction": 0.85,
    },
    {
        "id": "af3",
        "current_factor": 1.80,
        "derivative_factor": 1.80,
        "terminal_remaining_factor": 2.00,
        "current_soft_fraction": 0.85,
        "derivative_soft_fraction": 0.85,
    },
]

FIXED_REWARD = {
    "shape_mean_scale_m": 0.03,
    "shape_max_scale_m": 0.08,
    "ip_scale_a": 25000.0,
    "reward_scale": 1.0,
    "terminal_reward": -20.0,
    "terminal_remaining_cost": 0.0,
    "current_soft_fraction": 1.0,
    "current_bad_fraction": 1.4,
    "derivative_soft_fraction": 1.0,
    "derivative_bad_fraction": 1.4,
    "action_weight": 0.0,
    "delta_action_weight": 0.0,
}

CURRENT_CONSTRAINT_FIXED_REWARD = {
    "shape_mean_scale_m": 0.03,
    "shape_max_scale_m": 0.08,
    "ip_scale_a": 25000.0,
    "reward_scale": 1.0,
    "terminal_reward": -20.0,
    "terminal_remaining_cost": 50000.0,
    "current_bad_fraction": 1.20,
    "derivative_bad_fraction": 1.20,
    "action_weight": 0.01,
    "delta_action_weight": 0.025,
}

CURRENT_CONSTRAINT_FIXED_SIM = {
    "terminate_on_current_limit": True,
    "current_termination_over_limit_a": 50000.0,
    "current_termination_grace_steps": 50,
    "current_hard_termination_fraction": 1.40,
}

FIXED_HORIZON_BROAD_SHAPE_REGIMES = [
    {"id": "s0", "shape_mean_weight": 1.0, "shape_max_weight": 0.25},
    {"id": "s1", "shape_mean_weight": 2.0, "shape_max_weight": 0.50},
    {"id": "s2", "shape_mean_weight": 4.0, "shape_max_weight": 1.00},
]

FIXED_HORIZON_BROAD_IP_REGIMES = [
    {"id": "i0", "ip_weight": 0.75},
    {"id": "i1", "ip_weight": 1.50},
    {"id": "i2", "ip_weight": 3.00},
]

FIXED_HORIZON_BROAD_SAFETY_REGIMES = [
    {
        "id": "a0",
        "current_weight": 1.0,
        "current_soft_fraction": 0.90,
        "derivative_weight": 0.10,
        "derivative_soft_fraction": 0.90,
    },
    {
        "id": "a1",
        "current_weight": 2.0,
        "current_soft_fraction": 0.90,
        "derivative_weight": 0.25,
        "derivative_soft_fraction": 0.85,
    },
    {
        "id": "a2",
        "current_weight": 4.0,
        "current_soft_fraction": 0.85,
        "derivative_weight": 0.50,
        "derivative_soft_fraction": 0.85,
    },
    {
        "id": "a3",
        "current_weight": 6.0,
        "current_soft_fraction": 0.80,
        "derivative_weight": 0.75,
        "derivative_soft_fraction": 0.80,
    },
]

FIXED_HORIZON_FOCUS_SHAPE_FACTORS = [
    {"id": "sf0", "factor": 0.75},
    {"id": "sf1", "factor": 1.00},
    {"id": "sf2", "factor": 1.25},
]

FIXED_HORIZON_FOCUS_IP_FACTORS = [
    {"id": "if0", "factor": 0.70},
    {"id": "if1", "factor": 1.00},
    {"id": "if2", "factor": 1.30},
]

FIXED_HORIZON_FOCUS_SAFETY_FACTORS = [
    {
        "id": "af0",
        "current_factor": 0.70,
        "derivative_factor": 0.70,
        "current_soft_fraction": 0.90,
        "derivative_soft_fraction": 0.90,
    },
    {
        "id": "af1",
        "current_factor": 1.00,
        "derivative_factor": 1.00,
        "current_soft_fraction": "center",
        "derivative_soft_fraction": "center",
    },
    {
        "id": "af2",
        "current_factor": 1.35,
        "derivative_factor": 1.35,
        "current_soft_fraction": 0.85,
        "derivative_soft_fraction": 0.85,
    },
    {
        "id": "af3",
        "current_factor": 1.80,
        "derivative_factor": 1.80,
        "current_soft_fraction": 0.80,
        "derivative_soft_fraction": 0.80,
    },
]

FIXED_HORIZON_FIXED_REWARD = {
    "shape_mean_scale_m": 0.03,
    "shape_max_scale_m": 0.08,
    "ip_scale_a": 25000.0,
    "reward_scale": 1.0,
    "terminal_reward": -20.0,
    "terminal_remaining_cost": 0.0,
    "current_bad_fraction": 1.20,
    "derivative_bad_fraction": 1.20,
    "action_weight": 0.01,
    "delta_action_weight": 0.025,
}

FIXED_HORIZON_FIXED_SIM = {
    "terminate_on_boundary_loss": False,
    "terminate_on_current_limit": False,
}

SATURATION_BROAD_SHAPE_REGIMES = [
    {"id": "s0", "shape_mean_weight": 0.75, "shape_max_weight": 0.1875},
    {"id": "s1", "shape_mean_weight": 1.5, "shape_max_weight": 0.375},
    {"id": "s2", "shape_mean_weight": 2.5, "shape_max_weight": 0.625},
]

SATURATION_BROAD_IP_REGIMES = [
    {"id": "i0", "ip_weight": 0.75},
    {"id": "i1", "ip_weight": 1.5},
    {"id": "i2", "ip_weight": 3.0},
]

SATURATION_BROAD_SAFETY_REGIMES = [
    {"id": "a0", "current_weight": 2.0, "derivative_weight": 0.25, "actuator_saturation_weight": 2.0},
    {"id": "a1", "current_weight": 4.2, "derivative_weight": 0.525, "actuator_saturation_weight": 4.0},
    {"id": "a2", "current_weight": 6.0, "derivative_weight": 0.75, "actuator_saturation_weight": 8.0},
    {"id": "a3", "current_weight": 8.0, "derivative_weight": 1.0, "actuator_saturation_weight": 12.0},
]

SATURATION_FIXED_REWARD = {
    "shape_mean_scale_m": 0.03,
    "shape_max_scale_m": 0.08,
    "ip_scale_a": 25000.0,
    "reward_scale": 1.0,
    "terminal_reward": -20.0,
    "terminal_remaining_cost": 0.0,
    "current_soft_fraction": 0.90,
    "current_bad_fraction": 1.20,
    "derivative_soft_fraction": 0.90,
    "derivative_bad_fraction": 1.20,
    "action_weight": 0.01,
    "delta_action_weight": 0.025,
}

SATURATION_FIXED_SIM = {
    "terminate_on_boundary_loss": False,
    "terminate_on_current_limit": False,
    "current_saturation_fraction": 1.15,
}


def _rounded(value: float) -> float:
    return float(round(float(value), 8))


def _check_profile(profile: str) -> str:
    if profile not in {PROFILE_LEGAL, PROFILE_CURRENT_CONSTRAINT, PROFILE_FIXED_HORIZON, PROFILE_SATURATION}:
        raise ValueError(f"Unknown reward sweep profile: {profile}")
    return profile


def _fixed_reward(profile: str) -> dict[str, Any]:
    profile = _check_profile(profile)
    if profile == PROFILE_CURRENT_CONSTRAINT:
        return dict(CURRENT_CONSTRAINT_FIXED_REWARD)
    if profile == PROFILE_FIXED_HORIZON:
        return dict(FIXED_HORIZON_FIXED_REWARD)
    if profile == PROFILE_SATURATION:
        return dict(SATURATION_FIXED_REWARD)
    return dict(FIXED_REWARD)


def _fixed_sim(profile: str) -> dict[str, Any]:
    profile = _check_profile(profile)
    if profile == PROFILE_CURRENT_CONSTRAINT:
        return dict(CURRENT_CONSTRAINT_FIXED_SIM)
    if profile == PROFILE_FIXED_HORIZON:
        return dict(FIXED_HORIZON_FIXED_SIM)
    if profile == PROFILE_SATURATION:
        return dict(SATURATION_FIXED_SIM)
    return {}


def _broad_shape_regimes(profile: str) -> list[dict[str, Any]]:
    profile = _check_profile(profile)
    if profile == PROFILE_CURRENT_CONSTRAINT:
        return CURRENT_CONSTRAINT_BROAD_SHAPE_REGIMES
    if profile == PROFILE_FIXED_HORIZON:
        return FIXED_HORIZON_BROAD_SHAPE_REGIMES
    if profile == PROFILE_SATURATION:
        return SATURATION_BROAD_SHAPE_REGIMES
    return BROAD_SHAPE_REGIMES


def _broad_ip_regimes(profile: str) -> list[dict[str, Any]]:
    profile = _check_profile(profile)
    if profile == PROFILE_CURRENT_CONSTRAINT:
        return CURRENT_CONSTRAINT_BROAD_IP_REGIMES
    if profile == PROFILE_FIXED_HORIZON:
        return FIXED_HORIZON_BROAD_IP_REGIMES
    if profile == PROFILE_SATURATION:
        return SATURATION_BROAD_IP_REGIMES
    return BROAD_IP_REGIMES


def _broad_safety_regimes(profile: str) -> list[dict[str, Any]]:
    profile = _check_profile(profile)
    if profile == PROFILE_CURRENT_CONSTRAINT:
        return CURRENT_CONSTRAINT_BROAD_SAFETY_REGIMES
    if profile == PROFILE_FIXED_HORIZON:
        return FIXED_HORIZON_BROAD_SAFETY_REGIMES
    if profile == PROFILE_SATURATION:
        return SATURATION_BROAD_SAFETY_REGIMES
    return BROAD_ACTUATOR_REGIMES


def _focus_shape_factors(profile: str) -> list[dict[str, Any]]:
    profile = _check_profile(profile)
    if profile == PROFILE_CURRENT_CONSTRAINT:
        return CURRENT_CONSTRAINT_FOCUS_SHAPE_FACTORS
    if profile == PROFILE_FIXED_HORIZON:
        return FIXED_HORIZON_FOCUS_SHAPE_FACTORS
    return FOCUS_SHAPE_FACTORS


def _focus_ip_factors(profile: str) -> list[dict[str, Any]]:
    profile = _check_profile(profile)
    if profile == PROFILE_CURRENT_CONSTRAINT:
        return CURRENT_CONSTRAINT_FOCUS_IP_FACTORS
    if profile == PROFILE_FIXED_HORIZON:
        return FIXED_HORIZON_FOCUS_IP_FACTORS
    return FOCUS_IP_FACTORS


def _focus_safety_factors(profile: str) -> list[dict[str, Any]]:
    profile = _check_profile(profile)
    if profile == PROFILE_CURRENT_CONSTRAINT:
        return CURRENT_CONSTRAINT_FOCUS_SAFETY_FACTORS
    if profile == PROFILE_FIXED_HORIZON:
        return FIXED_HORIZON_FOCUS_SAFETY_FACTORS
    return FOCUS_ACTUATOR_FACTORS


def _soft_fraction(value: Any, center_reward: dict[str, float], key: str, default: float) -> float:
    if value == "center":
        return float(center_reward.get(key, default))
    return float(value)


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


def build_broad_variants(profile: str = PROFILE_LEGAL) -> list[dict[str, Any]]:
    profile = _check_profile(profile)
    variants: list[dict[str, Any]] = []
    index = 0
    fixed_reward = _fixed_reward(profile)
    fixed_sim = _fixed_sim(profile)
    prefix = "s" if profile == PROFILE_SATURATION else "b"
    for shape in _broad_shape_regimes(profile):
        for ip in _broad_ip_regimes(profile):
            for actuator in _broad_safety_regimes(profile):
                name = f"{shape['id']}_{ip['id']}_{actuator['id']}"
                reward = {
                    **fixed_reward,
                    "shape_mean_weight": shape["shape_mean_weight"],
                    "shape_max_weight": shape["shape_max_weight"],
                    "ip_weight": ip["ip_weight"],
                    "current_weight": actuator["current_weight"],
                    "derivative_weight": actuator["derivative_weight"],
                }
                if "current_soft_fraction" in actuator:
                    reward["current_soft_fraction"] = actuator["current_soft_fraction"]
                if "derivative_soft_fraction" in actuator:
                    reward["derivative_soft_fraction"] = actuator["derivative_soft_fraction"]
                if "terminal_remaining_cost" in actuator:
                    reward["terminal_remaining_cost"] = actuator["terminal_remaining_cost"]
                if "actuator_saturation_weight" in actuator:
                    reward["actuator_saturation_weight"] = actuator["actuator_saturation_weight"]
                extra = {"actuator_regime": actuator["id"]}
                if fixed_sim:
                    extra["sim"] = fixed_sim
                variants.append(
                    _variant(
                        index=index,
                        prefix=prefix,
                        name=name,
                        shape_regime=str(shape["id"]),
                        ip_regime=str(ip["id"]),
                        current_regime=str(actuator["id"]),
                        derivative_regime=str(actuator["id"]),
                        reward=reward,
                        extra=extra,
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
    optional = ("current_soft_fraction", "derivative_soft_fraction", "terminal_reward", "terminal_remaining_cost")
    return {key: float(reward[key]) for key in (*required, *optional) if key in reward}


def build_focused_variants(center_reward: dict[str, float], profile: str = PROFILE_LEGAL) -> list[dict[str, Any]]:
    profile = _check_profile(profile)
    variants: list[dict[str, Any]] = []
    index = 0
    fixed_reward = _fixed_reward(profile)
    fixed_sim = _fixed_sim(profile)
    for shape in _focus_shape_factors(profile):
        for ip in _focus_ip_factors(profile):
            for actuator in _focus_safety_factors(profile):
                name = f"{shape['id']}_{ip['id']}_{actuator['id']}"
                current_soft = _soft_fraction(
                    actuator.get("current_soft_fraction", fixed_reward.get("current_soft_fraction", 1.0)),
                    center_reward,
                    "current_soft_fraction",
                    float(fixed_reward.get("current_soft_fraction", 1.0)),
                )
                derivative_soft = _soft_fraction(
                    actuator.get("derivative_soft_fraction", fixed_reward.get("derivative_soft_fraction", 1.0)),
                    center_reward,
                    "derivative_soft_fraction",
                    float(fixed_reward.get("derivative_soft_fraction", 1.0)),
                )
                reward = {
                    **fixed_reward,
                    "shape_mean_weight": _rounded(center_reward["shape_mean_weight"] * float(shape["factor"])),
                    "shape_max_weight": _rounded(center_reward["shape_max_weight"] * float(shape["factor"])),
                    "ip_weight": _rounded(center_reward["ip_weight"] * float(ip["factor"])),
                    "current_weight": _rounded(center_reward["current_weight"] * float(actuator["current_factor"])),
                    "derivative_weight": _rounded(center_reward["derivative_weight"] * float(actuator["derivative_factor"])),
                    "terminal_reward": _rounded(center_reward.get("terminal_reward", fixed_reward["terminal_reward"])),
                    "terminal_remaining_cost": _rounded(
                        center_reward.get("terminal_remaining_cost", fixed_reward["terminal_remaining_cost"])
                        * float(actuator.get("terminal_remaining_factor", 1.0))
                    ),
                    "current_soft_fraction": current_soft,
                    "derivative_soft_fraction": derivative_soft,
                }
                extra = {
                    "actuator_regime": actuator["id"],
                    "shape_factor": shape["factor"],
                    "ip_factor": ip["factor"],
                    "current_factor": actuator["current_factor"],
                    "derivative_factor": actuator["derivative_factor"],
                    "terminal_remaining_factor": actuator.get("terminal_remaining_factor", 1.0),
                }
                if fixed_sim:
                    extra["sim"] = fixed_sim
                variants.append(
                    _variant(
                        index=index,
                        prefix="f",
                        name=name,
                        shape_regime=str(shape["id"]),
                        ip_regime=str(ip["id"]),
                        current_regime=str(actuator["id"]),
                        derivative_regime=str(actuator["id"]),
                        reward=reward,
                        extra=extra,
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


def build_variants(
    sweep_pass: str = "broad",
    center_reward: dict[str, float] | None = None,
    variant_budget: int | None = None,
    *,
    profile: str = PROFILE_LEGAL,
) -> list[dict[str, Any]]:
    profile = _check_profile(profile)
    if sweep_pass == "broad":
        variants = build_broad_variants(profile)
    elif sweep_pass == "focused":
        if profile == PROFILE_SATURATION:
            raise ValueError("saturation reward sweep profile is one-pass only")
        if center_reward is None:
            raise ValueError("focused sweep requires center_reward")
        variants = build_focused_variants(center_reward, profile)
    else:
        raise ValueError(f"Unknown reward sweep pass: {sweep_pass}")
    return _select_evenly(variants, int(variant_budget)) if variant_budget is not None else variants


def build_manifest(
    sweep_pass: str = "broad",
    center_reward: dict[str, float] | None = None,
    *,
    profile: str = PROFILE_LEGAL,
    variant_budget: int | None = None,
    runs_per_array_task: int | None = None,
    array_task_count: int | None = None,
) -> dict[str, Any]:
    profile = _check_profile(profile)
    variants = build_variants(sweep_pass=sweep_pass, center_reward=center_reward, variant_budget=variant_budget, profile=profile)
    runs_per_task = int(runs_per_array_task if runs_per_array_task is not None else 3)
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
        "description": f"{len(variants)}-run T15 CSV segmented-profile {profile} reward sweep ({sweep_pass})",
        "profile": profile,
        "sweep_pass": sweep_pass,
        "variant_count": len(variants),
        "runs_per_array_task": runs_per_task,
        "array_task_count": array_task_count,
        "fixed_reward": _fixed_reward(profile),
        "fixed_sim": _fixed_sim(profile),
        "variants": variants,
    }
    if sweep_pass == "broad":
        manifest.update(
            {
                "shape_regimes": _broad_shape_regimes(profile),
                "ip_regimes": _broad_ip_regimes(profile),
                "actuator_regimes": _broad_safety_regimes(profile),
            }
        )
    else:
        manifest.update(
            {
                "center_reward": center_reward,
                "shape_factors": _focus_shape_factors(profile),
                "ip_factors": _focus_ip_factors(profile),
                "actuator_factors": _focus_safety_factors(profile),
            }
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write deterministic two-pass reward sweep manifests.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pass", dest="sweep_pass", choices=("broad", "focused"), default="broad")
    parser.add_argument("--profile", choices=(PROFILE_LEGAL, PROFILE_CURRENT_CONSTRAINT, PROFILE_FIXED_HORIZON, PROFILE_SATURATION), default=PROFILE_LEGAL)
    parser.add_argument("--center", type=Path, default=None, help="physical_best_candidate.json for focused pass")
    parser.add_argument("--variant-budget", type=int, default=None)
    parser.add_argument("--runs-per-array-task", type=int, default=None)
    parser.add_argument("--array-task-count", type=int, default=None)
    args = parser.parse_args(argv)
    center_reward = load_center_reward(args.center) if args.sweep_pass == "focused" and args.center is not None else None
    manifest = build_manifest(
        sweep_pass=args.sweep_pass,
        center_reward=center_reward,
        profile=args.profile,
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
