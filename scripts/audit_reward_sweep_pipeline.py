#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.build_reward_sweep_manifest import build_manifest
except ModuleNotFoundError:  # pragma: no cover - used when run as python3 scripts/...
    from build_reward_sweep_manifest import build_manifest


ACTIVE_JOB_FILES = [
    "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_pass1_broad.sbatch",
    "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_pass2_focused.sbatch",
    "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_current_constraint_pass1.sbatch",
    "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_current_constraint_pass2.sbatch",
    "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_fixed_horizon_pass1.sbatch",
    "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_fixed_horizon_pass2.sbatch",
    "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_saturation_pass1.sbatch",
    "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_saturation_pass2.sbatch",
    "jobs/aggregate_t15_reward_sweep_pass1.sbatch",
    "jobs/aggregate_t15_reward_sweep_final.sbatch",
]


def audit(root: Path) -> list[str]:
    issues: list[str] = []
    policy_pipeline = (root / "tokamak_rl_v2/training/policy_pipeline.py").read_text(encoding="utf-8")
    builder = (root / "scripts/build_t15_csv_initial_state_library.py").read_text(encoding="utf-8")
    runner = (root / "scripts/run_reward_sweep_candidate.py").read_text(encoding="utf-8")

    if "validate_split_nonoverlap" in policy_pipeline:
        issues.append("production policy preflight still references validate_split_nonoverlap")
    if "validate_split_nonoverlap" in builder:
        issues.append("CSV initial-state builder still references validate_split_nonoverlap")

    broad = build_manifest("broad", runs_per_array_task=3, array_task_count=12)
    if int(broad.get("variant_count", -1)) != 36:
        issues.append(f"pass1 broad manifest has {broad.get('variant_count')} variants, expected 36")
    center = {
        "shape_mean_weight": 2.25,
        "shape_max_weight": 0.5625,
        "ip_weight": 5.0,
        "current_weight": 1.2,
        "derivative_weight": 0.9,
    }
    focused = build_manifest("focused", center, runs_per_array_task=3, array_task_count=12)
    if int(focused.get("variant_count", -1)) != 36:
        issues.append(f"pass2 focused manifest has {focused.get('variant_count')} variants, expected 36")

    if "--no-save-checkpoints" not in runner or "--no-export" not in runner or "--reward-sweep-mode" not in runner:
        issues.append("reward sweep candidate runner is missing no-checkpoint/no-export/sweep-mode flags")
    if 'shutil.rmtree(output_dir / "exports"' not in runner or 'shutil.rmtree(output_dir / "checkpoints"' not in runner:
        issues.append("reward sweep candidate runner does not clean exports/checkpoints")

    for rel in ACTIVE_JOB_FILES:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        if "reward288" in text:
            issues.append(f"{rel} still contains stale reward288 naming")
        if "--eval-max-steps 500" in text or "--eval-max-steps \"500\"" in text:
            issues.append(f"{rel} still hardcodes 500-step eval")
        if "build_reward_sweep_manifest.py" in text and "aggregate_t15_reward_sweep_pass1" not in rel:
            issues.append(f"{rel} writes a manifest inside an array task; manifests should be prepared upstream")
        if "pass1" in rel and "aggregate" not in rel and "missing pass1 reward-sweep manifest" not in text:
            issues.append(f"{rel} does not require the prebuilt pass1 manifest")
        if "pass2" in rel and "aggregate" not in rel and "missing pass2 reward-sweep manifest" not in text:
            issues.append(f"{rel} does not require the aggregate-built pass2 manifest")
        if "sweep_t15_csv_segmented_profile_rewards_12gpu" in rel and "REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}" not in text:
            issues.append(f"{rel} must default REPLAY_CAPACITY_EPISODES to 288 for 2000-step A100 sweeps")
        if rel.endswith("aggregate_t15_reward_sweep_pass1.sbatch") and "pass2_focused/variants.json" not in text:
            issues.append(f"{rel} does not build the focused pass2 manifest")

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cheap static audit for the two-pass reward-sweep pipeline.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    issues = audit(args.root.resolve())
    if issues:
        print("reward sweep pipeline audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 2
    print("reward sweep pipeline audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
