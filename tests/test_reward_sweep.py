from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

from scripts.aggregate_reward_sweep import aggregate, score_eval_row, write_outputs
from scripts.audit_reward_sweep_pipeline import audit as audit_reward_sweep_pipeline
from scripts.build_reward_sweep_manifest import (
    PROFILE_CURRENT_CONSTRAINT,
    PROFILE_FIXED_HORIZON,
    PROFILE_SATURATION,
    PROFILE_TCV_DERIVATIVE,
    PROFILE_TCV_DELTA_JDOT,
    PROFILE_TCV_DELTA_NO_TERMINATION,
    PROFILE_TCV_DELTA_TERMINATION_F002,
    PROFILE_TCV_QUALITY,
    build_manifest,
    build_variants,
)
from scripts.build_reward_sweep_rerun_manifest import build as build_rerun_manifest
from scripts.run_reward_sweep_candidate import run_candidate
from scripts.summarize_two_pass_reward_sweep_physical import summarize_two_pass
from scripts.summarize_reward_sweep_physical import summarize
from scripts.submit_saturation_reward_sweep import submit_onepass
from scripts.submit_saturation_two_pass_reward_sweep import main as submit_saturation_two_pass_main
from scripts.submit_two_pass_reward_sweep import submit_chain


ROOT = Path(__file__).resolve().parents[1]
PASS1_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_96gpu_pass1_broad.sbatch"
PASS2_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_96gpu_pass2_focused.sbatch"
PASS1_12_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_pass1_broad.sbatch"
PASS2_12_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_pass2_focused.sbatch"
PASS1_CURRENT_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_current_constraint_pass1.sbatch"
PASS2_CURRENT_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_current_constraint_pass2.sbatch"
PASS1_FIXED_HORIZON_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_fixed_horizon_pass1.sbatch"
PASS2_FIXED_HORIZON_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_fixed_horizon_pass2.sbatch"
SATURATION_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_saturation_onepass.sbatch"
SATURATION_AGG_JOB = ROOT / "jobs/aggregate_t15_reward_sweep_onepass.sbatch"
SATURATION_PASS1_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_saturation_pass1.sbatch"
SATURATION_PASS2_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_saturation_pass2.sbatch"
TCV_QUALITY_PASS1_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_tcv_quality_pass1.sbatch"
TCV_QUALITY_PASS2_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_tcv_quality_pass2.sbatch"
RERUN_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_rerun_1gpu.sbatch"


def test_reward_sweep_broad_manifest_has_36_explicit_unique_variants() -> None:
    variants = build_variants("broad")
    assert len(variants) == 36
    assert [variant["index"] for variant in variants] == list(range(36))
    assert len({variant["name"] for variant in variants}) == 36
    assert len({variant["folder"] for variant in variants}) == 36
    assert variants[0]["folder"] == "b000_s0_i0_a0"
    assert variants[-1]["folder"] == "b035_s3_i2_a2"
    actuator_by_regime = {
        variant["actuator_regime"]: (
            variant["reward"]["current_weight"],
            variant["reward"]["derivative_weight"],
            variant["reward"]["current_soft_fraction"],
            variant["reward"]["current_bad_fraction"],
            variant["reward"]["derivative_soft_fraction"],
            variant["reward"]["derivative_bad_fraction"],
            variant["reward"]["action_weight"],
            variant["reward"]["delta_action_weight"],
        )
        for variant in variants
        if variant["shape_regime"] == "s0" and variant["ip_regime"] == "i0"
    }
    assert actuator_by_regime == {
        "a0": (0.3, 0.3, 1.0, 1.4, 1.0, 1.4, 0.0, 0.0),
        "a1": (1.2, 0.9, 1.0, 1.4, 1.0, 1.4, 0.0, 0.0),
        "a2": (2.5, 1.5, 1.0, 1.4, 1.0, 1.4, 0.0, 0.0),
    }


def test_reward_sweep_focused_manifest_has_36_explicit_unique_variants() -> None:
    center = {
        "shape_mean_weight": 2.25,
        "shape_max_weight": 0.5625,
        "ip_weight": 5.0,
        "current_weight": 1.2,
        "derivative_weight": 0.9,
    }
    variants = build_variants("focused", center)
    assert len(variants) == 36
    assert [variant["index"] for variant in variants] == list(range(36))
    assert len({variant["folder"] for variant in variants}) == 36
    assert variants[0]["folder"] == "f000_sf0_if0_af0"
    assert variants[-1]["folder"] == "f035_sf2_if2_af3"
    assert variants[0]["reward"]["shape_mean_weight"] == 1.6875
    assert variants[0]["reward"]["ip_weight"] == 4.0
    assert variants[0]["reward"]["current_weight"] == 0.72
    assert variants[0]["reward"]["derivative_weight"] == 0.63
    assert variants[0]["reward"]["current_soft_fraction"] == 1.0
    assert variants[0]["reward"]["current_bad_fraction"] == 1.4
    assert variants[0]["reward"]["action_weight"] == 0.0
    assert variants[0]["reward"]["delta_action_weight"] == 0.0


def test_current_constraint_broad_manifest_has_36_and_current_termination() -> None:
    variants = build_variants("broad", profile=PROFILE_CURRENT_CONSTRAINT)
    assert len(variants) == 36
    assert variants[0]["folder"] == "b000_s0_i0_a0"
    assert variants[-1]["folder"] == "b035_s2_i2_a3"
    first = variants[0]
    assert first["reward"]["shape_mean_weight"] == 1.0
    assert first["reward"]["shape_max_weight"] == 0.25
    assert first["reward"]["ip_weight"] == 0.75
    assert first["reward"]["current_weight"] == 2.0
    assert first["reward"]["current_soft_fraction"] == 0.90
    assert first["reward"]["current_bad_fraction"] == 1.20
    assert first["reward"]["derivative_weight"] == 0.10
    assert first["reward"]["derivative_soft_fraction"] == 0.90
    assert first["reward"]["derivative_bad_fraction"] == 1.20
    assert first["reward"]["terminal_reward"] == -20.0
    assert first["reward"]["terminal_remaining_cost"] == 25000.0
    assert first["reward"]["action_weight"] == 0.01
    assert first["reward"]["delta_action_weight"] == 0.025
    assert first["sim"] == {
        "terminate_on_current_limit": True,
        "current_termination_over_limit_a": 50000.0,
        "current_termination_grace_steps": 50,
        "current_hard_termination_fraction": 1.40,
    }


def test_fixed_horizon_broad_manifest_has_36_and_no_termination() -> None:
    variants = build_variants("broad", profile=PROFILE_FIXED_HORIZON)
    assert len(variants) == 36
    assert variants[0]["folder"] == "b000_s0_i0_a0"
    assert variants[-1]["folder"] == "b035_s2_i2_a3"
    first = variants[0]
    assert first["reward"]["shape_mean_weight"] == 1.0
    assert first["reward"]["shape_max_weight"] == 0.25
    assert first["reward"]["ip_weight"] == 0.75
    assert first["reward"]["current_weight"] == 1.0
    assert first["reward"]["current_soft_fraction"] == 0.90
    assert first["reward"]["current_bad_fraction"] == 1.20
    assert first["reward"]["derivative_weight"] == 0.10
    assert first["reward"]["derivative_soft_fraction"] == 0.90
    assert first["reward"]["derivative_bad_fraction"] == 1.20
    assert first["reward"]["terminal_reward"] == -20.0
    assert first["reward"]["terminal_remaining_cost"] == 0.0
    assert first["reward"]["action_weight"] == 0.01
    assert first["reward"]["delta_action_weight"] == 0.025
    assert first["sim"] == {
        "terminate_on_boundary_loss": False,
        "terminate_on_current_limit": False,
    }


def test_saturation_broad_manifest_has_36_and_saturation_overrides() -> None:
    variants = build_variants("broad", profile=PROFILE_SATURATION)
    assert len(variants) == 36
    assert [variant["index"] for variant in variants] == list(range(36))
    assert len({variant["folder"] for variant in variants}) == 36
    assert variants[0]["folder"] == "s000_s0_i0_a0"
    assert variants[-1]["folder"] == "s035_s2_i2_a3"
    first = variants[0]
    assert first["reward"]["shape_mean_weight"] == 0.75
    assert first["reward"]["shape_max_weight"] == 0.1875
    assert first["reward"]["ip_weight"] == 0.75
    assert first["reward"]["current_weight"] == 1.5
    assert first["reward"]["current_soft_fraction"] == 0.90
    assert first["reward"]["current_bad_fraction"] == 1.05
    assert first["reward"]["derivative_weight"] == 0.20
    assert first["reward"]["current_usage_weight"] == 0.25
    assert first["reward"]["derivative_usage_weight"] == 0.05
    assert first["reward"]["derivative_soft_fraction"] == 0.90
    assert first["reward"]["derivative_bad_fraction"] == 1.20
    assert first["reward"]["actuator_saturation_weight"] == 2.0
    assert first["reward"]["terminal_remaining_cost"] == 0.0
    assert first["reward"]["action_weight"] == 0.01
    assert first["reward"]["delta_action_weight"] == 0.025
    assert first["sim"] == {
        "terminate_on_boundary_loss": False,
        "terminate_on_current_limit": False,
        "current_saturation_fraction": 1.05,
    }
    saturation_weights = {
        variant["actuator_regime"]: variant["reward"]["actuator_saturation_weight"]
        for variant in variants
        if variant["shape_regime"] == "s0" and variant["ip_regime"] == "i0"
    }
    assert saturation_weights == {"a0": 2.0, "a1": 4.0, "a2": 8.0, "a3": 12.0}


def test_tcv_derivative_manifest_has_36_and_operational_termination_overrides() -> None:
    variants = build_variants("broad", profile=PROFILE_TCV_DERIVATIVE)
    assert len(variants) == 36
    assert variants[0]["folder"] == "b000_s0_i0_a0"
    assert variants[-1]["folder"] == "b035_s2_i2_a3"
    first = variants[0]
    assert first["reward"]["kind"] == "tcv_derivative"
    assert first["reward"]["smoothmax_alpha"] == -5.0
    assert first["reward"]["reward_scale"] == 0.01
    assert first["reward"]["terminal_reward"] == -5.0
    assert first["reward"]["ip_scale_a"] == 15000.0
    assert first["reward"]["shape_mean_weight"] == 1.0
    assert first["reward"]["shape_max_weight"] == 0.25
    assert first["reward"]["ip_weight"] == 0.5
    assert first["reward"]["current_weight"] == 1.0
    assert first["reward"]["derivative_weight"] == 0.25
    assert first["reward"]["actuator_saturation_weight"] == 0.25
    assert first["reward"]["action_weight"] == 0.0
    assert first["reward"]["delta_action_weight"] == 0.0
    assert first["sim"]["action_contract"] == "delta_jdot"
    assert first["sim"]["delta_derivative_scale_aps"] == 500000.0
    assert first["sim"]["delta_derivative_limits_aps"]["pfc"]["pfc5"] == 1191036.96
    assert first["sim"]["delta_derivative_limits_aps"]["sol"]["sol1"] == 5889842.0
    assert first["sim"]["terminate_on_boundary_loss"] is True
    assert first["sim"]["terminate_on_current_limit"] is True
    assert first["sim"]["current_termination_over_limit_a"] == 0.0
    assert first["sim"]["current_termination_grace_steps"] == 1
    assert first["sim"]["current_hard_termination_fraction"] == 1.20
    assert first["sim"]["current_saturation_fraction"] == 1.0


def test_tcv_derivative_focused_manifest_has_12_variants() -> None:
    center = {
        "shape_mean_weight": 2.0,
        "shape_max_weight": 0.5,
        "ip_weight": 1.0,
        "current_weight": 2.0,
        "derivative_weight": 0.5,
        "actuator_saturation_weight": 0.5,
    }
    manifest = build_manifest(
        "focused",
        center_reward=center,
        profile=PROFILE_TCV_DERIVATIVE,
        runs_per_array_task=1,
        array_task_count=12,
    )
    variants = manifest["variants"]
    assert manifest["variant_count"] == 12
    assert variants[0]["folder"] == "f000_sf0_if0_af0"
    assert variants[-1]["folder"] == "f011_sf2_if1_af1"
    assert variants[0]["reward"]["kind"] == "tcv_derivative"
    assert variants[0]["reward"]["current_weight"] == 1.5
    assert variants[-1]["reward"]["current_weight"] == 3.0


def test_tcv_delta_jdot_manifest_has_60_and_delta_contract() -> None:
    manifest = build_manifest(
        "broad",
        profile=PROFILE_TCV_DELTA_JDOT,
        runs_per_array_task=5,
        array_task_count=12,
    )
    variants = manifest["variants"]
    assert manifest["variant_count"] == 60
    assert manifest["runs_per_array_task"] == 5
    assert manifest["array_task_count"] == 12
    assert variants[0]["folder"] == "b000_s0_i0_a0"
    assert variants[-1]["folder"] == "b059_s3_i2_a4"
    assert variants[0]["reward"]["kind"] == "tcv_derivative"
    assert variants[0]["reward"]["shape_mean_weight"] == 0.5
    assert variants[0]["reward"]["ip_weight"] == 0.25
    assert variants[0]["reward"]["current_weight"] == 0.5
    assert variants[-1]["reward"]["shape_mean_weight"] == 4.0
    assert variants[-1]["reward"]["ip_weight"] == 1.5
    assert variants[-1]["reward"]["current_weight"] == 8.0
    assert variants[0]["sim"]["action_contract"] == "delta_jdot"
    assert variants[0]["sim"]["delta_derivative_scale_aps"] == 500000.0
    assert variants[0]["sim"]["delta_derivative_limits_aps"]["pfc"]["pfc0"] == 163347.0
    assert variants[0]["sim"]["delta_derivative_limits_aps"]["sol"]["sol1"] == 5889842.0
    assert variants[0]["sim"]["current_saturation_fraction"] == 1.0


def test_tcv_delta_jdot_focused_manifest_has_12_variants() -> None:
    center = {
        "shape_mean_weight": 1.0,
        "shape_max_weight": 0.25,
        "ip_weight": 0.75,
        "current_weight": 2.0,
        "derivative_weight": 0.5,
        "actuator_saturation_weight": 0.5,
    }
    manifest = build_manifest(
        "focused",
        center_reward=center,
        profile=PROFILE_TCV_DELTA_JDOT,
        runs_per_array_task=1,
        array_task_count=12,
    )
    variants = manifest["variants"]
    assert manifest["variant_count"] == 12
    assert variants[0]["folder"] == "f000_sf0_if0_af0"
    assert variants[-1]["folder"] == "f011_sf2_if1_af1"
    assert variants[0]["reward"]["kind"] == "tcv_derivative"
    assert variants[0]["reward"]["current_weight"] == 1.5
    assert variants[-1]["reward"]["current_weight"] == 3.0
    assert all(variant["sim"]["action_contract"] == "delta_jdot" for variant in variants)
    assert all(variant["sim"]["delta_derivative_limits_aps"]["pfc"]["pfc5"] == 1191036.96 for variant in variants)
    assert all(variant["sim"]["delta_derivative_limits_aps"]["sol"]["sol2"] == 1946208.8 for variant in variants)


def test_tcv_delta_termination_f002_manifest_has_12_termination_variants() -> None:
    manifest = build_manifest(
        "broad",
        profile=PROFILE_TCV_DELTA_TERMINATION_F002,
        runs_per_array_task=1,
        array_task_count=12,
    )
    variants = manifest["variants"]
    assert manifest["variant_count"] == 12
    assert manifest["runs_per_array_task"] == 1
    assert manifest["array_task_count"] == 12
    assert variants[0]["folder"] == "t000_f110_g001"
    assert variants[-1]["folder"] == "t011_f130_g025"
    assert {variant["reward"]["shape_mean_weight"] for variant in variants} == {3.2}
    assert {variant["reward"]["shape_max_weight"] for variant in variants} == {0.8}
    assert {variant["reward"]["ip_weight"] for variant in variants} == {1.8}
    assert {variant["reward"]["current_weight"] for variant in variants} == {0.75}
    assert {variant["reward"]["derivative_weight"] for variant in variants} == {0.1875}
    assert {variant["reward"]["actuator_saturation_weight"] for variant in variants} == {0.1875}
    assert {variant["reward"]["terminal_reward"] for variant in variants} == {-5.0}
    assert {variant["sim"]["terminate_on_boundary_loss"] for variant in variants} == {True}
    assert {variant["sim"]["terminate_on_current_limit"] for variant in variants} == {True}
    assert {variant["sim"]["action_contract"] for variant in variants} == {"delta_jdot"}
    assert {variant["sim"]["delta_derivative_scale_aps"] for variant in variants} == {500000.0}
    assert {variant["sim"]["delta_derivative_limits_aps"]["pfc"]["pfc2"] for variant in variants} == {87838.08}
    assert {variant["sim"]["delta_derivative_limits_aps"]["sol"]["sol0"] for variant in variants} == {1437338.8}
    assert {variant["sim"]["current_hard_termination_fraction"] for variant in variants} == {1.10, 1.15, 1.20, 1.30}
    assert {variant["sim"]["current_termination_grace_steps"] for variant in variants} == {1, 8, 25}
    assert all(variant["sim"]["current_termination_over_limit_a"] == 0.0 for variant in variants)


def test_tcv_delta_no_termination_manifest_has_36_variants() -> None:
    manifest = build_manifest(
        "broad",
        profile=PROFILE_TCV_DELTA_NO_TERMINATION,
        runs_per_array_task=3,
        array_task_count=12,
    )
    variants = manifest["variants"]
    assert manifest["variant_count"] == 36
    assert manifest["runs_per_array_task"] == 3
    assert manifest["array_task_count"] == 12
    assert variants[0]["folder"] == "n000_s0_i0_a0"
    assert variants[-1]["folder"] == "n035_s2_i2_a3"
    assert {variant["reward"]["kind"] for variant in variants} == {"tcv_derivative"}
    assert {variant["sim"]["action_contract"] for variant in variants} == {"delta_jdot"}
    assert {variant["sim"]["delta_derivative_scale_aps"] for variant in variants} == {500000.0}
    assert {variant["sim"]["delta_derivative_limits_aps"]["pfc"]["pfc4"] for variant in variants} == {404364.0}
    assert {variant["sim"]["delta_derivative_limits_aps"]["sol"]["sol1"] for variant in variants} == {5889842.0}
    assert {variant["sim"]["terminate_on_boundary_loss"] for variant in variants} == {False}
    assert {variant["sim"]["terminate_on_current_limit"] for variant in variants} == {False}
    assert {variant["training"]["production_mode"] for variant in variants} == {False}
    assert {variant["reward"]["shape_mean_weight"] for variant in variants} == {1.6, 3.2, 6.4}
    assert {variant["reward"]["ip_weight"] for variant in variants} == {0.9, 1.8, 3.6}
    assert {variant["reward"]["current_weight"] for variant in variants} == {0.75, 1.5, 3.0, 6.0}


def test_current_constraint_focused_manifest_uses_center_soft_fractions() -> None:
    center = {
        "shape_mean_weight": 10.0,
        "shape_max_weight": 3.0,
        "ip_weight": 3.0,
        "current_weight": 4.0,
        "current_soft_fraction": 0.90,
        "derivative_weight": 0.25,
        "derivative_soft_fraction": 0.85,
        "terminal_remaining_cost": 50000.0,
    }
    variants = build_variants("focused", center, profile=PROFILE_CURRENT_CONSTRAINT)
    assert len(variants) == 36
    assert variants[0]["folder"] == "f000_sf0_if0_af0"
    assert variants[-1]["folder"] == "f035_sf2_if2_af3"
    assert variants[0]["reward"]["shape_mean_weight"] == 7.5
    assert variants[0]["reward"]["ip_weight"] == 2.1
    assert variants[0]["reward"]["current_weight"] == 3.0
    assert variants[0]["reward"]["derivative_weight"] == 0.1875
    assert variants[0]["reward"]["terminal_remaining_cost"] == 37500.0
    assert variants[0]["reward"]["current_soft_fraction"] == 0.90
    assert variants[0]["reward"]["derivative_soft_fraction"] == 0.90
    center_variant = variants[17]
    assert center_variant["folder"] == "f017_sf1_if1_af1"
    assert center_variant["reward"]["shape_mean_weight"] == 10.0
    assert center_variant["reward"]["ip_weight"] == 3.0
    assert center_variant["reward"]["current_weight"] == 4.0
    assert center_variant["reward"]["derivative_weight"] == 0.25
    assert center_variant["reward"]["terminal_remaining_cost"] == 50000.0
    assert center_variant["reward"]["current_soft_fraction"] == 0.90
    assert center_variant["reward"]["derivative_soft_fraction"] == 0.85
    assert center_variant["sim"]["terminate_on_current_limit"] is True


def test_fixed_horizon_focused_manifest_uses_center_soft_fractions() -> None:
    center = {
        "shape_mean_weight": 2.0,
        "shape_max_weight": 0.5,
        "ip_weight": 1.5,
        "current_weight": 2.0,
        "current_soft_fraction": 0.90,
        "derivative_weight": 0.25,
        "derivative_soft_fraction": 0.85,
        "terminal_remaining_cost": 0.0,
    }
    variants = build_variants("focused", center, profile=PROFILE_FIXED_HORIZON)
    assert len(variants) == 36
    assert variants[0]["folder"] == "f000_sf0_if0_af0"
    assert variants[-1]["folder"] == "f035_sf2_if2_af3"
    assert variants[0]["reward"]["shape_mean_weight"] == 1.5
    assert variants[0]["reward"]["ip_weight"] == 1.05
    assert variants[0]["reward"]["current_weight"] == 1.4
    assert variants[0]["reward"]["derivative_weight"] == 0.175
    assert variants[0]["reward"]["terminal_remaining_cost"] == 0.0
    assert variants[0]["reward"]["current_soft_fraction"] == 0.90
    assert variants[0]["reward"]["derivative_soft_fraction"] == 0.90
    center_variant = variants[17]
    assert center_variant["folder"] == "f017_sf1_if1_af1"
    assert center_variant["reward"]["shape_mean_weight"] == 2.0
    assert center_variant["reward"]["ip_weight"] == 1.5
    assert center_variant["reward"]["current_weight"] == 2.0
    assert center_variant["reward"]["derivative_weight"] == 0.25
    assert center_variant["reward"]["current_soft_fraction"] == 0.90
    assert center_variant["reward"]["derivative_soft_fraction"] == 0.85
    assert center_variant["sim"]["terminate_on_boundary_loss"] is False
    assert center_variant["sim"]["terminate_on_current_limit"] is False


def test_reward_sweep_manifests_have_36_variants_without_hidden_subsampling() -> None:
    broad = build_manifest("broad", runs_per_array_task=3, array_task_count=12)
    assert broad["variant_count"] == 36
    assert broad["runs_per_array_task"] == 3
    assert broad["array_task_count"] == 12
    assert broad["variants"][0]["folder"].startswith("b000_")
    assert broad["variants"][-1]["folder"].startswith("b035_")
    assert all("source_index" not in variant for variant in broad["variants"])

    center = {
        "shape_mean_weight": 2.25,
        "shape_max_weight": 0.5625,
        "ip_weight": 5.0,
        "current_weight": 1.2,
        "derivative_weight": 0.9,
    }
    focused = build_manifest("focused", center, runs_per_array_task=3, array_task_count=12)
    assert focused["variant_count"] == 36
    assert focused["runs_per_array_task"] == 3
    assert focused["array_task_count"] == 12
    assert focused["variants"][0]["folder"].startswith("f000_")
    assert focused["variants"][-1]["folder"].startswith("f035_")
    assert all("source_index" not in variant for variant in focused["variants"])


def test_current_constraint_manifest_has_exact_36_variants() -> None:
    broad = build_manifest("broad", profile=PROFILE_CURRENT_CONSTRAINT, runs_per_array_task=3, array_task_count=12)
    assert broad["profile"] == PROFILE_CURRENT_CONSTRAINT
    assert broad["variant_count"] == 36
    assert broad["fixed_sim"]["terminate_on_current_limit"] is True
    assert broad["fixed_reward"]["action_weight"] == 0.01
    assert broad["fixed_reward"]["terminal_remaining_cost"] == 50000.0


def test_fixed_horizon_manifest_has_exact_36_variants() -> None:
    broad = build_manifest("broad", profile=PROFILE_FIXED_HORIZON, runs_per_array_task=3, array_task_count=12)
    assert broad["profile"] == PROFILE_FIXED_HORIZON
    assert broad["variant_count"] == 36
    assert broad["fixed_sim"] == {
        "terminate_on_boundary_loss": False,
        "terminate_on_current_limit": False,
    }
    assert broad["fixed_reward"]["action_weight"] == 0.01
    assert broad["fixed_reward"]["delta_action_weight"] == 0.025
    assert broad["fixed_reward"]["terminal_remaining_cost"] == 0.0


def test_saturation_manifest_has_exact_36_variants() -> None:
    broad = build_manifest("broad", profile=PROFILE_SATURATION, runs_per_array_task=3, array_task_count=12)
    assert broad["profile"] == PROFILE_SATURATION
    assert broad["variant_count"] == 36
    assert broad["runs_per_array_task"] == 3
    assert broad["array_task_count"] == 12
    assert broad["variants"][0]["folder"] == "s000_s0_i0_a0"
    assert broad["variants"][-1]["folder"] == "s035_s2_i2_a3"
    assert broad["fixed_sim"] == {
        "terminate_on_boundary_loss": False,
        "terminate_on_current_limit": False,
        "current_saturation_fraction": 1.05,
    }
    assert all("actuator_saturation_weight" in variant["reward"] for variant in broad["variants"])
    assert all("current_usage_weight" in variant["reward"] for variant in broad["variants"])
    assert all("derivative_usage_weight" in variant["reward"] for variant in broad["variants"])


def test_saturation_focused_manifest_has_12_local_variants() -> None:
    center = {
        "shape_mean_weight": 1.5,
        "shape_max_weight": 0.375,
        "ip_weight": 1.5,
        "current_weight": 3.0,
        "derivative_weight": 0.40,
        "current_usage_weight": 0.50,
        "derivative_usage_weight": 0.10,
        "actuator_saturation_weight": 4.0,
    }
    focused = build_manifest("focused", center, profile=PROFILE_SATURATION, runs_per_array_task=1, array_task_count=12)
    assert focused["profile"] == PROFILE_SATURATION
    assert focused["variant_count"] == 12
    assert focused["runs_per_array_task"] == 1
    assert focused["array_task_count"] == 12
    assert focused["variants"][0]["folder"] == "f000_sf0_if0_af0"
    assert focused["variants"][-1]["folder"] == "f011_sf2_if1_af1"
    first = focused["variants"][0]
    assert first["reward"]["shape_mean_weight"] == 1.2
    assert first["reward"]["shape_max_weight"] == 0.3
    assert first["reward"]["ip_weight"] == 1.2
    assert first["reward"]["current_weight"] == 2.55
    assert first["reward"]["derivative_weight"] == 0.34
    assert first["reward"]["current_usage_weight"] == 0.375
    assert first["reward"]["derivative_usage_weight"] == 0.075
    assert first["reward"]["actuator_saturation_weight"] == 3.0
    assert first["sim"] == {
        "terminate_on_boundary_loss": False,
        "terminate_on_current_limit": False,
        "current_saturation_fraction": 1.05,
    }


def test_reward_sweep_array_task_mappings() -> None:
    broad = build_variants("broad")
    for task_id in range(12):
        mapped = [task_id * 3 + local_index for local_index in range(3)]
        assert mapped == [broad[index]["index"] for index in mapped]
    focused = build_variants(
        "focused",
        {
            "shape_mean_weight": 2.25,
            "shape_max_weight": 0.5625,
            "ip_weight": 5.0,
            "current_weight": 1.2,
            "derivative_weight": 0.9,
        },
    )
    for task_id in range(12):
        mapped = [task_id * 3 + local_index for local_index in range(3)]
        assert mapped == [focused[index]["index"] for index in mapped]
    assert 11 * 3 + 2 == 35


def test_reward_sweep_candidate_applies_sim_overrides_to_generated_config(tmp_path: Path, monkeypatch) -> None:
    variant = build_variants("broad", profile=PROFILE_CURRENT_CONSTRAINT)[0]
    manifest = tmp_path / "variants.json"
    manifest.write_text(json.dumps({"variants": [variant]}), encoding="utf-8")
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps(
            {
                "name": "base",
                "sim": {"terminate_on_current_limit": False},
                "reference": {"ip": {}},
                "reward": {},
                "training": {},
                "learner": {},
            }
        ),
        encoding="utf-8",
    )

    class Result:
        returncode = 0

    monkeypatch.setattr("scripts.run_reward_sweep_candidate.subprocess.run", lambda *args, **kwargs: Result())
    sweep_root = tmp_path / "sweep"
    args = Namespace(
        manifest=manifest,
        variant_index=0,
        base_config=base_config,
        sweep_root=sweep_root,
        train_env_steps=100,
        eval_env_steps=50,
        num_envs=4,
        batch_size=2,
        replay_capacity_episodes=288,
        rollout_chunk_length=64,
        updates_per_rollout_chunk=32,
        wandb_project="test",
        wandb_mode="offline",
        device="cuda:0",
        sim_config_path=Path("/sim.toml"),
        initial_state_library=Path("/states.npz"),
        reference_limits=Path("/limits.json"),
    )

    assert run_candidate(args) == 0
    generated = json.loads((sweep_root / "generated_configs" / f"{variant['folder']}.json").read_text(encoding="utf-8"))
    assert generated["sim"]["terminate_on_current_limit"] is True
    assert generated["sim"]["current_termination_over_limit_a"] == 50000.0
    assert generated["sim"]["current_termination_grace_steps"] == 50
    assert generated["sim"]["current_hard_termination_fraction"] == 1.40
    assert generated["reward"]["action_weight"] == 0.01
    assert generated["reward"]["delta_action_weight"] == 0.025
    assert generated["reward"]["terminal_remaining_cost"] == 25000.0


def test_reward_sweep_candidate_applies_fixed_horizon_sim_overrides(tmp_path: Path, monkeypatch) -> None:
    variant = build_variants("broad", profile=PROFILE_FIXED_HORIZON)[0]
    manifest = tmp_path / "variants.json"
    manifest.write_text(json.dumps({"variants": [variant]}), encoding="utf-8")
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps(
            {
                "name": "base",
                "sim": {"terminate_on_boundary_loss": True, "terminate_on_current_limit": True, "max_episode_steps": 2000},
                "reference": {"ip": {}},
                "reward": {},
                "training": {},
                "learner": {},
            }
        ),
        encoding="utf-8",
    )

    class Result:
        returncode = 0

    monkeypatch.setattr("scripts.run_reward_sweep_candidate.subprocess.run", lambda *args, **kwargs: Result())
    sweep_root = tmp_path / "sweep"
    args = Namespace(
        manifest=manifest,
        variant_index=0,
        base_config=base_config,
        sweep_root=sweep_root,
        train_env_steps=100,
        eval_env_steps=50,
        num_envs=4,
        batch_size=2,
        replay_capacity_episodes=288,
        rollout_chunk_length=64,
        updates_per_rollout_chunk=32,
        wandb_project="test",
        wandb_mode="offline",
        device="cuda:0",
        sim_config_path=Path("/sim.toml"),
        initial_state_library=Path("/states.npz"),
        reference_limits=Path("/limits.json"),
    )

    assert run_candidate(args) == 0
    generated = json.loads((sweep_root / "generated_configs" / f"{variant['folder']}.json").read_text(encoding="utf-8"))
    assert generated["sim"]["terminate_on_boundary_loss"] is False
    assert generated["sim"]["terminate_on_current_limit"] is False
    assert generated["training"]["eval_max_steps"] == 2000
    assert generated["reward"]["terminal_remaining_cost"] == 0.0
    assert generated["reward"]["action_weight"] == 0.01
    assert generated["reward"]["delta_action_weight"] == 0.025


def test_reward_sweep_candidate_applies_saturation_overrides(tmp_path: Path, monkeypatch) -> None:
    variant = build_variants("broad", profile=PROFILE_SATURATION)[0]
    manifest = tmp_path / "variants.json"
    manifest.write_text(json.dumps({"variants": [variant]}), encoding="utf-8")
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps(
            {
                "name": "base",
                "sim": {"terminate_on_boundary_loss": True, "terminate_on_current_limit": True, "max_episode_steps": 2000},
                "reference": {"ip": {}},
                "reward": {},
                "training": {},
                "learner": {},
            }
        ),
        encoding="utf-8",
    )

    class Result:
        returncode = 0

    monkeypatch.setattr("scripts.run_reward_sweep_candidate.subprocess.run", lambda *args, **kwargs: Result())
    sweep_root = tmp_path / "sweep"
    args = Namespace(
        manifest=manifest,
        variant_index=0,
        base_config=base_config,
        sweep_root=sweep_root,
        train_env_steps=100,
        eval_env_steps=50,
        num_envs=4,
        batch_size=2,
        replay_capacity_episodes=288,
        rollout_chunk_length=64,
        updates_per_rollout_chunk=32,
        wandb_project="test",
        wandb_mode="offline",
        device="cuda:0",
        sim_config_path=Path("/sim.toml"),
        initial_state_library=Path("/states.npz"),
        reference_limits=Path("/limits.json"),
    )

    assert run_candidate(args) == 0
    generated = json.loads((sweep_root / "generated_configs" / f"{variant['folder']}.json").read_text(encoding="utf-8"))
    assert generated["sim"]["terminate_on_boundary_loss"] is False
    assert generated["sim"]["terminate_on_current_limit"] is False
    assert generated["sim"]["current_saturation_fraction"] == 1.05
    assert generated["reward"]["actuator_saturation_weight"] == 2.0
    assert generated["reward"]["current_usage_weight"] == 0.25
    assert generated["reward"]["derivative_usage_weight"] == 0.05
    assert generated["reward"]["current_soft_fraction"] == 0.90
    assert generated["reward"]["current_bad_fraction"] == 1.05
    assert generated["reward"]["terminal_remaining_cost"] == 0.0


def test_reward_sweep_job_blocks_stale_name_leaks() -> None:
    for path in (
        PASS1_JOB,
        PASS2_JOB,
        PASS1_12_JOB,
        PASS2_12_JOB,
        PASS1_CURRENT_JOB,
        PASS2_CURRENT_JOB,
        PASS1_FIXED_HORIZON_JOB,
        PASS2_FIXED_HORIZON_JOB,
        SATURATION_JOB,
        SATURATION_PASS1_JOB,
        SATURATION_PASS2_JOB,
        RERUN_JOB,
    ):
        text = path.read_text(encoding="utf-8")
        assert "unset RUN_NAME" in text
        assert "unset TRAIN_OUTPUT" in text
        assert "unset WANDB_PROJECT_NAME" in text
        assert "run_reward_sweep_candidate.py" in text
        assert "apply_t15_actuator_limits.py" in text
    runner = (ROOT / "scripts/run_reward_sweep_candidate.py").read_text(encoding="utf-8")
    assert "--no-save-checkpoints" in runner
    assert "--reward-sweep-mode" in runner
    assert "--no-export" in runner
    assert "--wandb-optional" in runner
    assert "shutil.rmtree(output_dir / \"exports\"" in runner
    assert "shutil.rmtree(output_dir / \"checkpoints\"" in runner
    assert "REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}" in PASS1_12_JOB.read_text(encoding="utf-8")
    assert "REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}" in PASS2_12_JOB.read_text(encoding="utf-8")
    assert "REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}" in PASS1_CURRENT_JOB.read_text(encoding="utf-8")
    assert "REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}" in PASS2_CURRENT_JOB.read_text(encoding="utf-8")
    assert "REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}" in PASS1_FIXED_HORIZON_JOB.read_text(encoding="utf-8")
    assert "REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}" in PASS2_FIXED_HORIZON_JOB.read_text(encoding="utf-8")
    assert "REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}" in SATURATION_JOB.read_text(encoding="utf-8")
    assert "REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}" in SATURATION_PASS1_JOB.read_text(encoding="utf-8")
    assert "REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}" in SATURATION_PASS2_JOB.read_text(encoding="utf-8")
    for path in (
        ROOT / "jobs/aggregate_t15_reward_sweep_pass1.sbatch",
        ROOT / "jobs/aggregate_t15_reward_sweep_final.sbatch",
        SATURATION_AGG_JOB,
        PASS1_12_JOB,
        PASS2_12_JOB,
        SATURATION_PASS1_JOB,
        SATURATION_PASS2_JOB,
    ):
        assert "reward288" not in path.read_text(encoding="utf-8")


def test_reward_sweep_pipeline_static_audit_passes() -> None:
    assert audit_reward_sweep_pipeline(ROOT) == []


def test_reward_sweep_score_penalizes_bad_current_and_completion() -> None:
    good = {
        "shape_error_mean_m_late": "0.03",
        "shape_error_max_m_late": "0.08",
        "ip_error_a_late": "25000",
        "current_over_limit_a_late": "0",
        "current_over_limit_fraction_late": "0",
        "mean_episode_completion": "1.0",
        "boundary_found_late_min": "1.0",
    }
    bad = dict(good)
    bad["current_over_limit_a_late"] = "20000"
    bad["mean_episode_completion"] = "0.5"
    assert score_eval_row(bad) > score_eval_row(good)


def test_reward_sweep_aggregator_handles_complete_and_missing_runs(tmp_path: Path) -> None:
    manifest = build_manifest()
    manifest["variants"] = manifest["variants"][:2]
    (tmp_path / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    run_dir = tmp_path / manifest["variants"][0]["folder"]
    run_dir.mkdir(parents=True)
    with (run_dir / "eval_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "shape_error_mean_m_late",
                "shape_error_max_m_late",
                "ip_error_a_late",
                "current_over_limit_a_late",
                "current_over_limit_fraction_late",
                "mean_episode_completion",
                "boundary_found_late_min",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": 100000,
                "shape_error_mean_m_late": 0.03,
                "shape_error_max_m_late": 0.08,
                "ip_error_a_late": 25000,
                "current_over_limit_a_late": 0,
                "current_over_limit_fraction_late": 0,
                "mean_episode_completion": 1.0,
                "boundary_found_late_min": 1.0,
            }
        )
    (run_dir / "policy_validation.json").write_text(json.dumps({"status": "sweep_completed"}), encoding="utf-8")

    result = aggregate(tmp_path)
    write_outputs(tmp_path, result)

    assert result["summaries"][0]["variant_index"] == 0
    assert result["failures"][0]["variant_index"] == 1
    assert (tmp_path / "reward_sweep_summary.csv").exists()
    assert (tmp_path / "reward_sweep_top20.md").exists()
    assert (tmp_path / "reward_sweep_best.json").exists()
    assert (tmp_path / "reward_sweep_failures.csv").exists()


def _write_physical_run(
    root: Path,
    variant: dict,
    *,
    completion: float = 1.0,
    boundary: float = 1.0,
    shape: float = 0.04,
    ip: float = 40000.0,
    current_max: float = 0.0,
    current_fraction: float = 0.0,
    current_usage_loss: float = 0.2,
    derivative_usage_loss: float = 0.05,
    saturation_fraction: float = 0.0,
    saturation_loss: float = 0.0,
    actor_eval: bool = True,
) -> None:
    run_dir = root / variant["folder"]
    run_dir.mkdir(parents=True)
    (run_dir / "reward_variant.json").write_text(json.dumps({"variant": variant}), encoding="utf-8")
    validation = {"status": "sweep_completed"}
    if actor_eval:
        validation["actor_eval"] = {
            "mean_episode_completion": completion,
            "boundary_found_late_min": boundary,
            "terminated_boundary": 0.0 if boundary >= 0.999 else 0.1,
            "shape_error_mean_m_late": shape,
            "shape_error_max_m_late": shape * 2.0,
            "ip_error_a_late": ip,
            "current_over_limit_a_late_max": current_max,
            "current_over_limit_fraction_late": current_fraction,
            "current_usage_fraction_late_max": 0.8 if current_max == 0.0 else 1.2,
            "current_usage_mean_fraction_late": 0.4,
            "current_usage_loss_late": current_usage_loss,
            "derivative_usage_mean_fraction_late": 0.2,
            "derivative_usage_loss_late": derivative_usage_loss,
            "action_rms_late": 0.1,
            "delta_action_rms_late": 0.01,
            "action_saturation_fraction_late": saturation_fraction,
            "action_saturation_delta_rms_late": saturation_fraction * 0.1,
            "actuator_saturation_loss_late": saturation_loss,
        }
    (run_dir / "policy_validation.json").write_text(json.dumps(validation), encoding="utf-8")
    with (run_dir / "eval_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env_step",
                "mean_episode_completion",
                "boundary_found_late_min",
                "shape_error_mean_m_late",
                "ip_error_a_late",
                "current_over_limit_fraction_late",
                "current_usage_loss_late",
                "derivative_usage_loss_late",
                "action_saturation_fraction_late",
                "actuator_saturation_loss_late",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "env_step": 100000,
                "mean_episode_completion": completion,
                "boundary_found_late_min": boundary,
                "shape_error_mean_m_late": shape,
                "ip_error_a_late": ip,
                "current_over_limit_fraction_late": current_fraction,
                "current_usage_loss_late": current_usage_loss,
                "derivative_usage_loss_late": derivative_usage_loss,
                "action_saturation_fraction_late": saturation_fraction,
                "actuator_saturation_loss_late": saturation_loss,
            }
        )


def test_physical_sweep_summary_uses_actor_eval_and_marks_missing_actor_eval(tmp_path: Path) -> None:
    manifest = build_manifest()
    manifest["variants"] = manifest["variants"][:2]
    (tmp_path / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_physical_run(tmp_path, manifest["variants"][0], ip=45000.0)
    _write_physical_run(tmp_path, manifest["variants"][1], actor_eval=False)

    out_dir = tmp_path / "analysis"
    summarize(tmp_path, out_dir, top=10, min_completion=0.95, min_boundary_late=0.999, max_terminated_boundary=0.001, max_current_over_limit_a_max=250000.0, max_current_over_limit_fraction_late=0.5)

    rows = list(csv.DictReader((out_dir / "physical_sweep_summary.csv").open()))
    assert len(rows) == 2
    assert rows[0]["folder"] == manifest["variants"][0]["folder"]
    assert rows[0]["selection_valid"] == "True"
    assert rows[0]["ip_error_a_late"] == "45000.0"
    assert rows[1]["selection_valid"] == "False"
    assert rows[1]["selection_reason"] == "missing_actor_eval"
    assert (out_dir / "physical_best_candidate.json").exists()
    assert (out_dir / "physical_selection_report.md").exists()


def test_physical_sweep_summary_prefers_padded_actor_eval_metrics(tmp_path: Path) -> None:
    manifest = build_manifest()
    manifest["variants"] = manifest["variants"][:1]
    (tmp_path / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_physical_run(tmp_path, manifest["variants"][0], completion=1.0, boundary=1.0, shape=0.02, ip=10000.0)
    validation_path = tmp_path / manifest["variants"][0]["folder"] / "policy_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["actor_eval"].update(
        {
            "full_episode_success": 0.0,
            "termination_failure_fraction": 1.0,
            "padded_boundary_found_late_min": 0.0,
            "padded_shape_error_mean_m_late": 0.10,
            "padded_shape_error_max_m_late": 0.10,
            "padded_ip_error_a_late": 100000.0,
            "padded_current_over_limit_a_late_max": 20000.0,
            "padded_current_over_limit_fraction_late": 1.0,
            "padded_current_usage_fraction_late_max": 1.20,
        }
    )
    validation_path.write_text(json.dumps(validation), encoding="utf-8")

    out_dir = tmp_path / "analysis"
    summarize(tmp_path, out_dir, top=10, min_completion=0.95, min_boundary_late=0.999, max_terminated_boundary=0.001, max_current_over_limit_a_max=250000.0, max_current_over_limit_fraction_late=0.5)

    rows = list(csv.DictReader((out_dir / "physical_sweep_summary.csv").open()))
    assert rows[0]["uses_padded_metrics"] == "True"
    assert rows[0]["selection_valid"] == "False"
    assert rows[0]["selection_reason"] == "low_full_episode_success"
    assert rows[0]["boundary_found_late_min"] == "0.0"
    assert rows[0]["shape_error_mean_m_late"] == "0.1"
    assert rows[0]["ip_error_a_late"] == "100000.0"


def test_physical_sweep_best_candidate_falls_back_to_imperfect_actor_eval(tmp_path: Path) -> None:
    manifest = build_manifest()
    manifest["variants"] = manifest["variants"][:2]
    (tmp_path / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_physical_run(tmp_path, manifest["variants"][0], completion=0.4, boundary=0.9, shape=0.08, ip=90000.0)
    _write_physical_run(tmp_path, manifest["variants"][1], actor_eval=False)

    out_dir = tmp_path / "analysis"
    summarize(tmp_path, out_dir, top=10, min_completion=0.95, min_boundary_late=0.999, max_terminated_boundary=0.001, max_current_over_limit_a_max=250000.0, max_current_over_limit_fraction_late=0.5)

    best = json.loads((out_dir / "physical_best_candidate.json").read_text(encoding="utf-8"))
    assert best["valid_candidates"] == 0
    assert best["best_candidate_passed_hard_filters"] is False
    assert best["best_candidate"]["folder"] == manifest["variants"][0]["folder"]
    assert "shape_mean_weight" in best["best_candidate"]


def test_physical_sweep_pareto_excludes_dominated_runs(tmp_path: Path) -> None:
    variants = build_variants()[:2]
    (tmp_path / "variants.json").write_text(json.dumps({"variants": variants}), encoding="utf-8")
    _write_physical_run(tmp_path, variants[0], shape=0.03, ip=30000.0, current_max=0.0, current_fraction=0.0)
    _write_physical_run(tmp_path, variants[1], shape=0.06, ip=60000.0, current_max=1000.0, current_fraction=0.1)

    out_dir = tmp_path / "analysis"
    summarize(tmp_path, out_dir, top=10, min_completion=0.95, min_boundary_late=0.999, max_terminated_boundary=0.001, max_current_over_limit_a_max=250000.0, max_current_over_limit_fraction_late=0.5)

    front = list(csv.DictReader((out_dir / "physical_pareto_front.csv").open()))
    assert [row["folder"] for row in front] == [variants[0]["folder"]]


def test_physical_sweep_penalizes_saturation_usage(tmp_path: Path) -> None:
    variants = build_variants("broad", profile=PROFILE_SATURATION)[:2]
    (tmp_path / "variants.json").write_text(json.dumps({"variants": variants}), encoding="utf-8")
    _write_physical_run(tmp_path, variants[0], saturation_fraction=0.0, saturation_loss=0.0)
    _write_physical_run(tmp_path, variants[1], saturation_fraction=0.8, saturation_loss=0.4)

    out_dir = tmp_path / "analysis"
    summarize(
        tmp_path,
        out_dir,
        top=10,
        min_completion=0.95,
        min_boundary_late=0.999,
        max_terminated_boundary=0.001,
        max_current_over_limit_a_max=250000.0,
        max_current_over_limit_fraction_late=0.5,
    )

    rows = list(csv.DictReader((out_dir / "physical_sweep_summary.csv").open()))
    assert rows[0]["folder"] == variants[0]["folder"]
    assert rows[0]["action_saturation_fraction_late"] == "0.0"
    assert rows[1]["folder"] == variants[1]["folder"]
    assert rows[1]["action_saturation_fraction_late"] == "0.8"


def test_physical_sweep_regime_summary_groups_reward_axes(tmp_path: Path) -> None:
    manifest = build_manifest()
    manifest["variants"] = manifest["variants"][:4]
    (tmp_path / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    for variant in manifest["variants"]:
        _write_physical_run(tmp_path, variant)

    out_dir = tmp_path / "analysis"
    summarize(tmp_path, out_dir, top=2, min_completion=0.95, min_boundary_late=0.999, max_terminated_boundary=0.001, max_current_over_limit_a_max=250000.0, max_current_over_limit_fraction_late=0.5)

    regimes = list(csv.DictReader((out_dir / "physical_regime_summary.csv").open()))
    assert {row["regime_kind"] for row in regimes} == {"shape", "ip", "current", "derivative"}
    assert any(row["regime_kind"] == "current" and row["regime"] == "a0" for row in regimes)


def test_reward_sweep_rerun_manifest_finds_missing_and_failed_variants(tmp_path: Path) -> None:
    pass1 = tmp_path / "pass1_broad"
    manifest = build_manifest("broad")
    manifest["variants"] = manifest["variants"][:3]
    pass1.mkdir()
    (pass1 / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_physical_run(pass1, manifest["variants"][0])
    failed_dir = pass1 / manifest["variants"][1]["folder"]
    failed_dir.mkdir()
    (failed_dir / "policy_validation.json").write_text(json.dumps({"status": "sweep_failed_training"}), encoding="utf-8")

    rerun = build_rerun_manifest(tmp_path)

    reasons = {item["folder"]: item["reason"] for item in rerun["variants"]}
    assert reasons[manifest["variants"][1]["folder"]] == "sweep_failed_training"
    assert reasons[manifest["variants"][2]["folder"]] == "missing_folder"
    assert rerun["missing_or_failed_count"] == 2


def test_two_pass_summary_writes_final_recommendation(tmp_path: Path) -> None:
    pass1 = tmp_path / "pass1_broad"
    pass2 = tmp_path / "pass2_focused"
    pass1.mkdir()
    pass2.mkdir()
    pass1_manifest = build_manifest("broad")
    pass1_manifest["variants"] = pass1_manifest["variants"][:2]
    center = pass1_manifest["variants"][0]["reward"]
    pass2_manifest = build_manifest("focused", center)
    pass2_manifest["variants"] = pass2_manifest["variants"][:2]
    (pass1 / "variants.json").write_text(json.dumps(pass1_manifest), encoding="utf-8")
    (pass2 / "variants.json").write_text(json.dumps(pass2_manifest), encoding="utf-8")
    _write_physical_run(pass1, pass1_manifest["variants"][0], shape=0.04, ip=40000.0)
    _write_physical_run(pass1, pass1_manifest["variants"][1], shape=0.05, ip=50000.0)
    _write_physical_run(pass2, pass2_manifest["variants"][0], shape=0.03, ip=25000.0)
    _write_physical_run(pass2, pass2_manifest["variants"][1], shape=0.06, ip=60000.0, current_max=1000.0)

    out_dir = tmp_path / "selection"
    result = summarize_two_pass(tmp_path, out_dir)

    assert result["recommended_candidate"]["sweep_pass"] == "pass2_focused"
    assert result["recommended_candidate"]["folder"] == pass2_manifest["variants"][0]["folder"]
    assert (out_dir / "pass1_physical_summary.csv").exists()
    assert (out_dir / "pass1_physical_best_candidate.json").exists()
    assert (out_dir / "pass2_physical_summary.csv").exists()
    assert (out_dir / "combined_physical_summary.csv").exists()
    assert (out_dir / "combined_pareto_front.csv").exists()
    assert (out_dir / "combined_regime_summary.csv").exists()
    assert (out_dir / "missing_or_failed_variants.json").exists()
    assert (out_dir / "final_reward_recommendation.json").exists()
    assert (out_dir / "final_reward_selection_report.md").exists()


def test_two_pass_summary_keeps_best_available_when_no_hard_filter_passes(tmp_path: Path) -> None:
    pass1 = tmp_path / "pass1_broad"
    pass2 = tmp_path / "pass2_focused"
    pass1.mkdir()
    pass2.mkdir()
    pass1_manifest = build_manifest("broad")
    pass1_manifest["variants"] = pass1_manifest["variants"][:1]
    pass2_manifest = build_manifest("focused", pass1_manifest["variants"][0]["reward"])
    pass2_manifest["variants"] = pass2_manifest["variants"][:1]
    (pass1 / "variants.json").write_text(json.dumps(pass1_manifest), encoding="utf-8")
    (pass2 / "variants.json").write_text(json.dumps(pass2_manifest), encoding="utf-8")
    _write_physical_run(pass1, pass1_manifest["variants"][0], completion=0.4, boundary=0.0, shape=0.10, ip=100000.0)
    _write_physical_run(pass2, pass2_manifest["variants"][0], completion=0.5, boundary=0.0, shape=0.08, ip=90000.0)

    out_dir = tmp_path / "selection"
    result = summarize_two_pass(tmp_path, out_dir)

    assert result["valid_candidates"] == 0
    assert result["passes_hard_filters"] is False
    assert result["recommended_for_long_training"] is False
    assert result["best_available_candidate"]["sweep_pass"] == "pass2_focused"
    assert result["recommended_candidate"]["folder"] == pass2_manifest["variants"][0]["folder"]


def test_submit_two_pass_chain_uses_slurm_dependencies(tmp_path: Path, monkeypatch) -> None:
    submitted: list[list[str]] = []
    released: list[list[str]] = []
    jobids = iter(["111\n", "112\n", "113\n", "114\n"])

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, check, text, stdout, stderr):
        if args[0] == "scontrol":
            released.append(list(args))
            return Result("")
        submitted.append(list(args))
        return Result(next(jobids))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.submit_two_pass_reward_sweep.subprocess.run", fake_run)

    payload = submit_chain(
        pass1_job=Path("jobs/pass1.sbatch"),
        pass1_aggregate_job=Path("jobs/agg1.sbatch"),
        pass2_job=Path("jobs/pass2.sbatch"),
        final_aggregate_job=Path("jobs/final.sbatch"),
        root_prefix="outputs/t15_reward_sweep72_legal_1m",
    )

    assert payload["pass1_jobid"] == "111"
    assert payload["pass1_aggregate_jobid"] == "112"
    assert payload["pass2_jobid"] == "113"
    assert payload["final_aggregate_jobid"] == "114"
    assert payload["root"] == "outputs/t15_reward_sweep72_legal_1m_111"
    assert "--hold" in submitted[0]
    assert "--export=ALL,SWEEP_ROOT_PREFIX=outputs/t15_reward_sweep72_legal_1m" in submitted[0]
    assert submitted[1][2] == "--dependency=afterany:111"
    assert submitted[2][2] == "--dependency=afterok:112"
    assert submitted[3][2] == "--dependency=afterany:113"
    assert (tmp_path / payload["root"] / "selection" / "submission_chain.json").exists()
    assert (tmp_path / payload["root"] / "pass1_broad" / "variants.json").exists()
    assert payload["pass1_manifest"] == f"{payload['root']}/pass1_broad/variants.json"
    assert payload["pass2_manifest"] == f"{payload['root']}/pass2_focused/variants.json"
    assert released == [["scontrol", "release", "111"]]


def test_submit_saturation_onepass_uses_single_array_and_aggregate(tmp_path: Path, monkeypatch) -> None:
    submitted: list[list[str]] = []
    released: list[list[str]] = []
    jobids = iter(["211\n", "212\n"])

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, check, text, stdout, stderr):
        if args[0] == "scontrol":
            released.append(list(args))
            return Result("")
        submitted.append(list(args))
        return Result(next(jobids))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.submit_saturation_reward_sweep.subprocess.run", fake_run)

    payload = submit_onepass(
        sweep_job=Path("jobs/saturation.sbatch"),
        aggregate_job=Path("jobs/aggregate.sbatch"),
        root_prefix="outputs/t15_reward_sweep36_saturation_1m",
    )

    assert payload["sweep_jobid"] == "211"
    assert payload["aggregate_jobid"] == "212"
    assert payload["root"] == "outputs/t15_reward_sweep36_saturation_1m_211"
    assert "--hold" in submitted[0]
    assert "--export=ALL,SWEEP_ROOT_PREFIX=outputs/t15_reward_sweep36_saturation_1m" in submitted[0]
    assert submitted[1][2] == "--dependency=afterany:211"
    assert len(submitted) == 2
    assert "center_json" not in payload
    assert "pass2_jobid" not in payload
    assert (tmp_path / payload["root"] / "selection" / "submission_chain.json").exists()
    assert (tmp_path / payload["root"] / "variants.json").exists()
    manifest = json.loads((tmp_path / payload["root"] / "variants.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == PROFILE_SATURATION
    assert manifest["variant_count"] == 36
    assert released == [["scontrol", "release", "211"]]


def test_submit_saturation_two_pass_uses_36_plus_12_chain(tmp_path: Path, monkeypatch) -> None:
    submitted: list[list[str]] = []
    released: list[list[str]] = []
    jobids = iter(["311\n", "312\n", "313\n", "314\n"])

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, check, text, stdout, stderr):
        if args[0] == "scontrol":
            released.append(list(args))
            return Result("")
        submitted.append(list(args))
        return Result(next(jobids))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.submit_two_pass_reward_sweep.subprocess.run", fake_run)

    assert (
        submit_saturation_two_pass_main(
            [
                "--pass1-job",
                "jobs/sat-pass1.sbatch",
                "--pass1-aggregate-job",
                "jobs/agg1.sbatch",
                "--pass2-job",
                "jobs/sat-pass2.sbatch",
                "--final-aggregate-job",
                "jobs/final.sbatch",
                "--root-prefix",
                "outputs/t15_reward_sweep48_saturation_1m",
            ]
        )
        == 0
    )

    root = tmp_path / "outputs/t15_reward_sweep48_saturation_1m_311"
    pass1_manifest = json.loads((root / "pass1_broad" / "variants.json").read_text(encoding="utf-8"))
    payload = json.loads((root / "selection" / "submission_chain.json").read_text(encoding="utf-8"))

    assert pass1_manifest["profile"] == PROFILE_SATURATION
    assert pass1_manifest["variant_count"] == 36
    assert pass1_manifest["runs_per_array_task"] == 3
    assert pass1_manifest["array_task_count"] == 12
    assert payload["pass2_manifest"] == "outputs/t15_reward_sweep48_saturation_1m_311/pass2_focused/variants.json"
    assert payload["jobs"]["pass1"] == "jobs/sat-pass1.sbatch"
    assert payload["jobs"]["pass2"] == "jobs/sat-pass2.sbatch"
    assert "--export=ALL,SWEEP_ROOT_PREFIX=outputs/t15_reward_sweep48_saturation_1m" in submitted[0]
    assert submitted[1][2] == "--dependency=afterany:311"
    assert submitted[2][2] == "--dependency=afterok:312"
    assert submitted[3][2] == "--dependency=afterany:313"
    assert released == [["scontrol", "release", "311"]]


def test_submit_chain_supports_60_candidate_pass1(tmp_path: Path, monkeypatch) -> None:
    submitted: list[list[str]] = []
    released: list[list[str]] = []
    jobids = iter(["411\n", "412\n", "413\n", "414\n"])

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, check, text, stdout, stderr):
        if args[0] == "scontrol":
            released.append(list(args))
            return Result("")
        submitted.append(list(args))
        return Result(next(jobids))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.submit_two_pass_reward_sweep.subprocess.run", fake_run)

    payload = submit_chain(
        pass1_job=Path("jobs/tcv-delta-pass1.sbatch"),
        pass1_aggregate_job=Path("jobs/agg1.sbatch"),
        pass2_job=Path("jobs/tcv-delta-pass2.sbatch"),
        final_aggregate_job=Path("jobs/final.sbatch"),
        root_prefix="outputs/t15_reward_sweep72_tcv_delta_jdot_2m5m",
        profile=PROFILE_TCV_DELTA_JDOT,
        pass1_runs_per_array_task=5,
        pass1_array_task_count=12,
    )

    root = tmp_path / "outputs/t15_reward_sweep72_tcv_delta_jdot_2m5m_411"
    pass1_manifest = json.loads((root / "pass1_broad" / "variants.json").read_text(encoding="utf-8"))

    assert payload["root"] == "outputs/t15_reward_sweep72_tcv_delta_jdot_2m5m_411"
    assert payload["pass1_runs_per_array_task"] == 5
    assert payload["pass1_array_task_count"] == 12
    assert pass1_manifest["profile"] == PROFILE_TCV_DELTA_JDOT
    assert pass1_manifest["variant_count"] == 60
    assert pass1_manifest["runs_per_array_task"] == 5
    assert pass1_manifest["array_task_count"] == 12
    assert "--export=ALL,SWEEP_ROOT_PREFIX=outputs/t15_reward_sweep72_tcv_delta_jdot_2m5m" in submitted[0]
    assert released == [["scontrol", "release", "411"]]
