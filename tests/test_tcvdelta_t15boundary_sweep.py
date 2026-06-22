from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.sweeps.tcvdelta_t15boundary import (
    CANDIDATES_PER_TASK,
    VARIANT_COUNT,
    build_manifest,
    generate_candidate_config,
    summarize_root,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml"


def test_tcvdelta_t15boundary_manifest_has_36_mapped_variants() -> None:
    manifest = build_manifest()
    variants = manifest["variants"]
    assert isinstance(variants, list)
    assert len(variants) == VARIANT_COUNT
    assert variants[0]["name"] == "s000_t0_q0_a0"
    assert variants[-1]["name"] == "s035_t2_q2_a3"
    assert len({variant["name"] for variant in variants}) == VARIANT_COUNT
    for variant in variants:
        index = int(variant["index"])
        assert int(variant["task_id"]) == index // CANDIDATES_PER_TASK
        assert int(variant["local_index"]) == index % CANDIDATES_PER_TASK


def test_tcvjdot_t15boundary_candidate_config_loads_with_repaired_invariants(tmp_path: Path) -> None:
    manifest = build_manifest()
    cfg_dict = generate_candidate_config(
        base_config=BASE_CONFIG,
        manifest=manifest,
        variant_index=35,
        output_dir=str(tmp_path / "candidate"),
        train_steps=2_000_000,
        eval_steps=200_000,
        num_envs=256,
        replay_capacity_episodes=288,
        rl_root=str(ROOT),
        sim_root=str((ROOT.parent / "tokamak-sim").resolve()),
    )
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(cfg_dict), encoding="utf-8")
    cfg = load_experiment_config(path)
    assert cfg.reward.kind == "tcv_derivative"
    assert cfg.reward.terminal_reward == -80.0
    assert cfg.reward.shape_mean_weight == 5.0
    assert cfg.reward.shape_max_weight == 1.25
    assert cfg.reward.ip_weight == 3.0
    assert cfg.reward.current_weight == 3.0
    assert cfg.reward.derivative_weight == 0.75
    assert cfg.reward.actuator_saturation_weight == 0.75
    assert cfg.reference.boundary.kind == "t15_replay_segment_conditioned"
    assert cfg.sim.action_contract == "jdot_command"
    assert cfg.observation.actor_kind == "controller_state_v4"
    assert cfg.sim.delta_derivative_limits_aps is None
    assert cfg.sim.terminate_on_boundary_loss is True
    assert cfg.sim.terminate_on_current_limit is True
    assert cfg.sim.current_hard_termination_fraction == 1.2
    assert cfg.sim.current_termination_grace_steps == 1
    assert cfg.sim.current_saturation_fraction == 1.0
    assert cfg.training.steps == 2_000_000
    assert cfg.training.num_envs == 256
    assert cfg.training.save_checkpoints is False


def test_tcvdelta_t15boundary_aggregator_handles_success_and_failure(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    manifest = write_manifest(root / "variants.json")
    first = manifest["variants"][0]
    second = manifest["variants"][1]
    assert isinstance(first, dict) and isinstance(second, dict)
    first_dir = root / str(first["name"])
    first_dir.mkdir(parents=True)
    with (first_dir / "eval_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "mean_episode_completion",
                "full_episode_success",
                "min_episode_completion",
                "boundary_found_late_min",
                "current_over_limit_5ka_fraction_late",
                "current_over_limit_1pct_fraction_late",
                "shape_error_mean_m_late",
                "ip_error_a_late",
                "action_saturation_fraction_late",
                "delta_action_rms_late",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": 100,
                "mean_episode_completion": 1.0,
                "full_episode_success": 1.0,
                "min_episode_completion": 1.0,
                "boundary_found_late_min": 1.0,
                "current_over_limit_5ka_fraction_late": 0.0,
                "current_over_limit_1pct_fraction_late": 0.0,
                "shape_error_mean_m_late": 0.02,
                "ip_error_a_late": 12000,
                "action_saturation_fraction_late": 0.0,
                "delta_action_rms_late": 0.02,
            }
        )
    (first_dir / "policy_validation.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    second_dir = root / str(second["name"])
    second_dir.mkdir(parents=True)
    (second_dir / "policy_validation.json").write_text(json.dumps({"status": "sweep_failed_training"}), encoding="utf-8")

    result = summarize_root(root)
    assert result["rows"] == VARIANT_COUNT
    best = json.loads((root / "selection/best_available_candidate.json").read_text(encoding="utf-8"))
    assert best["name"] == first["name"]
    assert (root / "selection/reward_search_summary.csv").exists()
    assert (root / "selection/reward_search_report.md").exists()


def test_tcvdelta_t15boundary_jobs_pass_bash_syntax() -> None:
    jobs = [
        ROOT / "jobs/sweep_t15_csv_segmented_profile_tcvdelta_t15boundary_12gpu_36x2m.sbatch",
        ROOT / "jobs/aggregate_tcvdelta_t15boundary_reward_sweep.sbatch",
    ]
    for job in jobs:
        subprocess.run(["bash", "-n", str(job)], check=True)


def test_tcvdelta_t15boundary_sweep_job_uses_local_replay_gpu_path() -> None:
    job = ROOT / "jobs/sweep_t15_csv_segmented_profile_tcvdelta_t15boundary_12gpu_36x2m.sbatch"
    text = job.read_text(encoding="utf-8")
    assert "--sim-compute-backend gpu" in text
    assert "--sim-gpu-device cuda:0" in text
    assert "--distributed-mode local_replay" in text
    assert "--distributed-mode single" not in text
