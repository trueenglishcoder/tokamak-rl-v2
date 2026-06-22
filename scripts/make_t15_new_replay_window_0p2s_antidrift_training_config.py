#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokamak_rl_v2.config import load_experiment_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the 0.2s T15 replay-window anti-drift training config.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--eval-steps", type=int, required=True)
    parser.add_argument("--checkpoint-steps", type=int, required=True)
    parser.add_argument("--replay-capacity-episodes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--rollout-chunk-length", type=int, required=True)
    parser.add_argument("--updates-per-rollout-chunk", type=int, required=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    base_path = Path(args.base_config)
    cfg = json.loads(base_path.read_text(encoding="utf-8"))
    cfg["name"] = "t15_new_replay_window_0p2s_tcvjdot_t1q1a1_antidrift_64gpu_10m"

    cfg["sim"]["config_path"] = "/workspace/tokamak-sim/configs/T15MD_new_data.toml"
    cfg["sim"]["csv_initial_state_library"] = "/workspace/tokamak-rl-v2/data/processed/t15_new_replay_window_0p1s_initial_states.npz"
    cfg["sim"]["csv_initial_state_split"] = "train"
    cfg["reference"]["boundary"]["replay_reference_dir"] = "/workspace/tokamak-sim/runs/t15md_limited_replay_dataset_sigmaL_3856_3857_3858_3863_3864"

    reward = cfg["reward"]
    reward["kind"] = "tcv_derivative"
    reward["terminal_reward"] = -20.0
    reward["terminal_remaining_cost"] = 2.0
    reward["shape_mean_weight"] = 3.2
    reward["shape_max_weight"] = 0.8
    reward["ip_weight"] = 1.8
    reward["current_weight"] = 0.75
    reward["derivative_weight"] = 0.1875
    reward["current_drift_weight"] = 1.5
    reward["current_drift_bad_fraction"] = 0.10
    reward["mean_jdot_bias_weight"] = 1.0
    reward["mean_jdot_bias_bad_fraction"] = 0.08
    reward["actuator_saturation_weight"] = 0.1875
    reward["reward_scale"] = 0.01
    reward["smoothmax_alpha"] = -5.0
    reward["ip_scale_a"] = 25000.0

    training = cfg["training"]
    training["steps"] = int(args.steps)
    training["num_envs"] = int(args.num_envs)
    training["output_dir"] = f"/workspace/tokamak-rl-v2/{args.output_dir}"
    training["save_checkpoints"] = bool(args.save_checkpoints)
    training["checkpoint_interval_steps"] = int(args.checkpoint_steps)
    training["milestone_checkpoint_interval_steps"] = int(args.checkpoint_steps)
    training["eval_checkpoint_top_k"] = 10
    training["keep_latest_checkpoint"] = True
    training["eval_interval_steps"] = int(args.eval_steps)
    training["eval_max_steps"] = int(cfg["sim"]["max_episode_steps"])
    training["distributed_mode"] = "local_replay"
    training["actor_workers"] = 1
    training["production_mode"] = True

    learner = cfg["learner"]
    learner["replay_capacity_episodes"] = int(args.replay_capacity_episodes)
    learner["batch_size"] = int(args.batch_size)
    learner["rollout_chunk_length"] = int(args.rollout_chunk_length)
    learner["updates_per_rollout_chunk"] = int(args.updates_per_rollout_chunk)

    _check(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    load_experiment_config(out)
    print(out)
    return 0


def _check(cfg: dict) -> None:
    episode_steps = int(cfg["sim"]["max_episode_steps"])
    preview_steps = int(cfg["observation"]["target_preview_steps"])
    preview_stride = int(cfg["observation"]["target_preview_stride"])
    checks = {
        "reference.ip.kind": cfg["reference"]["ip"]["kind"] == "replay_window",
        "reference.boundary.kind": cfg["reference"]["boundary"]["kind"] == "t15_replay_segment_conditioned",
        "reference.duration_s": float(cfg["reference"]["duration_s"]) == 0.2,
        "reference.t_step": float(cfg["reference"]["t_step"]) == 0.001,
        "sim.max_episode_steps": episode_steps == 200,
        "training.eval_max_steps": int(cfg["training"]["eval_max_steps"]) == 200,
        "preview_shape_matches_warm_start": preview_steps == 10 and preview_stride == 10,
        "preview_within_episode": preview_steps * preview_stride <= episode_steps,
        "reward.kind": cfg["reward"]["kind"] == "tcv_derivative",
        "sim.action_contract": cfg["sim"]["action_contract"] == "jdot_command",
        "sim.no_delta_derivative_limits": "delta_derivative_limits_aps" not in cfg["sim"],
        "sim.terminate_on_boundary_loss": cfg["sim"]["terminate_on_boundary_loss"] is True,
        "sim.terminate_on_current_limit": cfg["sim"]["terminate_on_current_limit"] is True,
        "sim.current_hard_termination_fraction": float(cfg["sim"]["current_hard_termination_fraction"]) == 1.2,
        "sim.current_termination_grace_steps": int(cfg["sim"]["current_termination_grace_steps"]) == 1,
        "sim.current_saturation_fraction": float(cfg["sim"]["current_saturation_fraction"]) == 1.0,
        "reward.terminal_reward": float(cfg["reward"]["terminal_reward"]) == -20.0,
        "reward.terminal_remaining_cost": float(cfg["reward"]["terminal_remaining_cost"]) == 2.0,
        "reward.current_drift_weight": float(cfg["reward"]["current_drift_weight"]) == 1.5,
        "reward.current_drift_bad_fraction": float(cfg["reward"]["current_drift_bad_fraction"]) == 0.10,
        "reward.mean_jdot_bias_weight": float(cfg["reward"]["mean_jdot_bias_weight"]) == 1.0,
        "reward.mean_jdot_bias_bad_fraction": float(cfg["reward"]["mean_jdot_bias_bad_fraction"]) == 0.08,
        "reward.reward_scale": float(cfg["reward"]["reward_scale"]) == 0.01,
        "reward.smoothmax_alpha": float(cfg["reward"]["smoothmax_alpha"]) == -5.0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"generated 0.2s anti-drift config failed sanity checks: {failed}")


if __name__ == "__main__":
    raise SystemExit(main())
