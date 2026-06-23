#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate canonical 0.1s replay-window oracle training config.")
    ap.add_argument("--base-config", default="configs/experiments/t15_new_replay_window_0p1s_tcvjdot_mpo.yaml")
    ap.add_argument("--output-config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reward-preset", choices=("t1q1a1", "balanced"), default="t1q1a1")
    ap.add_argument("--steps", type=int, default=10000000)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--eval-interval-steps", type=int, default=500000)
    ap.add_argument("--checkpoint-interval-steps", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--action-samples", type=int, default=64)
    ap.add_argument("--early-stop-patience-evals", type=int, default=5)
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    cfg["name"] = f"t15_new_replay_window_0p1s_tcvjdot_{args.reward_preset}_oracle"
    cfg["training"]["steps"] = int(args.steps)
    cfg["training"]["num_envs"] = int(args.num_envs)
    cfg["training"]["output_dir"] = str(args.output_dir)
    cfg["training"]["eval_interval_steps"] = int(args.eval_interval_steps)
    cfg["training"]["checkpoint_interval_steps"] = int(args.checkpoint_interval_steps)
    cfg["training"]["milestone_checkpoint_interval_steps"] = int(args.checkpoint_interval_steps)
    cfg["training"]["eval_checkpoint_top_k"] = 10
    cfg["training"]["keep_latest_checkpoint"] = True
    cfg["training"]["distributed_mode"] = "local_replay"
    cfg["training"]["production_mode"] = True
    cfg["training"]["eval_max_steps"] = 100
    cfg["training"]["save_checkpoints"] = True
    cfg["training"]["seed"] = int(args.seed)
    cfg["training"]["early_stop_patience_evals"] = int(args.early_stop_patience_evals)
    cfg["training"]["early_stop_min_delta"] = 0.0

    cfg["sim"]["config_path"] = "/workspace/tokamak-sim/configs/T15MD_new_data.toml"
    cfg["sim"]["csv_initial_state_library"] = "/workspace/tokamak-rl-v2/data/processed/t15_new_replay_window_0p1s_oracle_initial_states.npz"
    cfg["sim"]["csv_initial_state_split"] = "train"
    cfg["sim"]["action_contract"] = "jdot_command"
    cfg["reference"]["boundary"]["replay_reference_dir"] = "/workspace/tokamak-rl-v2/data/processed/t15_new_replay_window_0p1s_oracle_targets"
    cfg["observation"]["actor_kind"] = "controller_state_v6"
    cfg["observation"]["critic_kind"] = "compact_training_state_v2"
    cfg["observation"]["ip_rate_scale_aps"] = 500000.0
    cfg["observation"]["boundary_rate_scale_mps"] = 1.0
    cfg["learner"]["unroll_length"] = 100
    cfg["learner"]["min_replay_sequence_length"] = 100
    cfg["learner"]["rollout_chunk_length"] = 100
    cfg["learner"]["batch_size"] = 32
    cfg["learner"]["updates_per_rollout_chunk"] = 64
    cfg["learner"]["replay_capacity_episodes"] = 1024
    cfg["learner"]["action_samples"] = int(args.action_samples)

    reward = cfg["reward"]
    reward["kind"] = "tcv_derivative"
    reward["terminal_reward"] = -20.0
    reward["current_weight"] = 0.75
    reward["derivative_weight"] = 0.1875
    reward["actuator_saturation_weight"] = 0.1875
    reward["reward_scale"] = 0.01
    if args.reward_preset == "balanced":
        reward["shape_mean_weight"] = 2.4
        reward["shape_max_weight"] = 0.6
        reward["ip_weight"] = 3.0
        reward["ip_scale_a"] = 15000.0
        reward["smoothmax_alpha"] = -1.0
    else:
        reward["shape_mean_weight"] = 3.2
        reward["shape_max_weight"] = 0.8
        reward["ip_weight"] = 1.8
        reward["ip_scale_a"] = 25000.0
        reward["smoothmax_alpha"] = -5.0

    _validate(cfg, reward_preset=args.reward_preset)
    out = Path(args.output_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(out)
    return 0


def _validate(cfg: dict[str, object], *, reward_preset: str) -> None:
    checks = {
        "reference.ip.kind": cfg["reference"]["ip"]["kind"] == "replay_window",
        "reference.boundary.kind": cfg["reference"]["boundary"]["kind"] == "t15_replay_segment_conditioned",
        "oracle_reference_dir": "oracle_targets" in cfg["reference"]["boundary"]["replay_reference_dir"],
        "oracle_reset_library": "oracle_initial_states" in cfg["sim"]["csv_initial_state_library"],
        "sim.action_contract": cfg["sim"]["action_contract"] == "jdot_command",
        "observation.actor_kind": cfg["observation"]["actor_kind"] == "controller_state_v6",
        "observation.critic_kind": cfg["observation"]["critic_kind"] == "compact_training_state_v2",
        "episode_steps": int(cfg["sim"]["max_episode_steps"]) == 100,
        "duration": float(cfg["reference"]["duration_s"]) == 0.1,
        "unroll_length": int(cfg["learner"]["unroll_length"]) == 100,
        "min_replay_sequence_length": int(cfg["learner"]["min_replay_sequence_length"]) == 100,
        "rollout_chunk_length": int(cfg["learner"]["rollout_chunk_length"]) == 100,
        "batch_size": int(cfg["learner"]["batch_size"]) == 32,
        "action_samples": int(cfg["learner"]["action_samples"]) == 64,
        "num_envs": int(cfg["training"]["num_envs"]) == 2048,
        "production_mode": cfg["training"]["production_mode"] is True,
        "reward.kind": cfg["reward"]["kind"] == "tcv_derivative",
    }
    if reward_preset == "balanced":
        checks["balanced_ip_weight"] = float(cfg["reward"]["ip_weight"]) == 3.0
        checks["balanced_ip_scale"] = float(cfg["reward"]["ip_scale_a"]) == 15000.0
        checks["balanced_smoothmax"] = float(cfg["reward"]["smoothmax_alpha"]) == -1.0
    else:
        checks["center_ip_weight"] = float(cfg["reward"]["ip_weight"]) == 1.8
        checks["center_ip_scale"] = float(cfg["reward"]["ip_scale_a"]) == 25000.0
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"oracle training config failed checks: {failed}")


if __name__ == "__main__":
    raise SystemExit(main())
