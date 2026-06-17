#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return raw


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cleanup(output_dir: Path) -> None:
    shutil.rmtree(output_dir / "checkpoints", ignore_errors=True)
    shutil.rmtree(output_dir / "exports", ignore_errors=True)


def _write_failure(output_dir: Path, *, status: str, reason: str, return_code: int | None = None) -> None:
    path = output_dir / "policy_validation.json"
    if path.exists():
        return
    payload: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "output_dir": str(output_dir),
    }
    if return_code is not None:
        payload["return_code"] = int(return_code)
    _write_json(path, payload)


def run_candidate(args: argparse.Namespace) -> int:
    manifest = _load_json(args.manifest)
    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise ValueError(f"{args.manifest} has no variants list")
    variant_index = int(args.variant_index)
    if variant_index < 0 or variant_index >= len(variants):
        print(f"variant_index={variant_index} out of range; skipping")
        return 0
    variant = variants[variant_index]
    if not isinstance(variant, dict):
        raise ValueError(f"Variant {variant_index} is not an object")

    folder = str(variant["folder"])
    output_dir = args.sweep_root / folder
    generated_config = args.sweep_root / "generated_configs" / f"{folder}.json"
    base_path = args.base_config

    cfg = _load_json(base_path)
    base_name = str(cfg.get("name") or base_path.stem)
    cfg["name"] = f"{base_name}_{folder}"
    cfg.setdefault("sim", {})["config_path"] = str(args.sim_config_path)
    cfg["sim"]["csv_initial_state_library"] = str(args.initial_state_library)
    cfg.setdefault("reference", {}).setdefault("ip", {})["limits_path"] = str(args.reference_limits)
    cfg.setdefault("reward", {}).update(variant["reward"])
    cfg.setdefault("training", {})["steps"] = int(args.train_env_steps)
    cfg["training"]["output_dir"] = str(output_dir)
    cfg["training"]["save_checkpoints"] = False
    cfg["training"]["checkpoint_interval_steps"] = max(1, int(args.train_env_steps))
    cfg["training"]["eval_interval_steps"] = max(1, int(args.eval_env_steps))
    cfg["training"]["num_envs"] = int(args.num_envs)
    cfg.setdefault("learner", {})["replay_capacity_episodes"] = int(args.replay_capacity_episodes)
    cfg["learner"]["batch_size"] = int(args.batch_size)
    cfg["learner"]["rollout_chunk_length"] = int(args.rollout_chunk_length)
    cfg["learner"]["updates_per_rollout_chunk"] = int(args.updates_per_rollout_chunk)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    generated_config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    _write_json(
        output_dir / "reward_variant.json",
        {
            "variant_index": variant_index,
            "variant": variant,
            "manifest": str(args.manifest),
            "base_config": str(base_path),
            "generated_config": str(generated_config),
            "output_dir": str(output_dir),
        },
    )

    print(f"variant_index={variant_index}")
    print(f"variant_name={variant.get('name', folder)}")
    print(f"variant_folder={folder}")
    print(f"generated_config={generated_config}")
    print(f"train_output={output_dir}")
    print(f"run_name={folder}")
    sys.stdout.flush()

    cmd = [
        sys.executable,
        "scripts/train_policy_pipeline.py",
        "--config",
        str(generated_config),
        "--output-dir",
        str(output_dir),
        "--steps",
        str(int(args.train_env_steps)),
        "--device",
        args.device,
        "--sim-compute-backend",
        "gpu",
        "--sim-gpu-device",
        args.device,
        "--num-envs",
        str(int(args.num_envs)),
        "--distributed-mode",
        "local_replay",
        "--actor-workers",
        "1",
        "--batch-size",
        str(int(args.batch_size)),
        "--replay-capacity-episodes",
        str(int(args.replay_capacity_episodes)),
        "--rollout-chunk-length",
        str(int(args.rollout_chunk_length)),
        "--updates-per-rollout-chunk",
        str(int(args.updates_per_rollout_chunk)),
        "--hidden-dim",
        "256",
        "--critic-hidden-dim",
        "256",
        "--critic-mlp-hidden-dim",
        "256",
        "--action-samples",
        "20",
        "--actor-update-chunk-size",
        "2048",
        "--no-save-checkpoints",
        "--eval-interval-steps",
        str(int(args.eval_env_steps)),
        "--eval-episodes",
        "128",
        "--eval-max-steps",
        "500",
        "--controller-rollout-steps",
        "0",
        "--reward-sweep-mode",
        "--no-export",
        "--wandb",
        "--wandb-project",
        args.wandb_project,
        "--wandb-name",
        folder,
        "--wandb-group",
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-optional",
    ]

    env = os.environ.copy()
    env.setdefault("WANDB__SERVICE_WAIT", "30")
    try:
        result = subprocess.run(cmd, env=env, check=False)
    finally:
        _cleanup(output_dir)
    print(f"variant_index={variant_index} run_status={result.returncode}")
    if result.returncode != 0:
        _write_failure(
            output_dir,
            status="sweep_failed_training",
            reason="train_policy_pipeline.py returned nonzero",
            return_code=int(result.returncode),
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one fault-isolated reward sweep candidate.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant-index", type=int, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--train-env-steps", type=int, required=True)
    parser.add_argument("--eval-env-steps", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--replay-capacity-episodes", type=int, required=True)
    parser.add_argument("--rollout-chunk-length", type=int, required=True)
    parser.add_argument("--updates-per-rollout-chunk", type=int, required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-config-path", type=Path, default=Path("/workspace/tokamak-sim/configs/T15MD_new_data.toml"))
    parser.add_argument("--initial-state-library", type=Path, default=Path("/workspace/tokamak-rl-v2/data/processed/t15_csv_initial_states.npz"))
    parser.add_argument("--reference-limits", type=Path, default=Path("/workspace/tokamak-rl-v2/data/processed/t15_reference_limits.json"))
    args = parser.parse_args(argv)
    return run_candidate(args)


if __name__ == "__main__":
    raise SystemExit(main())
