from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.config.loader import _validate_experiment_config
from tokamak_rl_v2.config.schema import RewardConfig
from tokamak_rl_v2.training.trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume-checkpoint", default=None)
    ap.add_argument("--reward-config", default=None, help="JSON/YAML reward override, for example outputs/.../best_reward.yaml")
    ap.add_argument("--sim-compute-backend", choices=("cpu", "gpu"), default=None)
    ap.add_argument("--sim-gpu-device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--unroll-length", type=int, default=None)
    ap.add_argument("--replay-capacity-episodes", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--critic-hidden-dim", type=int, default=None)
    ap.add_argument("--critic-mlp-hidden-dim", type=int, default=None)
    ap.add_argument("--rollout-chunk-length", type=int, default=None)
    ap.add_argument("--updates-per-rollout-chunk", type=int, default=None)
    ap.add_argument("--action-samples", type=int, default=None)
    ap.add_argument("--actor-update-chunk-size", type=int, default=None)
    ap.add_argument("--checkpoint-interval-steps", type=int, default=None)
    ap.add_argument("--eval-interval-steps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--eval-max-steps", type=int, default=None)
    ap.add_argument("--actor-workers", type=int, default=None)
    ap.add_argument("--actor-devices", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="tokamak-rl-v2")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    if args.reward_config is not None:
        cfg = replace(cfg, reward=_load_reward_override(args.reward_config, base=cfg.reward))
    if args.sim_compute_backend is not None or args.sim_gpu_device is not None:
        cfg = replace(cfg, sim=replace(cfg.sim, compute_backend=args.sim_compute_backend if args.sim_compute_backend is not None else cfg.sim.compute_backend, gpu_device=args.sim_gpu_device if args.sim_gpu_device is not None else cfg.sim.gpu_device))
    if any(v is not None for v in (args.batch_size, args.unroll_length, args.replay_capacity_episodes, args.rollout_chunk_length, args.updates_per_rollout_chunk, args.action_samples, args.actor_update_chunk_size)):
        cfg = replace(
            cfg,
            learner=replace(
                cfg.learner,
                batch_size=args.batch_size if args.batch_size is not None else cfg.learner.batch_size,
                unroll_length=args.unroll_length if args.unroll_length is not None else cfg.learner.unroll_length,
                replay_capacity_episodes=args.replay_capacity_episodes if args.replay_capacity_episodes is not None else cfg.learner.replay_capacity_episodes,
                rollout_chunk_length=args.rollout_chunk_length if args.rollout_chunk_length is not None else cfg.learner.rollout_chunk_length,
                updates_per_rollout_chunk=args.updates_per_rollout_chunk if args.updates_per_rollout_chunk is not None else cfg.learner.updates_per_rollout_chunk,
                action_samples=args.action_samples if args.action_samples is not None else cfg.learner.action_samples,
                actor_update_chunk_size=args.actor_update_chunk_size if args.actor_update_chunk_size is not None else cfg.learner.actor_update_chunk_size,
            ),
        )
    if any(v is not None for v in (args.hidden_dim, args.critic_hidden_dim, args.critic_mlp_hidden_dim)):
        cfg = replace(
            cfg,
            network=replace(
                cfg.network,
                hidden_dim=args.hidden_dim if args.hidden_dim is not None else cfg.network.hidden_dim,
                critic_hidden_dim=args.critic_hidden_dim if args.critic_hidden_dim is not None else cfg.network.critic_hidden_dim,
                critic_mlp_hidden_dim=args.critic_mlp_hidden_dim if args.critic_mlp_hidden_dim is not None else cfg.network.critic_mlp_hidden_dim,
            ),
        )
    if any(v is not None for v in (args.checkpoint_interval_steps, args.eval_interval_steps, args.eval_episodes, args.eval_max_steps, args.actor_workers, args.actor_devices)):
        cfg = replace(
            cfg,
            training=replace(
                cfg.training,
                checkpoint_interval_steps=args.checkpoint_interval_steps if args.checkpoint_interval_steps is not None else cfg.training.checkpoint_interval_steps,
                eval_interval_steps=args.eval_interval_steps if args.eval_interval_steps is not None else cfg.training.eval_interval_steps,
                eval_episodes=args.eval_episodes if args.eval_episodes is not None else cfg.training.eval_episodes,
                eval_max_steps=args.eval_max_steps if args.eval_max_steps is not None else cfg.training.eval_max_steps,
                actor_workers=args.actor_workers if args.actor_workers is not None else cfg.training.actor_workers,
                actor_devices=_device_list(args.actor_devices) if args.actor_devices is not None else cfg.training.actor_devices,
            ),
        )
    if args.steps is not None and int(args.steps) <= 0:
        raise ValueError("--steps must be positive")
    if args.num_envs is not None and int(args.num_envs) <= 0:
        raise ValueError("--num-envs must be positive")
    _validate_experiment_config(cfg)
    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled":
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name or cfg.name, mode=args.wandb_mode, config={"experiment": cfg.name})
    trainer = Trainer(cfg, steps=args.steps, num_envs=args.num_envs, device=args.device, output_dir=args.output_dir, wandb_run=wandb_run, resume_checkpoint=args.resume_checkpoint)
    result = trainer.train()
    if wandb_run is not None:
        wandb_run.finish()
    print(result)
    return 0



def _load_reward_override(path: str, *, base: RewardConfig) -> RewardConfig:
    source = Path(path).resolve()
    text = source.read_text(encoding="utf-8")
    raw: object
    if source.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - only used when PyYAML is missing locally
            raise RuntimeError(f"reward override requires JSON or PyYAML for YAML files: {source}") from exc
        raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ValueError(f"reward override must be a mapping: {source}")
    values = {}
    for field in RewardConfig.__dataclass_fields__:
        default = getattr(base, field)
        value = raw.get(field, default)
        values[field] = str(value) if isinstance(default, str) else float(value)
    reward = RewardConfig(**values)
    if reward.shape_bad_m <= reward.shape_good_m:
        raise ValueError("reward override must have shape_bad_m > shape_good_m")
    if reward.ip_bad_a <= reward.ip_good_a:
        raise ValueError("reward override must have ip_bad_a > ip_good_a")
    if reward.current_bad_a <= reward.current_good_a:
        raise ValueError("reward override must have current_bad_a > current_good_a")
    if reward.derivative_bad <= reward.derivative_good:
        raise ValueError("reward override must have derivative_bad > derivative_good")
    if reward.reward_scale <= 0.0:
        raise ValueError("reward override must have positive reward_scale")
    if reward.tracking_combiner not in {"smooth_min", "weighted_mean", "geometric_mean", "product"}:
        raise ValueError(f"reward override has unsupported tracking_combiner: {reward.tracking_combiner}")
    if reward.shape_aggregator not in {"smooth_worst", "mean", "geometric_mean"}:
        raise ValueError(f"reward override has unsupported shape_aggregator: {reward.shape_aggregator}")
    return reward

def _device_list(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("device list must not be empty")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
