from __future__ import annotations

import argparse
from dataclasses import replace

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.training.trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    ap.add_argument("--output-dir", default=None)
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
    ap.add_argument("--checkpoint-interval-steps", type=int, default=None)
    ap.add_argument("--eval-interval-steps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--eval-max-steps", type=int, default=None)
    ap.add_argument("--actor-workers", type=int, default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="tokamak-rl-v2")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    if args.sim_compute_backend is not None or args.sim_gpu_device is not None:
        cfg = replace(cfg, sim=replace(cfg.sim, compute_backend=args.sim_compute_backend or cfg.sim.compute_backend, gpu_device=args.sim_gpu_device or cfg.sim.gpu_device))
    if any(v is not None for v in (args.batch_size, args.unroll_length, args.replay_capacity_episodes, args.rollout_chunk_length, args.updates_per_rollout_chunk, args.action_samples)):
        cfg = replace(
            cfg,
            learner=replace(
                cfg.learner,
                batch_size=args.batch_size or cfg.learner.batch_size,
                unroll_length=args.unroll_length or cfg.learner.unroll_length,
                replay_capacity_episodes=args.replay_capacity_episodes or cfg.learner.replay_capacity_episodes,
                rollout_chunk_length=args.rollout_chunk_length or cfg.learner.rollout_chunk_length,
                updates_per_rollout_chunk=args.updates_per_rollout_chunk or cfg.learner.updates_per_rollout_chunk,
                action_samples=args.action_samples or cfg.learner.action_samples,
            ),
        )
    if any(v is not None for v in (args.hidden_dim, args.critic_hidden_dim, args.critic_mlp_hidden_dim)):
        cfg = replace(
            cfg,
            network=replace(
                cfg.network,
                hidden_dim=args.hidden_dim or cfg.network.hidden_dim,
                critic_hidden_dim=args.critic_hidden_dim or cfg.network.critic_hidden_dim,
                critic_mlp_hidden_dim=args.critic_mlp_hidden_dim or cfg.network.critic_mlp_hidden_dim,
            ),
        )
    if any(v is not None for v in (args.checkpoint_interval_steps, args.eval_interval_steps, args.eval_episodes, args.eval_max_steps, args.actor_workers)):
        cfg = replace(
            cfg,
            training=replace(
                cfg.training,
                checkpoint_interval_steps=args.checkpoint_interval_steps or cfg.training.checkpoint_interval_steps,
                eval_interval_steps=args.eval_interval_steps or cfg.training.eval_interval_steps,
                eval_episodes=args.eval_episodes or cfg.training.eval_episodes,
                eval_max_steps=args.eval_max_steps or cfg.training.eval_max_steps,
                actor_workers=args.actor_workers or cfg.training.actor_workers,
            ),
        )
    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled":
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name or cfg.name, mode=args.wandb_mode, config={"experiment": cfg.name})
    trainer = Trainer(cfg, steps=args.steps, num_envs=args.num_envs, device=args.device, output_dir=args.output_dir, wandb_run=wandb_run)
    result = trainer.train()
    if wandb_run is not None:
        wandb_run.finish()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
