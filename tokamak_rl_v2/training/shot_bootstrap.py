from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tokamak_rl_v2.config.schema import ExperimentConfig
from tokamak_rl_v2.training.trainer import Trainer


@dataclass(frozen=True, slots=True)
class ShotBootstrapConfig:
    steps: int = 2000
    lr: float = 1.0e-4
    target_std: float = 0.08
    std_loss_weight: float = 0.05
    log_interval: int = 50


def run_shot_bootstrap(
    trainer: Trainer,
    *,
    config: ExperimentConfig,
    bootstrap: ShotBootstrapConfig,
    output_dir: Path,
    wandb_run=None,
) -> dict[str, Any]:
    """Pretrain actor from idealized shot-current trajectories and seed replay."""

    if config.sim.shot_fragments is None:
        raise ValueError("shot bootstrap requires sim.shot_fragments")
    if int(config.training.actor_workers) > 1:
        raise ValueError("shot bootstrap currently supports actor_workers=1")
    if int(bootstrap.steps) <= 0:
        return {"enabled": False, "steps": 0}

    env = trainer.env
    obs = env.reset()
    optimizer = torch.optim.Adam(trainer.actor.parameters(), lr=float(bootstrap.lr))

    metrics_path = output_dir / "shot_bootstrap_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "loss",
        "mse_loss",
        "std_loss",
        "teacher_action_rms",
        "actor_action_rms",
        "action_abs_error",
        "mean_reward",
        "replay_size",
        "elapsed_s",
        "steps_per_second",
    ]
    last_row: dict[str, float] = {}
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        start_time = time.monotonic()
        last_log_time = start_time
        last_log_step = 0
        for step in range(1, int(bootstrap.steps) + 1):
            with torch.no_grad():
                teacher_action = env.shot_fragment_teacher_action()

            obs_for_update = obs
            actor_out = trainer.actor(obs_for_update)
            mse_loss = F.mse_loss(torch.tanh(actor_out.mean), teacher_action)
            target_std = torch.full_like(actor_out.std, float(bootstrap.target_std))
            std_loss = F.mse_loss(actor_out.std, target_std)
            loss = mse_loss + float(bootstrap.std_loss_weight) * std_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.actor.parameters(), 10.0)
            optimizer.step()

            with torch.no_grad():
                actor_mean_after = trainer.actor.deterministic(obs_for_update)
                batch_step = env.step(teacher_action)
                discount = torch.full((trainer.num_envs,), float(config.learner.discount), dtype=torch.float32, device=trainer.device)
                done = batch_step.terminated | batch_step.truncated
                trainer.replay.add_batch(obs, batch_step.applied_action, batch_step.reward, discount, batch_step.obs, done)
                obs = env.reset_indices(done) if bool(torch.any(done).item()) else batch_step.obs

            if step % max(int(bootstrap.log_interval), 1) == 0 or step == 1 or step == int(bootstrap.steps):
                with torch.no_grad():
                    abs_error = torch.mean(torch.abs(actor_mean_after - teacher_action))
                    now = time.monotonic()
                    interval_s = max(now - last_log_time, 1.0e-9)
                    interval_steps = max(step - last_log_step, 1)
                    row = {
                        "step": float(step),
                        "loss": float(loss.detach().cpu()),
                        "mse_loss": float(mse_loss.detach().cpu()),
                        "std_loss": float(std_loss.detach().cpu()),
                        "teacher_action_rms": float(torch.sqrt(torch.mean(teacher_action.pow(2))).detach().cpu()),
                        "actor_action_rms": float(torch.sqrt(torch.mean(actor_mean_after.pow(2))).detach().cpu()),
                        "action_abs_error": float(abs_error.detach().cpu()),
                        "mean_reward": float(torch.mean(batch_step.reward).detach().cpu()),
                        "replay_size": float(trainer.replay.size),
                        "elapsed_s": float(now - start_time),
                        "steps_per_second": float(interval_steps / interval_s),
                    }
                writer.writerow(row)
                f.flush()
                last_row = row
                last_log_time = now
                last_log_step = step
                print(
                    "shot_bootstrap "
                    f"step={step}/{int(bootstrap.steps)} "
                    f"loss={row['loss']:.6g} "
                    f"teacher_rms={row['teacher_action_rms']:.4f} "
                    f"actor_rms={row['actor_action_rms']:.4f} "
                    f"abs_error={row['action_abs_error']:.4f} "
                    f"steps_per_second={row['steps_per_second']:.3f}",
                    flush=True,
                )
                if wandb_run is not None:
                    env_step = int(step) * int(trainer.num_envs)
                    wandb_run.log({"global_step": env_step, **{f"bootstrap/{k}": v for k, v in row.items() if k != "step"}}, step=env_step)

    trainer.target_actor.load_state_dict(trainer.actor.state_dict())
    return {
        "enabled": True,
        "kind": "idealized_shot_current_behavior_cloning",
        "steps": int(bootstrap.steps),
        "env_steps": int(bootstrap.steps) * int(trainer.num_envs),
        "metrics_csv": str(metrics_path),
        "final_metrics": last_row,
        "replay_size": int(trainer.replay.size),
    }
