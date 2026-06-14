from __future__ import annotations

import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tokamak_control.control.hinf_joint import HinftyJointController
from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections

from tokamak_rl_v2.config.schema import ExperimentConfig
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.training.trainer import Trainer


@dataclass(frozen=True, slots=True)
class HinfBootstrapConfig:
    steps: int = 5000
    lr: float = 1.0e-4
    q_error: float = 3.0e6
    q_ip: float = 1.0e7
    gamma: float = 7905.694150420949
    u_clip: float = 2.0e6
    j_curr: float = 0.0
    target_std: float = 0.08
    std_loss_weight: float = 0.05
    log_interval: int = 100
    reset_interval_steps: int = 250


def run_hinf_bootstrap(
    trainer: Trainer,
    *,
    config: ExperimentConfig,
    bootstrap: HinfBootstrapConfig,
    output_dir: Path,
    wandb_run=None,
) -> dict[str, Any]:
    """Pretrain actor on H∞ teacher fragments and seed replay with teacher data."""

    if int(config.training.actor_workers) > 1:
        raise ValueError("H∞ bootstrap currently supports actor_workers=1")
    if int(bootstrap.steps) <= 0:
        return {"enabled": False, "steps": 0}

    teacher_cfg = _cpu_teacher_config(config)
    teacher_env = TokamakMagneticControlEnv(
        teacher_cfg,
        batch_size=trainer.num_envs,
        device=torch.device("cpu"),
        seed=int(config.training.seed) + 310000,
    )
    controllers = [_new_teacher_controller(bootstrap) for _ in range(trainer.num_envs)]
    obs_cpu = teacher_env.reset()
    for controller in controllers:
        controller.reset()

    optimizer = torch.optim.Adam(trainer.actor.parameters(), lr=float(bootstrap.lr))
    metrics_path = output_dir / "hinf_bootstrap_metrics.csv"
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
            teacher_action_cpu = _hinf_teacher_action(teacher_env, controllers, bootstrap)
            obs = obs_cpu.to(device=trainer.device, dtype=torch.float32)
            teacher_action = teacher_action_cpu.to(device=trainer.device, dtype=torch.float32)

            actor_out = trainer.actor(obs)
            mse_loss = F.mse_loss(actor_out.mean, teacher_action)
            target_std = torch.full_like(actor_out.std, float(bootstrap.target_std))
            std_loss = F.mse_loss(actor_out.std, target_std)
            loss = mse_loss + float(bootstrap.std_loss_weight) * std_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.actor.parameters(), 10.0)
            optimizer.step()

            with torch.no_grad():
                out = teacher_env.step(teacher_action_cpu)
                done = out.terminated | out.truncated
                if int(bootstrap.reset_interval_steps) > 0 and step % int(bootstrap.reset_interval_steps) == 0:
                    done = torch.ones_like(done, dtype=torch.bool)
                discount = torch.full((trainer.num_envs,), float(config.learner.discount), dtype=torch.float32, device=trainer.device)
                trainer.replay.add_batch(
                    obs_cpu.to(device=trainer.device, dtype=torch.float32),
                    teacher_action,
                    out.reward.to(device=trainer.device, dtype=torch.float32),
                    discount,
                    out.obs.to(device=trainer.device, dtype=torch.float32),
                    done.to(device=trainer.device, dtype=torch.bool),
                )
                if bool(torch.any(done).item()):
                    done_cpu = done.detach().cpu().numpy().astype(bool)
                    for index, is_done in enumerate(done_cpu.tolist()):
                        if is_done:
                            controllers[index].reset()
                    obs_cpu = teacher_env.reset_indices(done)
                else:
                    obs_cpu = out.obs

            if step % max(int(bootstrap.log_interval), 1) == 0 or step == 1 or step == int(bootstrap.steps):
                with torch.no_grad():
                    actor_mean = trainer.actor(obs).mean
                    abs_error = torch.mean(torch.abs(actor_mean - teacher_action))
                    now = time.monotonic()
                    interval_s = max(now - last_log_time, 1.0e-9)
                    interval_steps = max(step - last_log_step, 1)
                    row = {
                        "step": float(step),
                        "loss": float(loss.detach().cpu()),
                        "mse_loss": float(mse_loss.detach().cpu()),
                        "std_loss": float(std_loss.detach().cpu()),
                        "teacher_action_rms": float(torch.sqrt(torch.mean(teacher_action.pow(2))).detach().cpu()),
                        "actor_action_rms": float(torch.sqrt(torch.mean(actor_mean.pow(2))).detach().cpu()),
                        "action_abs_error": float(abs_error.detach().cpu()),
                        "mean_reward": float(torch.mean(out.reward).detach().cpu()),
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
                    "hinf_bootstrap "
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
        "steps": int(bootstrap.steps),
        "env_steps": int(bootstrap.steps) * int(trainer.num_envs),
        "teacher": {
            "controller": "hinf_joint",
            "q_error": float(bootstrap.q_error),
            "q_ip": float(bootstrap.q_ip),
            "gamma": float(bootstrap.gamma),
            "u_clip": float(bootstrap.u_clip),
            "j_curr": float(bootstrap.j_curr),
            "reset_interval_steps": int(bootstrap.reset_interval_steps),
        },
        "metrics_csv": str(metrics_path),
        "final_metrics": last_row,
        "replay_size": int(trainer.replay.size),
    }


def _cpu_teacher_config(config: ExperimentConfig) -> ExperimentConfig:
    return replace(
        config,
        sim=replace(config.sim, compute_backend="cpu", gpu_device="cpu"),
        training=replace(config.training, actor_workers=1, actor_devices=()),
    )


def _new_teacher_controller(config: HinfBootstrapConfig) -> HinftyJointController:
    return HinftyJointController(
        q_error=float(config.q_error),
        q_ip=float(config.q_ip),
        gamma=float(config.gamma),
        u_clip=float(config.u_clip),
        j_curr=float(config.j_curr),
    )


@torch.no_grad()
def _hinf_teacher_action(env: TokamakMagneticControlEnv, controllers: list[HinftyJointController], config: HinfBootstrapConfig) -> Tensor:
    if env.config.sim.compute_backend != "cpu":
        raise ValueError("H∞ teacher action generation requires a CPU env")
    if env.reference is None:
        raise RuntimeError("teacher env has not been reset")

    ip_ref, _ref_points, ref_radii = env._reference_at()
    ip_ref_np = ip_ref.detach().cpu().numpy().astype(float)
    ref_radii_np = ref_radii.detach().cpu().numpy().astype(float)
    actions: list[np.ndarray] = []
    center = (float(env.cfg.physics.R0), float(env.cfg.physics.Z0))
    for index, model in enumerate(env._cpu_models):
        try:
            boundary_poly, _level, _status = find_plasma_boundary_with_status(
                model.state.psi,
                model.grid,
                center,
                n_levels=80,
                limiter_shape=env.cfg.limiter_shape,
                boundary_mode=env.cfg.boundary_mode,
            )
            _ = radii_from_polyline_ray_intersections(boundary_poly, center, env.angles)
        except BoundaryNotFoundError:
            boundary_poly = None
        control = controllers[index].compute_control(
            model=model,
            psi=np.asarray(model.state.psi, dtype=float),
            boundary_poly=boundary_poly,
            center=center,
            measure_angles=env.angles,
            ref_radii=ref_radii_np[index, : int(env.config.sim.angles)],
            Ip_ref=float(ip_ref_np[index]),
        )
        physical = np.concatenate([
            np.asarray(control.pfc_derivs, dtype=float).reshape(-1),
            np.asarray(control.sol_derivs, dtype=float).reshape(-1),
        ])
        actions.append(physical)

    physical_t = torch.as_tensor(np.stack(actions, axis=0), dtype=torch.float32, device=env.device)
    derivative_limits = torch.where(
        torch.isfinite(env.derivative_limits) & (env.derivative_limits > 0.0),
        env.derivative_limits,
        torch.ones_like(env.derivative_limits),
    )
    normalized = torch.nan_to_num(physical_t / derivative_limits[None, :], nan=0.0, posinf=1.0, neginf=-1.0)
    normalized = torch.clamp(normalized, -1.0, 1.0)
    return env._project_action_to_current_safety(normalized).detach()
