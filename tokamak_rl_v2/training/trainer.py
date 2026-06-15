from __future__ import annotations

import csv
import json
import queue
import time
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional local dependency
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total=None, desc=None, unit=None, dynamic_ncols=None):
            self.total = total
        def update(self, n=1):
            return None
        def set_postfix(self, **kwargs):
            return None
        def close(self):
            return None

from tokamak_rl_v2.config.schema import ExperimentConfig
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.export import export_deterministic_actor
from tokamak_rl_v2.networks import CRITIC_ACTION_INPUT_KIND, FeedForwardGaussianActor, RecurrentQCritic
from tokamak_rl_v2.training.distributed import broadcast_actor, start_actor_workers, stop_actor_workers
from tokamak_rl_v2.training.mpo import MaximumAPosterioriPolicyOptimiser
from tokamak_rl_v2.training.replay import FIFOSequenceReplay


class Trainer:
    def __init__(self, config: ExperimentConfig, *, steps: int | None = None, num_envs: int | None = None, device: str | None = None, output_dir: str | Path | None = None, wandb_run=None, resume_checkpoint: str | Path | None = None) -> None:
        self.config = config
        self.steps = int(config.training.steps if steps is None else steps)
        self.num_envs = int(config.training.num_envs if num_envs is None else num_envs)
        self.device = _resolve_device(config.training.device if device is None else device)
        self.output_dir = Path(config.training.output_dir if output_dir is None else output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run
        self.resume_checkpoint = Path(resume_checkpoint) if resume_checkpoint is not None else None
        self.start_step = 0
        self.start_updates = 0
        self._last_actor_devices: tuple[str, ...] = ()
        self._last_envs_per_worker: int | None = None
        self._last_total_training_envs: int | None = None
        torch.manual_seed(int(config.training.seed))
        np.random.seed(int(config.training.seed))
        self.env = TokamakMagneticControlEnv(config, batch_size=self.num_envs, device=self.device, seed=int(config.training.seed))
        self.actor = FeedForwardGaussianActor(self.env.obs_dim, self.env.action_dim, config.network.hidden_dim, min_std=config.network.actor_min_std, initial_std=config.network.actor_initial_std).to(self.device)
        self.critic = RecurrentQCritic(self.env.obs_dim, self.env.action_dim, config.network.critic_hidden_dim, config.network.critic_mlp_hidden_dim).to(self.device)
        self.target_actor = FeedForwardGaussianActor(self.env.obs_dim, self.env.action_dim, config.network.hidden_dim, min_std=config.network.actor_min_std, initial_std=config.network.actor_initial_std).to(self.device)
        self.target_critic = RecurrentQCritic(self.env.obs_dim, self.env.action_dim, config.network.critic_hidden_dim, config.network.critic_mlp_hidden_dim).to(self.device)
        self.learner = MaximumAPosterioriPolicyOptimiser(actor=self.actor, critic=self.critic, target_actor=self.target_actor, target_critic=self.target_critic, config=config.learner, device=self.device)
        self.replay = FIFOSequenceReplay(capacity_episodes=int(config.learner.replay_capacity_episodes), max_episode_steps=int(config.sim.max_episode_steps), active_envs=self.num_envs, obs_dim=self.env.obs_dim, action_dim=self.env.action_dim, device=self.device)
        self.schema = self.env.export_schema()
        self.normalization = self.env.normalization()
        self.best_eval = -float("inf")
        self.best_eval_details: dict[str, float] = {}
        self.best_actor_state_dict: dict[str, torch.Tensor] | None = None
        self._configure_wandb_metrics()

    def _configure_wandb_metrics(self) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.define_metric("global_step")
            self.wandb_run.define_metric("env_step")
            self.wandb_run.define_metric("decision_step")
            for prefix in ("train/*", "reward/*", "eval/*"):
                self.wandb_run.define_metric(prefix, step_metric="global_step")
        except Exception as exc:
            print(f"warning: failed to configure W&B metrics: {exc}", file=sys.stderr)

    def _wandb_log(self, values: dict[str, object], *, step: int) -> None:
        if self.wandb_run is None:
            return
        raw_step = int(step)
        if int(self.config.training.actor_workers) > 1:
            env_step = raw_step
            decision_step = raw_step
        else:
            decision_step = raw_step
            env_step = raw_step * self.num_envs
        payload = {"global_step": int(env_step), "env_step": int(env_step), "decision_step": int(decision_step), **values}
        self.wandb_run.log(payload, step=int(env_step))

    def train(self) -> dict[str, Any]:
        if int(self.config.training.actor_workers) > 1:
            return self._train_distributed()
        self._write_config_snapshot()
        losses_path = self.output_dir / "losses.csv"
        metrics_path = self.output_dir / "metrics.json"
        rewards_path = self.output_dir / "reward_components.csv"
        eval_path = self.output_dir / "eval_history.csv"
        with losses_path.open("w", newline="", encoding="utf-8") as loss_f, rewards_path.open("w", newline="", encoding="utf-8") as reward_f:
            loss_writer = csv.DictWriter(loss_f, fieldnames=["step", "critic_loss", "actor_loss", "mean_kl", "std_kl", "q_mean", "target_q_mean", "actor_mle_loss", "actor_param_delta_norm", "action_mean_abs", "action_std_mean", "sampled_q_spread", "policy_weight_entropy", "policy_weight_max", "mpo_temperature", "mean_kl_penalty", "std_kl_penalty", "env_steps_per_second"])
            reward_writer = None
            loss_writer.writeheader()
            if self.resume_checkpoint is None and self.replay.size > 0:
                self.replay.start_new_episodes()
            obs = self.env.reset()
            if self.resume_checkpoint is not None:
                obs = self._load_checkpoint(self.resume_checkpoint)
            start = time.time()
            updates = int(self.start_updates)
            progress = tqdm(total=max(self.steps - self.start_step, 0), desc="train", unit="step", dynamic_ncols=True)
            for step in range(self.start_step + 1, self.steps + 1):
                with torch.no_grad():
                    action, _logp, _mean = self.actor.sample(obs)
                batch_step = self.env.step(action)
                discount = torch.full((self.num_envs,), float(self.config.learner.discount), dtype=torch.float32, device=self.device)
                done = batch_step.terminated | batch_step.truncated
                self.replay.add_batch(obs, batch_step.applied_action, batch_step.reward, discount, batch_step.obs, done)
                obs = self.env.reset_indices(done) if bool(torch.any(done).item()) else batch_step.obs
                metrics = None
                if self.replay.ready(self.config.learner.unroll_length, self.config.learner.batch_size) and step % int(self.config.learner.rollout_chunk_length) == 0:
                    for _ in range(int(self.config.learner.updates_per_rollout_chunk)):
                        seq = self.replay.sample(batch_size=self.config.learner.batch_size, sequence_length=self.config.learner.unroll_length)
                        metrics = self.learner.update(seq)
                        updates += 1
                    speed = float(step * self.num_envs) / max(time.time() - start, 1.0e-9)
                    row = {"step": step, "env_steps_per_second": speed, **asdict(metrics)}
                    loss_writer.writerow(row); loss_f.flush()
                    self._wandb_log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=step)
                comps = batch_step.info.get("reward_components", {}) if isinstance(batch_step.info, dict) else {}
                if comps:
                    flat = {"step": step}
                    for name, value in comps.items():
                        flat[name] = float(np.nanmean(value))
                    if reward_writer is None:
                        reward_writer = csv.DictWriter(reward_f, fieldnames=list(flat.keys()))
                        reward_writer.writeheader()
                    reward_writer.writerow(flat); reward_f.flush()
                    self._wandb_log({f"reward/{k}": v for k, v in flat.items() if k != "step"}, step=step)
                if self._save_checkpoints_enabled() and step % int(self.config.training.checkpoint_interval_steps) == 0:
                    self._save_checkpoint("latest.pt", step=step, updates=updates)
                if step % int(self.config.training.eval_interval_steps) == 0:
                    eval_metrics = self.evaluate_detailed(max_steps=int(self.config.training.eval_max_steps), episodes=int(self.config.training.eval_episodes))
                    score = self._selection_score(eval_metrics)
                    eval_metrics["selection_score"] = score
                    _append_csv_row(eval_path, {"step": step, "env_step": step * self.num_envs, **eval_metrics})
                    self._wandb_log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)
                    if score > self.best_eval:
                        self.best_eval = score
                        self.best_eval_details = dict(eval_metrics)
                        self._remember_best_actor()
                        if self._save_checkpoints_enabled():
                            self._save_checkpoint("best.pt", step=step, updates=updates)
                        self._export("exports/best_actor", step=step, updates=updates, eval_score=score)
                progress.update(1)
                progress.set_postfix(replay=self.replay.size, updates=updates, reward=f"{float(batch_step.reward.mean().detach().cpu()):.4f}", refresh=False)
            progress.close()
        final = {
            "start_step": self.start_step,
            "steps": self.steps,
            "decision_steps": self.steps,
            "env_steps": self.steps * self.num_envs,
            "updates": updates,
            "best_eval": self.best_eval,
            "best_eval_details": self.best_eval_details,
            "device": str(self.device),
            "output_dir": str(self.output_dir),
            "total_training_envs": self.num_envs,
            "save_checkpoints": self._save_checkpoints_enabled(),
        }
        if self._save_checkpoints_enabled():
            self._save_checkpoint("final.pt", step=self.steps, updates=updates)
        self._export("exports/final_actor", step=self.steps, updates=updates, eval_score=self.best_eval)
        metrics_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        return final


    def _train_distributed(self) -> dict[str, Any]:
        self._write_config_snapshot()
        worker_count = int(self.config.training.actor_workers)
        envs_per_worker = self._distributed_envs_per_worker(worker_count)
        actor_devices = self._resolve_actor_devices(worker_count)
        self._last_actor_devices = actor_devices
        self._last_envs_per_worker = envs_per_worker
        self._last_total_training_envs = envs_per_worker * worker_count
        if self.resume_checkpoint is not None:
            raise ValueError("distributed actor-worker training checkpoints are not exactly resumable yet; use actor_workers=1 for resumable training or start a fresh distributed run")
        processes, param_queues, data_q, stop = start_actor_workers(
            config=self.config,
            actor_state_dict=self.actor.state_dict(),
            worker_count=worker_count,
            envs_per_worker=envs_per_worker,
            rollout_chunk_length=int(self.config.learner.rollout_chunk_length),
            actor_devices=actor_devices,
            seed=int(self.config.training.seed),
        )
        losses_path = self.output_dir / "losses.csv"
        metrics_path = self.output_dir / "metrics.json"
        rewards_path = self.output_dir / "reward_components.csv"
        eval_path = self.output_dir / "eval_history.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        updates = int(self.start_updates)
        env_steps = int(self.start_step)
        last_checkpoint_step = 0
        last_eval_step = 0
        worker_rollout_counts = {str(index): 0 for index in range(worker_count)}
        start = time.time()
        try:
            with losses_path.open("w", newline="", encoding="utf-8") as loss_f, rewards_path.open("w", newline="", encoding="utf-8") as reward_f:
                loss_writer = csv.DictWriter(loss_f, fieldnames=["step", "critic_loss", "actor_loss", "mean_kl", "std_kl", "q_mean", "target_q_mean", "actor_mle_loss", "actor_param_delta_norm", "action_mean_abs", "action_std_mean", "sampled_q_spread", "policy_weight_entropy", "policy_weight_max", "mpo_temperature", "mean_kl_penalty", "std_kl_penalty", "env_steps_per_second"])
                loss_writer.writeheader()
                reward_writer = None
                progress = tqdm(total=max(self.steps - self.start_step, 0), desc="distributed-train", unit="step", dynamic_ncols=True)
                while env_steps < self.steps:
                    self._raise_for_dead_actor_workers(processes, actor_devices)
                    try:
                        payload = data_q.get(timeout=300.0)
                    except queue.Empty as exc:
                        self._raise_for_dead_actor_workers(processes, actor_devices)
                        raise RuntimeError("timed out waiting for actor worker rollout data") from exc
                    T, B = int(payload["reward"].shape[0]), int(payload["reward"].shape[1])
                    worker_index = str(int(payload.get("worker_index", -1)))
                    if worker_index in worker_rollout_counts:
                        worker_rollout_counts[worker_index] += T * B
                    worker_lane_offset = int(payload.get("worker_index", 0)) * envs_per_worker
                    worker_lanes = list(range(worker_lane_offset, worker_lane_offset + B))
                    for t in range(T):
                        self.replay.add_batch(
                            torch.as_tensor(payload["obs"][t], dtype=torch.float32, device=self.device),
                            torch.as_tensor(payload["action"][t], dtype=torch.float32, device=self.device),
                            torch.as_tensor(payload["reward"][t], dtype=torch.float32, device=self.device),
                            torch.as_tensor(payload["discount"][t], dtype=torch.float32, device=self.device),
                            torch.as_tensor(payload["next_obs"][t], dtype=torch.float32, device=self.device),
                            torch.as_tensor(payload["done"][t], dtype=torch.bool, device=self.device),
                            lane_indices=worker_lanes,
                        )
                    env_steps += T * B
                    reward_components = payload.get("reward_components", {})
                    if isinstance(reward_components, dict) and reward_components:
                        flat = {"step": env_steps}
                        for name, value in reward_components.items():
                            flat[str(name)] = float(np.nanmean(np.asarray(value, dtype=float)))
                        if reward_writer is None:
                            reward_writer = csv.DictWriter(reward_f, fieldnames=list(flat.keys()))
                            reward_writer.writeheader()
                        reward_writer.writerow(flat); reward_f.flush()
                        self._wandb_log({f"reward/{k}": v for k, v in flat.items() if k != "step"}, step=env_steps)
                    while self.replay.ready(self.config.learner.unroll_length, self.config.learner.batch_size):
                        for _ in range(int(self.config.learner.updates_per_rollout_chunk)):
                            seq = self.replay.sample(batch_size=self.config.learner.batch_size, sequence_length=self.config.learner.unroll_length)
                            metrics = self.learner.update(seq)
                            updates += 1
                        speed = float(env_steps) / max(time.time() - start, 1.0e-9)
                        row = {"step": env_steps, "env_steps_per_second": speed, **asdict(metrics)}
                        loss_writer.writerow(row); loss_f.flush()
                        self._wandb_log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=env_steps)
                        if updates % 4 == 0:
                            broadcast_actor(param_queues, self.actor.state_dict())
                        break
                    if self._save_checkpoints_enabled() and env_steps - last_checkpoint_step >= max(int(self.config.training.checkpoint_interval_steps), 1):
                        self._save_checkpoint("latest.pt", step=env_steps, updates=updates)
                        last_checkpoint_step = env_steps
                    if env_steps - last_eval_step >= max(int(self.config.training.eval_interval_steps), 1):
                        eval_metrics = self.evaluate_detailed(max_steps=int(self.config.training.eval_max_steps), episodes=int(self.config.training.eval_episodes))
                        score = self._selection_score(eval_metrics)
                        eval_metrics["selection_score"] = score
                        _append_csv_row(eval_path, {"step": env_steps, "env_step": env_steps, **eval_metrics})
                        self._wandb_log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=env_steps)
                        if score > self.best_eval:
                            self.best_eval = score
                            self.best_eval_details = dict(eval_metrics)
                            self._remember_best_actor()
                            if self._save_checkpoints_enabled():
                                self._save_checkpoint("best.pt", step=env_steps, updates=updates)
                            self._export("exports/best_actor", step=env_steps, updates=updates, eval_score=score)
                        last_eval_step = env_steps
                    progress.update(min(T * B, max(self.steps - (env_steps - T * B), 0)))
                    progress.set_postfix(replay=self.replay.size, updates=updates, refresh=False)
                progress.close()
        finally:
            stop_actor_workers(processes, stop, param_queues=param_queues, data_q=data_q)
        if self._save_checkpoints_enabled():
            self._save_checkpoint("final.pt", step=env_steps, updates=updates)
        self._export("exports/final_actor", step=env_steps, updates=updates, eval_score=None)
        final = {
            "start_step": self.start_step,
            "steps": env_steps,
            "env_steps": env_steps,
            "updates": updates,
            "best_eval": self.best_eval,
            "best_eval_details": self.best_eval_details,
            "device": str(self.device),
            "learner_device": str(self.device),
            "output_dir": str(self.output_dir),
            "actor_workers": worker_count,
            "actor_devices": list(actor_devices),
            "envs_per_worker": envs_per_worker,
            "total_training_envs": envs_per_worker * worker_count,
            "worker_rollout_counts": worker_rollout_counts,
            "save_checkpoints": self._save_checkpoints_enabled(),
        }
        metrics_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        return final

    def _distributed_envs_per_worker(self, worker_count: int) -> int:
        if worker_count < 2:
            raise ValueError("distributed training requires actor_workers >= 2")
        if self.num_envs < worker_count:
            raise ValueError("num_envs must be at least actor_workers for distributed training")
        if self.num_envs % worker_count != 0:
            raise ValueError("num_envs must be divisible by actor_workers so every actor GPU receives the same batch size")
        return self.num_envs // worker_count

    def _resolve_actor_devices(self, worker_count: int) -> tuple[str, ...]:
        configured = tuple(str(device).strip() for device in self.config.training.actor_devices if str(device).strip())
        if configured:
            if len(configured) < worker_count:
                raise ValueError(f"actor_workers={worker_count} requires at least {worker_count} actor_devices, got {len(configured)}")
            devices = configured[:worker_count]
        elif self.config.sim.compute_backend == "gpu":
            raise ValueError("GPU distributed training requires --actor-devices, for example --actor-devices cuda:1,cuda:2,cuda:3")
        else:
            devices = tuple(str(self.device) for _ in range(worker_count))
        for raw in devices:
            dev = torch.device(raw)
            if self.config.sim.compute_backend == "gpu" and dev.type != "cuda":
                raise ValueError(f"GPU simulator actor workers require CUDA devices, got {raw}")
            if dev.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(f"CUDA actor device requested but torch.cuda.is_available() is false: {raw}")
                if dev.index is not None and dev.index >= torch.cuda.device_count():
                    raise RuntimeError(f"CUDA actor device index is not visible: {raw}; visible device count is {torch.cuda.device_count()}")
        return devices

    @staticmethod
    def _raise_for_dead_actor_workers(processes, actor_devices: tuple[str, ...]) -> None:
        dead = []
        for index, proc in enumerate(processes):
            exitcode = proc.exitcode
            if exitcode is not None and exitcode != 0:
                device = actor_devices[index] if index < len(actor_devices) else "unknown"
                dead.append(f"worker={index} device={device} exitcode={exitcode}")
        if dead:
            raise RuntimeError("actor worker failed: " + "; ".join(dead))

    @torch.no_grad()
    def evaluate(self, *, episodes: int, max_steps: int) -> float:
        return float(self.evaluate_detailed(episodes=episodes, max_steps=max_steps)["mean_return"])

    @torch.no_grad()
    def evaluate_detailed(self, *, episodes: int, max_steps: int, policy: Literal["actor", "no_control"] = "actor", seed_offset: int = 100000) -> dict[str, float]:
        if policy not in {"actor", "no_control"}:
            raise ValueError(f"unsupported evaluation policy: {policy}")
        batch_size = max(1, min(int(episodes), self.num_envs))
        env = TokamakMagneticControlEnv(self.config, batch_size=batch_size, device=self.device, seed=int(self.config.training.seed) + int(seed_offset))
        obs = env.reset()
        returns: list[float] = []
        episode_steps: list[int] = []
        totals = torch.zeros((env.batch_size,), dtype=torch.float32, device=self.device)
        steps = torch.zeros((env.batch_size,), dtype=torch.long, device=self.device)
        component_values: dict[str, list[float]] = {}
        early_component_values: dict[str, list[float]] = {}
        late_component_values: dict[str, list[float]] = {}
        while len(returns) < int(episodes):
            if policy == "actor":
                action = self.actor.deterministic(obs)
            else:
                action = torch.zeros((env.batch_size, env.action_dim), dtype=torch.float32, device=self.device)
            out = env.step(action)
            totals += out.reward
            steps += 1
            comps = out.info.get("reward_components", {}) if isinstance(out.info, dict) else {}
            if isinstance(comps, dict):
                max_steps_f = max(float(max_steps), 1.0)
                progress = (steps.to(torch.float32) / max_steps_f).detach().cpu().numpy().reshape(-1)
                early_cutoff = max(0.2, 1.0 / max_steps_f)
                for name, value in comps.items():
                    arr = np.asarray(value, dtype=float).reshape(-1)
                    if arr.size:
                        finite = np.isfinite(arr)
                        component_values.setdefault(str(name), []).extend(arr[finite].astype(float).tolist())
                        if progress.shape == arr.shape:
                            early = finite & (progress <= early_cutoff)
                            late = finite & (progress >= 0.8)
                            if np.any(early):
                                early_component_values.setdefault(str(name), []).extend(arr[early].astype(float).tolist())
                            if np.any(late):
                                late_component_values.setdefault(str(name), []).extend(arr[late].astype(float).tolist())
            done = out.terminated | out.truncated | (steps >= int(max_steps))
            if bool(torch.any(done).item()):
                done_cpu = done.detach().cpu().numpy().astype(bool)
                totals_cpu = totals.detach().cpu().numpy().astype(float)
                steps_cpu = steps.detach().cpu().numpy().astype(int)
                for index, is_done in enumerate(done_cpu):
                    if is_done and len(returns) < int(episodes):
                        returns.append(float(totals_cpu[index]))
                        episode_steps.append(int(steps_cpu[index]))
                totals = torch.where(done, torch.zeros_like(totals), totals)
                steps = torch.where(done, torch.zeros_like(steps), steps)
                obs = env.reset_indices(done) if len(returns) < int(episodes) else out.obs
            else:
                obs = out.obs
        selected_returns = np.asarray(returns[: int(episodes)], dtype=float)
        selected_steps = np.asarray(episode_steps[: int(episodes)], dtype=float)
        max_steps_f = max(float(max_steps), 1.0)
        metrics: dict[str, float] = {"mean_return": float(np.nanmean(selected_returns)) if selected_returns.size else float("nan")}
        if selected_steps.size:
            completion = selected_steps / max_steps_f
            metrics["mean_episode_steps"] = float(np.nanmean(selected_steps))
            metrics["min_episode_steps"] = float(np.nanmin(selected_steps))
            metrics["mean_episode_completion"] = float(np.nanmean(completion))
            metrics["min_episode_completion"] = float(np.nanmin(completion))
        max_metrics = {
            "shape_error_mean_m",
            "shape_error_max_m",
            "ip_error_a",
            "current_over_limit_a",
            "current_usage_fraction",
            "derivative_usage",
            "max_abs_action",
            "action_rms",
            "delta_action_rms",
            "base_physical_cost",
            "physical_cost",
            "shape_loss",
            "ip_loss",
            "terminated_boundary",
            "terminated_current",
        }
        min_metrics = {"current_margin_fraction", "boundary_found"}
        for name, values in component_values.items():
            arr = np.asarray(values, dtype=float)
            if arr.size:
                metrics[name] = float(np.nanmean(arr))
                if name in max_metrics:
                    metrics[f"{name}_max"] = float(np.nanmax(arr))
                if name in min_metrics:
                    metrics[f"{name}_min"] = float(np.nanmin(arr))
                if name == "current_over_limit_a":
                    metrics["current_over_limit_fraction"] = float(np.nanmean(arr > 0.0))
        profile_metrics = max_metrics | min_metrics | {"episode_progress"}
        drift_metrics = {"shape_error_mean_m", "shape_error_max_m", "ip_error_a", "physical_cost", "base_physical_cost", "current_usage_fraction"}
        for name in profile_metrics:
            early = np.asarray(early_component_values.get(name, []), dtype=float)
            late = np.asarray(late_component_values.get(name, []), dtype=float)
            if early.size:
                metrics[f"{name}_early"] = float(np.nanmean(early))
            if late.size:
                metrics[f"{name}_late"] = float(np.nanmean(late))
                if name in max_metrics:
                    metrics[f"{name}_late_max"] = float(np.nanmax(late))
                if name in min_metrics:
                    metrics[f"{name}_late_min"] = float(np.nanmin(late))
                if name == "current_over_limit_a":
                    metrics["current_over_limit_fraction_late"] = float(np.nanmean(late > 0.0))
            if early.size and late.size and name in drift_metrics:
                metrics[f"{name}_late_minus_early"] = float(np.nanmean(late) - np.nanmean(early))
        return metrics

    @staticmethod
    def _selection_score(metrics: dict[str, float]) -> float:
        score = -float(metrics.get("physical_cost_late", metrics.get("physical_cost", float("inf"))))
        current_over = float(metrics.get("current_over_limit_a_late_max", metrics.get("current_over_limit_a_max", metrics.get("current_over_limit_a", 0.0))))
        boundary_found = float(metrics.get("boundary_found_late_min", metrics.get("boundary_found_min", metrics.get("boundary_found", 1.0))))
        episode_completion = float(metrics.get("min_episode_completion", metrics.get("mean_episode_completion", 1.0)))
        shape_drift = float(metrics.get("shape_error_mean_m_late_minus_early", 0.0))
        ip_drift = float(metrics.get("ip_error_a_late_minus_early", 0.0))
        if np.isfinite(current_over) and current_over > 0.0:
            score -= 1.0e6 + min(current_over, 1.0e6)
        if np.isfinite(boundary_found) and boundary_found < 0.999:
            score -= 1.0e6 * (0.999 - boundary_found)
        if np.isfinite(episode_completion) and episode_completion < 0.95:
            score -= 1.0e6 * (0.95 - episode_completion)
        if np.isfinite(shape_drift) and shape_drift > 0.0:
            score -= 1000.0 * shape_drift
        if np.isfinite(ip_drift) and ip_drift > 0.0:
            score -= ip_drift / 1000.0
        return float(score)

    def _metadata(self, *, step: int, updates: int, eval_score: float | None = None) -> dict[str, object]:
        exact_resume_supported = not bool(self._last_actor_devices)
        raw_step = int(step)
        if self._last_actor_devices:
            env_step = raw_step
            decision_step = raw_step
        else:
            decision_step = raw_step
            env_step = raw_step * self.num_envs
        out: dict[str, object] = {"experiment": self.config.name, "step": raw_step, "decision_step": decision_step, "env_step": env_step, "updates": int(updates), "eval_score": eval_score, "device": str(self.device), "learner_device": str(self.device), "algorithm": "Maximum a Posteriori Policy Optimisation", "plant": "tokamak-sim", "sim_compute_backend": self.config.sim.compute_backend, "exact_resume_supported": exact_resume_supported}
        if self._last_actor_devices:
            out["actor_devices"] = list(self._last_actor_devices)
            out["actor_workers"] = len(self._last_actor_devices)
            out["envs_per_worker"] = self._last_envs_per_worker
            out["total_training_envs"] = self._last_total_training_envs
            out["resume_limitation"] = "actor-worker environment states are not checkpointed"
        return out

    def restore_best_actor(self) -> bool:
        if self.best_actor_state_dict is None:
            return False
        self.actor.load_state_dict({name: value.to(self.device) for name, value in self.best_actor_state_dict.items()})
        return True

    def _remember_best_actor(self) -> None:
        self.best_actor_state_dict = {name: value.detach().cpu().clone() for name, value in self.actor.state_dict().items()}

    def _save_checkpoints_enabled(self) -> bool:
        return bool(self.config.training.save_checkpoints)

    def _save_checkpoint(self, name: str, *, step: int, updates: int) -> Path:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / name
        rng_state: dict[str, object] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        try:
            env_state = self.env.state_dict()
        except RuntimeError:
            if int(self.config.training.actor_workers) > 1:
                env_state = None
            else:
                raise
        torch.save({
            "checkpoint_version": 2,
            "critic_action_input_kind": CRITIC_ACTION_INPUT_KIND,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "learner_state": self.learner.state_dict(),
            "replay_state": self.replay.state_dict(),
            "env_state": env_state,
            "rng_state": rng_state,
            "best_eval": self.best_eval,
            "best_eval_details": self.best_eval_details,
            "schema": self.schema,
            "normalization": self.normalization,
            "metadata": self._metadata(step=step, updates=updates),
            "network": asdict(self.config.network),
            "learner": asdict(self.config.learner),
            "reward": asdict(self.config.reward),
            "sim": self._sim_resume_fragment(),
            "training_state": {"step": int(step), "updates": int(updates)},
        }, path)
        return path

    def _load_checkpoint(self, path: str | Path, *, restore_env: bool = True) -> torch.Tensor:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
        data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if int(data.get("checkpoint_version", 0)) < 2:
            raise ValueError(f"checkpoint is not resumable by this trainer: {checkpoint_path}")
        self._validate_checkpoint(data, checkpoint_path)
        self.actor.load_state_dict(data["actor_state_dict"])
        self.critic.load_state_dict(data["critic_state_dict"])
        self.target_actor.load_state_dict(data["target_actor_state_dict"])
        self.target_critic.load_state_dict(data["target_critic_state_dict"])
        self.learner.load_state_dict(data["learner_state"])
        self.replay.load_state_dict(data["replay_state"])
        self.best_eval = float(data.get("best_eval", -float("inf")))
        raw_details = data.get("best_eval_details", {})
        self.best_eval_details = dict(raw_details) if isinstance(raw_details, dict) else {}
        training_state = data.get("training_state", {})
        self.start_step = int(training_state.get("step", data.get("metadata", {}).get("step", 0)))
        self.start_updates = int(training_state.get("updates", data.get("metadata", {}).get("updates", 0)))
        if self.steps <= self.start_step:
            raise ValueError(f"resume target steps must be greater than checkpoint step: target={self.steps}, checkpoint={self.start_step}")
        rng = data.get("rng_state", {})
        if isinstance(rng, dict):
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].detach().cpu())
            if self.device.type == "cuda":
                if "torch_cuda" not in rng:
                    raise ValueError(f"checkpoint does not contain CUDA RNG state and cannot be exactly resumed on {self.device}: {checkpoint_path}")
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            elif "torch_cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "python" in rng:
                random.setstate(rng["python"])
        if restore_env:
            if data.get("env_state") is None:
                raise ValueError(f"checkpoint does not contain environment state and cannot be exactly resumed: {checkpoint_path}")
            return self.env.load_state_dict(data["env_state"])
        return self.env.reset()

    def _validate_checkpoint(self, data: dict[str, object], path: Path) -> None:
        if data.get("critic_action_input_kind") != CRITIC_ACTION_INPUT_KIND:
            raise ValueError(f"checkpoint critic action input convention mismatch: {path}")
        if data.get("schema", {}).get("observation_kind") != self.schema.get("observation_kind"):
            raise ValueError(f"checkpoint observation schema mismatch: {path}")
        if int(data.get("schema", {}).get("obs_dim", -1)) != int(self.schema.get("obs_dim", -2)):
            raise ValueError(f"checkpoint observation dimension mismatch: {path}")
        if int(data.get("schema", {}).get("action_dim", -1)) != int(self.schema.get("action_dim", -2)):
            raise ValueError(f"checkpoint action dimension mismatch: {path}")
        expected = {
            "network": asdict(self.config.network),
            "learner": asdict(self.config.learner),
            "reward": asdict(self.config.reward),
            "sim": self._sim_resume_fragment(),
        }
        for name, value in expected.items():
            if data.get(name) != value:
                raise ValueError(f"checkpoint {name} config mismatch: {path}")

    def _sim_resume_fragment(self) -> object:
        fragment = self._config_fragment(self.config.sim)
        if isinstance(fragment, dict) and "gpu_device" in fragment:
            fragment = dict(fragment)
            fragment["gpu_device"] = "<runtime>"
        return fragment

    @staticmethod
    def _config_fragment(obj) -> object:
        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if is_dataclass(value):
                return {k: convert(v) for k, v in asdict(value).items()}
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value
        return convert(obj)

    def _export(self, relative: str, *, step: int, updates: int, eval_score: float | None) -> Path:
        return export_deterministic_actor(actor=self.actor, export_dir=self.output_dir / relative, schema=self.schema, normalization=self.normalization, metadata=self._metadata(step=step, updates=updates, eval_score=eval_score))

    def _write_config_snapshot(self) -> None:
        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if is_dataclass(obj):
                return {k: convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj
        (self.output_dir / "config_snapshot.json").write_text(json.dumps(convert(self.config), indent=2), encoding="utf-8")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(value)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if dev.type == "cuda" and dev.index is not None and dev.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index is not visible: {value}; visible device count is {torch.cuda.device_count()}")
    return dev


def _append_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        extra_fields = [key for key in row if key not in fields]
        if extra_fields:
            fields.extend(extra_fields)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for old_row in rows:
                    writer.writerow(old_row)
    else:
        fields = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
