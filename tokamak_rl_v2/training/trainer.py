from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import time
import random
import sys
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
import torch.distributed as dist
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


def _value_to_numpy(value: object) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _component_mean(value: object) -> float:
    if torch.is_tensor(value):
        tensor = value.detach().to(dtype=torch.float32)
        finite = torch.isfinite(tensor)
        if not bool(torch.any(finite).item()):
            return float("nan")
        return float(torch.mean(tensor[finite]).detach().cpu().item())
    arr = np.asarray(value, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _distributed_mean_scalars(values: dict[str, float], *, device: torch.device, enabled: bool) -> dict[str, float]:
    if not enabled or not (dist.is_available() and dist.is_initialized()) or int(dist.get_world_size()) <= 1:
        return values
    out: dict[str, float] = {}
    for key in sorted(values):
        value = float(values[key])
        finite = math_isfinite(value)
        pair = torch.tensor([value if finite else 0.0, 1.0 if finite else 0.0], dtype=torch.float64, device=device)
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
        count = float(pair[1].detach().cpu().item())
        out[key] = float(pair[0].detach().cpu().item() / count) if count > 0.0 else float("nan")
    return out


class _RewardComponentAccumulator:
    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}

    def add(self, components: Mapping[str, object]) -> None:
        for raw_name, raw_value in components.items():
            name = str(raw_name)
            if torch.is_tensor(raw_value):
                tensor = raw_value.detach().to(device=self.device, dtype=torch.float32)
            else:
                tensor = torch.as_tensor(raw_value, dtype=torch.float32, device=self.device)
            finite = torch.isfinite(tensor)
            value_sum = torch.where(finite, tensor, torch.zeros_like(tensor)).sum()
            count = finite.to(dtype=torch.float32).sum()
            if name not in self.sums:
                self.sums[name] = torch.zeros((), dtype=torch.float32, device=self.device)
                self.counts[name] = torch.zeros((), dtype=torch.float32, device=self.device)
            self.sums[name] = self.sums[name] + value_sum
            self.counts[name] = self.counts[name] + count

    def means(self, *, distributed: bool = False, reset: bool = True) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in sorted(self.sums):
            pair = torch.stack([self.sums[name], self.counts[name]]).to(dtype=torch.float64)
            if distributed and dist.is_available() and dist.is_initialized() and int(dist.get_world_size()) > 1:
                dist.all_reduce(pair, op=dist.ReduceOp.SUM)
            count = float(pair[1].detach().cpu().item())
            out[name] = float(pair[0].detach().cpu().item() / count) if count > 0.0 else float("nan")
        if reset:
            self.clear()
        return out

    def clear(self) -> None:
        self.sums.clear()
        self.counts.clear()


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def _eval_max_steps_for_config(config: ExperimentConfig) -> int:
    if bool(config.training.production_mode):
        return int(config.sim.max_episode_steps)
    return int(config.training.eval_max_steps)


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        steps: int | None = None,
        num_envs: int | None = None,
        device: str | None = None,
        output_dir: str | Path | None = None,
        wandb_run=None,
        resume_checkpoint: str | Path | None = None,
        export_policy: bool = True,
    ) -> None:
        self.distributed_mode = str(config.training.distributed_mode)
        self.distributed_rank = 0
        self.distributed_world_size = 1
        self.distributed_local_rank = 0
        requested_num_envs = int(config.training.num_envs if num_envs is None else num_envs)
        requested_device = config.training.device if device is None else device
        self.global_num_envs = requested_num_envs
        if self.distributed_mode == "local_replay":
            config, requested_device, requested_num_envs = self._configure_local_replay_runtime(config, requested_num_envs=requested_num_envs, requested_device=str(requested_device))
        self.config = config
        self.steps = int(config.training.steps if steps is None else steps)
        self.num_envs = int(requested_num_envs)
        self.device = _resolve_device(str(requested_device))
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.output_dir = Path(config.training.output_dir if output_dir is None else output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run
        self.resume_checkpoint = Path(resume_checkpoint) if resume_checkpoint is not None else None
        self.export_policy = bool(export_policy)
        self.start_step = 0
        self.start_updates = 0
        self._last_actor_devices: tuple[str, ...] = ()
        self._last_envs_per_worker: int | None = None
        self._last_total_training_envs: int | None = None
        base_seed = int(config.training.seed)
        rank_seed = base_seed + int(self.distributed_rank) * 1000003
        torch.manual_seed(base_seed)
        np.random.seed(rank_seed)
        self.env = TokamakMagneticControlEnv(config, batch_size=self.num_envs, device=self.device, seed=rank_seed)
        torch.manual_seed(base_seed)
        self.actor = FeedForwardGaussianActor(self.env.obs_dim, self.env.action_dim, config.network.hidden_dim, min_std=config.network.actor_min_std, initial_std=config.network.actor_initial_std).to(self.device)
        self.critic = RecurrentQCritic(self.env.critic_obs_dim, self.env.action_dim, config.network.critic_hidden_dim, config.network.critic_mlp_hidden_dim).to(self.device)
        self.target_actor = FeedForwardGaussianActor(self.env.obs_dim, self.env.action_dim, config.network.hidden_dim, min_std=config.network.actor_min_std, initial_std=config.network.actor_initial_std).to(self.device)
        self.target_critic = RecurrentQCritic(self.env.critic_obs_dim, self.env.action_dim, config.network.critic_hidden_dim, config.network.critic_mlp_hidden_dim).to(self.device)
        self.learner = MaximumAPosterioriPolicyOptimiser(actor=self.actor, critic=self.critic, target_actor=self.target_actor, target_critic=self.target_critic, config=config.learner, device=self.device, distributed=self.distributed_mode == "local_replay")
        if self.distributed_mode == "local_replay":
            self._broadcast_trainable_state()
            torch.manual_seed(rank_seed + 17)
        self.replay = FIFOSequenceReplay(capacity_episodes=int(config.learner.replay_capacity_episodes), max_episode_steps=int(config.sim.max_episode_steps), active_envs=self.num_envs, obs_dim=self.env.obs_dim, critic_obs_dim=self.env.critic_obs_dim, action_dim=self.env.action_dim, device=self.device)
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

    def _failure_result(self, status: str, *, env_steps: int, updates: int, details: Mapping[str, object] | None = None) -> dict[str, Any]:
        final = {
            "status": str(status),
            "start_step": self.start_step,
            "steps": int(env_steps),
            "env_steps": int(env_steps),
            "updates": int(updates),
            "best_eval": self.best_eval,
            "best_eval_details": self.best_eval_details,
            "device": str(self.device),
            "output_dir": str(self.output_dir),
            "distributed_mode": self.distributed_mode,
            "rank": int(self.distributed_rank),
            "world_size": int(self.distributed_world_size),
            "local_envs_per_rank": int(self.num_envs),
            "total_training_envs": int(self.global_num_envs),
            "failure_details": dict(details or {}),
        }
        if self._rank0():
            (self.output_dir / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        return final

    def _configure_local_replay_runtime(self, config: ExperimentConfig, *, requested_num_envs: int, requested_device: str) -> tuple[ExperimentConfig, str, int]:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        if world_size > 1 and not dist.is_initialized():
            wants_cuda = requested_device != "cpu" and torch.cuda.is_available()
            backend = "nccl" if wants_cuda else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
        self.distributed_rank = rank
        self.distributed_world_size = world_size
        self.distributed_local_rank = local_rank
        if requested_num_envs % world_size != 0:
            raise ValueError(f"local_replay requires global num_envs divisible by WORLD_SIZE: num_envs={requested_num_envs}, WORLD_SIZE={world_size}")
        if requested_device == "cpu":
            resolved_device = "cpu"
        elif torch.cuda.is_available():
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(f"LOCAL_RANK={local_rank} is not visible; CUDA device count is {torch.cuda.device_count()}")
            resolved_device = f"cuda:{local_rank}"
            torch.cuda.set_device(torch.device(resolved_device))
        else:
            resolved_device = "cpu"
        if config.sim.compute_backend == "gpu" and not resolved_device.startswith("cuda"):
            raise RuntimeError("local_replay with sim.compute_backend=gpu requires CUDA")
        local_envs = requested_num_envs // world_size
        if int(config.training.actor_workers) != 1:
            raise ValueError("local_replay does not use actor_workers; set actor_workers=1")
        cfg = replace(
            config,
            sim=replace(config.sim, gpu_device=resolved_device) if config.sim.compute_backend == "gpu" else config.sim,
            training=replace(config.training, num_envs=requested_num_envs, device=resolved_device, distributed_mode="local_replay"),
        )
        return cfg, resolved_device, local_envs

    def _rank0(self) -> bool:
        return int(self.distributed_rank) == 0

    def _distributed_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _broadcast_trainable_state(self) -> None:
        if not self._distributed_initialized() or int(dist.get_world_size()) <= 1:
            return
        for module in (self.actor, self.critic, self.target_actor, self.target_critic):
            for tensor in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(tensor.data, src=0)
        dist.broadcast(self.learner.log_mean_kl_penalty.data, src=0)
        dist.broadcast(self.learner.log_std_kl_penalty.data, src=0)
        dist.broadcast(self.learner.last_temperature.data, src=0)

    def _distributed_all_bool(self, value: bool) -> bool:
        if not self._distributed_initialized() or int(dist.get_world_size()) <= 1:
            return bool(value)
        flag = torch.tensor([1 if value else 0], dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(int(flag.detach().cpu().item()) == 1)

    def _barrier(self) -> None:
        if self._distributed_initialized() and int(dist.get_world_size()) > 1:
            dist.barrier()

    def _wandb_log(self, values: dict[str, object], *, step: int) -> None:
        if self.wandb_run is None:
            return
        raw_step = int(step)
        if self.distributed_mode == "local_replay" or int(self.config.training.actor_workers) > 1:
            env_step = raw_step
            decision_step = raw_step
        else:
            decision_step = raw_step
            env_step = raw_step * self.num_envs
        payload = {"global_step": int(env_step), "env_step": int(env_step), "decision_step": int(decision_step), **values}
        try:
            self.wandb_run.log(payload, step=int(env_step))
        except Exception as exc:
            print(f"warning: W&B log failed; continuing with disk outputs: {exc}", file=sys.stderr)

    def _min_replay_sequence_length(self) -> int:
        return int(getattr(self.config.learner, "min_replay_sequence_length", self.config.learner.unroll_length))

    def _replay_ready(self) -> bool:
        return self.replay.ready(
            self.config.learner.unroll_length,
            self.config.learner.batch_size,
            min_sequence_length=self._min_replay_sequence_length(),
        )

    def _sample_replay(self):
        return self.replay.sample(
            batch_size=self.config.learner.batch_size,
            sequence_length=self.config.learner.unroll_length,
            min_sequence_length=self._min_replay_sequence_length(),
        )

    @staticmethod
    def _replay_health_fields() -> list[str]:
        return [
            "updates",
            "replay_ready",
            "replay_size",
            "replay_completed_episodes",
            "replay_full_sequence_eligible_episodes",
            "replay_min_sequence_eligible_episodes",
            "replay_mean_episode_length",
            "replay_min_episode_length",
            "replay_max_episode_length",
            "learner_no_update_steps",
            "learner_no_update_warning",
        ]

    def _replay_health(self, *, step: int, updates: int, last_update_step: int) -> dict[str, float]:
        stats = self.replay.stats(
            sequence_length=int(self.config.learner.unroll_length),
            min_sequence_length=self._min_replay_sequence_length(),
        )
        no_update_steps = max(0, int(step) - int(last_update_step))
        warning = no_update_steps > int(self.config.learner.rollout_chunk_length)
        return {
            "updates": float(updates),
            "replay_ready": 1.0 if self._replay_ready() else 0.0,
            **stats,
            "learner_no_update_steps": float(no_update_steps),
            "learner_no_update_warning": 1.0 if warning else 0.0,
        }

    def _min_replay_health_check_chunks(self) -> int:
        rollout = max(int(self.config.learner.rollout_chunk_length), 1)
        episode_steps = max(int(self.config.sim.max_episode_steps), 1)
        # Full-episode tasks cannot produce eligible closed episodes before at
        # least one lane reaches the configured horizon. Give the replay one
        # extra rollout chunk after that before treating missing sequences as a
        # real learner failure.
        return max(2, (episode_steps + rollout - 1) // rollout + 1)

    def train(self) -> dict[str, Any]:
        if self.distributed_mode == "local_replay":
            return self._train_local_replay_distributed()
        if int(self.config.training.actor_workers) > 1:
            return self._train_distributed()
        self._write_config_snapshot()
        losses_path = self.output_dir / "losses.csv"
        metrics_path = self.output_dir / "metrics.json"
        rewards_path = self.output_dir / "reward_components.csv"
        eval_path = self.output_dir / "eval_history.csv"
        health_path = self.output_dir / "replay_health.csv"
        replay_health_fields = self._replay_health_fields()
        with losses_path.open("w", newline="", encoding="utf-8") as loss_f, rewards_path.open("w", newline="", encoding="utf-8") as reward_f, health_path.open("w", newline="", encoding="utf-8") as health_f:
            loss_writer = csv.DictWriter(loss_f, fieldnames=["step", "critic_loss", "actor_loss", "mean_kl", "std_kl", "q_mean", "target_q_mean", "actor_mle_loss", "actor_param_delta_norm", "action_mean_abs", "action_std_mean", "sampled_q_spread", "policy_weight_entropy", "policy_weight_max", "mpo_temperature", "mean_kl_penalty", "std_kl_penalty", "env_steps_per_second", *replay_health_fields])
            health_writer = csv.DictWriter(health_f, fieldnames=["step", *replay_health_fields])
            reward_writer = None
            loss_writer.writeheader()
            health_writer.writeheader()
            if self.resume_checkpoint is None and self.replay.size > 0:
                self.replay.start_new_episodes()
            obs = self.env.reset()
            critic_obs = self.env.critic_obs()
            if self.resume_checkpoint is not None:
                obs = self._load_checkpoint(self.resume_checkpoint)
                critic_obs = self.env.critic_obs()
            start = time.time()
            updates = int(self.start_updates)
            last_update_step = int(self.start_step)
            progress = tqdm(total=max(self.steps - self.start_step, 0), desc="train", unit="step", dynamic_ncols=True)
            for step in range(self.start_step + 1, self.steps + 1):
                with torch.no_grad():
                    action, _logp, _mean = self.actor.sample(obs)
                batch_step = self.env.step(action)
                discount = torch.full((self.num_envs,), float(self.config.learner.discount), dtype=torch.float32, device=self.device)
                done = batch_step.terminated | batch_step.truncated
                self.replay.add_batch(obs, batch_step.applied_action, batch_step.reward, discount, batch_step.obs, done, critic_obs=critic_obs, next_critic_obs=batch_step.critic_obs)
                obs = self.env.reset_indices(done) if bool(torch.any(done).item()) else batch_step.obs
                critic_obs = self.env.critic_obs() if bool(torch.any(done).item()) else batch_step.critic_obs
                metrics = None
                if step % int(self.config.learner.rollout_chunk_length) == 0:
                    replay_health = self._replay_health(step=step, updates=updates, last_update_step=last_update_step)
                    health_writer.writerow({"step": step, **replay_health}); health_f.flush()
                    self._wandb_log({f"train/{k}": v for k, v in replay_health.items()}, step=step)
                else:
                    replay_health = None
                if self._replay_ready() and step % int(self.config.learner.rollout_chunk_length) == 0:
                    for _ in range(int(self.config.learner.updates_per_rollout_chunk)):
                        seq = self._sample_replay()
                        metrics = self.learner.update(seq)
                        updates += 1
                    last_update_step = step
                    speed = float(step * self.num_envs) / max(time.time() - start, 1.0e-9)
                    replay_health = self._replay_health(step=step, updates=updates, last_update_step=last_update_step)
                    row = {"step": step, "env_steps_per_second": speed, **asdict(metrics), **replay_health}
                    loss_writer.writerow(row); loss_f.flush()
                    self._wandb_log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=step)
                comps = batch_step.info.get("reward_components", {}) if isinstance(batch_step.info, dict) else {}
                if comps:
                    flat = {"step": step}
                    for name, value in comps.items():
                        flat[name] = _component_mean(value)
                    if reward_writer is None:
                        reward_writer = csv.DictWriter(reward_f, fieldnames=list(flat.keys()))
                        reward_writer.writeheader()
                    reward_writer.writerow(flat); reward_f.flush()
                    self._wandb_log({f"reward/{k}": v for k, v in flat.items() if k != "step"}, step=step)
                if self._keep_latest_checkpoint_enabled() and step % int(self.config.training.checkpoint_interval_steps) == 0:
                    self._save_checkpoint("latest.pt", step=step, updates=updates)
                if step % int(self.config.training.eval_interval_steps) == 0:
                    eval_metrics = self.evaluate_detailed(max_steps=_eval_max_steps_for_config(self.config), episodes=int(self.config.training.eval_episodes))
                    score = self._selection_score(eval_metrics)
                    eval_metrics["selection_score"] = score
                    _append_csv_row(eval_path, {"step": step, "env_step": step * self.num_envs, **eval_metrics})
                    self._wandb_log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)
                    if self._retained_eval_checkpoints_enabled():
                        self._save_retained_eval_checkpoint(step=step, updates=updates, score=score, eval_metrics=eval_metrics)
                    if score > self.best_eval:
                        self.best_eval = score
                        self.best_eval_details = dict(eval_metrics)
                        self._remember_best_actor()
                        if self._save_checkpoints_enabled() and not self._retained_eval_checkpoints_enabled():
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
        if self._save_final_checkpoint_enabled():
            self._save_checkpoint("final.pt", step=self.steps, updates=updates)
        self._export("exports/final_actor", step=self.steps, updates=updates, eval_score=self.best_eval)
        metrics_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        return final


    def _train_local_replay_distributed(self) -> dict[str, Any]:
        if int(self.config.training.actor_workers) != 1:
            raise ValueError("local_replay does not use actor_workers")
        self._write_config_snapshot()
        losses_path = self.output_dir / "losses.csv"
        metrics_path = self.output_dir / "metrics.json"
        rewards_path = self.output_dir / "reward_components.csv"
        eval_path = self.output_dir / "eval_history.csv"
        health_path = self.output_dir / "replay_health.csv"
        replay_health_fields = [
            *self._replay_health_fields(),
            "replay_ready_all",
            "distributed_world_size",
            "local_envs_per_rank",
            "global_envs",
        ]
        loss_fields = ["step", "critic_loss", "actor_loss", "mean_kl", "std_kl", "q_mean", "target_q_mean", "actor_mle_loss", "actor_param_delta_norm", "action_mean_abs", "action_std_mean", "sampled_q_spread", "policy_weight_entropy", "policy_weight_max", "mpo_temperature", "mean_kl_penalty", "std_kl_penalty", "env_steps_per_second", *replay_health_fields]

        loss_f = reward_f = health_f = None
        loss_writer = health_writer = reward_writer = None
        if self._rank0():
            append_logs = self.resume_checkpoint is not None
            write_loss_header = (not append_logs) or not losses_path.exists() or losses_path.stat().st_size == 0
            write_health_header = (not append_logs) or not health_path.exists() or health_path.stat().st_size == 0
            loss_f = losses_path.open("a" if append_logs else "w", newline="", encoding="utf-8")
            reward_f = rewards_path.open("a" if append_logs else "w", newline="", encoding="utf-8")
            health_f = health_path.open("a" if append_logs else "w", newline="", encoding="utf-8")
            loss_writer = csv.DictWriter(loss_f, fieldnames=loss_fields)
            health_writer = csv.DictWriter(health_f, fieldnames=["step", *replay_health_fields])
            if write_loss_header:
                loss_writer.writeheader()
            if write_health_header:
                health_writer.writeheader()
            if append_logs and rewards_path.exists() and rewards_path.stat().st_size > 0:
                with rewards_path.open("r", newline="", encoding="utf-8") as existing_reward_f:
                    reader = csv.reader(existing_reward_f)
                    try:
                        reward_header = next(reader)
                    except StopIteration:
                        reward_header = []
                if reward_header:
                    reward_writer = csv.DictWriter(reward_f, fieldnames=reward_header)

        obs = self.env.reset()
        critic_obs = self.env.critic_obs()
        if self.resume_checkpoint is not None:
            obs = self._load_checkpoint(self.resume_checkpoint, restore_env=False, restore_replay=False, restore_rng=False)
            critic_obs = self.env.critic_obs()
            # Exact distributed resume would require per-rank env/replay/RNG state.
            # The saved checkpoint is rank-0 state, so warm resume keeps the learned
            # weights/optimiser state and reseeds each rank independently while replay refills.
            rank_seed = int(self.config.training.seed) + int(self.distributed_rank) * 1000003 + int(self.start_step) + 17
            torch.manual_seed(rank_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(rank_seed)
            np.random.seed(rank_seed % (2**32 - 1))
            random.seed(rank_seed)
            self._broadcast_trainable_state()
        start = time.time()
        updates = int(self.start_updates)
        local_step = 0
        env_steps = int(self.start_step)
        last_update_step = int(self.start_step)
        last_checkpoint_step = int(self.start_step)
        last_eval_step = int(self.start_step)
        rollout_chunks_seen = 0
        reward_acc = _RewardComponentAccumulator(self.device)
        progress = tqdm(total=max(self.steps - self.start_step, 0), desc="local-replay-train", unit="env-step", dynamic_ncols=True) if self._rank0() else None

        try:
            while env_steps < self.steps:
                assert obs is not None
                assert critic_obs is not None
                with torch.no_grad():
                    action, _logp, _mean = self.actor.sample(obs)
                batch_step = self.env.step(action)
                discount = torch.full((self.num_envs,), float(self.config.learner.discount), dtype=torch.float32, device=self.device)
                done = batch_step.terminated | batch_step.truncated
                self.replay.add_batch(obs, batch_step.applied_action, batch_step.reward, discount, batch_step.obs, done, critic_obs=critic_obs, next_critic_obs=batch_step.critic_obs)
                obs = self.env.reset_indices(done) if bool(torch.any(done).item()) else batch_step.obs
                critic_obs = self.env.critic_obs() if bool(torch.any(done).item()) else batch_step.critic_obs

                local_step += 1
                previous_env_steps = env_steps
                env_steps += int(self.global_num_envs)
                comps = batch_step.info.get("reward_components", {}) if isinstance(batch_step.info, dict) else {}
                if isinstance(comps, Mapping) and comps:
                    reward_acc.add(comps)

                if progress is not None:
                    progress.update(min(env_steps, self.steps) - min(previous_env_steps, self.steps))
                    progress.set_postfix(replay=self.replay.size, updates=updates, reward=f"{float(batch_step.reward.mean().detach().cpu()):.4f}", refresh=False)

                chunk_due = local_step % int(self.config.learner.rollout_chunk_length) == 0 or env_steps >= self.steps
                if not chunk_due:
                    continue
                rollout_chunks_seen += 1

                reward_means = reward_acc.means(distributed=self._distributed_initialized(), reset=True)
                if self._rank0() and reward_means:
                    flat = {"step": env_steps, **reward_means}
                    if reward_writer is None:
                        assert reward_f is not None
                        reward_writer = csv.DictWriter(reward_f, fieldnames=list(flat.keys()))
                        reward_writer.writeheader()
                    reward_writer.writerow(flat)
                    assert reward_f is not None
                    reward_f.flush()
                    self._wandb_log({f"reward/{k}": v for k, v in flat.items() if k != "step"}, step=env_steps)

                replay_health_local = self._replay_health(step=env_steps, updates=updates, last_update_step=last_update_step)
                ready_all = self._distributed_all_bool(bool(replay_health_local["replay_ready"]))
                replay_health_local["replay_ready_all"] = 1.0 if ready_all else 0.0
                replay_health_local["distributed_world_size"] = float(self.distributed_world_size)
                replay_health_local["local_envs_per_rank"] = float(self.num_envs)
                replay_health_local["global_envs"] = float(self.global_num_envs)
                replay_health = _distributed_mean_scalars(replay_health_local, device=self.device, enabled=self._distributed_initialized())
                if self._rank0() and health_writer is not None:
                    health_writer.writerow({"step": env_steps, **replay_health})
                    assert health_f is not None
                    health_f.flush()
                    self._wandb_log({f"train/{k}": v for k, v in replay_health.items()}, step=env_steps)

                if rollout_chunks_seen >= self._min_replay_health_check_chunks() and not ready_all:
                    status = "failed_replay_health"
                    details = {
                        "env_steps": int(env_steps),
                        "rollout_chunks_seen": int(rollout_chunks_seen),
                        "min_replay_health_check_chunks": int(self._min_replay_health_check_chunks()),
                        "replay_health": replay_health,
                    }
                    return self._failure_result(status, env_steps=env_steps, updates=updates, details=details)

                metrics = None
                if ready_all:
                    for _ in range(int(self.config.learner.updates_per_rollout_chunk)):
                        seq = self._sample_replay()
                        metrics = self.learner.update(seq)
                        updates += 1
                    last_update_step = env_steps
                    speed = float(env_steps) / max(time.time() - start, 1.0e-9)
                    replay_health_local = self._replay_health(step=env_steps, updates=updates, last_update_step=last_update_step)
                    replay_health_local["replay_ready_all"] = 1.0
                    replay_health_local["distributed_world_size"] = float(self.distributed_world_size)
                    replay_health_local["local_envs_per_rank"] = float(self.num_envs)
                    replay_health_local["global_envs"] = float(self.global_num_envs)
                    replay_health = _distributed_mean_scalars(replay_health_local, device=self.device, enabled=self._distributed_initialized())
                    metrics_mean = _distributed_mean_scalars(asdict(metrics), device=self.device, enabled=self._distributed_initialized()) if metrics is not None else {}
                    row = {"step": env_steps, "env_steps_per_second": speed, **metrics_mean, **replay_health}
                    if self._rank0() and loss_writer is not None:
                        loss_writer.writerow(row)
                        assert loss_f is not None
                        loss_f.flush()
                        self._wandb_log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=env_steps)

                if self._rank0() and self._keep_latest_checkpoint_enabled() and env_steps - last_checkpoint_step >= max(int(self.config.training.checkpoint_interval_steps), 1):
                    self._save_checkpoint("latest.pt", step=env_steps, updates=updates)
                    last_checkpoint_step = env_steps

                if self._rank0() and env_steps - last_eval_step >= max(int(self.config.training.eval_interval_steps), 1):
                    eval_metrics = self.evaluate_detailed(max_steps=_eval_max_steps_for_config(self.config), episodes=int(self.config.training.eval_episodes))
                    score = self._selection_score(eval_metrics)
                    eval_metrics["selection_score"] = score
                    _append_csv_row(eval_path, {"step": env_steps, "env_step": env_steps, **eval_metrics})
                    self._wandb_log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=env_steps)
                    if self._retained_eval_checkpoints_enabled():
                        self._save_retained_eval_checkpoint(step=env_steps, updates=updates, score=score, eval_metrics=eval_metrics)
                    if score > self.best_eval:
                        self.best_eval = score
                        self.best_eval_details = dict(eval_metrics)
                        self._remember_best_actor()
                        if self._save_checkpoints_enabled() and not self._retained_eval_checkpoints_enabled():
                            self._save_checkpoint("best.pt", step=env_steps, updates=updates)
                        self._export("exports/best_actor", step=env_steps, updates=updates, eval_score=score)
                    last_eval_step = env_steps

                self._barrier()
        finally:
            if progress is not None:
                progress.close()
            for handle in (loss_f, reward_f, health_f):
                if handle is not None:
                    handle.close()

        if self._rank0():
            if self._save_final_checkpoint_enabled():
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
                "distributed_mode": "local_replay",
                "rank": int(self.distributed_rank),
                "world_size": int(self.distributed_world_size),
                "local_envs_per_rank": int(self.num_envs),
                "total_training_envs": int(self.global_num_envs),
                "save_checkpoints": self._save_checkpoints_enabled(),
                "status": "completed",
            }
            metrics_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        else:
            final = {
                "start_step": self.start_step,
                "steps": env_steps,
                "env_steps": env_steps,
                "updates": updates,
                "device": str(self.device),
                "output_dir": str(self.output_dir),
                "distributed_mode": "local_replay",
                "rank": int(self.distributed_rank),
                "world_size": int(self.distributed_world_size),
                "local_envs_per_rank": int(self.num_envs),
                "total_training_envs": int(self.global_num_envs),
            }
        self._barrier()
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
        health_path = self.output_dir / "replay_health.csv"
        replay_health_fields = self._replay_health_fields()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        updates = int(self.start_updates)
        env_steps = int(self.start_step)
        last_update_step = int(self.start_step)
        last_checkpoint_step = 0
        last_eval_step = 0
        worker_rollout_counts = {str(index): 0 for index in range(worker_count)}
        start = time.time()
        try:
            with losses_path.open("w", newline="", encoding="utf-8") as loss_f, rewards_path.open("w", newline="", encoding="utf-8") as reward_f, health_path.open("w", newline="", encoding="utf-8") as health_f:
                loss_writer = csv.DictWriter(loss_f, fieldnames=["step", "critic_loss", "actor_loss", "mean_kl", "std_kl", "q_mean", "target_q_mean", "actor_mle_loss", "actor_param_delta_norm", "action_mean_abs", "action_std_mean", "sampled_q_spread", "policy_weight_entropy", "policy_weight_max", "mpo_temperature", "mean_kl_penalty", "std_kl_penalty", "env_steps_per_second", *replay_health_fields])
                health_writer = csv.DictWriter(health_f, fieldnames=["step", *replay_health_fields])
                loss_writer.writeheader()
                health_writer.writeheader()
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
                            critic_obs=torch.as_tensor(payload.get("critic_obs", payload["obs"])[t], dtype=torch.float32, device=self.device),
                            next_critic_obs=torch.as_tensor(payload.get("next_critic_obs", payload["next_obs"])[t], dtype=torch.float32, device=self.device),
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
                    replay_health = self._replay_health(step=env_steps, updates=updates, last_update_step=last_update_step)
                    health_writer.writerow({"step": env_steps, **replay_health}); health_f.flush()
                    self._wandb_log({f"train/{k}": v for k, v in replay_health.items()}, step=env_steps)
                    while self._replay_ready():
                        for _ in range(int(self.config.learner.updates_per_rollout_chunk)):
                            seq = self._sample_replay()
                            metrics = self.learner.update(seq)
                            updates += 1
                        last_update_step = env_steps
                        speed = float(env_steps) / max(time.time() - start, 1.0e-9)
                        replay_health = self._replay_health(step=env_steps, updates=updates, last_update_step=last_update_step)
                        row = {"step": env_steps, "env_steps_per_second": speed, **asdict(metrics), **replay_health}
                        loss_writer.writerow(row); loss_f.flush()
                        self._wandb_log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=env_steps)
                        if updates % 4 == 0:
                            broadcast_actor(param_queues, self.actor.state_dict())
                        break
                    if self._keep_latest_checkpoint_enabled() and env_steps - last_checkpoint_step >= max(int(self.config.training.checkpoint_interval_steps), 1):
                        self._save_checkpoint("latest.pt", step=env_steps, updates=updates)
                        last_checkpoint_step = env_steps
                    if env_steps - last_eval_step >= max(int(self.config.training.eval_interval_steps), 1):
                        eval_metrics = self.evaluate_detailed(max_steps=_eval_max_steps_for_config(self.config), episodes=int(self.config.training.eval_episodes))
                        score = self._selection_score(eval_metrics)
                        eval_metrics["selection_score"] = score
                        _append_csv_row(eval_path, {"step": env_steps, "env_step": env_steps, **eval_metrics})
                        self._wandb_log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=env_steps)
                        if self._retained_eval_checkpoints_enabled():
                            self._save_retained_eval_checkpoint(step=env_steps, updates=updates, score=score, eval_metrics=eval_metrics)
                        if score > self.best_eval:
                            self.best_eval = score
                            self.best_eval_details = dict(eval_metrics)
                            self._remember_best_actor()
                            if self._save_checkpoints_enabled() and not self._retained_eval_checkpoints_enabled():
                                self._save_checkpoint("best.pt", step=env_steps, updates=updates)
                            self._export("exports/best_actor", step=env_steps, updates=updates, eval_score=score)
                        last_eval_step = env_steps
                    progress.update(min(T * B, max(self.steps - (env_steps - T * B), 0)))
                    progress.set_postfix(replay=self.replay.size, updates=updates, refresh=False)
                progress.close()
        finally:
            stop_actor_workers(processes, stop, param_queues=param_queues, data_q=data_q)
        if self._save_final_checkpoint_enabled():
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
        eval_config = self._evaluation_config()
        env = TokamakMagneticControlEnv(eval_config, batch_size=batch_size, device=self.device, seed=int(eval_config.training.seed) + int(seed_offset))
        obs = env.reset()
        returns: list[float] = []
        episode_steps: list[int] = []
        totals = torch.zeros((env.batch_size,), dtype=torch.float32, device=self.device)
        steps = torch.zeros((env.batch_size,), dtype=torch.long, device=self.device)
        component_values: dict[str, list[float]] = {}
        early_component_values: dict[str, list[float]] = {}
        late_component_values: dict[str, list[float]] = {}
        padded_component_values: dict[str, list[float]] = {}
        padded_late_component_values: dict[str, list[float]] = {}
        full_episode_successes: list[float] = []
        max_steps_f = max(float(max_steps), 1.0)
        failure_padded_names = {
            "shape_error_mean_m",
            "shape_error_max_m",
            "ip_error_a",
            "current_over_limit_a",
            "current_usage_fraction",
            "boundary_found",
        }
        failure_ip_error_a = max(100000.0, 4.0 * float(eval_config.reward.ip_scale_a))
        failure_shape_error_m = float(eval_config.reward.boundary_missing_error_m)
        current_termination_over_limit_a = float(eval_config.sim.current_termination_over_limit_a)
        current_hard_termination_fraction = float(eval_config.sim.current_hard_termination_fraction)

        def _append_padded(name: str, values: np.ndarray, progress: np.ndarray) -> None:
            if name not in failure_padded_names or values.size == 0:
                return
            finite = np.isfinite(values)
            if np.any(finite):
                padded_component_values.setdefault(name, []).extend(values[finite].astype(float).tolist())
            if progress.shape == values.shape:
                late = finite & (progress >= 0.8)
                if np.any(late):
                    padded_late_component_values.setdefault(name, []).extend(values[late].astype(float).tolist())

        def _append_padded_failure(name: str, value: float, remaining: int, late_count: int) -> None:
            if remaining <= 0:
                return
            padded_component_values.setdefault(name, []).extend([float(value)] * int(remaining))
            if late_count > 0:
                padded_late_component_values.setdefault(name, []).extend([float(value)] * int(late_count))

        while len(returns) < int(episodes):
            if policy == "actor":
                action = self.actor.deterministic(obs)
            else:
                action = torch.zeros((env.batch_size, env.action_dim), dtype=torch.float32, device=self.device)
            out = env.step(action)
            totals += out.reward
            steps += 1
            comps = out.info.get("reward_components", {}) if isinstance(out.info, dict) else {}
            comp_arrays: dict[str, np.ndarray] = {}
            if isinstance(comps, dict):
                progress = (steps.to(torch.float32) / max_steps_f).detach().cpu().numpy().reshape(-1)
                early_cutoff = max(0.2, 1.0 / max_steps_f)
                for name, value in comps.items():
                    arr = np.asarray(_value_to_numpy(value), dtype=float).reshape(-1)
                    comp_arrays[str(name)] = arr
                    if arr.size:
                        finite = np.isfinite(arr)
                        component_values.setdefault(str(name), []).extend(arr[finite].astype(float).tolist())
                        _append_padded(str(name), arr, progress)
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
                terminated_cpu = out.terminated.detach().cpu().numpy().astype(bool)
                truncated_cpu = out.truncated.detach().cpu().numpy().astype(bool)
                totals_cpu = totals.detach().cpu().numpy().astype(float)
                steps_cpu = steps.detach().cpu().numpy().astype(int)

                def terminal_component(name: str, index: int, default: float = 0.0) -> float:
                    arr = comp_arrays.get(name)
                    if arr is None or index >= arr.size:
                        return float(default)
                    value = float(arr[index])
                    return value if np.isfinite(value) else float(default)

                for index, is_done in enumerate(done_cpu):
                    if is_done and len(returns) < int(episodes):
                        returns.append(float(totals_cpu[index]))
                        episode_steps.append(int(steps_cpu[index]))
                        finished_horizon = int(steps_cpu[index]) >= int(max_steps)
                        success = finished_horizon and not bool(terminated_cpu[index])
                        full_episode_successes.append(1.0 if success else 0.0)
                        remaining = max(int(max_steps) - int(steps_cpu[index]), 0)
                        if remaining > 0:
                            future_progress = (np.arange(int(steps_cpu[index]) + 1, int(max_steps) + 1, dtype=float) / max_steps_f)
                            late_count = int(np.count_nonzero(future_progress >= 0.8))
                            terminal_ip_error = terminal_component("ip_error_a", index, failure_ip_error_a)
                            terminal_current_over = terminal_component("current_over_limit_a", index, 0.0)
                            terminal_current_usage = terminal_component("current_usage_fraction", index, 0.0)
                            current_terminated = terminal_component("terminated_current", index, 0.0) > 0.5
                            if current_terminated:
                                terminal_current_over = max(float(terminal_current_over), current_termination_over_limit_a)
                                terminal_current_usage = max(float(terminal_current_usage), current_hard_termination_fraction)
                            _append_padded_failure("boundary_found", 0.0, remaining, late_count)
                            _append_padded_failure("shape_error_mean_m", failure_shape_error_m, remaining, late_count)
                            _append_padded_failure("shape_error_max_m", failure_shape_error_m, remaining, late_count)
                            _append_padded_failure("ip_error_a", max(failure_ip_error_a, float(terminal_ip_error)), remaining, late_count)
                            _append_padded_failure("current_over_limit_a", terminal_current_over, remaining, late_count)
                            _append_padded_failure("current_usage_fraction", terminal_current_usage, remaining, late_count)
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
        if full_episode_successes:
            successes = np.asarray(full_episode_successes[: int(episodes)], dtype=float)
            metrics["full_episode_success"] = float(np.nanmean(successes))
            metrics["termination_failure_fraction"] = float(np.nanmean(successes < 0.5))
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
            "physical_cost",
            "shape_mean_loss",
            "shape_max_loss",
            "ip_loss",
            "current_loss",
            "derivative_loss",
            "action_loss",
            "delta_action_loss",
            "terminated_boundary",
            "terminated_current",
            "current_over_limit_steps",
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
        drift_metrics = {"shape_error_mean_m", "shape_error_max_m", "ip_error_a", "physical_cost", "current_usage_fraction"}
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
        padded_max_metrics = {"shape_error_mean_m", "shape_error_max_m", "ip_error_a", "current_over_limit_a", "current_usage_fraction"}
        padded_min_metrics = {"boundary_found"}
        for name, values in padded_component_values.items():
            arr = np.asarray(values, dtype=float)
            if arr.size:
                metrics[f"padded_{name}"] = float(np.nanmean(arr))
                if name in padded_max_metrics:
                    metrics[f"padded_{name}_max"] = float(np.nanmax(arr))
                if name in padded_min_metrics:
                    metrics[f"padded_{name}_min"] = float(np.nanmin(arr))
                if name == "current_over_limit_a":
                    metrics["padded_current_over_limit_fraction"] = float(np.nanmean(arr > 0.0))
        for name, values in padded_late_component_values.items():
            arr = np.asarray(values, dtype=float)
            if arr.size:
                metrics[f"padded_{name}_late"] = float(np.nanmean(arr))
                if name in padded_max_metrics:
                    metrics[f"padded_{name}_late_max"] = float(np.nanmax(arr))
                if name in padded_min_metrics:
                    metrics[f"padded_{name}_late_min"] = float(np.nanmin(arr))
                if name == "current_over_limit_a":
                    metrics["padded_current_over_limit_fraction_late"] = float(np.nanmean(arr > 0.0))
        return metrics

    def _evaluation_config(self) -> ExperimentConfig:
        if str(self.config.sim.reset_source) != "csv_initial_states":
            return self.config
        return replace(self.config, sim=replace(self.config.sim, csv_initial_state_split="holdout"))

    @staticmethod
    def _selection_score(metrics: dict[str, float]) -> float:
        def metric(name: str, default: float) -> float:
            try:
                value = float(metrics.get(name, default))
            except (TypeError, ValueError):
                return default
            return value if np.isfinite(value) else default

        completion = metric("mean_episode_completion", 0.0)
        full_success = metric("full_episode_success", completion)
        min_completion = metric("min_episode_completion", completion)
        boundary_late = metric("padded_boundary_found_late_min", metric("boundary_found_late_min", metric("boundary_found_min", metric("boundary_found", 0.0))))
        terminated_boundary = metric("terminated_boundary_late_max", metric("terminated_boundary_max", metric("terminated_boundary", 0.0)))
        current_over = metric("padded_current_over_limit_a_late_max", metric("current_over_limit_a_late_max", metric("current_over_limit_a_max", metric("current_over_limit_a", 0.0))))
        current_fraction = metric("padded_current_over_limit_fraction_late", metric("current_over_limit_fraction_late", metric("current_over_limit_fraction", 0.0)))
        shape_mean = metric("padded_shape_error_mean_m_late", metric("shape_error_mean_m_late", metric("shape_error_mean_m", 1.0)))
        shape_max = metric("padded_shape_error_max_m_late", metric("shape_error_max_m_late", metric("shape_error_max_m", 1.0)))
        ip_error = metric("padded_ip_error_a_late", metric("ip_error_a_late", metric("ip_error_a", 1.0e6)))
        action_rms = metric("action_rms_late", metric("action_rms", 1.0))
        delta_action_rms = metric("delta_action_rms_late", metric("delta_action_rms", 1.0))

        objective = (
            200.0 * max(0.95 - completion, 0.0)
            + 200.0 * max(0.95 - full_success, 0.0)
            + 200.0 * max(0.90 - min_completion, 0.0)
            + 1000.0 * max(0.999 - boundary_late, 0.0)
            + 100.0 * max(terminated_boundary, 0.0)
            + 3.0 * max(current_over, 0.0) / 20000.0
            + 2.0 * max(current_fraction, 0.0)
            + 2.0 * max(shape_mean, 0.0) / 0.03
            + 1.0 * max(shape_max, 0.0) / 0.08
            + 2.0 * max(ip_error, 0.0) / 25000.0
            + 0.5 * max(action_rms, 0.0) / 0.5
            + 0.25 * max(delta_action_rms, 0.0) / 0.1
        )
        return float(-objective)

    def _metadata(self, *, step: int, updates: int, eval_score: float | None = None) -> dict[str, object]:
        exact_resume_supported = not bool(self._last_actor_devices) and self.distributed_mode != "local_replay"
        raw_step = int(step)
        if self._last_actor_devices or self.distributed_mode == "local_replay":
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
        if self.distributed_mode == "local_replay":
            out["distributed_mode"] = "local_replay"
            out["rank"] = int(self.distributed_rank)
            out["world_size"] = int(self.distributed_world_size)
            out["local_envs"] = int(self.num_envs)
            out["total_training_envs"] = int(self.global_num_envs)
            out["resume_limitation"] = "local-replay distributed environment/replay state is sharded per rank"
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

    def _keep_latest_checkpoint_enabled(self) -> bool:
        return self._save_checkpoints_enabled() and bool(self.config.training.keep_latest_checkpoint)

    def _retained_eval_checkpoints_enabled(self) -> bool:
        return self._save_checkpoints_enabled() and (
            int(self.config.training.eval_checkpoint_top_k) > 0
            or int(self.config.training.milestone_checkpoint_interval_steps) > 0
        )

    def _save_final_checkpoint_enabled(self) -> bool:
        return self._save_checkpoints_enabled() and not self._retained_eval_checkpoints_enabled()

    def _save_retained_eval_checkpoint(self, *, step: int, updates: int, score: float, eval_metrics: Mapping[str, object]) -> Path:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"eval_step_{int(step):012d}.pt"
        checkpoint_path = self._save_checkpoint(checkpoint_name, step=step, updates=updates, eval_score=score)
        entry = {
            "step": int(step),
            "updates": int(updates),
            "score": float(score),
            "path": checkpoint_name,
            "milestone": self._is_milestone_checkpoint_step(step),
            "eval_metrics": self._json_scalar_mapping(eval_metrics),
        }
        entries_by_path: dict[str, dict[str, object]] = {}
        for old in self._load_eval_checkpoint_index(ckpt_dir):
            if not isinstance(old, dict):
                continue
            name = str(old.get("path", ""))
            if not name:
                continue
            path = ckpt_dir / name
            if path.exists() or path.is_symlink() or name == checkpoint_name:
                entries_by_path[name] = dict(old)
        entries_by_path[checkpoint_name] = entry

        entries = list(entries_by_path.values())
        top_k = max(0, int(self.config.training.eval_checkpoint_top_k))
        top_entries = self._sort_checkpoint_entries(entries)[:top_k] if top_k > 0 else []
        keep_names = {str(item["path"]) for item in top_entries}
        keep_names.update(str(item["path"]) for item in entries if bool(item.get("milestone", False)))

        retained: list[dict[str, object]] = []
        for item in entries:
            name = str(item.get("path", ""))
            path = ckpt_dir / name
            if name in keep_names:
                if path.exists() or path.is_symlink():
                    retained.append(item)
                continue
            if path.exists() or path.is_symlink():
                path.unlink()

        retained = self._sort_checkpoint_entries(retained)
        best = retained[0] if retained else None
        if best is not None:
            self._point_best_checkpoint_at(ckpt_dir / str(best["path"]))
        self._write_eval_checkpoint_index(ckpt_dir, retained)
        return checkpoint_path

    def _is_milestone_checkpoint_step(self, step: int) -> bool:
        interval = int(self.config.training.milestone_checkpoint_interval_steps)
        return interval > 0 and int(step) > 0 and int(step) % interval == 0

    @staticmethod
    def _load_eval_checkpoint_index(ckpt_dir: Path) -> list[dict[str, object]]:
        path = ckpt_dir / "eval_checkpoints.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        raw = data.get("checkpoints", []) if isinstance(data, dict) else []
        return [dict(item) for item in raw if isinstance(item, dict)]

    def _write_eval_checkpoint_index(self, ckpt_dir: Path, entries: list[dict[str, object]]) -> None:
        path = ckpt_dir / "eval_checkpoints.json"
        data = {
            "top_k": int(self.config.training.eval_checkpoint_top_k),
            "milestone_interval_steps": int(self.config.training.milestone_checkpoint_interval_steps),
            "best_checkpoint": str(entries[0]["path"]) if entries else None,
            "checkpoints": entries,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _sort_checkpoint_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
        def key(item: dict[str, object]) -> tuple[int, float, int]:
            raw_score = item.get("score", -float("inf"))
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = -float("inf")
            finite = 1 if np.isfinite(score) else 0
            if not finite:
                score = -float("inf")
            return finite, score, int(item.get("step", 0))

        return sorted(entries, key=key, reverse=True)

    @staticmethod
    def _json_scalar_mapping(values: Mapping[str, object]) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in values.items():
            if isinstance(value, (str, bool)) or value is None:
                out[str(key)] = value
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            out[str(key)] = number if np.isfinite(number) else None
        return out

    @staticmethod
    def _point_best_checkpoint_at(target: Path) -> None:
        best = target.parent / "best.pt"
        if best.exists() or best.is_symlink():
            best.unlink()
        try:
            best.symlink_to(target.name)
            return
        except OSError:
            pass
        try:
            os.link(target, best)
            return
        except OSError:
            pass
        shutil.copy2(target, best)

    def _save_checkpoint(self, name: str, *, step: int, updates: int, eval_score: float | None = None) -> Path:
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
            if self.distributed_mode == "local_replay":
                env_state = None
            else:
                env_state = self.env.state_dict()
        except RuntimeError:
            if int(self.config.training.actor_workers) > 1 or self.distributed_mode == "local_replay":
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
            "metadata": self._metadata(step=step, updates=updates, eval_score=eval_score),
            "network": asdict(self.config.network),
            "learner": asdict(self.config.learner),
            "reward": asdict(self.config.reward),
            "sim": self._sim_resume_fragment(),
            "training_state": {"step": int(step), "updates": int(updates)},
        }, path)
        return path

    def _load_checkpoint(self, path: str | Path, *, restore_env: bool = True, restore_replay: bool = True, restore_rng: bool = True) -> torch.Tensor:
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
        if restore_replay:
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
        if restore_rng and isinstance(rng, dict):
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

    def _export(self, relative: str, *, step: int, updates: int, eval_score: float | None) -> Path | None:
        if not self.export_policy:
            return None
        return export_deterministic_actor(actor=self.actor, export_dir=self.output_dir / relative, schema=self.schema, normalization=self.normalization, metadata=self._metadata(step=step, updates=updates, eval_score=eval_score))

    def _write_config_snapshot(self) -> None:
        if self.distributed_mode == "local_replay" and not self._rank0():
            return
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
