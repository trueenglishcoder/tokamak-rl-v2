from __future__ import annotations

import csv
import json
import queue
import time
import random
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
from tokamak_rl_v2.networks import FeedForwardGaussianActor, RecurrentQCritic
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
        self.actor = FeedForwardGaussianActor(self.env.obs_dim, self.env.action_dim, config.network.hidden_dim).to(self.device)
        self.critic = RecurrentQCritic(self.env.obs_dim, self.env.action_dim, config.network.critic_hidden_dim, config.network.critic_mlp_hidden_dim).to(self.device)
        self.target_actor = FeedForwardGaussianActor(self.env.obs_dim, self.env.action_dim, config.network.hidden_dim).to(self.device)
        self.target_critic = RecurrentQCritic(self.env.obs_dim, self.env.action_dim, config.network.critic_hidden_dim, config.network.critic_mlp_hidden_dim).to(self.device)
        self.learner = MaximumAPosterioriPolicyOptimiser(actor=self.actor, critic=self.critic, target_actor=self.target_actor, target_critic=self.target_critic, config=config.learner, device=self.device)
        self.replay = FIFOSequenceReplay(capacity_episodes=int(config.learner.replay_capacity_episodes), max_episode_steps=int(config.sim.max_episode_steps), active_envs=self.num_envs, obs_dim=self.env.obs_dim, action_dim=self.env.action_dim, device=self.device)
        self.schema = self.env.export_schema()
        self.normalization = self.env.normalization()
        self.best_eval = -float("inf")
        self.best_eval_details: dict[str, float] = {}
        self._configure_wandb_metrics()

    def _configure_wandb_metrics(self) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.define_metric("global_step")
            for prefix in ("train/*", "reward/*", "eval/*"):
                self.wandb_run.define_metric(prefix, step_metric="global_step")
        except Exception:
            pass

    def _wandb_log(self, values: dict[str, object], *, step: int) -> None:
        if self.wandb_run is None:
            return
        payload = {"global_step": int(step), **values}
        self.wandb_run.log(payload, step=int(step))

    def train(self) -> dict[str, Any]:
        if int(self.config.training.actor_workers) > 1:
            return self._train_distributed()
        self._write_config_snapshot()
        losses_path = self.output_dir / "losses.csv"
        metrics_path = self.output_dir / "metrics.json"
        rewards_path = self.output_dir / "reward_components.csv"
        with losses_path.open("w", newline="", encoding="utf-8") as loss_f, rewards_path.open("w", newline="", encoding="utf-8") as reward_f:
            loss_writer = csv.DictWriter(loss_f, fieldnames=["step", "critic_loss", "actor_loss", "mean_kl", "std_kl", "q_mean", "target_q_mean", "actor_mle_loss", "actor_param_delta_norm", "action_mean_abs", "action_std_mean", "sampled_q_spread", "policy_weight_entropy", "env_steps_per_second"])
            reward_writer = None
            loss_writer.writeheader()
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
                self.replay.add_batch(obs, action, batch_step.reward, discount, batch_step.obs, done)
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
                if step % int(self.config.training.checkpoint_interval_steps) == 0:
                    self._save_checkpoint("latest.pt", step=step, updates=updates)
                if step % int(self.config.training.eval_interval_steps) == 0:
                    eval_metrics = self.evaluate_detailed(max_steps=int(self.config.training.eval_max_steps), episodes=int(self.config.training.eval_episodes))
                    score = float(eval_metrics["mean_return"])
                    self._wandb_log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)
                    if score > self.best_eval:
                        self.best_eval = score
                        self.best_eval_details = dict(eval_metrics)
                        self._save_checkpoint("best.pt", step=step, updates=updates)
                        self._export("exports/best_actor", step=step, updates=updates, eval_score=score)
                progress.update(1)
                progress.set_postfix(replay=self.replay.size, updates=updates, reward=f"{float(batch_step.reward.mean().detach().cpu()):.4f}", refresh=False)
            progress.close()
        final = {"start_step": self.start_step, "steps": self.steps, "updates": updates, "best_eval": self.best_eval, "best_eval_details": self.best_eval_details, "device": str(self.device), "output_dir": str(self.output_dir)}
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
            self._load_checkpoint(self.resume_checkpoint, restore_env=False)
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        updates = int(self.start_updates)
        env_steps = int(self.start_step)
        last_checkpoint_step = 0
        last_eval_step = 0
        worker_rollout_counts = {str(index): 0 for index in range(worker_count)}
        start = time.time()
        try:
            with losses_path.open("w", newline="", encoding="utf-8") as loss_f:
                loss_writer = csv.DictWriter(loss_f, fieldnames=["step", "critic_loss", "actor_loss", "mean_kl", "std_kl", "q_mean", "target_q_mean", "actor_mle_loss", "actor_param_delta_norm", "action_mean_abs", "action_std_mean", "sampled_q_spread", "policy_weight_entropy", "env_steps_per_second"])
                loss_writer.writeheader()
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
                    if env_steps - last_checkpoint_step >= max(int(self.config.training.checkpoint_interval_steps), 1):
                        self._save_checkpoint("latest.pt", step=env_steps, updates=updates)
                        last_checkpoint_step = env_steps
                    if env_steps - last_eval_step >= max(int(self.config.training.eval_interval_steps), 1):
                        eval_metrics = self.evaluate_detailed(max_steps=int(self.config.training.eval_max_steps), episodes=int(self.config.training.eval_episodes))
                        score = float(eval_metrics["mean_return"])
                        self._wandb_log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=env_steps)
                        if score > self.best_eval:
                            self.best_eval = score
                            self.best_eval_details = dict(eval_metrics)
                            self._save_checkpoint("best.pt", step=env_steps, updates=updates)
                            self._export("exports/best_actor", step=env_steps, updates=updates, eval_score=score)
                        last_eval_step = env_steps
                    progress.update(min(T * B, max(self.steps - (env_steps - T * B), 0)))
                    progress.set_postfix(replay=self.replay.size, updates=updates, refresh=False)
                progress.close()
        finally:
            stop_actor_workers(processes, stop)
        self._save_checkpoint("final.pt", step=env_steps, updates=updates)
        self._export("exports/final_actor", step=env_steps, updates=updates, eval_score=None)
        final = {
            "start_step": self.start_step,
            "steps": env_steps,
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
    def evaluate_detailed(self, *, episodes: int, max_steps: int, policy: Literal["actor", "no_control"] = "actor") -> dict[str, float]:
        if policy not in {"actor", "no_control"}:
            raise ValueError(f"unsupported evaluation policy: {policy}")
        batch_size = max(1, min(int(episodes), self.num_envs))
        env = TokamakMagneticControlEnv(self.config, batch_size=batch_size, device=self.device, seed=int(self.config.training.seed) + 100000)
        obs = env.reset()
        returns: list[float] = []
        totals = torch.zeros((env.batch_size,), dtype=torch.float32, device=self.device)
        steps = torch.zeros((env.batch_size,), dtype=torch.long, device=self.device)
        component_values: dict[str, list[float]] = {}
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
                for name, value in comps.items():
                    arr = np.asarray(value, dtype=float).reshape(-1)
                    if arr.size:
                        component_values.setdefault(str(name), []).extend(arr[np.isfinite(arr)].astype(float).tolist())
            done = out.terminated | out.truncated | (steps >= int(max_steps))
            if bool(torch.any(done).item()):
                done_cpu = done.detach().cpu().numpy().astype(bool)
                totals_cpu = totals.detach().cpu().numpy().astype(float)
                for index, is_done in enumerate(done_cpu):
                    if is_done and len(returns) < int(episodes):
                        returns.append(float(totals_cpu[index]))
                totals = torch.where(done, torch.zeros_like(totals), totals)
                steps = torch.where(done, torch.zeros_like(steps), steps)
                obs = env.reset_indices(done) if len(returns) < int(episodes) else out.obs
            else:
                obs = out.obs
        selected_returns = np.asarray(returns[: int(episodes)], dtype=float)
        metrics: dict[str, float] = {"mean_return": float(np.nanmean(selected_returns)) if selected_returns.size else float("nan")}
        for name, values in component_values.items():
            arr = np.asarray(values, dtype=float)
            if arr.size:
                metrics[name] = float(np.nanmean(arr))
        return metrics

    def _metadata(self, *, step: int, updates: int, eval_score: float | None = None) -> dict[str, object]:
        out: dict[str, object] = {"experiment": self.config.name, "step": int(step), "updates": int(updates), "eval_score": eval_score, "device": str(self.device), "learner_device": str(self.device), "algorithm": "Maximum a Posteriori Policy Optimisation", "plant": "tokamak-sim", "sim_compute_backend": self.config.sim.compute_backend}
        if self._last_actor_devices:
            out["actor_devices"] = list(self._last_actor_devices)
            out["actor_workers"] = len(self._last_actor_devices)
            out["envs_per_worker"] = self._last_envs_per_worker
            out["total_training_envs"] = self._last_total_training_envs
        return out

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
            env_state = None
        torch.save({
            "checkpoint_version": 2,
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
            if "torch_cuda" in rng and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
                except Exception:
                    pass
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
