from __future__ import annotations

import csv
import json
import time
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
    def __init__(self, config: ExperimentConfig, *, steps: int | None = None, num_envs: int | None = None, device: str | None = None, output_dir: str | Path | None = None, wandb_run=None) -> None:
        self.config = config
        self.steps = int(config.training.steps if steps is None else steps)
        self.num_envs = int(config.training.num_envs if num_envs is None else num_envs)
        self.device = _resolve_device(config.training.device if device is None else device)
        self.output_dir = Path(config.training.output_dir if output_dir is None else output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run
        torch.manual_seed(int(config.training.seed))
        np.random.seed(int(config.training.seed))
        self.env = TokamakMagneticControlEnv(config, batch_size=self.num_envs, device=self.device, seed=int(config.training.seed))
        self.actor = FeedForwardGaussianActor(self.env.obs_dim, self.env.action_dim, config.network.hidden_dim).to(self.device)
        self.critic = RecurrentQCritic(self.env.obs_dim, self.env.action_dim, config.network.critic_hidden_dim, config.network.critic_mlp_hidden_dim).to(self.device)
        self.target_actor = FeedForwardGaussianActor(self.env.obs_dim, self.env.action_dim, config.network.hidden_dim).to(self.device)
        self.target_critic = RecurrentQCritic(self.env.obs_dim, self.env.action_dim, config.network.critic_hidden_dim, config.network.critic_mlp_hidden_dim).to(self.device)
        self.learner = MaximumAPosterioriPolicyOptimiser(actor=self.actor, critic=self.critic, target_actor=self.target_actor, target_critic=self.target_critic, config=config.learner, device=self.device)
        capacity = int(config.learner.replay_capacity_episodes) * int(config.sim.max_episode_steps)
        self.replay = FIFOSequenceReplay(capacity_steps=capacity, obs_dim=self.env.obs_dim, action_dim=self.env.action_dim, device=self.device)
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
            loss_writer = csv.DictWriter(loss_f, fieldnames=["step", "critic_loss", "actor_loss", "mean_kl", "std_kl", "q_mean", "target_q_mean", "env_steps_per_second"])
            reward_writer = None
            loss_writer.writeheader()
            obs = self.env.reset()
            start = time.time()
            updates = 0
            progress = tqdm(total=self.steps, desc="train", unit="step", dynamic_ncols=True)
            for step in range(1, self.steps + 1):
                with torch.no_grad():
                    action, _logp, _mean = self.actor.sample(obs)
                batch_step = self.env.step(action)
                discount = torch.full((self.num_envs,), float(self.config.learner.discount), dtype=torch.float32, device=self.device)
                done = batch_step.terminated | batch_step.truncated
                self.replay.add_batch(obs, action, batch_step.reward, discount, batch_step.obs, done)
                obs = batch_step.obs
                if bool(torch.any(done).item()):
                    obs = self.env.reset()
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
        final = {"steps": self.steps, "updates": updates, "best_eval": self.best_eval, "best_eval_details": self.best_eval_details, "device": str(self.device), "output_dir": str(self.output_dir)}
        self._save_checkpoint("final.pt", step=self.steps, updates=updates)
        self._export("exports/final_actor", step=self.steps, updates=updates, eval_score=self.best_eval)
        metrics_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        return final


    def _train_distributed(self) -> dict[str, Any]:
        self._write_config_snapshot()
        worker_count = int(self.config.training.actor_workers)
        envs_per_worker = max(1, self.num_envs // worker_count)
        processes, param_queues, data_q, stop = start_actor_workers(
            config=self.config,
            actor_state_dict=self.actor.state_dict(),
            worker_count=worker_count,
            envs_per_worker=envs_per_worker,
            rollout_chunk_length=int(self.config.learner.rollout_chunk_length),
            device=str(self.device),
            seed=int(self.config.training.seed),
        )
        losses_path = self.output_dir / "losses.csv"
        metrics_path = self.output_dir / "metrics.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        updates = 0
        env_steps = 0
        start = time.time()
        try:
            with losses_path.open("w", newline="", encoding="utf-8") as loss_f:
                loss_writer = csv.DictWriter(loss_f, fieldnames=["step", "critic_loss", "actor_loss", "mean_kl", "std_kl", "q_mean", "target_q_mean", "env_steps_per_second"])
                loss_writer.writeheader()
                progress = tqdm(total=self.steps, desc="distributed-train", unit="step", dynamic_ncols=True)
                while env_steps < self.steps:
                    payload = data_q.get(timeout=300.0)
                    T, B = int(payload["reward"].shape[0]), int(payload["reward"].shape[1])
                    for t in range(T):
                        self.replay.add_batch(
                            torch.as_tensor(payload["obs"][t], dtype=torch.float32, device=self.device),
                            torch.as_tensor(payload["action"][t], dtype=torch.float32, device=self.device),
                            torch.as_tensor(payload["reward"][t], dtype=torch.float32, device=self.device),
                            torch.as_tensor(payload["discount"][t], dtype=torch.float32, device=self.device),
                            torch.as_tensor(payload["next_obs"][t], dtype=torch.float32, device=self.device),
                            torch.as_tensor(payload["done"][t], dtype=torch.bool, device=self.device),
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
                    if env_steps % max(int(self.config.training.checkpoint_interval_steps), 1) < T * B:
                        self._save_checkpoint("latest.pt", step=env_steps, updates=updates)
                    progress.update(min(T * B, max(self.steps - (env_steps - T * B), 0)))
                    progress.set_postfix(replay=self.replay.size, updates=updates, refresh=False)
                progress.close()
        finally:
            stop_actor_workers(processes, stop)
        self._save_checkpoint("final.pt", step=env_steps, updates=updates)
        self._export("exports/final_actor", step=env_steps, updates=updates, eval_score=None)
        final = {"steps": env_steps, "updates": updates, "best_eval": self.best_eval, "device": str(self.device), "output_dir": str(self.output_dir), "actor_workers": worker_count}
        metrics_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        return final

    @torch.no_grad()
    def evaluate(self, *, episodes: int, max_steps: int) -> float:
        return float(self.evaluate_detailed(episodes=episodes, max_steps=max_steps)["mean_return"])

    @torch.no_grad()
    def evaluate_detailed(self, *, episodes: int, max_steps: int, policy: Literal["actor", "no_control"] = "actor") -> dict[str, float]:
        if policy not in {"actor", "no_control"}:
            raise ValueError(f"unsupported evaluation policy: {policy}")
        env = TokamakMagneticControlEnv(self.config, batch_size=max(1, min(int(episodes), self.num_envs)), device=self.device, seed=int(self.config.training.seed) + 100000)
        returns: list[float] = []
        component_values: dict[str, list[float]] = {}
        remaining = int(episodes)
        while remaining > 0:
            obs = env.reset()
            total = torch.zeros((env.batch_size,), dtype=torch.float32, device=self.device)
            for _ in range(int(max_steps)):
                if policy == "actor":
                    action = self.actor.deterministic(obs)
                else:
                    action = torch.zeros((env.batch_size, env.action_dim), dtype=torch.float32, device=self.device)
                out = env.step(action)
                total += out.reward
                comps = out.info.get("reward_components", {}) if isinstance(out.info, dict) else {}
                if isinstance(comps, dict):
                    for name, value in comps.items():
                        arr = np.asarray(value, dtype=float).reshape(-1)
                        if arr.size:
                            component_values.setdefault(str(name), []).extend(arr[np.isfinite(arr)].astype(float).tolist())
                obs = out.obs
                if bool(torch.all(out.terminated | out.truncated).item()):
                    break
            returns.extend(total.detach().cpu().numpy().astype(float).tolist())
            remaining -= env.batch_size
        selected_returns = np.asarray(returns[: int(episodes)], dtype=float)
        metrics: dict[str, float] = {"mean_return": float(np.nanmean(selected_returns)) if selected_returns.size else float("nan")}
        for name, values in component_values.items():
            arr = np.asarray(values, dtype=float)
            if arr.size:
                metrics[name] = float(np.nanmean(arr))
        return metrics

    def _metadata(self, *, step: int, updates: int, eval_score: float | None = None) -> dict[str, object]:
        return {"experiment": self.config.name, "step": int(step), "updates": int(updates), "eval_score": eval_score, "device": str(self.device), "algorithm": "Maximum a Posteriori Policy Optimisation", "plant": "tokamak-sim"}

    def _save_checkpoint(self, name: str, *, step: int, updates: int) -> Path:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / name
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "schema": self.schema,
            "normalization": self.normalization,
            "metadata": self._metadata(step=step, updates=updates),
            "network": asdict(self.config.network),
            "learner": asdict(self.config.learner),
        }, path)
        return path

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
    return dev
