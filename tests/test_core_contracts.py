from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.networks import FeedForwardGaussianActor, RecurrentQCritic
from tokamak_rl_v2.rewards import T15StaticBoundaryReward
from tokamak_rl_v2.rewards import transforms
from tokamak_rl_v2.training.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/t15_static_boundary.yaml"


def _small_config(tmp_path: Path):
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=12))
    cfg = replace(cfg, network=replace(cfg.network, hidden_dim=16, critic_hidden_dim=16, critic_mlp_hidden_dim=16))
    cfg = replace(cfg, learner=replace(cfg.learner, batch_size=4, unroll_length=2, rollout_chunk_length=2, updates_per_rollout_chunk=1, action_samples=4))
    cfg = replace(cfg, training=replace(cfg.training, output_dir=tmp_path, steps=8, num_envs=2, checkpoint_interval_steps=8, eval_interval_steps=8, eval_episodes=2, eval_max_steps=4))
    return cfg


def test_network_shapes() -> None:
    actor = FeedForwardGaussianActor(obs_dim=17, action_dim=5, hidden_dim=16)
    critic = RecurrentQCritic(obs_dim=17, action_dim=5, lstm_hidden_dim=16, mlp_hidden_dim=16)
    obs = torch.zeros((3, 17))
    out = actor(obs)
    assert out.mean.shape == (3, 5)
    assert out.std.shape == (3, 5)
    q, state = critic(obs, torch.zeros((3, 5)))
    assert q.shape == (3, 1)
    assert state.h.shape[-1] == 16


def test_quality_transforms_hit_declared_points() -> None:
    x = torch.tensor([0.005, 0.05], dtype=torch.float32)
    y = transforms.softplus(x, good=0.005, bad=0.05)
    assert torch.isclose(y[0], torch.tensor(1.0), atol=1.0e-5)
    assert torch.isclose(y[1], torch.tensor(0.1), atol=1.0e-4)
    s = transforms.sigmoid(torch.tensor([500.0, 20000.0]), good=500.0, bad=20000.0)
    assert s[0] > 0.94
    assert s[1] < 0.06


def test_environment_reset_step_contract() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=4))
    env = TokamakMagneticControlEnv(cfg, batch_size=2, device="cpu", seed=1)
    obs = env.reset()
    assert obs.shape == (2, env.obs_dim)
    result = env.step(torch.zeros((2, env.action_dim)))
    assert result.obs.shape == (2, env.obs_dim)
    assert result.reward.shape == (2,)
    assert torch.isfinite(result.reward).all()


def test_small_training_writes_export(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)
    result = trainer.train()
    assert result["updates"] > 0
    assert (tmp_path / "checkpoints" / "final.pt").exists()
    assert (tmp_path / "exports" / "final_actor" / "policy_weights.npz").exists()
    assert (tmp_path / "losses.csv").exists()
    assert (tmp_path / "reward_components.csv").exists()


def test_small_distributed_training_writes_export(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    cfg = replace(cfg, training=replace(cfg.training, actor_workers=2, eval_interval_steps=1000))
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)
    result = trainer.train()
    assert result["updates"] > 0
    assert result["actor_workers"] == 2
    assert (tmp_path / "checkpoints" / "final.pt").exists()
    assert (tmp_path / "exports" / "final_actor" / "policy_weights.npz").exists()
