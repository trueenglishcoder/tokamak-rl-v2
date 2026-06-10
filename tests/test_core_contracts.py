from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.config.schema import LearnerConfig
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.networks import FeedForwardGaussianActor, RecurrentQCritic
from tokamak_rl_v2.rewards import T15StaticBoundaryReward
from tokamak_rl_v2.rewards import transforms
from tokamak_rl_v2.training.mpo import MaximumAPosterioriPolicyOptimiser
from tokamak_rl_v2.training.trainer import Trainer
from tokamak_rl_v2.training.reward_search import _rank_rows, main as reward_search_main
from tokamak_rl_v2.training.cli import _device_list


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
    schema = env.export_schema()
    assert schema["observation_kind"] == "joint_state_v1"
    assert "diagnostics" not in schema
    assert "psi_flat" in schema["feature_order"]
    assert "measured_boundary_radii" in schema["feature_order"]
    assert "boundary_radii_error" in schema["feature_order"]
    assert env.obs_dim == schema["feature_slices"]["target_preview"][1]
    assert "flux_scale" not in env.normalization()
    assert "field_scale" not in env.normalization()
    assert "bdot_scale" not in env.normalization()
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
    cfg = replace(cfg, training=replace(cfg.training, actor_workers=2, actor_devices=("cpu", "cpu"), eval_interval_steps=1000))
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)
    result = trainer.train()
    assert result["updates"] > 0
    assert result["actor_workers"] == 2
    assert result["learner_device"] == "cpu"
    assert result["actor_devices"] == ["cpu", "cpu"]
    assert result["envs_per_worker"] == 1
    assert result["total_training_envs"] == 2
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert set(metrics["worker_rollout_counts"]) == {"0", "1"}
    assert sum(metrics["worker_rollout_counts"].values()) >= result["steps"]
    assert (tmp_path / "checkpoints" / "final.pt").exists()
    assert (tmp_path / "exports" / "final_actor" / "policy_weights.npz").exists()


def test_distributed_training_requires_enough_actor_devices(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    cfg = replace(cfg, training=replace(cfg.training, num_envs=3, actor_workers=3, actor_devices=("cpu", "cpu"), eval_interval_steps=1000))
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)
    with pytest.raises(ValueError, match="actor_workers=3 requires at least 3 actor_devices"):
        trainer.train()


def test_training_cli_device_list_parser_accepts_explicit_cuda_devices() -> None:
    assert _device_list("cuda:1,cuda:2,cuda:3") == ("cuda:1", "cuda:2", "cuda:3")


def test_config_loader_reads_explicit_training_devices(tmp_path: Path) -> None:
    data = json.loads(CONFIG.read_text())
    data["training"]["device"] = "cuda:0"
    data["training"]["actor_workers"] = 2
    data["training"]["actor_devices"] = ["cuda:1", "cuda:2"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    cfg = load_experiment_config(path)
    assert cfg.training.device == "cuda:0"
    assert cfg.training.actor_workers == 2
    assert cfg.training.actor_devices == ("cuda:1", "cuda:2")


def test_reward_search_dry_run_writes_candidates(tmp_path: Path) -> None:
    out = tmp_path / "search"
    code = reward_search_main([
        "--config", str(CONFIG),
        "--output-dir", str(out),
        "--dry-run",
        "--max-candidates", "2",
        "--shape-good-values", "0.004,0.006",
        "--shape-bad-values", "0.05",
    ])
    assert code == 0
    assert (out / "search_manifest.json").exists()
    results = (out / "results.csv").read_text()
    assert "candidate" in results
    assert "dry_run" in results


def test_export_metadata_records_update_count(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    result = Trainer(cfg, device="cpu", output_dir=tmp_path).train()
    import json
    metadata = json.loads((tmp_path / "exports" / "final_actor" / "metadata.json").read_text())
    assert metadata["updates"] == result["updates"]
    assert metadata["updates"] > 0


def test_evaluate_detailed_reports_physical_metrics(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)
    metrics = trainer.evaluate_detailed(episodes=2, max_steps=4, policy="no_control")
    assert "mean_return" in metrics
    assert "shape_error_mean_m" in metrics
    assert "shape_error_max_m" in metrics
    assert "ip_error_a" in metrics
    assert "current_over_limit_a" in metrics
    assert "boundary_found" in metrics
    assert np.isfinite(metrics["mean_return"])


def test_successive_halving_dry_run_writes_stage_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "staged_search"
    code = reward_search_main([
        "--config", str(CONFIG),
        "--output-dir", str(out),
        "--strategy", "successive_halving",
        "--dry-run",
        "--max-candidates", "2",
        "--stage-steps", "2,3,4",
        "--stage-keep", "2,1,1",
        "--stage-eval-episodes", "1,1,1",
        "--shape-good-values", "0.004,0.006",
        "--shape-bad-values", "0.05",
    ])
    assert code == 0
    manifest = (out / "search_manifest.json").read_text()
    assert '"strategy": "successive_halving"' in manifest
    assert '"baseline_policy": "no_control"' in manifest
    assert (out / "stage_00_baseline" / "results.csv").exists()
    assert (out / "baseline_results.csv").exists()


def test_reward_search_ranking_uses_physical_metrics_before_return() -> None:
    rows = [
        {
            "candidate": 0,
            "status": "ok",
            "eval.boundary_found": 1.0,
            "eval.current_over_limit_a": 0.0,
            "eval.shape_error_mean_m": 0.09,
            "eval.ip_error_a": 5000.0,
            "eval.improvement_over_no_control": 5.0,
            "eval.delta_action_quality": 0.9,
            "eval.action_quality": 0.9,
            "eval.mean_return": 100.0,
        },
        {
            "candidate": 1,
            "status": "ok",
            "eval.boundary_found": 1.0,
            "eval.current_over_limit_a": 0.0,
            "eval.shape_error_mean_m": 0.04,
            "eval.ip_error_a": 7000.0,
            "eval.improvement_over_no_control": 1.0,
            "eval.delta_action_quality": 0.8,
            "eval.action_quality": 0.8,
            "eval.mean_return": 50.0,
        },
        {
            "candidate": 2,
            "status": "ok",
            "eval.boundary_found": 0.5,
            "eval.current_over_limit_a": 0.0,
            "eval.shape_error_mean_m": 0.01,
            "eval.ip_error_a": 100.0,
            "eval.improvement_over_no_control": 100.0,
            "eval.delta_action_quality": 1.0,
            "eval.action_quality": 1.0,
            "eval.mean_return": 500.0,
        },
    ]
    ranked = _rank_rows(rows)
    assert ranked[0]["candidate"] == 1
    assert ranked[-1]["candidate"] == 2
    assert ranked[-1]["promotion_reason"] == "rejected_boundary_not_reliable"


def test_chunked_sampled_q_values_match_unbatched_reference() -> None:
    torch.manual_seed(123)
    obs_dim = 9
    action_dim = 3
    actor = FeedForwardGaussianActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=8)
    critic = RecurrentQCritic(obs_dim=obs_dim, action_dim=action_dim, lstm_hidden_dim=8, mlp_hidden_dim=8)
    target_actor = FeedForwardGaussianActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=8)
    target_critic = RecurrentQCritic(obs_dim=obs_dim, action_dim=action_dim, lstm_hidden_dim=8, mlp_hidden_dim=8)
    learner = MaximumAPosterioriPolicyOptimiser(
        actor=actor,
        critic=critic,
        target_actor=target_actor,
        target_critic=target_critic,
        config=LearnerConfig(batch_size=2, unroll_length=2, action_samples=4, actor_update_chunk_size=2),
        device="cpu",
    )
    obs = torch.randn((5, obs_dim), dtype=torch.float32)
    sampled = torch.randn((4, 5, action_dim), dtype=torch.float32)
    chunked = learner._sampled_q_values(obs, sampled)
    obs_rep = obs[None, :, :].expand(4, -1, -1).reshape(20, obs_dim)
    act_rep = sampled.reshape(20, action_dim)
    reference, _ = critic(obs_rep, act_rep)
    assert torch.allclose(chunked, reference.reshape(4, 5), atol=1.0e-6)


def test_successive_halving_parallel_cpu_workers_write_results(tmp_path: Path) -> None:
    out = tmp_path / "parallel_search"
    code = reward_search_main([
        "--config", str(CONFIG),
        "--output-dir", str(out),
        "--strategy", "successive_halving",
        "--stage-steps", "2,3,4",
        "--stage-keep", "1,1,1",
        "--stage-eval-episodes", "1,1,1",
        "--max-candidates", "2",
        "--parallel-candidates", "2",
        "--gpu-devices", "cpu,cpu",
        "--num-envs", "1",
        "--device", "cpu",
        "--sim-compute-backend", "cpu",
        "--batch-size", "2",
        "--unroll-length", "2",
        "--replay-capacity-episodes", "1",
        "--rollout-chunk-length", "2",
        "--updates-per-rollout-chunk", "1",
        "--hidden-dim", "16",
        "--critic-hidden-dim", "16",
        "--critic-mlp-hidden-dim", "16",
        "--action-samples", "4",
        "--actor-update-chunk-size", "2",
        "--eval-max-steps", "4",
        "--shape-good-values", "0.004,0.006",
        "--shape-bad-values", "0.05",
    ])
    assert code == 0
    assert (out / "stage_01_short" / "results.csv").exists()
    assert (out / "stage_01_short" / "candidate_0000" / "worker_result.json").exists()
    assert "worker_returncode" in (out / "stage_01_short" / "results.csv").read_text()
