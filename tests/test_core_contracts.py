from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.config.schema import LearnerConfig, RewardConfig
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.networks import FeedForwardGaussianActor, RecurrentQCritic
from tokamak_rl_v2.rewards import T15StaticBoundaryReward
from tokamak_rl_v2.rewards import transforms
from tokamak_rl_v2.training.mpo import MaximumAPosterioriPolicyOptimiser
from tokamak_rl_v2.training.replay import FIFOSequenceReplay, SequenceBatch
from tokamak_rl_v2.training.trainer import Trainer
from tokamak_rl_v2.training.reward_search import _promotion_reason, _rank_rows, main as reward_search_main
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


def test_tracking_reward_cannot_be_replaced_by_low_action() -> None:
    reward_fn = T15StaticBoundaryReward(RewardConfig(reward_scale=1.0), control_rate_hz=1000.0)
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    boundary = ref.clone()
    boundary[1, :, 0] = 0.20
    zero_action = torch.zeros((2, 9), dtype=torch.float32)
    active_action = torch.ones((2, 9), dtype=torch.float32)
    common = dict(
        ip=torch.tensor([200000.0, 200000.0]),
        ip_ref=torch.tensor([200000.0, 200000.0]),
        boundary_points=boundary,
        reference_points=ref,
        previous_action=zero_action,
        current_over_limit_a=torch.zeros((2,), dtype=torch.float32),
        derivative_usage=torch.zeros((2,), dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.zeros((2,), dtype=torch.bool),
    )
    zero = reward_fn(action=zero_action, **common)
    active = reward_fn(action=active_action, **common)
    assert zero.reward[0] > zero.reward[1]
    assert active.reward[0] > zero.reward[1]
    assert active.reward[0] < zero.reward[0]


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


def test_episode_replay_never_crosses_env_or_done_boundaries() -> None:
    replay = FIFOSequenceReplay(capacity_episodes=4, max_episode_steps=5, active_envs=2, obs_dim=2, action_dim=1, device="cpu")
    for t in range(4):
        obs = torch.tensor([[0.0, float(t)], [1.0, float(t)]])
        next_obs = obs + torch.tensor([[0.0, 0.1], [0.0, 0.1]])
        done = torch.tensor([t == 1, False])
        replay.add_batch(obs, torch.zeros((2, 1)), torch.zeros((2,)), torch.ones((2,)), next_obs, done)
    assert replay.ready(sequence_length=2, batch_size=8)
    for _ in range(20):
        batch = replay.sample(batch_size=8, sequence_length=2)
        assert torch.all(batch.obs[:, :, 0] == batch.obs[:, :1, 0])
        assert not torch.any(batch.done[:, :-1])


def test_environment_reset_indices_only_resets_done_slot() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=8))
    env = TokamakMagneticControlEnv(cfg, batch_size=2, device="cpu", seed=7)
    env.reset()
    ref_before = env.reference.ip.detach().clone()
    action = torch.full((2, env.action_dim), 0.25, dtype=torch.float32)
    out = env.step(action)
    assert torch.all(env.step_index == 1)
    obs = env.reset_indices(torch.tensor([True, False]))
    assert obs.shape == (2, env.obs_dim)
    assert int(env.step_index[0].item()) == 0
    assert int(env.step_index[1].item()) == 1
    assert torch.allclose(env.reference.ip[1], ref_before[1])
    assert torch.allclose(env.previous_action[1], action[1])
    assert torch.allclose(env.previous_action[0], torch.zeros_like(env.previous_action[0]))


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


def test_reward_search_weight_grid_generates_expected_candidates(tmp_path: Path) -> None:
    out = tmp_path / "weight_grid_search"
    code = reward_search_main([
        "--config", str(CONFIG),
        "--output-dir", str(out),
        "--dry-run",
        "--shape-good-values", "0.003",
        "--shape-bad-values", "0.04,0.08",
        "--ip-good-values", "500",
        "--ip-bad-values", "25000,40000",
        "--shape-weight-values", "2.0,4.0",
        "--ip-weight-values", "1.5,3.0",
    ])
    assert code == 0
    manifest = json.loads((out / "search_manifest.json").read_text())
    assert manifest["candidate_count"] == 16
    assert manifest["reward_values"]["shape_weight"] == [2.0, 4.0]
    assert manifest["reward_values"]["ip_weight"] == [1.5, 3.0]




def test_control_discovery_preset_is_broad_and_not_local_grid(tmp_path: Path) -> None:
    out = tmp_path / "control_discovery"
    code = reward_search_main([
        "--config", str(CONFIG),
        "--output-dir", str(out),
        "--strategy", "successive_halving",
        "--candidate-preset", "control_discovery",
        "--dry-run",
        "--stage-steps", "2,3,4",
        "--stage-keep", "4,2,1",
        "--stage-eval-episodes", "1,1,1",
    ])
    assert code == 0
    manifest = json.loads((out / "search_manifest.json").read_text())
    assert manifest["candidate_preset"] == "control_discovery"
    assert manifest["candidate_count"] >= 24
    rows = list(csv.DictReader((out / "results.csv").open()))
    assert len({row["reward.tracking_combiner"] for row in rows}) >= 3
    assert len({row["reward.action_penalty_weight"] for row in rows}) >= 3
    assert len({row["reward.ip_weight"] for row in rows}) >= 6
    assert len({row["reward.shape_bad_m"] for row in rows}) >= 6


def test_reward_search_rejects_low_control_activity() -> None:
    row = {
        "status": "ok",
        "eval.boundary_found": 1.0,
        "eval.current_over_limit_a": 0.0,
        "eval.shape_error_mean_m": 0.10,
        "eval.ip_error_a": 90000.0,
        "eval.shape_improvement_over_no_control_m": -0.001,
        "eval.ip_improvement_over_no_control_a": 40000.0,
        "eval.action_rms": 0.001,
        "search.min_action_rms": 0.005,
        "search.max_shape_error_m": 0.14,
        "search.max_ip_error_a": 105000.0,
    }
    assert _promotion_reason(row) == "rejected_low_control_activity"
    active = dict(row, **{"eval.action_rms": 0.02})
    assert _promotion_reason(active) == "eligible"

def test_export_metadata_records_update_count(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    result = Trainer(cfg, device="cpu", output_dir=tmp_path).train()
    metadata = json.loads((tmp_path / "exports" / "final_actor" / "metadata.json").read_text())
    assert metadata["updates"] == result["updates"]
    assert metadata["updates"] > 0


def test_training_checkpoint_resume_restores_replay_and_counters(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    cfg = _small_config(first_dir)
    cfg = replace(cfg, training=replace(cfg.training, steps=4, output_dir=first_dir, checkpoint_interval_steps=4, eval_interval_steps=1000))
    first = Trainer(cfg, device="cpu", output_dir=first_dir).train()
    checkpoint = first_dir / "checkpoints" / "final.pt"
    assert checkpoint.exists()
    second_dir = tmp_path / "second"
    resumed_cfg = replace(cfg, training=replace(cfg.training, steps=6, output_dir=second_dir, checkpoint_interval_steps=6, eval_interval_steps=1000))
    resumed = Trainer(resumed_cfg, device="cpu", output_dir=second_dir, resume_checkpoint=checkpoint).train()
    assert resumed["start_step"] == 4
    assert resumed["steps"] == 6
    state = torch.load(second_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
    assert state["training_state"]["step"] == 6
    assert state["replay_state"]["size"] > 0


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
    assert ranked[0]["candidate"] == 0
    assert ranked[-1]["candidate"] == 2
    assert ranked[-1]["promotion_reason"] == "rejected_boundary_not_reliable"


def test_reward_search_rejects_candidates_worse_than_no_control() -> None:
    rows = [
        {
            "candidate": 0,
            "status": "ok",
            "eval.boundary_found": 1.0,
            "eval.current_over_limit_a": 0.0,
            "eval.shape_error_mean_m": 0.18,
            "eval.ip_error_a": 100000.0,
            "eval.shape_improvement_over_no_control_m": -0.04,
            "eval.ip_improvement_over_no_control_a": 60000.0,
            "eval.improvement_over_no_control": 5.0,
            "search.require_no_control_improvement": 1,
            "eval.delta_action_quality": 1.0,
            "eval.action_quality": 1.0,
            "eval.mean_return": 60.0,
        },
        {
            "candidate": 1,
            "status": "ok",
            "eval.boundary_found": 1.0,
            "eval.current_over_limit_a": 0.0,
            "eval.shape_error_mean_m": 0.10,
            "eval.ip_error_a": 180000.0,
            "eval.shape_improvement_over_no_control_m": 0.03,
            "eval.ip_improvement_over_no_control_a": -20000.0,
            "eval.improvement_over_no_control": 5.0,
            "search.require_no_control_improvement": 1,
            "eval.delta_action_quality": 1.0,
            "eval.action_quality": 1.0,
            "eval.mean_return": 60.0,
        },
        {
            "candidate": 2,
            "status": "ok",
            "eval.boundary_found": 1.0,
            "eval.current_over_limit_a": 0.0,
            "eval.shape_error_mean_m": 0.10,
            "eval.ip_error_a": 100000.0,
            "eval.shape_improvement_over_no_control_m": 0.03,
            "eval.ip_improvement_over_no_control_a": 60000.0,
            "eval.improvement_over_no_control": 5.0,
            "search.require_no_control_improvement": 1,
            "eval.delta_action_quality": 1.0,
            "eval.action_quality": 1.0,
            "eval.mean_return": 60.0,
        },
    ]
    ranked = _rank_rows(rows)
    assert ranked[0]["candidate"] == 2
    reasons = {row["candidate"]: row["promotion_reason"] for row in ranked}
    assert reasons[0] == "rejected_shape_not_better_than_no_control"
    assert reasons[1] == "rejected_ip_not_better_than_no_control"


def test_reward_search_allows_bounded_shape_degradation_for_ip_gain() -> None:
    row = {
        "status": "ok",
        "eval.boundary_found": 1.0,
        "eval.current_over_limit_a": 0.0,
        "eval.shape_error_mean_m": 0.11,
        "eval.ip_error_a": 80000.0,
        "eval.shape_improvement_over_no_control_m": -0.014,
        "eval.ip_improvement_over_no_control_a": 20000.0,
        "search.max_shape_degradation_m": 0.015,
        "search.min_ip_improvement_a": 15000.0,
    }
    assert _promotion_reason(row) == "eligible"
    too_much_shape_loss = dict(row, **{"eval.shape_improvement_over_no_control_m": -0.020})
    assert _promotion_reason(too_much_shape_loss) == "rejected_shape_degradation_over_limit"
    not_enough_ip_gain = dict(row, **{"eval.ip_improvement_over_no_control_a": 1000.0})
    assert _promotion_reason(not_enough_ip_gain) == "rejected_ip_improvement_below_minimum"


def test_sequence_sampled_q_values_match_recurrent_reference() -> None:
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
        config=LearnerConfig(batch_size=2, unroll_length=3, action_samples=4, actor_update_chunk_size=2),
        device="cpu",
    )
    obs = torch.randn((2, 3, obs_dim), dtype=torch.float32)
    sampled = torch.randn((4, 2, 3, action_dim), dtype=torch.float32)
    mask = torch.ones((2, 3), dtype=torch.float32)
    chunked = learner._sampled_q_values(obs, sampled, mask=mask)
    reference = []
    for k in range(4):
        q, _ = critic(obs, sampled[k], mask=mask)
        reference.append(q)
    assert torch.allclose(chunked, torch.stack(reference, dim=0), atol=1.0e-6)


def test_actor_update_changes_policy_parameters_on_sequence_batch() -> None:
    torch.manual_seed(321)
    obs_dim = 7
    action_dim = 2
    actor = FeedForwardGaussianActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=8)
    critic = RecurrentQCritic(obs_dim=obs_dim, action_dim=action_dim, lstm_hidden_dim=8, mlp_hidden_dim=8)
    target_actor = FeedForwardGaussianActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=8)
    target_critic = RecurrentQCritic(obs_dim=obs_dim, action_dim=action_dim, lstm_hidden_dim=8, mlp_hidden_dim=8)
    learner = MaximumAPosterioriPolicyOptimiser(
        actor=actor,
        critic=critic,
        target_actor=target_actor,
        target_critic=target_critic,
        config=LearnerConfig(batch_size=4, unroll_length=3, action_samples=5),
        device="cpu",
    )
    batch = SequenceBatch(
        obs=torch.randn((4, 3, obs_dim)),
        action=torch.tanh(torch.randn((4, 3, action_dim))),
        reward=torch.randn((4, 3)),
        discount=torch.full((4, 3), 0.99),
        next_obs=torch.randn((4, 3, obs_dim)),
        done=torch.zeros((4, 3), dtype=torch.bool),
        mask=torch.ones((4, 3)),
    )
    metrics = learner.update(batch)
    assert metrics.actor_param_delta_norm > 0.0
    assert np.isfinite(metrics.sampled_q_spread)
    assert np.isfinite(metrics.policy_weight_entropy)



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
    stage1_text = (out / "stage_01_short" / "results.csv").read_text()
    assert "worker_returncode" in stage1_text
    assert "checkpoint_path" in stage1_text


def test_stage_only_search_resumes_from_previous_results(tmp_path: Path) -> None:
    out1 = tmp_path / "stage1"
    code = reward_search_main([
        "--config", str(CONFIG),
        "--output-dir", str(out1),
        "--strategy", "successive_halving",
        "--stage-only", "stage_01_short",
        "--stage-steps", "2,3,4",
        "--stage-keep", "2,1,1",
        "--stage-eval-episodes", "1,1,1",
        "--parallel-candidates", "1",
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
    out2 = tmp_path / "stage2"
    code = reward_search_main([
        "--config", str(CONFIG),
        "--output-dir", str(out2),
        "--strategy", "successive_halving",
        "--stage-only", "stage_02_medium",
        "--baseline-results", str(out1 / "baseline_results.csv"),
        "--previous-stage-results", str(out1 / "stage_01_short" / "results.csv"),
        "--stage-input-count", "1",
        "--stage-steps", "2,3,4",
        "--stage-keep", "2,1,1",
        "--stage-eval-episodes", "1,1,1",
        "--parallel-candidates", "1",
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
    assert (out2 / "baseline_results.csv").exists()
    stage2_results = (out2 / "stage_02_medium" / "results.csv")
    assert stage2_results.exists()
    text = stage2_results.read_text()
    assert "resumed_from_checkpoint" in text
    assert "start_step" in text
    assert (out2 / "stage_02_medium" / "promoted_candidates.csv").exists()
