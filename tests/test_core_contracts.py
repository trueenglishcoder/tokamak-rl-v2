from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.config.schema import IpReferenceConfig, LearnerConfig, RewardConfig
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.networks import FeedForwardGaussianActor, RecurrentQCritic
from tokamak_rl_v2.rewards import T15PhysicalReward
from tokamak_rl_v2.export.cli import main as export_cli_main
from tokamak_rl_v2.training.mpo import MaximumAPosterioriPolicyOptimiser
from scripts.calibrate_physical_reward import Candidate, _write_candidate_config
from tokamak_rl_v2.training.policy_pipeline import evaluate_policy_gates, run_reset_sanity
from tokamak_rl_v2.training.replay import FIFOSequenceReplay, SequenceBatch
from tokamak_rl_v2.training.trainer import Trainer
from tokamak_rl_v2.env.references import _segmented_ip, _segment_lengths, generate_reference_batch
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
    assert torch.mean(out.std).item() == pytest.approx(0.2, rel=1.0e-3)
    q, state = critic(obs, torch.zeros((3, 5)))
    assert q.shape == (3, 1)
    assert state.h.shape[-1] == 16


def test_critic_reads_normalized_env_actions_without_extra_squash() -> None:
    critic = RecurrentQCritic(obs_dim=3, action_dim=2, lstm_hidden_dim=8, mlp_hidden_dim=8)
    obs = torch.tensor([[0.25, -0.5, 0.75]], dtype=torch.float32)
    action = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}

    def capture_lstm_input(_module, args):
        captured["x"] = args[0].detach().clone()

    handle = critic.lstm.register_forward_pre_hook(capture_lstm_input)
    try:
        critic(obs, action)
    finally:
        handle.remove()

    lstm_input = captured["x"][0, 0]
    assert torch.allclose(lstm_input[:3], obs[0])
    assert torch.allclose(lstm_input[3:], action[0])
    assert not torch.allclose(lstm_input[3:], torch.tanh(action[0]))


def test_physical_reward_cannot_replace_tracking_with_low_action() -> None:
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0), control_rate_hz=1000.0)
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
        current_usage_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        current_margin_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        derivative_usage=torch.zeros((2,), dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.zeros((2,), dtype=torch.bool),
    )
    zero = reward_fn(action=zero_action, **common)
    active = reward_fn(action=active_action, **common)
    assert zero.reward[0] > zero.reward[1]
    assert active.reward[0] > zero.reward[1]
    assert active.reward[0] < zero.reward[0]


def test_reward_components_remain_finite_when_boundary_is_missing() -> None:
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0), control_rate_hz=1000.0)
    boundary = torch.full((1, 32, 2), float("nan"), dtype=torch.float32)
    ref = torch.zeros((1, 32, 2), dtype=torch.float32)
    rb = reward_fn(
        ip=torch.tensor([200000.0]),
        ip_ref=torch.tensor([200000.0]),
        boundary_points=boundary,
        reference_points=ref,
        action=torch.zeros((1, 9), dtype=torch.float32),
        previous_action=torch.zeros((1, 9), dtype=torch.float32),
        current_over_limit_a=torch.zeros((1,), dtype=torch.float32),
        current_usage_fraction=torch.full((1,), 0.5, dtype=torch.float32),
        current_margin_fraction=torch.full((1,), 0.5, dtype=torch.float32),
        derivative_usage=torch.zeros((1,), dtype=torch.float32),
        boundary_found=torch.zeros((1,), dtype=torch.bool),
        terminated=torch.ones((1,), dtype=torch.bool),
    )
    assert torch.isfinite(rb.reward).all()
    for value in rb.components.values():
        assert torch.isfinite(value).all()
    assert float(rb.components["shape_error_mean_m"].item()) == pytest.approx(0.1)


def test_physical_reward_makes_missing_boundary_expensive() -> None:
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0, shape_bad_m=0.025, boundary_missing_error_m=0.1), control_rate_hz=1000.0)
    rb = reward_fn(
        ip=torch.tensor([200000.0]),
        ip_ref=torch.tensor([200000.0]),
        boundary_points=torch.full((1, 32, 2), float("nan"), dtype=torch.float32),
        reference_points=torch.zeros((1, 32, 2), dtype=torch.float32),
        action=torch.zeros((1, 9), dtype=torch.float32),
        previous_action=torch.zeros((1, 9), dtype=torch.float32),
        current_over_limit_a=torch.zeros((1,), dtype=torch.float32),
        current_usage_fraction=torch.full((1,), 0.5, dtype=torch.float32),
        current_margin_fraction=torch.full((1,), 0.5, dtype=torch.float32),
        derivative_usage=torch.zeros((1,), dtype=torch.float32),
        boundary_found=torch.zeros((1,), dtype=torch.bool),
        terminated=torch.zeros((1,), dtype=torch.bool),
    )
    assert float(rb.components["shape_error_mean_m"].item()) == pytest.approx(0.1)
    assert float(rb.components["shape_loss"].item()) > 3.0
    assert float(rb.components["physical_cost"].item()) > 6.0


def test_physical_reward_improves_when_tracking_errors_improve() -> None:
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0, shape_bad_m=0.03, ip_bad_a=40000.0), control_rate_hz=1000.0)
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    boundary = ref.clone()
    boundary[1, :, 0] = 0.06
    action = torch.zeros((2, 9), dtype=torch.float32)
    rb = reward_fn(
        ip=torch.tensor([200000.0, 260000.0]),
        ip_ref=torch.tensor([200000.0, 200000.0]),
        boundary_points=boundary,
        reference_points=ref,
        action=action,
        previous_action=action,
        current_over_limit_a=torch.zeros((2,), dtype=torch.float32),
        current_usage_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        current_margin_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        derivative_usage=torch.zeros((2,), dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.zeros((2,), dtype=torch.bool),
    )
    assert rb.reward[0] > rb.reward[1]
    assert rb.components["shape_loss"][0] < rb.components["shape_loss"][1]
    assert rb.components["ip_loss"][0] < rb.components["ip_loss"][1]


def test_physical_reward_current_margin_warns_before_limit() -> None:
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0), control_rate_hz=1000.0)
    ref = torch.zeros((3, 32, 2), dtype=torch.float32)
    action = torch.zeros((3, 9), dtype=torch.float32)
    rb = reward_fn(
        ip=torch.full((3,), 200000.0),
        ip_ref=torch.full((3,), 200000.0),
        boundary_points=ref,
        reference_points=ref,
        action=action,
        previous_action=action,
        current_over_limit_a=torch.tensor([0.0, 0.0, 1000.0], dtype=torch.float32),
        current_usage_fraction=torch.tensor([0.50, 0.95, 1.20], dtype=torch.float32),
        current_margin_fraction=torch.tensor([0.50, 0.05, -0.20], dtype=torch.float32),
        derivative_usage=torch.zeros((3,), dtype=torch.float32),
        boundary_found=torch.ones((3,), dtype=torch.bool),
        terminated=torch.zeros((3,), dtype=torch.bool),
    )
    assert float(rb.components["current_margin_loss"][0].item()) == pytest.approx(0.0)
    assert float(rb.components["current_margin_loss"][1].item()) > 0.0
    assert rb.components["current_margin_loss"][2] > rb.components["current_margin_loss"][1]
    assert rb.reward[0] > rb.reward[1] > rb.reward[2]


def test_physical_reward_actuator_penalties_start_near_saturation() -> None:
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0), control_rate_hz=1000.0)
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    low_action = torch.full((1, 9), 0.5, dtype=torch.float32)
    high_action = torch.full((1, 9), 0.95, dtype=torch.float32)
    action = torch.cat([low_action, high_action], dim=0)
    previous = torch.zeros((2, 9), dtype=torch.float32)
    rb = reward_fn(
        ip=torch.full((2,), 200000.0),
        ip_ref=torch.full((2,), 200000.0),
        boundary_points=ref,
        reference_points=ref,
        action=action,
        previous_action=previous,
        current_over_limit_a=torch.zeros((2,), dtype=torch.float32),
        current_usage_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        current_margin_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        derivative_usage=torch.tensor([0.5, 0.95], dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.zeros((2,), dtype=torch.bool),
    )
    assert float(rb.components["derivative_loss"][0].item()) == pytest.approx(0.0)
    assert float(rb.components["action_saturation_loss"][0].item()) == pytest.approx(0.0)
    assert rb.components["derivative_loss"][1] > 0.0
    assert rb.components["action_saturation_loss"][1] > 0.0
    assert rb.components["delta_action_loss"][1] > rb.components["delta_action_loss"][0]


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


def test_action_scale_caps_physical_derivative_usage() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=4, action_scale=0.25))
    env = TokamakMagneticControlEnv(cfg, batch_size=2, device="cpu", seed=2)
    env.reset()
    result = env.step(torch.ones((2, env.action_dim), dtype=torch.float32))
    comps = result.info["reward_components"]
    assert float(np.nanmax(comps["derivative_usage"])) <= 0.25001
    assert np.allclose(np.asarray(env.normalization()["derivative_scale"]), env.raw_derivative_limits.detach().cpu().numpy() * 0.25)


def test_hold_reset_boundary_uses_observed_reset_boundary() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=4))
    cfg = replace(cfg, reference=replace(cfg.reference, boundary=replace(cfg.reference.boundary, kind="hold_reset_boundary")))
    env = TokamakMagneticControlEnv(cfg, batch_size=2, device="cpu", seed=3)
    obs = env.reset()
    schema = env.export_schema()
    err0, err1 = schema["feature_slices"]["boundary_radii_error"]
    found0, found1 = schema["feature_slices"]["boundary_found"]
    assert torch.max(torch.abs(obs[:, err0:err1])).item() == pytest.approx(0.0, abs=1.0e-6)
    assert torch.min(obs[:, found0:found1]).item() == pytest.approx(1.0)


def test_hold_reset_boundary_reset_is_deterministic_for_fixed_seed() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=4))
    cfg = replace(cfg, reference=replace(cfg.reference, boundary=replace(cfg.reference.boundary, kind="hold_reset_boundary")))
    env_a = TokamakMagneticControlEnv(cfg, batch_size=2, device="cpu", seed=5)
    env_b = TokamakMagneticControlEnv(cfg, batch_size=2, device="cpu", seed=5)
    obs_a = env_a.reset()
    obs_b = env_b.reset()
    assert torch.allclose(obs_a, obs_b)
    assert env_a.reference is not None and env_b.reference is not None
    assert torch.allclose(env_a.reference.radii, env_b.reference.radii)


def test_policy_pipeline_reset_sanity_uses_hold_boundary() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=4))
    cfg = replace(cfg, reference=replace(cfg.reference, boundary=replace(cfg.reference.boundary, kind="hold_reset_boundary")))
    report = run_reset_sanity(cfg, device="cpu", num_envs=2)
    assert report["max_abs_boundary_radii_error_m"] == pytest.approx(0.0, abs=1.0e-6)
    assert report["boundary_found_mean"] == pytest.approx(1.0)


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


def test_environment_step_uses_post_step_reference_index() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=8))
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=11)
    obs0 = env.reset()
    ip_ref_slice = env.export_schema()["feature_slices"]["ip_ref"]
    ref0 = float(obs0[0, ip_ref_slice[0]].item())
    result = env.step(torch.zeros((1, env.action_dim)))
    ref1_obs = float(result.obs[0, ip_ref_slice[0]].item())
    ref1_expected = float(env.reference.ip[0, 1].item() / 5.0e5)
    assert int(env.step_index[0].item()) == 1
    assert ref1_obs == pytest.approx(ref1_expected, abs=1.0e-7)
    if float(env.reference.ip[0, 0].item()) != float(env.reference.ip[0, 1].item()):
        assert ref1_obs != pytest.approx(ref0, abs=1.0e-9)


def test_ip_reference_segment_count_controls_generation() -> None:
    cfg = load_experiment_config(CONFIG)
    lengths = _segment_lengths(cfg.reference.ip, int(cfg.sim.max_episode_steps), np.random.default_rng(123))
    assert int(np.sum(lengths)) == int(cfg.sim.max_episode_steps)
    assert int(cfg.reference.ip.segment_count_min) <= len(lengths) <= int(cfg.reference.ip.segment_count_max)
    assert int(np.min(lengths)) >= int(cfg.reference.ip.segment_min_steps)
    assert int(np.max(lengths)) <= int(cfg.reference.ip.segment_max_steps)
    batch = generate_reference_batch(
        config=cfg.reference,
        initial_ip=np.asarray([250000.0, 300000.0]),
        initial_parameters=np.asarray([[1.4, 0.0, 0.55, 1.2, 0.2], [1.42, -0.01, 0.58, 1.25, 0.18]]),
        steps=int(cfg.sim.max_episode_steps),
        device="cpu",
        seed=456,
    )
    assert batch.ip.shape == (2, int(cfg.sim.max_episode_steps) + 1)
    assert torch.allclose(batch.ip[:, 0], torch.tensor([250000.0, 300000.0], dtype=torch.float64))


def test_ip_reference_inserts_hold_between_opposite_ramps() -> None:
    cfg = IpReferenceConfig(
        min=100000.0,
        max=160000.0,
        rate_limit=2.0e6,
        segment_min_steps=5,
        segment_max_steps=8,
        segment_count_min=20,
        segment_count_max=20,
        hold_probability=0.0,
    )
    saw_hold = False
    for seed in range(40):
        values = _segmented_ip(cfg, 125000.0, 120, np.random.default_rng(seed), dt=0.001)
        assert np.all(values > 0.0)
        signs = np.sign(np.diff(values))
        signs[np.abs(np.diff(values)) < 1.0e-7] = 0.0
        runs = [int(signs[0])] if signs.size else []
        for sign in signs[1:]:
            sign_i = int(sign)
            if sign_i != runs[-1]:
                runs.append(sign_i)
        saw_hold = saw_hold or 0 in runs
        for left, right in zip(runs, runs[1:], strict=False):
            assert not (left != 0 and right != 0 and left != right)
    assert saw_hold


def test_config_rejects_ip_reference_ranges_that_cross_zero(tmp_path: Path) -> None:
    data = json.loads(CONFIG.read_text())
    data["reference"]["ip"]["min"] = -100000.0
    data["reference"]["ip"]["max"] = 100000.0
    bad_reference = tmp_path / "bad_reference_ip.json"
    bad_reference.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="reference.ip range must stay strictly on one side of zero"):
        load_experiment_config(bad_reference)

    data = json.loads(CONFIG.read_text())
    data["sim"]["initial_ranges"]["ip"]["min"] = -1000.0
    data["sim"]["initial_ranges"]["ip"]["max"] = 1000.0
    bad_initial = tmp_path / "bad_initial_ip.json"
    bad_initial.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="sim.initial_ranges.ip must stay strictly on one side of zero"):
        load_experiment_config(bad_initial)

    data = json.loads(CONFIG.read_text())
    data["sim"]["initial_ranges"]["ip"]["min"] = -130000.0
    data["sim"]["initial_ranges"]["ip"]["max"] = -120000.0
    bad_sign = tmp_path / "bad_initial_ip_sign.json"
    bad_sign.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="sim.initial_ranges.ip must have the same sign as reference.ip"):
        load_experiment_config(bad_sign)


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


def test_config_loader_rejects_invalid_values(tmp_path: Path) -> None:
    data = json.loads(CONFIG.read_text())
    data["reward"]["unused_reward_key"] = "not_real"
    bad_reward = tmp_path / "bad_reward.json"
    bad_reward.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported keys"):
        load_experiment_config(bad_reward)

    data = json.loads(CONFIG.read_text())
    data["learner"]["action_samples"] = 1
    bad_learner = tmp_path / "bad_learner.json"
    bad_learner.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="action_samples"):
        load_experiment_config(bad_learner)

    data = json.loads(CONFIG.read_text())
    data["reward"]["mode"] = "not_real"
    bad_mode = tmp_path / "bad_reward_mode.json"
    bad_mode.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="mode"):
        load_experiment_config(bad_mode)

    data = json.loads(CONFIG.read_text())
    data["reward"]["current_margin_start_fraction"] = 1.0
    bad_current_margin = tmp_path / "bad_current_margin.json"
    bad_current_margin.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="current_margin_start_fraction"):
        load_experiment_config(bad_current_margin)

    data = json.loads(CONFIG.read_text())
    data["reward"]["delta_action_bad"] = data["reward"]["delta_action_penalty_start"]
    bad_delta = tmp_path / "bad_delta.json"
    bad_delta.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="delta_action_bad"):
        load_experiment_config(bad_delta)

    data = json.loads(CONFIG.read_text())
    data["sim"]["action_scale"] = 0.0
    bad_action_scale = tmp_path / "bad_action_scale.json"
    bad_action_scale.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="action_scale"):
        load_experiment_config(bad_action_scale)

    data = json.loads(CONFIG.read_text())
    data["network"]["actor_initial_std"] = 1.0e-5
    bad_actor_std = tmp_path / "bad_actor_std.json"
    bad_actor_std.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="actor_initial_std"):
        load_experiment_config(bad_actor_std)

    data = json.loads(CONFIG.read_text())
    data["reward"]["boundary_missing_error_m"] = -1.0
    bad_missing_boundary = tmp_path / "bad_missing_boundary.json"
    bad_missing_boundary.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="boundary_missing_error"):
        load_experiment_config(bad_missing_boundary)


def test_policy_pipeline_gates_require_learning_signals() -> None:
    actor_eval = {
        "boundary_found": 1.0,
        "current_over_limit_a": 0.0,
        "shape_error_mean_m": 0.02,
        "ip_error_a": 70000.0,
        "action_rms": 0.02,
    }
    no_control = {"ip_error_a": 100000.0}
    tail_losses = {"tail100.policy_weight_max": 0.06, "tail100.sampled_q_spread": 1.0e-4}
    gates = evaluate_policy_gates(
        actor_eval=actor_eval,
        no_control=no_control,
        tail_losses=tail_losses,
        action_samples=20,
        min_boundary_found=0.999,
        max_current_over_limit_a=0.0,
        max_shape_error_m=0.03,
        min_ip_improvement_frac=0.25,
        min_ip_improvement_a=20000.0,
        min_action_rms=0.005,
        max_action_rms=0.5,
        min_policy_weight_extra=1.0e-4,
        min_sampled_q_spread=1.0e-8,
        require_controller_rollout=True,
        controller_rollout={"status": "ok"},
    )
    assert gates["passed"] is True

    stalled = evaluate_policy_gates(
        actor_eval=dict(actor_eval, action_rms=0.0),
        no_control=no_control,
        tail_losses={"tail100.policy_weight_max": 0.05, "tail100.sampled_q_spread": 0.0},
        action_samples=20,
        min_boundary_found=0.999,
        max_current_over_limit_a=0.0,
        max_shape_error_m=0.03,
        min_ip_improvement_frac=0.25,
        min_ip_improvement_a=20000.0,
        min_action_rms=0.005,
        max_action_rms=0.5,
        min_policy_weight_extra=1.0e-4,
        min_sampled_q_spread=1.0e-8,
        require_controller_rollout=True,
        controller_rollout={"status": "ok"},
    )
    reasons = {check["name"]: check["passed"] for check in stalled["checks"]}
    assert stalled["passed"] is False
    assert reasons["action_rms_min"] is False
    assert reasons["mpo_policy_weights_nonuniform"] is False
    assert reasons["mpo_sampled_q_spread"] is False

    current_spike = evaluate_policy_gates(
        actor_eval=dict(actor_eval, current_over_limit_a=0.0, current_over_limit_a_max=1.0, current_over_limit_fraction=0.01),
        no_control=no_control,
        tail_losses=tail_losses,
        action_samples=20,
        min_boundary_found=0.999,
        max_current_over_limit_a=0.0,
        max_shape_error_m=0.03,
        min_ip_improvement_frac=0.25,
        min_ip_improvement_a=20000.0,
        min_action_rms=0.005,
        max_action_rms=0.5,
        min_policy_weight_extra=1.0e-4,
        min_sampled_q_spread=1.0e-8,
        require_controller_rollout=True,
        controller_rollout={"status": "ok"},
    )
    spike_reasons = {check["name"]: check["passed"] for check in current_spike["checks"]}
    assert spike_reasons["current_limit"] is False


def test_experiment_configs_use_neutral_output_names() -> None:
    for path in (ROOT / "configs/experiments").glob("*.yaml"):
        data = json.loads(path.read_text())
        output_dir = str(data.get("training", {}).get("output_dir", ""))
        assert "candidate" not in output_dir


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


def test_training_checkpoint_resume_rejects_old_critic_action_input_kind(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    cfg = _small_config(first_dir)
    trainer = Trainer(cfg, device="cpu", output_dir=first_dir)
    trainer.env.reset()
    checkpoint = trainer._save_checkpoint("old_critic_semantics.pt", step=0, updates=0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state.pop("critic_action_input_kind", None)
    torch.save(state, checkpoint)

    resumed = Trainer(cfg, device="cpu", output_dir=tmp_path / "resume", resume_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="critic action input"):
        resumed._load_checkpoint(checkpoint)


def test_checkpoint_save_does_not_hide_single_env_state_errors(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)
    trainer.env.reset()

    def broken_state_dict():
        raise RuntimeError("broken environment serialization")

    trainer.env.state_dict = broken_state_dict  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="broken environment serialization"):
        trainer._save_checkpoint("bad.pt", step=0, updates=0)


def test_export_cli_rejects_malformed_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "malformed.pt"
    torch.save(["not", "a", "checkpoint"], checkpoint)
    with pytest.raises(ValueError, match="training-state dictionary"):
        export_cli_main(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "export")])


def test_export_cli_rejects_obsolete_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "obsolete.pt"
    torch.save(
        {
            "checkpoint_version": 1,
            "schema": {"observation_kind": "diagnostic_v0", "obs_dim": 4, "action_dim": 2},
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="obsolete"):
        export_cli_main(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "export")])


def test_export_cli_writes_policy_bundle_from_valid_checkpoint(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    cfg = _small_config(train_dir)
    Trainer(cfg, device="cpu", output_dir=train_dir).train()
    export_dir = tmp_path / "manual_export"
    assert export_cli_main(["--checkpoint", str(train_dir / "checkpoints" / "final.pt"), "--out", str(export_dir)]) == 0
    assert (export_dir / "actor.pt").exists()
    assert (export_dir / "policy_weights.npz").exists()
    schema = json.loads((export_dir / "controller_schema.json").read_text())
    assert schema["observation_kind"] == "joint_state_v1"


def test_distributed_resume_fails_clearly_because_worker_envs_are_not_checkpointed(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    cfg = _small_config(first_dir)
    cfg = replace(cfg, training=replace(cfg.training, steps=4, output_dir=first_dir, checkpoint_interval_steps=4, eval_interval_steps=1000))
    Trainer(cfg, device="cpu", output_dir=first_dir).train()
    checkpoint = first_dir / "checkpoints" / "final.pt"
    dist_dir = tmp_path / "dist"
    dist_cfg = replace(
        cfg,
        training=replace(
            cfg.training,
            steps=8,
            output_dir=dist_dir,
            actor_workers=2,
            actor_devices=("cpu", "cpu"),
            eval_interval_steps=1000,
        ),
    )
    trainer = Trainer(dist_cfg, device="cpu", output_dir=dist_dir, resume_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="not exactly resumable"):
        trainer.train()


def test_evaluate_detailed_reports_physical_metrics(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)
    metrics = trainer.evaluate_detailed(episodes=2, max_steps=4, policy="no_control")
    assert "mean_return" in metrics
    assert "shape_error_mean_m" in metrics
    assert "shape_error_max_m" in metrics
    assert "ip_error_a" in metrics
    assert "current_over_limit_a" in metrics
    assert "current_over_limit_a_max" in metrics
    assert "current_over_limit_fraction" in metrics
    assert "current_usage_fraction" in metrics
    assert "current_usage_fraction_max" in metrics
    assert "current_margin_fraction" in metrics
    assert "current_margin_fraction_min" in metrics
    assert "derivative_usage" in metrics
    assert "derivative_usage_max" in metrics
    assert "max_abs_action" in metrics
    assert "max_abs_action_max" in metrics
    assert "physical_cost" in metrics
    assert "boundary_found" in metrics
    assert "boundary_found_min" in metrics
    assert np.isfinite(metrics["mean_return"])


def test_mpo_e_step_keeps_uniform_weights_when_sampled_q_is_uniform() -> None:
    torch.manual_seed(777)
    obs_dim = 5
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
        config=LearnerConfig(batch_size=2, unroll_length=2, action_samples=6, mpo_epsilon=0.1),
        device="cpu",
    )
    batch = SequenceBatch(
        obs=torch.randn((2, 2, obs_dim)),
        action=torch.zeros((2, 2, action_dim)),
        reward=torch.zeros((2, 2)),
        discount=torch.full((2, 2), 0.99),
        next_obs=torch.randn((2, 2, obs_dim)),
        done=torch.zeros((2, 2), dtype=torch.bool),
        mask=torch.ones((2, 2)),
    )

    def uniform_q(obs, sampled_actions, *, mask=None):
        return torch.zeros((sampled_actions.shape[0], obs.shape[0], obs.shape[1]), dtype=obs.dtype, device=obs.device)

    learner._sampled_q_values = uniform_q  # type: ignore[method-assign]
    metrics = learner._actor_update(batch)
    assert metrics[8] == pytest.approx(float(np.log(6.0)), rel=1.0e-5)
    assert metrics[9] == pytest.approx(1.0 / 6.0, rel=1.0e-5)
    assert np.isfinite(metrics[10])


def test_mpo_e_step_solves_batch_dual_for_distinct_sampled_q() -> None:
    torch.manual_seed(778)
    obs_dim = 5
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
        config=LearnerConfig(batch_size=2, unroll_length=2, action_samples=6, mpo_epsilon=0.1),
        device="cpu",
    )
    batch = SequenceBatch(
        obs=torch.randn((2, 2, obs_dim)),
        action=torch.zeros((2, 2, action_dim)),
        reward=torch.zeros((2, 2)),
        discount=torch.full((2, 2), 0.99),
        next_obs=torch.randn((2, 2, obs_dim)),
        done=torch.zeros((2, 2), dtype=torch.bool),
        mask=torch.ones((2, 2)),
    )

    def ranked_q(obs, sampled_actions, *, mask=None):
        ranks = torch.linspace(0.0, 0.05, sampled_actions.shape[0], dtype=obs.dtype, device=obs.device)
        return ranks[:, None, None].expand(-1, obs.shape[0], obs.shape[1])

    learner._sampled_q_values = ranked_q  # type: ignore[method-assign]
    metrics = learner._actor_update(batch)
    entropy = metrics[8]
    kl_to_uniform = float(np.log(6.0)) - entropy
    assert kl_to_uniform == pytest.approx(0.1, rel=5.0e-2, abs=5.0e-3)
    assert metrics[9] > 1.0 / 6.0
    assert metrics[10] < 1.0
    assert metrics[11] > 0.0
    assert metrics[12] > 0.0



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


def test_reward_calibration_candidate_config_keeps_sim_paths_valid(tmp_path: Path) -> None:
    candidate_path = tmp_path / "generated" / "candidate.json"
    candidate_path.parent.mkdir(parents=True)
    _write_candidate_config(CONFIG, candidate_path, Candidate("test", {"shape_weight": 3.0}))
    data = json.loads(candidate_path.read_text(encoding="utf-8"))
    sim_config = Path(data["sim"]["config_path"])
    assert sim_config.is_absolute()
    assert sim_config.exists()
    assert data["reward"]["shape_weight"] == pytest.approx(3.0)
