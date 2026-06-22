from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.config.schema import BoundaryReferenceConfig, DeltaDerivativeLimits, IpReferenceConfig, LearnerConfig, ReferenceConfig, RewardConfig
from tokamak_rl_v2.env import BatchStep, TokamakMagneticControlEnv
from tokamak_rl_v2.networks import FeedForwardGaussianActor, RecurrentQCritic
from tokamak_rl_v2.rewards import T15PhysicalReward, T15TCVDerivativeReward, T15TCVQualityReward
from tokamak_rl_v2.export.cli import main as export_cli_main
from tokamak_rl_v2.training.mpo import MaximumAPosterioriPolicyOptimiser
from tokamak_rl_v2.training.policy_pipeline import _ArrayReferenceScenario, _write_baseline_report, evaluate_policy_gates, run_reset_sanity
from tokamak_rl_v2.training.replay import FIFOSequenceReplay, SequenceBatch
from tokamak_rl_v2.training.trainer import Trainer, _append_csv_row
from tokamak_rl_v2.env.references import T15ReplayBoundaryLibrary, _segmented_ip, _segment_lengths, generate_reference_batch
from tokamak_rl_v2.training.cli import _device_list


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/t15_static_boundary.yaml"
PRODUCTION_CONFIG = ROOT / "configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml"
FIXED_HORIZON_HOLD_CONFIG = ROOT / "configs/experiments/t15_csv_hold_ip_fixed_horizon.yaml"
FIXED_HORIZON_EASY_SEGMENTED_CONFIG = ROOT / "configs/experiments/t15_csv_easy_segmented_fixed_horizon.yaml"


def _small_config(tmp_path: Path):
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=12))
    cfg = replace(cfg, network=replace(cfg.network, hidden_dim=16, critic_hidden_dim=16, critic_mlp_hidden_dim=16))
    cfg = replace(cfg, learner=replace(cfg.learner, batch_size=4, unroll_length=2, rollout_chunk_length=2, updates_per_rollout_chunk=1, action_samples=4))
    cfg = replace(cfg, training=replace(cfg.training, output_dir=tmp_path, steps=8, num_envs=2, checkpoint_interval_steps=8, eval_interval_steps=8, eval_episodes=2, eval_max_steps=4))
    return cfg


def _sequence_batch(
    *,
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    discount: torch.Tensor,
    next_obs: torch.Tensor,
    done: torch.Tensor,
    mask: torch.Tensor,
    critic_obs: torch.Tensor | None = None,
    next_critic_obs: torch.Tensor | None = None,
) -> SequenceBatch:
    return SequenceBatch(
        obs=obs,
        critic_obs=obs if critic_obs is None else critic_obs,
        action=action,
        reward=reward,
        discount=discount,
        next_obs=next_obs,
        next_critic_obs=next_obs if next_critic_obs is None else next_critic_obs,
        done=done,
        mask=mask,
    )


def test_network_shapes() -> None:
    actor = FeedForwardGaussianActor(obs_dim=17, action_dim=5, hidden_dim=16)
    critic = RecurrentQCritic(obs_dim=17, action_dim=5, lstm_hidden_dim=16, mlp_hidden_dim=16)
    obs = torch.zeros((3, 17))
    out = actor(obs)
    assert out.mean.shape == (3, 5)
    assert out.std.shape == (3, 5)
    det = actor.deterministic(obs)
    assert det.shape == (3, 5)
    assert torch.allclose(det, torch.tanh(out.mean))
    assert torch.max(torch.abs(det)).item() <= 1.0
    assert torch.mean(out.std).item() == pytest.approx(0.2, rel=1.0e-3)
    q, state = critic(obs, torch.zeros((3, 5)))
    assert q.shape == (3, 1)
    assert state.h.shape[-1] == 16


def test_tcv_derivative_current_termination_uses_configured_fraction_and_grace() -> None:
    env = object.__new__(TokamakMagneticControlEnv)
    env.config = SimpleNamespace(
        reward=SimpleNamespace(kind="tcv_derivative"),
        sim=SimpleNamespace(
            terminate_on_current_limit=True,
            current_termination_over_limit_a=0.0,
            current_termination_grace_steps=2,
            current_hard_termination_fraction=1.20,
        ),
    )
    env.current_over_limit_steps = torch.zeros(3, dtype=torch.int64)
    current_over_limit = torch.tensor([1000.0, 1000.0, 1000.0])
    usage = torch.tensor([1.01, 1.19, 1.21])

    terminated, hard, grace = TokamakMagneticControlEnv._current_termination(
        env,
        current_over_limit=current_over_limit,
        current_usage_fraction=usage,
    )
    assert terminated.tolist() == [False, False, False]
    assert hard.tolist() == [False, False, False]
    assert grace.tolist() == [False, False, False]
    assert env.current_over_limit_steps.tolist() == [0, 0, 1]

    terminated, hard, grace = TokamakMagneticControlEnv._current_termination(
        env,
        current_over_limit=current_over_limit,
        current_usage_fraction=usage,
    )
    assert terminated.tolist() == [False, False, True]
    assert hard.tolist() == [False, False, False]
    assert grace.tolist() == [False, False, True]


def test_tcv_derivative_current_termination_grace_one_is_immediate() -> None:
    env = object.__new__(TokamakMagneticControlEnv)
    env.config = SimpleNamespace(
        reward=SimpleNamespace(kind="tcv_derivative"),
        sim=SimpleNamespace(
            terminate_on_current_limit=True,
            current_termination_over_limit_a=0.0,
            current_termination_grace_steps=1,
            current_hard_termination_fraction=1.20,
        ),
    )
    env.current_over_limit_steps = torch.zeros(2, dtype=torch.int64)

    terminated, _hard, grace = TokamakMagneticControlEnv._current_termination(
        env,
        current_over_limit=torch.tensor([0.0, 0.0]),
        current_usage_fraction=torch.tensor([1.20, 1.2001]),
    )
    assert terminated.tolist() == [False, True]
    assert grace.tolist() == [False, True]


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


def test_physical_reward_tracks_errors_and_action_cost() -> None:
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
    assert active.reward[0] < zero.reward[0]
    assert float(active.components["action_rms"][0].item()) > float(zero.components["action_rms"][0].item())
    assert float(zero.components["physical_cost"][0].item()) == pytest.approx(0.0)


def test_tcv_quality_reward_improves_with_better_physical_errors() -> None:
    reward_fn = T15TCVQualityReward(
        RewardConfig(
            kind="tcv_quality",
            reward_scale=1.0,
            smoothmax_alpha=5.0,
            ip_scale_a=15000.0,
            boundary_missing_error_m=1.0,
            boundary_missing_weight=20.0,
            shape_mean_weight=1.0,
            shape_max_weight=0.25,
            ip_weight=1.0,
            current_weight=1.0,
            derivative_weight=0.5,
            actuator_saturation_weight=1.0,
            action_weight=0.0,
            delta_action_weight=0.0,
        ),
        control_rate_hz=1000.0,
    )
    ref = torch.zeros((3, 32, 2), dtype=torch.float32)
    boundary = ref.clone()
    boundary[1, :, 0] = 0.20
    action = torch.zeros((3, 9), dtype=torch.float32)
    rb = reward_fn(
        ip=torch.tensor([200000.0, 250000.0, 200000.0]),
        ip_ref=torch.tensor([200000.0, 200000.0, 200000.0]),
        boundary_points=boundary,
        reference_points=ref,
        action=action,
        previous_action=action,
        requested_action=torch.stack([action[0], action[1], torch.ones((9,), dtype=torch.float32)]),
        current_over_limit_a=torch.zeros((3,), dtype=torch.float32),
        current_usage_fraction=torch.tensor([0.5, 0.95, 0.5], dtype=torch.float32),
        current_margin_fraction=torch.tensor([0.5, 0.05, 0.5], dtype=torch.float32),
        derivative_usage=torch.tensor([0.1, 0.95, 0.1], dtype=torch.float32),
        boundary_found=torch.tensor([True, True, False]),
        terminated=torch.tensor([False, False, True]),
    )
    assert rb.reward[0] > rb.reward[1]
    assert rb.reward[1] > rb.reward[2]
    assert rb.components["physical_cost"][2] > rb.components["physical_cost"][1]
    assert rb.components["actuator_saturation_loss"][2] > 0.0


def test_tcv_derivative_reward_uses_terminal_reward_replacement() -> None:
    reward_fn = T15TCVDerivativeReward(
        RewardConfig(
            kind="tcv_derivative",
            reward_scale=0.01,
            smoothmax_alpha=-5.0,
            terminal_reward=-5.0,
            terminal_remaining_cost=0.0,
            shape_mean_weight=1.0,
            shape_max_weight=0.25,
            ip_weight=1.0,
            current_weight=1.0,
            derivative_weight=0.5,
            actuator_saturation_weight=0.5,
            boundary_missing_weight=20.0,
            current_usage_weight=0.0,
            derivative_usage_weight=0.0,
            action_weight=0.0,
            delta_action_weight=0.0,
        ),
        control_rate_hz=1000.0,
    )
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    action = torch.zeros((2, 9), dtype=torch.float32)
    rb = reward_fn(
        ip=torch.tensor([200000.0, 200000.0]),
        ip_ref=torch.tensor([200000.0, 200000.0]),
        boundary_points=ref,
        reference_points=ref,
        action=action,
        previous_action=action,
        requested_action=action,
        current_over_limit_a=torch.zeros((2,), dtype=torch.float32),
        current_usage_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        current_margin_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        derivative_usage=torch.zeros((2,), dtype=torch.float32),
        boundary_found=torch.tensor([True, True]),
        terminated=torch.tensor([False, True]),
    )
    assert float(rb.reward[0]) > 0.0
    assert float(rb.reward[1]) == pytest.approx(-0.05)
    assert float(rb.components["terminal_reward_raw"][1]) == pytest.approx(-5.0)
    assert float(rb.components["terminal_reward_scaled"][1]) == pytest.approx(-0.05)
    assert float(rb.components["terminal_total_penalty"][1]) == pytest.approx(-0.05)


def test_tcv_derivative_current_and_saturation_rewards_are_not_inverted() -> None:
    reward_fn = T15TCVDerivativeReward(
        RewardConfig(
            kind="tcv_derivative",
            reward_scale=1.0,
            smoothmax_alpha=-5.0,
            shape_mean_weight=0.0,
            shape_max_weight=0.0,
            ip_weight=0.0,
            current_weight=1.0,
            derivative_weight=0.0,
            actuator_saturation_weight=1.0,
            boundary_missing_weight=0.0,
            current_soft_fraction=0.9,
            current_bad_fraction=1.0,
        ),
        control_rate_hz=1000.0,
    )
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    action = torch.zeros((2, 9), dtype=torch.float32)
    rb = reward_fn(
        ip=torch.full((2,), 200000.0),
        ip_ref=torch.full((2,), 200000.0),
        boundary_points=ref,
        reference_points=ref,
        action=action,
        previous_action=action,
        requested_action=torch.stack([torch.zeros((9,), dtype=torch.float32), torch.ones((9,), dtype=torch.float32)]),
        applied_delta_action=torch.zeros((2, 9), dtype=torch.float32),
        current_over_limit_a=torch.tensor([0.0, 1000.0], dtype=torch.float32),
        current_usage_fraction=torch.tensor([0.5, 1.2], dtype=torch.float32),
        current_margin_fraction=torch.tensor([0.5, -0.2], dtype=torch.float32),
        derivative_usage=torch.zeros((2,), dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.zeros((2,), dtype=torch.bool),
    )
    assert float(rb.components["current_loss"][0]) == pytest.approx(0.0)
    assert float(rb.components["current_loss"][1]) == pytest.approx(1.0)
    assert float(rb.components["tcv_saturation_component_loss"][0]) == pytest.approx(0.0)
    assert float(rb.components["tcv_saturation_component_loss"][1]) == pytest.approx(1.0)
    assert float(rb.components["tcv_quality"][0]) > float(rb.components["tcv_quality"][1])


def test_tcv_derivative_actuator_effort_uses_realized_delta_jdot() -> None:
    reward_fn = T15TCVDerivativeReward(
        RewardConfig(
            kind="tcv_derivative",
            reward_scale=1.0,
            smoothmax_alpha=-5.0,
            derivative_weight=1.0,
            shape_mean_weight=0.0,
            shape_max_weight=0.0,
            ip_weight=0.0,
            current_weight=0.0,
            actuator_saturation_weight=0.0,
            boundary_missing_weight=0.0,
            derivative_bad_fraction=1.0,
        ),
        control_rate_hz=1000.0,
    )
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    accumulated = torch.full((2, 9), 0.95, dtype=torch.float32)
    previous = torch.zeros_like(accumulated)
    rb = reward_fn(
        ip=torch.full((2,), 200000.0),
        ip_ref=torch.full((2,), 200000.0),
        boundary_points=ref,
        reference_points=ref,
        action=accumulated,
        previous_action=previous,
        requested_action=torch.zeros_like(accumulated),
        applied_delta_action=torch.stack([torch.zeros((9,), dtype=torch.float32), torch.full((9,), 0.95, dtype=torch.float32)]),
        current_over_limit_a=torch.zeros((2,), dtype=torch.float32),
        current_usage_fraction=torch.zeros((2,), dtype=torch.float32),
        current_margin_fraction=torch.ones((2,), dtype=torch.float32),
        derivative_usage=torch.full((2,), 0.95, dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.zeros((2,), dtype=torch.bool),
    )
    assert float(rb.components["derivative_loss"][0]) < 0.05
    assert float(rb.components["derivative_loss"][1]) > 0.5


def test_physical_reward_penalizes_rejected_actuator_command() -> None:
    reward_fn = T15PhysicalReward(
        RewardConfig(reward_scale=1.0, actuator_saturation_weight=4.0, action_weight=0.0, delta_action_weight=0.0),
        control_rate_hz=1000.0,
    )
    ref = torch.zeros((1, 32, 2), dtype=torch.float32)
    applied = torch.zeros((1, 9), dtype=torch.float32)
    requested = torch.ones((1, 9), dtype=torch.float32)
    common = dict(
        ip=torch.tensor([200000.0]),
        ip_ref=torch.tensor([200000.0]),
        boundary_points=ref,
        reference_points=ref,
        previous_action=applied,
        current_over_limit_a=torch.zeros((1,), dtype=torch.float32),
        current_usage_fraction=torch.full((1,), 0.5, dtype=torch.float32),
        current_margin_fraction=torch.full((1,), 0.5, dtype=torch.float32),
        derivative_usage=torch.zeros((1,), dtype=torch.float32),
        boundary_found=torch.ones((1,), dtype=torch.bool),
        terminated=torch.zeros((1,), dtype=torch.bool),
    )
    clean = reward_fn(action=applied, requested_action=applied, **common)
    saturated = reward_fn(action=applied, requested_action=requested, **common)
    assert float(clean.components["actuator_saturation_loss"].item()) == pytest.approx(0.0)
    assert float(saturated.components["actuator_saturation_loss"].item()) == pytest.approx(1.0)
    assert float(saturated.components["action_saturation_delta_rms"].item()) == pytest.approx(1.0)
    assert float(saturated.components["action_saturation_fraction"].item()) == pytest.approx(1.0)
    assert saturated.reward[0] < clean.reward[0]


def test_physical_reward_penalizes_always_on_coil_usage_below_soft_limit() -> None:
    reward_fn = T15PhysicalReward(
        RewardConfig(
            reward_scale=1.0,
            current_weight=0.0,
            derivative_weight=0.0,
            current_usage_weight=2.0,
            derivative_usage_weight=3.0,
            action_weight=0.0,
            delta_action_weight=0.0,
            actuator_saturation_weight=0.0,
        ),
        control_rate_hz=1000.0,
    )
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    action = torch.zeros((2, 9), dtype=torch.float32)
    common = dict(
        ip=torch.tensor([200000.0, 200000.0]),
        ip_ref=torch.tensor([200000.0, 200000.0]),
        boundary_points=ref,
        reference_points=ref,
        action=action,
        previous_action=action,
        current_over_limit_a=torch.zeros((2,), dtype=torch.float32),
        current_usage_fraction=torch.tensor([0.4, 0.8], dtype=torch.float32),
        current_margin_fraction=torch.tensor([0.6, 0.2], dtype=torch.float32),
        derivative_usage=torch.tensor([0.2, 0.6], dtype=torch.float32),
        current_usage_loss=torch.tensor([0.04, 0.16], dtype=torch.float32),
        derivative_usage_loss=torch.tensor([0.01, 0.09], dtype=torch.float32),
        current_usage_mean_fraction=torch.tensor([0.2, 0.4], dtype=torch.float32),
        derivative_usage_mean_fraction=torch.tensor([0.1, 0.3], dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.zeros((2,), dtype=torch.bool),
    )
    rb = reward_fn(**common)
    assert float(rb.components["current_loss"][0].item()) == pytest.approx(0.0)
    assert float(rb.components["derivative_loss"][0].item()) == pytest.approx(0.0)
    assert float(rb.components["current_usage_loss"][0].item()) == pytest.approx(0.04)
    assert float(rb.components["derivative_usage_loss"][1].item()) == pytest.approx(0.09)
    assert float(rb.components["current_usage_mean_fraction"][1].item()) == pytest.approx(0.4)
    assert rb.reward[1] < rb.reward[0]


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
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0, shape_mean_scale_m=0.025, boundary_missing_error_m=0.1), control_rate_hz=1000.0)
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
    assert float(rb.components["shape_mean_loss"].item()) > 3.0
    assert float(rb.components["physical_cost"].item()) > 10.0


def test_physical_reward_improves_when_tracking_errors_improve() -> None:
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0, shape_mean_scale_m=0.03, ip_scale_a=40000.0), control_rate_hz=1000.0)
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
    assert rb.components["shape_mean_loss"][0] < rb.components["shape_mean_loss"][1]
    assert rb.components["ip_loss"][0] < rb.components["ip_loss"][1]


def test_physical_reward_worst_boundary_point_matters() -> None:
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0), control_rate_hz=1000.0)
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    boundary = ref.clone()
    boundary[0, :, 0] = 0.01
    boundary[1, 0, 0] = 0.10
    action = torch.zeros((2, 9), dtype=torch.float32)
    rb = reward_fn(
        ip=torch.full((2,), 200000.0),
        ip_ref=torch.full((2,), 200000.0),
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
    assert rb.components["shape_error_mean_m"][1] < rb.components["shape_error_mean_m"][0]
    assert rb.components["shape_max_loss"][1] > rb.components["shape_max_loss"][0]


def test_physical_reward_terminal_penalty_applies_to_hard_failures() -> None:
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0, terminal_reward=-100.0), control_rate_hz=1000.0)
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    action = torch.zeros((2, 9), dtype=torch.float32)
    rb = reward_fn(
        ip=torch.tensor([200000.0, 200000.0]),
        ip_ref=torch.tensor([200000.0, 200000.0]),
        boundary_points=ref,
        reference_points=ref,
        action=action,
        previous_action=action,
        current_over_limit_a=torch.zeros((2,), dtype=torch.float32),
        current_usage_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        current_margin_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        derivative_usage=torch.zeros((2,), dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.tensor([False, True], dtype=torch.bool),
    )
    assert rb.reward[1] < rb.reward[0]
    assert float((rb.reward[0] - rb.reward[1]).item()) == pytest.approx(100.0)


def test_physical_reward_terminal_remaining_cost_is_worse_earlier() -> None:
    reward_fn = T15PhysicalReward(
        RewardConfig(reward_scale=1.0, terminal_reward=-100.0, terminal_remaining_cost=1000.0),
        control_rate_hz=1000.0,
    )
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    action = torch.zeros((2, 9), dtype=torch.float32)
    rb = reward_fn(
        ip=torch.tensor([200000.0, 200000.0]),
        ip_ref=torch.tensor([200000.0, 200000.0]),
        boundary_points=ref,
        reference_points=ref,
        action=action,
        previous_action=action,
        current_over_limit_a=torch.zeros((2,), dtype=torch.float32),
        current_usage_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        current_margin_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        derivative_usage=torch.zeros((2,), dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.tensor([True, True], dtype=torch.bool),
        episode_progress=torch.tensor([0.25, 0.75], dtype=torch.float32),
    )
    assert rb.reward[0] < rb.reward[1]
    assert float((rb.reward[1] - rb.reward[0]).item()) == pytest.approx(500.0)
    comps = rb.components
    assert float(comps["terminal_remaining_loss"][0].item()) == pytest.approx(750.0)
    assert float(comps["terminal_remaining_loss"][1].item()) == pytest.approx(250.0)
    assert float(comps["terminal_total_penalty"][0].item()) == pytest.approx(-850.0)


def test_physical_reward_terminal_remaining_cost_does_not_change_live_steps() -> None:
    base = T15PhysicalReward(RewardConfig(reward_scale=1.0, terminal_remaining_cost=0.0), control_rate_hz=1000.0)
    survival = T15PhysicalReward(RewardConfig(reward_scale=1.0, terminal_remaining_cost=100000.0), control_rate_hz=1000.0)
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    action = torch.zeros((2, 9), dtype=torch.float32)
    kwargs = dict(
        ip=torch.tensor([200000.0, 220000.0]),
        ip_ref=torch.tensor([210000.0, 210000.0]),
        boundary_points=ref,
        reference_points=ref,
        action=action,
        previous_action=action,
        current_over_limit_a=torch.zeros((2,), dtype=torch.float32),
        current_usage_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        current_margin_fraction=torch.full((2,), 0.5, dtype=torch.float32),
        derivative_usage=torch.zeros((2,), dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.zeros((2,), dtype=torch.bool),
        episode_progress=torch.tensor([0.25, 0.75], dtype=torch.float32),
    )
    base_rb = base(**kwargs)
    survival_rb = survival(**kwargs)
    assert torch.allclose(base_rb.reward, survival_rb.reward)
    assert torch.count_nonzero(survival_rb.components["terminal_remaining_loss"]).item() == 0
    assert torch.count_nonzero(survival_rb.components["terminal_total_penalty"]).item() == 0


def test_physical_reward_penalizes_current_usage() -> None:
    reward_fn = T15PhysicalReward(RewardConfig(reward_scale=1.0), control_rate_hz=1000.0)
    ref = torch.zeros((2, 32, 2), dtype=torch.float32)
    action = torch.zeros((2, 9), dtype=torch.float32)
    action[1] = 0.95
    rb = reward_fn(
        ip=torch.full((2,), 200000.0),
        ip_ref=torch.full((2,), 200000.0),
        boundary_points=ref,
        reference_points=ref,
        action=action,
        previous_action=torch.zeros((2, 9), dtype=torch.float32),
        current_over_limit_a=torch.tensor([0.0, 1000.0], dtype=torch.float32),
        current_usage_fraction=torch.tensor([0.50, 1.20], dtype=torch.float32),
        current_margin_fraction=torch.tensor([0.50, -0.20], dtype=torch.float32),
        derivative_usage=torch.tensor([0.10, 0.95], dtype=torch.float32),
        boundary_found=torch.ones((2,), dtype=torch.bool),
        terminated=torch.zeros((2,), dtype=torch.bool),
    )
    assert rb.reward[0] > rb.reward[1]
    assert rb.components["current_over_limit_a"][1] > rb.components["current_over_limit_a"][0]
    assert float(rb.components["current_loss"][0].item()) == pytest.approx(0.0)
    assert rb.components["current_loss"][1] > rb.components["current_loss"][0]
    assert rb.components["derivative_usage"][1] > rb.components["derivative_usage"][0]
    assert rb.components["max_abs_action"][1] > rb.components["max_abs_action"][0]
    for removed in ("shape_quality", "ip_quality", "current_quality", "combined_quality", "current_limit_loss", "time_weight"):
        assert removed not in rb.components


def test_physical_reward_actuator_losses_start_above_legal_envelope() -> None:
    reward_fn = T15PhysicalReward(
        RewardConfig(
            reward_scale=1.0,
            current_soft_fraction=1.0,
            current_bad_fraction=1.4,
            derivative_soft_fraction=1.0,
            derivative_bad_fraction=1.4,
            current_weight=1.0,
            derivative_weight=1.0,
            action_weight=0.0,
            delta_action_weight=0.0,
        ),
        control_rate_hz=1000.0,
    )
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
        current_usage_fraction=torch.tensor([0.75, 1.0, 1.2], dtype=torch.float32),
        current_margin_fraction=torch.tensor([0.25, 0.0, -0.2], dtype=torch.float32),
        derivative_usage=torch.tensor([0.5, 1.0, 1.2], dtype=torch.float32),
        boundary_found=torch.ones((3,), dtype=torch.bool),
        terminated=torch.zeros((3,), dtype=torch.bool),
    )
    assert float(rb.components["current_loss"][0].item()) == pytest.approx(0.0)
    assert float(rb.components["current_loss"][1].item()) == pytest.approx(0.0)
    assert rb.components["current_loss"][2] > 0.0
    assert float(rb.components["derivative_loss"][0].item()) == pytest.approx(0.0)
    assert float(rb.components["derivative_loss"][1].item()) == pytest.approx(0.0)
    assert rb.components["derivative_loss"][2] > 0.0


def test_environment_reset_step_contract() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=4))
    env = TokamakMagneticControlEnv(cfg, batch_size=2, device="cpu", seed=1)
    schema = env.export_schema()
    assert schema["observation_kind"] == "controller_state_v3"
    assert schema["critic_observation_kind"] == "privileged_training_state_v1"
    assert "diagnostics" not in schema
    assert "psi_flat" not in schema["feature_order"]
    assert "psi_flat_normalized" in schema["critic_feature_order"]
    assert "previous_action" in schema["feature_order"]
    assert "previous_derivative_command" in schema["feature_order"]
    assert "measured_boundary_radii" in schema["feature_order"]
    assert "boundary_radii_error" in schema["feature_order"]
    assert env.obs_dim == schema["feature_slices"]["target_preview"][1]
    assert env.critic_obs_dim == schema["critic_feature_slices"]["derivative_usage"][1]
    assert "flux_scale" not in env.normalization()
    assert "field_scale" not in env.normalization()
    assert "bdot_scale" not in env.normalization()
    assert env.normalization()["critic_psi_normalization"] == "per_reset_standardization"
    assert env.normalization()["action_contract"] == "delta_jdot_derivative_command_v3"
    assert env.normalization()["delta_derivative_scale_aps"] == pytest.approx(500000.0)
    assert len(env.normalization()["delta_derivative_limits_aps"]) == env.action_dim
    if cfg.reward.kind == "tcv_derivative":
        assert "current_saturation_fraction" not in env.normalization()
    else:
        assert env.normalization()["current_saturation_fraction"] == pytest.approx(float(cfg.sim.current_saturation_fraction))
    obs = env.reset()
    assert obs.shape == (2, env.obs_dim)
    assert env.critic_obs().shape == (2, env.critic_obs_dim)
    psi0, psi1 = schema["critic_feature_slices"]["psi_flat_normalized"]
    psi = env.critic_obs()[:, int(psi0) : int(psi1)]
    assert torch.allclose(torch.mean(psi, dim=1), torch.zeros((2,), dtype=torch.float32), atol=1.0e-5)
    assert torch.allclose(torch.std(psi, dim=1, unbiased=False), torch.ones((2,), dtype=torch.float32), atol=1.0e-4)
    result = env.step(torch.zeros((2, env.action_dim)))
    assert result.obs.shape == (2, env.obs_dim)
    assert result.critic_obs.shape == (2, env.critic_obs_dim)
    assert result.requested_action.shape == (2, env.action_dim)
    assert result.applied_action.shape == (2, env.action_dim)
    assert result.reward.shape == (2,)
    assert torch.isfinite(result.requested_action).all()
    assert torch.isfinite(result.reward).all()
    assert torch.isfinite(result.applied_action).all()


def test_action_scale_caps_physical_derivative_usage() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=4, action_scale=0.25, action_contract="absolute_derivative"))
    env = TokamakMagneticControlEnv(cfg, batch_size=2, device="cpu", seed=2)
    env.reset()
    result = env.step(torch.ones((2, env.action_dim), dtype=torch.float32))
    comps = result.info["reward_components"]
    assert float(np.nanmax(comps["derivative_usage"])) <= 0.25001
    assert np.allclose(np.asarray(env.normalization()["derivative_scale"]), env.raw_derivative_limits.detach().cpu().numpy() * 0.25)


def test_delta_jdot_action_accumulates_derivative_command() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(
        cfg,
        sim=replace(
            cfg.sim,
            compute_backend="cpu",
            max_episode_steps=4,
            action_contract="delta_jdot",
            delta_derivative_limits_aps=DeltaDerivativeLimits(
                pfc=(100000.0, 200000.0, 300000.0, 400000.0, 500000.0, 600000.0),
                sol=(700000.0, 800000.0, 900000.0),
            ),
            terminate_on_current_limit=False,
            current_saturation_fraction=1.0e6,
        ),
    )
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=3)
    env.reset()
    action = torch.ones((1, env.action_dim), dtype=torch.float32)
    first = env.step(action)
    expected_delta = env.delta_action_to_command_norm[None, :]
    assert env.delta_derivative_scale.detach().cpu().numpy().tolist() == pytest.approx(
        [100000.0, 200000.0, 300000.0, 400000.0, 500000.0, 600000.0, 700000.0, 800000.0, 900000.0]
    )
    assert torch.allclose(first.requested_action, action)
    assert torch.allclose(first.applied_action, expected_delta, atol=1.0e-6)
    second = env.step(action)
    assert torch.allclose(second.applied_action, torch.clamp(2.0 * expected_delta, -1.0, 1.0), atol=1.0e-6)
    assert torch.allclose(env.previous_action, action)
    assert torch.allclose(env.previous_derivative_command, second.applied_action)
    schema = env.export_schema()
    assert "previous_derivative_command" in schema["feature_order"]
    assert env.normalization()["previous_action_semantics"] == "previous_requested_delta_action"
    assert env.normalization()["previous_derivative_command_semantics"] == "applied_accumulated_derivative_command"


def test_delta_jdot_command_clipping_penalizes_unrealized_delta() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(
        cfg,
        sim=replace(
            cfg.sim,
            compute_backend="cpu",
            max_episode_steps=4,
            action_contract="delta_jdot",
            delta_derivative_limits_aps=DeltaDerivativeLimits(
                pfc=(1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9),
                sol=(1.0e9, 1.0e9, 1.0e9),
            ),
            terminate_on_current_limit=False,
            current_saturation_fraction=1.0e6,
        ),
        reward=replace(cfg.reward, actuator_saturation_weight=4.0),
    )
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=4)
    env.reset()
    result = env.step(torch.ones((1, env.action_dim), dtype=torch.float32))
    comps = result.info["reward_components"]
    assert env.normalization()["action_contract"] == "delta_jdot_derivative_command_v3"
    assert torch.allclose(result.applied_action, torch.ones_like(result.applied_action), atol=1.0e-6)
    assert float(np.nanmax(comps["action_saturation_delta_rms"])) > 0.0
    assert float(np.nanmax(comps["actuator_saturation_loss"])) > 0.0


def test_delta_jdot_contract_does_not_project_current_runaway_command() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(
        cfg,
        sim=replace(
            cfg.sim,
            compute_backend="cpu",
            max_episode_steps=4,
            action_contract="delta_jdot",
            terminate_on_current_limit=False,
            current_saturation_fraction=1.05,
        ),
        reward=replace(cfg.reward, actuator_saturation_weight=4.0),
    )
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=12)
    env.reset()
    limit = float(env.current_limits[0].item())
    upper = 1.05 * limit
    env._cpu_models[0].state.pfc_currents[0] = upper - 1.0
    action = torch.zeros((1, env.action_dim), dtype=torch.float32)
    action[0, 0] = 1.0
    result = env.step(action)
    comps = result.info["reward_components"]
    next_current = float(env._cpu_models[0].state.pfc_currents[0])
    assert next_current > upper
    assert torch.allclose(result.requested_action, action)
    assert result.applied_action[0, 0].item() == pytest.approx(env.delta_action_to_command_norm[0].item())
    assert env.previous_action[0, 0].item() == pytest.approx(action[0, 0].item())
    assert env.previous_derivative_command[0, 0].item() == pytest.approx(result.applied_action[0, 0].item())
    assert float(np.nanmax(comps["action_saturation_delta_rms"])) == pytest.approx(0.0)
    assert float(np.nanmax(comps["action_saturation_fraction"])) == pytest.approx(0.0)
    assert float(np.nanmax(comps["actuator_saturation_loss"])) == pytest.approx(0.0)
    assert float(np.nanmax(comps["current_usage_fraction"])) > 1.0
    assert bool(result.terminated[0].item()) is False


def test_tcv_derivative_mode_does_not_project_current_runaway_command() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(
        cfg,
        sim=replace(
            cfg.sim,
            compute_backend="cpu",
            max_episode_steps=4,
            action_contract="delta_jdot",
            delta_derivative_limits_aps=DeltaDerivativeLimits(
                pfc=(163347.0, 310755.0, 87838.08, 153214.2, 404364.0, 1191036.96),
                sol=(1437338.8, 5889842.0, 1946208.8),
            ),
            terminate_on_boundary_loss=True,
            terminate_on_current_limit=True,
            current_hard_termination_fraction=1.20,
            current_termination_grace_steps=1,
            current_saturation_fraction=1.0,
        ),
        reward=replace(
            cfg.reward,
            kind="tcv_derivative",
            reward_scale=0.01,
            smoothmax_alpha=-5.0,
            terminal_reward=-5.0,
            terminal_remaining_cost=0.0,
            current_usage_weight=0.0,
            derivative_usage_weight=0.0,
            action_weight=0.0,
            delta_action_weight=0.0,
        ),
    )
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=12)
    env.reset()
    limit = float(env.current_limits[0].item())
    env._cpu_models[0].state.pfc_currents[0] = 1.21 * limit
    action = torch.zeros((1, env.action_dim), dtype=torch.float32)
    action[0, 0] = 1.0
    result = env.step(action)
    comps = result.info["reward_components"]
    expected_command = action * env.delta_action_to_command_norm[None, :]
    assert torch.allclose(result.requested_action, action)
    assert torch.allclose(result.applied_action, expected_command, atol=1.0e-6)
    assert env.normalization()["action_contract"] == "delta_jdot_derivative_command_v3"
    assert env.normalization()["delta_derivative_limits_aps"][5] == pytest.approx(1191036.96)
    assert "current_saturation_fraction" not in env.normalization()
    assert bool(result.terminated[0].item()) is True
    assert float(np.nanmax(comps["terminated_current"])) == pytest.approx(1.0)
    assert float(np.nanmax(comps["action_saturation_delta_rms"])) == pytest.approx(0.0)


def test_current_aware_saturation_leaves_safe_command_unchanged() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(
        cfg,
        sim=replace(
            cfg.sim,
            compute_backend="cpu",
            max_episode_steps=4,
            action_contract="absolute_derivative",
            terminate_on_current_limit=False,
            current_saturation_fraction=1.05,
        ),
    )
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=13)
    env.reset()
    action = torch.zeros((1, env.action_dim), dtype=torch.float32)
    action[0, 0] = 0.01
    result = env.step(action)
    comps = result.info["reward_components"]
    assert torch.allclose(result.applied_action, action, atol=1.0e-6)
    assert float(np.nanmax(comps["action_saturation_delta_rms"])) == pytest.approx(0.0)
    assert float(np.nanmax(comps["actuator_saturation_loss"])) == pytest.approx(0.0)


def test_small_current_limit_violation_gets_grace_before_termination() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(
        cfg,
        sim=replace(
            cfg.sim,
            compute_backend="cpu",
            max_episode_steps=16,
            terminate_on_current_limit=True,
            current_termination_over_limit_a=5000.0,
            current_termination_grace_steps=3,
            current_hard_termination_fraction=1.10,
        ),
    )
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=12)
    env.reset()
    limit = float(env.current_limits[0].item())
    action = torch.zeros((1, env.action_dim), dtype=torch.float32)
    env._cpu_models[0].state.pfc_currents[0] = 1.04 * limit
    first = env.step(action)
    assert bool(first.terminated[0].item()) is False
    assert int(env.current_over_limit_steps[0].item()) == 1
    env._cpu_models[0].state.pfc_currents[0] = 0.90 * limit
    recovered = env.step(action)
    assert bool(recovered.terminated[0].item()) is False
    assert int(env.current_over_limit_steps[0].item()) == 0

    for _ in range(2):
        env._cpu_models[0].state.pfc_currents[0] = 1.04 * limit
        result = env.step(action)
    comps = result.info["reward_components"]
    assert bool(result.terminated[0].item()) is False
    assert float(np.nanmax(comps["terminated_current"])) == pytest.approx(0.0)
    env._cpu_models[0].state.pfc_currents[0] = 1.04 * limit
    result = env.step(action)
    comps = result.info["reward_components"]
    assert float(np.nanmax(comps["current_over_limit_a"])) > 0.0
    assert float(np.nanmax(comps["current_usage_fraction"])) > 1.0
    assert bool(result.terminated[0].item()) is True
    assert float(np.nanmax(comps["terminated_current"])) == pytest.approx(1.0)
    assert float(np.nanmax(comps["terminated_current_grace"])) == pytest.approx(1.0)
    assert torch.allclose(result.applied_action, action)
    assert torch.allclose(env.previous_action, action)
    assert torch.allclose(env.previous_derivative_command, result.applied_action)


def test_severe_current_limit_violation_terminates_immediately() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(
        cfg,
        sim=replace(
            cfg.sim,
            compute_backend="cpu",
            max_episode_steps=4,
            terminate_on_current_limit=True,
            current_termination_over_limit_a=5000.0,
            current_termination_grace_steps=8,
            current_hard_termination_fraction=1.05,
        ),
    )
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=12)
    env.reset()
    limit = float(env.current_limits[0].item())
    env._cpu_models[0].state.pfc_currents[0] = 1.06 * limit
    result = env.step(torch.zeros((1, env.action_dim), dtype=torch.float32))
    comps = result.info["reward_components"]
    assert bool(result.terminated[0].item()) is True
    assert float(np.nanmax(comps["terminated_current_hard"])) == pytest.approx(1.0)


def test_current_limit_violation_is_logged_but_not_terminal_when_disabled() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(
        cfg,
        sim=replace(
            cfg.sim,
            compute_backend="cpu",
            max_episode_steps=4,
            terminate_on_current_limit=False,
            current_termination_over_limit_a=5000.0,
            current_termination_grace_steps=2,
            current_hard_termination_fraction=1.01,
        ),
    )
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=12)
    env.reset()
    limit = float(env.current_limits[0].item())
    env._cpu_models[0].state.pfc_currents[0] = 1.08 * limit
    result = env.step(torch.zeros((1, env.action_dim), dtype=torch.float32))
    comps = result.info["reward_components"]
    assert bool(result.terminated[0].item()) is False
    assert float(np.nanmax(comps["current_over_limit_a"])) > 0.0
    assert float(np.nanmax(comps["current_usage_fraction"])) > 1.0
    assert float(np.nanmax(comps["terminated_current"])) == pytest.approx(0.0)
    assert int(env.current_over_limit_steps[0].item()) == 1


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


def test_policy_pipeline_writes_no_control_baseline_before_training(tmp_path: Path) -> None:
    _write_baseline_report(
        tmp_path,
        reset_sanity={"boundary_found_mean": 1.0},
        no_control_selection={"ip_error_a": 10.0},
        no_control={"ip_error_a": 20.0},
        selection_seed_offset=100000,
        holdout_seed_offset=200000,
    )
    data = json.loads((tmp_path / "no_control_baseline.json").read_text(encoding="utf-8"))
    assert data["status"] == "ready_for_training"
    assert data["evaluation_seed_offsets"] == {"selection": 100000, "holdout": 200000}
    assert data["no_control_selection"]["ip_error_a"] == pytest.approx(10.0)
    assert data["no_control"]["ip_error_a"] == pytest.approx(20.0)


def test_array_reference_scenario_preserves_changing_ip_and_radii() -> None:
    scenario = _ArrayReferenceScenario(
        ip=np.array([100000.0, 150000.0, 200000.0], dtype=float),
        radii=np.array([[0.30, 0.31], [0.32, 0.33], [0.34, 0.35]], dtype=float),
        dt=0.01,
    )
    angles = np.array([0.0, 1.0], dtype=float)
    assert scenario.Ip_ref(0.0) == pytest.approx(100000.0)
    assert scenario.Ip_ref(0.01) == pytest.approx(150000.0)
    assert scenario.Ip_ref(0.02) == pytest.approx(200000.0)
    assert np.allclose(scenario.ref_radii(angles, 0.01), np.array([0.32, 0.33]))


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


def test_replay_start_new_episodes_prevents_external_reset_crossing() -> None:
    replay = FIFOSequenceReplay(capacity_episodes=4, max_episode_steps=6, active_envs=1, obs_dim=1, action_dim=1, device="cpu")
    replay.add_batch(
        torch.tensor([[1.0]]),
        torch.zeros((1, 1)),
        torch.zeros((1,)),
        torch.ones((1,)),
        torch.tensor([[2.0]]),
        torch.zeros((1,), dtype=torch.bool),
    )
    replay.add_batch(
        torch.tensor([[2.0]]),
        torch.zeros((1, 1)),
        torch.zeros((1,)),
        torch.ones((1,)),
        torch.tensor([[3.0]]),
        torch.zeros((1,), dtype=torch.bool),
    )
    replay.start_new_episodes()
    replay.add_batch(
        torch.tensor([[100.0]]),
        torch.zeros((1, 1)),
        torch.zeros((1,)),
        torch.ones((1,)),
        torch.tensor([[101.0]]),
        torch.zeros((1,), dtype=torch.bool),
    )

    assert not replay.ready(sequence_length=3, batch_size=1)


def test_replay_samples_short_terminal_episodes_with_padding_mask() -> None:
    replay = FIFOSequenceReplay(capacity_episodes=4, max_episode_steps=8, active_envs=1, obs_dim=1, action_dim=1, device="cpu")
    for t in range(3):
        replay.add_batch(
            torch.tensor([[float(t)]]),
            torch.tensor([[0.1 * t]]),
            torch.tensor([float(t)]),
            torch.ones((1,)),
            torch.tensor([[float(t + 1)]]),
            torch.tensor([t == 2], dtype=torch.bool),
        )

    assert not replay.ready(sequence_length=6, batch_size=2)
    assert replay.ready(sequence_length=6, batch_size=2, min_sequence_length=3)
    batch = replay.sample(batch_size=2, sequence_length=6, min_sequence_length=3)
    assert batch.obs.shape == (2, 6, 1)
    assert torch.all(batch.mask[:, :3] == 1.0)
    assert torch.all(batch.mask[:, 3:] == 0.0)
    assert torch.all(batch.done[:, 2])


def test_replay_batched_insert_preserves_lane_boundaries_and_size() -> None:
    replay = FIFOSequenceReplay(capacity_episodes=6, max_episode_steps=4, active_envs=3, obs_dim=2, action_dim=1, device="cpu")
    for t in range(3):
        obs = torch.tensor([[0.0, float(t)], [1.0, float(t)], [2.0, float(t)]])
        done = torch.tensor([False, t == 1, False])
        replay.add_batch(obs, torch.zeros((3, 1)), torch.zeros((3,)), torch.ones((3,)), obs + 0.5, done)

    assert replay.size == 9
    assert replay.completed_episodes >= 1
    batch = replay.sample(batch_size=12, sequence_length=2)
    assert torch.all(batch.obs[:, :, 0] == batch.obs[:, :1, 0])
    assert not torch.any(batch.done[:, :-1])


def test_environment_reset_indices_only_resets_done_slot() -> None:
    cfg = load_experiment_config(CONFIG)
    cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", max_episode_steps=8, action_contract="absolute_derivative"))
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


def test_trainer_replay_stores_requested_actor_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _small_config(tmp_path)
    cfg = replace(cfg, training=replace(cfg.training, steps=1, eval_interval_steps=99))
    trainer = Trainer(cfg)
    raw_action = torch.ones((trainer.num_envs, trainer.env.action_dim), dtype=torch.float32, device=trainer.device)
    applied_action = torch.full_like(raw_action, 0.25)
    obs0 = torch.zeros((trainer.num_envs, trainer.env.obs_dim), dtype=torch.float32, device=trainer.device)
    obs1 = torch.ones_like(obs0)
    critic0 = torch.zeros((trainer.num_envs, trainer.env.critic_obs_dim), dtype=torch.float32, device=trainer.device)
    critic1 = torch.ones_like(critic0)

    def fake_sample(_obs):
        return raw_action.clone(), None, None

    def fake_step(_action):
        return BatchStep(
            obs=obs1.clone(),
            critic_obs=critic1.clone(),
            requested_action=raw_action.clone(),
            applied_action=applied_action.clone(),
            reward=torch.zeros((trainer.num_envs,), dtype=torch.float32, device=trainer.device),
            terminated=torch.zeros((trainer.num_envs,), dtype=torch.bool, device=trainer.device),
            truncated=torch.zeros((trainer.num_envs,), dtype=torch.bool, device=trainer.device),
            info={},
        )

    monkeypatch.setattr(trainer.actor, "sample", fake_sample)
    monkeypatch.setattr(trainer.env, "reset", lambda: obs0.clone())
    monkeypatch.setattr(trainer.env, "critic_obs", lambda: critic0.clone())
    monkeypatch.setattr(trainer.env, "step", fake_step)
    monkeypatch.setattr(trainer, "_export", lambda *args, **kwargs: None)

    trainer.train()

    for slot in trainer.replay.active_slots.detach().cpu().tolist():
        assert torch.allclose(trainer.replay.action[int(slot), 0], raw_action[0])


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


def test_production_segmented_profile_uses_2000_step_t15_scale_segments() -> None:
    cfg = load_experiment_config(PRODUCTION_CONFIG)
    assert int(cfg.sim.max_episode_steps) == 2000
    assert cfg.reference.duration_s == pytest.approx(2.0)
    assert cfg.reference.duration_s == pytest.approx(float(cfg.sim.max_episode_steps) * float(cfg.reference.t_step))
    assert int(cfg.training.eval_max_steps) == 2000
    assert cfg.reference.ip.ramp_rate_reference == "robust_mean"
    assert cfg.reference.ip.ramp_up_rate_min_fraction == pytest.approx(0.3)
    assert cfg.reference.ip.ramp_up_rate_fraction == pytest.approx(0.55)
    assert cfg.reference.ip.ramp_down_rate_min_fraction == pytest.approx(0.3)
    assert cfg.reference.ip.ramp_down_rate_fraction == pytest.approx(0.55)
    assert int(cfg.reference.ip.segment_min_steps) == 300
    assert int(cfg.reference.ip.segment_max_steps) == 800
    assert int(cfg.reference.ip.segment_count_min) == 3
    assert int(cfg.reference.ip.segment_count_max) == 5
    assert int(cfg.reference.ip.hold_min_steps) == 300
    assert int(cfg.reference.ip.hold_max_steps) == 800
    assert int(cfg.reference.ip.final_hold_min_steps) == 0

    lengths = _segment_lengths(cfg.reference.ip, int(cfg.sim.max_episode_steps), np.random.default_rng(123))
    assert int(np.sum(lengths)) == 2000
    assert 3 <= len(lengths) <= 5
    assert int(np.min(lengths)) >= 300
    assert int(np.max(lengths)) <= 800


def test_production_config_loads_real_t15_delta_jdot_limits() -> None:
    cfg = load_experiment_config(PRODUCTION_CONFIG)
    assert cfg.sim.delta_derivative_limits_aps is not None
    assert cfg.sim.delta_derivative_limits_aps.pfc == pytest.approx(
        (163347.0, 310755.0, 87838.08, 153214.2, 404364.0, 1191036.96)
    )
    assert cfg.sim.delta_derivative_limits_aps.sol == pytest.approx(
        (1437338.8, 5889842.0, 1946208.8)
    )


def test_production_reference_generates_t15_replay_conditioned_boundary() -> None:
    cfg = load_experiment_config(PRODUCTION_CONFIG)
    assert cfg.reference.boundary.kind == "t15_replay_segment_conditioned"
    assert cfg.reference.boundary.replay_reference_dir is not None
    library = T15ReplayBoundaryLibrary(cfg.reference.boundary.replay_reference_dir, theta_count=int(cfg.reference.theta_count))
    with np.load(cfg.sim.csv_initial_state_library) as data:
        split = np.asarray(data["split"]).astype(str)
        keep = np.flatnonzero(split == "train")[:2]
        initial_ip = np.asarray(data["ip0"], dtype=float)[keep]
        shot_ids = np.asarray(data["shot_id"]).astype(str)[keep]
        source_indices = np.asarray(data["source_index"], dtype=np.int64)[keep]
        source_times_s = np.asarray(data["time_s"], dtype=float)[keep]
    reset_boundary_radii = np.ones((2, int(cfg.reference.theta_count)), dtype=float)
    reset_boundary_points = np.zeros((2, int(cfg.reference.theta_count), 2), dtype=float)
    batch = generate_reference_batch(
        config=cfg.reference,
        initial_ip=initial_ip,
        initial_parameters=np.zeros((2, 5), dtype=float),
        initial_boundary_points=reset_boundary_points,
        initial_boundary_radii=reset_boundary_radii,
        shot_ids=shot_ids,
        source_indices=source_indices,
        source_times_s=source_times_s,
        boundary_replay_library=library,
        boundary_center=(1.5, 0.0),
        steps=int(cfg.sim.max_episode_steps),
        device="cpu",
        seed=77,
    )
    assert batch.ip.shape == (2, 2001)
    assert batch.radii.shape == (2, 2001, int(cfg.reference.theta_count))
    assert batch.points.shape == (2, 2001, int(cfg.reference.theta_count), 2)
    assert torch.all(torch.isfinite(batch.radii))


def test_t15_replay_boundary_reference_uses_time_segment_not_ip_sort(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    np.savez(
        replay_dir / "lqr_boundary_reference_1_smoothed.npz",
        shot=np.asarray([1]),
        step=np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
        t=np.asarray([0.0, 0.001, 0.002, 0.003, 0.004], dtype=float),
        angles_rad=np.asarray([0.0, np.pi], dtype=float),
        Ip=np.asarray([100.0, 200.0, 100.0, 200.0, 100.0], dtype=float),
        radii_true=np.asarray(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
                [1.4, 2.4],
            ],
            dtype=float,
        ),
        boundary_found=np.ones((5,), dtype=bool),
    )
    cfg = ReferenceConfig(
        duration_s=0.003,
        t_step=0.001,
        theta_count=2,
        seed=1,
        ip=IpReferenceConfig(kind="hold_reset", min=0.0, max=1.0, rate_limit=0.0),
        boundary=BoundaryReferenceConfig(kind="t15_replay_segment_conditioned", replay_reference_dir=replay_dir),
    )
    library = T15ReplayBoundaryLibrary(replay_dir, theta_count=2)
    batch = generate_reference_batch(
        config=cfg,
        initial_ip=np.asarray([123.0], dtype=float),
        initial_parameters=np.zeros((1, 5), dtype=float),
        initial_boundary_points=np.zeros((1, 2, 2), dtype=float),
        initial_boundary_radii=np.asarray([[10.0, 20.0]], dtype=float),
        shot_ids=np.asarray(["1"]),
        source_indices=np.asarray([0]),
        boundary_replay_library=library,
        boundary_center=(1.5, 0.0),
        steps=3,
        device="cpu",
        seed=2,
    )
    expected = torch.tensor(
        [[[10.0, 20.0], [10.1, 20.1], [10.2, 20.2], [10.3, 20.3]]],
        dtype=torch.float64,
    )
    assert torch.allclose(batch.radii, expected)


def test_actuator_legality_analysis_reports_delta_jdot_plus_twenty_percent(tmp_path: Path) -> None:
    from scripts.analyze_t15_actuator_legality import analyze

    coils = tmp_path / "coils"
    coils.mkdir()
    rows = [
        [0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.001, 0.0, 20.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.002, 0.0, 10.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.003, 0.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    with (coils / "t15md_0001_coils.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(rows)

    result = analyze(tmp_path, min_dt_s=0.0005)
    limits = result["recommended_limits"]["delta_derivative_aps_plus_20pct"]
    assert limits["pfc"][0] == pytest.approx(24000.0)
    assert limits["sol"][1] == pytest.approx(36000.0)


def test_fixed_horizon_diagnostic_configs_disable_training_terminations() -> None:
    hold = load_experiment_config(FIXED_HORIZON_HOLD_CONFIG)
    easy = load_experiment_config(FIXED_HORIZON_EASY_SEGMENTED_CONFIG)
    for cfg in (hold, easy):
        assert cfg.training.production_mode is False
        assert cfg.sim.reset_source == "csv_initial_states"
        assert cfg.reference.boundary.kind == "hold_reset_boundary"
        assert int(cfg.sim.max_episode_steps) == 2000
        assert cfg.reference.duration_s == pytest.approx(2.0)
        assert cfg.sim.terminate_on_boundary_loss is False
        assert cfg.sim.terminate_on_current_limit is False
        assert cfg.reward.terminal_remaining_cost == pytest.approx(0.0)
        assert cfg.learner.discount == pytest.approx(0.9995)
    assert hold.reference.ip.kind == "hold_reset"
    assert easy.reference.ip.kind == "segmented_profile"
    assert easy.reference.ip.max_delta_fraction == pytest.approx(0.10)
    assert int(easy.reference.ip.segment_min_steps) == 800
    assert int(easy.reference.ip.segment_max_steps) == 1600


def test_fixed_horizon_hold_reset_reference_uses_reset_ip() -> None:
    cfg = load_experiment_config(FIXED_HORIZON_HOLD_CONFIG)
    reset_ip = np.asarray([123456.0, 234567.0], dtype=float)
    reset_boundary_points = np.zeros((2, int(cfg.reference.theta_count), 2), dtype=float)
    reset_boundary_radii = np.ones((2, int(cfg.reference.theta_count)), dtype=float)
    batch = generate_reference_batch(
        config=cfg.reference,
        initial_ip=reset_ip,
        initial_parameters=np.zeros((2, 5), dtype=float),
        initial_boundary_points=reset_boundary_points,
        initial_boundary_radii=reset_boundary_radii,
        steps=int(cfg.sim.max_episode_steps),
        device="cpu",
        seed=11,
    )
    assert batch.ip.shape == (2, int(cfg.sim.max_episode_steps) + 1)
    assert torch.allclose(batch.ip[0], torch.full_like(batch.ip[0], reset_ip[0]))
    assert torch.allclose(batch.ip[1], torch.full_like(batch.ip[1], reset_ip[1]))


def test_hold_reset_ip_reference_uses_actual_reset_ip() -> None:
    cfg = load_experiment_config(CONFIG)
    hold_ip = replace(
        cfg.reference.ip,
        kind="hold_reset",
        min=100000.0,
        max=160000.0,
        hold_probability=1.0,
    )
    reference = replace(cfg.reference, ip=hold_ip)
    batch = generate_reference_batch(
        config=reference,
        initial_ip=np.asarray([124800.0, 125100.0]),
        initial_parameters=np.asarray([[1.4, 0.0, 0.55, 1.2, 0.2], [1.42, -0.01, 0.58, 1.25, 0.18]]),
        steps=20,
        device="cpu",
        seed=123,
    )
    assert torch.allclose(batch.ip[0], torch.full((21,), 124800.0, dtype=torch.float64))
    assert torch.allclose(batch.ip[1], torch.full((21,), 125100.0, dtype=torch.float64))


def test_sim_limit_scales_expand_current_and_derivative_limits() -> None:
    cfg = load_experiment_config(CONFIG)
    base_cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", current_limit_scale=1.0, derivative_limit_scale=1.0))
    scaled_cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu", current_limit_scale=1.2, derivative_limit_scale=1.2))
    base_env = TokamakMagneticControlEnv(base_cfg, batch_size=1, device="cpu", seed=123)
    scaled_env = TokamakMagneticControlEnv(scaled_cfg, batch_size=1, device="cpu", seed=123)
    assert np.allclose(scaled_env.current_limits.cpu().numpy(), base_env.current_limits.cpu().numpy() * 1.2)
    assert np.allclose(scaled_env.raw_derivative_limits.cpu().numpy(), base_env.raw_derivative_limits.cpu().numpy() * 1.2)
    assert np.allclose(scaled_env.derivative_limits.cpu().numpy(), base_env.derivative_limits.cpu().numpy() * 1.2)
    assert scaled_env.cfg.physics.pfc_deriv_limit == pytest.approx(float(base_env.cfg.physics.pfc_deriv_limit) * 1.2)
    assert scaled_env.cfg.physics.sol_deriv_limit == pytest.approx(float(base_env.cfg.physics.sol_deriv_limit) * 1.2)


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
    assert not (tmp_path / "checkpoints" / "final.pt").exists()
    assert (tmp_path / "exports" / "final_actor" / "policy_weights.npz").exists()
    assert (tmp_path / "losses.csv").exists()
    assert (tmp_path / "reward_components.csv").exists()
    assert (tmp_path / "replay_health.csv").exists()


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
    assert not (tmp_path / "checkpoints" / "final.pt").exists()
    assert (tmp_path / "exports" / "final_actor" / "policy_weights.npz").exists()
    assert (tmp_path / "replay_health.csv").exists()


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


def test_config_loader_accepts_local_replay_and_rejects_actor_workers(tmp_path: Path) -> None:
    data = json.loads(CONFIG.read_text())
    data["training"]["distributed_mode"] = "local_replay"
    path = tmp_path / "local_replay.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    cfg = load_experiment_config(path)
    assert cfg.training.distributed_mode == "local_replay"

    data["training"]["actor_workers"] = 2
    bad_path = tmp_path / "bad_local_replay_workers.json"
    bad_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="local_replay does not use actor_workers"):
        load_experiment_config(bad_path)

    data = json.loads(CONFIG.read_text())
    data["training"]["distributed_mode"] = "not_real"
    bad_mode = tmp_path / "bad_distributed_mode.json"
    bad_mode.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="distributed_mode"):
        load_experiment_config(bad_mode)


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
    data["reward"]["shape_max_bad_m"] = 0.08
    bad_shape_max = tmp_path / "bad_shape_max.json"
    bad_shape_max.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stale keys.*shape_max_bad_m"):
        load_experiment_config(bad_shape_max)

    data = json.loads(CONFIG.read_text())
    data["reward"]["current_margin_start_fraction"] = 0.75
    bad_current_margin = tmp_path / "bad_current_margin.json"
    bad_current_margin.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stale keys.*current_margin_start_fraction"):
        load_experiment_config(bad_current_margin)

    data = json.loads(CONFIG.read_text())
    data["reward"]["delta_action_bad"] = 1.0
    bad_delta = tmp_path / "bad_delta.json"
    bad_delta.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stale keys.*delta_action_bad"):
        load_experiment_config(bad_delta)

    data = json.loads(CONFIG.read_text())
    data["sim"]["action_scale"] = 0.0
    bad_action_scale = tmp_path / "bad_action_scale.json"
    bad_action_scale.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="action_scale"):
        load_experiment_config(bad_action_scale)

    data = json.loads(CONFIG.read_text())
    data["sim"]["delta_derivative_limits_aps"] = {
        "pfc": {
            "pfc0": 1.0,
            "pfc1": 1.0,
            "pfc2": -1.0,
            "pfc3": 1.0,
            "pfc4": 1.0,
            "pfc5": 1.0,
        },
        "sol": {
            "sol0": 1.0,
            "sol1": 1.0,
            "sol2": 1.0,
        },
    }
    bad_delta_limits = tmp_path / "bad_delta_limits.json"
    bad_delta_limits.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="delta_derivative_limits_aps"):
        load_experiment_config(bad_delta_limits)

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

    data = json.loads(CONFIG.read_text())
    data["reward"]["terminal_remaining_cost"] = -1.0
    bad_terminal_remaining = tmp_path / "bad_terminal_remaining.json"
    bad_terminal_remaining.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal_remaining_cost"):
        load_experiment_config(bad_terminal_remaining)

    data = json.loads(CONFIG.read_text())
    data["reward"]["actuator_saturation_weight"] = -1.0
    bad_saturation_weight = tmp_path / "bad_saturation_weight.json"
    bad_saturation_weight.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="actuator_saturation_weight"):
        load_experiment_config(bad_saturation_weight)

    data = json.loads(CONFIG.read_text())
    data["reward"]["current_usage_weight"] = -1.0
    bad_current_usage_weight = tmp_path / "bad_current_usage_weight.json"
    bad_current_usage_weight.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="current_usage_weight"):
        load_experiment_config(bad_current_usage_weight)

    data = json.loads(CONFIG.read_text())
    data["reward"]["derivative_usage_weight"] = -1.0
    bad_derivative_usage_weight = tmp_path / "bad_derivative_usage_weight.json"
    bad_derivative_usage_weight.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="derivative_usage_weight"):
        load_experiment_config(bad_derivative_usage_weight)

    data = json.loads(CONFIG.read_text())
    data["sim"]["current_saturation_fraction"] = 0.99
    bad_current_saturation = tmp_path / "bad_current_saturation.json"
    bad_current_saturation.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="current_saturation_fraction"):
        load_experiment_config(bad_current_saturation)

    data = json.loads(CONFIG.read_text())
    data["reward"]["late_error_weight"] = 1.0
    bad_late_weight = tmp_path / "bad_late_weight.json"
    bad_late_weight.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stale keys.*late_error_weight"):
        load_experiment_config(bad_late_weight)

    data = json.loads(CONFIG.read_text())
    data["reward"]["projection_bad"] = 0.05
    bad_projection_bad = tmp_path / "bad_projection_bad.json"
    bad_projection_bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stale keys.*projection_bad"):
        load_experiment_config(bad_projection_bad)

    data = json.loads(CONFIG.read_text())
    data["reward"]["projection_weight"] = 1.0
    bad_projection_weight = tmp_path / "bad_projection_weight.json"
    bad_projection_weight.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stale keys.*projection_weight"):
        load_experiment_config(bad_projection_weight)

    data = json.loads(CONFIG.read_text())
    data["reference"]["ip"]["kind"] = "teleport"
    bad_ip_kind = tmp_path / "bad_ip_kind.json"
    bad_ip_kind.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="reference.ip.kind"):
        load_experiment_config(bad_ip_kind)

    data = json.loads(CONFIG.read_text())
    data["sim"]["project_actions_to_current_limits"] = True
    bad_projection = tmp_path / "bad_projection.json"
    bad_projection.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stale keys.*project_actions_to_current_limits"):
        load_experiment_config(bad_projection)


def test_policy_pipeline_gates_require_learning_signals() -> None:
    actor_eval = {
        "boundary_found": 1.0,
        "boundary_found_late_min": 1.0,
        "current_over_limit_a": 0.0,
        "current_over_limit_a_max": 0.0,
        "current_over_limit_a_late_max": 0.0,
        "shape_error_mean_m": 0.02,
        "shape_error_mean_m_late": 0.02,
        "ip_error_a": 70000.0,
        "ip_error_a_late": 70000.0,
        "action_rms": 0.02,
        "mean_episode_completion": 1.0,
        "min_episode_completion": 1.0,
        "mean_episode_steps": 500.0,
        "min_episode_steps": 500.0,
    }
    no_control = {"ip_error_a": 100000.0, "ip_error_a_late": 100000.0}
    tail_losses = {"tail100.policy_weight_max": 0.06, "tail100.sampled_q_spread": 1.0e-4}
    controller_rollout = {
        "status": "ok",
        "mean_episode_completion": 1.0,
        "min_episode_completion": 1.0,
        "boundary_found_mean": 1.0,
        "boundary_found_late_min": 1.0,
        "current_over_limit_a_max": 0.0,
        "current_over_limit_a_late_max": 0.0,
        "shape_error_mean_m": 0.02,
        "shape_error_late_m": 0.02,
        "ip_error_a": 30000.0,
        "ip_error_late_a": 30000.0,
    }
    gate_kwargs = {
        "action_samples": 20,
        "min_boundary_found": 0.999,
        "max_current_over_limit_a": 0.0,
        "max_shape_error_m": 0.03,
        "min_ip_improvement_frac": 0.25,
        "min_ip_improvement_a": 20000.0,
        "max_ip_error_a": None,
        "max_ip_error_late_a": None,
        "min_action_rms": 0.005,
        "max_action_rms": 0.5,
        "min_mean_episode_completion": 0.95,
        "min_episode_completion": 0.95,
        "min_baseline_ip_error_a": 0.0,
        "min_baseline_ip_error_late_a": 0.0,
        "always_require_ip_improvement": False,
        "min_policy_weight_extra": 1.0e-4,
        "min_sampled_q_spread": 1.0e-8,
        "include_mpo_gates": True,
        "require_controller_rollout": True,
        "controller_rollout": controller_rollout,
        "max_controller_shape_error_m": 0.03,
        "max_controller_ip_error_a": 40000.0,
    }
    gates = evaluate_policy_gates(
        actor_eval=actor_eval,
        no_control=no_control,
        tail_losses=tail_losses,
        **gate_kwargs,
    )
    assert gates["passed"] is True

    stalled = evaluate_policy_gates(
        actor_eval=dict(actor_eval, action_rms=0.0),
        no_control=no_control,
        tail_losses={"tail100.policy_weight_max": 0.05, "tail100.sampled_q_spread": 0.0},
        **gate_kwargs,
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
        **gate_kwargs,
    )
    spike_reasons = {check["name"]: check["passed"] for check in current_spike["checks"]}
    assert spike_reasons["current_limit"] is False

    late_drift = evaluate_policy_gates(
        actor_eval=dict(actor_eval, shape_error_mean_m_late=0.05, current_over_limit_a_late_max=0.0, boundary_found_late_min=1.0, ip_error_a_late=90000.0),
        no_control=dict(no_control, ip_error_a_late=100000.0),
        tail_losses=tail_losses,
        **gate_kwargs,
    )
    late_reasons = {check["name"]: check["passed"] for check in late_drift["checks"]}
    assert late_reasons["shape_error_late"] is False
    assert late_reasons["ip_improvement_late"] is False

    fake_short_eval = evaluate_policy_gates(
        actor_eval=dict(actor_eval, mean_episode_completion=0.003, min_episode_completion=0.002),
        no_control=no_control,
        tail_losses=tail_losses,
        **gate_kwargs,
    )
    fake_reasons = {check["name"]: check["passed"] for check in fake_short_eval["checks"]}
    assert fake_short_eval["passed"] is False
    assert fake_reasons["episode_completion"] is False

    bad_controller = evaluate_policy_gates(
        actor_eval=actor_eval,
        no_control=no_control,
        tail_losses=tail_losses,
        **{**gate_kwargs, "controller_rollout": dict(controller_rollout, shape_error_mean_m=0.12, shape_error_late_m=0.21, ip_error_a=100000.0, ip_error_late_a=130000.0)},
    )
    controller_reasons = {check["name"]: check["passed"] for check in bad_controller["checks"]}
    assert bad_controller["passed"] is False
    assert controller_reasons["controller_shape_error"] is False
    assert controller_reasons["controller_ip_error"] is False


def test_policy_pipeline_shot_family_gates_reject_easy_tasks_and_missing_ip_gain() -> None:
    actor_eval = {
        "boundary_found": 1.0,
        "boundary_found_late_min": 1.0,
        "current_over_limit_a": 0.0,
        "current_over_limit_a_max": 0.0,
        "current_over_limit_a_late_max": 0.0,
        "shape_error_mean_m": 0.02,
        "shape_error_mean_m_late": 0.02,
        "ip_error_a": 82000.0,
        "ip_error_a_late": 90000.0,
        "action_rms": 0.02,
        "mean_episode_completion": 0.96,
        "min_episode_completion": 0.91,
    }
    no_control = {"ip_error_a": 90000.0, "ip_error_a_late": 95000.0}
    tail_losses = {"tail100.policy_weight_max": 0.06, "tail100.sampled_q_spread": 1.0e-4}
    controller_rollout = {
        "status": "ok",
        "boundary_found_mean": 1.0,
        "current_over_limit_a_max": 0.0,
        "shape_error_mean_m": 0.02,
        "shape_error_late_m": 0.02,
        "ip_error_a": 24000.0,
        "ip_error_late_a": 24000.0,
    }
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
        max_ip_error_a=25000.0,
        max_ip_error_late_a=25000.0,
        min_action_rms=0.005,
        max_action_rms=0.5,
        min_mean_episode_completion=0.95,
        min_episode_completion=0.90,
        min_baseline_ip_error_a=100000.0,
        min_baseline_ip_error_late_a=100000.0,
        always_require_ip_improvement=True,
        min_policy_weight_extra=1.0e-4,
        min_sampled_q_spread=1.0e-8,
        include_mpo_gates=True,
        require_controller_rollout=True,
        controller_rollout=controller_rollout,
        max_controller_shape_error_m=0.03,
        max_controller_ip_error_a=25000.0,
    )
    reasons = {check["name"]: check["passed"] for check in gates["checks"]}
    assert reasons["task_difficulty"] is False
    assert reasons["ip_error_mean"] is False
    assert reasons["ip_error_late"] is False
    assert reasons["ip_improvement"] is False
    assert reasons["ip_improvement_late"] is False


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
    cfg = replace(cfg, training=replace(cfg.training, steps=4, output_dir=first_dir, save_checkpoints=True, checkpoint_interval_steps=4, eval_interval_steps=1000))
    first = Trainer(cfg, device="cpu", output_dir=first_dir).train()
    checkpoint = first_dir / "checkpoints" / "final.pt"
    assert checkpoint.exists()
    second_dir = tmp_path / "second"
    resumed_cfg = replace(cfg, training=replace(cfg.training, steps=6, output_dir=second_dir, save_checkpoints=True, checkpoint_interval_steps=6, eval_interval_steps=1000))
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

    state["critic_action_input_kind"] = "normalized_action_v1"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="critic action input"):
        resumed._load_checkpoint(checkpoint)

    state["critic_action_input_kind"] = "requested_delta_jdot_v1"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="critic action input"):
        resumed._load_checkpoint(checkpoint)


def test_training_checkpoint_resume_rejects_missing_reference_fragment(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    cfg = _small_config(first_dir)
    trainer = Trainer(cfg, device="cpu", output_dir=first_dir)
    trainer.env.reset()
    checkpoint = trainer._save_checkpoint("missing_reference.pt", step=0, updates=0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state.pop("reference", None)
    torch.save(state, checkpoint)

    resumed = Trainer(cfg, device="cpu", output_dir=tmp_path / "resume", resume_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="reference config mismatch"):
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


def test_eval_checkpoint_retention_keeps_top_k_and_milestones(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    cfg = replace(
        cfg,
        training=replace(
            cfg.training,
            save_checkpoints=True,
            eval_checkpoint_top_k=2,
            milestone_checkpoint_interval_steps=4,
        ),
    )
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)
    trainer.env.reset()

    for step, score in ((2, 0.1), (4, -10.0), (6, 0.2), (8, -20.0), (10, 0.3)):
        trainer._save_retained_eval_checkpoint(
            step=step,
            updates=step,
            score=score,
            eval_metrics={"selection_score": score, "mean_episode_completion": 1.0},
        )

    ckpt_dir = tmp_path / "checkpoints"
    manifest = json.loads((ckpt_dir / "eval_checkpoints.json").read_text())
    kept = {entry["path"] for entry in manifest["checkpoints"]}
    assert kept == {
        "eval_step_000000000004.pt",
        "eval_step_000000000006.pt",
        "eval_step_000000000008.pt",
        "eval_step_000000000010.pt",
    }
    assert not (ckpt_dir / "eval_step_000000000002.pt").exists()
    assert (ckpt_dir / "eval_step_000000000004.pt").exists()
    assert (ckpt_dir / "eval_step_000000000008.pt").exists()

    best = ckpt_dir / "best.pt"
    assert best.exists() or best.is_symlink()
    state = torch.load(best, map_location="cpu", weights_only=False)
    assert state["training_state"]["step"] == 10
    assert state["metadata"]["eval_score"] == 0.3


def test_local_replay_replay_health_waits_for_episode_horizon(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    cfg = replace(cfg, sim=replace(cfg.sim, max_episode_steps=500))
    cfg = replace(cfg, learner=replace(cfg.learner, rollout_chunk_length=64))
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)

    assert trainer._min_replay_health_check_chunks() == 9


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
    cfg = replace(cfg, training=replace(cfg.training, save_checkpoints=True))
    Trainer(cfg, device="cpu", output_dir=train_dir).train()
    export_dir = tmp_path / "manual_export"
    assert export_cli_main(["--checkpoint", str(train_dir / "checkpoints" / "final.pt"), "--out", str(export_dir)]) == 0
    assert (export_dir / "actor.pt").exists()
    assert (export_dir / "policy_weights.npz").exists()
    schema = json.loads((export_dir / "controller_schema.json").read_text())
    assert schema["observation_kind"] == "controller_state_v3"


def test_distributed_resume_fails_clearly_because_worker_envs_are_not_checkpointed(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    cfg = _small_config(first_dir)
    cfg = replace(cfg, training=replace(cfg.training, steps=4, output_dir=first_dir, save_checkpoints=True, checkpoint_interval_steps=4, eval_interval_steps=1000))
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
    cfg = replace(cfg, sim=replace(cfg.sim, terminate_on_current_limit=False, terminate_on_boundary_loss=False, max_episode_steps=4))
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)
    metrics = trainer.evaluate_detailed(episodes=2, max_steps=4, policy="no_control", seed_offset=123456)
    repeat = trainer.evaluate_detailed(episodes=2, max_steps=4, policy="no_control", seed_offset=123456)
    holdout = trainer.evaluate_detailed(episodes=2, max_steps=4, policy="no_control", seed_offset=123457)
    assert "mean_return" in metrics
    assert "mean_episode_steps" in metrics
    assert "min_episode_steps" in metrics
    assert "mean_episode_completion" in metrics
    assert "min_episode_completion" in metrics
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
    assert "physical_cost_late" in metrics
    assert "shape_mean_loss" in metrics
    assert "shape_max_loss" in metrics
    assert "current_loss" in metrics
    assert "shape_error_mean_m_late" in metrics
    assert "shape_error_mean_m_late_minus_early" in metrics
    assert "ip_error_a_late" in metrics
    assert "boundary_found" in metrics
    assert "boundary_found_late_min" in metrics
    assert "boundary_found_min" in metrics
    assert "full_episode_success" in metrics
    assert "termination_failure_fraction" in metrics
    assert "padded_shape_error_mean_m_late" in metrics
    assert "padded_shape_error_max_m_late" in metrics
    assert "padded_ip_error_a_late" in metrics
    assert "padded_boundary_found_late_min" in metrics
    assert "padded_current_over_limit_a_late_max" in metrics
    assert "padded_current_over_limit_fraction_late" in metrics
    assert metrics["full_episode_success"] == pytest.approx(1.0)
    assert metrics["termination_failure_fraction"] == pytest.approx(0.0)
    assert metrics["padded_shape_error_mean_m_late"] == pytest.approx(metrics["shape_error_mean_m_late"])
    assert metrics["padded_ip_error_a_late"] == pytest.approx(metrics["ip_error_a_late"])
    assert metrics["padded_boundary_found_late_min"] == pytest.approx(metrics["boundary_found_late_min"])
    assert np.isfinite(metrics["mean_return"])
    assert 0.0 < metrics["mean_episode_completion"] <= 1.0
    assert repeat["mean_return"] == pytest.approx(metrics["mean_return"])
    assert np.isfinite(holdout["mean_return"])


def test_evaluate_detailed_failure_padding_marks_short_termination_bad(tmp_path: Path) -> None:
    cfg = _small_config(tmp_path)
    cfg = replace(
        cfg,
        sim=replace(
            cfg.sim,
            max_episode_steps=8,
            terminate_on_current_limit=True,
            terminate_on_boundary_loss=False,
            current_termination_over_limit_a=-1.0,
            current_termination_grace_steps=1,
            current_hard_termination_fraction=1.01,
        ),
    )
    trainer = Trainer(cfg, device="cpu", output_dir=tmp_path)
    metrics = trainer.evaluate_detailed(episodes=2, max_steps=4, policy="no_control", seed_offset=123456)
    assert metrics["mean_episode_completion"] < 1.0
    assert metrics["full_episode_success"] == pytest.approx(0.0)
    assert metrics["termination_failure_fraction"] == pytest.approx(1.0)
    assert metrics["padded_boundary_found_late_min"] == pytest.approx(0.0)
    assert metrics["padded_shape_error_mean_m_late"] >= cfg.reward.boundary_missing_error_m
    assert metrics["padded_shape_error_max_m_late"] >= cfg.reward.boundary_missing_error_m
    assert metrics["padded_ip_error_a_late"] >= max(100000.0, 4.0 * cfg.reward.ip_scale_a)


def test_selection_score_prefers_survival_over_short_low_cost() -> None:
    short_policy = {
        "physical_cost_late": 0.10,
        "current_over_limit_a_late_max": 0.0,
        "boundary_found_late_min": 1.0,
        "min_episode_completion": 0.20,
        "mean_episode_completion": 0.25,
    }
    surviving_policy = {
        "physical_cost_late": 0.90,
        "current_over_limit_a_late_max": 0.0,
        "boundary_found_late_min": 1.0,
        "min_episode_completion": 1.0,
        "mean_episode_completion": 1.0,
    }
    assert Trainer._selection_score(surviving_policy) > Trainer._selection_score(short_policy)


def test_append_csv_row_preserves_header_when_metric_sets_change(tmp_path: Path) -> None:
    path = tmp_path / "eval_history.csv"
    _append_csv_row(path, {"step": 1, "mean_return": -1.0, "late_metric": 3.0, "selection_score": -10.0})
    _append_csv_row(path, {"step": 2, "mean_return": -2.0, "selection_score": -20.0})
    _append_csv_row(path, {"step": 3, "mean_return": -3.0, "new_metric": 7.0, "selection_score": -30.0})

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert [row["step"] for row in rows] == ["1", "2", "3"]
    assert [row["selection_score"] for row in rows] == ["-10.0", "-20.0", "-30.0"]
    assert rows[1]["late_metric"] == ""
    assert rows[0]["new_metric"] == ""
    assert rows[2]["new_metric"] == "7.0"


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
    batch = _sequence_batch(
        obs=torch.randn((2, 2, obs_dim)),
        action=torch.zeros((2, 2, action_dim)),
        reward=torch.zeros((2, 2)),
        discount=torch.full((2, 2), 0.99),
        next_obs=torch.randn((2, 2, obs_dim)),
        done=torch.zeros((2, 2), dtype=torch.bool),
        mask=torch.ones((2, 2)),
    )

    def uniform_q(obs, history_action, sampled_actions, *, mask=None):
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
    batch = _sequence_batch(
        obs=torch.randn((2, 2, obs_dim)),
        action=torch.zeros((2, 2, action_dim)),
        reward=torch.zeros((2, 2)),
        discount=torch.full((2, 2), 0.99),
        next_obs=torch.randn((2, 2, obs_dim)),
        done=torch.zeros((2, 2), dtype=torch.bool),
        mask=torch.ones((2, 2)),
    )

    def ranked_q(obs, history_action, sampled_actions, *, mask=None):
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


def test_mpo_update_accepts_padded_short_terminal_sequence() -> None:
    torch.manual_seed(781)
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
        config=LearnerConfig(batch_size=2, unroll_length=6, action_samples=4),
        device="cpu",
    )
    batch = _sequence_batch(
        obs=torch.randn((2, 6, obs_dim)),
        action=torch.zeros((2, 6, action_dim)),
        reward=torch.randn((2, 6)),
        discount=torch.full((2, 6), 0.99),
        next_obs=torch.randn((2, 6, obs_dim)),
        done=torch.tensor([[False, False, True, True, True, True], [False, True, True, True, True, True]], dtype=torch.bool),
        mask=torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32),
    )
    metrics = learner.update(batch)
    assert np.isfinite(metrics.critic_loss)
    assert np.isfinite(metrics.actor_loss)
    assert np.isfinite(metrics.sampled_q_spread)


def test_mpo_actor_update_honors_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(779)
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
        config=LearnerConfig(batch_size=2, unroll_length=3, action_samples=4, actor_update_chunk_size=2),
        device="cpu",
    )
    batch = _sequence_batch(
        obs=torch.randn((2, 3, obs_dim)),
        action=torch.zeros((2, 3, action_dim)),
        reward=torch.zeros((2, 3)),
        discount=torch.full((2, 3), 0.99),
        next_obs=torch.randn((2, 3, obs_dim)),
        done=torch.zeros((2, 3), dtype=torch.bool),
        mask=torch.ones((2, 3)),
    )

    def ranked_q(obs, history_action, sampled_actions, *, mask=None):
        ranks = torch.linspace(0.0, 0.05, sampled_actions.shape[0], dtype=obs.dtype, device=obs.device)
        return ranks[:, None, None].expand(-1, obs.shape[0], obs.shape[1])

    calls: list[int] = []
    original_forward = actor.forward

    def tracked_forward(x):
        calls.append(int(x.shape[0]))
        return original_forward(x)

    learner._sampled_q_values = ranked_q  # type: ignore[method-assign]
    monkeypatch.setattr(actor, "forward", tracked_forward)
    learner._actor_update(batch)
    assert calls == [2, 2, 2]



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
    history_action = torch.tanh(torch.randn((2, 3, action_dim), dtype=torch.float32))
    sampled = history_action.unsqueeze(0).repeat(4, 1, 1, 1)
    mask = torch.ones((2, 3), dtype=torch.float32)
    chunked = learner._sampled_q_values(obs, history_action, sampled, mask=mask)
    reference, _ = critic(obs, history_action, mask=mask)
    assert torch.allclose(chunked, reference.unsqueeze(0).repeat(4, 1, 1), atol=1.0e-6)


def test_sampled_q_values_use_replay_history_not_candidate_history() -> None:
    torch.manual_seed(124)
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
        config=LearnerConfig(batch_size=2, unroll_length=3, action_samples=2),
        device="cpu",
    )
    obs = torch.randn((2, 3, obs_dim), dtype=torch.float32)
    history_action = torch.tanh(torch.randn((2, 3, action_dim), dtype=torch.float32))
    sampled = torch.zeros((2, 2, 3, action_dim), dtype=torch.float32)
    sampled[:, :, 0] = torch.tanh(torch.randn((2, 2, action_dim), dtype=torch.float32))
    sampled[:, :, 1:] = 0.25
    mask = torch.ones((2, 3), dtype=torch.float32)

    q = learner._sampled_q_values(obs, history_action, sampled, mask=mask)
    assert torch.allclose(q[0, :, 1:], q[1, :, 1:], atol=1.0e-6)


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
    batch = _sequence_batch(
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
