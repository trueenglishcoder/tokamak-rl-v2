from __future__ import annotations

import torch

from tokamak_rl_v2.config.schema import RewardConfig
from tokamak_rl_v2.rewards.physical import T15TCVDerivativeReward


def _call_reward(current_usage: float, derivative_usage: float):
    cfg = RewardConfig(
        kind="tcv_derivative",
        current_margin_weight=1.0,
        current_margin_good_fraction=0.70,
        current_margin_bad_fraction=0.90,
        derivative_margin_weight=1.0,
        derivative_margin_good_fraction=0.25,
        derivative_margin_bad_fraction=0.50,
        smoothmax_alpha=-5.0,
    )
    reward = T15TCVDerivativeReward(cfg, control_rate_hz=1000.0)
    batch = reward(
        ip=torch.tensor([200000.0]),
        ip_ref=torch.tensor([200000.0]),
        boundary_points=torch.zeros((1, 32, 2)),
        reference_points=torch.zeros((1, 32, 2)),
        action=torch.zeros((1, 9)),
        previous_action=torch.zeros((1, 9)),
        current_over_limit_a=torch.tensor([0.0]),
        current_usage_fraction=torch.tensor([current_usage]),
        current_margin_fraction=torch.tensor([1.0 - current_usage]),
        derivative_usage=torch.tensor([derivative_usage]),
        boundary_found=torch.tensor([True]),
        terminated=torch.tensor([False]),
    )
    return batch.components


def test_safety_margin_components_are_zero_in_good_region() -> None:
    comps = _call_reward(current_usage=0.50, derivative_usage=0.10)
    assert float(comps["current_margin_loss"][0]) == 0.0
    assert float(comps["derivative_margin_loss"][0]) == 0.0


def test_safety_margin_components_activate_before_hard_limit() -> None:
    comps = _call_reward(current_usage=0.80, derivative_usage=0.40)
    assert 0.0 < float(comps["current_margin_loss"][0]) < 1.0
    assert 0.0 < float(comps["derivative_margin_loss"][0]) < 1.0


def test_safety_margin_components_cap_in_bad_region() -> None:
    comps = _call_reward(current_usage=1.10, derivative_usage=0.80)
    assert float(comps["current_margin_loss"][0]) == 1.0
    assert float(comps["derivative_margin_loss"][0]) == 1.0


def test_tcv_derivative_reward_uses_mean_usage_weights() -> None:
    cfg = RewardConfig(
        kind="tcv_derivative",
        current_usage_weight=2.0,
        derivative_usage_weight=2.0,
        current_bad_fraction=1.0,
        derivative_margin_bad_fraction=1.0,
        smoothmax_alpha=-5.0,
        reward_scale=1.0,
    )
    reward = T15TCVDerivativeReward(cfg, control_rate_hz=1000.0)

    def call(mean_current: float, mean_derivative: float):
        return reward(
            ip=torch.tensor([200000.0]),
            ip_ref=torch.tensor([200000.0]),
            boundary_points=torch.zeros((1, 32, 2)),
            reference_points=torch.zeros((1, 32, 2)),
            action=torch.zeros((1, 9)),
            previous_action=torch.zeros((1, 9)),
            current_over_limit_a=torch.tensor([0.0]),
            current_usage_fraction=torch.tensor([0.20]),
            current_usage_mean_fraction=torch.tensor([mean_current]),
            current_margin_fraction=torch.tensor([0.80]),
            derivative_usage=torch.tensor([0.20]),
            derivative_usage_mean_fraction=torch.tensor([mean_derivative]),
            boundary_found=torch.tensor([True]),
            terminated=torch.tensor([False]),
        )

    low = call(0.10, 0.10)
    high = call(0.80, 0.80)
    assert float(low.components["current_usage_mean_loss"][0]) < float(high.components["current_usage_mean_loss"][0])
    assert float(low.components["derivative_usage_mean_loss"][0]) < float(high.components["derivative_usage_mean_loss"][0])
    assert float(low.reward[0]) > float(high.reward[0])
