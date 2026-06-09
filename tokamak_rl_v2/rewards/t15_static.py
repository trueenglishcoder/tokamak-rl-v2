from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tokamak_rl_v2.config.schema import RewardConfig
from tokamak_rl_v2.rewards import combiners, transforms


@dataclass(frozen=True, slots=True)
class RewardBatch:
    reward: Tensor
    components: dict[str, Tensor]


class T15StaticBoundaryReward:
    """TCV reward-transform recipe adapted to T15 static-boundary control."""

    def __init__(self, config: RewardConfig, *, control_rate_hz: float) -> None:
        self.config = config
        self.control_rate_hz = float(control_rate_hz)

    def __call__(
        self,
        *,
        ip: Tensor,
        ip_ref: Tensor,
        boundary_points: Tensor,
        reference_points: Tensor,
        action: Tensor,
        previous_action: Tensor,
        current_over_limit_a: Tensor,
        derivative_margin: Tensor,
        boundary_found: Tensor,
        terminated: Tensor,
    ) -> RewardBatch:
        c = self.config
        shape_error = torch.linalg.norm(boundary_points - reference_points, dim=-1)
        shape_quality_points = transforms.softplus(shape_error, good=c.shape_good_m, bad=c.shape_bad_m)
        r_shape = combiners.smooth_max(shape_quality_points, alpha=-1.0, dim=-1)
        ip_error = torch.abs(ip - ip_ref)
        r_ip = transforms.softplus(ip_error, good=c.ip_good_a, bad=c.ip_bad_a)
        current_error = torch.clamp(current_over_limit_a, min=0.0)
        r_current = transforms.softplus(current_error, good=c.current_good_a, bad=max(c.current_bad_a, c.current_good_a + 1.0))
        delta_action = action - previous_action
        action_mag = torch.sqrt(torch.mean(action.pow(2), dim=-1))
        delta_mag = torch.sqrt(torch.mean(delta_action.pow(2), dim=-1))
        r_action = transforms.softplus(action_mag, good=0.0, bad=1.0)
        r_delta = transforms.softplus(delta_mag, good=0.0, bad=1.0)
        qualities = torch.stack([r_shape, r_ip, r_current, r_action, r_delta], dim=-1)
        combined = combiners.smooth_max(qualities, alpha=-0.5, dim=-1)
        combined = torch.where(boundary_found, combined, torch.zeros_like(combined))
        reward = combined * float(c.reward_scale)
        terminal = torch.full_like(reward, float(c.terminal_reward) * float(c.reward_scale))
        reward = torch.where(terminated, terminal, reward)
        return RewardBatch(
            reward=reward,
            components={
                "shape_error_mean_m": torch.mean(shape_error, dim=-1),
                "shape_error_max_m": torch.max(shape_error, dim=-1).values,
                "shape_quality": r_shape,
                "ip_error_a": ip_error,
                "ip_quality": r_ip,
                "current_over_limit_a": current_error,
                "current_quality": r_current,
                "action_quality": r_action,
                "delta_action_quality": r_delta,
                "combined_quality": combined,
                "boundary_found": boundary_found.to(dtype=reward.dtype),
            },
        )
