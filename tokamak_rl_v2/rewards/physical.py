from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tokamak_rl_v2.config.schema import RewardConfig


@dataclass(frozen=True, slots=True)
class RewardBatch:
    reward: Tensor
    components: dict[str, Tensor]


class T15PhysicalReward:
    """Dense negative physical-cost reward for simulator-only T15 RL."""

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
        current_usage_fraction: Tensor,
        current_margin_fraction: Tensor,
        derivative_usage: Tensor,
        boundary_found: Tensor,
        terminated: Tensor,
        episode_progress: Tensor | None = None,
    ) -> RewardBatch:
        c = self.config

        found = boundary_found.reshape(-1, 1).to(dtype=torch.bool)
        raw_shape_error = torch.linalg.norm(boundary_points - reference_points, dim=-1)
        finite_missing = torch.full_like(raw_shape_error, float(c.boundary_missing_error_m))
        shape_error = torch.nan_to_num(
            raw_shape_error,
            nan=float(c.boundary_missing_error_m),
            posinf=float(c.boundary_missing_error_m),
            neginf=float(c.boundary_missing_error_m),
        )
        shape_error = torch.where(found, shape_error, finite_missing)
        shape_error_mean = torch.mean(shape_error, dim=-1)
        shape_error_max = torch.max(shape_error, dim=-1).values

        ip_error = torch.abs(ip - ip_ref)
        delta_action = action - previous_action
        action_rms = torch.sqrt(torch.mean(action.pow(2), dim=-1))
        delta_action_rms = torch.sqrt(torch.mean(delta_action.pow(2), dim=-1))
        max_abs_action = torch.max(torch.abs(action), dim=-1).values

        shape_mean_loss = _huber(shape_error_mean / max(float(c.shape_mean_scale_m), 1.0e-12))
        shape_max_loss = _huber(shape_error_max / max(float(c.shape_max_scale_m), 1.0e-12))
        ip_loss = _huber(ip_error / max(float(c.ip_scale_a), 1.0e-12))
        current_loss = _threshold_square(current_usage_fraction, start=float(c.current_soft_fraction), bad=1.0)
        derivative_loss = _threshold_square(derivative_usage, start=float(c.derivative_soft_fraction), bad=1.0)
        action_loss = torch.mean(action.pow(2), dim=-1)
        delta_action_loss = torch.mean(delta_action.pow(2), dim=-1)

        physical_cost = (
            float(c.shape_mean_weight) * shape_mean_loss
            + float(c.shape_max_weight) * shape_max_loss
            + float(c.ip_weight) * ip_loss
            + float(c.current_weight) * current_loss
            + float(c.derivative_weight) * derivative_loss
            + float(c.action_weight) * action_loss
            + float(c.delta_action_weight) * delta_action_loss
        )
        reward = -float(c.reward_scale) * physical_cost
        reward = torch.where(
            terminated,
            reward + torch.full_like(reward, float(c.terminal_reward) * float(c.reward_scale)),
            reward,
        )
        if episode_progress is None:
            progress = torch.zeros_like(reward)
        else:
            progress = torch.clamp(episode_progress.to(dtype=reward.dtype, device=reward.device).reshape_as(reward), 0.0, 1.0)

        return RewardBatch(
            reward=reward,
            components={
                "shape_error_mean_m": shape_error_mean,
                "shape_error_max_m": shape_error_max,
                "ip_error_a": ip_error,
                "current_over_limit_a": torch.clamp(current_over_limit_a, min=0.0),
                "current_usage_fraction": torch.clamp(current_usage_fraction, min=0.0),
                "current_margin_fraction": current_margin_fraction,
                "derivative_usage": torch.clamp(derivative_usage, min=0.0),
                "max_abs_action": max_abs_action,
                "action_rms": action_rms,
                "delta_action_rms": delta_action_rms,
                "episode_progress": progress,
                "physical_cost": physical_cost,
                "shape_mean_loss": shape_mean_loss,
                "shape_max_loss": shape_max_loss,
                "ip_loss": ip_loss,
                "current_loss": current_loss,
                "derivative_loss": derivative_loss,
                "action_loss": action_loss,
                "delta_action_loss": delta_action_loss,
                "boundary_found": boundary_found.to(dtype=reward.dtype),
            },
        )


def _huber(x: Tensor, *, delta: float = 1.0) -> Tensor:
    z = torch.abs(x)
    d = float(delta)
    return torch.where(z <= d, 0.5 * z.pow(2), d * (z - 0.5 * d))


def _threshold_square(x: Tensor, *, start: float, bad: float) -> Tensor:
    width = max(float(bad) - float(start), 1.0e-12)
    z = torch.clamp((x - float(start)) / width, min=0.0)
    return z.pow(2)
