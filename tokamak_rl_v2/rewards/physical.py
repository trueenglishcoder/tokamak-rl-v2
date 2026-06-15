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
    """Single physical-cost reward for T15 magnetic-control training."""

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

        boundary_mask = boundary_found.reshape(-1, 1)
        raw_shape_error = torch.linalg.norm(boundary_points - reference_points, dim=-1)
        missing_shape_error = float(c.boundary_missing_error_m) if float(c.boundary_missing_error_m) > 0.0 else float(c.shape_bad_m)
        finite_bad_shape = torch.full_like(raw_shape_error, missing_shape_error)
        shape_error = torch.nan_to_num(raw_shape_error, nan=float(c.shape_bad_m), posinf=float(c.shape_bad_m), neginf=float(c.shape_bad_m))
        shape_error = torch.where(boundary_mask, shape_error, finite_bad_shape)
        shape_error_mean = torch.mean(shape_error, dim=-1)
        shape_error_max = torch.max(shape_error, dim=-1).values

        ip_error = torch.abs(ip - ip_ref)
        current_error = torch.clamp(current_over_limit_a, min=0.0)
        delta_action = action - previous_action
        action_rms = torch.sqrt(torch.mean(action.pow(2), dim=-1))
        delta_action_rms = torch.sqrt(torch.mean(delta_action.pow(2), dim=-1))
        max_abs_action = torch.max(torch.abs(action), dim=-1).values

        shape_loss = _huber01(shape_error_mean / max(float(c.shape_bad_m), 1.0e-12)) + 0.25 * _huber01(shape_error_max / max(float(c.shape_max_bad_m), 1.0e-12))
        ip_loss = _huber01(ip_error / max(float(c.ip_bad_a), 1.0e-12))

        physical_cost = float(c.shape_weight) * shape_loss + float(c.ip_weight) * ip_loss
        if episode_progress is None:
            progress = torch.zeros_like(physical_cost)
        else:
            progress = torch.clamp(episode_progress.to(dtype=physical_cost.dtype, device=physical_cost.device).reshape_as(physical_cost), 0.0, 1.0)
        reward = -float(c.reward_scale) * physical_cost
        terminal = torch.full_like(reward, float(c.terminal_reward) * float(c.reward_scale))
        reward = torch.where(terminated, reward + terminal, reward)

        return RewardBatch(
            reward=reward,
            components={
                "shape_error_mean_m": shape_error_mean,
                "shape_error_max_m": shape_error_max,
                "ip_error_a": ip_error,
                "current_over_limit_a": current_error,
                "current_usage_fraction": torch.clamp(current_usage_fraction, min=0.0),
                "current_margin_fraction": current_margin_fraction,
                "derivative_usage": torch.clamp(derivative_usage, min=0.0),
                "max_abs_action": max_abs_action,
                "action_rms": action_rms,
                "delta_action_rms": delta_action_rms,
                "episode_progress": progress,
                "base_physical_cost": physical_cost,
                "physical_cost": physical_cost,
                "shape_loss": shape_loss,
                "ip_loss": ip_loss,
                "boundary_found": boundary_found.to(dtype=reward.dtype),
            },
        )


def _huber01(x: Tensor) -> Tensor:
    x = torch.clamp(x, min=0.0)
    return torch.where(x <= 1.0, 0.5 * x.pow(2), x - 0.5)
