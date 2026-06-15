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
    """TCV-style adapted quality reward for T15 magnetic-control training."""

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

        point_quality = _quality_between(raw_shape_error, good=float(c.shape_good_m), bad=float(c.shape_bad_m))
        missing_quality = torch.zeros_like(point_quality)
        point_quality = torch.where(boundary_mask, point_quality, missing_quality)
        shape_loss_points = 1.0 - point_quality
        shape_loss = _smooth_max(shape_loss_points, sharpness=10.0)
        shape_quality = torch.clamp(1.0 - shape_loss, 0.0, 1.0)

        ip_quality = _quality_between(ip_error, good=float(c.ip_good_a), bad=float(c.ip_bad_a))
        ip_loss = 1.0 - ip_quality

        current_quality = _quality_between(current_usage_fraction, good=float(c.current_good_fraction), bad=float(c.current_bad_fraction))
        current_limit_loss = 1.0 - current_quality

        combined_quality = _weighted_geometric_mean(
            (shape_quality, ip_quality, current_quality),
            (float(c.shape_weight), float(c.ip_weight), float(c.current_weight)),
        )
        physical_cost = 1.0 - combined_quality
        if episode_progress is None:
            progress = torch.zeros_like(physical_cost)
        else:
            progress = torch.clamp(episode_progress.to(dtype=physical_cost.dtype, device=physical_cost.device).reshape_as(physical_cost), 0.0, 1.0)
        max_step_reward = float(c.max_episode_reward) / max(self.control_rate_hz, 1.0e-12)
        reward = float(c.reward_scale) * max_step_reward * combined_quality
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
                "current_limit_loss": current_limit_loss,
                "shape_quality": shape_quality,
                "ip_quality": ip_quality,
                "current_quality": current_quality,
                "combined_quality": combined_quality,
                "boundary_found": boundary_found.to(dtype=reward.dtype),
            },
        )


def _quality_between(x: Tensor, *, good: float, bad: float) -> Tensor:
    good_f = float(good)
    bad_f = float(bad)
    width = max(bad_f - good_f, 1.0e-12)
    z = torch.clamp((x - good_f) / width, min=0.0)
    return torch.exp(-3.0 * z.pow(2)).clamp(0.0, 1.0)


def _smooth_max(x: Tensor, *, sharpness: float) -> Tensor:
    if x.shape[-1] == 1:
        return x[..., 0]
    weights = torch.softmax(float(sharpness) * x, dim=-1)
    return torch.sum(weights * x, dim=-1)


def _weighted_geometric_mean(values: tuple[Tensor, ...], weights: tuple[float, ...]) -> Tensor:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    total_weight = max(sum(float(w) for w in weights), 1.0e-12)
    out = torch.zeros_like(values[0])
    for value, weight in zip(values, weights, strict=True):
        out = out + float(weight) * torch.log(torch.clamp(value, min=1.0e-6, max=1.0))
    return torch.exp(out / total_weight).clamp(0.0, 1.0)
