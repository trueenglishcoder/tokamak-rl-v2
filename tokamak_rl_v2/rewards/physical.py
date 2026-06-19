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
        requested_action: Tensor | None = None,
        current_over_limit_a: Tensor,
        current_usage_fraction: Tensor,
        current_margin_fraction: Tensor,
        derivative_usage: Tensor,
        boundary_found: Tensor,
        terminated: Tensor,
        current_usage_loss: Tensor | None = None,
        derivative_usage_loss: Tensor | None = None,
        current_usage_mean_fraction: Tensor | None = None,
        derivative_usage_mean_fraction: Tensor | None = None,
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
        requested = action if requested_action is None else requested_action.to(dtype=action.dtype, device=action.device)
        delta_action = action - previous_action
        action_rms = torch.sqrt(torch.mean(action.pow(2), dim=-1))
        delta_action_rms = torch.sqrt(torch.mean(delta_action.pow(2), dim=-1))
        max_abs_action = torch.max(torch.abs(action), dim=-1).values
        saturation_delta = requested - action
        requested_action_rms = torch.sqrt(torch.mean(requested.pow(2), dim=-1))
        action_saturation_delta_rms = torch.sqrt(torch.mean(saturation_delta.pow(2), dim=-1))
        action_saturation_delta_max = torch.max(torch.abs(saturation_delta), dim=-1).values
        action_saturation_fraction = torch.mean((torch.abs(saturation_delta) > 1.0e-6).to(dtype=action.dtype), dim=-1)

        shape_mean_loss = _huber(shape_error_mean / max(float(c.shape_mean_scale_m), 1.0e-12))
        shape_max_loss = _huber(shape_error_max / max(float(c.shape_max_scale_m), 1.0e-12))
        ip_loss = _huber(ip_error / max(float(c.ip_scale_a), 1.0e-12))
        current_loss = _threshold_square(current_usage_fraction, start=float(c.current_soft_fraction), bad=float(c.current_bad_fraction))
        derivative_loss = _threshold_square(derivative_usage, start=float(c.derivative_soft_fraction), bad=float(c.derivative_bad_fraction))
        current_usage_cost = (
            torch.clamp(current_usage_fraction, min=0.0).pow(2)
            if current_usage_loss is None
            else torch.clamp(current_usage_loss.to(dtype=action.dtype, device=action.device).reshape_as(current_usage_fraction), min=0.0)
        )
        derivative_usage_cost = (
            torch.clamp(derivative_usage, min=0.0).pow(2)
            if derivative_usage_loss is None
            else torch.clamp(derivative_usage_loss.to(dtype=action.dtype, device=action.device).reshape_as(derivative_usage), min=0.0)
        )
        current_usage_mean = (
            torch.clamp(current_usage_fraction, min=0.0)
            if current_usage_mean_fraction is None
            else torch.clamp(current_usage_mean_fraction.to(dtype=action.dtype, device=action.device).reshape_as(current_usage_fraction), min=0.0)
        )
        derivative_usage_mean = (
            torch.clamp(derivative_usage, min=0.0)
            if derivative_usage_mean_fraction is None
            else torch.clamp(derivative_usage_mean_fraction.to(dtype=action.dtype, device=action.device).reshape_as(derivative_usage), min=0.0)
        )
        action_loss = torch.mean(action.pow(2), dim=-1)
        delta_action_loss = torch.mean(delta_action.pow(2), dim=-1)
        actuator_saturation_loss = torch.mean(saturation_delta.pow(2), dim=-1)

        physical_cost = (
            float(c.shape_mean_weight) * shape_mean_loss
            + float(c.shape_max_weight) * shape_max_loss
            + float(c.ip_weight) * ip_loss
            + float(c.current_weight) * current_loss
            + float(c.derivative_weight) * derivative_loss
            + float(c.current_usage_weight) * current_usage_cost
            + float(c.derivative_usage_weight) * derivative_usage_cost
            + float(c.action_weight) * action_loss
            + float(c.delta_action_weight) * delta_action_loss
            + float(c.actuator_saturation_weight) * actuator_saturation_loss
        )
        if episode_progress is None:
            progress = torch.zeros_like(physical_cost)
        else:
            progress = torch.clamp(episode_progress.to(dtype=physical_cost.dtype, device=physical_cost.device).reshape_as(physical_cost), 0.0, 1.0)

        terminal_mask = terminated.to(dtype=torch.bool, device=physical_cost.device).reshape_as(physical_cost)
        terminal_remaining_loss = torch.where(
            terminal_mask,
            torch.clamp(1.0 - progress, min=0.0) * float(c.terminal_remaining_cost),
            torch.zeros_like(physical_cost),
        )
        immediate_terminal_penalty = torch.where(
            terminal_mask,
            torch.full_like(physical_cost, float(c.terminal_reward) * float(c.reward_scale)),
            torch.zeros_like(physical_cost),
        )
        remaining_terminal_penalty = -float(c.reward_scale) * terminal_remaining_loss
        terminal_total_penalty = immediate_terminal_penalty + remaining_terminal_penalty

        reward = -float(c.reward_scale) * physical_cost
        reward = reward + terminal_total_penalty

        return RewardBatch(
            reward=reward,
            components={
                "shape_error_mean_m": shape_error_mean,
                "shape_error_max_m": shape_error_max,
                "ip_error_a": ip_error,
                "current_over_limit_a": torch.clamp(current_over_limit_a, min=0.0),
                "current_usage_fraction": torch.clamp(current_usage_fraction, min=0.0),
                "current_usage_mean_fraction": current_usage_mean,
                "current_margin_fraction": current_margin_fraction,
                "derivative_usage": torch.clamp(derivative_usage, min=0.0),
                "derivative_usage_mean_fraction": derivative_usage_mean,
                "max_abs_action": max_abs_action,
                "action_rms": action_rms,
                "requested_action_rms": requested_action_rms,
                "applied_action_rms": action_rms,
                "delta_action_rms": delta_action_rms,
                "action_saturation_delta_rms": action_saturation_delta_rms,
                "action_saturation_delta_max": action_saturation_delta_max,
                "action_saturation_fraction": action_saturation_fraction,
                "episode_progress": progress,
                "physical_cost": physical_cost,
                "shape_mean_loss": shape_mean_loss,
                "shape_max_loss": shape_max_loss,
                "ip_loss": ip_loss,
                "current_loss": current_loss,
                "derivative_loss": derivative_loss,
                "current_usage_loss": current_usage_cost,
                "derivative_usage_loss": derivative_usage_cost,
                "action_loss": action_loss,
                "delta_action_loss": delta_action_loss,
                "actuator_saturation_loss": actuator_saturation_loss,
                "terminal_remaining_loss": terminal_remaining_loss,
                "terminal_total_penalty": terminal_total_penalty,
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
