from __future__ import annotations

from dataclasses import dataclass
import math

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
        applied_delta_action: Tensor | None = None,
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
        boundary_missing_loss = (~found.reshape(-1)).to(dtype=boundary_points.dtype, device=boundary_points.device)

        ip_error = torch.abs(ip - ip_ref)
        requested = action if requested_action is None else requested_action.to(dtype=action.dtype, device=action.device)
        delta_action = (
            action - previous_action
            if applied_delta_action is None
            else applied_delta_action.to(dtype=action.dtype, device=action.device)
        )
        action_rms = torch.sqrt(torch.mean(action.pow(2), dim=-1))
        delta_action_rms = torch.sqrt(torch.mean(delta_action.pow(2), dim=-1))
        max_abs_action = torch.max(torch.abs(action), dim=-1).values
        max_abs_delta_action = torch.max(torch.abs(delta_action), dim=-1).values
        saturation_reference = action if applied_delta_action is None else delta_action
        saturation_delta = requested - saturation_reference
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
            float(c.boundary_missing_weight) * boundary_missing_loss
            + float(c.shape_mean_weight) * shape_mean_loss
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
                "boundary_missing_loss": boundary_missing_loss,
            },
        )


class T15TCVQualityReward:
    """TCV-style bounded quality reward translated to derivative actions."""

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
        applied_delta_action: Tensor | None = None,
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
        boundary_missing_loss = (~found.reshape(-1)).to(dtype=boundary_points.dtype, device=boundary_points.device)

        ip_error = torch.abs(ip - ip_ref)
        requested = action if requested_action is None else requested_action.to(dtype=action.dtype, device=action.device)
        delta_action = (
            action - previous_action
            if applied_delta_action is None
            else applied_delta_action.to(dtype=action.dtype, device=action.device)
        )
        action_rms = torch.sqrt(torch.mean(action.pow(2), dim=-1))
        delta_action_rms = torch.sqrt(torch.mean(delta_action.pow(2), dim=-1))
        max_abs_action = torch.max(torch.abs(action), dim=-1).values
        max_abs_delta_action = torch.max(torch.abs(delta_action), dim=-1).values
        saturation_reference = action if applied_delta_action is None else delta_action
        saturation_delta = requested - saturation_reference
        requested_action_rms = torch.sqrt(torch.mean(requested.pow(2), dim=-1))
        action_saturation_delta_rms = torch.sqrt(torch.mean(saturation_delta.pow(2), dim=-1))
        action_saturation_delta_max = torch.max(torch.abs(saturation_delta), dim=-1).values
        action_saturation_fraction = torch.mean((torch.abs(saturation_delta) > 1.0e-6).to(dtype=action.dtype), dim=-1)

        alpha = float(c.smoothmax_alpha)
        shape_point_loss = shape_error / max(float(c.shape_mean_scale_m), 1.0e-12)
        shape_mean_loss = _smoothmax(shape_point_loss, dim=-1, alpha=alpha)
        shape_max_loss = shape_error_max / max(float(c.shape_max_scale_m), 1.0e-12)
        ip_loss = ip_error / max(float(c.ip_scale_a), 1.0e-12)
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

        component_losses = torch.stack(
            [
                float(c.shape_mean_weight) * shape_mean_loss,
                float(c.shape_max_weight) * shape_max_loss,
                float(c.ip_weight) * ip_loss,
                float(c.current_weight) * current_loss,
                float(c.derivative_weight) * derivative_loss,
                float(c.current_usage_weight) * current_usage_cost,
                float(c.derivative_usage_weight) * derivative_usage_cost,
                float(c.action_weight) * action_loss,
                float(c.delta_action_weight) * delta_action_loss,
                float(c.actuator_saturation_weight) * actuator_saturation_loss,
                float(c.boundary_missing_weight) * boundary_missing_loss,
            ],
            dim=-1,
        )
        combined_loss = _smoothmax(component_losses, dim=-1, alpha=alpha)
        quality = 1.0 - combined_loss

        if episode_progress is None:
            progress = torch.zeros_like(combined_loss)
        else:
            progress = torch.clamp(episode_progress.to(dtype=combined_loss.dtype, device=combined_loss.device).reshape_as(combined_loss), 0.0, 1.0)
        terminal_mask = terminated.to(dtype=torch.bool, device=combined_loss.device).reshape_as(combined_loss)
        terminal_remaining_loss = torch.where(
            terminal_mask,
            torch.clamp(1.0 - progress, min=0.0) * float(c.terminal_remaining_cost),
            torch.zeros_like(combined_loss),
        )
        immediate_terminal_penalty = torch.where(
            terminal_mask,
            torch.full_like(combined_loss, float(c.terminal_reward) * float(c.reward_scale)),
            torch.zeros_like(combined_loss),
        )
        remaining_terminal_penalty = -float(c.reward_scale) * terminal_remaining_loss
        terminal_total_penalty = immediate_terminal_penalty + remaining_terminal_penalty

        reward = float(c.reward_scale) * quality + terminal_total_penalty

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
                "physical_cost": combined_loss,
                "tcv_quality": quality,
                "tcv_combined_loss": combined_loss,
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
                "boundary_missing_loss": boundary_missing_loss,
            },
        )


class T15TCVDerivativeReward:
    """Source-locked TCV reward structure translated from voltages to derivatives."""

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
        applied_delta_action: Tensor | None = None,
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
        boundary_missing_loss = (~found.reshape(-1)).to(dtype=boundary_points.dtype, device=boundary_points.device)

        ip_error = torch.abs(ip - ip_ref)
        requested = action if requested_action is None else requested_action.to(dtype=action.dtype, device=action.device)
        delta_action = (
            action - previous_action
            if applied_delta_action is None
            else applied_delta_action.to(dtype=action.dtype, device=action.device)
        )
        action_rms = torch.sqrt(torch.mean(action.pow(2), dim=-1))
        delta_action_rms = torch.sqrt(torch.mean(delta_action.pow(2), dim=-1))
        max_abs_action = torch.max(torch.abs(action), dim=-1).values
        max_abs_delta_action = torch.max(torch.abs(delta_action), dim=-1).values
        saturation_reference = action if applied_delta_action is None else delta_action
        saturation_delta = requested - saturation_reference
        requested_action_rms = torch.sqrt(torch.mean(requested.pow(2), dim=-1))
        action_saturation_delta_rms = torch.sqrt(torch.mean(saturation_delta.pow(2), dim=-1))
        action_saturation_delta_max = torch.max(torch.abs(saturation_delta), dim=-1).values
        action_saturation_fraction = torch.mean((torch.abs(saturation_delta) > 1.0e-6).to(dtype=action.dtype), dim=-1)

        shape_point_error = _tcv_smoothmax(shape_error, dim=-1, alpha=abs(float(c.smoothmax_alpha)))
        shape_reward = _tcv_softplus(shape_point_error, bad=max(float(c.shape_mean_scale_m), 1.0e-12), good=0.0)
        shape_max_reward = _tcv_softplus(shape_error_max, bad=max(float(c.shape_max_scale_m), 1.0e-12), good=0.0)
        ip_reward = _tcv_softplus(ip_error, bad=max(float(c.ip_scale_a), 1.0e-12), good=0.0)
        current_reward = _tcv_clipped_linear(current_usage_fraction, bad=float(c.current_bad_fraction), good=float(c.current_soft_fraction))
        derivative_reward = _tcv_softplus(max_abs_delta_action, bad=float(c.derivative_bad_fraction), good=0.0)
        actuator_saturation_loss = torch.mean(saturation_delta.pow(2), dim=-1)
        saturation_reward = _tcv_clipped_linear(
            torch.sqrt(torch.clamp(actuator_saturation_loss, min=0.0)),
            bad=1.0,
            good=0.0,
        )
        boundary_reward = torch.where(
            found.reshape(-1),
            torch.ones_like(ip_reward),
            torch.zeros_like(ip_reward),
        )

        component_rewards = torch.stack(
            [
                shape_reward,
                shape_max_reward,
                ip_reward,
                current_reward,
                derivative_reward,
                saturation_reward,
                boundary_reward,
            ],
            dim=-1,
        )
        weights = torch.as_tensor(
            [
                float(c.shape_mean_weight),
                float(c.shape_max_weight),
                float(c.ip_weight),
                float(c.current_weight),
                float(c.derivative_weight),
                float(c.actuator_saturation_weight),
                float(c.boundary_missing_weight),
            ],
            dtype=component_rewards.dtype,
            device=component_rewards.device,
        )
        quality = _tcv_smoothmax(component_rewards, dim=-1, alpha=float(c.smoothmax_alpha), weights=weights)
        normal_reward = float(c.reward_scale) * quality

        terminal_mask = terminated.to(dtype=torch.bool, device=quality.device).reshape_as(quality)
        terminal_reward = torch.full_like(normal_reward, float(c.terminal_reward) * float(c.reward_scale))
        reward = torch.where(terminal_mask, terminal_reward, normal_reward)
        if episode_progress is None:
            progress = torch.zeros_like(quality)
        else:
            progress = torch.clamp(episode_progress.to(dtype=quality.dtype, device=quality.device).reshape_as(quality), 0.0, 1.0)
        terminal_total_penalty = torch.where(terminal_mask, terminal_reward, torch.zeros_like(terminal_reward))

        shape_mean_loss = 1.0 - shape_reward
        shape_max_loss = 1.0 - shape_max_reward
        ip_loss = 1.0 - ip_reward
        current_loss = 1.0 - current_reward
        derivative_loss = 1.0 - derivative_reward
        saturation_component_loss = 1.0 - saturation_reward
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
                "max_abs_delta_action": max_abs_delta_action,
                "action_rms": action_rms,
                "requested_action_rms": requested_action_rms,
                "applied_action_rms": action_rms,
                "delta_action_rms": delta_action_rms,
                "action_saturation_delta_rms": action_saturation_delta_rms,
                "action_saturation_delta_max": action_saturation_delta_max,
                "action_saturation_fraction": action_saturation_fraction,
                "episode_progress": progress,
                "physical_cost": 1.0 - quality,
                "tcv_quality": quality,
                "tcv_combined_loss": 1.0 - quality,
                "shape_mean_loss": shape_mean_loss,
                "shape_max_loss": shape_max_loss,
                "ip_loss": ip_loss,
                "current_loss": current_loss,
                "derivative_loss": derivative_loss,
                "current_usage_loss": current_loss,
                "derivative_usage_loss": derivative_loss,
                "action_loss": torch.zeros_like(quality),
                "delta_action_loss": torch.zeros_like(quality),
                "actuator_saturation_loss": actuator_saturation_loss,
                "tcv_saturation_component_loss": saturation_component_loss,
                "terminal_remaining_loss": torch.zeros_like(quality),
                "terminal_total_penalty": terminal_total_penalty,
                "boundary_found": boundary_found.to(dtype=reward.dtype),
                "boundary_missing_loss": boundary_missing_loss,
            },
        )


def build_reward(config: RewardConfig, *, control_rate_hz: float) -> T15PhysicalReward | T15TCVQualityReward | T15TCVDerivativeReward:
    if config.kind == "tcv_derivative":
        return T15TCVDerivativeReward(config, control_rate_hz=control_rate_hz)
    if config.kind in {"tcv_quality", "tcv_quality_legacy"}:
        return T15TCVQualityReward(config, control_rate_hz=control_rate_hz)
    return T15PhysicalReward(config, control_rate_hz=control_rate_hz)


def _huber(x: Tensor, *, delta: float = 1.0) -> Tensor:
    z = torch.abs(x)
    d = float(delta)
    return torch.where(z <= d, 0.5 * z.pow(2), d * (z - 0.5 * d))


def _smoothmax(x: Tensor, *, dim: int, alpha: float) -> Tensor:
    a = max(float(alpha), 1.0e-6)
    return torch.logsumexp(a * x, dim=dim) / a - torch.log(torch.as_tensor(x.shape[dim], dtype=x.dtype, device=x.device)) / a


def _threshold_square(x: Tensor, *, start: float, bad: float) -> Tensor:
    width = max(float(bad) - float(start), 1.0e-12)
    z = torch.clamp((x - float(start)) / width, min=0.0)
    return z.pow(2)


def _tcv_scale(v: Tensor, a: float, b: float, c: float, d: float) -> Tensor:
    den = float(b) - float(a)
    if abs(den) < 1.0e-12:
        den = 1.0e-12 if den >= 0.0 else -1.0e-12
    v01 = (v - float(a)) / den
    return float(c) + v01 * (float(d) - float(c))


def _tcv_logistic(v: Tensor) -> Tensor:
    return torch.sigmoid(torch.clamp(v, -50.0, 50.0))


def _tcv_softplus(errors: Tensor, *, bad: float, good: float = 0.0) -> Tensor:
    low = -math.log(19.0)
    return torch.clamp(2.0 * _tcv_logistic(_tcv_scale(errors, float(good), float(bad), 0.0, low)), 0.0, 1.0)


def _tcv_clipped_linear(errors: Tensor, *, bad: float, good: float = 0.0) -> Tensor:
    return torch.clamp(_tcv_scale(errors, float(bad), float(good), 0.0, 1.0), 0.0, 1.0)


def _tcv_smoothmax(x: Tensor, *, dim: int, alpha: float, weights: Tensor | None = None) -> Tensor:
    if weights is None:
        weights_t = torch.ones((x.shape[dim],), dtype=x.dtype, device=x.device)
    else:
        weights_t = weights.to(dtype=x.dtype, device=x.device)
    if weights_t.shape != (x.shape[dim],):
        raise ValueError("tcv smoothmax weights must match the combined dimension")
    finite_weight = torch.clamp(weights_t, min=0.0)
    if float(torch.sum(finite_weight).detach().cpu()) <= 0.0:
        return torch.full(x.shape[:dim] + x.shape[dim + 1 :], float("nan"), dtype=x.dtype, device=x.device)
    if math.isinf(float(alpha)):
        return torch.max(x, dim=dim).values if float(alpha) > 0.0 else torch.min(x, dim=dim).values
    view_shape = [1] * x.ndim
    view_shape[dim] = x.shape[dim]
    log_weights = torch.log(torch.clamp(finite_weight, min=1.0e-45)).reshape(view_shape)
    logits = log_weights + float(alpha) * x
    soft_weights = torch.exp(logits - torch.logsumexp(logits, dim=dim, keepdim=True))
    return torch.sum(x * soft_weights, dim=dim)
