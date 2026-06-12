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
        derivative_usage: Tensor,
        boundary_found: Tensor,
        terminated: Tensor,
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
        shape_quality_points = transforms.softplus(shape_error, good=c.shape_good_m, bad=c.shape_bad_m)
        if c.shape_aggregator == "smooth_worst":
            r_shape = combiners.smooth_max(shape_quality_points, alpha=-1.0, dim=-1)
        elif c.shape_aggregator == "mean":
            r_shape = combiners.mean(shape_quality_points, dim=-1)
        elif c.shape_aggregator == "geometric_mean":
            r_shape = combiners.geometric_mean(shape_quality_points, dim=-1)
        else:
            raise ValueError(f"unsupported shape_aggregator: {c.shape_aggregator}")
        ip_error = torch.abs(ip - ip_ref)
        r_ip = transforms.softplus(ip_error, good=c.ip_good_a, bad=c.ip_bad_a)
        current_error = torch.clamp(current_over_limit_a, min=0.0)
        r_current = transforms.softplus(current_error, good=c.current_good_a, bad=max(c.current_bad_a, c.current_good_a + 1.0))
        derivative_usage = torch.clamp(derivative_usage, min=0.0)
        r_derivative = transforms.softplus(derivative_usage, good=c.derivative_good, bad=max(c.derivative_bad, c.derivative_good + 1.0e-6))
        delta_action = action - previous_action
        action_mag = torch.sqrt(torch.mean(action.pow(2), dim=-1))
        delta_mag = torch.sqrt(torch.mean(delta_action.pow(2), dim=-1))
        tracking = torch.stack([r_shape, r_ip], dim=-1)
        tracking_weights = torch.as_tensor([c.shape_weight, c.ip_weight], dtype=tracking.dtype, device=tracking.device)
        if c.tracking_combiner == "smooth_min":
            tracking_quality = combiners.smooth_max(tracking, alpha=-5.0, weights=tracking_weights, dim=-1)
        elif c.tracking_combiner == "weighted_mean":
            tracking_quality = combiners.mean(tracking, weights=tracking_weights, dim=-1)
        elif c.tracking_combiner == "geometric_mean":
            tracking_quality = combiners.geometric_mean(tracking, weights=tracking_weights, dim=-1)
        elif c.tracking_combiner == "product":
            normalized_weights = tracking_weights / torch.clamp(torch.sum(tracking_weights), min=1.0e-12)
            tracking_quality = combiners.multiply(tracking, weights=normalized_weights, dim=-1)
        else:
            raise ValueError(f"unsupported tracking_combiner: {c.tracking_combiner}")
        action_penalty = torch.clamp(float(c.action_penalty_weight) * action_mag.pow(2), min=0.0)
        delta_action_penalty = torch.clamp(float(c.delta_action_penalty_weight) * delta_mag.pow(2), min=0.0)
        r_action = torch.clamp(1.0 - action_penalty, 0.0, 1.0)
        r_delta = torch.clamp(1.0 - delta_action_penalty, 0.0, 1.0)
        regularization_quality = torch.clamp(1.0 - action_penalty - delta_action_penalty, 0.0, 1.0)
        combined = tracking_quality * r_current * r_derivative * regularization_quality
        combined = torch.where(boundary_found, combined, torch.zeros_like(combined))
        terminal = torch.full_like(combined, float(c.terminal_reward) * float(c.reward_scale))
        if c.mode == "quality":
            reward = combined * float(c.reward_scale)
            reward = torch.where(terminated, terminal, reward)
            dense_components: dict[str, Tensor] = {}
        elif c.mode == "dense_physical":
            shape_scale = max(float(c.shape_bad_m), 1.0e-12)
            ip_scale = max(float(c.ip_bad_a), 1.0e-12)
            current_scale = max(float(c.current_bad_a), 1.0e-12)
            derivative_scale = max(float(c.derivative_bad), 1.0e-12)
            dense_shape_loss = _huber01(shape_error_mean / shape_scale) + 0.25 * _huber01(shape_error_max / shape_scale)
            dense_ip_loss = _huber01(ip_error / ip_scale)
            dense_current_loss = _huber01(current_error / current_scale)
            dense_derivative_loss = _huber01(derivative_usage / derivative_scale)
            dense_action_loss = float(c.action_penalty_weight) * action_mag.pow(2) + float(c.delta_action_penalty_weight) * delta_mag.pow(2)
            dense_loss = (
                float(c.shape_weight) * dense_shape_loss
                + float(c.ip_weight) * dense_ip_loss
                + float(c.current_weight) * dense_current_loss
                + float(c.derivative_weight) * dense_derivative_loss
                + dense_action_loss
            )
            reward = -float(c.reward_scale) * dense_loss
            reward = torch.where(terminated, reward + terminal, reward)
            dense_components = {
                "dense_loss": dense_loss,
                "dense_shape_loss": dense_shape_loss,
                "dense_ip_loss": dense_ip_loss,
                "dense_current_loss": dense_current_loss,
                "dense_derivative_loss": dense_derivative_loss,
                "dense_action_loss": dense_action_loss,
                "dense_weighted_shape_loss": float(c.shape_weight) * dense_shape_loss,
                "dense_weighted_ip_loss": float(c.ip_weight) * dense_ip_loss,
                "dense_weighted_current_loss": float(c.current_weight) * dense_current_loss,
                "dense_weighted_derivative_loss": float(c.derivative_weight) * dense_derivative_loss,
            }
        else:
            raise ValueError(f"unsupported reward mode: {c.mode}")
        return RewardBatch(
            reward=reward,
            components={
                "shape_error_mean_m": shape_error_mean,
                "shape_error_max_m": shape_error_max,
                "shape_quality": r_shape,
                "ip_error_a": ip_error,
                "ip_quality": r_ip,
                "tracking_quality": tracking_quality,
                "shape_aggregator_smooth_worst": torch.full_like(reward, 1.0 if c.shape_aggregator == "smooth_worst" else 0.0),
                "tracking_combiner_smooth_min": torch.full_like(reward, 1.0 if c.tracking_combiner == "smooth_min" else 0.0),
                "current_over_limit_a": current_error,
                "current_quality": r_current,
                "derivative_usage": derivative_usage,
                "derivative_quality": r_derivative,
                "action_rms": action_mag,
                "delta_action_rms": delta_mag,
                "action_penalty": action_penalty,
                "delta_action_penalty": delta_action_penalty,
                "action_quality": r_action,
                "delta_action_quality": r_delta,
                "regularization_quality": regularization_quality,
                "combined_quality": combined,
                "boundary_found": boundary_found.to(dtype=reward.dtype),
                **dense_components,
            },
        )


def _huber01(x: Tensor) -> Tensor:
    x = torch.clamp(x, min=0.0)
    return torch.where(x <= 1.0, 0.5 * x.pow(2), x - 0.5)
