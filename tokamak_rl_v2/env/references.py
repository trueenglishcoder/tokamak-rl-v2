from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from tokamak_rl_v2.config.schema import BoundaryReferenceConfig, InitialRanges, IpReferenceConfig, ReferenceConfig

PARAMETER_ORDER = ("R0", "Z0", "A0", "kappa", "delta")


@dataclass(frozen=True, slots=True)
class ReferenceBatch:
    ip: Tensor
    parameters: Tensor
    points: Tensor
    radii: Tensor
    theta: Tensor


def boundary_points_from_parameters(parameters: Tensor, theta: Tensor) -> Tensor:
    R0 = parameters[..., 0][..., None]
    Z0 = parameters[..., 1][..., None]
    A0 = parameters[..., 2][..., None]
    kappa = parameters[..., 3][..., None]
    delta = parameters[..., 4][..., None]
    sin_t = torch.sin(theta)
    R = R0 + A0 * torch.cos(theta) - delta * A0 * sin_t.pow(2)
    Z = Z0 + A0 * kappa * sin_t
    return torch.stack([R, Z], dim=-1)


def radii_from_points(points: Tensor, center: Tensor) -> Tensor:
    return torch.linalg.norm(points - center[..., None, :], dim=-1)


def sample_initial_conditions(rng: np.random.Generator, ranges: InitialRanges, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ip = rng.uniform(ranges.ip.min, ranges.ip.max, size=(batch_size,))
    pfc = np.stack([rng.uniform(r.min, r.max, size=(batch_size,)) for r in ranges.pfc_currents], axis=1)
    sol = np.stack([rng.uniform(r.min, r.max, size=(batch_size,)) for r in ranges.sol_currents], axis=1)
    params = np.stack([rng.uniform(ranges.boundary_parameters[name].min, ranges.boundary_parameters[name].max, size=(batch_size,)) for name in PARAMETER_ORDER], axis=1)
    return ip.astype(float), pfc.astype(float), sol.astype(float), params.astype(float)


def generate_reference_batch(
    *,
    config: ReferenceConfig,
    initial_ip: np.ndarray,
    initial_parameters: np.ndarray,
    steps: int,
    device: torch.device | str,
    seed: int,
    initial_boundary_points: Tensor | np.ndarray | None = None,
    initial_boundary_radii: Tensor | np.ndarray | None = None,
) -> ReferenceBatch:
    dev = torch.device(device)
    rng = np.random.default_rng(int(seed))
    B = int(np.asarray(initial_ip).reshape(-1).shape[0])
    ip = np.zeros((B, int(steps) + 1), dtype=np.float64)
    params = np.zeros((B, int(steps) + 1, 5), dtype=np.float64)
    theta = torch.linspace(-torch.pi, torch.pi, int(config.theta_count) + 1, dtype=torch.float64, device=dev)[:-1]
    for b in range(B):
        ip[b] = _segmented_ip(config.ip, float(initial_ip[b]), int(steps), rng, dt=float(config.t_step))
        if config.boundary.kind != "hold_reset_boundary":
            params[b] = _boundary_params(config.boundary, np.asarray(initial_parameters[b], dtype=float), int(steps), rng, dt=float(config.t_step))
    params_t = torch.as_tensor(params, dtype=torch.float64, device=dev)
    if config.boundary.kind == "hold_reset_boundary":
        if initial_boundary_points is None or initial_boundary_radii is None:
            raise ValueError("hold_reset_boundary requires initial_boundary_points and initial_boundary_radii")
        points0 = torch.nan_to_num(torch.as_tensor(initial_boundary_points, dtype=torch.float64, device=dev), nan=0.0, posinf=0.0, neginf=0.0).reshape(B, int(config.theta_count), 2)
        radii0 = torch.nan_to_num(torch.as_tensor(initial_boundary_radii, dtype=torch.float64, device=dev), nan=0.0, posinf=0.0, neginf=0.0).reshape(B, int(config.theta_count))
        points = points0[:, None, :, :].repeat(1, int(steps) + 1, 1, 1)
        radii = radii0[:, None, :].repeat(1, int(steps) + 1, 1)
        centers = torch.mean(points0, dim=1)
        params_t = torch.zeros((B, int(steps) + 1, 5), dtype=torch.float64, device=dev)
        params_t[..., 0:2] = centers[:, None, :]
    else:
        points = boundary_points_from_parameters(params_t, theta)
        centers = params_t[..., 0:2]
        radii = radii_from_points(points, centers)
    return ReferenceBatch(
        ip=torch.as_tensor(ip, dtype=torch.float64, device=dev),
        parameters=params_t,
        points=points,
        radii=radii,
        theta=theta,
    )


def _segmented_ip(cfg: IpReferenceConfig, start: float, steps: int, rng: np.random.Generator, *, dt: float) -> np.ndarray:
    values = np.zeros((steps + 1,), dtype=float)
    values[0] = float(np.clip(start, cfg.min, cfg.max))
    k = 0
    target = values[0]
    previous_ramp_direction = 0
    for seg_len in _segment_lengths(cfg, int(steps), rng):
        if rng.random() < float(cfg.hold_probability):
            next_target = target
        else:
            next_target = float(rng.uniform(cfg.min, cfg.max))
        max_delta = float(cfg.rate_limit) * float(seg_len) * float(dt)
        next_target = float(np.clip(next_target, target - max_delta, target + max_delta))
        next_target = float(np.clip(next_target, cfg.min, cfg.max))
        ramp_direction = _direction(next_target - target)
        if previous_ramp_direction and ramp_direction and ramp_direction != previous_ramp_direction:
            next_target = target
            ramp_direction = 0
        ramp = np.linspace(target, next_target, int(seg_len) + 1, dtype=float)[1:]
        values[k + 1 : k + int(seg_len) + 1] = ramp
        target = next_target
        previous_ramp_direction = ramp_direction
        k += int(seg_len)
    return values


def _direction(value: float, *, atol: float = 1.0e-9) -> int:
    value_f = float(value)
    if value_f > float(atol):
        return 1
    if value_f < -float(atol):
        return -1
    return 0


def _segment_lengths(cfg: IpReferenceConfig, steps: int, rng: np.random.Generator) -> np.ndarray:
    if steps <= 0:
        return np.zeros((0,), dtype=int)
    min_len = max(1, int(cfg.segment_min_steps))
    max_len = max(min_len, int(cfg.segment_max_steps))
    min_count = max(1, int(cfg.segment_count_min))
    max_count = max(min_count, int(cfg.segment_count_max))
    if steps < min_len:
        # Very short smoke-test episodes cannot satisfy production segment
        # length/count constraints. Keep them continuous instead of failing
        # before the trainer can exercise reset/replay/checkpoint logic.
        return np.asarray([int(steps)], dtype=int)
    feasible_min = max(min_count, int(np.ceil(float(steps) / float(max_len))))
    feasible_max = min(max_count, int(np.floor(float(steps) / float(min_len))))
    if feasible_min > feasible_max:
        if steps <= max_len:
            return np.asarray([int(steps)], dtype=int)
        raise ValueError(
            "Ip reference segment constraints cannot cover episode length: "
            f"steps={steps}, segment_min_steps={min_len}, segment_max_steps={max_len}, "
            f"segment_count_min={min_count}, segment_count_max={max_count}"
        )
    count = int(rng.integers(feasible_min, feasible_max + 1))
    lengths = np.full((count,), min_len, dtype=int)
    remaining = int(steps - int(np.sum(lengths)))
    capacities = np.full((count,), max_len - min_len, dtype=int)
    while remaining > 0:
        available = np.flatnonzero(capacities > 0)
        if available.size == 0:
            raise ValueError("Ip reference segment allocation exhausted unexpectedly")
        idx = int(rng.choice(available))
        add = int(rng.integers(1, min(int(capacities[idx]), remaining) + 1))
        lengths[idx] += add
        capacities[idx] -= add
        remaining -= add
    rng.shuffle(lengths)
    return lengths


def _boundary_params(cfg: BoundaryReferenceConfig, start: np.ndarray, steps: int, rng: np.random.Generator, *, dt: float) -> np.ndarray:
    out = np.repeat(start.reshape(1, 5), steps + 1, axis=0)
    if cfg.kind == "static_initial_parameters":
        return out
    if cfg.kind != "rate_limited_parameters":
        raise ValueError(f"unknown boundary reference kind: {cfg.kind}")
    # Rate-limited generation around the initial handover parameters. Bounds are
    # expected to be enforced before this by the sampled initial ranges.
    for k in range(1, steps + 1):
        prev = out[k - 1].copy()
        for i, name in enumerate(PARAMETER_ORDER):
            limit = float(cfg.rate_limits.get(name, 0.0))
            if limit > 0.0:
                prev[i] += rng.uniform(-limit * dt, limit * dt)
        out[k] = prev
    return out
