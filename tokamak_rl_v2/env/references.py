from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from tokamak_rl_v2.config.schema import BoundaryReferenceConfig, InitialRanges, IpReferenceConfig, ReferenceConfig
from tokamak_rl_v2.env.t15_reference_limits import T15ReferenceLimits, load_reference_limits

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
        if config.ip.kind == "hold_reset":
            ip[b] = float(initial_ip[b])
        elif config.ip.kind == "segmented_profile":
            ip[b] = _segmented_profile_ip(config.ip, float(initial_ip[b]), int(steps), rng, dt=float(config.t_step))
        else:
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


def _segmented_profile_ip(cfg: IpReferenceConfig, start: float, steps: int, rng: np.random.Generator, *, dt: float) -> np.ndarray:
    if cfg.limits_path is None:
        raise ValueError("segmented_profile requires limits_path")
    limits = load_reference_limits(cfg.limits_path)
    lo = float(limits.ip_p01_a)
    hi = float(limits.ip_p99_a)
    start_ip = float(start)
    if not (lo <= start_ip <= hi):
        raise ValueError(f"segmented_profile reset Ip {start_ip:g} is outside production bounds [{lo:g}, {hi:g}]")
    width = max(hi - lo, 1.0)
    max_delta = float(cfg.max_delta_fraction) * width
    positive_rate_base, negative_rate_base = _segmented_profile_rate_bases(cfg, limits)
    positive_rate_min = float(cfg.ramp_up_rate_min_fraction) * float(positive_rate_base)
    positive_rate_max = float(cfg.ramp_up_rate_fraction) * float(positive_rate_base)
    negative_rate_min = float(cfg.ramp_down_rate_min_fraction) * float(negative_rate_base)
    negative_rate_max = float(cfg.ramp_down_rate_fraction) * float(negative_rate_base)
    # Cosine-eased ramps have a peak derivative of pi/2 times their mean
    # derivative. Sample the mean ramp below the configured signed-rate limit
    # so the finite-difference trajectory itself obeys the same bound.
    ramp_peak_factor = 1.7 if bool(cfg.smooth_ramps) else 1.0
    positive_delta_rate_max = positive_rate_max / ramp_peak_factor
    negative_delta_rate_max = negative_rate_max / ramp_peak_factor
    positive_peak_rate_max = positive_rate_max
    negative_peak_rate_max = negative_rate_max
    min_hold = max(1, int(cfg.hold_min_steps))
    max_hold = max(min_hold, int(cfg.hold_max_steps))
    final_hold = min(max(0, int(cfg.final_hold_min_steps)), max(0, int(steps) - 1))
    available = int(steps) - final_hold

    for _attempt in range(512):
        if available < 2:
            continue
        lengths = _segment_lengths(cfg, int(available), rng)
        if int(lengths.size) < 2:
            continue
        kinds = _sample_segment_kinds(cfg, lengths, rng)
        if kinds is None:
            continue
        out = np.full((int(steps) + 1,), np.nan, dtype=float)
        current = start_ip
        cursor = 0
        saw_hold = False
        saw_nonzero_ramp = False
        valid = True
        for seg_len, kind in zip(lengths.tolist(), kinds.tolist(), strict=True):
            segment_steps = int(seg_len)
            if kind == 0:
                out[cursor : cursor + segment_steps + 1] = current
                cursor += segment_steps
                saw_hold = True
                continue
            delta = _profile_ramp_delta(
                cfg,
                current=current,
                direction=int(kind),
                steps=segment_steps,
                lo=lo,
                hi=hi,
                max_delta=max_delta,
                positive_rate_min=positive_rate_min,
                positive_rate_max=positive_delta_rate_max,
                negative_rate_min=negative_rate_min,
                negative_rate_max=negative_delta_rate_max,
                dt=dt,
                rng=rng,
            )
            if delta is None or delta <= 1.0e-6:
                valid = False
                break
            target = float(current + float(kind) * delta)
            out[cursor : cursor + segment_steps + 1] = _monotone_ramp(current, target, segment_steps, smooth=bool(cfg.smooth_ramps))
            cursor += segment_steps
            current = target
            saw_nonzero_ramp = True
        if not valid or not saw_hold or not saw_nonzero_ramp:
            continue
        out[cursor:] = current
        if (
            np.all(np.isfinite(out))
            and np.all(out > 0.0)
            and np.nanmin(out) >= lo
            and np.nanmax(out) <= hi
            and _reference_signed_rate_ok(out, max_positive_rate=positive_peak_rate_max, max_negative_abs_rate=negative_peak_rate_max, dt=dt)
        ):
            return out
    raise ValueError("failed to sample a segmented_profile Ip reference that fits the episode")


def _segmented_profile_rate_bases(cfg: IpReferenceConfig, limits: T15ReferenceLimits) -> tuple[float, float]:
    if cfg.ramp_rate_reference == "robust_mean":
        if limits.positive_ramp_mean_a_per_s is None or limits.negative_ramp_abs_mean_a_per_s is None:
            raise ValueError("reference.ip.ramp_rate_reference=robust_mean requires rebuilt t15_reference_limits.json with ramp mean fields")
        return float(limits.positive_ramp_mean_a_per_s), float(limits.negative_ramp_abs_mean_a_per_s)
    return float(limits.positive_dipdt_p95_a_per_s), float(limits.negative_dipdt_abs_p95_a_per_s)


def _sample_segment_kinds(cfg: IpReferenceConfig, lengths: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    count = int(np.asarray(lengths).size)
    if count < 2:
        return None
    hold_min = max(1, int(cfg.hold_min_steps))
    hold_max = max(hold_min, int(cfg.hold_max_steps))
    lengths_arr = np.asarray(lengths, dtype=int).reshape(-1)
    hold_candidates = np.flatnonzero((lengths_arr >= hold_min) & (lengths_arr <= hold_max))
    if hold_candidates.size == 0:
        return None
    kinds = np.zeros((count,), dtype=np.int8)
    forced_hold = int(rng.choice(hold_candidates))
    hold_eligible = np.zeros((count,), dtype=bool)
    hold_eligible[hold_candidates] = True
    previous_was_ramp = False
    saw_ramp = False
    for idx in range(count):
        if idx == forced_hold:
            kinds[idx] = 0
            previous_was_ramp = False
            continue

        must_hold = previous_was_ramp
        can_hold = bool(hold_eligible[idx])
        if must_hold:
            if not can_hold:
                return None
            kinds[idx] = 0
            previous_was_ramp = False
            continue

        if can_hold and rng.random() < float(cfg.hold_probability):
            kinds[idx] = 0
            previous_was_ramp = False
            continue

        direction = int(rng.choice((-1, 1)))
        kinds[idx] = np.int8(direction)
        previous_was_ramp = True
        saw_ramp = True
    if not saw_ramp:
        ramp_candidates = np.flatnonzero(np.arange(count) != forced_hold)
        if ramp_candidates.size == 0:
            return None
        kinds[int(rng.choice(ramp_candidates))] = np.int8(int(rng.choice((-1, 1))))
    if not np.any(kinds == 0) or not np.any(kinds != 0):
        return None
    return kinds


def _profile_ramp_delta(
    cfg: IpReferenceConfig,
    *,
    current: float,
    direction: int,
    steps: int,
    lo: float,
    hi: float,
    max_delta: float,
    positive_rate_min: float,
    positive_rate_max: float,
    negative_rate_min: float,
    negative_rate_max: float,
    dt: float,
    rng: np.random.Generator,
) -> float | None:
    step_count = max(1, int(steps))
    if int(direction) > 0:
        room = max(float(hi) - float(current), 0.0)
        rate_low = float(positive_rate_min)
        rate_high = float(positive_rate_max)
        delta_high = min(
            float(max_delta) * float(cfg.plateau_max_fraction),
            room,
            rate_high * float(dt) * float(step_count),
        )
        delta_low = max(float(max_delta) * float(cfg.plateau_min_fraction), rate_low * float(dt) * float(step_count))
    else:
        room = max(float(current) - float(lo), 0.0)
        rate_low = float(negative_rate_min)
        rate_high = float(negative_rate_max)
        delta_high = min(
            float(max_delta) * float(cfg.end_max_fraction),
            room,
            rate_high * float(dt) * float(step_count),
        )
        delta_low = max(float(max_delta) * float(cfg.end_min_fraction), rate_low * float(dt) * float(step_count))
    if not np.isfinite(delta_high) or delta_high <= 1.0e-6:
        return None
    delta_low = max(float(delta_low), 1.0e-6)
    if delta_low > delta_high:
        return None
    return float(rng.uniform(delta_low, delta_high))


def _monotone_ramp(start: float, end: float, steps: int, *, smooth: bool) -> np.ndarray:
    n = max(1, int(steps))
    t = np.linspace(0.0, 1.0, n + 1, dtype=float)
    if smooth:
        t = 0.5 - 0.5 * np.cos(np.pi * t)
    return float(start) + (float(end) - float(start)) * t


def _smooth_reference_corners(values: np.ndarray, *, smoothing_steps: int) -> np.ndarray:
    width = max(0, int(smoothing_steps))
    if width < 2 or values.size < 2 * width + 3:
        return np.asarray(values, dtype=float)
    out = np.asarray(values, dtype=float).copy()
    # Light zero-phase smoothing with edge padding preserves the first value
    # after we restore it in the caller and removes actuator-hostile corners.
    w = width if width % 2 == 1 else width + 1
    pad = w // 2
    kernel_x = np.linspace(-np.pi, np.pi, w)
    kernel = 0.5 + 0.5 * np.cos(kernel_x)
    kernel = kernel / np.sum(kernel)
    padded = np.pad(out, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _reference_signed_rate_ok(values: np.ndarray, *, max_positive_rate: float, max_negative_abs_rate: float, dt: float) -> bool:
    rates = np.diff(np.asarray(values, dtype=float)) / max(float(dt), 1.0e-12)
    if not np.all(np.isfinite(rates)):
        return False
    positive = rates[rates > 0.0]
    negative = -rates[rates < 0.0]
    positive_ok = positive.size == 0 or float(np.nanmax(positive)) <= float(max_positive_rate) * (1.0 + 1.0e-6)
    negative_ok = negative.size == 0 or float(np.nanmax(negative)) <= float(max_negative_abs_rate) * (1.0 + 1.0e-6)
    return bool(positive_ok and negative_ok)


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
