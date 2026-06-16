from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokamak_rl_v2.config.schema import Range, ShotFragmentConfig


@dataclass(frozen=True, slots=True)
class ShotFragmentSample:
    ip_reference: np.ndarray


class ShotFragmentLibrary:
    """Generate idealized positive Ip trajectories anchored to the reset Ip0."""

    def __init__(self, config: ShotFragmentConfig, *, dt: float) -> None:
        if config.kind != "idealized_t15_trapezoid":
            raise ValueError("only idealized_t15_trapezoid shot fragments are supported")
        if config.ip_a is None:
            raise ValueError("idealized_t15_trapezoid requires ip_a")
        self.config = config
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("shot fragment dt must be finite and positive")

    def sample(self, rng: np.random.Generator, *, count: int, steps: int, initial_ip: np.ndarray) -> ShotFragmentSample:
        count_i = int(count)
        steps_i = int(steps)
        if count_i <= 0:
            raise ValueError("count must be positive")
        if steps_i <= 0:
            raise ValueError("steps must be positive")
        initial = np.asarray(initial_ip, dtype=float).reshape(-1)
        if initial.shape != (count_i,):
            raise ValueError(f"initial_ip must have shape ({count_i},)")
        offsets = np.arange(steps_i + 1, dtype=float) * self.dt
        ip_ref = np.zeros((count_i, steps_i + 1), dtype=float)
        for b in range(count_i):
            ramp_up_s = _sample_range(rng, self.config.ramp_up_s)
            hold_s = _sample_range(rng, self.config.hold_s)
            ramp_down_s = _sample_range(rng, self.config.ramp_down_s)
            plateau = max(_sample_range(rng, self.config.ip_a.plateau), 1.0)
            end = max(_sample_range(rng, self.config.ip_a.end), 1.0)
            curve = _profile_values(
                offsets,
                start=float(initial[b]),
                plateau=plateau,
                end=end,
                ramp_up_s=ramp_up_s,
                hold_s=hold_s,
                ramp_down_s=ramp_down_s,
            )
            if float(self.config.corner_smoothing_s) > 0.0:
                curve = _smooth_1d(
                    curve,
                    window_steps=max(1, int(round(float(self.config.corner_smoothing_s) / self.dt))),
                )
            curve = np.clip(curve, 1.0, None)
            curve[0] = float(initial[b])
            ip_ref[b] = curve
        return ShotFragmentSample(ip_reference=ip_ref)


def _sample_range(rng: np.random.Generator, item: Range) -> float:
    return float(rng.uniform(float(item.min), float(item.max)))


def _profile_values(
    t: np.ndarray,
    *,
    start: float,
    plateau: float,
    end: float,
    ramp_up_s: float,
    hold_s: float,
    ramp_down_s: float,
) -> np.ndarray:
    times = np.asarray(t, dtype=float)
    ru = max(float(ramp_up_s), 1.0e-9)
    hold = max(float(hold_s), 0.0)
    rd = max(float(ramp_down_s), 1.0e-9)
    down_start = ru + hold
    up_u = np.clip(times / ru, 0.0, 1.0)
    down_u = np.clip((times - down_start) / rd, 0.0, 1.0)
    up = float(start) + (float(plateau) - float(start)) * _smoothstep(up_u)
    down = float(plateau) + (float(end) - float(plateau)) * _smoothstep(down_u)
    return np.where(times <= down_start, up, down)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _smooth_1d(values: np.ndarray, *, window_steps: int) -> np.ndarray:
    width = int(window_steps)
    if width <= 1 or values.size < 3:
        return np.asarray(values, dtype=float).copy()
    if width % 2 == 0:
        width += 1
    pad = width // 2
    kernel = np.full((width,), 1.0 / float(width), dtype=float)
    padded = np.pad(np.asarray(values, dtype=float), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


__all__ = ["ShotFragmentLibrary", "ShotFragmentSample"]
