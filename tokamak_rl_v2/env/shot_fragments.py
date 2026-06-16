from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tokamak_rl_v2.config.schema import ShotFragmentConfig


@dataclass(frozen=True, slots=True)
class ShotFragmentTemplate:
    shot_id: str
    plateau_ip: float
    ramp_up_end_s: float
    trace_end_s: float
    ip_at_1s: float


@dataclass(frozen=True, slots=True)
class ShotFragmentSample:
    ip_reference: np.ndarray
    shot_ids: tuple[str, ...]


class ShotFragmentLibrary:
    """Build one coherent 1-second ramp-and-hold Ip template per real T15 shot."""

    def __init__(self, config: ShotFragmentConfig, *, dt: float, config_path: Path) -> None:
        if config.kind != "idealized_t15_trapezoid":
            raise ValueError("only idealized_t15_trapezoid shot fragments are supported")
        self.config = config
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("shot fragment dt must be finite and positive")
        sim_root = Path(config_path).resolve().parents[1]
        ip_root = sim_root / "data" / "t15_data_new" / "ip"
        templates: dict[str, ShotFragmentTemplate] = {}
        for shot_id in config.shot_ids:
            path = ip_root / f"t15md_{shot_id}_ip.csv"
            if not path.exists():
                raise FileNotFoundError(f"missing raw T15 Ip trace: {path}")
            times_s, ip_a = _load_ip_trace(path)
            templates[str(shot_id)] = _build_template(
                shot_id=str(shot_id),
                times_s=times_s,
                ip_a=ip_a,
                corner_smoothing_s=float(config.corner_smoothing_s),
                default_dt=self.dt,
            )
        self.templates = templates

    def sample(
        self,
        rng: np.random.Generator,
        *,
        count: int,
        steps: int,
        initial_ip: np.ndarray,
        shot_ids: tuple[str, ...],
    ) -> ShotFragmentSample:
        del rng
        count_i = int(count)
        steps_i = int(steps)
        if count_i <= 0:
            raise ValueError("count must be positive")
        if steps_i <= 0:
            raise ValueError("steps must be positive")
        initial = np.asarray(initial_ip, dtype=float).reshape(-1)
        selected_shot_ids = tuple(str(shot_id) for shot_id in shot_ids)
        if initial.shape != (count_i,):
            raise ValueError(f"initial_ip must have shape ({count_i},)")
        if len(selected_shot_ids) != count_i:
            raise ValueError(f"shot_ids must contain {count_i} entries")
        offsets = np.arange(steps_i + 1, dtype=float) * self.dt
        ip_ref = np.zeros((count_i, steps_i + 1), dtype=float)
        for b, shot_id in enumerate(selected_shot_ids):
            if shot_id not in self.templates:
                raise ValueError(f"shot fragment template is unavailable for shot {shot_id}")
            template = self.templates[shot_id]
            plateau_ip = max(float(template.plateau_ip), float(initial[b]), 1.0)
            curve = _ramp_hold_profile(
                offsets,
                start=float(initial[b]),
                plateau=plateau_ip,
                ramp_up_end_s=float(template.ramp_up_end_s),
            )
            curve = np.clip(curve, 1.0, None)
            curve[0] = float(initial[b])
            ip_ref[b] = curve
        return ShotFragmentSample(ip_reference=ip_ref, shot_ids=selected_shot_ids)


def _build_template(
    *,
    shot_id: str,
    times_s: np.ndarray,
    ip_a: np.ndarray,
    corner_smoothing_s: float,
    default_dt: float,
) -> ShotFragmentTemplate:
    trace_end_s = float(times_s[-1])
    plateau_end_s = min(1.0, trace_end_s)
    smoothed_ip = _smoothed_trace(ip_a, times_s=times_s, corner_smoothing_s=corner_smoothing_s, default_dt=default_dt)
    plateau_ip = _estimate_plateau_ip(times_s, ip_a, trace_end_s=trace_end_s, plateau_end_s=plateau_end_s, smoothed_ip=smoothed_ip)
    ramp_up_end_s = _first_crossing_time(times_s, smoothed_ip, threshold=0.98 * plateau_ip)
    ramp_up_end_s = float(np.clip(ramp_up_end_s, default_dt, max(plateau_end_s, default_dt)))
    ip_at_1s = float(_interpolate_at(times_s, ip_a, t=1.0))
    return ShotFragmentTemplate(
        shot_id=shot_id,
        plateau_ip=float(max(plateau_ip, 1.0)),
        ramp_up_end_s=ramp_up_end_s,
        trace_end_s=trace_end_s,
        ip_at_1s=ip_at_1s,
    )


def _estimate_plateau_ip(
    times_s: np.ndarray,
    ip_a: np.ndarray,
    *,
    trace_end_s: float,
    plateau_end_s: float,
    smoothed_ip: np.ndarray,
) -> float:
    if plateau_end_s <= 0.0:
        return float(max(ip_a[-1], 1.0))
    if trace_end_s >= 0.55:
        plateau_ip = _median_over_window(times_s, ip_a, start_s=0.55, end_s=plateau_end_s)
    else:
        plateau_ip = _median_over_window(times_s, ip_a, start_s=max(0.0, trace_end_s * 0.8), end_s=plateau_end_s)
    for _ in range(2):
        ramp_up_end_s = _first_crossing_time(times_s, smoothed_ip, threshold=0.98 * plateau_ip)
        plateau_start_s = max(0.55, ramp_up_end_s) if trace_end_s >= 0.55 else ramp_up_end_s
        plateau_ip = _median_over_window(times_s, ip_a, start_s=plateau_start_s, end_s=plateau_end_s)
    return float(max(plateau_ip, 1.0))


def _load_ip_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, delimiter=";", dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 2:
        raise ValueError(f"raw Ip trace must have exactly two columns: {path}")
    times_s = np.asarray(raw[:, 0], dtype=float).reshape(-1)
    ip_a = np.asarray(raw[:, 1], dtype=float).reshape(-1)
    finite = np.isfinite(times_s) & np.isfinite(ip_a)
    if not np.any(finite):
        raise ValueError(f"raw Ip trace has no finite samples: {path}")
    times_s = times_s[finite]
    ip_a = ip_a[finite]
    order = np.argsort(times_s, kind="stable")
    times_s = times_s[order]
    ip_a = ip_a[order]
    if times_s.size < 2:
        raise ValueError(f"raw Ip trace must contain at least two samples: {path}")
    return times_s, ip_a


def _median_over_window(times_s: np.ndarray, values: np.ndarray, *, start_s: float, end_s: float) -> float:
    start = float(np.clip(start_s, float(times_s[0]), float(times_s[-1])))
    end = float(np.clip(end_s, start, float(times_s[-1])))
    mask = (times_s >= start) & (times_s <= end)
    if np.any(mask):
        return float(np.median(values[mask]))
    idx = int(np.argmin(np.abs(times_s - start)))
    return float(values[idx])


def _first_crossing_time(times_s: np.ndarray, values: np.ndarray, *, threshold: float) -> float:
    target = float(threshold)
    meets = np.flatnonzero(values >= target)
    if meets.size == 0:
        return float(times_s[-1])
    idx = int(meets[0])
    if idx == 0:
        return float(times_s[0])
    left_t = float(times_s[idx - 1])
    right_t = float(times_s[idx])
    left_v = float(values[idx - 1])
    right_v = float(values[idx])
    if not np.isfinite(left_v) or not np.isfinite(right_v) or right_t <= left_t or right_v <= left_v:
        return right_t
    alpha = (target - left_v) / max(right_v - left_v, 1.0e-12)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return left_t + alpha * (right_t - left_t)


def _interpolate_at(times_s: np.ndarray, values: np.ndarray, *, t: float) -> float:
    target = float(np.clip(t, float(times_s[0]), float(times_s[-1])))
    return float(np.interp(target, times_s, values))


def _smoothed_trace(ip_a: np.ndarray, *, times_s: np.ndarray, corner_smoothing_s: float, default_dt: float) -> np.ndarray:
    if float(corner_smoothing_s) <= 0.0:
        return np.asarray(ip_a, dtype=float).copy()
    dt_candidates = np.diff(times_s)
    dt_candidates = dt_candidates[np.isfinite(dt_candidates) & (dt_candidates > 0.0)]
    sample_dt = float(np.median(dt_candidates)) if dt_candidates.size else float(default_dt)
    window_steps = max(1, int(round(float(corner_smoothing_s) / max(sample_dt, 1.0e-12))))
    return _smooth_1d(ip_a, window_steps=window_steps)


def _ramp_hold_profile(t: np.ndarray, *, start: float, plateau: float, ramp_up_end_s: float) -> np.ndarray:
    times = np.asarray(t, dtype=float)
    ramp_end = max(float(ramp_up_end_s), 1.0e-9)
    u = np.clip(times / ramp_end, 0.0, 1.0)
    ramp = float(start) + (float(plateau) - float(start)) * _smoothstep(u)
    return np.where(times <= ramp_end, ramp, float(plateau))


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


__all__ = ["ShotFragmentLibrary", "ShotFragmentSample", "ShotFragmentTemplate"]
