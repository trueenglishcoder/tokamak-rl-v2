from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokamak_rl_v2.config.schema import Range, ShotFragmentConfig, TrapezoidValueRanges


@dataclass(frozen=True, slots=True)
class ShotFragmentSample:
    ip0: np.ndarray
    pfc_currents: np.ndarray
    sol_currents: np.ndarray
    ip_reference: np.ndarray
    pfc_current_reference: np.ndarray
    sol_current_reference: np.ndarray
    shot_ids: tuple[str, ...]
    start_times_s: np.ndarray


class ShotFragmentLibrary:
    """Generate idealized T15-like trapezoid fragments without replaying shot files."""

    def __init__(self, config: ShotFragmentConfig, *, n_pfc: int, n_sol: int, dt: float) -> None:
        if config.kind != "idealized_t15_trapezoid":
            raise ValueError("only idealized_t15_trapezoid shot fragments are supported")
        if config.ip_a is None:
            raise ValueError("idealized_t15_trapezoid requires ip_a")
        self.config = config
        self.n_pfc = int(n_pfc)
        self.n_sol = int(n_sol)
        self.dt = float(dt)
        if self.n_pfc <= 0 or self.n_sol <= 0:
            raise ValueError("shot fragments require positive PFC and SOL counts")
        if len(config.pfc_currents) != self.n_pfc:
            raise ValueError("pfc current profile count does not match simulator")
        if len(config.sol_currents) != self.n_sol:
            raise ValueError("sol current profile count does not match simulator")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("shot fragment dt must be finite and positive")

    def sample(self, rng: np.random.Generator, *, count: int, steps: int) -> ShotFragmentSample:
        count_i = int(count)
        steps_i = int(steps)
        if count_i <= 0:
            raise ValueError("count must be positive")
        if steps_i <= 0:
            raise ValueError("steps must be positive")
        offsets = np.arange(steps_i + 1, dtype=float) * self.dt
        episode_duration = float(steps_i) * self.dt

        ip_ref = np.zeros((count_i, steps_i + 1), dtype=float)
        pfc_ref = np.zeros((count_i, steps_i + 1, self.n_pfc), dtype=float)
        sol_ref = np.zeros((count_i, steps_i + 1, self.n_sol), dtype=float)
        ip0 = np.zeros((count_i,), dtype=float)
        pfc = np.zeros((count_i, self.n_pfc), dtype=float)
        sol = np.zeros((count_i, self.n_sol), dtype=float)
        start_times = np.zeros((count_i,), dtype=float)
        labels: list[str] = []

        for b in range(count_i):
            profile = self._sample_profile(rng, min_duration_s=episode_duration + float(self.config.trim_end_s))
            start_min = float(self.config.start_time_min_s)
            start_max = profile["duration_s"] - episode_duration - float(self.config.trim_end_s)
            if self.config.start_time_max_s is not None:
                start_max = min(start_max, float(self.config.start_time_max_s))
            if start_max < start_min:
                raise ValueError("sampled trapezoid profile is too short for requested fragment")
            start = float(rng.uniform(start_min, start_max)) if start_max > start_min else float(start_min)
            query_t = start + offsets
            ip_curve = _profile_values(
                query_t,
                start=float(profile["ip_start"]),
                plateau=float(profile["ip_plateau"]),
                end=float(profile["ip_end"]),
                ramp_up_s=float(profile["ramp_up_s"]),
                hold_s=float(profile["hold_s"]),
                ramp_down_s=float(profile["ramp_down_s"]),
            )
            if float(self.config.corner_smoothing_s) > 0.0:
                ip_curve = _smooth_1d(ip_curve, window_steps=max(1, int(round(float(self.config.corner_smoothing_s) / self.dt))))
            ip_curve = np.clip(ip_curve, 1.0, None)
            sol_curve = _current_profile_values(
                query_t,
                profiles=profile["sol_profiles"],
                ramp_up_s=float(profile["ramp_up_s"]),
                hold_s=float(profile["hold_s"]),
                ramp_down_s=float(profile["ramp_down_s"]),
            )
            pfc_curve = _current_profile_values(
                query_t,
                profiles=profile["pfc_profiles"],
                ramp_up_s=float(profile["ramp_up_s"]),
                hold_s=float(profile["hold_s"]),
                ramp_down_s=float(profile["ramp_down_s"]),
            )
            if float(self.config.corner_smoothing_s) > 0.0:
                window = max(1, int(round(float(self.config.corner_smoothing_s) / self.dt)))
                sol_curve = _smooth_2d(sol_curve, window_steps=window)
                pfc_curve = _smooth_2d(pfc_curve, window_steps=window)
            ip_ref[b] = ip_curve
            ip0[b] = float(ip_curve[0])
            sol_ref[b] = sol_curve
            pfc_ref[b] = pfc_curve
            sol[b] = sol_curve[0]
            pfc[b] = pfc_curve[0]
            start_times[b] = start
            labels.append("idealized_t15_trapezoid")

        return ShotFragmentSample(
            ip0=ip0,
            pfc_currents=pfc,
            sol_currents=sol,
            ip_reference=ip_ref,
            pfc_current_reference=pfc_ref,
            sol_current_reference=sol_ref,
            shot_ids=tuple(labels),
            start_times_s=start_times,
        )

    def _sample_profile(self, rng: np.random.Generator, *, min_duration_s: float) -> dict[str, object]:
        for _attempt in range(256):
            ramp_up_s = _sample_range(rng, self.config.ramp_up_s)
            hold_s = _sample_range(rng, self.config.hold_s)
            ramp_down_s = _sample_range(rng, self.config.ramp_down_s)
            duration_s = ramp_up_s + hold_s + ramp_down_s
            if duration_s < float(min_duration_s):
                continue
            ip = self.config.ip_a
            assert ip is not None
            return {
                "duration_s": duration_s,
                "ramp_up_s": ramp_up_s,
                "hold_s": hold_s,
                "ramp_down_s": ramp_down_s,
                "ip_start": _sample_range(rng, ip.start),
                "ip_plateau": _sample_range(rng, ip.plateau),
                "ip_end": _sample_range(rng, ip.end),
                "sol_profiles": tuple(_sample_value_profile(rng, item) for item in self.config.sol_currents),
                "pfc_profiles": tuple(_sample_value_profile(rng, item) for item in self.config.pfc_currents),
            }
        raise ValueError("could not sample a long enough idealized trapezoid profile")

def _sample_range(rng: np.random.Generator, item: Range) -> float:
    return float(rng.uniform(float(item.min), float(item.max)))


def _sample_value_profile(rng: np.random.Generator, item: TrapezoidValueRanges) -> tuple[float, float, float]:
    return (_sample_range(rng, item.start), _sample_range(rng, item.plateau), _sample_range(rng, item.end))


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


def _current_profile_values(
    t: np.ndarray,
    *,
    profiles: tuple[tuple[float, float, float], ...],
    ramp_up_s: float,
    hold_s: float,
    ramp_down_s: float,
) -> np.ndarray:
    values = [
        _profile_values(
            t,
            start=start,
            plateau=plateau,
            end=end,
            ramp_up_s=ramp_up_s,
            hold_s=hold_s,
            ramp_down_s=ramp_down_s,
        )
        for start, plateau, end in profiles
    ]
    return np.stack(values, axis=1)


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


def _smooth_2d(values: np.ndarray, *, window_steps: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("values must be 2D")
    return np.stack([_smooth_1d(arr[:, i], window_steps=window_steps) for i in range(arr.shape[1])], axis=1)


__all__ = ["ShotFragmentLibrary", "ShotFragmentSample"]
