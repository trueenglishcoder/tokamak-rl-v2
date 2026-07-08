from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import math

import numpy as np

from tokamak_rl_v2.config.machine_envelope import MachineEnvelope, load_machine_envelope

TARGET_SCHEMA = "t15_proxy_target_trajectory_v1"
TARGET_FILE = "t15_target_trajectory_targets.npz"
INITIAL_FILE = "t15_proxy_target_v1_initial_states.npz"
WINDOW_STEPS = 100
DIFFICULTY_ORDER = ("hold", "slow", "medium", "fast")
DEFAULT_BALANCED_DIFFICULTY_FRACTIONS = {"hold": 0.20, "slow": 0.35, "medium": 0.30, "fast": 0.15}


@dataclass(frozen=True, slots=True)
class ParentSeedLibrary:
    ip: np.ndarray
    radii: np.ndarray
    split: np.ndarray
    shot_id: np.ndarray
    source_index: np.ndarray
    time_s: np.ndarray
    difficulty_bin: np.ndarray
    ip0: np.ndarray
    pfc0: np.ndarray
    sol0: np.ndarray
    params0: np.ndarray | None

    @property
    def count(self) -> int:
        return int(self.ip.shape[0])


@dataclass(frozen=True, slots=True)
class TargetDatasetSummary:
    dataset_dir: Path
    target_path: Path
    initial_state_path: Path
    windows: int
    train_windows: int
    holdout_windows: int
    parent_count: int
    window_steps: int
    window_stride_steps: int
    ip_min_a: float
    ip_max_a: float
    max_abs_ip_rate_a_per_s: float
    min_limiter_margin_m: float
    family_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    difficulty_selection: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "dataset_dir": str(self.dataset_dir),
            "target_path": str(self.target_path),
            "initial_state_path": str(self.initial_state_path),
            "windows": int(self.windows),
            "train_windows": int(self.train_windows),
            "holdout_windows": int(self.holdout_windows),
            "parent_count": int(self.parent_count),
            "window_steps": int(self.window_steps),
            "window_stride_steps": int(self.window_stride_steps),
            "ip_min_a": float(self.ip_min_a),
            "ip_max_a": float(self.ip_max_a),
            "max_abs_ip_rate_a_per_s": float(self.max_abs_ip_rate_a_per_s),
            "min_limiter_margin_m": float(self.min_limiter_margin_m),
            "family_counts": dict(self.family_counts),
            "difficulty_counts": dict(self.difficulty_counts),
        }
        if self.difficulty_selection is not None:
            out["difficulty_selection"] = self.difficulty_selection
        return out


def build_target_dataset(
    *,
    target_seed_path: Path,
    initial_library_path: Path,
    machine_envelope_path: Path,
    out_dir: Path,
    limiter_shape: np.ndarray,
    boundary_center: tuple[float, float],
    theta_count: int = 32,
    train_parents: int = 48,
    holdout_parents: int = 8,
    min_steps: int = 1000,
    max_steps: int = 1500,
    window_steps: int = WINDOW_STEPS,
    window_stride_steps: int = 1,
    dt: float = 0.001,
    seed: int = 123,
    split_holdout_fraction: float = 0.15,
    target_train_windows: int | None = None,
    target_holdout_windows: int | None = None,
    target_difficulty_fractions: Mapping[str, float] | None = None,
) -> TargetDatasetSummary:
    """Build target-only Ip/boundary windows with overlapping cuts.

    The produced targets do not include oracle actions.  They are desired
    references for the policy to track by interacting with tokamak-sim.  The
    seed libraries are used only to choose reasonable reset currents and initial
    target shapes; generated parents are cut into overlapping windows using
    ``window_stride_steps``.
    """

    if int(window_stride_steps) <= 0:
        raise ValueError("window_stride_steps must be positive")
    if int(window_stride_steps) > int(window_steps):
        raise ValueError("window_stride_steps should be <= window_steps for overlapping/contiguous coverage")
    if int(window_stride_steps) != 1:
        # The project requirement for this batch is one-step-overlapping windows.
        raise ValueError("t15_proxy_target_v1 requires window_stride_steps=1 so windows overlap at every step")
    if int(min_steps) < int(window_steps):
        raise ValueError("min_steps must be >= window_steps")
    if int(max_steps) < int(min_steps):
        raise ValueError("max_steps must be >= min_steps")
    if float(dt) <= 0.0 or not np.isfinite(float(dt)):
        raise ValueError("dt must be finite and positive")

    envelope = load_machine_envelope(machine_envelope_path)
    rng = np.random.default_rng(int(seed))
    seeds = load_parent_seed_library(target_seed_path=target_seed_path, initial_library_path=initial_library_path)
    theta = np.linspace(-np.pi, np.pi, int(theta_count), endpoint=False, dtype=np.float64)
    limiter = np.asarray(limiter_shape, dtype=np.float64).reshape(-1, 2)
    center = np.asarray(boundary_center, dtype=np.float64).reshape(2)
    limiter_radii = limiter_radii_at_angles(limiter, center=center, theta=theta)
    usable_radii = np.maximum(limiter_radii - float(envelope.proxy_training.boundary_margin_m), 1.0e-6)

    parent_rows: list[dict[str, Any]] = []
    target_train = int(train_parents)
    target_holdout = int(holdout_parents)
    if target_train <= 0 or target_holdout <= 0:
        raise ValueError("train_parents and holdout_parents must both be positive")
    for split_name, count in (("train", target_train), ("holdout", target_holdout)):
        split_indices = np.nonzero(seeds.split == split_name)[0]
        if split_indices.size == 0:
            split_indices = np.arange(seeds.count, dtype=np.int64)
        for parent_local in range(int(count)):
            seed_index = int(split_indices[int(rng.integers(0, int(split_indices.size)))])
            steps = int(rng.integers(int(min_steps), int(max_steps) + 1))
            parent_id = (970000 if split_name == "train" else 980000) + parent_local
            parent_rows.append(
                _generate_parent(
                    seeds=seeds,
                    seed_index=seed_index,
                    parent_id=parent_id,
                    split=split_name,
                    steps=steps,
                    dt=float(dt),
                    theta=theta,
                    usable_radii=usable_radii,
                    envelope=envelope,
                    rng=rng,
                )
            )

    windows = []
    for parent in parent_rows:
        windows.extend(
            _windows_from_parent(
                parent=parent,
                window_steps=int(window_steps),
                stride=int(window_stride_steps),
                dt=float(dt),
                envelope=envelope,
            )
        )
    if not windows:
        raise RuntimeError("target generator produced no windows")

    difficulty_selection: dict[str, Any] | None = None
    normalized_fractions = _normalize_difficulty_fractions(target_difficulty_fractions)
    if normalized_fractions is not None:
        windows, difficulty_selection = _select_balanced_difficulty_windows(
            windows,
            fractions=normalized_fractions,
            target_train_windows=target_train_windows,
            target_holdout_windows=target_holdout_windows,
            rng=rng,
        )
        if not windows:
            raise RuntimeError("balanced target selection produced no windows")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / TARGET_FILE
    initial_path = out_dir / INITIAL_FILE
    _write_target_npz(target_path, windows, theta=theta, envelope=envelope, dt=float(dt), window_steps=int(window_steps))
    _write_initial_npz(initial_path, windows)
    # Compatibility name for existing run tooling that still says oracle/replay.
    # The reference library will prefer TARGET_FILE, so this copy is only for
    # older ad-hoc diagnostics.  It intentionally omits real_jdot_action.
    compat_path = out_dir / "t15_replay_window_oracle_targets.npz"
    if compat_path != target_path:
        _write_target_npz(compat_path, windows, theta=theta, envelope=envelope, dt=float(dt), window_steps=int(window_steps))

    summary = summarize_target_dataset(target_path=target_path, initial_state_path=initial_path, limiter_shape=limiter, boundary_center=tuple(center), dt=float(dt))
    if difficulty_selection is not None:
        summary = replace(summary, difficulty_selection=difficulty_selection)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(out_dir / "target_feasibility_report.md", summary=summary, envelope=envelope)
    return summary


def load_parent_seed_library(*, target_seed_path: Path, initial_library_path: Path) -> ParentSeedLibrary:
    with np.load(Path(target_seed_path), allow_pickle=False) as target, np.load(Path(initial_library_path), allow_pickle=False) as initial:
        required_target = {"ip_target", "boundary_radii", "shot_id", "source_index", "time_s"}
        required_initial = {"ip0", "pfc0", "sol0", "shot_id", "source_index", "time_s"}
        missing_target = sorted(required_target - set(target.files))
        missing_initial = sorted(required_initial - set(initial.files))
        if missing_target or missing_initial:
            raise ValueError(f"seed libraries missing arrays: target={missing_target} initial={missing_initial}")
        ip = np.asarray(target["ip_target"], dtype=np.float64)
        radii = np.asarray(target["boundary_radii"], dtype=np.float64)
        split = np.asarray(target["split"]).astype(str).reshape(-1) if "split" in target.files else np.full(ip.shape[0], "train")
        shot_id = np.asarray(target["shot_id"]).astype(str).reshape(-1)
        source_index = np.asarray(target["source_index"], dtype=np.int64).reshape(-1)
        time_s = np.asarray(target["time_s"], dtype=np.float64).reshape(-1)
        difficulty = np.asarray(target["difficulty_bin"]).astype(str).reshape(-1) if "difficulty_bin" in target.files else np.full(ip.shape[0], "seed")
        pfc0_all = np.asarray(initial["pfc0"], dtype=np.float64)
        sol0_all = np.asarray(initial["sol0"], dtype=np.float64)
        ip0_all = np.asarray(initial["ip0"], dtype=np.float64).reshape(-1)
        params0_all = np.asarray(initial["params0"], dtype=np.float64) if "params0" in initial.files else None
        init_shot = np.asarray(initial["shot_id"]).astype(str).reshape(-1)
        init_source = np.asarray(initial["source_index"], dtype=np.int64).reshape(-1)

    if ip.ndim != 2 or radii.ndim != 3:
        raise ValueError("seed target arrays must have ip_target [N,T] and boundary_radii [N,T,A]")
    count = int(ip.shape[0])
    if radii.shape[0] != count or split.shape != (count,) or shot_id.shape != (count,) or source_index.shape != (count,):
        raise ValueError("seed target metadata shapes do not match target row count")
    key_to_initial = {(str(init_shot[i]), int(init_source[i])): i for i in range(init_shot.shape[0])}
    pfc0 = np.empty((count, pfc0_all.shape[1]), dtype=np.float64)
    sol0 = np.empty((count, sol0_all.shape[1]), dtype=np.float64)
    ip0 = np.empty((count,), dtype=np.float64)
    params0 = None if params0_all is None else np.zeros((count, 5), dtype=np.float64)
    for row in range(count):
        key = (str(shot_id[row]), int(source_index[row]))
        init_row = key_to_initial.get(key)
        if init_row is None:
            init_row = int(np.argmin(np.abs(ip0_all - float(ip[row, 0]))))
        pfc0[row] = pfc0_all[init_row]
        sol0[row] = sol0_all[init_row]
        ip0[row] = float(ip0_all[init_row])
        if params0 is not None and params0_all is not None:
            params0[row] = params0_all[init_row]
    for name, arr in (("ip", ip), ("radii", radii), ("pfc0", pfc0), ("sol0", sol0), ("ip0", ip0)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"seed libraries contain non-finite {name}")
    if np.any(radii <= 0.0):
        raise ValueError("seed boundary radii must be positive")
    return ParentSeedLibrary(
        ip=ip,
        radii=radii,
        split=split,
        shot_id=shot_id,
        source_index=source_index,
        time_s=time_s,
        difficulty_bin=difficulty,
        ip0=ip0,
        pfc0=pfc0,
        sol0=sol0,
        params0=params0,
    )


def _generate_parent(
    *,
    seeds: ParentSeedLibrary,
    seed_index: int,
    parent_id: int,
    split: str,
    steps: int,
    dt: float,
    theta: np.ndarray,
    usable_radii: np.ndarray,
    envelope: MachineEnvelope,
    rng: np.random.Generator,
) -> dict[str, Any]:
    seed_ip0 = float(seeds.ip[int(seed_index), 0])
    ip = _generate_ip_profile(seed_ip0, steps=steps, dt=dt, envelope=envelope, rng=rng)
    radii0 = np.asarray(seeds.radii[int(seed_index), 0, : theta.shape[0]], dtype=np.float64)
    radii0 = np.minimum(np.maximum(radii0, 1.0e-4), usable_radii * 0.98)
    radii, family = _generate_boundary_profile(radii0, steps=steps, dt=dt, usable_radii=usable_radii, envelope=envelope, rng=rng)
    difficulty = _difficulty_bin(ip=ip, radii=radii, dt=dt, envelope=envelope, family=family)
    return {
        "parent_id": int(parent_id),
        "split": str(split),
        "steps": int(steps),
        "dt": float(dt),
        "ip": ip.astype(np.float64),
        "radii": radii.astype(np.float64),
        "family": str(family),
        "difficulty_bin": str(difficulty),
        "seed_shot_id": str(seeds.shot_id[int(seed_index)]),
        "seed_source_index": int(seeds.source_index[int(seed_index)]),
        "seed_time_s": float(seeds.time_s[int(seed_index)]),
        "pfc0": np.asarray(seeds.pfc0[int(seed_index)], dtype=np.float64),
        "sol0": np.asarray(seeds.sol0[int(seed_index)], dtype=np.float64),
        "params0": None if seeds.params0 is None else np.asarray(seeds.params0[int(seed_index)], dtype=np.float64),
    }


def _generate_ip_profile(start: float, *, steps: int, dt: float, envelope: MachineEnvelope, rng: np.random.Generator) -> np.ndarray:
    lo, hi = envelope.proxy_training.ip_range_a
    max_rate = float(envelope.proxy_training.ip_rate_a_per_s)
    values = np.empty((int(steps) + 1,), dtype=np.float64)
    current = float(np.clip(start, lo, hi))
    values[0] = current
    cursor = 0
    family = rng.choice(["hold", "smooth_ramp", "ramp_hold_ramp", "smooth_step", "s_curve"], p=[0.16, 0.28, 0.24, 0.16, 0.16])
    if family == "hold":
        values[:] = current
        return values
    segments = _partition_steps(int(steps), min_segment=80, max_segment=320, rng=rng)
    previous_direction = 0
    for seg_len in segments:
        if cursor >= int(steps):
            break
        seg_len = min(int(seg_len), int(steps) - cursor)
        if family == "ramp_hold_ramp" and rng.random() < 0.35:
            target = current
        else:
            direction = int(rng.choice([-1, 1]))
            if previous_direction and direction != previous_direction and rng.random() < 0.55:
                direction = previous_direction
            rate_fraction = float(rng.uniform(0.08, 0.65))
            max_delta = rate_fraction * max_rate * float(seg_len) * float(dt)
            target = float(np.clip(current + direction * max_delta, lo, hi))
            previous_direction = 0 if abs(target - current) < 1e-9 else (1 if target > current else -1)
        ramp = _smooth_between(current, target, int(seg_len), kind=family)
        values[cursor : cursor + seg_len + 1] = ramp
        current = float(ramp[-1])
        cursor += seg_len
    values[cursor:] = current
    rates = np.diff(values) / float(dt)
    peak = float(np.max(np.abs(rates))) if rates.size else 0.0
    if peak > max_rate:
        values = values[0] + (values - values[0]) * (max_rate / (peak + 1e-12))
    return np.clip(values, lo, hi)


def _generate_boundary_profile(
    radii0: np.ndarray,
    *,
    steps: int,
    dt: float,
    usable_radii: np.ndarray,
    envelope: MachineEnvelope,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    angles = int(radii0.shape[0])
    radii = np.empty((int(steps) + 1, angles), dtype=np.float64)
    radii[0] = np.minimum(np.maximum(radii0, 1.0e-4), usable_radii * 0.985)
    family = str(rng.choice(["boundary_hold", "mean_shift", "tilt_mode", "elliptic_mode", "coupled_shape"], p=[0.16, 0.28, 0.18, 0.20, 0.18]))
    if family == "boundary_hold":
        radii[:] = radii[0]
        return radii, family
    max_rate = float(envelope.proxy_training.boundary_rate_m_per_s)
    # Low-dimensional smooth modes in radius space.  These are target shapes,
    # not inferred actuator answers.
    theta = np.linspace(-np.pi, np.pi, angles, endpoint=False, dtype=np.float64)
    mode_count = {"mean_shift": 1, "tilt_mode": 2, "elliptic_mode": 3, "coupled_shape": 4}[family]
    modes = [np.ones_like(theta)]
    if mode_count >= 2:
        modes.append(np.sin(theta))
    if mode_count >= 3:
        modes.append(np.cos(2.0 * theta))
    if mode_count >= 4:
        modes.append(np.sin(2.0 * theta))
    modes_arr = np.stack(modes, axis=0)
    amplitude_limits = []
    base_room = np.minimum(usable_radii - radii[0], radii[0] - 1.0e-4)
    base_room_scalar = max(float(np.percentile(base_room, 20)), 0.002)
    for _mode in modes:
        amplitude_limits.append(float(np.clip(0.55 * base_room_scalar, 0.002, 0.06)))
    segment_targets = rng.uniform(-1.0, 1.0, size=(len(modes),)) * np.asarray(amplitude_limits, dtype=np.float64)
    segments = _partition_steps(int(steps), min_segment=120, max_segment=360, rng=rng)
    coeff_current = np.zeros((len(modes),), dtype=np.float64)
    cursor = 0
    for seg_len in segments:
        if cursor >= int(steps):
            break
        seg_len = min(int(seg_len), int(steps) - cursor)
        coeff_target = rng.uniform(-1.0, 1.0, size=(len(modes),)) * np.asarray(amplitude_limits, dtype=np.float64)
        if family == "mean_shift":
            coeff_target[1:] = 0.0
        coeff_path = np.stack([_smooth_between(coeff_current[i], coeff_target[i], seg_len, kind="s_curve") for i in range(len(modes))], axis=1)
        delta = coeff_path @ modes_arr
        chunk = radii[0][None, :] + delta
        chunk = _limit_boundary_rate(chunk, previous=radii[cursor], max_rate=max_rate, dt=dt)
        chunk = np.minimum(np.maximum(chunk, 1.0e-4), usable_radii[None, :] * 0.995)
        radii[cursor : cursor + seg_len + 1] = chunk
        coeff_current = coeff_path[-1]
        cursor += seg_len
    radii[cursor:] = radii[cursor]
    radii = _limit_boundary_rate(radii, previous=radii[0], max_rate=max_rate, dt=dt)
    return np.minimum(np.maximum(radii, 1.0e-4), usable_radii[None, :] * 0.995), family


def _limit_boundary_rate(values: np.ndarray, *, previous: np.ndarray, max_rate: float, dt: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    prev = np.asarray(previous, dtype=np.float64).copy()
    max_delta = float(max_rate) * float(dt)
    for i in range(out.shape[0]):
        lo = prev - max_delta
        hi = prev + max_delta
        out[i] = np.minimum(np.maximum(out[i], lo), hi)
        prev = out[i]
    return out


def _windows_from_parent(*, parent: Mapping[str, Any], window_steps: int, stride: int, dt: float, envelope: MachineEnvelope) -> list[dict[str, Any]]:
    ip = np.asarray(parent["ip"], dtype=np.float64)
    radii = np.asarray(parent["radii"], dtype=np.float64)
    if ip.shape[0] != radii.shape[0]:
        raise ValueError("parent ip/radii length mismatch")
    max_start = int(ip.shape[0] - int(window_steps) - 1)
    if max_start < 0:
        return []
    rows = []
    for start in range(0, max_start + 1, int(stride)):
        end = start + int(window_steps)
        ip_window = ip[start : end + 1].astype(np.float32)
        radii_window = radii[start : end + 1].astype(np.float32)
        window_difficulty = _difficulty_bin(
            ip=np.asarray(ip_window, dtype=np.float64),
            radii=np.asarray(radii_window, dtype=np.float64),
            dt=float(dt),
            envelope=envelope,
            family=str(parent["family"]),
        )
        rows.append(
            {
                "shot_id": int(parent["parent_id"]),
                "source_index": int(start),
                "time_s": float(start) * float(dt),
                "split": str(parent["split"]),
                "difficulty_bin": str(window_difficulty),
                "family": str(parent["family"]),
                "seed_shot_id": str(parent["seed_shot_id"]),
                "seed_source_index": int(parent["seed_source_index"]),
                "seed_time_s": float(parent["seed_time_s"]),
                "ip_target": ip_window,
                "boundary_radii": radii_window,
                "ip0": float(ip[start]),
                "pfc0": np.asarray(parent["pfc0"], dtype=np.float32),
                "sol0": np.asarray(parent["sol0"], dtype=np.float32),
                "params0": np.zeros((5,), dtype=np.float32) if parent.get("params0") is None else np.asarray(parent["params0"], dtype=np.float32),
            }
        )
    return rows



def parse_difficulty_fractions(text: str | None) -> dict[str, float] | None:
    """Parse difficulty fractions such as ``hold=0.2,slow=0.35,medium=0.3,fast=0.15``."""

    if text is None or str(text).strip() == "":
        return None
    out: dict[str, float] = {}
    for raw_item in str(text).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"difficulty fraction item must be name=value, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in DIFFICULTY_ORDER:
            raise ValueError(f"unknown difficulty {name!r}; expected one of {DIFFICULTY_ORDER}")
        fraction = float(value)
        if not np.isfinite(fraction) or fraction < 0.0:
            raise ValueError(f"difficulty fraction for {name!r} must be finite and non-negative")
        out[name] = fraction
    return _normalize_difficulty_fractions(out)


def _normalize_difficulty_fractions(fractions: Mapping[str, float] | None) -> dict[str, float] | None:
    if fractions is None:
        return None
    unknown = sorted(set(fractions) - set(DIFFICULTY_ORDER))
    if unknown:
        raise ValueError(f"unknown difficulty fraction keys: {unknown}")
    values = {name: float(fractions.get(name, 0.0)) for name in DIFFICULTY_ORDER}
    for name, value in values.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"difficulty fraction for {name!r} must be finite and non-negative")
    total = float(sum(values.values()))
    if total <= 0.0:
        raise ValueError("difficulty fractions must sum to a positive value")
    return {name: values[name] / total for name in DIFFICULTY_ORDER}


def _count_difficulty(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in DIFFICULTY_ORDER}
    for row in rows:
        name = str(row.get("difficulty_bin", "unknown"))
        counts[name] = int(counts.get(name, 0)) + 1
    return {name: count for name, count in counts.items() if count > 0}


def _quota_counts(total: int, fractions: Mapping[str, float]) -> dict[str, int]:
    total = int(total)
    if total < 0:
        raise ValueError("quota total must be non-negative")
    exact = {name: float(fractions.get(name, 0.0)) * float(total) for name in DIFFICULTY_ORDER}
    quotas = {name: int(math.floor(exact[name])) for name in DIFFICULTY_ORDER}
    remainder = total - int(sum(quotas.values()))
    order = sorted(DIFFICULTY_ORDER, key=lambda name: (exact[name] - math.floor(exact[name]), fractions.get(name, 0.0)), reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def _select_balanced_difficulty_windows(
    windows: list[dict[str, Any]],
    *,
    fractions: Mapping[str, float],
    target_train_windows: int | None,
    target_holdout_windows: int | None,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_counts_all = _count_difficulty(windows)
    selected: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "target_difficulty_fractions": {name: float(fractions.get(name, 0.0)) for name in DIFFICULTY_ORDER},
        "raw_difficulty_counts": raw_counts_all,
        "raw_split_difficulty_counts": {},
        "target_split_windows": {},
        "target_split_difficulty_quotas": {},
        "selected_split_difficulty_counts": {},
        "unused_split_difficulty_counts": {},
        "underfilled_split_difficulties": {},
    }
    for split in ("train", "holdout"):
        split_rows = [row for row in windows if str(row.get("split")) == split]
        requested = target_train_windows if split == "train" else target_holdout_windows
        target_count = len(split_rows) if requested is None else int(requested)
        if target_count < 0:
            raise ValueError(f"target_{split}_windows must be non-negative")
        target_count = min(target_count, len(split_rows))
        chosen, split_meta = _select_balanced_split_windows(split_rows, target_count=target_count, fractions=fractions, rng=rng)
        selected.extend(chosen)
        meta["raw_split_difficulty_counts"][split] = split_meta["raw_counts"]
        meta["target_split_windows"][split] = int(target_count)
        meta["target_split_difficulty_quotas"][split] = split_meta["target_quotas"]
        meta["selected_split_difficulty_counts"][split] = split_meta["selected_counts"]
        meta["unused_split_difficulty_counts"][split] = split_meta["unused_counts"]
        meta["underfilled_split_difficulties"][split] = split_meta["underfilled_difficulties"]
    selected.sort(key=lambda row: (str(row.get("split")), int(row.get("shot_id", 0)), int(row.get("source_index", 0))))
    meta["selected_difficulty_counts"] = _count_difficulty(selected)
    meta["selected_windows"] = int(len(selected))
    return selected, meta


def _select_balanced_split_windows(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    fractions: Mapping[str, float],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_diff: dict[str, list[int]] = {name: [] for name in DIFFICULTY_ORDER}
    overflow: list[int] = []
    for idx, row in enumerate(rows):
        name = str(row.get("difficulty_bin", "unknown"))
        if name in by_diff:
            by_diff[name].append(idx)
        else:
            overflow.append(idx)
    raw_counts = {name: len(indices) for name, indices in by_diff.items() if indices}
    if overflow:
        raw_counts["unknown"] = len(overflow)
    quotas = _quota_counts(int(target_count), fractions)
    selected_indices: list[int] = []
    underfilled: dict[str, dict[str, int]] = {}
    for name in DIFFICULTY_ORDER:
        available = list(by_diff[name])
        rng.shuffle(available)
        take = min(int(quotas[name]), len(available))
        selected_indices.extend(available[:take])
        if take < int(quotas[name]):
            underfilled[name] = {"requested": int(quotas[name]), "available": int(len(available)), "selected": int(take)}
    selected_set = set(selected_indices)
    deficit = int(target_count) - len(selected_indices)
    if deficit > 0:
        remaining = [idx for idx in range(len(rows)) if idx not in selected_set]
        rng.shuffle(remaining)
        selected_indices.extend(remaining[:deficit])
    selected_indices = selected_indices[: int(target_count)]
    selected_set = set(selected_indices)
    selected_rows = [rows[idx] for idx in selected_indices]
    selected_counts = _count_difficulty(selected_rows)
    unused_rows = [rows[idx] for idx in range(len(rows)) if idx not in selected_set]
    unused_counts = _count_difficulty(unused_rows)
    return selected_rows, {
        "raw_counts": raw_counts,
        "target_quotas": {name: int(value) for name, value in quotas.items() if value > 0},
        "selected_counts": selected_counts,
        "unused_counts": unused_counts,
        "underfilled_difficulties": underfilled,
    }

def summarize_target_dataset(
    *,
    target_path: Path,
    initial_state_path: Path,
    limiter_shape: np.ndarray,
    boundary_center: tuple[float, float],
    dt: float,
) -> TargetDatasetSummary:
    with np.load(target_path, allow_pickle=False) as target, np.load(initial_state_path, allow_pickle=False) as initial:
        ip = np.asarray(target["ip_target"], dtype=np.float64)
        radii = np.asarray(target["boundary_radii"], dtype=np.float64)
        split = np.asarray(target["split"]).astype(str).reshape(-1)
        family = np.asarray(target["family"]).astype(str).reshape(-1) if "family" in target.files else np.full(ip.shape[0], "unknown")
        difficulty = np.asarray(target["difficulty_bin"]).astype(str).reshape(-1)
        source_index = np.asarray(target["source_index"], dtype=np.int64).reshape(-1)
        shot_id = np.asarray(target["shot_id"]).astype(str).reshape(-1)
        initial_count = int(np.asarray(initial["ip0"]).shape[0])
    if initial_count != ip.shape[0]:
        raise ValueError(f"initial row count {initial_count} != target row count {ip.shape[0]}")
    theta = np.linspace(-np.pi, np.pi, radii.shape[-1], endpoint=False, dtype=np.float64)
    center = np.asarray(boundary_center, dtype=np.float64).reshape(2)
    points = center[None, None, None, :] + radii[..., None] * np.stack([np.cos(theta), np.sin(theta)], axis=-1)[None, None, :, :]
    margin = signed_limiter_margin(points.reshape(-1, 2), np.asarray(limiter_shape, dtype=np.float64)).reshape(ip.shape[0], ip.shape[1], radii.shape[-1])
    family_counts = {str(name): int(np.sum(family == name)) for name in sorted(set(family.tolist()))}
    difficulty_counts = {str(name): int(np.sum(difficulty == name)) for name in sorted(set(difficulty.tolist()))}
    parent_count = len(set(zip(shot_id.tolist(), source_index.tolist())))
    # Parent count above is not enough because source_index differs per window; use shot_id unique.
    parent_count = len(set(shot_id.tolist()))
    return TargetDatasetSummary(
        dataset_dir=Path(target_path).parent,
        target_path=Path(target_path),
        initial_state_path=Path(initial_state_path),
        windows=int(ip.shape[0]),
        train_windows=int(np.sum(split == "train")),
        holdout_windows=int(np.sum(split == "holdout")),
        parent_count=int(parent_count),
        window_steps=int(ip.shape[1] - 1),
        window_stride_steps=int(np.min(np.diff(np.sort(np.unique(source_index))))) if source_index.size > 1 and np.unique(source_index).size > 1 else 1,
        ip_min_a=float(np.min(ip)),
        ip_max_a=float(np.max(ip)),
        max_abs_ip_rate_a_per_s=float(np.max(np.abs(np.diff(ip, axis=1))) / max(float(dt), 1.0e-12)),
        min_limiter_margin_m=float(np.min(margin)),
        family_counts=family_counts,
        difficulty_counts=difficulty_counts,
    )


def audit_target_dataset(
    *,
    dataset_dir: Path,
    limiter_shape: np.ndarray,
    boundary_center: tuple[float, float],
    dt: float = 0.001,
    out_dir: Path | None = None,
) -> TargetDatasetSummary:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    target_path = dataset_dir / TARGET_FILE
    if not target_path.exists():
        target_path = dataset_dir / "t15_replay_window_oracle_targets.npz"
    initial_path = dataset_dir / INITIAL_FILE
    summary = summarize_target_dataset(
        target_path=target_path,
        initial_state_path=initial_path,
        limiter_shape=limiter_shape,
        boundary_center=boundary_center,
        dt=float(dt),
    )
    if out_dir is not None:
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "target_feasibility_summary.json").write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        _write_report(out_dir / "target_feasibility_report.md", summary=summary, envelope=None)
    return summary


def limiter_radii_at_angles(limiter: np.ndarray, *, center: np.ndarray, theta: np.ndarray) -> np.ndarray:
    poly = np.asarray(limiter, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 3:
        raise ValueError("limiter polygon needs at least 3 points")
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    c = np.asarray(center, dtype=np.float64).reshape(2)
    out = np.empty((theta.shape[0],), dtype=np.float64)
    for i, angle in enumerate(theta.tolist()):
        direction = np.asarray([math.cos(float(angle)), math.sin(float(angle))], dtype=np.float64)
        hits = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
            hit = _ray_segment_intersection(c, direction, a, b)
            if hit is not None:
                hits.append(hit)
        if not hits:
            raise ValueError(f"limiter is not star-shaped around center for theta index {i}")
        out[i] = float(min(hit for hit in hits if hit > 1.0e-9))
    return out


def signed_limiter_margin(points: np.ndarray, limiter: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(limiter, dtype=np.float64).reshape(-1, 2)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    inside = _points_in_polygon_or_on_edge(pts, poly)
    dist = _distance_to_polygon_edges(pts, poly)
    return np.where(inside, dist, -dist)


def _ray_segment_intersection(origin: np.ndarray, direction: np.ndarray, a: np.ndarray, b: np.ndarray) -> float | None:
    v = b - a
    mat = np.column_stack([direction, -v])
    det = float(np.linalg.det(mat))
    if abs(det) < 1.0e-12:
        return None
    rhs = a - origin
    t, u = np.linalg.solve(mat, rhs)
    if t >= -1.0e-10 and -1.0e-10 <= u <= 1.0 + 1.0e-10:
        return float(max(t, 0.0))
    return None


def _points_in_polygon_or_on_edge(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    for xa, ya, xb, yb in zip(x1, y1, x2, y2, strict=True):
        # Edge-on points are valid.
        cross = (x - xa) * (yb - ya) - (y - ya) * (xb - xa)
        dot = (x - xa) * (x - xb) + (y - ya) * (y - yb)
        on = (np.abs(cross) <= 1.0e-10) & (dot <= 1.0e-10)
        inside |= on
        intersects = ((ya > y) != (yb > y)) & (x < (xb - xa) * (y - ya) / ((yb - ya) + 1.0e-300) + xa)
        inside ^= intersects
    return inside


def _distance_to_polygon_edges(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    out = np.full((points.shape[0],), np.inf, dtype=np.float64)
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 0.0:
            continue
        t = np.clip(((points - a[None, :]) @ ab) / denom, 0.0, 1.0)
        nearest = a[None, :] + t[:, None] * ab[None, :]
        d = np.sqrt(np.sum((points - nearest) ** 2, axis=1))
        out = np.minimum(out, d)
    return out


def _partition_steps(total: int, *, min_segment: int, max_segment: int, rng: np.random.Generator) -> list[int]:
    remaining = int(total)
    out = []
    while remaining > 0:
        if remaining <= int(max_segment):
            out.append(remaining)
            break
        seg = int(rng.integers(int(min_segment), int(max_segment) + 1))
        out.append(seg)
        remaining -= seg
    return out


def _smooth_between(start: float, target: float, steps: int, *, kind: str) -> np.ndarray:
    u = np.linspace(0.0, 1.0, int(steps) + 1, dtype=np.float64)
    if kind in {"smooth_step", "s_curve", "ramp_hold_ramp"}:
        w = u * u * (3.0 - 2.0 * u)
    else:
        w = u
    return float(start) + (float(target) - float(start)) * w


def _difficulty_bin(*, ip: np.ndarray, radii: np.ndarray, dt: float, envelope: MachineEnvelope, family: str) -> str:
    ip_rate = float(np.max(np.abs(np.diff(ip))) / max(float(dt), 1.0e-12)) if ip.size > 1 else 0.0
    br = float(np.max(np.abs(np.diff(radii, axis=0))) / max(float(dt), 1.0e-12)) if radii.shape[0] > 1 else 0.0
    ip_score = ip_rate / max(float(envelope.proxy_training.ip_rate_a_per_s), 1.0e-12)
    b_score = br / max(float(envelope.proxy_training.boundary_rate_m_per_s), 1.0e-12)
    score = max(ip_score, b_score)
    if family.endswith("hold") or score < 0.05:
        return "hold"
    if score < 0.25:
        return "slow"
    if score < 0.55:
        return "medium"
    return "fast"


def _write_target_npz(path: Path, rows: list[Mapping[str, Any]], *, theta: np.ndarray, envelope: MachineEnvelope, dt: float, window_steps: int) -> None:
    def arr(name: str, dtype=None):
        return np.asarray([row[name] for row in rows], dtype=dtype)

    ip = np.stack([np.asarray(row["ip_target"], dtype=np.float32) for row in rows], axis=0)
    radii = np.stack([np.asarray(row["boundary_radii"], dtype=np.float32) for row in rows], axis=0)
    if ip.shape[1] != int(window_steps) + 1 or radii.shape[1] != int(window_steps) + 1:
        raise ValueError("window arrays have unexpected length")
    np.savez_compressed(
        path,
        schema=np.asarray([TARGET_SCHEMA]),
        target_only=np.asarray([True]),
        shot_id=arr("shot_id", np.int64),
        source_index=arr("source_index", np.int64),
        time_s=arr("time_s", np.float64),
        split=arr("split").astype(str),
        difficulty_bin=arr("difficulty_bin").astype(str),
        family=arr("family").astype(str),
        seed_shot_id=arr("seed_shot_id").astype(str),
        seed_source_index=arr("seed_source_index", np.int64),
        seed_time_s=arr("seed_time_s", np.float64),
        ip_target=ip,
        boundary_radii=radii,
        theta=theta.astype(np.float64),
        dt=np.asarray([float(dt)], dtype=np.float64),
        window_steps=np.asarray([int(window_steps)], dtype=np.int64),
        proxy_ip_range_a=np.asarray(envelope.proxy_training.ip_range_a, dtype=np.float64),
        proxy_ip_rate_a_per_s=np.asarray([envelope.proxy_training.ip_rate_a_per_s], dtype=np.float64),
        proxy_boundary_rate_m_per_s=np.asarray([envelope.proxy_training.boundary_rate_m_per_s], dtype=np.float64),
    )


def _write_initial_npz(path: Path, rows: list[Mapping[str, Any]]) -> None:
    def arr(name: str, dtype=None):
        return np.asarray([row[name] for row in rows], dtype=dtype)

    np.savez_compressed(
        path,
        schema=np.asarray([TARGET_SCHEMA + "_initial_states"]),
        shot_id=arr("shot_id", np.int64),
        source_index=arr("source_index", np.int64),
        time_s=arr("time_s", np.float64),
        split=arr("split").astype(str),
        difficulty_bin=arr("difficulty_bin").astype(str),
        ip0=arr("ip0", np.float32),
        pfc0=np.stack([np.asarray(row["pfc0"], dtype=np.float32) for row in rows], axis=0),
        sol0=np.stack([np.asarray(row["sol0"], dtype=np.float32) for row in rows], axis=0),
        params0=np.stack([np.asarray(row["params0"], dtype=np.float32) for row in rows], axis=0),
    )


def _write_report(path: Path, *, summary: TargetDatasetSummary, envelope: MachineEnvelope | None) -> None:
    lines = [
        "# T15 proxy target-only dataset report",
        "",
        "This dataset contains desired target trajectories only. It does not contain oracle coil actions.",
        "Windows are one-step overlapping cuts from longer generated parents.",
        "",
        f"- windows: {summary.windows}",
        f"- train windows: {summary.train_windows}",
        f"- holdout windows: {summary.holdout_windows}",
        f"- parent count: {summary.parent_count}",
        f"- window steps: {summary.window_steps}",
        f"- window stride steps: {summary.window_stride_steps}",
        f"- Ip range A: [{summary.ip_min_a:.6g}, {summary.ip_max_a:.6g}]",
        f"- max abs Ip rate A/s: {summary.max_abs_ip_rate_a_per_s:.6g}",
        f"- minimum limiter margin m: {summary.min_limiter_margin_m:.6g}",
        "",
        "## Family counts",
        "",
    ]
    for key, value in sorted(summary.family_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Difficulty counts", ""])
    for key, value in sorted(summary.difficulty_counts.items()):
        lines.append(f"- {key}: {value}")
    if summary.difficulty_selection is not None:
        lines.extend(["", "## Difficulty selection", "", "```json", json.dumps(summary.difficulty_selection, indent=2, sort_keys=True), "```"])
    if envelope is not None:
        lines.extend(
            [
                "",
                "## Proxy envelope",
                "",
                f"- Ip range A: {list(envelope.proxy_training.ip_range_a)}",
                f"- Ip rate A/s: {envelope.proxy_training.ip_rate_a_per_s:g}",
                f"- boundary margin m: {envelope.proxy_training.boundary_margin_m:g}",
                f"- boundary rate m/s: {envelope.proxy_training.boundary_rate_m_per_s:g}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
