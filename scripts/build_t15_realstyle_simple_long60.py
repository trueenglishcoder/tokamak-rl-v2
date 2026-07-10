#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_t15_synthetic_long30_trim50_plain_gpu1e6_oracle_windows import (
    DT,
    WINDOW_STEPS,
    _difficulty_bin,
    _write_initial_library,
    _write_oracle_npz,
)
from build_t15_synthetic_long_preview import RealSpace, _features_to_radii, _load_real_space, _repo_path, _require_inputs


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PARENT_ID_BASE = 960000
HOLDOUT_PARENT_ID_BASE = 970000
ALLOWED_FAMILIES = ("ramp_up_hold_ramp_down", "ramp_up_hold", "hold_ramp_down", "ramp_up")


@dataclass(frozen=True, slots=True)
class ResetPool:
    features: np.ndarray
    currents: np.ndarray
    shot_id: np.ndarray
    source_index: np.ndarray
    time_s: np.ndarray


@dataclass(frozen=True, slots=True)
class StyleStats:
    segment_lengths: np.ndarray
    ip_up_rate: np.ndarray
    ip_down_rate: np.ndarray
    feature_rate_abs: np.ndarray
    real_parent_ip_range: np.ndarray
    real_parent_radii_mean_range: np.ndarray


@dataclass(frozen=True, slots=True)
class BuildSpace:
    real: RealSpace
    split: str
    reset_pool: ResetPool
    style: StyleStats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build t15_realstyle_simple_long60: simple real-style 0.5-1.5 s synthetic "
            "T15 parent shots from first-500 real reset states, cut into all overlapping "
            "100-step replay-window oracle targets."
        )
    )
    parser.add_argument(
        "--oracle-target",
        type=Path,
        default=Path(
            "data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/"
            "t15_replay_window_oracle_targets.npz"
        ),
    )
    parser.add_argument(
        "--initial-library",
        type=Path,
        default=Path("data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/t15_realstyle_simple_long60"))
    parser.add_argument(
        "--initial-library-out",
        type=Path,
        default=Path("data/processed/t15_realstyle_simple_long60_initial_states.npz"),
    )
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-examples", type=int, default=6)
    parser.add_argument("--train-parents", type=int, default=60)
    parser.add_argument("--holdout-parents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--min-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--reset-max-source-index", type=int, default=500)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--wiggle-room", type=float, default=1.15)
    parser.add_argument("--radii-margin-m", type=float, default=0.025)
    parser.add_argument("--current-envelope-margin", type=float, default=0.08)
    parser.add_argument("--max-cloud-rows", type=int, default=20000)
    parser.add_argument("--knn-k", type=int, default=16)
    args = parser.parse_args(argv)

    if int(args.min_steps) < WINDOW_STEPS:
        raise ValueError(f"--min-steps must be >= {WINDOW_STEPS}")
    if int(args.max_steps) < int(args.min_steps):
        raise ValueError("--max-steps must be >= --min-steps")
    if int(args.reset_max_source_index) <= 0:
        raise ValueError("--reset-max-source-index must be positive")

    target_path = _repo_path(args.oracle_target)
    initial_path = _repo_path(args.initial_library)
    out_dir = _repo_path(args.out_dir)
    initial_out = _repo_path(args.initial_library_out)
    _require_inputs(target_path=target_path, initial_path=initial_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    train_space = _load_build_space(target_path=target_path, initial_path=initial_path, split="train", args=args, rng=rng)
    holdout_space = _load_build_space(
        target_path=target_path, initial_path=initial_path, split="holdout", args=args, rng=rng
    )

    if bool(args.preview_only):
        parents = _make_parent_set(
            space=train_space,
            count=int(args.preview_examples),
            parent_id_base=TRAIN_PARENT_ID_BASE,
            split="preview",
            args=args,
            rng=rng,
        )
        _write_parent_outputs(out_dir=out_dir, parents=parents, preview_count=len(parents))
        summary = _summary(
            args=args,
            target_path=target_path,
            initial_path=initial_path,
            out_dir=out_dir,
            rows=[],
            parents=parents,
            train_space=train_space,
            holdout_space=holdout_space,
            initial_out=initial_out,
            preview_only=True,
        )
        _write_summary_files(out_dir=out_dir, summary=summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0

    train_parents = _make_parent_set(
        space=train_space,
        count=int(args.train_parents),
        parent_id_base=TRAIN_PARENT_ID_BASE,
        split="train",
        args=args,
        rng=rng,
    )
    holdout_parents = _make_parent_set(
        space=holdout_space,
        count=int(args.holdout_parents),
        parent_id_base=HOLDOUT_PARENT_ID_BASE,
        split="holdout",
        args=args,
        rng=rng,
    )
    parents = [*train_parents, *holdout_parents]

    rows: list[dict[str, Any]] = []
    for parent in parents:
        rows.extend(_windows_from_parent(parent=parent, current_limits=train_space.real.current_limits))
    if not rows:
        raise RuntimeError("t15_realstyle_simple_long60 produced zero windows")

    oracle_path = out_dir / "t15_replay_window_oracle_targets.npz"
    _write_oracle_npz(
        oracle_path,
        rows,
        current_limits=train_space.real.current_limits,
        derivative_limits=train_space.real.derivative_limits,
    )
    _write_initial_library(initial_out, rows)
    _write_parent_outputs(out_dir=out_dir, parents=parents, preview_count=int(args.preview_examples))

    summary = _summary(
        args=args,
        target_path=target_path,
        initial_path=initial_path,
        out_dir=out_dir,
        rows=rows,
        parents=parents,
        train_space=train_space,
        holdout_space=holdout_space,
        initial_out=initial_out,
        preview_only=False,
    )
    _write_summary_files(out_dir=out_dir, summary=summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def _load_build_space(
    *,
    target_path: Path,
    initial_path: Path,
    split: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> BuildSpace:
    real = _load_real_space(
        target_path=target_path,
        initial_path=initial_path,
        dt=DT,
        pca_components=int(args.pca_components),
        wiggle_room=float(args.wiggle_room),
        radii_margin_m=float(args.radii_margin_m),
        current_envelope_margin=float(args.current_envelope_margin),
        max_cloud_rows=int(args.max_cloud_rows),
        safe_split=str(split),
        rng=rng,
    )
    reset_pool = _load_reset_pool(
        target_path=target_path,
        initial_path=initial_path,
        real=real,
        split=split,
        reset_max_source_index=int(args.reset_max_source_index),
    )
    style = _load_style_stats(target_path=target_path, real=real, split=split)
    print(
        "[realstyle-simple] "
        f"split={split} reset_rows={reset_pool.features.shape[0]} "
        f"segment_samples={style.segment_lengths.shape[0]}",
        flush=True,
    )
    return BuildSpace(real=real, split=str(split), reset_pool=reset_pool, style=style)


def _load_reset_pool(
    *,
    target_path: Path,
    initial_path: Path,
    real: RealSpace,
    split: str,
    reset_max_source_index: int,
) -> ResetPool:
    with np.load(target_path, allow_pickle=False) as target, np.load(initial_path, allow_pickle=False) as init:
        split_arr = np.asarray(target["split"]).astype(str)
        source_index = np.asarray(target["source_index"], dtype=np.int64)
        mask = (split_arr == str(split)) & (source_index < int(reset_max_source_index))
        if not np.any(mask):
            raise ValueError(
                f"no reset rows for split={split!r} with source_index < {int(reset_max_source_index)}"
            )
        ip0 = np.asarray(target["ip_target"], dtype=np.float64)[mask, 0]
        radii0 = np.asarray(target["boundary_radii"], dtype=np.float64)[mask, 0, :]
        pfc0 = np.asarray(init["pfc0"], dtype=np.float64)[mask]
        sol0 = np.asarray(init["sol0"], dtype=np.float64)[mask]
        shot_id = np.asarray(target["shot_id"], dtype=np.int64)[mask]
        src = source_index[mask]
        time_s = np.asarray(target["time_s"], dtype=np.float64)[mask]

    coeffs0 = (radii0 - real.radii_mean.reshape(1, -1)) @ real.pca_components.T
    features = np.concatenate([ip0.reshape(-1, 1), coeffs0], axis=1)
    currents = np.concatenate([pfc0, sol0], axis=1)
    return ResetPool(
        features=features.astype(np.float64),
        currents=currents.astype(np.float64),
        shot_id=shot_id,
        source_index=src,
        time_s=time_s,
    )


def _load_style_stats(*, target_path: Path, real: RealSpace, split: str) -> StyleStats:
    series = _stitch_real_series(target_path=target_path, real=real, split=split)
    segment_lengths: list[int] = []
    ip_up_rates: list[float] = []
    ip_down_rates: list[float] = []
    feature_rate_abs: list[np.ndarray] = []
    parent_ip_range: list[float] = []
    parent_radii_mean_range: list[float] = []

    for item in series:
        features = item["features"]
        radii = item["radii"]
        if features.shape[0] < 3:
            continue
        df = np.diff(features, axis=0) / DT
        ip_rate = df[:, 0]
        pos = ip_rate[ip_rate > 1000.0]
        neg = -ip_rate[ip_rate < -1000.0]
        if pos.size:
            ip_up_rates.extend(pos.tolist())
        if neg.size:
            ip_down_rates.extend(neg.tolist())
        feature_rate_abs.append(np.abs(df[:, 1:]))
        parent_ip_range.append(float(np.ptp(features[:, 0])))
        parent_radii_mean_range.append(float(np.ptp(np.mean(radii, axis=1))))
        segment_lengths.extend(_segment_lengths_from_series(features[:, 0]).tolist())

    if not segment_lengths:
        segment_lengths = [120, 220, 360]
    if not ip_up_rates:
        ip_up_rates = [150000.0]
    if not ip_down_rates:
        ip_down_rates = [150000.0]
    if not feature_rate_abs:
        feature_rate = np.ones((1, real.pca_components.shape[0]), dtype=np.float64)
    else:
        feature_rate = np.concatenate(feature_rate_abs, axis=0)
    return StyleStats(
        segment_lengths=np.asarray(segment_lengths, dtype=np.int64),
        ip_up_rate=np.asarray(ip_up_rates, dtype=np.float64),
        ip_down_rate=np.asarray(ip_down_rates, dtype=np.float64),
        feature_rate_abs=feature_rate.astype(np.float64),
        real_parent_ip_range=np.asarray(parent_ip_range or [0.0], dtype=np.float64),
        real_parent_radii_mean_range=np.asarray(parent_radii_mean_range or [0.0], dtype=np.float64),
    )


def _stitch_real_series(*, target_path: Path, real: RealSpace, split: str) -> list[dict[str, np.ndarray]]:
    with np.load(target_path, allow_pickle=False) as target:
        split_arr = np.asarray(target["split"]).astype(str)
        shot_id = np.asarray(target["shot_id"], dtype=np.int64)
        source_index = np.asarray(target["source_index"], dtype=np.int64)
        ip = np.asarray(target["ip_target"], dtype=np.float64)
        radii = np.asarray(target["boundary_radii"], dtype=np.float64)
    out: list[dict[str, np.ndarray]] = []
    for shot in sorted(np.unique(shot_id[split_arr == str(split)]).tolist()):
        rows = np.where((split_arr == str(split)) & (shot_id == int(shot)))[0]
        if rows.size == 0:
            continue
        point_ip: dict[int, float] = {}
        point_radii: dict[int, np.ndarray] = {}
        for row in rows[np.argsort(source_index[rows])]:
            start = int(source_index[row])
            for offset in range(ip.shape[1]):
                key = start + offset
                if key not in point_ip:
                    point_ip[key] = float(ip[row, offset])
                    point_radii[key] = radii[row, offset].astype(np.float64, copy=True)
        keys = np.asarray(sorted(point_ip), dtype=np.int64)
        if keys.size < WINDOW_STEPS + 1:
            continue
        ip_series = np.asarray([point_ip[int(k)] for k in keys], dtype=np.float64)
        radii_series = np.stack([point_radii[int(k)] for k in keys], axis=0)
        coeffs = (radii_series - real.radii_mean.reshape(1, -1)) @ real.pca_components.T
        features = np.concatenate([ip_series.reshape(-1, 1), coeffs], axis=1)
        out.append({"shot_id": int(shot), "features": features, "radii": radii_series})
    return out


def _segment_lengths_from_series(values: np.ndarray) -> np.ndarray:
    step = np.diff(np.asarray(values, dtype=np.float64))
    if step.size == 0:
        return np.asarray([100], dtype=np.int64)
    thresh = max(50.0, 0.08 * float(np.percentile(np.abs(step), 90.0)))
    sign = np.zeros_like(step, dtype=np.int64)
    sign[step > thresh] = 1
    sign[step < -thresh] = -1
    runs: list[int] = []
    start = 0
    for i in range(1, sign.shape[0]):
        if sign[i] != sign[start]:
            runs.append(i - start)
            start = i
    runs.append(sign.shape[0] - start)
    return np.asarray([max(1, int(v)) for v in runs], dtype=np.int64)


def _make_parent_set(
    *,
    space: BuildSpace,
    count: int,
    parent_id_base: int,
    split: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    parents: list[dict[str, Any]] = []
    for local_idx in range(int(count)):
        parent = _generate_parent(
            space=space,
            parent_id=int(parent_id_base + local_idx),
            split=str(split),
            min_steps=int(args.min_steps),
            max_steps=int(args.max_steps),
            knn_k=int(args.knn_k),
            rng=rng,
        )
        parents.append(parent)
        print(
            "[realstyle-simple] "
            f"{split} parent={local_idx + 1}/{count} id={parent['parent_id']} "
            f"family={parent['ip_family']} steps={parent['steps']} "
            f"feature_scale={parent['feature_motion_scale']:.3f} current_scale={parent['current_motion_scale']:.3f}",
            flush=True,
        )
    return parents


def _generate_parent(
    *,
    space: BuildSpace,
    parent_id: int,
    split: str,
    min_steps: int,
    max_steps: int,
    knn_k: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    steps = int(rng.integers(int(min_steps), int(max_steps) + 1))
    reset_idx = int(rng.integers(0, space.reset_pool.features.shape[0]))
    start_feature = space.reset_pool.features[reset_idx].astype(np.float64, copy=True)
    start_current = space.reset_pool.currents[reset_idx].astype(np.float64, copy=True)
    ip_family = str(rng.choice(np.asarray(ALLOWED_FAMILIES)))
    ip = _sample_ip_profile(space=space, start_ip=float(start_feature[0]), steps=steps, family=ip_family, rng=rng)
    features = _sample_boundary_features(space=space, start_feature=start_feature, ip=ip, rng=rng)
    features[:, 0] = ip
    features, feature_scale = _project_features(features=features, start_feature=start_feature, space=space)
    features, radii, radii_scale = _project_radii(features=features, start_feature=start_feature, space=space)
    feature_scale = min(feature_scale, radii_scale)
    currents = _infer_currents(real=space.real, features=features, start_current=start_current, knn_k=knn_k)
    currents, current_scale = _scale_current_motion(currents=currents, start_current=start_current, real=space.real)
    jdot = np.diff(currents, axis=0) / DT
    action = jdot / space.real.derivative_limits.reshape(1, -1)
    if np.any(np.abs(action) > 1.0 + 1.0e-6):
        raise RuntimeError("deterministic current scaling failed to satisfy derivative limits")
    if np.any(np.abs(currents) > space.real.current_limits.reshape(1, -1) + 1.0e-3):
        raise RuntimeError("deterministic current scaling failed to satisfy current limits")
    return {
        "parent_id": int(parent_id),
        "split": str(split),
        "steps": int(steps),
        "ip_family": ip_family,
        "boundary_families": [],
        "features": features.astype(np.float64),
        "ip": features[:, 0].astype(np.float64),
        "radii": radii.astype(np.float64),
        "currents": currents.astype(np.float64),
        "jdot": jdot.astype(np.float64),
        "real_jdot_action": action.astype(np.float64),
        "reset_shot_id": int(space.reset_pool.shot_id[reset_idx]),
        "reset_source_index": int(space.reset_pool.source_index[reset_idx]),
        "reset_time_s": float(space.reset_pool.time_s[reset_idx]),
        "feature_motion_scale": float(feature_scale),
        "current_motion_scale": float(current_scale),
    }


def _sample_ip_profile(
    *,
    space: BuildSpace,
    start_ip: float,
    steps: int,
    family: str,
    rng: np.random.Generator,
) -> np.ndarray:
    kinds = _family_kinds(family)
    lengths = _sample_segment_lengths(space.style.segment_lengths, steps=steps, count=len(kinds), rng=rng)
    low = float(space.real.feature_low[0])
    high = float(space.real.feature_high[0])
    current = float(np.clip(start_ip, low, high))
    values = [current]
    for kind, duration in zip(kinds, lengths):
        if kind == "hold":
            end = current
        else:
            rates = space.style.ip_up_rate if kind == "up" else space.style.ip_down_rate
            rate = _sample_rate(rates, rng=rng, low_q=35.0, high_q=85.0)
            delta = float(rate * int(duration) * DT)
            end = current + delta if kind == "up" else current - delta
            end = _project_scalar_segment(current=current, proposed=end, low=low, high=high)
        segment = np.linspace(current, end, int(duration) + 1, dtype=np.float64)
        values.extend(segment[1:].tolist())
        current = float(end)
    out = np.asarray(values, dtype=np.float64)
    if out.shape[0] != int(steps) + 1:
        out = _resample_1d(out, int(steps) + 1)
    out[0] = start_ip
    return np.clip(out, low, high)


def _sample_boundary_features(
    *,
    space: BuildSpace,
    start_feature: np.ndarray,
    ip: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    steps = ip.shape[0] - 1
    dims = start_feature.shape[0]
    features = np.repeat(start_feature.reshape(1, -1), steps + 1, axis=0)
    features[:, 0] = ip
    boundary_dims = dims - 1
    if boundary_dims <= 0:
        return features
    active_count = int(rng.integers(1, min(3, boundary_dims) + 1))
    active_dims = rng.choice(np.arange(1, dims), size=active_count, replace=False)
    for dim in active_dims:
        family = str(rng.choice(np.asarray(ALLOWED_FAMILIES)))
        direction = 1.0 if rng.random() < 0.5 else -1.0
        profile = _sample_feature_profile(
            space=space,
            start=float(start_feature[dim]),
            dim=int(dim),
            steps=steps,
            family=family,
            direction=direction,
            rng=rng,
        )
        features[:, dim] = profile
    return features


def _sample_feature_profile(
    *,
    space: BuildSpace,
    start: float,
    dim: int,
    steps: int,
    family: str,
    direction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    kinds = _family_kinds(family)
    lengths = _sample_segment_lengths(space.style.segment_lengths, steps=steps, count=len(kinds), rng=rng)
    low = float(space.real.feature_low[dim])
    high = float(space.real.feature_high[dim])
    current = float(np.clip(start, low, high))
    values = [current]
    rate_samples = space.style.feature_rate_abs[:, dim - 1]
    rate = _sample_rate(rate_samples, rng=rng, low_q=35.0, high_q=90.0)
    for kind, duration in zip(kinds, lengths):
        if kind == "hold":
            end = current
        else:
            sign = direction if kind == "up" else -direction
            delta = float(sign * rate * int(duration) * DT)
            end = _project_scalar_segment(current=current, proposed=current + delta, low=low, high=high)
        segment = np.linspace(current, end, int(duration) + 1, dtype=np.float64)
        values.extend(segment[1:].tolist())
        current = float(end)
    out = np.asarray(values, dtype=np.float64)
    if out.shape[0] != int(steps) + 1:
        out = _resample_1d(out, int(steps) + 1)
    out[0] = start
    return np.clip(out, low, high)


def _family_kinds(family: str) -> tuple[str, ...]:
    if family == "ramp_up_hold_ramp_down":
        return ("up", "hold", "down")
    if family == "ramp_up_hold":
        return ("up", "hold")
    if family == "hold_ramp_down":
        return ("hold", "down")
    if family == "ramp_up":
        return ("up",)
    raise ValueError(f"unsupported family {family!r}")


def _sample_segment_lengths(samples: np.ndarray, *, steps: int, count: int, rng: np.random.Generator) -> np.ndarray:
    count = int(count)
    if count <= 1:
        return np.asarray([int(steps)], dtype=np.int64)
    raw = rng.choice(np.asarray(samples, dtype=np.float64), size=count, replace=True)
    raw = np.maximum(raw, 1.0)
    min_seg = min(max(50, int(0.08 * steps)), max(1, int(steps // count)))
    remaining = int(steps) - count * min_seg
    if remaining <= 0:
        lengths = np.full(count, int(steps) // count, dtype=np.int64)
        lengths[-1] += int(steps) - int(np.sum(lengths))
        return lengths
    extra = np.floor(remaining * raw / float(np.sum(raw))).astype(np.int64)
    lengths = extra + min_seg
    lengths[-1] += int(steps) - int(np.sum(lengths))
    return np.maximum(lengths, 1)


def _sample_rate(samples: np.ndarray, *, rng: np.random.Generator, low_q: float, high_q: float) -> float:
    clean = np.asarray(samples, dtype=np.float64)
    clean = clean[np.isfinite(clean) & (clean > 0.0)]
    if clean.size == 0:
        return 1.0
    lo = float(np.percentile(clean, low_q))
    hi = float(np.percentile(clean, high_q))
    if hi < lo:
        lo, hi = hi, lo
    if hi <= 0.0:
        return max(1.0, lo)
    return float(rng.uniform(max(0.0, lo), hi))


def _project_scalar_segment(*, current: float, proposed: float, low: float, high: float) -> float:
    if low <= proposed <= high:
        return float(proposed)
    if proposed > high:
        return float(high)
    return float(low)


def _project_features(*, features: np.ndarray, start_feature: np.ndarray, space: BuildSpace) -> tuple[np.ndarray, float]:
    raw = np.asarray(features, dtype=np.float64)
    delta = raw - start_feature.reshape(1, -1)
    scale = np.ones(raw.shape[1], dtype=np.float64)
    for dim in range(raw.shape[1]):
        lo = float(space.real.feature_low[dim])
        hi = float(space.real.feature_high[dim])
        pos = delta[:, dim] > 0.0
        neg = delta[:, dim] < 0.0
        if np.any(pos):
            scale[dim] = min(scale[dim], float(np.min((hi - start_feature[dim]) / delta[pos, dim])))
        if np.any(neg):
            scale[dim] = min(scale[dim], float(np.min((lo - start_feature[dim]) / delta[neg, dim])))
    scale = np.clip(scale, 0.0, 1.0)
    out = start_feature.reshape(1, -1) + delta * scale.reshape(1, -1)
    out[0] = start_feature
    out = np.minimum(np.maximum(out, space.real.feature_low.reshape(1, -1)), space.real.feature_high.reshape(1, -1))
    out[0] = start_feature
    return out, float(np.min(scale))


def _project_radii(
    *,
    features: np.ndarray,
    start_feature: np.ndarray,
    space: BuildSpace,
) -> tuple[np.ndarray, np.ndarray, float]:
    def safe_at(scale: float) -> tuple[np.ndarray, np.ndarray, bool]:
        candidate = features.copy()
        candidate[:, 1:] = start_feature.reshape(1, -1)[:, 1:] + scale * (
            candidate[:, 1:] - start_feature.reshape(1, -1)[:, 1:]
        )
        radii = _features_to_radii(space.real, candidate)
        ok = bool(
            np.all(np.isfinite(radii))
            and np.all(radii >= space.real.radii_low.reshape(1, -1))
            and np.all(radii <= space.real.radii_high.reshape(1, -1))
        )
        return candidate, radii, ok

    candidate, radii, ok = safe_at(1.0)
    if ok:
        return candidate, radii, 1.0
    best_features, best_radii, _ = safe_at(0.0)
    lo = 0.0
    hi = 1.0
    best_scale = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        features_mid, radii_mid, ok_mid = safe_at(mid)
        if ok_mid:
            lo = mid
            best_scale = mid
            best_features = features_mid
            best_radii = radii_mid
        else:
            hi = mid
    return best_features, best_radii, float(best_scale)


def _infer_currents(*, real: RealSpace, features: np.ndarray, start_current: np.ndarray, knn_k: int) -> np.ndarray:
    query = (features - real.feature_center.reshape(1, -1)) / real.feature_scale.reshape(1, -1)
    values = _knn_weighted_average(query, real.features_norm, real.currents, k=int(knn_k))
    values = _lowpass_2d(values, width=_adaptive_odd_width(values.shape[0], target=121))
    shifted = start_current.reshape(1, -1) + (values - values[0].reshape(1, -1))
    shifted[0] = start_current
    return shifted


def _knn_weighted_average(query: np.ndarray, cloud: np.ndarray, values: np.ndarray, *, k: int) -> np.ndarray:
    k = max(1, min(int(k), cloud.shape[0]))
    out = np.empty((query.shape[0], values.shape[1]), dtype=np.float64)
    chunk = 128
    for start in range(0, query.shape[0], chunk):
        stop = min(query.shape[0], start + chunk)
        diff = query[start:stop, None, :] - cloud[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        local_dist = np.take_along_axis(dist2, idx, axis=1)
        weights = 1.0 / (local_dist + 1.0e-6)
        weights /= np.sum(weights, axis=1, keepdims=True)
        out[start:stop] = np.sum(values[idx] * weights[:, :, None], axis=1)
    return out


def _scale_current_motion(
    *,
    currents: np.ndarray,
    start_current: np.ndarray,
    real: RealSpace,
) -> tuple[np.ndarray, float]:
    delta = currents - start_current.reshape(1, -1)

    def ok(scale: float) -> bool:
        candidate = start_current.reshape(1, -1) + scale * delta
        if np.any(np.abs(candidate) > real.current_limits.reshape(1, -1) + 1.0e-6):
            return False
        jdot = np.diff(candidate, axis=0) / DT
        if np.any(np.abs(jdot) > real.derivative_limits.reshape(1, -1) + 1.0e-6):
            return False
        return True

    if ok(1.0):
        out = currents.copy()
        out[0] = start_current
        return out, 1.0
    lo = 0.0
    hi = 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    out = start_current.reshape(1, -1) + lo * delta
    out[0] = start_current
    return out, float(lo)


def _windows_from_parent(*, parent: dict[str, Any], current_limits: np.ndarray) -> list[dict[str, Any]]:
    steps = int(parent["steps"])
    ip = np.asarray(parent["ip"], dtype=np.float64)
    radii = np.asarray(parent["radii"], dtype=np.float64)
    currents = np.asarray(parent["currents"], dtype=np.float64)
    action = np.asarray(parent["real_jdot_action"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for start in range(0, steps - WINDOW_STEPS + 1):
        end = start + WINDOW_STEPS
        current0 = currents[start]
        ip_target = ip[start : end + 1]
        rows.append(
            {
                "shot_id": int(parent["parent_id"]),
                "split": str(parent["split"]),
                "source_index": int(start),
                "time_s": float(start * DT),
                "ip0": float(ip_target[0]),
                "pfc0": current0[:6].astype(np.float32),
                "sol0": current0[6:].astype(np.float32),
                "ip_target": ip_target.astype(np.float32),
                "boundary_radii": radii[start : end + 1].astype(np.float32),
                "real_jdot_action": action[start:end].astype(np.float32),
                "difficulty_bin": _difficulty_bin(float(ip_target[-1] - ip_target[0])),
                "oracle_ip_mean_error_a": 0.0,
                "oracle_ip_max_error_a": 0.0,
                "parent_reset_shot_id": int(parent["reset_shot_id"]),
                "parent_reset_source_index": int(parent["reset_source_index"]),
                "parent_reset_time_s": float(parent["reset_time_s"]),
            }
        )
    return rows


def _write_parent_outputs(*, out_dir: Path, parents: list[dict[str, Any]], preview_count: int) -> None:
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    _write_preview_plots(out_dir=preview_dir, parents=parents[: max(0, int(preview_count))])
    _write_parent_npz(out_dir / "t15_realstyle_simple_long60_parents.npz", parents)


def _write_parent_npz(path: Path, parents: list[dict[str, Any]]) -> None:
    if not parents:
        return
    max_points = max(int(p["steps"]) + 1 for p in parents)
    max_steps = max(int(p["steps"]) for p in parents)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(["t15_realstyle_simple_long60_parents_v1"]),
        parent_id=np.asarray([int(p["parent_id"]) for p in parents], dtype=np.int64),
        split=np.asarray([str(p["split"]) for p in parents]),
        steps=np.asarray([int(p["steps"]) for p in parents], dtype=np.int64),
        ip_family=np.asarray([str(p["ip_family"]) for p in parents]),
        reset_shot_id=np.asarray([int(p["reset_shot_id"]) for p in parents], dtype=np.int64),
        reset_source_index=np.asarray([int(p["reset_source_index"]) for p in parents], dtype=np.int64),
        reset_time_s=np.asarray([float(p["reset_time_s"]) for p in parents], dtype=np.float64),
        feature_motion_scale=np.asarray([float(p["feature_motion_scale"]) for p in parents], dtype=np.float64),
        current_motion_scale=np.asarray([float(p["current_motion_scale"]) for p in parents], dtype=np.float64),
        ip=np.stack([_pad_2d(np.asarray(p["ip"])[:, None], max_len=max_points)[:, 0] for p in parents], axis=0),
        radii=np.stack([_pad_2d(np.asarray(p["radii"]), max_len=max_points) for p in parents], axis=0),
        currents=np.stack([_pad_2d(np.asarray(p["currents"]), max_len=max_points) for p in parents], axis=0),
        jdot=np.stack([_pad_2d(np.asarray(p["jdot"]), max_len=max_steps) for p in parents], axis=0),
    )


def _summary(
    *,
    args: argparse.Namespace,
    target_path: Path,
    initial_path: Path,
    out_dir: Path,
    rows: list[dict[str, Any]],
    parents: list[dict[str, Any]],
    train_space: BuildSpace,
    holdout_space: BuildSpace,
    initial_out: Path,
    preview_only: bool,
) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for row in rows:
        split_counts[str(row["split"])] = split_counts.get(str(row["split"]), 0) + 1
        difficulty_counts[str(row["difficulty_bin"])] = difficulty_counts.get(str(row["difficulty_bin"]), 0) + 1
    current_usage = [
        float(np.max(np.abs(np.asarray(p["currents"], dtype=np.float64)) / train_space.real.current_limits.reshape(1, -1)))
        for p in parents
    ]
    action_usage = [float(np.max(np.abs(np.asarray(p["real_jdot_action"], dtype=np.float64)))) for p in parents]
    summary = {
        "schema": "t15_realstyle_simple_long60_summary_v1",
        "preview_only": bool(preview_only),
        "source_oracle_target": str(target_path),
        "source_initial_library": str(initial_path),
        "target_dir": str(out_dir),
        "oracle_path": str(out_dir / "t15_replay_window_oracle_targets.npz"),
        "initial_library": str(initial_out),
        "allowed_families": list(ALLOWED_FAMILIES),
        "parent_steps": {"min": int(args.min_steps), "max": int(args.max_steps)},
        "reset_max_source_index": int(args.reset_max_source_index),
        "train_parents": int(args.train_parents) if not preview_only else 0,
        "holdout_parents": int(args.holdout_parents) if not preview_only else 0,
        "preview_parents": len(parents) if preview_only else min(int(args.preview_examples), len(parents)),
        "accepted_windows": int(len(rows)),
        "split_counts": dict(sorted(split_counts.items())),
        "difficulty_bins": dict(sorted(difficulty_counts.items())),
        "current_jdot_limits": {
            "current_limits": train_space.real.current_limits.tolist(),
            "derivative_limits": train_space.real.derivative_limits.tolist(),
            "max_current_usage": float(np.max(current_usage)) if current_usage else math.nan,
            "mean_current_usage": float(np.mean(current_usage)) if current_usage else math.nan,
            "max_action_usage": float(np.max(action_usage)) if action_usage else math.nan,
            "mean_action_usage": float(np.mean(action_usage)) if action_usage else math.nan,
        },
        "safe_space": {
            "train_reset_rows_first_500": int(train_space.reset_pool.features.shape[0]),
            "holdout_reset_rows_first_500": int(holdout_space.reset_pool.features.shape[0]),
            "feature_low": train_space.real.feature_low.tolist(),
            "feature_high": train_space.real.feature_high.tolist(),
            "radii_low_min": float(np.min(train_space.real.radii_low)),
            "radii_high_max": float(np.max(train_space.real.radii_high)),
        },
        "parent_summaries": [_parent_summary(p) for p in parents],
    }
    return summary


def _write_summary_files(*, out_dir: Path, summary: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(out_dir / "report.md", summary)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# T15 Real-Style Simple Long60",
        "",
        f"- Preview only: {summary['preview_only']}",
        f"- Train parents: {summary['train_parents']}",
        f"- Holdout parents: {summary['holdout_parents']}",
        f"- Accepted windows: {summary['accepted_windows']}",
        f"- Parent steps: {summary['parent_steps']['min']}..{summary['parent_steps']['max']}",
        f"- Reset max source index: {summary['reset_max_source_index']}",
        f"- Max current usage: {summary['current_jdot_limits']['max_current_usage']:.4f}",
        f"- Max normalized Jdot action: {summary['current_jdot_limits']['max_action_usage']:.4f}",
        "",
        "## Allowed Families",
        "",
    ]
    lines.extend([f"- `{name}`" for name in summary["allowed_families"]])
    lines.extend(["", "## Splits", "", "| split | windows |", "|---|---:|"])
    if summary["split_counts"]:
        for split, count in summary["split_counts"].items():
            lines.append(f"| `{split}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(["", "## Difficulty Bins", "", "| bin | windows |", "|---|---:|"])
    if summary["difficulty_bins"]:
        for key, count in summary["difficulty_bins"].items():
            lines.append(f"| `{key}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(["", "## Parent Summary", "", "| parent | split | family | steps | windows | reset | current abs max | action max |"])
    lines.append("|---:|---|---|---:|---:|---|---:|---:|")
    for parent in summary["parent_summaries"][:24]:
        lines.append(
            f"| {parent['parent_id']} | `{parent['split']}` | `{parent['ip_family']}` | "
            f"{parent['steps']} | {parent['windows']} | "
            f"{parent['reset_shot_id']}:{parent['reset_source_index']} | "
            f"{parent['current_abs_max']:.4g} | {parent['action_abs_max']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parent_summary(parent: dict[str, Any]) -> dict[str, Any]:
    ip = np.asarray(parent["ip"], dtype=np.float64)
    radii = np.asarray(parent["radii"], dtype=np.float64)
    currents = np.asarray(parent["currents"], dtype=np.float64)
    action = np.asarray(parent["real_jdot_action"], dtype=np.float64)
    return {
        "parent_id": int(parent["parent_id"]),
        "split": str(parent["split"]),
        "ip_family": str(parent["ip_family"]),
        "steps": int(parent["steps"]),
        "windows": int(parent["steps"]) - WINDOW_STEPS + 1,
        "ip_min": float(np.min(ip)),
        "ip_max": float(np.max(ip)),
        "ip_delta": float(ip[-1] - ip[0]),
        "boundary_mean_range_m": float(np.ptp(np.mean(radii, axis=1))),
        "boundary_angle_range_m": float(np.mean(np.ptp(radii, axis=0))),
        "current_abs_max": float(np.max(np.abs(currents))),
        "action_abs_max": float(np.max(np.abs(action))),
        "feature_motion_scale": float(parent["feature_motion_scale"]),
        "current_motion_scale": float(parent["current_motion_scale"]),
        "reset_shot_id": int(parent["reset_shot_id"]),
        "reset_source_index": int(parent["reset_source_index"]),
        "reset_time_s": float(parent["reset_time_s"]),
    }


def _write_preview_plots(*, out_dir: Path, parents: list[dict[str, Any]]) -> None:
    if not parents:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["PFC0", "PFC1", "PFC2", "PFC3", "PFC4", "PFC5", "SOL0", "SOL1", "SOL2"]
    angle_idx = np.linspace(0, 31, 8, dtype=int)
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    for i, parent in enumerate(parents):
        steps = int(parent["steps"])
        t = np.arange(steps + 1, dtype=np.float64) * DT
        tj = np.arange(steps, dtype=np.float64) * DT
        ip = np.asarray(parent["ip"], dtype=np.float64)
        radii = np.asarray(parent["radii"], dtype=np.float64)
        currents = np.asarray(parent["currents"], dtype=np.float64)
        jdot = np.asarray(parent["jdot"], dtype=np.float64)
        fig, axes = plt.subplots(7, 1, figsize=(12, 18), constrained_layout=True)
        axes[0].plot(t, ip, linewidth=1.8)
        axes[0].set_ylabel("Ip [A]")
        axes[0].grid(True, alpha=0.25)
        axes[1].plot(t, np.mean(radii, axis=1), label="mean radius", linewidth=1.6)
        axes[1].plot(t, np.min(radii, axis=1), label="min radius", linewidth=1.0)
        axes[1].plot(t, np.max(radii, axis=1), label="max radius", linewidth=1.0)
        axes[1].set_ylabel("radii [m]")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.25)
        for idx in angle_idx:
            axes[2].plot(t, radii[:, idx], linewidth=1.1, label=f"a{int(idx)}")
        axes[2].set_ylabel("selected radii [m]")
        axes[2].legend(ncol=4, fontsize=8, loc="best")
        axes[2].grid(True, alpha=0.25)
        for step, name in ((0, "start"), (steps // 2, "middle"), (steps, "end")):
            axes[3].plot(theta, radii[step], linewidth=1.5, label=name)
        axes[3].set_ylabel("boundary radius [m]")
        axes[3].legend(loc="best")
        axes[3].grid(True, alpha=0.25)
        for c in range(6):
            axes[4].plot(t, currents[:, c], linewidth=1.0, label=labels[c])
        axes[4].set_ylabel("PFC current [A]")
        axes[4].legend(ncol=3, fontsize=8, loc="best")
        axes[4].grid(True, alpha=0.25)
        for c in range(6, 9):
            axes[5].plot(t, currents[:, c], linewidth=1.0, label=labels[c])
        axes[5].set_ylabel("SOL current [A]")
        axes[5].legend(ncol=3, fontsize=8, loc="best")
        axes[5].grid(True, alpha=0.25)
        for c in range(jdot.shape[1]):
            axes[6].plot(tj, jdot[:, c], linewidth=0.85, label=labels[c])
        axes[6].set_ylabel("approx Jdot [A/s]")
        axes[6].set_xlabel("time [s]")
        axes[6].legend(ncol=3, fontsize=8, loc="best")
        axes[6].grid(True, alpha=0.25)
        fig.suptitle(
            f"t15_realstyle_simple_long60 preview {i:02d} ({steps} steps, {parent['ip_family']})",
            fontsize=14,
        )
        fig.savefig(out_dir / f"t15_realstyle_simple_long60_preview_{i:02d}.png", dpi=140)
        plt.close(fig)


def _resample_1d(values: np.ndarray, size: int) -> np.ndarray:
    old_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
    new_x = np.linspace(0.0, 1.0, int(size), dtype=np.float64)
    return np.interp(new_x, old_x, values)


def _lowpass_2d(values: np.ndarray, *, width: int) -> np.ndarray:
    width = int(width)
    if width < 3 or values.shape[0] < width:
        return values.astype(np.float64, copy=True)
    if width % 2 == 0:
        width += 1
    kernel = np.hanning(width).astype(np.float64)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        return values.astype(np.float64, copy=True)
    kernel /= kernel_sum
    pad = width // 2
    out = np.empty_like(values, dtype=np.float64)
    for dim in range(values.shape[1]):
        out[:, dim] = np.convolve(np.pad(values[:, dim], (pad, pad), mode="edge"), kernel, mode="valid")[: values.shape[0]]
    return out


def _adaptive_odd_width(length: int, *, target: int) -> int:
    length = int(length)
    target = int(target)
    if length < 3:
        return 1
    width = min(target, length if length % 2 == 1 else length - 1)
    return max(3, int(width))


def _pad_2d(values: np.ndarray, *, max_len: int) -> np.ndarray:
    arr = np.asarray(values)
    out = np.full((int(max_len), arr.shape[1]), np.nan, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


if __name__ == "__main__":
    raise SystemExit(main())
