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
from build_t15_synthetic_long_preview import (
    RealSpace,
    _features_to_radii,
    _load_real_space,
    _repo_path,
    _require_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PARENT_ID_BASE = 940000
HOLDOUT_PARENT_ID_BASE = 950000


@dataclass(frozen=True, slots=True)
class EmpiricalSpace:
    real: RealSpace
    split: str
    reset_features: np.ndarray
    reset_currents: np.ndarray
    reset_shot_id: np.ndarray
    reset_source_index: np.ndarray
    reset_time_s: np.ndarray
    feature_low: np.ndarray
    feature_high: np.ndarray
    radii_low: np.ndarray
    radii_high: np.ndarray
    velocity_samples: np.ndarray
    velocity_center: np.ndarray
    velocity_scale: np.ndarray
    knn_features: np.ndarray
    knn_velocities: np.ndarray
    knn_currents: np.ndarray
    knn_keys: np.ndarray
    key_center: np.ndarray
    key_scale: np.ndarray
    real_ip_rate_abs: np.ndarray
    real_radii_step_abs: np.ndarray
    real_radii_range: np.ndarray


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build t15_synth_empirical_long60: new long synthetic T15 trajectories "
            "from real reset states and empirical derivative statistics, cut into "
            "overlapping 100-step replay-window oracle targets."
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
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/t15_synth_empirical_long60"))
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-examples", type=int, default=6)
    parser.add_argument("--train-parents", type=int, default=60)
    parser.add_argument("--holdout-parents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--wiggle-room", type=float, default=1.15)
    parser.add_argument("--ip-high-a", type=float, default=500000.0)
    parser.add_argument("--radii-margin-m", type=float, default=0.025)
    parser.add_argument("--current-envelope-margin", type=float, default=0.08)
    parser.add_argument("--max-cloud-rows", type=int, default=12000)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--dt", type=float, default=DT)
    args = parser.parse_args(argv)

    if int(args.min_steps) < WINDOW_STEPS:
        raise ValueError(f"--min-steps must be >= {WINDOW_STEPS}")
    if int(args.max_steps) < int(args.min_steps):
        raise ValueError("--max-steps must be >= --min-steps")
    if float(args.dt) <= 0.0:
        raise ValueError("--dt must be positive")

    target_path = _repo_path(args.oracle_target)
    initial_path = _repo_path(args.initial_library)
    out_dir = _repo_path(args.out_dir)
    _require_inputs(target_path=target_path, initial_path=initial_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    print("[synth-empirical] loading train empirical space", flush=True)
    train_space = _load_empirical_space(
        target_path=target_path,
        initial_path=initial_path,
        split="train",
        args=args,
        rng=rng,
    )
    print("[synth-empirical] loading holdout empirical space", flush=True)
    holdout_space = _load_empirical_space(
        target_path=target_path,
        initial_path=initial_path,
        split="holdout",
        args=args,
        rng=rng,
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
        _write_parent_outputs(out_dir=out_dir, parents=parents, train_space=train_space, preview_count=len(parents))
        summary = _summary(
            args=args,
            target_path=target_path,
            initial_path=initial_path,
            out_dir=out_dir,
            rows=[],
            parents=parents,
            train_space=train_space,
            holdout_space=holdout_space,
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
        raise RuntimeError("t15_synth_empirical_long60 produced zero windows")

    oracle_path = out_dir / "t15_replay_window_oracle_targets.npz"
    initial_out = out_dir / "t15_synth_empirical_long60_initial_states.npz"
    _write_oracle_npz(
        oracle_path,
        rows,
        current_limits=train_space.real.current_limits,
        derivative_limits=train_space.real.derivative_limits,
    )
    _write_initial_library(initial_out, rows)
    _write_parent_outputs(out_dir=out_dir, parents=parents, train_space=train_space, preview_count=int(args.preview_examples))

    summary = _summary(
        args=args,
        target_path=target_path,
        initial_path=initial_path,
        out_dir=out_dir,
        rows=rows,
        parents=parents,
        train_space=train_space,
        holdout_space=holdout_space,
        preview_only=False,
    )
    _write_summary_files(out_dir=out_dir, summary=summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def _load_empirical_space(
    *,
    target_path: Path,
    initial_path: Path,
    split: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> EmpiricalSpace:
    real = _load_real_space(
        target_path=target_path,
        initial_path=initial_path,
        dt=float(args.dt),
        pca_components=int(args.pca_components),
        wiggle_room=float(args.wiggle_room),
        radii_margin_m=float(args.radii_margin_m),
        current_envelope_margin=float(args.current_envelope_margin),
        max_cloud_rows=int(args.max_cloud_rows),
        safe_split=str(split),
        rng=rng,
    )
    arrays = _load_split_arrays(target_path=target_path, initial_path=initial_path, split=split, real=real, dt=float(args.dt))
    features_seq = arrays["features"]
    currents_seq = arrays["currents"]
    radii_seq = arrays["radii"]
    velocities_seq = np.diff(features_seq, axis=1) / float(args.dt)
    velocities_seq = _smooth_velocity_sequences(velocities_seq)

    feature_low = _expand_bounds(real.feature_low, real.feature_high, float(args.wiggle_room))[0]
    feature_high = _expand_bounds(real.feature_low, real.feature_high, float(args.wiggle_room))[1]
    feature_high[0] = float(getattr(args, "ip_high_a", 500000.0))
    radii_low = real.radii_low
    radii_high = real.radii_high

    reset_features = features_seq[:, 0, :]
    reset_currents = currents_seq[:, 0, :]
    flat_features = features_seq[:, :-1, :].reshape(-1, features_seq.shape[-1])
    flat_velocities = velocities_seq.reshape(-1, velocities_seq.shape[-1])
    flat_currents = currents_seq[:, :-1, :].reshape(-1, currents_seq.shape[-1])

    cloud_idx = _sample_indices(flat_features.shape[0], max_rows=int(args.max_cloud_rows), rng=rng)
    knn_features = flat_features[cloud_idx]
    knn_velocities = flat_velocities[cloud_idx]
    knn_currents = flat_currents[cloud_idx]
    velocity_center = np.median(flat_velocities, axis=0)
    velocity_scale = np.percentile(np.abs(flat_velocities - velocity_center.reshape(1, -1)), 90, axis=0)
    velocity_scale = np.where(np.isfinite(velocity_scale) & (velocity_scale > 1.0e-12), velocity_scale, 1.0)

    key = _make_knn_key(
        features=knn_features,
        velocities=knn_velocities,
        feature_center=real.feature_center,
        feature_scale=real.feature_scale,
        velocity_center=velocity_center,
        velocity_scale=velocity_scale,
    )
    key_center = np.median(key, axis=0)
    key_scale = np.percentile(np.abs(key - key_center.reshape(1, -1)), 90, axis=0)
    key_scale = np.where(np.isfinite(key_scale) & (key_scale > 1.0e-12), key_scale, 1.0)
    knn_keys = (key - key_center.reshape(1, -1)) / key_scale.reshape(1, -1)

    return EmpiricalSpace(
        real=real,
        split=str(split),
        reset_features=reset_features,
        reset_currents=reset_currents,
        reset_shot_id=arrays["shot_id"],
        reset_source_index=arrays["source_index"],
        reset_time_s=arrays["time_s"],
        feature_low=feature_low,
        feature_high=feature_high,
        radii_low=radii_low,
        radii_high=radii_high,
        velocity_samples=flat_velocities,
        velocity_center=velocity_center,
        velocity_scale=velocity_scale,
        knn_features=knn_features,
        knn_velocities=knn_velocities,
        knn_currents=knn_currents,
        knn_keys=knn_keys,
        key_center=key_center,
        key_scale=key_scale,
        real_ip_rate_abs=np.abs(velocities_seq[..., 0]).reshape(-1),
        real_radii_step_abs=np.mean(np.abs(np.diff(radii_seq, axis=1)), axis=2).reshape(-1),
        real_radii_range=np.ptp(np.mean(radii_seq, axis=2), axis=1),
    )


def _load_split_arrays(
    *,
    target_path: Path,
    initial_path: Path,
    split: str,
    real: RealSpace,
    dt: float,
) -> dict[str, np.ndarray]:
    with np.load(target_path, allow_pickle=False) as target, np.load(initial_path, allow_pickle=False) as init:
        split_arr = np.asarray(target["split"]).astype(str)
        mask = split_arr == str(split)
        if not np.any(mask):
            raise ValueError(f"split {split!r} matched zero rows in {target_path}")
        ip = np.asarray(target["ip_target"], dtype=np.float64)[mask]
        radii = np.asarray(target["boundary_radii"], dtype=np.float64)[mask]
        action = np.asarray(target["real_jdot_action"], dtype=np.float64)[mask]
        shot_id = np.asarray(target["shot_id"], dtype=np.int64)[mask]
        source_index = np.asarray(target["source_index"], dtype=np.int64)[mask]
        time_s = np.asarray(target["time_s"], dtype=np.float64)[mask]
        pfc0 = np.asarray(init["pfc0"], dtype=np.float64)[mask]
        sol0 = np.asarray(init["sol0"], dtype=np.float64)[mask]

    coeffs = (radii.reshape(-1, radii.shape[-1]) - real.radii_mean.reshape(1, -1)) @ real.pca_components.T
    features = np.concatenate([ip.reshape(-1, 1), coeffs], axis=1).reshape(ip.shape[0], ip.shape[1], -1)
    initial = np.concatenate([pfc0, sol0], axis=1)
    jdot = action * real.derivative_limits.reshape(1, 1, -1)
    currents = np.empty((initial.shape[0], action.shape[1] + 1, initial.shape[1]), dtype=np.float64)
    currents[:, 0, :] = initial
    currents[:, 1:, :] = initial[:, None, :] + np.cumsum(jdot * float(dt), axis=1)
    return {
        "ip": ip,
        "radii": radii,
        "features": features,
        "currents": currents,
        "shot_id": shot_id,
        "source_index": source_index,
        "time_s": time_s,
    }


def _make_parent_set(
    *,
    space: EmpiricalSpace,
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
            dt=float(args.dt),
            knn_k=int(args.knn_k),
            rng=rng,
        )
        parents.append(parent)
        print(
            "[synth-empirical] "
            f"{split} parent={local_idx + 1}/{count} id={parent['parent_id']} "
            f"steps={parent['steps']} feature_scale={parent['feature_motion_scale']:.3f} "
            f"current_scale={parent['current_motion_scale']:.3f}",
            flush=True,
        )
    return parents


def _generate_parent(
    *,
    space: EmpiricalSpace,
    parent_id: int,
    split: str,
    min_steps: int,
    max_steps: int,
    dt: float,
    knn_k: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    steps = int(rng.integers(int(min_steps), int(max_steps) + 1))
    reset_idx = int(rng.integers(0, space.reset_features.shape[0]))
    start_feature = space.reset_features[reset_idx].astype(np.float64, copy=True)
    start_current = space.reset_currents[reset_idx].astype(np.float64, copy=True)

    velocities = _sample_empirical_velocity_process(space=space, steps=steps, rng=rng)
    features = np.empty((steps + 1, start_feature.shape[0]), dtype=np.float64)
    features[0] = start_feature
    features[1:] = start_feature.reshape(1, -1) + np.cumsum(velocities * float(dt), axis=0)
    features, bounds_scale = _project_features(features=features, start_feature=start_feature, space=space)
    features, radii, radii_scale = _project_radii(features=features, start_feature=start_feature, space=space)
    feature_scale = min(bounds_scale, radii_scale)
    velocities = np.diff(features, axis=0) / float(dt)

    currents = _infer_currents(space=space, features=features, velocities=velocities, start_current=start_current, knn_k=knn_k)
    currents, current_scale = _scale_current_motion(currents=currents, start_current=start_current, space=space, dt=dt)
    jdot = np.diff(currents, axis=0) / float(dt)
    action = jdot / space.real.derivative_limits.reshape(1, -1)
    if np.any(np.abs(action) > 1.0 + 1.0e-6):
        raise RuntimeError("deterministic current scaling failed to satisfy derivative limits")
    if np.any(np.abs(currents) > space.real.current_limits.reshape(1, -1) + 1.0e-3):
        raise RuntimeError("deterministic current scaling failed to satisfy current limits")

    return {
        "parent_id": int(parent_id),
        "split": str(split),
        "steps": int(steps),
        "features": features.astype(np.float64),
        "ip": features[:, 0].astype(np.float64),
        "radii": radii.astype(np.float64),
        "currents": currents.astype(np.float64),
        "jdot": jdot.astype(np.float64),
        "real_jdot_action": action.astype(np.float64),
        "reset_shot_id": int(space.reset_shot_id[reset_idx]),
        "reset_source_index": int(space.reset_source_index[reset_idx]),
        "reset_time_s": float(space.reset_time_s[reset_idx]),
        "feature_motion_scale": float(feature_scale),
        "current_motion_scale": float(current_scale),
    }


def _sample_empirical_velocity_process(
    *,
    space: EmpiricalSpace,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dims = space.velocity_samples.shape[1]
    min_interval = 120
    max_interval = 320
    knots = [0]
    while knots[-1] < int(steps):
        knots.append(min(int(steps), knots[-1] + int(rng.integers(min_interval, max_interval + 1))))
    if knots[-1] != int(steps):
        knots.append(int(steps))
    knot_count = len(knots)
    idx = rng.integers(0, space.velocity_samples.shape[0], size=knot_count)
    knot_vel = space.velocity_samples[idx].astype(np.float64, copy=True)

    # Small correlated perturbation keeps trajectories new while preserving
    # empirical derivative scale and cross-parameter coupling.
    noise = rng.normal(0.0, 0.12, size=knot_vel.shape) * space.velocity_scale.reshape(1, -1)
    knot_vel += noise
    low = np.percentile(space.velocity_samples, 1.0, axis=0)
    high = np.percentile(space.velocity_samples, 99.0, axis=0)
    knot_vel = np.minimum(np.maximum(knot_vel, low.reshape(1, -1)), high.reshape(1, -1))

    velocities = np.empty((int(steps), dims), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(knots[:-1], knots[1:])):
        duration = max(1, int(hi - lo))
        a = knot_vel[i]
        b = knot_vel[i + 1]
        alpha = _smoothstep(np.linspace(0.0, 1.0, duration, endpoint=False, dtype=np.float64)).reshape(-1, 1)
        velocities[lo:hi] = a.reshape(1, -1) + alpha * (b - a).reshape(1, -1)
    velocities = _lowpass_2d(velocities, width=81)
    if velocities.shape[0] != int(steps):
        raise AssertionError("velocity process length mismatch")
    return velocities


def _project_features(*, features: np.ndarray, start_feature: np.ndarray, space: EmpiricalSpace) -> tuple[np.ndarray, float]:
    raw = np.asarray(features, dtype=np.float64)
    delta = raw - start_feature.reshape(1, -1)
    scale = np.ones(raw.shape[1], dtype=np.float64)
    for dim in range(raw.shape[1]):
        lo = float(space.feature_low[dim])
        hi = float(space.feature_high[dim])
        pos = delta[:, dim] > 0.0
        neg = delta[:, dim] < 0.0
        if np.any(pos):
            scale[dim] = min(scale[dim], float(np.min((hi - start_feature[dim]) / delta[pos, dim])))
        if np.any(neg):
            scale[dim] = min(scale[dim], float(np.min((lo - start_feature[dim]) / delta[neg, dim])))
    scale = np.clip(scale, 0.0, 1.0)
    out = start_feature.reshape(1, -1) + delta * scale.reshape(1, -1)
    out[0] = start_feature
    # Numerical guard only. The geometric projection above is what prevents
    # synthetic ramp-to-clamp corners.
    out = np.minimum(np.maximum(out, space.feature_low.reshape(1, -1)), space.feature_high.reshape(1, -1))
    out[0] = start_feature
    return out, float(np.min(scale))


def _project_radii(
    *,
    features: np.ndarray,
    start_feature: np.ndarray,
    space: EmpiricalSpace,
) -> tuple[np.ndarray, np.ndarray, float]:
    def safe_at(scale: float) -> tuple[np.ndarray, np.ndarray, bool]:
        candidate = features.copy()
        candidate[:, 1:] = start_feature.reshape(1, -1)[:, 1:] + scale * (
            candidate[:, 1:] - start_feature.reshape(1, -1)[:, 1:]
        )
        radii = _features_to_radii(space.real, candidate)
        ok = bool(
            np.all(np.isfinite(radii))
            and np.all(radii >= space.radii_low.reshape(1, -1))
            and np.all(radii <= space.radii_high.reshape(1, -1))
        )
        return candidate, radii, ok

    candidate, radii, ok = safe_at(1.0)
    if ok:
        return candidate, radii, 1.0
    lo = 0.0
    hi = 1.0
    best_features, best_radii, _ = safe_at(0.0)
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


def _infer_currents(
    *,
    space: EmpiricalSpace,
    features: np.ndarray,
    velocities: np.ndarray,
    start_current: np.ndarray,
    knn_k: int,
) -> np.ndarray:
    point_vel = np.empty_like(features)
    point_vel[:-1] = velocities
    point_vel[-1] = velocities[-1] if velocities.shape[0] else 0.0
    query_key = _make_knn_key(
        features=features,
        velocities=point_vel,
        feature_center=space.real.feature_center,
        feature_scale=space.real.feature_scale,
        velocity_center=space.velocity_center,
        velocity_scale=space.velocity_scale,
    )
    query_key = (query_key - space.key_center.reshape(1, -1)) / space.key_scale.reshape(1, -1)
    pred = _knn_weighted_average(query_key, space.knn_keys, space.knn_currents, k=int(knn_k))
    # KNN is a local state lookup, so nearest-neighbor identity can change from
    # one millisecond to the next. Smooth the inferred current path before
    # differencing; otherwise the diagnostic oracle action becomes needle-like
    # and downstream current scaling collapses otherwise reasonable parents.
    pred = _lowpass_2d(pred, width=_adaptive_odd_width(pred.shape[0], target=121))
    shifted = start_current.reshape(1, -1) + (pred - pred[0].reshape(1, -1))
    shifted[0] = start_current
    return shifted


def _scale_current_motion(
    *,
    currents: np.ndarray,
    start_current: np.ndarray,
    space: EmpiricalSpace,
    dt: float,
) -> tuple[np.ndarray, float]:
    delta = currents - start_current.reshape(1, -1)

    def ok(scale: float) -> bool:
        candidate = start_current.reshape(1, -1) + scale * delta
        if np.any(np.abs(candidate) > space.real.current_limits.reshape(1, -1) + 1.0e-6):
            return False
        jdot = np.diff(candidate, axis=0) / float(dt)
        if np.any(np.abs(jdot) > space.real.derivative_limits.reshape(1, -1) + 1.0e-6):
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
    if ip.shape != (steps + 1,):
        raise ValueError(f"parent {parent['parent_id']} ip shape mismatch: {ip.shape}")
    if radii.shape != (steps + 1, 32):
        raise ValueError(f"parent {parent['parent_id']} radii shape mismatch: {radii.shape}")
    if currents.shape != (steps + 1, 9):
        raise ValueError(f"parent {parent['parent_id']} current shape mismatch: {currents.shape}")
    if action.shape != (steps, 9):
        raise ValueError(f"parent {parent['parent_id']} action shape mismatch: {action.shape}")

    rows: list[dict[str, Any]] = []
    for start in range(0, ip.shape[0] - WINDOW_STEPS):
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


def _write_parent_outputs(
    *,
    out_dir: Path,
    parents: list[dict[str, Any]],
    train_space: EmpiricalSpace,
    preview_count: int,
) -> None:
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    _write_preview_plots(out_dir=preview_dir, parents=parents[: max(0, int(preview_count))], real=train_space.real, dt=DT)
    _write_parent_npz(out_dir / "t15_synth_empirical_long60_parents.npz", parents)


def _write_parent_npz(path: Path, parents: list[dict[str, Any]]) -> None:
    if not parents:
        return
    max_points = max(int(p["steps"]) + 1 for p in parents)
    max_steps = max(int(p["steps"]) for p in parents)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(["t15_synth_empirical_long60_parents_v1"]),
        parent_id=np.asarray([int(p["parent_id"]) for p in parents], dtype=np.int64),
        split=np.asarray([str(p["split"]) for p in parents]),
        steps=np.asarray([int(p["steps"]) for p in parents], dtype=np.int64),
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
    train_space: EmpiricalSpace,
    holdout_space: EmpiricalSpace,
    preview_only: bool,
) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for row in rows:
        split_counts[str(row["split"])] = split_counts.get(str(row["split"]), 0) + 1
        difficulty_counts[str(row["difficulty_bin"])] = difficulty_counts.get(str(row["difficulty_bin"]), 0) + 1
    gen_ip_rates = _cat([np.diff(np.asarray(p["ip"], dtype=np.float64)) / DT for p in parents])
    gen_radii_step = _cat([np.mean(np.abs(np.diff(np.asarray(p["radii"], dtype=np.float64), axis=0)), axis=1) for p in parents])
    gen_radii_range = np.asarray([float(np.ptp(np.mean(np.asarray(p["radii"], dtype=np.float64), axis=1))) for p in parents])
    current_usage = []
    action_usage = []
    current_scale = []
    feature_scale = []
    for p in parents:
        currents = np.asarray(p["currents"], dtype=np.float64)
        action = np.asarray(p["real_jdot_action"], dtype=np.float64)
        current_usage.append(float(np.max(np.abs(currents) / train_space.real.current_limits.reshape(1, -1))))
        action_usage.append(float(np.max(np.abs(action))))
        current_scale.append(float(p["current_motion_scale"]))
        feature_scale.append(float(p["feature_motion_scale"]))
    summary = {
        "schema": "t15_synth_empirical_long60_summary_v1",
        "preview_only": bool(preview_only),
        "source_oracle_target": str(target_path),
        "source_initial_library": str(initial_path),
        "target_dir": str(out_dir),
        "oracle_path": str(out_dir / "t15_replay_window_oracle_targets.npz"),
        "initial_library": str(out_dir / "t15_synth_empirical_long60_initial_states.npz"),
        "parent_steps": {"min": int(args.min_steps), "max": int(args.max_steps)},
        "train_parents": int(args.train_parents) if not preview_only else 0,
        "holdout_parents": int(args.holdout_parents) if not preview_only else 0,
        "preview_parents": len(parents) if preview_only else min(int(args.preview_examples), len(parents)),
        "accepted_windows": int(len(rows)),
        "split_counts": dict(sorted(split_counts.items())),
        "difficulty_bins": dict(sorted(difficulty_counts.items())),
        "scale_factors": {
            "feature_motion_min": float(np.min(feature_scale)) if feature_scale else math.nan,
            "feature_motion_mean": float(np.mean(feature_scale)) if feature_scale else math.nan,
            "current_motion_min": float(np.min(current_scale)) if current_scale else math.nan,
            "current_motion_mean": float(np.mean(current_scale)) if current_scale else math.nan,
        },
        "current_jdot_limits": {
            "current_limits": train_space.real.current_limits.tolist(),
            "derivative_limits": train_space.real.derivative_limits.tolist(),
            "max_current_usage": float(np.max(current_usage)) if current_usage else math.nan,
            "mean_current_usage": float(np.mean(current_usage)) if current_usage else math.nan,
            "max_action_usage": float(np.max(action_usage)) if action_usage else math.nan,
            "mean_action_usage": float(np.mean(action_usage)) if action_usage else math.nan,
        },
        "real_vs_generated_style": {
            "real_ip_rate_abs_p50": _finite_percentile(train_space.real_ip_rate_abs, 50),
            "real_ip_rate_abs_p90": _finite_percentile(train_space.real_ip_rate_abs, 90),
            "generated_ip_rate_abs_p50": _finite_percentile(np.abs(gen_ip_rates), 50),
            "generated_ip_rate_abs_p90": _finite_percentile(np.abs(gen_ip_rates), 90),
            "real_boundary_step_mean_abs_p50": _finite_percentile(train_space.real_radii_step_abs, 50),
            "real_boundary_step_mean_abs_p90": _finite_percentile(train_space.real_radii_step_abs, 90),
            "generated_boundary_step_mean_abs_p50": _finite_percentile(gen_radii_step, 50),
            "generated_boundary_step_mean_abs_p90": _finite_percentile(gen_radii_step, 90),
            "real_boundary_parent_range_p50": _finite_percentile(train_space.real_radii_range, 50),
            "real_boundary_parent_range_p90": _finite_percentile(train_space.real_radii_range, 90),
            "generated_boundary_parent_range_p50": _finite_percentile(gen_radii_range, 50),
            "generated_boundary_parent_range_p90": _finite_percentile(gen_radii_range, 90),
        },
        "safe_space": {
            "train_reset_rows": int(train_space.reset_features.shape[0]),
            "holdout_reset_rows": int(holdout_space.reset_features.shape[0]),
            "synthetic_ip_high_a": float(train_space.feature_high[0]),
            "feature_low": train_space.feature_low.tolist(),
            "feature_high": train_space.feature_high.tolist(),
            "radii_low_min": float(np.min(train_space.radii_low)),
            "radii_high_max": float(np.max(train_space.radii_high)),
        },
        "parent_summaries": [_parent_summary(p) for p in parents],
    }
    return summary


def _write_summary_files(*, out_dir: Path, summary: dict[str, Any]) -> None:
    (out_dir / "t15_synth_empirical_long60_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(out_dir / "t15_synth_empirical_long60_report.md", summary)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# T15 Synth Empirical Long60",
        "",
        f"- Preview only: {summary['preview_only']}",
        f"- Train parents: {summary['train_parents']}",
        f"- Holdout parents: {summary['holdout_parents']}",
        f"- Accepted windows: {summary['accepted_windows']}",
        f"- Parent steps: {summary['parent_steps']['min']}..{summary['parent_steps']['max']}",
        f"- Synthetic Ip high: {summary['safe_space']['synthetic_ip_high_a']:.0f} A",
        f"- Max current usage: {summary['current_jdot_limits']['max_current_usage']:.4f}",
        f"- Max normalized Jdot action: {summary['current_jdot_limits']['max_action_usage']:.4f}",
        f"- Mean current scale: {summary['scale_factors']['current_motion_mean']:.4f}",
        "",
        "## Splits",
        "",
        "| split | windows |",
        "|---|---:|",
    ]
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
    lines.extend(
        [
            "",
            "## Style Metrics",
            "",
            "| metric | real | generated |",
            "|---|---:|---:|",
        ]
    )
    style = summary["real_vs_generated_style"]
    pairs = [
        ("Ip |rate| p50 [A/s]", "real_ip_rate_abs_p50", "generated_ip_rate_abs_p50"),
        ("Ip |rate| p90 [A/s]", "real_ip_rate_abs_p90", "generated_ip_rate_abs_p90"),
        ("Boundary step mean p50 [m]", "real_boundary_step_mean_abs_p50", "generated_boundary_step_mean_abs_p50"),
        ("Boundary step mean p90 [m]", "real_boundary_step_mean_abs_p90", "generated_boundary_step_mean_abs_p90"),
        ("Boundary parent range p50 [m]", "real_boundary_parent_range_p50", "generated_boundary_parent_range_p50"),
        ("Boundary parent range p90 [m]", "real_boundary_parent_range_p90", "generated_boundary_parent_range_p90"),
    ]
    for label, real_key, gen_key in pairs:
        lines.append(f"| {label} | {style[real_key]:.6g} | {style[gen_key]:.6g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parent_summary(parent: dict[str, Any]) -> dict[str, Any]:
    ip = np.asarray(parent["ip"], dtype=np.float64)
    radii = np.asarray(parent["radii"], dtype=np.float64)
    currents = np.asarray(parent["currents"], dtype=np.float64)
    action = np.asarray(parent["real_jdot_action"], dtype=np.float64)
    return {
        "parent_id": int(parent["parent_id"]),
        "split": str(parent["split"]),
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


def _write_preview_plots(*, out_dir: Path, parents: list[dict[str, Any]], real: RealSpace, dt: float) -> None:
    if not parents:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["PFC0", "PFC1", "PFC2", "PFC3", "PFC4", "PFC5", "SOL0", "SOL1", "SOL2"]
    theta = np.linspace(0.0, 2.0 * np.pi, real.radii_mean.shape[0], endpoint=False)
    angle_idx = np.linspace(0, real.radii_mean.shape[0] - 1, 8, dtype=int)
    for i, parent in enumerate(parents):
        steps = int(parent["steps"])
        t = np.arange(steps + 1, dtype=np.float64) * float(dt)
        tj = np.arange(steps, dtype=np.float64) * float(dt)
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
        fig.suptitle(f"t15_synth_empirical_long60 preview {i:02d} ({steps} steps)", fontsize=14)
        fig.savefig(out_dir / f"t15_synth_empirical_long60_preview_{i:02d}.png", dpi=140)
        plt.close(fig)


def _make_knn_key(
    *,
    features: np.ndarray,
    velocities: np.ndarray,
    feature_center: np.ndarray,
    feature_scale: np.ndarray,
    velocity_center: np.ndarray,
    velocity_scale: np.ndarray,
) -> np.ndarray:
    feature_key = (features - feature_center.reshape(1, -1)) / feature_scale.reshape(1, -1)
    velocity_key = (velocities - velocity_center.reshape(1, -1)) / velocity_scale.reshape(1, -1)
    return np.concatenate([feature_key, 0.25 * velocity_key], axis=1)


def _knn_weighted_average(query: np.ndarray, cloud: np.ndarray, values: np.ndarray, *, k: int) -> np.ndarray:
    k = max(1, min(int(k), cloud.shape[0]))
    out = np.empty((query.shape[0], values.shape[1]), dtype=np.float64)
    chunk = 256
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


def _smooth_velocity_sequences(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1])
    smoothed = _lowpass_2d(flat, width=21)
    return smoothed.reshape(values.shape)


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


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def _expand_bounds(low: np.ndarray, high: np.ndarray, wiggle_room: float) -> tuple[np.ndarray, np.ndarray]:
    center = 0.5 * (low + high)
    half = 0.5 * (high - low) * float(wiggle_room)
    return center - half, center + half


def _sample_indices(n: int, *, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if int(max_rows) <= 0 or n <= int(max_rows):
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=int(max_rows), replace=False).astype(np.int64))


def _pad_2d(arr: np.ndarray, *, max_len: int) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[:, None]
    out = np.full((max_len, arr.shape[1]), np.nan, dtype=np.float64)
    out[: arr.shape[0]] = arr
    return out


def _cat(items: list[np.ndarray]) -> np.ndarray:
    good = [np.asarray(x).reshape(-1) for x in items if np.asarray(x).size]
    if not good:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(good)


def _finite_percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.percentile(arr, q))


if __name__ == "__main__":
    raise SystemExit(main())
