#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class RealSpace:
    ip: np.ndarray
    radii: np.ndarray
    features: np.ndarray
    currents: np.ndarray
    source_shot_id: np.ndarray
    source_index: np.ndarray
    source_time_s: np.ndarray
    jdot: np.ndarray
    action: np.ndarray
    current_limits: np.ndarray
    derivative_limits: np.ndarray
    radii_mean: np.ndarray
    pca_components: np.ndarray
    feature_center: np.ndarray
    feature_scale: np.ndarray
    features_norm: np.ndarray
    feature_low: np.ndarray
    feature_high: np.ndarray
    radii_low: np.ndarray
    radii_high: np.ndarray
    current_low: np.ndarray
    current_high: np.ndarray
    ip_rate_abs_p99: float
    action_abs_p99: np.ndarray
    action_jump_abs_p99: np.ndarray


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small preview of new long synthetic T15 trajectories inside the "
            "safe coupled space of the successful real replay-window target library."
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
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/t15_synthetic_long_preview"))
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--wiggle-room", type=float, default=1.15)
    parser.add_argument("--safe-split", default="train", help="Use this split from the real library, or 'all'.")
    parser.add_argument("--radii-margin-m", type=float, default=0.025)
    parser.add_argument("--current-envelope-margin", type=float, default=0.08)
    parser.add_argument("--max-cloud-rows", type=int, default=20000)
    parser.add_argument("--knn", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    target_path = _repo_path(args.oracle_target)
    initial_path = _repo_path(args.initial_library)
    out_dir = _repo_path(args.out_dir)
    _require_inputs(target_path=target_path, initial_path=initial_path)

    rng = np.random.default_rng(int(args.seed))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[synthetic-preview] loading safe space from {target_path} split={args.safe_split}",
        flush=True,
    )
    real = _load_real_space(
        target_path=target_path,
        initial_path=initial_path,
        dt=float(args.dt),
        pca_components=int(args.pca_components),
        wiggle_room=float(args.wiggle_room),
        radii_margin_m=float(args.radii_margin_m),
        current_envelope_margin=float(args.current_envelope_margin),
        max_cloud_rows=int(args.max_cloud_rows),
        safe_split=str(args.safe_split),
        rng=rng,
    )
    print(
        "[synthetic-preview] "
        f"safe cloud rows={real.features.shape[0]} pca={real.pca_components.shape[0]} "
        f"current dims={real.currents.shape[1]}",
        flush=True,
    )
    trajectories, reject_counts = _generate_previews(
        real=real,
        examples=int(args.examples),
        min_steps=int(args.min_steps),
        max_steps=int(args.max_steps),
        dt=float(args.dt),
        wiggle_room=float(args.wiggle_room),
        knn=int(args.knn),
        max_attempts=int(args.max_attempts),
        progress_every=int(args.progress_every),
        rng=rng,
    )
    if not trajectories:
        summary = {
            "accepted": 0,
            "requested": int(args.examples),
            "reject_counts": reject_counts,
        }
        (out_dir / "synthetic_long_preview_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise RuntimeError(f"no synthetic long trajectories accepted; reject_counts={reject_counts}")

    npz_path = out_dir / "synthetic_long_preview_trajectories.npz"
    _write_preview_npz(npz_path, trajectories)
    summary = _write_summary(
        out_dir=out_dir,
        real=real,
        trajectories=trajectories,
        reject_counts=reject_counts,
        target_path=target_path,
        initial_path=initial_path,
        args=args,
    )
    if bool(args.plot):
        _write_plots(out_dir=out_dir, trajectories=trajectories, real=real, dt=float(args.dt))

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _repo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _require_inputs(*, target_path: Path, initial_path: Path) -> None:
    missing = [str(p) for p in (target_path, initial_path) if not p.exists()]
    if not missing:
        return
    raise FileNotFoundError(
        "Missing canonical real replay-window artifacts. Download these from the server first, "
        "or run this preview on the server inside the RL container:\n"
        f"  {target_path}\n"
        f"  {initial_path}\n"
        f"Missing: {missing}"
    )


def _load_real_space(
    *,
    target_path: Path,
    initial_path: Path,
    dt: float,
    pca_components: int,
    wiggle_room: float,
    radii_margin_m: float,
    current_envelope_margin: float,
    max_cloud_rows: int,
    rng: np.random.Generator,
    safe_split: str,
) -> RealSpace:
    with np.load(target_path, allow_pickle=False) as target, np.load(initial_path, allow_pickle=False) as init:
        ip = np.asarray(target["ip_target"], dtype=np.float64)
        radii = np.asarray(target["boundary_radii"], dtype=np.float64)
        action = np.asarray(target["real_jdot_action"], dtype=np.float64)
        current_limits = np.asarray(target["current_limits"], dtype=np.float64)
        derivative_limits = np.asarray(target["derivative_limits"], dtype=np.float64)
        shot_id = np.asarray(target["shot_id"], dtype=np.int64) if "shot_id" in target.files else np.arange(ip.shape[0])
        source_index = (
            np.asarray(target["source_index"], dtype=np.int64)
            if "source_index" in target.files
            else np.zeros((ip.shape[0],), dtype=np.int64)
        )
        time_s = (
            np.asarray(target["time_s"], dtype=np.float64)
            if "time_s" in target.files
            else np.zeros((ip.shape[0],), dtype=np.float64)
        )
        pfc0 = np.asarray(init["pfc0"], dtype=np.float64)
        sol0 = np.asarray(init["sol0"], dtype=np.float64)
        split = np.asarray(target["split"]).astype(str) if "split" in target.files else np.full(ip.shape[0], "")

    if ip.ndim != 2 or radii.ndim != 3 or action.ndim != 3:
        raise ValueError("expected ip [N,T+1], radii [N,T+1,A], action [N,T,C]")
    if ip.shape[:2] != radii.shape[:2] or action.shape[0] != ip.shape[0] or action.shape[1] != ip.shape[1] - 1:
        raise ValueError(f"incompatible shapes: ip={ip.shape}, radii={radii.shape}, action={action.shape}")
    if pfc0.shape[0] != ip.shape[0] or sol0.shape[0] != ip.shape[0]:
        raise ValueError("initial library row count does not match target library")
    if safe_split != "all":
        mask = split == str(safe_split)
        if not np.any(mask):
            raise ValueError(f"safe split {safe_split!r} matched zero rows")
        ip = ip[mask]
        radii = radii[mask]
        action = action[mask]
        shot_id = shot_id[mask]
        source_index = source_index[mask]
        time_s = time_s[mask]
        pfc0 = pfc0[mask]
        sol0 = sol0[mask]

    initial = np.concatenate([pfc0, sol0], axis=1)
    jdot = action * derivative_limits.reshape(1, 1, -1)
    currents = np.empty((initial.shape[0], action.shape[1] + 1, initial.shape[1]), dtype=np.float64)
    currents[:, 0, :] = initial
    currents[:, 1:, :] = initial[:, None, :] + np.cumsum(jdot * float(dt), axis=1)

    radii_flat = radii.reshape(-1, radii.shape[-1])
    sample = _sample_rows(radii_flat, max_rows=max_cloud_rows, rng=rng)
    radii_mean, components = _pca(sample, n_components=int(pca_components))
    coeffs = (radii_flat - radii_mean.reshape(1, -1)) @ components.T
    features = np.concatenate([ip.reshape(-1, 1), coeffs], axis=1)

    current_flat = currents.reshape(-1, currents.shape[-1])
    point_count = ip.shape[1]
    point_offset = np.tile(np.arange(point_count, dtype=np.int64), ip.shape[0])
    shot_flat = np.repeat(shot_id, point_count)
    source_index_flat = np.repeat(source_index, point_count) + point_offset
    time_flat = np.repeat(time_s, point_count) + point_offset.astype(np.float64) * float(dt)
    action_flat = action.reshape(-1, action.shape[-1])
    jdot_flat = jdot.reshape(-1, jdot.shape[-1])
    ip_rate = np.diff(ip, axis=1).reshape(-1) / float(dt)
    action_jump = np.diff(action, axis=1).reshape(-1, action.shape[-1])
    cloud_idx = _sample_indices(features.shape[0], max_rows=max_cloud_rows, rng=rng)
    feature_cloud = features[cloud_idx]
    current_cloud = current_flat[cloud_idx]
    feature_center = np.median(feature_cloud, axis=0)
    feature_scale = np.percentile(np.abs(feature_cloud - feature_center.reshape(1, -1)), 90, axis=0)
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 1.0e-12), feature_scale, 1.0)
    features_norm = (feature_cloud - feature_center.reshape(1, -1)) / feature_scale.reshape(1, -1)

    low_q = max(0.0, 0.002 / max(float(wiggle_room), 1.0))
    high_q = 1.0 - low_q
    return RealSpace(
        ip=ip,
        radii=radii,
        features=feature_cloud,
        currents=current_cloud,
        source_shot_id=shot_flat[cloud_idx],
        source_index=source_index_flat[cloud_idx],
        source_time_s=time_flat[cloud_idx],
        jdot=jdot_flat,
        action=action_flat,
        current_limits=current_limits,
        derivative_limits=derivative_limits,
        radii_mean=radii_mean,
        pca_components=components,
        feature_center=feature_center,
        feature_scale=feature_scale,
        features_norm=features_norm,
        feature_low=np.quantile(features, low_q, axis=0),
        feature_high=np.quantile(features, high_q, axis=0),
        radii_low=np.min(radii_flat, axis=0) - float(radii_margin_m),
        radii_high=np.max(radii_flat, axis=0) + float(radii_margin_m),
        current_low=np.min(current_flat, axis=0) - current_envelope_margin * current_limits,
        current_high=np.max(current_flat, axis=0) + current_envelope_margin * current_limits,
        ip_rate_abs_p99=float(np.percentile(np.abs(ip_rate), 99.0) * float(wiggle_room)),
        action_abs_p99=np.percentile(np.abs(action_flat), 99.5, axis=0) * float(wiggle_room),
        action_jump_abs_p99=np.percentile(np.abs(action_jump), 99.5, axis=0) * float(wiggle_room),
    )


def _sample_rows(arr: np.ndarray, *, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    idx = _sample_indices(arr.shape[0], max_rows=max_rows, rng=rng)
    return arr[idx]


def _sample_indices(n: int, *, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if int(max_rows) <= 0 or n <= int(max_rows):
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=int(max_rows), replace=False).astype(np.int64))


def _pca(values: np.ndarray, *, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    centered = values - mean.reshape(1, -1)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(max(1, int(n_components)), vt.shape[0])
    return mean.astype(np.float64), vt[:k].astype(np.float64)


def _generate_previews(
    *,
    real: RealSpace,
    examples: int,
    min_steps: int,
    max_steps: int,
    dt: float,
    wiggle_room: float,
    knn: int,
    max_attempts: int,
    progress_every: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, np.ndarray | int]], dict[str, int]]:
    accepted: list[dict[str, np.ndarray | int]] = []
    reject_counts: dict[str, int] = {}
    attempts = 0
    while len(accepted) < int(examples) and attempts < int(max_attempts):
        attempts += 1
        steps = int(rng.integers(int(min_steps), int(max_steps) + 1))
        start_idx = int(rng.integers(0, real.features.shape[0]))
        start_current = real.currents[start_idx].copy()
        features = _sample_feature_trajectory(real=real, start=real.features[start_idx], steps=steps, rng=rng)
        reason = _check_features(real, features)
        if reason is not None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            _maybe_print_progress(attempts, accepted, reject_counts, progress_every=progress_every)
            continue
        radii = _features_to_radii(real, features)
        reason = _check_radii(real, radii)
        if reason is not None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            _maybe_print_progress(attempts, accepted, reject_counts, progress_every=progress_every)
            continue
        currents = _estimate_currents(real=real, features=features, knn=int(knn))
        currents = _align_currents_to_start(currents, start_current=start_current)
        currents = _smooth_currents(currents, width=31)
        reason = _check_currents(real, currents, dt=dt, wiggle_room=wiggle_room)
        if reason is not None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            _maybe_print_progress(attempts, accepted, reject_counts, progress_every=progress_every)
            continue
        accepted.append(
            {
                "steps": steps,
                "features": features,
                "ip": features[:, 0],
                "radii": radii,
                "currents": currents,
                "jdot": np.diff(currents, axis=0) / float(dt),
                "reset_shot_id": int(real.source_shot_id[start_idx]),
                "reset_source_index": int(real.source_index[start_idx]),
                "reset_time_s": float(real.source_time_s[start_idx]),
            }
        )
        print(
            f"[synthetic-preview] accepted={len(accepted)}/{examples} attempts={attempts} steps={steps}",
            flush=True,
        )
    _maybe_print_progress(attempts, accepted, reject_counts, progress_every=1)
    return accepted, reject_counts


def _maybe_print_progress(
    attempts: int,
    accepted: list[dict[str, np.ndarray | int]],
    reject_counts: dict[str, int],
    *,
    progress_every: int,
) -> None:
    every = max(1, int(progress_every))
    if attempts <= 0 or attempts % every != 0:
        return
    top = ", ".join(f"{k}={v}" for k, v in sorted(reject_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4])
    print(
        f"[synthetic-preview] attempts={attempts} accepted={len(accepted)} rejects: {top or 'none'}",
        flush=True,
    )


def _sample_feature_trajectory(*, real: RealSpace, start: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    dims = start.shape[0]
    ip_profile, edges = _sample_ip_profile(real=real, start_ip=float(start[0]), steps=steps, rng=rng)
    features = np.empty((steps + 1, dims), dtype=np.float64)
    features[0] = start
    features[:, 0] = ip_profile
    current = start.copy()
    current[0] = ip_profile[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        duration = max(1, int(hi - lo))
        target = _sample_safe_waypoint(
            real=real,
            current=current,
            duration=duration,
            target_ip=float(ip_profile[hi]),
            rng=rng,
        )
        segment = _interpolate_waypoint(current, target, duration)
        segment[:, 0] = ip_profile[lo : hi + 1]
        features[lo + 1 : hi + 1] = segment[1:]
        current = target
        current[0] = ip_profile[hi]
    return features


def _sample_ip_profile(
    *,
    real: RealSpace,
    start_ip: float,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    low = float(real.feature_low[0])
    high = float(real.feature_high[0])
    start_ip = float(np.clip(start_ip, low, high))
    max_rate = max(1.0e5, float(real.ip_rate_abs_p99))

    mode = str(
        rng.choice(
            np.asarray(["hold", "ramp", "hold_ramp", "ramp_hold", "ramp_hold_reverse"]),
            p=np.asarray([0.08, 0.34, 0.20, 0.28, 0.10]),
        )
    )
    edges = _ip_segment_edges(mode=mode, steps=int(steps), rng=rng)
    profile = np.empty((int(steps) + 1,), dtype=np.float64)
    profile[0] = start_ip
    current = start_ip
    first_direction = _choose_ip_direction(current=current, low=low, high=high, rng=rng)

    for segment_idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        duration = max(1, int(hi - lo))
        kind = _ip_segment_kind(mode=mode, segment_idx=segment_idx, segment_count=len(edges) - 1)
        direction = first_direction
        if mode == "ramp_hold_reverse" and segment_idx == len(edges) - 2:
            direction = -first_direction
        if kind == "hold":
            end = _sample_ip_hold_endpoint(current=current, low=low, high=high, max_rate=max_rate, duration=duration, rng=rng)
        else:
            end = _sample_ip_ramp_endpoint(
                current=current,
                low=low,
                high=high,
                max_rate=max_rate,
                duration=duration,
                direction=direction,
                rng=rng,
            )
        profile[lo : hi + 1] = np.linspace(current, end, duration + 1, dtype=np.float64)
        current = float(end)

    return np.clip(profile, low, high), edges


def _ip_segment_edges(*, mode: str, steps: int, rng: np.random.Generator) -> np.ndarray:
    if mode in {"hold", "ramp"} or steps < 360:
        return np.asarray([0, steps], dtype=np.int64)
    min_seg = min(max(150, steps // 7), max(1, steps // 3))
    if mode in {"hold_ramp", "ramp_hold"}:
        cut = int(rng.integers(min_seg, steps - min_seg + 1))
        return np.asarray([0, cut, steps], dtype=np.int64)
    cut1 = int(rng.integers(min_seg, steps - 2 * min_seg + 1))
    cut2 = int(rng.integers(cut1 + min_seg, steps - min_seg + 1))
    return np.asarray([0, cut1, cut2, steps], dtype=np.int64)


def _ip_segment_kind(*, mode: str, segment_idx: int, segment_count: int) -> str:
    if mode == "hold":
        return "hold"
    if mode == "ramp":
        return "ramp"
    if mode == "hold_ramp":
        return "hold" if segment_idx == 0 else "ramp"
    if mode == "ramp_hold":
        return "ramp" if segment_idx == 0 else "hold"
    if mode == "ramp_hold_reverse":
        if segment_idx == 0 or segment_idx == segment_count - 1:
            return "ramp"
        return "hold"
    return "ramp"


def _choose_ip_direction(*, current: float, low: float, high: float, rng: np.random.Generator) -> int:
    up_room = max(0.0, high - current)
    down_room = max(0.0, current - low)
    if up_room < 5000.0 and down_room < 5000.0:
        return 1
    if up_room < 5000.0:
        return -1
    if down_room < 5000.0:
        return 1
    p_up = 0.25 + 0.5 * up_room / max(up_room + down_room, 1.0)
    return 1 if rng.random() < p_up else -1


def _sample_ip_hold_endpoint(
    *,
    current: float,
    low: float,
    high: float,
    max_rate: float,
    duration: int,
    rng: np.random.Generator,
) -> float:
    return float(np.clip(current, low, high))


def _sample_ip_ramp_endpoint(
    *,
    current: float,
    low: float,
    high: float,
    max_rate: float,
    duration: int,
    direction: int,
    rng: np.random.Generator,
) -> float:
    direction = 1 if int(direction) >= 0 else -1
    room = (high - current) if direction > 0 else (current - low)
    if room <= 5000.0:
        return _sample_ip_hold_endpoint(current=current, low=low, high=high, max_rate=max_rate, duration=duration, rng=rng)
    duration_s = float(duration) * 0.001
    rate = float(rng.uniform(0.35, 0.85) * max_rate)
    max_delta = min(room, rate * duration_s)
    min_delta = min(max_delta, max(8000.0, 0.35 * max_delta))
    delta = float(rng.uniform(min_delta, max_delta)) if max_delta > 1000.0 else 0.0
    return float(np.clip(current + direction * delta, low, high))


def _sample_safe_waypoint(
    *,
    real: RealSpace,
    current: np.ndarray,
    duration: int,
    target_ip: float,
    rng: np.random.Generator,
) -> np.ndarray:
    current_n = (current - real.feature_center) / real.feature_scale
    dist = np.sqrt(np.sum((real.features_norm - current_n.reshape(1, -1)) ** 2, axis=1))

    # Longer segments may travel farther, but the target is always a coupled
    # state sampled from the real cloud rather than independently sampled axes.
    max_dist = np.clip(0.20 + 0.004 * float(duration), 0.45, 1.75)
    min_dist = 0.04 if rng.random() > 0.18 else 0.0
    ip_tol = max(12000.0, 0.07 * (float(real.feature_high[0]) - float(real.feature_low[0])))
    ip_mask = np.abs(real.features[:, 0] - float(target_ip)) <= ip_tol
    candidates = np.flatnonzero((dist >= min_dist) & (dist <= max_dist) & ip_mask)
    if candidates.size == 0:
        ip_tol = max(25000.0, 0.14 * (float(real.feature_high[0]) - float(real.feature_low[0])))
        ip_mask = np.abs(real.features[:, 0] - float(target_ip)) <= ip_tol
        candidates = np.flatnonzero((dist >= min_dist) & (dist <= max_dist) & ip_mask)
    if candidates.size == 0:
        candidates = np.argsort(dist)[: min(128, dist.shape[0])]
    idx = int(rng.choice(candidates))
    target = real.features[idx].copy()
    target[0] = float(target_ip)

    # Make it new without leaving the local safe manifold: perturb only a small
    # distance in normalized feature space, then clip to the safe bounds.
    perturb_n = rng.normal(0.0, 0.025, size=current.shape[0])
    perturb_n[0] = 0.0
    target = target + perturb_n * real.feature_scale
    return np.minimum(np.maximum(target, real.feature_low), real.feature_high)


def _interpolate_waypoint(start: np.ndarray, target: np.ndarray, duration: int) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, int(duration) + 1, dtype=np.float64).reshape(-1, 1)
    # Linear in the middle with eased bends at the joins. This keeps long shots
    # readable while avoiding a hard derivative corner at every waypoint.
    bend = 0.5 - 0.5 * np.cos(np.pi * alpha)
    return start.reshape(1, -1) + bend * (target - start).reshape(1, -1)


def _check_features(real: RealSpace, features: np.ndarray) -> str | None:
    if not np.all(np.isfinite(features)):
        return "feature_nonfinite"
    if np.any(features < real.feature_low.reshape(1, -1)):
        return "feature_below_safe_space"
    if np.any(features > real.feature_high.reshape(1, -1)):
        return "feature_above_safe_space"
    return None


def _features_to_radii(real: RealSpace, features: np.ndarray) -> np.ndarray:
    coeffs = features[:, 1:]
    return real.radii_mean.reshape(1, -1) + coeffs @ real.pca_components


def _check_radii(real: RealSpace, radii: np.ndarray) -> str | None:
    if not np.all(np.isfinite(radii)):
        return "radii_nonfinite"
    if np.any(radii < real.radii_low.reshape(1, -1)):
        return "radii_below_safe_space"
    if np.any(radii > real.radii_high.reshape(1, -1)):
        return "radii_above_safe_space"
    return None


def _estimate_currents(*, real: RealSpace, features: np.ndarray, knn: int) -> np.ndarray:
    query = (features - real.feature_center.reshape(1, -1)) / real.feature_scale.reshape(1, -1)
    out = np.empty((features.shape[0], real.currents.shape[1]), dtype=np.float64)
    k = max(1, min(int(knn), real.features_norm.shape[0]))
    for start in range(0, query.shape[0], 128):
        block = query[start : start + 128]
        dist2 = np.sum((block[:, None, :] - real.features_norm[None, :, :]) ** 2, axis=2)
        idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        local_dist = np.take_along_axis(dist2, idx, axis=1)
        weights = 1.0 / (local_dist + 1.0e-6)
        weights /= np.sum(weights, axis=1, keepdims=True)
        out[start : start + block.shape[0]] = np.sum(real.currents[idx] * weights[:, :, None], axis=1)
    return out


def _align_currents_to_start(currents: np.ndarray, *, start_current: np.ndarray) -> np.ndarray:
    out = np.asarray(currents, dtype=np.float64).copy()
    start = np.asarray(start_current, dtype=np.float64).reshape(1, -1)
    if out.ndim != 2 or start.shape[1] != out.shape[1]:
        raise ValueError(f"current shape mismatch: currents={out.shape} start={start.shape}")
    out += start - out[:1]
    out[0] = start[0]
    return out


def _smooth_currents(currents: np.ndarray, *, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width <= 1:
        return currents
    pad = width // 2
    kernel = np.ones((width,), dtype=np.float64) / float(width)
    padded = np.pad(currents, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(currents)
    for c in range(currents.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")[: currents.shape[0]]
    out[0] = currents[0]
    out[-1] = currents[-1]
    return out


def _check_currents(real: RealSpace, currents: np.ndarray, *, dt: float, wiggle_room: float) -> str | None:
    if not np.all(np.isfinite(currents)):
        return "current_nonfinite"
    if np.any(np.abs(currents) > real.current_limits.reshape(1, -1)):
        return "current_hard_limit"
    if np.any(currents < real.current_low.reshape(1, -1)):
        return "current_below_safe_space"
    if np.any(currents > real.current_high.reshape(1, -1)):
        return "current_above_safe_space"
    jdot = np.diff(currents, axis=0) / float(dt)
    if np.any(np.abs(jdot) > real.derivative_limits.reshape(1, -1)):
        return "jdot_hard_limit"
    action = jdot / real.derivative_limits.reshape(1, -1)
    if np.any(np.abs(action) > np.maximum(real.action_abs_p99.reshape(1, -1), 0.05) * float(wiggle_room)):
        return "action_outside_realistic_range"
    action_jump = np.diff(action, axis=0)
    if action_jump.size and np.any(
        np.abs(action_jump) > np.maximum(real.action_jump_abs_p99.reshape(1, -1), 0.03) * float(wiggle_room)
    ):
        return "action_jump_outside_realistic_range"
    return None


def _write_preview_npz(path: Path, trajectories: list[dict[str, np.ndarray | int]]) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray("t15_synthetic_long_preview_v1"),
        steps=np.asarray([int(t["steps"]) for t in trajectories], dtype=np.int64),
        reset_shot_id=np.asarray([int(t.get("reset_shot_id", -1)) for t in trajectories], dtype=np.int64),
        reset_source_index=np.asarray([int(t.get("reset_source_index", -1)) for t in trajectories], dtype=np.int64),
        reset_time_s=np.asarray([float(t.get("reset_time_s", np.nan)) for t in trajectories], dtype=np.float64),
        ip=np.asarray([_pad_2d(np.asarray(t["ip"], dtype=np.float64), max_len=max(int(x["steps"]) + 1 for x in trajectories)) for t in trajectories]),
        radii=np.asarray([_pad_2d(np.asarray(t["radii"], dtype=np.float64), max_len=max(int(x["steps"]) + 1 for x in trajectories)) for t in trajectories]),
        currents=np.asarray([_pad_2d(np.asarray(t["currents"], dtype=np.float64), max_len=max(int(x["steps"]) + 1 for x in trajectories)) for t in trajectories]),
        jdot=np.asarray([_pad_2d(np.asarray(t["jdot"], dtype=np.float64), max_len=max(int(x["steps"]) for x in trajectories)) for t in trajectories]),
    )


def _pad_2d(arr: np.ndarray, *, max_len: int) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[:, None]
    out = np.full((max_len, arr.shape[1]), np.nan, dtype=np.float64)
    out[: arr.shape[0]] = arr
    return out


def _write_summary(
    *,
    out_dir: Path,
    real: RealSpace,
    trajectories: list[dict[str, np.ndarray | int]],
    reject_counts: dict[str, int],
    target_path: Path,
    initial_path: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    accepted_steps = [int(t["steps"]) for t in trajectories]
    action_max = []
    current_usage = []
    for t in trajectories:
        currents = np.asarray(t["currents"], dtype=np.float64)
        jdot = np.asarray(t["jdot"], dtype=np.float64)
        action = jdot / real.derivative_limits.reshape(1, -1)
        action_max.append(float(np.max(np.abs(action))))
        current_usage.append(float(np.max(np.abs(currents) / real.current_limits.reshape(1, -1))))
    summary = {
        "schema": "t15_synthetic_long_preview_summary_v1",
        "oracle_target": str(target_path),
        "initial_library": str(initial_path),
        "safe_split": str(args.safe_split),
        "accepted": len(trajectories),
        "requested": int(args.examples),
        "reject_counts": dict(sorted(reject_counts.items())),
        "steps": {
            "min": int(np.min(accepted_steps)),
            "max": int(np.max(accepted_steps)),
            "mean": float(np.mean(accepted_steps)),
        },
        "safe_space": {
            "feature_names": ["Ip", *[f"boundary_pca_{i}" for i in range(real.pca_components.shape[0])]],
            "feature_low": real.feature_low.tolist(),
            "feature_high": real.feature_high.tolist(),
            "radii_low_min": float(np.min(real.radii_low)),
            "radii_high_max": float(np.max(real.radii_high)),
            "current_limits": real.current_limits.tolist(),
            "derivative_limits": real.derivative_limits.tolist(),
        },
        "accepted_current_usage_fraction_max": {
            "mean": float(np.mean(current_usage)),
            "max": float(np.max(current_usage)),
        },
        "accepted_action_abs_max": {
            "mean": float(np.mean(action_max)),
            "max": float(np.max(action_max)),
        },
    }
    (out_dir / "synthetic_long_preview_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(out_dir / "synthetic_long_preview_report.md", summary)
    return summary


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# T15 Synthetic Long Preview",
        "",
        f"- Accepted: {summary['accepted']} / {summary['requested']}",
        f"- Step range: {summary['steps']['min']}..{summary['steps']['max']}",
        f"- Mean max current usage: {summary['accepted_current_usage_fraction_max']['mean']:.4f}",
        f"- Max current usage: {summary['accepted_current_usage_fraction_max']['max']:.4f}",
        f"- Mean max normalized action: {summary['accepted_action_abs_max']['mean']:.4f}",
        f"- Max normalized action: {summary['accepted_action_abs_max']['max']:.4f}",
        "",
        "## Rejections",
        "",
        "| reason | count |",
        "|---|---:|",
    ]
    rejects = summary.get("reject_counts", {})
    if isinstance(rejects, dict) and rejects:
        for key, value in rejects.items():
            lines.append(f"| `{key}` | {value} |")
    else:
        lines.append("| none | 0 |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(*, out_dir: Path, trajectories: list[dict[str, np.ndarray | int]], real: RealSpace, dt: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["PFC0", "PFC1", "PFC2", "PFC3", "PFC4", "PFC5", "SOL0", "SOL1", "SOL2"]
    theta = np.linspace(0.0, 2.0 * np.pi, real.radii_mean.shape[0], endpoint=False)
    angle_idx = np.linspace(0, real.radii_mean.shape[0] - 1, 8, dtype=int)

    for i, item in enumerate(trajectories):
        steps = int(item["steps"])
        t = np.arange(steps + 1, dtype=np.float64) * float(dt)
        ip = np.asarray(item["ip"], dtype=np.float64)
        radii = np.asarray(item["radii"], dtype=np.float64)
        currents = np.asarray(item["currents"], dtype=np.float64)
        jdot = np.asarray(item["jdot"], dtype=np.float64)
        tj = np.arange(steps, dtype=np.float64) * float(dt)

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

        fig.suptitle(f"Synthetic long preview {i:02d} ({steps} steps)", fontsize=14)
        fig.savefig(out_dir / f"synthetic_long_preview_{i:02d}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
