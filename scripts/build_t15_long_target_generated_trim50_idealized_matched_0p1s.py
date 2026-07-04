#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import build_t15_simple_manifold_generated_trim50_idealized_0p1s as simple


DEFAULT_BOUNDARY_PARAM_DIR = Path("../tokamak-sim/output/t15_boundary_parameters_trim50_idealized_matched_gpu_plain_1e6")
DEFAULT_DATA_ROOT = Path("../tokamak-sim/data/t15_data_new_trim50_idealized_matched")
DEFAULT_MACHINE_CONFIG = Path(
    "../tokamak-sim/runs/t15md_limited_replay_dataset_trim50_idealized_matched_gpu_plain_1e6/"
    "T15MD_new_data_legacy_contour_limited_replay.toml"
)
DEFAULT_OUT_DIR = Path("data/processed/t15_long_target_generated_trim50_plain_gpu1e6_oracle_window_0p1s")
DEFAULT_INITIAL_STATES_OUT = Path(
    "data/processed/t15_long_target_generated_trim50_plain_gpu1e6_oracle_window_0p1s_initial_states.npz"
)
DEFAULT_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_long_target_generated_targets.npz"
DEFAULT_ORACLE_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_replay_window_oracle_targets.npz"
DEFAULT_PARENTS_OUT = DEFAULT_OUT_DIR / "t15_long_target_generated_parents.npz"
DEFAULT_TRAIN_SHOTS = ("3856", "3857", "3858", "3863")
DEFAULT_HOLDOUT_SHOTS = ("3864",)

PARAM_COLUMNS = simple.PARAM_COLUMNS
COIL_NAMES = simple.COIL_NAMES

# The conservative "safe state space" used by the earlier simple-manifold
# generator. Keep these explicit so this long-parent path cannot quietly drift
# into the broad independent-box generator again.
SAFE_IP = (140285.0, 426402.0)
SAFE_A0 = (0.434, 0.659)
SAFE_E = (0.134, 0.296)
SAFE_DELTA = (0.070, 0.177)


@dataclass(frozen=True, slots=True)
class SourceShot:
    shot: str
    split: str
    time_s: np.ndarray
    source_index: np.ndarray
    x: np.ndarray
    params: np.ndarray
    coils: np.ndarray


@dataclass(frozen=True, slots=True)
class ResetPoint:
    shot: str
    split: str
    source_index: int
    time_s: float
    x: np.ndarray
    params: np.ndarray
    coils: np.ndarray


@dataclass(frozen=True, slots=True)
class MoveSample:
    shot: str
    start: int
    dx: np.ndarray
    dcoils: np.ndarray


@dataclass(frozen=True, slots=True)
class ParentTarget:
    parent_id: int
    source_shot: str
    split: str
    reset_source_index: int
    reset_time_s: float
    parent_steps: int
    waypoint_count: int
    time_s: np.ndarray
    x: np.ndarray
    params: np.ndarray
    radii: np.ndarray
    coils: np.ndarray
    state_distance_p95: float


@dataclass(frozen=True, slots=True)
class WindowTarget:
    parent: ParentTarget
    parent_step: int
    source_index: int
    time_s: float
    ip_ref: np.ndarray
    params_ref: np.ndarray
    radii_ref: np.ndarray
    coil_witness: np.ndarray
    difficulty_bin: str
    move_distance: float


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build synthetic 1.0-1.2 s Ip/boundary/coils parent trajectories from "
            "real reset states and the conservative replay-derived safe state space, "
            "then cut them densely into 100-step replay-window oracle targets."
        )
    )
    parser.add_argument("--boundary-param-dir", type=Path, default=DEFAULT_BOUNDARY_PARAM_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--machine-config", type=Path, default=DEFAULT_MACHINE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-states-out", type=Path, default=DEFAULT_INITIAL_STATES_OUT)
    parser.add_argument("--targets-out", type=Path, default=DEFAULT_TARGETS_OUT)
    parser.add_argument("--oracle-targets-out", type=Path, default=DEFAULT_ORACLE_TARGETS_OUT)
    parser.add_argument("--parents-out", type=Path, default=DEFAULT_PARENTS_OUT)
    parser.add_argument("--train-shots", nargs="+", default=list(DEFAULT_TRAIN_SHOTS))
    parser.add_argument("--holdout-shots", nargs="+", default=list(DEFAULT_HOLDOUT_SHOTS))
    parser.add_argument("--window-steps", type=int, default=100)
    parser.add_argument("--parent-count", type=int, default=12)
    parser.add_argument("--parent-min-steps", type=int, default=1000)
    parser.add_argument("--parent-max-steps", type=int, default=1200)
    parser.add_argument("--parent-stride", type=int, default=1)
    parser.add_argument("--parent-segment-min-steps", type=int, default=180)
    parser.add_argument("--parent-segment-max-steps", type=int, default=320)
    parser.add_argument("--join-blend-steps", type=int, default=12)
    parser.add_argument("--state-distance-limit", type=float, default=0.08)
    parser.add_argument("--move-distance-limit", type=float, default=0.08)
    parser.add_argument("--current-usage-cap", type=float, default=0.75)
    parser.add_argument("--endpoint-distance-min", type=float, default=0.006)
    parser.add_argument("--endpoint-distance-max", type=float, default=0.11)
    parser.add_argument("--endpoint-mix-min", type=float, default=0.92)
    parser.add_argument("--endpoint-mix-max", type=float, default=1.05)
    parser.add_argument("--hold-probability", type=float, default=0.22)
    parser.add_argument("--max-attempts-per-parent", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    window_steps = int(args.window_steps)
    if window_steps != 100:
        raise SystemExit("this builder intentionally emits 0.1 s / 100-step windows")
    if int(args.parent_min_steps) < window_steps:
        raise SystemExit("parent_min_steps must be at least window_steps")
    if int(args.parent_max_steps) < int(args.parent_min_steps):
        raise SystemExit("parent_max_steps must be >= parent_min_steps")
    if int(args.parent_stride) <= 0:
        raise SystemExit("parent_stride must be positive")
    if int(args.parent_segment_min_steps) <= 0 or int(args.parent_segment_max_steps) < int(args.parent_segment_min_steps):
        raise SystemExit("invalid parent segment length bounds")

    rng = np.random.default_rng(int(args.seed))
    train_shots = tuple(str(int(v)) for v in args.train_shots)
    holdout_shots = tuple(str(int(v)) for v in args.holdout_shots)
    overlap = sorted(set(train_shots) & set(holdout_shots), key=int)
    if overlap:
        raise SystemExit("train and holdout shots overlap: " + ", ".join(overlap))

    limits = simple._load_limits(args.machine_config.resolve())
    sources = _load_sources(
        args.boundary_param_dir.resolve(),
        args.data_root.resolve(),
        train_shots=train_shots,
        holdout_shots=holdout_shots,
    )
    theta = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    all_points = _safe_reset_points_from_sources(
        sources,
        limits=limits,
        current_usage_cap=float(args.current_usage_cap),
    )
    if not all_points:
        raise SystemExit("no real replay rows survived the conservative safe-state reset filter")
    state_space = _state_space_from_points(all_points, limits=limits)
    move_samples = _move_samples_from_sources(sources, steps=window_steps, limits=limits)
    move_space = _move_space_from_samples(move_samples, limits=limits)
    envelope = _build_envelope(sources=sources, theta=theta, limits=limits)

    split_counts = _split_parent_counts(int(args.parent_count), points=all_points)
    parents: list[ParentTarget] = []
    parent_rejections = Counter()
    for split, count in split_counts.items():
        split_points = [p for p in all_points if p.split == split]
        parents.extend(
            _generate_parents(
                split_points,
                count=count,
                first_parent_id=len(parents),
                rng=rng,
                parent_min_steps=int(args.parent_min_steps),
                parent_max_steps=int(args.parent_max_steps),
                segment_min_steps=int(args.parent_segment_min_steps),
                segment_max_steps=int(args.parent_segment_max_steps),
                join_blend_steps=int(args.join_blend_steps),
                endpoint_distance_min=float(args.endpoint_distance_min),
                endpoint_distance_max=float(args.endpoint_distance_max),
                endpoint_mix_min=float(args.endpoint_mix_min),
                endpoint_mix_max=float(args.endpoint_mix_max),
                hold_probability=float(args.hold_probability),
                theta=theta,
                limits=limits,
                envelope=envelope,
                state_space=state_space,
                move_samples=move_samples,
                state_distance_limit=float(args.state_distance_limit),
                current_usage_cap=float(args.current_usage_cap),
                max_attempts_per_parent=int(args.max_attempts_per_parent),
                rejections=parent_rejections,
            )
        )

    window_rejections = Counter()
    windows = _cut_parents(
        parents,
        window_steps=window_steps,
        stride=int(args.parent_stride),
        move_space=move_space,
        limits=limits,
        move_distance_limit=float(args.move_distance_limit),
        rejections=window_rejections,
    )
    if not windows:
        raise SystemExit(f"no generated long-parent windows were accepted; window_rejections={dict(window_rejections)}")
    moving_windows = sum(1 for w in windows if w.move_distance > 1.0e-8)
    nonflat_ip_windows = sum(1 for w in windows if w.difficulty_bin != "flat")
    if moving_windows == 0 or nonflat_ip_windows == 0:
        raise SystemExit(
            "generated long-parent dataset collapsed to flat windows; "
            f"moving_windows={moving_windows}, nonflat_ip_windows={nonflat_ip_windows}, "
            f"window_rejections={dict(window_rejections)}"
        )

    _write_libraries(
        windows,
        parents=parents,
        initial_states_out=args.initial_states_out,
        targets_out=args.targets_out,
        oracle_targets_out=args.oracle_targets_out,
        parents_out=args.parents_out,
        limits=limits,
        train_shots=train_shots,
        holdout_shots=holdout_shots,
    )
    summary = _summary(
        args=args,
        sources=sources,
        safe_points=all_points,
        parents=parents,
        windows=windows,
        parent_rejections=parent_rejections,
        window_rejections=window_rejections,
        envelope=envelope,
        limits=limits,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "long_target_generated_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(args.out_dir / "summary" / "long_target_generated_dataset_report.md", summary)
    if args.plots:
        _write_plots(parents=parents, windows=windows, sources=sources, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_sources(
    boundary_param_dir: Path,
    data_root: Path,
    *,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
) -> list[SourceShot]:
    wanted = set(train_shots) | set(holdout_shots)
    rows_by_shot = simple._load_boundary_rows(boundary_param_dir, wanted=wanted)
    sources: list[SourceShot] = []
    holdout = set(holdout_shots)
    for shot in sorted(rows_by_shot, key=int):
        rows = rows_by_shot[shot]
        coil_t, coil_values = simple._load_coils(data_root / "coils" / f"t15md_{shot}_coils.csv")
        time_s = np.asarray([r["t"] for r in rows], dtype=float)
        nearest = np.asarray([int(np.argmin(np.abs(coil_t - float(t)))) for t in time_s], dtype=np.int64)
        x = np.asarray([[r["Ip"], r["A0"], r["kappa"] - 1.0, r["delta"]] for r in rows], dtype=float)
        params = np.asarray([[r[name] for name in PARAM_COLUMNS] for r in rows], dtype=float)
        coils = coil_values[nearest].astype(float)
        finite = (
            np.isfinite(time_s)
            & np.all(np.isfinite(x), axis=1)
            & np.all(np.isfinite(params), axis=1)
            & np.all(np.isfinite(coils), axis=1)
        )
        if not np.any(finite):
            continue
        sources.append(
            SourceShot(
                shot=str(shot),
                split="holdout" if shot in holdout else "train",
                time_s=time_s[finite],
                source_index=nearest[finite],
                x=x[finite],
                params=params[finite],
                coils=coils[finite],
            )
        )
    missing = sorted(wanted - {s.shot for s in sources}, key=int)
    if missing:
        raise ValueError("missing usable source shots: " + ", ".join(missing))
    return sources


def _safe_reset_points_from_sources(
    sources: list[SourceShot],
    *,
    limits: simple.Limits,
    current_usage_cap: float,
) -> list[ResetPoint]:
    rows: list[ResetPoint] = []
    for source in sources:
        for i in range(source.x.shape[0]):
            x = source.x[i]
            coils = source.coils[i]
            if not _x_inside_safe_bounds(x):
                continue
            sol_usage = np.max(np.abs(coils[:3]) / limits.sol_current)
            pfc_usage = np.max(np.abs(coils[3:]) / limits.pfc_current)
            if max(float(sol_usage), float(pfc_usage)) > float(current_usage_cap) + 1.0e-9:
                continue
            rows.append(
                ResetPoint(
                    shot=source.shot,
                    split=source.split,
                    source_index=int(source.source_index[i]),
                    time_s=float(source.time_s[i]),
                    x=np.asarray(x, dtype=float),
                    params=np.asarray(source.params[i], dtype=float),
                    coils=np.asarray(coils, dtype=float),
                )
            )
    missing_safe = sorted({s.shot for s in sources} - {p.shot for p in rows}, key=int)
    if missing_safe:
        raise ValueError("no safe reset rows for shots: " + ", ".join(missing_safe))
    return rows


def _state_space_from_points(points: list[ResetPoint], *, limits: simple.Limits) -> dict[str, object]:
    arr = np.asarray([simple._state_features(p.x, p.coils, limits=limits) for p in points], dtype=float)
    tree = simple.cKDTree(arr) if simple.cKDTree is not None else None
    return {"values": arr, "tree": tree}


def _move_space_from_sources(sources: list[SourceShot], *, steps: int, limits: simple.Limits) -> dict[str, object]:
    return _move_space_from_samples(_move_samples_from_sources(sources, steps=steps, limits=limits), limits=limits)


def _move_samples_from_sources(sources: list[SourceShot], *, steps: int, limits: simple.Limits) -> list[MoveSample]:
    samples: list[MoveSample] = []
    for source in sources:
        if source.x.shape[0] <= int(steps):
            continue
        for start in range(0, source.x.shape[0] - int(steps)):
            end = start + int(steps)
            if not _x_inside_safe_bounds(source.x[start]) or not _x_inside_safe_bounds(source.x[end]):
                continue
            dx = source.x[end] - source.x[start]
            dcoils = source.coils[end] - source.coils[start]
            samples.append(
                MoveSample(
                    shot=source.shot,
                    start=int(start),
                    dx=np.asarray(dx, dtype=float),
                    dcoils=np.asarray(dcoils, dtype=float),
                )
            )
    if not samples:
        raise ValueError("could not build 100-step real move-space from source shots")
    return samples


def _move_space_from_samples(samples: list[MoveSample], *, limits: simple.Limits) -> dict[str, object]:
    values = [simple._move_features(s.dx, s.dcoils, limits=limits) for s in samples]
    arr = np.asarray(values, dtype=float)
    tree = simple.cKDTree(arr) if simple.cKDTree is not None else None
    return {"values": arr, "tree": tree}


def _build_envelope(*, sources: list[SourceShot], theta: np.ndarray, limits: simple.Limits) -> dict[str, np.ndarray | float]:
    real_x = np.concatenate([s.x for s in sources], axis=0)
    real_params = np.concatenate([s.params for s in sources], axis=0)
    real_radii = np.concatenate([simple._radii_from_params(s.params, theta) for s in sources], axis=0)
    real_coils = np.concatenate([s.coils for s in sources], axis=0)
    sol_usage = np.nanmax(np.abs(real_coils[:, :3]) / float(limits.sol_current))
    pfc_usage = np.nanmax(np.abs(real_coils[:, 3:]) / float(limits.pfc_current))
    return {
        "x_lo": np.nanmin(real_x, axis=0),
        "x_hi": np.nanmax(real_x, axis=0),
        "params_lo": np.nanmin(real_params, axis=0),
        "params_hi": np.nanmax(real_params, axis=0),
        "radii_lo": np.nanmin(real_radii, axis=0) - 0.05,
        "radii_hi": np.nanmax(real_radii, axis=0) + 0.05,
        "real_current_usage_max": float(max(sol_usage, pfc_usage)),
    }


def _split_parent_counts(parent_count: int, *, points: list[ResetPoint]) -> dict[str, int]:
    if int(parent_count) < 2:
        raise ValueError("parent_count must be at least 2 so train and holdout are both represented")
    train_points = sum(1 for p in points if p.split == "train")
    holdout_points = sum(1 for p in points if p.split == "holdout")
    total = max(train_points + holdout_points, 1)
    holdout = max(1, int(round(float(parent_count) * holdout_points / total)))
    train = max(1, int(parent_count) - holdout)
    return {"train": train, "holdout": holdout}


def _generate_parents(
    points: list[ResetPoint],
    *,
    count: int,
    first_parent_id: int,
    rng: np.random.Generator,
    parent_min_steps: int,
    parent_max_steps: int,
    segment_min_steps: int,
    segment_max_steps: int,
    join_blend_steps: int,
    endpoint_distance_min: float,
    endpoint_distance_max: float,
    endpoint_mix_min: float,
    endpoint_mix_max: float,
    hold_probability: float,
    theta: np.ndarray,
    limits: simple.Limits,
    envelope: dict[str, np.ndarray | float],
    state_space: dict[str, object],
    move_samples: list[MoveSample],
    state_distance_limit: float,
    current_usage_cap: float,
    max_attempts_per_parent: int,
    rejections: Counter,
) -> list[ParentTarget]:
    if not points:
        raise RuntimeError("no reset points for split")
    by_shot: dict[str, list[ResetPoint]] = {}
    for point in points:
        by_shot.setdefault(point.shot, []).append(point)
    split_move_samples = [sample for sample in move_samples if sample.shot in by_shot]
    if not split_move_samples:
        raise RuntimeError("no real 100-step move primitives are available for this split")
    shots = sorted(by_shot, key=int)
    shot_cycle = np.resize(np.asarray(shots, dtype=object), max(int(count), len(shots)))
    rng.shuffle(shot_cycle)
    accepted: list[ParentTarget] = []
    attempts = 0
    max_attempts = max(int(max_attempts_per_parent) * max(int(count), 1), 1)
    while len(accepted) < int(count) and attempts < max_attempts:
        attempts += 1
        shot = str(shot_cycle[len(accepted) % len(shot_cycle)])
        reset = by_shot[shot][int(rng.integers(0, len(by_shot[shot])))]
        parent_steps = int(rng.integers(int(parent_min_steps), int(parent_max_steps) + 1))
        parent = _make_parent(
            reset,
            parent_id=first_parent_id + len(accepted),
            parent_steps=parent_steps,
            points=points,
            move_samples=split_move_samples,
            rng=rng,
            segment_min_steps=segment_min_steps,
            segment_max_steps=segment_max_steps,
            join_blend_steps=join_blend_steps,
            endpoint_distance_min=endpoint_distance_min,
            endpoint_distance_max=endpoint_distance_max,
            endpoint_mix_min=endpoint_mix_min,
            endpoint_mix_max=endpoint_mix_max,
            hold_probability=hold_probability,
            theta=theta,
            limits=limits,
        )
        ok, reason, state_distance = _parent_ok(
            parent,
            limits=limits,
            envelope=envelope,
            state_space=state_space,
            state_distance_limit=state_distance_limit,
            current_usage_cap=current_usage_cap,
        )
        if not ok:
            rejections[reason] += 1
            continue
        accepted.append(
            ParentTarget(
                parent_id=parent.parent_id,
                source_shot=parent.source_shot,
                split=parent.split,
                reset_source_index=parent.reset_source_index,
                reset_time_s=parent.reset_time_s,
                parent_steps=parent.parent_steps,
                waypoint_count=parent.waypoint_count,
                time_s=parent.time_s,
                x=parent.x,
                params=parent.params,
                radii=parent.radii,
                coils=parent.coils,
                state_distance_p95=float(state_distance),
            )
        )
    if len(accepted) < int(count):
        raise RuntimeError(f"accepted only {len(accepted)} / {count} long target parents; rejections={dict(rejections)}")
    return accepted


def _make_parent(
    reset: ResetPoint,
    *,
    parent_id: int,
    parent_steps: int,
    points: list[ResetPoint],
    move_samples: list[MoveSample],
    rng: np.random.Generator,
    segment_min_steps: int,
    segment_max_steps: int,
    join_blend_steps: int,
    endpoint_distance_min: float,
    endpoint_distance_max: float,
    endpoint_mix_min: float,
    endpoint_mix_max: float,
    hold_probability: float,
    theta: np.ndarray,
    limits: simple.Limits,
) -> ParentTarget:
    knots = _parent_knots(
        parent_steps,
        rng=rng,
        segment_min_steps=segment_min_steps,
        segment_max_steps=segment_max_steps,
    )
    way_x = [reset.x.copy()]
    way_coils = [reset.coils.copy()]
    current_x = reset.x.copy()
    current_coils = reset.coils.copy()
    last_was_hold = False
    point_features = np.asarray([simple._state_features(p.x, p.coils, limits=limits) for p in points], dtype=float)
    for knot in knots[1:]:
        do_hold = rng.random() < float(hold_probability) and not last_was_hold
        if do_hold:
            next_x = current_x.copy()
            next_coils = current_coils.copy()
            last_was_hold = True
        else:
            move = _choose_move_sample(
                current_x=current_x,
                current_coils=current_coils,
                points=points,
                point_features=point_features,
                move_samples=move_samples,
                rng=rng,
                limits=limits,
                distance_min=endpoint_distance_min,
                distance_max=endpoint_distance_max,
            )
            mix = float(rng.uniform(float(endpoint_mix_min), float(endpoint_mix_max)))
            seg_steps = int(knot)
            prev_step = int(knots[len(way_x) - 1])
            span = max(float(seg_steps - prev_step), 1.0)
            scale = mix * span / 100.0
            next_x = current_x + scale * move.dx
            next_coils = current_coils + scale * move.dcoils
            last_was_hold = False
        way_x.append(next_x)
        way_coils.append(next_coils)
        current_x = next_x
        current_coils = next_coils

    x = simple._piecewise_linear([int(k) for k in knots], way_x, steps=int(parent_steps), join_blend_steps=int(join_blend_steps))
    coils = simple._piecewise_linear(
        [int(k) for k in knots],
        way_coils,
        steps=int(parent_steps),
        join_blend_steps=int(join_blend_steps),
    )
    params = np.empty((int(parent_steps) + 1, 5), dtype=float)
    params[:, 0] = float(reset.params[0])
    params[:, 1] = float(reset.params[1])
    params[:, 2] = x[:, 1]
    params[:, 3] = 1.0 + x[:, 2]
    params[:, 4] = x[:, 3]
    radii = simple._radii_from_params(params, theta)
    return ParentTarget(
        parent_id=int(parent_id),
        source_shot=reset.shot,
        split=reset.split,
        reset_source_index=int(reset.source_index),
        reset_time_s=float(reset.time_s),
        parent_steps=int(parent_steps),
        waypoint_count=len(knots),
        time_s=reset.time_s + 0.001 * np.arange(int(parent_steps) + 1, dtype=float),
        x=x,
        params=params,
        radii=radii,
        coils=coils,
        state_distance_p95=float("nan"),
    )


def _choose_move_sample(
    *,
    current_x: np.ndarray,
    current_coils: np.ndarray,
    points: list[ResetPoint],
    point_features: np.ndarray,
    move_samples: list[MoveSample],
    rng: np.random.Generator,
    limits: simple.Limits,
    distance_min: float,
    distance_max: float,
) -> MoveSample:
    # Pick a real 100-step move primitive, but apply it from the current
    # synthetic state. This keeps the segment slope in the real replay move
    # family without copying a long source trajectory. The nearest-state check
    # prevents walking off the safe replay manifold.
    del points
    order = rng.choice(np.arange(len(move_samples)), size=min(512, len(move_samples)), replace=False)
    eligible: list[tuple[float, MoveSample]] = []
    fallback: list[tuple[float, MoveSample]] = []
    for idx in order:
        sample = move_samples[int(idx)]
        endpoint_x = current_x + sample.dx
        endpoint_coils = current_coils + sample.dcoils
        if not _x_inside_safe_bounds(endpoint_x):
            continue
        endpoint_feature = simple._state_features(endpoint_x, endpoint_coils, limits=limits)
        manifold_distance = float(np.sqrt(np.min(np.sum((point_features - endpoint_feature[None, :]) ** 2, axis=1))))
        move_distance = float(np.linalg.norm(simple._move_features(sample.dx, sample.dcoils, limits=limits)))
        if move_distance >= float(distance_min):
            fallback.append((manifold_distance, sample))
        if manifold_distance <= float(distance_max) and move_distance >= float(distance_min):
            eligible.append((manifold_distance, sample))
    if eligible:
        eligible.sort(key=lambda item: item[0])
        top = eligible[: min(64, len(eligible))]
        return top[int(rng.integers(0, len(top)))][1]
    if fallback:
        fallback.sort(key=lambda item: item[0])
        top = fallback[: min(64, len(fallback))]
        return top[int(rng.integers(0, len(top)))][1]
    return move_samples[int(rng.integers(0, len(move_samples)))]


def _choose_endpoint(
    *,
    current_x: np.ndarray,
    current_coils: np.ndarray,
    points: list[ResetPoint],
    point_features: np.ndarray,
    rng: np.random.Generator,
    limits: simple.Limits,
    distance_min: float,
    distance_max: float,
) -> ResetPoint:
    current = simple._state_features(current_x, current_coils, limits=limits)
    distances = np.sqrt(np.sum((point_features - current[None, :]) ** 2, axis=1))
    eligible = np.flatnonzero((distances >= float(distance_min)) & (distances <= float(distance_max)))
    if eligible.size == 0:
        order = np.argsort(distances)
        order = order[distances[order] > 1.0e-10]
        if order.size == 0:
            return points[int(rng.integers(0, len(points)))]
        eligible = order[: min(128, order.size)]
    return points[int(eligible[int(rng.integers(0, eligible.size))])]


def _parent_knots(
    parent_steps: int,
    *,
    rng: np.random.Generator,
    segment_min_steps: int,
    segment_max_steps: int,
) -> np.ndarray:
    interior: list[int] = []
    pos = 0
    while pos < int(parent_steps):
        seg = int(rng.integers(int(segment_min_steps), int(segment_max_steps) + 1))
        pos += seg
        if int(segment_min_steps) <= pos <= int(parent_steps) - int(segment_min_steps):
            interior.append(pos)
    return np.unique(np.asarray([0, *interior, int(parent_steps)], dtype=np.int64))


def _parent_ok(
    parent: ParentTarget,
    *,
    limits: simple.Limits,
    envelope: dict[str, np.ndarray | float],
    state_space: dict[str, object],
    state_distance_limit: float,
    current_usage_cap: float,
) -> tuple[bool, str, float]:
    x = parent.x
    coils = parent.coils
    radii = parent.radii
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(parent.params)) or not np.all(np.isfinite(radii)) or not np.all(np.isfinite(coils)):
        return False, "nonfinite", float("inf")
    if not _x_inside_safe_bounds(x):
        return False, "safe_state_bounds", float("inf")
    radii_lo = np.asarray(envelope["radii_lo"], dtype=float)
    radii_hi = np.asarray(envelope["radii_hi"], dtype=float)
    if np.nanmin(radii - radii_lo[None, :]) < -1.0e-8:
        return False, "radii_below_observed_envelope", float("inf")
    if np.nanmax(radii - radii_hi[None, :]) > 1.0e-8:
        return False, "radii_above_observed_envelope", float("inf")
    sol_usage = np.nanmax(np.abs(coils[:, :3]) / limits.sol_current)
    pfc_usage = np.nanmax(np.abs(coils[:, 3:]) / limits.pfc_current)
    usage = max(float(sol_usage), float(pfc_usage))
    if usage > 1.0 + 1.0e-9:
        return False, "hard_current_limit", float("inf")
    if usage > float(current_usage_cap) + 1.0e-9:
        return False, "safe_current_usage_cap", float("inf")
    jdot = np.diff(coils, axis=0) / 0.001
    if np.nanmax(np.abs(jdot[:, :3])) > limits.sol_deriv + 1.0e-6:
        return False, "sol_derivative_limit", float("inf")
    if np.nanmax(np.abs(jdot[:, 3:])) > limits.pfc_deriv + 1.0e-6:
        return False, "pfc_derivative_limit", float("inf")
    sampled = np.arange(0, parent.x.shape[0], 10, dtype=np.int64)
    if sampled[-1] != parent.x.shape[0] - 1:
        sampled = np.concatenate([sampled, np.asarray([parent.x.shape[0] - 1], dtype=np.int64)])
    features = np.asarray([simple._state_features(parent.x[i], parent.coils[i], limits=limits) for i in sampled], dtype=float)
    distances = simple._nearest_distance(features, state_space)
    p95 = float(np.percentile(distances, 95.0))
    if p95 > float(state_distance_limit):
        return False, "state_manifold_distance", p95
    return True, "ok", p95


def _cut_parents(
    parents: list[ParentTarget],
    *,
    window_steps: int,
    stride: int,
    move_space: dict[str, object],
    limits: simple.Limits,
    move_distance_limit: float,
    rejections: Counter,
) -> list[WindowTarget]:
    rows: list[WindowTarget] = []
    for parent in parents:
        for parent_step in range(0, int(parent.parent_steps) - int(window_steps) + 1, int(stride)):
            end = parent_step + int(window_steps)
            move = simple._move_features(parent.x[end] - parent.x[parent_step], parent.coils[end] - parent.coils[parent_step], limits=limits).reshape(1, -1)
            move_distance = float(simple._nearest_distance(move, move_space)[0])
            is_hold = (
                np.linalg.norm(parent.x[end] - parent.x[parent_step]) < 1.0e-9
                and np.linalg.norm(parent.coils[end] - parent.coils[parent_step]) < 1.0e-9
            )
            if not is_hold and move_distance > float(move_distance_limit):
                rejections["move_manifold_distance"] += 1
                continue
            sl = slice(parent_step, parent_step + int(window_steps) + 1)
            ip = parent.x[sl, 0].astype(np.float32)
            rows.append(
                WindowTarget(
                    parent=parent,
                    parent_step=int(parent_step),
                    source_index=int(parent_step),
                    time_s=float(parent.time_s[parent_step]),
                    ip_ref=ip,
                    params_ref=parent.params[sl].astype(np.float32),
                    radii_ref=parent.radii[sl].astype(np.float32),
                    coil_witness=parent.coils[sl].astype(np.float32),
                    difficulty_bin=simple._difficulty_bin(ip),
                    move_distance=0.0 if is_hold else move_distance,
                )
            )
    return rows


def _write_libraries(
    windows: list[WindowTarget],
    *,
    parents: list[ParentTarget],
    initial_states_out: Path,
    targets_out: Path,
    oracle_targets_out: Path,
    parents_out: Path,
    limits: simple.Limits,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
) -> None:
    for path in (initial_states_out, targets_out, oracle_targets_out, parents_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    n = len(windows)
    shot_id = np.asarray([str(900000 + int(w.parent.parent_id)) for w in windows], dtype="<U16")
    source_shot = np.asarray([w.parent.source_shot for w in windows], dtype="<U8")
    split = np.asarray([w.parent.split for w in windows], dtype="<U8")
    source_index = np.asarray([w.parent_step for w in windows], dtype=np.int64)
    source_source_index = np.asarray([w.parent.reset_source_index for w in windows], dtype=np.int64)
    time_s = np.asarray([w.time_s for w in windows], dtype=np.float64)
    parent_id = np.asarray([w.parent.parent_id for w in windows], dtype=np.int64)
    parent_step = np.asarray([w.parent_step for w in windows], dtype=np.int64)
    difficulty_bin = np.asarray([w.difficulty_bin for w in windows], dtype="<U16")
    mode = np.full((n,), "safe_synthetic_long_parent_cut", dtype="<U40")
    ip_ref = np.asarray([w.ip_ref for w in windows], dtype=np.float32)
    params_ref = np.asarray([w.params_ref for w in windows], dtype=np.float32)
    radii_ref = np.asarray([w.radii_ref for w in windows], dtype=np.float32)
    coil_witness = np.asarray([w.coil_witness for w in windows], dtype=np.float32)
    ip0 = ip_ref[:, 0].astype(np.float32)
    params0 = params_ref[:, 0, :].astype(np.float32)
    pfc0 = coil_witness[:, 0, 3:].astype(np.float32)
    sol0 = coil_witness[:, 0, :3].astype(np.float32)
    move_distance = np.asarray([w.move_distance for w in windows], dtype=np.float32)
    np.savez_compressed(
        initial_states_out,
        schema=np.asarray("t15_long_target_generated_trim50_plain_gpu1e6_initial_states_v2"),
        shot_id=shot_id,
        source_shot=source_shot,
        source_index=source_index,
        source_source_index=source_source_index,
        time_s=time_s,
        ip0=ip0,
        pfc0=pfc0,
        sol0=sol0,
        params0=params0,
        split=split,
        difficulty_bin=difficulty_bin,
        mode=mode,
        parent_id=parent_id,
        parent_step=parent_step,
        move_distance=move_distance,
    )
    action = simple._normalized_jdot_action(coil_witness, limits=limits).astype(np.float32)
    np.savez_compressed(
        targets_out,
        schema=np.asarray("t15_long_target_generated_trim50_plain_gpu1e6_targets_v2"),
        ip_ref=ip_ref,
        params_ref=params_ref,
        radii_ref=radii_ref,
        coil_witness=coil_witness,
        zone=np.full((n,), "safe_long_parent", dtype="<U24"),
        mode=mode,
        shot_id=shot_id,
        source_shot=source_shot,
        source_index=source_index,
        source_source_index=source_source_index,
        parent_id=parent_id,
        parent_step=parent_step,
        time_s=time_s,
        split=split,
        move_distance=move_distance,
        train_shots=np.asarray(train_shots, dtype="<U8"),
        holdout_shots=np.asarray(holdout_shots, dtype="<U8"),
    )
    np.savez_compressed(
        oracle_targets_out,
        schema=np.asarray("t15_replay_window_oracle_targets_v1"),
        shot_id=shot_id,
        source_shot=source_shot,
        split=split,
        source_index=source_index,
        source_source_index=source_source_index,
        time_s=time_s,
        difficulty_bin=difficulty_bin,
        mode=mode,
        parent_id=parent_id,
        parent_step=parent_step,
        move_distance=move_distance,
        ip0=ip0,
        pfc0=pfc0,
        sol0=sol0,
        params0=params0,
        ip_target=ip_ref,
        boundary_radii=radii_ref,
        real_jdot_action=action,
        oracle_ip_mean_error_a=np.zeros((n,), dtype=np.float32),
        oracle_ip_max_error_a=np.zeros((n,), dtype=np.float32),
        current_limits=np.asarray([limits.pfc_current] * 6 + [limits.sol_current] * 3, dtype=np.float32),
        derivative_limits=np.asarray([limits.pfc_deriv] * 6 + [limits.sol_deriv] * 3, dtype=np.float32),
        train_shots=np.asarray(train_shots, dtype="<U8"),
        holdout_shots=np.asarray(holdout_shots, dtype="<U8"),
    )
    parent_max_steps = max(p.parent_steps for p in parents)
    parent_x = np.full((len(parents), parent_max_steps + 1, 4), np.nan, dtype=np.float32)
    parent_params = np.full((len(parents), parent_max_steps + 1, 5), np.nan, dtype=np.float32)
    parent_radii = np.full((len(parents), parent_max_steps + 1, 32), np.nan, dtype=np.float32)
    parent_coils = np.full((len(parents), parent_max_steps + 1, 9), np.nan, dtype=np.float32)
    parent_time = np.full((len(parents), parent_max_steps + 1), np.nan, dtype=np.float64)
    for row, parent in enumerate(parents):
        count = parent.parent_steps + 1
        parent_x[row, :count] = parent.x.astype(np.float32)
        parent_params[row, :count] = parent.params.astype(np.float32)
        parent_radii[row, :count] = parent.radii.astype(np.float32)
        parent_coils[row, :count] = parent.coils.astype(np.float32)
        parent_time[row, :count] = parent.time_s.astype(np.float64)
    np.savez_compressed(
        parents_out,
        schema=np.asarray("t15_long_target_generated_trim50_plain_gpu1e6_parents_v2"),
        parent_id=np.asarray([p.parent_id for p in parents], dtype=np.int64),
        source_shot=np.asarray([p.source_shot for p in parents], dtype="<U8"),
        split=np.asarray([p.split for p in parents], dtype="<U8"),
        reset_source_index=np.asarray([p.reset_source_index for p in parents], dtype=np.int64),
        reset_time_s=np.asarray([p.reset_time_s for p in parents], dtype=np.float64),
        parent_steps=np.asarray([p.parent_steps for p in parents], dtype=np.int64),
        waypoint_count=np.asarray([p.waypoint_count for p in parents], dtype=np.int64),
        state_distance_p95=np.asarray([p.state_distance_p95 for p in parents], dtype=np.float32),
        time_s=parent_time,
        x=parent_x,
        params=parent_params,
        radii=parent_radii,
        coils=parent_coils,
    )


def _x_inside_safe_bounds(x: np.ndarray) -> bool:
    arr = np.asarray(x, dtype=float)
    return bool(
        np.nanmin(arr[..., 0]) >= SAFE_IP[0]
        and np.nanmax(arr[..., 0]) <= SAFE_IP[1]
        and np.nanmin(arr[..., 1]) >= SAFE_A0[0]
        and np.nanmax(arr[..., 1]) <= SAFE_A0[1]
        and np.nanmin(arr[..., 2]) >= SAFE_E[0]
        and np.nanmax(arr[..., 2]) <= SAFE_E[1]
        and np.nanmin(arr[..., 3]) >= SAFE_DELTA[0]
        and np.nanmax(arr[..., 3]) <= SAFE_DELTA[1]
    )


def _summary(
    *,
    args: argparse.Namespace,
    sources: list[SourceShot],
    safe_points: list[ResetPoint],
    parents: list[ParentTarget],
    windows: list[WindowTarget],
    parent_rejections: Counter,
    window_rejections: Counter,
    envelope: dict[str, np.ndarray | float],
    limits: simple.Limits,
) -> dict[str, object]:
    return {
        "schema": "t15_long_target_generated_trim50_plain_gpu1e6_summary_v2",
        "boundary_param_dir": str(args.boundary_param_dir),
        "data_root": str(args.data_root),
        "machine_config": str(args.machine_config),
        "initial_states": str(args.initial_states_out),
        "targets": str(args.targets_out),
        "oracle_targets": str(args.oracle_targets_out),
        "parents": str(args.parents_out),
        "source_shots": {s.shot: {"split": s.split, "rows": int(s.x.shape[0])} for s in sources},
        "safe_reset_rows_by_shot": dict(sorted(Counter(p.shot for p in safe_points).items(), key=lambda kv: int(kv[0]))),
        "parent_count": len(parents),
        "parent_steps": {
            "min": int(min(p.parent_steps for p in parents)),
            "max": int(max(p.parent_steps for p in parents)),
            "mean": float(np.mean([p.parent_steps for p in parents])),
        },
        "window_count": len(windows),
        "window_steps": int(args.window_steps),
        "parent_stride": int(args.parent_stride),
        "windows_by_split": dict(sorted(Counter(w.parent.split for w in windows).items())),
        "windows_by_source_shot": dict(sorted(Counter(w.parent.source_shot for w in windows).items(), key=lambda kv: int(kv[0]))),
        "difficulty_bins": dict(sorted(Counter(w.difficulty_bin for w in windows).items())),
        "parent_rejections": dict(sorted(parent_rejections.items())),
        "window_rejections": dict(sorted(window_rejections.items())),
        "moving_windows": int(sum(1 for w in windows if w.move_distance > 1.0e-8)),
        "nonflat_ip_windows": int(sum(1 for w in windows if w.difficulty_bin != "flat")),
        "state_distance_p95_max": float(max(p.state_distance_p95 for p in parents)),
        "move_distance_p95": float(np.percentile([w.move_distance for w in windows], 95.0)),
        "safe_state_bounds": {
            "Ip": SAFE_IP,
            "A0": SAFE_A0,
            "e": SAFE_E,
            "delta": SAFE_DELTA,
        },
        "state_distance_limit": float(args.state_distance_limit),
        "move_distance_limit": float(args.move_distance_limit),
        "current_usage_cap": float(args.current_usage_cap),
        "radii_envelope_margin_m": 0.05,
        "real_current_usage_max": float(envelope["real_current_usage_max"]),
        "limits": {
            "pfc_current": limits.pfc_current,
            "sol_current": limits.sol_current,
            "pfc_deriv": limits.pfc_deriv,
            "sol_deriv": limits.sol_deriv,
        },
        "note": (
            "Synthetic parents no longer reuse long real source segments. Each parent starts from a real safe reset row, "
            "then moves through piecewise-linear waypoints generated by applying real-sized 100-step replay move primitives "
            "from the safe state space. Dense 100-step cuts are retained only when their endpoint move is near a real "
            "100-step replay move."
        ),
    }


def _write_report(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Long Target Generated Dataset",
        "",
        f"- Parents: {summary['parent_count']}",
        f"- Windows: {summary['window_count']}",
        f"- Parent steps: {summary['parent_steps']}",
        f"- Safe reset rows by shot: {summary['safe_reset_rows_by_shot']}",
        f"- Windows by split: {summary['windows_by_split']}",
        f"- Windows by source shot: {summary['windows_by_source_shot']}",
        f"- Difficulty bins: {summary['difficulty_bins']}",
        f"- Parent rejections: {summary['parent_rejections']}",
        f"- Window rejections: {summary['window_rejections']}",
        f"- Max parent state-distance p95: {summary['state_distance_p95_max']:.6g}",
        f"- Window move-distance p95: {summary['move_distance_p95']:.6g}",
        "",
        str(summary["note"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plots(
    *,
    parents: list[ParentTarget],
    windows: list[WindowTarget],
    sources: list[SourceShot],
    out_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    subset = parents[: min(8, len(parents))]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    labels = ("Ip [A]", "A0 [m]", "e", "delta")
    for parent in subset:
        t = np.arange(parent.x.shape[0], dtype=float) * 0.001
        axes[0].plot(t, parent.x[:, 0], linewidth=1.0, alpha=0.75, label=f"p{parent.parent_id} reset {parent.source_shot}")
        axes[1].plot(t, parent.x[:, 1], linewidth=1.0, alpha=0.75)
        axes[2].plot(t, parent.x[:, 2], linewidth=1.0, alpha=0.75)
        axes[3].plot(t, parent.x[:, 3], linewidth=1.0, alpha=0.75)
    for axis, label in zip(axes, labels, strict=True):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(plot_dir / "sample_long_parent_ip_boundary_params.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for parent in subset:
        t = np.arange(parent.coils.shape[0] - 1, dtype=float) * 0.001
        jdot = np.diff(parent.coils, axis=0) / 0.001
        axes[0].plot(t, jdot[:, 3:], linewidth=0.8, alpha=0.55)
        axes[1].plot(t, jdot[:, :3], linewidth=0.8, alpha=0.55)
    axes[0].set_ylabel("PFC Jdot [A/s]")
    axes[1].set_ylabel("SOL Jdot [A/s]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "sample_long_parent_coil_witness_jdot.png", dpi=150)
    plt.close(fig)

    real_x = np.concatenate([s.x for s in sources], axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(real_x[:, 0] / 1000.0, real_x[:, 1], s=4, alpha=0.15, label="source states")
    end_x = np.asarray([p.x[-1] for p in parents], dtype=float)
    ax.scatter(end_x[:, 0] / 1000.0, end_x[:, 1], s=18, alpha=0.8, label="generated parent endpoints")
    ax.set_xlabel("Ip [kA]")
    ax.set_ylabel("A0 [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "parent_endpoint_ip_a0_overlay.png", dpi=150)
    plt.close(fig)

    rng = np.random.default_rng(123)
    chosen = [windows[int(i)] for i in rng.choice(np.arange(len(windows)), size=min(80, len(windows)), replace=False)]
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(101)
    for row in chosen:
        ax.plot(t, row.ip_ref, linewidth=0.8, alpha=0.25)
    ax.set_xlabel("window step")
    ax.set_ylabel("Ip target [A]")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "sample_cut_window_ip_targets.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
