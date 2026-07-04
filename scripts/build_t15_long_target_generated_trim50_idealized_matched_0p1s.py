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
DEFAULT_MACHINE_CONFIG = Path("../tokamak-sim/runs/t15md_limited_replay_dataset_trim50_idealized_matched_gpu_plain_1e6/T15MD_new_data_legacy_contour_limited_replay.toml")
DEFAULT_OUT_DIR = Path("data/processed/t15_long_target_generated_trim50_plain_gpu1e6_oracle_window_0p1s")
DEFAULT_INITIAL_STATES_OUT = Path("data/processed/t15_long_target_generated_trim50_plain_gpu1e6_oracle_window_0p1s_initial_states.npz")
DEFAULT_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_long_target_generated_targets.npz"
DEFAULT_ORACLE_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_replay_window_oracle_targets.npz"
DEFAULT_PARENTS_OUT = DEFAULT_OUT_DIR / "t15_long_target_generated_parents.npz"
DEFAULT_TRAIN_SHOTS = ("3856", "3857", "3858", "3863")
DEFAULT_HOLDOUT_SHOTS = ("3864",)

PARAM_COLUMNS = simple.PARAM_COLUMNS
COIL_NAMES = simple.COIL_NAMES


@dataclass(frozen=True, slots=True)
class SourceShot:
    shot: str
    split: str
    time_s: np.ndarray
    x: np.ndarray
    params: np.ndarray
    coils: np.ndarray


@dataclass(frozen=True, slots=True)
class ParentTarget:
    parent_id: int
    source_shot: str
    split: str
    source_start: int
    parent_steps: int
    scale: np.ndarray
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build long synthetic Ip/boundary target shots, then cut them into "
            "dense 0.1 s replay-window oracle targets. This keeps the old-good "
            "training contract: reset library + t15_replay_window_oracle_targets.npz."
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
    parser.add_argument("--parent-count", type=int, default=8)
    parser.add_argument("--parent-min-steps", type=int, default=1000)
    parser.add_argument("--parent-max-steps", type=int, default=1200)
    parser.add_argument("--parent-stride", type=int, default=1)
    parser.add_argument("--join-blend-steps", type=int, default=12)
    parser.add_argument("--perturb-fraction", type=float, default=0.035)
    parser.add_argument("--scale-min", type=float, default=0.88)
    parser.add_argument("--scale-max", type=float, default=1.12)
    parser.add_argument("--state-distance-limit", type=float, default=0.18)
    parser.add_argument("--max-attempts-per-parent", type=int, default=300)
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
    if float(args.perturb_fraction) < 0.0:
        raise SystemExit("perturb_fraction must be non-negative")

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
    real_x = np.concatenate([s.x for s in sources], axis=0)
    real_params = np.concatenate([s.params for s in sources], axis=0)
    real_radii = np.concatenate([simple._radii_from_params(s.params, theta) for s in sources], axis=0)
    real_coils = np.concatenate([s.coils for s in sources], axis=0)
    state_space = _state_space_from_sources(sources, limits=limits)
    envelope = _build_envelope(real_x=real_x, real_params=real_params, real_radii=real_radii, real_coils=real_coils, limits=limits)

    split_counts = _split_parent_counts(int(args.parent_count), sources=sources)
    parents: list[ParentTarget] = []
    rejections = Counter()
    for split, count in split_counts.items():
        split_sources = [s for s in sources if s.split == split]
        parents.extend(
            _generate_parents(
                split_sources,
                count=count,
                first_parent_id=len(parents),
                rng=rng,
                window_steps=window_steps,
                parent_min_steps=int(args.parent_min_steps),
                parent_max_steps=int(args.parent_max_steps),
                join_blend_steps=int(args.join_blend_steps),
                perturb_fraction=float(args.perturb_fraction),
                scale_min=float(args.scale_min),
                scale_max=float(args.scale_max),
                theta=theta,
                limits=limits,
                envelope=envelope,
                state_space=state_space,
                state_distance_limit=float(args.state_distance_limit),
                max_attempts_per_parent=int(args.max_attempts_per_parent),
                rejections=rejections,
            )
        )

    windows = _cut_parents(parents, window_steps=window_steps, stride=int(args.parent_stride))
    if not windows:
        raise SystemExit("no generated long-parent windows were produced")
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
        parents=parents,
        windows=windows,
        rejections=rejections,
        envelope=envelope,
        limits=limits,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "long_target_generated_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(args.out_dir / "summary" / "long_target_generated_dataset_report.md", summary)
    if args.plots:
        _write_plots(parents=parents, windows=windows, real_x=real_x, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_sources(boundary_param_dir: Path, data_root: Path, *, train_shots: tuple[str, ...], holdout_shots: tuple[str, ...]) -> list[SourceShot]:
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
        finite = np.isfinite(time_s) & np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(params), axis=1) & np.all(np.isfinite(coils), axis=1)
        if not np.any(finite):
            continue
        sources.append(
            SourceShot(
                shot=str(shot),
                split="holdout" if shot in holdout else "train",
                time_s=time_s[finite],
                x=x[finite],
                params=params[finite],
                coils=coils[finite],
            )
        )
    missing = sorted(wanted - {s.shot for s in sources}, key=int)
    if missing:
        raise ValueError("missing usable source shots: " + ", ".join(missing))
    return sources


def _state_space_from_sources(sources: list[SourceShot], *, limits: simple.Limits) -> dict[str, object]:
    values = []
    for source in sources:
        for i in range(source.x.shape[0]):
            values.append(simple._state_features(source.x[i], source.coils[i], limits=limits))
    arr = np.asarray(values, dtype=float)
    tree = simple.cKDTree(arr) if simple.cKDTree is not None else None
    return {"values": arr, "tree": tree}


def _build_envelope(
    *,
    real_x: np.ndarray,
    real_params: np.ndarray,
    real_radii: np.ndarray,
    real_coils: np.ndarray,
    limits: simple.Limits,
) -> dict[str, np.ndarray | float]:
    x_lo = np.nanmin(real_x, axis=0)
    x_hi = np.nanmax(real_x, axis=0)
    x_span = np.maximum(x_hi - x_lo, np.asarray([1.0, 1.0e-3, 1.0e-3, 1.0e-3], dtype=float))
    params_lo = np.nanmin(real_params, axis=0)
    params_hi = np.nanmax(real_params, axis=0)
    radii_lo = np.nanmin(real_radii, axis=0) - 0.05
    radii_hi = np.nanmax(real_radii, axis=0) + 0.05
    sol_usage = np.nanmax(np.abs(real_coils[:, :3]) / float(limits.sol_current))
    pfc_usage = np.nanmax(np.abs(real_coils[:, 3:]) / float(limits.pfc_current))
    return {
        "x_lo": x_lo,
        "x_hi": x_hi,
        "x_span": x_span,
        "params_lo": params_lo,
        "params_hi": params_hi,
        "radii_lo": radii_lo,
        "radii_hi": radii_hi,
        "real_current_usage_max": float(max(sol_usage, pfc_usage)),
    }


def _split_parent_counts(parent_count: int, *, sources: list[SourceShot]) -> dict[str, int]:
    if int(parent_count) < 2:
        raise ValueError("parent_count must be at least 2 so train and holdout are both represented")
    train_points = sum(int(s.x.shape[0]) for s in sources if s.split == "train")
    holdout_points = sum(int(s.x.shape[0]) for s in sources if s.split == "holdout")
    total = max(train_points + holdout_points, 1)
    holdout = max(1, int(round(float(parent_count) * holdout_points / total)))
    train = max(1, int(parent_count) - holdout)
    return {"train": train, "holdout": holdout}


def _generate_parents(
    sources: list[SourceShot],
    *,
    count: int,
    first_parent_id: int,
    rng: np.random.Generator,
    window_steps: int,
    parent_min_steps: int,
    parent_max_steps: int,
    join_blend_steps: int,
    perturb_fraction: float,
    scale_min: float,
    scale_max: float,
    theta: np.ndarray,
    limits: simple.Limits,
    envelope: dict[str, np.ndarray | float],
    state_space: dict[str, object],
    state_distance_limit: float,
    max_attempts_per_parent: int,
    rejections: Counter,
) -> list[ParentTarget]:
    usable = [s for s in sources if int(s.x.shape[0]) > max(window_steps, min(parent_min_steps, parent_max_steps))]
    if not usable:
        raise RuntimeError("no source shots are long enough for requested parent length")
    accepted: list[ParentTarget] = []
    attempts = 0
    max_attempts = max(int(max_attempts_per_parent) * max(int(count), 1), 1)
    while len(accepted) < int(count) and attempts < max_attempts:
        attempts += 1
        source = usable[int(rng.integers(0, len(usable)))]
        max_available_steps = int(source.x.shape[0]) - 1
        high = min(int(parent_max_steps), max_available_steps)
        low = min(int(parent_min_steps), high)
        if high < int(window_steps):
            rejections["source_too_short"] += 1
            continue
        parent_steps = int(rng.integers(low, high + 1))
        start_max = max_available_steps - parent_steps
        if start_max < 0:
            rejections["source_start_unavailable"] += 1
            continue
        start = int(rng.integers(0, start_max + 1))
        parent = _make_parent(
            source,
            parent_id=first_parent_id + len(accepted),
            start=start,
            parent_steps=parent_steps,
            rng=rng,
            join_blend_steps=join_blend_steps,
            perturb_fraction=perturb_fraction,
            scale_min=scale_min,
            scale_max=scale_max,
            theta=theta,
        )
        ok, reason, state_distance = _parent_ok(
            parent,
            limits=limits,
            envelope=envelope,
            state_space=state_space,
            state_distance_limit=state_distance_limit,
        )
        if not ok:
            rejections[reason] += 1
            continue
        accepted.append(
            ParentTarget(
                parent_id=parent.parent_id,
                source_shot=parent.source_shot,
                split=parent.split,
                source_start=parent.source_start,
                parent_steps=parent.parent_steps,
                scale=parent.scale,
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
    source: SourceShot,
    *,
    parent_id: int,
    start: int,
    parent_steps: int,
    rng: np.random.Generator,
    join_blend_steps: int,
    perturb_fraction: float,
    scale_min: float,
    scale_max: float,
    theta: np.ndarray,
) -> ParentTarget:
    sl = slice(int(start), int(start) + int(parent_steps) + 1)
    base_x = np.asarray(source.x[sl], dtype=float)
    base_params = np.asarray(source.params[sl], dtype=float)
    base_coils = np.asarray(source.coils[sl], dtype=float)
    base_time = np.asarray(source.time_s[sl], dtype=float)
    knots = _parent_knots(int(parent_steps), rng=rng)
    scale = rng.uniform(float(scale_min), float(scale_max), size=(4,))
    x_values = []
    base_start = base_x[0]
    real_span = np.maximum(np.nanmax(source.x, axis=0) - np.nanmin(source.x, axis=0), np.asarray([1.0, 1.0e-3, 1.0e-3, 1.0e-3]))
    for idx, k in enumerate(knots):
        value = base_start + scale * (base_x[int(k)] - base_start)
        if idx > 0 and float(perturb_fraction) > 0.0:
            # Low-frequency offset, zero at the reset, so target[0] stays exactly
            # aligned with the real reset state.
            noise = rng.normal(0.0, float(perturb_fraction), size=(4,)) * real_span
            taper = np.sin(np.pi * float(k) / max(float(parent_steps), 1.0))
            value = value + taper * noise
        x_values.append(value)
    x_values[0] = base_start.copy()
    x = simple._piecewise_linear([int(k) for k in knots], [np.asarray(v, dtype=float) for v in x_values], steps=int(parent_steps), join_blend_steps=int(join_blend_steps))
    params = np.empty((int(parent_steps) + 1, 5), dtype=float)
    params[:, 0] = float(base_params[0, 0])
    params[:, 1] = float(base_params[0, 1])
    params[:, 2] = x[:, 1]
    params[:, 3] = 1.0 + x[:, 2]
    params[:, 4] = x[:, 3]
    radii = simple._radii_from_params(params, theta)
    return ParentTarget(
        parent_id=int(parent_id),
        source_shot=source.shot,
        split=source.split,
        source_start=int(start),
        parent_steps=int(parent_steps),
        scale=np.asarray(scale, dtype=float),
        time_s=base_time,
        x=x,
        params=params,
        radii=radii,
        coils=base_coils,
        state_distance_p95=float("nan"),
    )


def _parent_knots(parent_steps: int, *, rng: np.random.Generator) -> np.ndarray:
    interior: list[int] = []
    pos = 0
    while pos < int(parent_steps):
        seg = int(rng.integers(180, 321))
        pos += seg
        if 120 <= pos <= int(parent_steps) - 120:
            interior.append(pos)
    knots = np.asarray([0, *interior, int(parent_steps)], dtype=np.int64)
    return np.unique(knots)


def _parent_ok(
    parent: ParentTarget,
    *,
    limits: simple.Limits,
    envelope: dict[str, np.ndarray | float],
    state_space: dict[str, object],
    state_distance_limit: float,
) -> tuple[bool, str, float]:
    x = parent.x
    coils = parent.coils
    radii = parent.radii
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(parent.params)) or not np.all(np.isfinite(radii)) or not np.all(np.isfinite(coils)):
        return False, "nonfinite", float("inf")
    x_lo = np.asarray(envelope["x_lo"], dtype=float)
    x_hi = np.asarray(envelope["x_hi"], dtype=float)
    x_span = np.asarray(envelope["x_span"], dtype=float)
    padded_lo = x_lo - np.asarray([0.03, 0.02, 0.05, 0.05]) * x_span
    padded_hi = x_hi + np.asarray([0.03, 0.02, 0.05, 0.05]) * x_span
    if np.any(np.nanmin(x, axis=0) < padded_lo) or np.any(np.nanmax(x, axis=0) > padded_hi):
        return False, "x_outside_observed_envelope", float("inf")
    radii_lo = np.asarray(envelope["radii_lo"], dtype=float)
    radii_hi = np.asarray(envelope["radii_hi"], dtype=float)
    if np.nanmin(radii - radii_lo[None, :]) < -1.0e-8:
        return False, "radii_below_observed_envelope", float("inf")
    if np.nanmax(radii - radii_hi[None, :]) > 1.0e-8:
        return False, "radii_above_observed_envelope", float("inf")
    sol_usage = np.nanmax(np.abs(coils[:, :3]) / limits.sol_current)
    pfc_usage = np.nanmax(np.abs(coils[:, 3:]) / limits.pfc_current)
    if max(float(sol_usage), float(pfc_usage)) > 1.0 + 1.0e-9:
        return False, "current_limit", float("inf")
    jdot = np.diff(coils, axis=0) / 0.001
    if np.nanmax(np.abs(jdot[:, :3])) > limits.sol_deriv + 1.0e-6:
        return False, "sol_derivative_limit", float("inf")
    if np.nanmax(np.abs(jdot[:, 3:])) > limits.pfc_deriv + 1.0e-6:
        return False, "pfc_derivative_limit", float("inf")
    sampled = np.arange(0, parent.x.shape[0], 25, dtype=np.int64)
    if sampled[-1] != parent.x.shape[0] - 1:
        sampled = np.concatenate([sampled, np.asarray([parent.x.shape[0] - 1], dtype=np.int64)])
    features = np.asarray([simple._state_features(parent.x[i], parent.coils[i], limits=limits) for i in sampled], dtype=float)
    distances = simple._nearest_distance(features, state_space)
    p95 = float(np.percentile(distances, 95.0))
    if p95 > float(state_distance_limit):
        return False, "state_manifold_distance", p95
    return True, "ok", p95


def _cut_parents(parents: list[ParentTarget], *, window_steps: int, stride: int) -> list[WindowTarget]:
    rows: list[WindowTarget] = []
    for parent in parents:
        for parent_step in range(0, int(parent.parent_steps) - int(window_steps) + 1, int(stride)):
            sl = slice(parent_step, parent_step + int(window_steps) + 1)
            ip = parent.x[sl, 0].astype(np.float32)
            rows.append(
                WindowTarget(
                    parent=parent,
                    parent_step=int(parent_step),
                    source_index=int(parent.source_start + parent_step),
                    time_s=float(parent.time_s[parent_step]),
                    ip_ref=ip,
                    params_ref=parent.params[sl].astype(np.float32),
                    radii_ref=parent.radii[sl].astype(np.float32),
                    coil_witness=parent.coils[sl].astype(np.float32),
                    difficulty_bin=simple._difficulty_bin(ip),
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
    # Oracle replay-window references are indexed by (shot_id, source_index).
    # Long generated parents can overlap the same real source shot/index range,
    # so each generated parent gets its own numeric synthetic shot id while the
    # real source shot is kept as provenance metadata.
    shot_id = np.asarray([str(900000 + int(w.parent.parent_id)) for w in windows], dtype="<U16")
    source_shot = np.asarray([w.parent.source_shot for w in windows], dtype="<U8")
    split = np.asarray([w.parent.split for w in windows], dtype="<U8")
    source_index = np.asarray([w.parent_step for w in windows], dtype=np.int64)
    source_source_index = np.asarray([w.source_index for w in windows], dtype=np.int64)
    time_s = np.asarray([w.time_s for w in windows], dtype=np.float64)
    parent_id = np.asarray([w.parent.parent_id for w in windows], dtype=np.int64)
    parent_step = np.asarray([w.parent_step for w in windows], dtype=np.int64)
    difficulty_bin = np.asarray([w.difficulty_bin for w in windows], dtype="<U16")
    mode = np.full((n,), "long_parent_cut", dtype="<U32")
    ip_ref = np.asarray([w.ip_ref for w in windows], dtype=np.float32)
    params_ref = np.asarray([w.params_ref for w in windows], dtype=np.float32)
    radii_ref = np.asarray([w.radii_ref for w in windows], dtype=np.float32)
    coil_witness = np.asarray([w.coil_witness for w in windows], dtype=np.float32)
    ip0 = ip_ref[:, 0].astype(np.float32)
    params0 = params_ref[:, 0, :].astype(np.float32)
    pfc0 = coil_witness[:, 0, 3:].astype(np.float32)
    sol0 = coil_witness[:, 0, :3].astype(np.float32)
    np.savez_compressed(
        initial_states_out,
        schema=np.asarray("t15_long_target_generated_trim50_plain_gpu1e6_initial_states_v1"),
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
    )
    action = simple._normalized_jdot_action(coil_witness, limits=limits).astype(np.float32)
    np.savez_compressed(
        targets_out,
        schema=np.asarray("t15_long_target_generated_trim50_plain_gpu1e6_targets_v1"),
        ip_ref=ip_ref,
        params_ref=params_ref,
        radii_ref=radii_ref,
        coil_witness=coil_witness,
        zone=np.full((n,), "long_parent", dtype="<U16"),
        mode=mode,
        shot_id=shot_id,
        source_shot=source_shot,
        source_index=source_index,
        source_source_index=source_source_index,
        parent_id=parent_id,
        parent_step=parent_step,
        time_s=time_s,
        split=split,
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
        schema=np.asarray("t15_long_target_generated_trim50_plain_gpu1e6_parents_v1"),
        parent_id=np.asarray([p.parent_id for p in parents], dtype=np.int64),
        source_shot=np.asarray([p.source_shot for p in parents], dtype="<U8"),
        split=np.asarray([p.split for p in parents], dtype="<U8"),
        source_start=np.asarray([p.source_start for p in parents], dtype=np.int64),
        parent_steps=np.asarray([p.parent_steps for p in parents], dtype=np.int64),
        scale=np.asarray([p.scale for p in parents], dtype=np.float32),
        state_distance_p95=np.asarray([p.state_distance_p95 for p in parents], dtype=np.float32),
        time_s=parent_time,
        x=parent_x,
        params=parent_params,
        radii=parent_radii,
        coils=parent_coils,
    )


def _summary(
    *,
    args: argparse.Namespace,
    sources: list[SourceShot],
    parents: list[ParentTarget],
    windows: list[WindowTarget],
    rejections: Counter,
    envelope: dict[str, np.ndarray | float],
    limits: simple.Limits,
) -> dict[str, object]:
    return {
        "schema": "t15_long_target_generated_trim50_plain_gpu1e6_summary_v1",
        "boundary_param_dir": str(args.boundary_param_dir),
        "data_root": str(args.data_root),
        "machine_config": str(args.machine_config),
        "initial_states": str(args.initial_states_out),
        "targets": str(args.targets_out),
        "oracle_targets": str(args.oracle_targets_out),
        "parents": str(args.parents_out),
        "source_shots": {s.shot: {"split": s.split, "rows": int(s.x.shape[0])} for s in sources},
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
        "windows_by_shot": dict(sorted(Counter(w.parent.source_shot for w in windows).items(), key=lambda kv: int(kv[0]))),
        "difficulty_bins": dict(sorted(Counter(w.difficulty_bin for w in windows).items())),
        "parent_rejections": dict(sorted(rejections.items())),
        "state_distance_p95_max": float(max(p.state_distance_p95 for p in parents)),
        "perturb_fraction": float(args.perturb_fraction),
        "scale_min": float(args.scale_min),
        "scale_max": float(args.scale_max),
        "radii_envelope_margin_m": 0.05,
        "real_current_usage_max": float(envelope["real_current_usage_max"]),
        "limits": {
            "pfc_current": limits.pfc_current,
            "sol_current": limits.sol_current,
            "pfc_deriv": limits.pfc_deriv,
            "sol_deriv": limits.sol_deriv,
        },
        "note": (
            "Targets are generated as long Ip/A0/e/delta parents anchored at real reset states, "
            "with low-frequency perturbations of idealized real-shot trends, then densely cut into "
            "overlapping 100-step replay-window targets."
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
        f"- Windows by split: {summary['windows_by_split']}",
        f"- Windows by shot: {summary['windows_by_shot']}",
        f"- Difficulty bins: {summary['difficulty_bins']}",
        f"- Parent rejections: {summary['parent_rejections']}",
        f"- Max parent state-distance p95: {summary['state_distance_p95_max']:.6g}",
        "",
        str(summary["note"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plots(*, parents: list[ParentTarget], windows: list[WindowTarget], real_x: np.ndarray, out_dir: Path) -> None:
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
        axes[0].plot(t, parent.x[:, 0], linewidth=1.0, alpha=0.75, label=f"p{parent.parent_id} shot {parent.source_shot}")
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
        t = np.arange(parent.coils.shape[0], dtype=float) * 0.001
        jdot = np.diff(parent.coils, axis=0) / 0.001
        axes[0].plot(t[:-1], jdot[:, 3:], linewidth=0.8, alpha=0.55)
        axes[1].plot(t[:-1], jdot[:, :3], linewidth=0.8, alpha=0.55)
    axes[0].set_ylabel("PFC Jdot [A/s]")
    axes[1].set_ylabel("SOL Jdot [A/s]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "sample_long_parent_coil_witness_jdot.png", dpi=150)
    plt.close(fig)

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
