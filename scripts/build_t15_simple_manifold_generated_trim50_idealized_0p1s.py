#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback for minimal environments
    cKDTree = None


DEFAULT_BOUNDARY_PARAM_DIR = Path("../tokamak-sim/output/t15_boundary_parameters_trim50_idealized_matched_gpu_plain_1e6")
DEFAULT_DATA_ROOT = Path("../tokamak-sim/data/t15_data_new_trim50_idealized_matched")
DEFAULT_MACHINE_CONFIG = Path("../tokamak-sim/runs/t15md_limited_replay_dataset_trim50_idealized_matched_gpu_plain_1e6/T15MD_new_data_legacy_contour_limited_replay.toml")
DEFAULT_OUT_DIR = Path("data/processed/t15_simple_manifold_generated_trim50_idealized_matched_0p1s")
DEFAULT_INITIAL_STATES_OUT = Path("data/processed/t15_simple_manifold_generated_trim50_idealized_matched_0p1s_initial_states.npz")
DEFAULT_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_feasible_generated_trim50_idealized_0p1s_targets.npz"
DEFAULT_ORACLE_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_replay_window_oracle_targets.npz"
DEFAULT_TRAIN_SHOTS = ("3856", "3857", "3858", "3863")
DEFAULT_HOLDOUT_SHOTS = ("3864",)

PARAM_COLUMNS = ("R0", "Z0", "A0", "kappa", "delta")
X_NAMES = ("Ip", "A0", "e", "delta")
COIL_NAMES = ("SOL0", "SOL1", "SOL2", "PFC0", "PFC1", "PFC2", "PFC3", "PFC4", "PFC5")
MODES = ("hold", "ramp", "hold_then_ramp", "ramp_then_hold", "ramp_then_ramp")


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    shot: str
    start_row: int
    source_index: int
    time_s: float
    split: str
    x: np.ndarray
    params: np.ndarray
    coils: np.ndarray


@dataclass(frozen=True, slots=True)
class Candidate:
    window: ReplayWindow
    mode: str
    ip_ref: np.ndarray
    params_ref: np.ndarray
    radii_ref: np.ndarray
    coil_witness: np.ndarray
    state_distance_max: float
    move_distance: float


@dataclass(frozen=True, slots=True)
class Limits:
    pfc_current: float
    sol_current: float
    pfc_deriv: float
    sol_deriv: float


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build simple, jointly feasible 0.1 s T15 target windows. "
            "Targets use only holds and linear ramps through replay-manifold states; "
            "internal joins may be locally blended, but episode edges are never eased."
        )
    )
    parser.add_argument("--boundary-param-dir", type=Path, default=DEFAULT_BOUNDARY_PARAM_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--machine-config", type=Path, default=DEFAULT_MACHINE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-states-out", type=Path, default=DEFAULT_INITIAL_STATES_OUT)
    parser.add_argument("--targets-out", type=Path, default=DEFAULT_TARGETS_OUT)
    parser.add_argument(
        "--oracle-targets-out",
        type=Path,
        default=DEFAULT_ORACLE_TARGETS_OUT,
        help=(
            "Optional replay-window/oracle target NPZ. This writes the exact "
            "t15_replay_window_oracle_targets.npz schema consumed by the "
            "successful replay-window training path."
        ),
    )
    parser.add_argument("--train-shots", nargs="+", default=list(DEFAULT_TRAIN_SHOTS))
    parser.add_argument("--holdout-shots", nargs="+", default=list(DEFAULT_HOLDOUT_SHOTS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--target-count", type=int, default=12000)
    parser.add_argument("--min-segment-steps", type=int, default=30)
    parser.add_argument("--join-blend-steps", type=int, default=4)
    parser.add_argument("--current-usage-cap", type=float, default=0.75)
    parser.add_argument("--state-distance-limit", type=float, default=0.08)
    parser.add_argument("--move-distance-limit", type=float, default=0.08)
    parser.add_argument("--min-ramp-scale", type=float, default=0.35)
    parser.add_argument("--max-ramp-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    steps = int(args.steps)
    if steps != 100:
        raise SystemExit("this builder intentionally targets 0.1 s / 100-step windows")
    if int(args.min_segment_steps) * 2 > steps:
        raise SystemExit("min_segment_steps must allow at least two segments inside the episode")
    if float(args.min_ramp_scale) <= 0.0 or float(args.max_ramp_scale) > 1.0 or float(args.min_ramp_scale) > float(args.max_ramp_scale):
        raise SystemExit("ramp scales must satisfy 0 < min <= max <= 1")

    rng = np.random.default_rng(int(args.seed))
    train_shots = tuple(str(int(v)) for v in args.train_shots)
    holdout_shots = tuple(str(int(v)) for v in args.holdout_shots)
    limits = _load_limits(args.machine_config.resolve())
    windows = _load_replay_windows(
        args.boundary_param_dir.resolve(),
        args.data_root.resolve(),
        train_shots=train_shots,
        holdout_shots=holdout_shots,
        steps=steps,
    )
    train_windows = [w for w in windows if w.split == "train"]
    holdout_windows = [w for w in windows if w.split == "holdout"]
    if not train_windows or not holdout_windows:
        raise SystemExit("simple manifold builder requires non-empty train and holdout replay windows")

    theta = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    real_params = np.concatenate([w.params for w in windows], axis=0)
    real_x = np.concatenate([w.x for w in windows], axis=0)
    real_radii = np.concatenate([_radii_from_params(w.params, theta) for w in windows], axis=0)
    radii_min = np.nanmin(real_radii, axis=0) - 0.05
    radii_max = np.nanmax(real_radii, axis=0) + 0.05

    state_space = _state_space(windows=windows, limits=limits)
    move_space = _move_space(windows=windows, limits=limits)

    split_counts = _split_target_counts(args.target_count, train_windows=train_windows, holdout_windows=holdout_windows)
    candidates: list[Candidate] = []
    rejection = Counter()
    for split, count in split_counts.items():
        source = train_windows if split == "train" else holdout_windows
        candidates.extend(
            _generate_candidates(
                source,
                count=count,
                rng=rng,
                steps=steps,
                min_segment_steps=int(args.min_segment_steps),
                join_blend_steps=int(args.join_blend_steps),
                min_ramp_scale=float(args.min_ramp_scale),
                max_ramp_scale=float(args.max_ramp_scale),
                theta=theta,
                radii_min=radii_min,
                radii_max=radii_max,
                limits=limits,
                current_usage_cap=float(args.current_usage_cap),
                state_space=state_space,
                move_space=move_space,
                state_distance_limit=float(args.state_distance_limit),
                move_distance_limit=float(args.move_distance_limit),
                rejection=rejection,
            )
        )

    if not candidates:
        raise SystemExit("no simple manifold candidates were accepted")
    _write_libraries(
        candidates,
        args.initial_states_out,
        args.targets_out,
        args.oracle_targets_out,
        limits=limits,
        train_shots=train_shots,
        holdout_shots=holdout_shots,
    )
    summary = _summary(candidates, rejection=rejection, windows=windows, args=args, limits=limits)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "simple_manifold_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.plots:
        _write_plots(candidates, real_x=real_x, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_limits(path: Path) -> Limits:
    if not path.exists():
        raise FileNotFoundError(f"machine config does not exist: {path}")
    text = path.read_text(encoding="utf-8")

    def value(name: str) -> float:
        match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE)
        if not match:
            raise ValueError(f"{path} missing {name}")
        return float(match.group(1))

    return Limits(
        pfc_current=value("pfc_current_limit"),
        sol_current=value("sol_current_limit"),
        pfc_deriv=value("pfc_deriv_limit"),
        sol_deriv=value("sol_deriv_limit"),
    )


def _load_replay_windows(
    boundary_param_dir: Path,
    data_root: Path,
    *,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
    steps: int,
) -> list[ReplayWindow]:
    rows_by_shot = _load_boundary_rows(boundary_param_dir, wanted=set(train_shots) | set(holdout_shots))
    coils_by_shot = {shot: _load_coils(data_root / "coils" / f"t15md_{shot}_coils.csv") for shot in rows_by_shot}
    windows: list[ReplayWindow] = []
    holdout = set(holdout_shots)
    for shot, rows in rows_by_shot.items():
        split = "holdout" if shot in holdout else "train"
        coil_t, coil_values = coils_by_shot[shot]
        for start in range(0, max(len(rows) - int(steps), 0)):
            segment = rows[start : start + int(steps) + 1]
            times = np.asarray([r["t"] for r in segment], dtype=float)
            if not np.all(np.isfinite(times)):
                continue
            if np.max(np.diff(times)) > 0.003:
                continue
            x = np.asarray([[r["Ip"], r["A0"], r["kappa"] - 1.0, r["delta"]] for r in segment], dtype=float)
            params = np.asarray([[r[name] for name in PARAM_COLUMNS] for r in segment], dtype=float)
            nearest = np.asarray([int(np.argmin(np.abs(coil_t - float(t)))) for t in times], dtype=np.int64)
            coils = coil_values[nearest].astype(float)
            if coils.shape != (int(steps) + 1, 9):
                raise ValueError(f"expected coil witness shape {(int(steps) + 1, 9)} for shot {shot}, got {coils.shape}")
            if np.all(np.isfinite(x)) and np.all(np.isfinite(params)) and np.all(np.isfinite(coils)):
                windows.append(
                    ReplayWindow(
                        shot=str(shot),
                        start_row=int(start),
                        source_index=int(nearest[0]),
                        time_s=float(times[0]),
                        split=split,
                        x=x,
                        params=params,
                        coils=coils,
                    )
                )
    return windows


def _load_boundary_rows(root: Path, *, wanted: set[str]) -> dict[str, list[dict[str, float]]]:
    if not root.exists():
        raise FileNotFoundError(f"boundary parameter directory does not exist: {root}")
    rows: dict[str, list[dict[str, float]]] = {}
    for path in sorted(root.glob("*_boundary_params.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = sorted({"shot", "step", "t", "Ip", *PARAM_COLUMNS, "fit_status"} - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{path} missing columns: {', '.join(missing)}")
            for raw in reader:
                if raw.get("fit_status") != "ok":
                    continue
                shot = str(int(raw["shot"]))
                if shot not in wanted:
                    continue
                row = {"step": float(raw["step"]), "t": float(raw["t"]), "Ip": float(raw["Ip"])}
                for name in PARAM_COLUMNS:
                    row[name] = float(raw[name])
                if all(np.isfinite(v) for v in row.values()):
                    rows.setdefault(shot, []).append(row)
    for shot in rows:
        rows[shot].sort(key=lambda r: r["t"])
    missing = sorted(wanted - set(rows), key=int)
    if missing:
        raise ValueError("missing boundary parameter CSVs for shots: " + ", ".join(missing))
    return rows


def _load_coils(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing idealized coil CSV: {path}")
    times: list[float] = []
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for raw in reader:
            if not raw:
                continue
            values = [float(v) for v in raw]
            if len(values) != 10:
                raise ValueError(f"{path} expected time + 9 coil columns, got {len(values)}")
            times.append(values[0])
            rows.append(values[1:])
    return np.asarray(times, dtype=float), np.asarray(rows, dtype=float)


def _state_space(*, windows: list[ReplayWindow], limits: Limits) -> dict[str, object]:
    rows = []
    for w in windows:
        rows.append(_state_features(w.x[0], w.coils[0], limits=limits))
        rows.append(_state_features(w.x[-1], w.coils[-1], limits=limits))
        rows.append(_state_features(w.x[w.x.shape[0] // 2], w.coils[w.coils.shape[0] // 2], limits=limits))
    values = np.asarray(rows, dtype=float)
    tree = cKDTree(values) if cKDTree is not None else None
    return {"values": values, "tree": tree}


def _move_space(*, windows: list[ReplayWindow], limits: Limits) -> dict[str, object]:
    values = np.asarray([_move_features(w.x[-1] - w.x[0], w.coils[-1] - w.coils[0], limits=limits) for w in windows], dtype=float)
    tree = cKDTree(values) if cKDTree is not None else None
    return {"values": values, "tree": tree}


def _state_features(x: np.ndarray, coils: np.ndarray, *, limits: Limits) -> np.ndarray:
    return np.asarray(
        [
            x[0] / 426401.0,
            x[1] / 0.6585472,
            x[2] / 0.295784,
            x[3] / 0.1762824,
            coils[0] / limits.sol_current,
            coils[1] / limits.sol_current,
            coils[2] / limits.sol_current,
            coils[3] / limits.pfc_current,
            coils[4] / limits.pfc_current,
            coils[5] / limits.pfc_current,
            coils[6] / limits.pfc_current,
            coils[7] / limits.pfc_current,
            coils[8] / limits.pfc_current,
        ],
        dtype=float,
    )


def _move_features(dx: np.ndarray, dcoils: np.ndarray, *, limits: Limits) -> np.ndarray:
    return np.asarray(
        [
            dx[0] / 73750.0,
            dx[1] / 0.0394,
            dx[2] / 0.0587,
            dx[3] / 0.0233,
            dcoils[0] / (limits.sol_deriv * 0.1),
            dcoils[1] / (limits.sol_deriv * 0.1),
            dcoils[2] / (limits.sol_deriv * 0.1),
            dcoils[3] / (limits.pfc_deriv * 0.1),
            dcoils[4] / (limits.pfc_deriv * 0.1),
            dcoils[5] / (limits.pfc_deriv * 0.1),
            dcoils[6] / (limits.pfc_deriv * 0.1),
            dcoils[7] / (limits.pfc_deriv * 0.1),
            dcoils[8] / (limits.pfc_deriv * 0.1),
        ],
        dtype=float,
    )


def _nearest_distance(features: np.ndarray, space: dict[str, object]) -> np.ndarray:
    arr = np.asarray(features, dtype=float)
    tree = space.get("tree")
    if tree is not None:
        distances, _ = tree.query(arr, k=1)
        return np.asarray(distances, dtype=float)
    real = np.asarray(space["values"], dtype=float)
    diff = arr[:, None, :] - real[None, :, :]
    return np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=1))


def _generate_candidates(
    windows: list[ReplayWindow],
    *,
    count: int,
    rng: np.random.Generator,
    steps: int,
    min_segment_steps: int,
    join_blend_steps: int,
    min_ramp_scale: float,
    max_ramp_scale: float,
    theta: np.ndarray,
    radii_min: np.ndarray,
    radii_max: np.ndarray,
    limits: Limits,
    current_usage_cap: float,
    state_space: dict[str, object],
    move_space: dict[str, object],
    state_distance_limit: float,
    move_distance_limit: float,
    rejection: Counter,
) -> list[Candidate]:
    accepted: list[Candidate] = []
    max_attempts = max(2000, int(count) * 150)
    attempts = 0
    mode_cycle = np.resize(np.asarray(MODES, dtype=object), max(int(count), len(MODES)))
    rng.shuffle(mode_cycle)
    while len(accepted) < int(count) and attempts < max_attempts:
        attempts += 1
        mode = str(mode_cycle[len(accepted) % int(mode_cycle.size)])
        window = windows[int(rng.integers(0, len(windows)))]
        scale = 0.0 if mode == "hold" else float(rng.uniform(min_ramp_scale, max_ramp_scale))
        x, coils = _simple_target_from_window(
            window,
            mode=mode,
            scale=scale,
            rng=rng,
            steps=steps,
            min_segment_steps=min_segment_steps,
            join_blend_steps=join_blend_steps,
        )
        params = np.empty((steps + 1, 5), dtype=float)
        params[:, 0] = float(window.params[0, 0])
        params[:, 1] = float(window.params[0, 1])
        params[:, 2] = x[:, 1]
        params[:, 3] = 1.0 + x[:, 2]
        params[:, 4] = x[:, 3]
        radii = _radii_from_params(params, theta)
        ok, reason, state_distance, move_distance = _candidate_ok(
            x=x,
            coils=coils,
            params=params,
            radii=radii,
            radii_min=radii_min,
            radii_max=radii_max,
            limits=limits,
            current_usage_cap=current_usage_cap,
            state_space=state_space,
            move_space=move_space,
            state_distance_limit=state_distance_limit,
            move_distance_limit=move_distance_limit,
            allow_zero_move=(mode == "hold"),
        )
        if not ok:
            rejection[f"{mode}:{reason}"] += 1
            continue
        accepted.append(
            Candidate(
                window=window,
                mode=mode,
                ip_ref=x[:, 0].astype(np.float32),
                params_ref=params.astype(np.float32),
                radii_ref=radii.astype(np.float32),
                coil_witness=coils.astype(np.float32),
                state_distance_max=float(state_distance),
                move_distance=float(move_distance),
            )
        )
    if len(accepted) < int(count):
        raise RuntimeError(f"accepted only {len(accepted)} / {count} simple manifold targets after {attempts} attempts")
    return accepted


def _simple_target_from_window(
    window: ReplayWindow,
    *,
    mode: str,
    scale: float,
    rng: np.random.Generator,
    steps: int,
    min_segment_steps: int,
    join_blend_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    start_x = window.x[0]
    start_j = window.coils[0]
    end_x = start_x + float(scale) * (window.x[-1] - start_x)
    end_j = start_j + float(scale) * (window.coils[-1] - start_j)
    if mode == "hold":
        way_t = [0, steps]
        way_x = [start_x, start_x]
        way_j = [start_j, start_j]
    elif mode == "ramp":
        way_t = [0, steps]
        way_x = [start_x, end_x]
        way_j = [start_j, end_j]
    elif mode == "hold_then_ramp":
        b = int(rng.integers(min_segment_steps, steps - min_segment_steps + 1))
        way_t = [0, b, steps]
        way_x = [start_x, start_x, end_x]
        way_j = [start_j, start_j, end_j]
    elif mode == "ramp_then_hold":
        b = int(rng.integers(min_segment_steps, steps - min_segment_steps + 1))
        way_t = [0, b, steps]
        way_x = [start_x, end_x, end_x]
        way_j = [start_j, end_j, end_j]
    elif mode == "ramp_then_ramp":
        b = int(rng.integers(min_segment_steps, steps - min_segment_steps + 1))
        mid_x = start_x + float(scale) * (window.x[b] - start_x)
        mid_j = start_j + float(scale) * (window.coils[b] - start_j)
        way_t = [0, b, steps]
        way_x = [start_x, mid_x, end_x]
        way_j = [start_j, mid_j, end_j]
    else:
        raise ValueError(f"unknown simple target mode: {mode}")
    x = _piecewise_linear(way_t, way_x, steps=steps, join_blend_steps=join_blend_steps)
    coils = _piecewise_linear(way_t, way_j, steps=steps, join_blend_steps=join_blend_steps)
    return x, coils


def _piecewise_linear(way_t: list[int], way_v: list[np.ndarray], *, steps: int, join_blend_steps: int) -> np.ndarray:
    times = np.arange(steps + 1, dtype=float)
    values = np.asarray(way_v, dtype=float)
    out = np.empty((steps + 1, values.shape[1]), dtype=float)
    for i in range(len(way_t) - 1):
        a = int(way_t[i])
        b = int(way_t[i + 1])
        if b <= a:
            raise ValueError("waypoint times must be strictly increasing")
        local = (times[a : b + 1] - float(a)) / float(b - a)
        out[a : b + 1] = values[i][None, :] + local[:, None] * (values[i + 1] - values[i])[None, :]
    # Only internal joins are blended. The first and last sample are never modified,
    # so ramps do not acquire an artificial flat/eased episode edge.
    blend = max(0, int(join_blend_steps))
    if blend <= 0 or len(way_t) <= 2:
        return out
    raw = out.copy()
    for join_idx in range(1, len(way_t) - 1):
        b = int(way_t[join_idx])
        lo = max(1, b - blend)
        hi = min(steps - 1, b + blend)
        if lo >= hi:
            continue
        left_a = int(way_t[join_idx - 1])
        right_b = int(way_t[join_idx + 1])
        left_span = max(float(b - left_a), 1.0)
        right_span = max(float(right_b - b), 1.0)
        left_slope = (values[join_idx] - values[join_idx - 1]) / left_span
        right_slope = (values[join_idx + 1] - values[join_idx]) / right_span
        raw[lo : hi + 1] = _monotone_hermite(
            p0=out[lo],
            p1=out[hi],
            m0=left_slope,
            m1=right_slope,
            steps=int(hi - lo),
        )
    return raw


def _monotone_hermite(*, p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, steps: int) -> np.ndarray:
    """Interpolate between two samples without overshooting either endpoint."""
    if int(steps) <= 0:
        return np.asarray(p0, dtype=float).reshape(1, -1)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    m0 = np.asarray(m0, dtype=float).copy()
    m1 = np.asarray(m1, dtype=float).copy()
    h = float(steps)
    secant = (p1 - p0) / h
    eps = 1.0e-14
    for i in range(secant.shape[0]):
        d = float(secant[i])
        if abs(d) <= eps:
            m0[i] = 0.0
            m1[i] = 0.0
            continue
        if float(m0[i]) * d <= 0.0:
            m0[i] = 0.0
        if float(m1[i]) * d <= 0.0:
            m1[i] = 0.0
        alpha = float(m0[i]) / d
        beta = float(m1[i]) / d
        norm = alpha * alpha + beta * beta
        if norm > 9.0:
            tau = 3.0 / float(np.sqrt(norm))
            m0[i] = tau * alpha * d
            m1[i] = tau * beta * d
    u = np.linspace(0.0, 1.0, int(steps) + 1, dtype=float).reshape(-1, 1)
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    return h00 * p0[None, :] + h10 * (h * m0)[None, :] + h01 * p1[None, :] + h11 * (h * m1)[None, :]


def _candidate_ok(
    *,
    x: np.ndarray,
    coils: np.ndarray,
    params: np.ndarray,
    radii: np.ndarray,
    radii_min: np.ndarray,
    radii_max: np.ndarray,
    limits: Limits,
    current_usage_cap: float,
    state_space: dict[str, object],
    move_space: dict[str, object],
    state_distance_limit: float,
    move_distance_limit: float,
    allow_zero_move: bool,
) -> tuple[bool, str, float, float]:
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(coils)) or not np.all(np.isfinite(params)) or not np.all(np.isfinite(radii)):
        return False, "nonfinite", np.inf, np.inf
    if np.nanmin(x[:, 0]) < 140285.0 or np.nanmax(x[:, 0]) > 426402.0:
        return False, "Ip_bounds", np.inf, np.inf
    if np.nanmin(x[:, 1]) < 0.434 or np.nanmax(x[:, 1]) > 0.659:
        return False, "A0_bounds", np.inf, np.inf
    if np.nanmin(x[:, 2]) < 0.134 or np.nanmax(x[:, 2]) > 0.296:
        return False, "e_bounds", np.inf, np.inf
    if np.nanmin(x[:, 3]) < 0.070 or np.nanmax(x[:, 3]) > 0.177:
        return False, "delta_bounds", np.inf, np.inf
    if np.nanmin(radii - radii_min[None, :]) < -1.0e-8 or np.nanmax(radii - radii_max[None, :]) > 1.0e-8:
        return False, "radii_envelope", np.inf, np.inf

    sol_usage = np.max(np.abs(coils[:, :3]) / limits.sol_current)
    pfc_usage = np.max(np.abs(coils[:, 3:]) / limits.pfc_current)
    current_usage = max(float(sol_usage), float(pfc_usage))
    if current_usage > 1.0 + 1.0e-9:
        return False, "hard_current_limit", np.inf, np.inf
    if current_usage > float(current_usage_cap) + 1.0e-9:
        return False, "replay_current_usage_cap", np.inf, np.inf
    jdot = np.diff(coils, axis=0) / 0.001
    if np.nanmax(np.abs(jdot[:, :3])) > limits.sol_deriv + 1.0e-6:
        return False, "sol_derivative_limit", np.inf, np.inf
    if np.nanmax(np.abs(jdot[:, 3:])) > limits.pfc_deriv + 1.0e-6:
        return False, "pfc_derivative_limit", np.inf, np.inf

    state_features = np.asarray([_state_features(x[i], coils[i], limits=limits) for i in range(0, x.shape[0], 5)], dtype=float)
    state_distances = _nearest_distance(state_features, state_space)
    state_distance = float(np.nanmax(state_distances))
    if state_distance > float(state_distance_limit):
        return False, "state_manifold_distance", state_distance, np.inf

    if allow_zero_move and np.linalg.norm(x[-1] - x[0]) < 1.0e-12 and np.linalg.norm(coils[-1] - coils[0]) < 1.0e-12:
        move_distance = 0.0
    else:
        move = _move_features(x[-1] - x[0], coils[-1] - coils[0], limits=limits).reshape(1, -1)
        move_distance = float(_nearest_distance(move, move_space)[0])
    if not allow_zero_move and move_distance > float(move_distance_limit):
        return False, "move_manifold_distance", state_distance, move_distance
    return True, "ok", state_distance, move_distance


def _radii_from_params(params: np.ndarray, theta: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=float)
    R0 = p[:, 0:1]
    Z0 = p[:, 1:2]
    A0 = p[:, 2:3]
    kappa = p[:, 3:4]
    delta = p[:, 4:5]
    sin_t = np.sin(theta).reshape(1, -1)
    R = R0 + A0 * np.cos(theta).reshape(1, -1) - delta * A0 * sin_t**2
    Z = Z0 + A0 * kappa * sin_t
    return np.sqrt((R - R0) ** 2 + (Z - Z0) ** 2)


def _split_target_counts(target_count: int, *, train_windows: list[ReplayWindow], holdout_windows: list[ReplayWindow]) -> dict[str, int]:
    total = len(train_windows) + len(holdout_windows)
    holdout = max(1, int(round(float(target_count) * len(holdout_windows) / max(total, 1))))
    train = max(1, int(target_count) - holdout)
    return {"train": train, "holdout": holdout}


def _write_libraries(
    candidates: list[Candidate],
    initial_states_out: Path,
    targets_out: Path,
    oracle_targets_out: Path,
    *,
    limits: Limits,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
) -> None:
    initial_states_out.parent.mkdir(parents=True, exist_ok=True)
    targets_out.parent.mkdir(parents=True, exist_ok=True)
    oracle_targets_out.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(candidates)
    source_index = np.arange(row_count, dtype=np.int64)
    shot_id = np.asarray([c.window.shot for c in candidates], dtype="<U8")
    split = np.asarray([c.window.split for c in candidates], dtype="<U8")
    zone = np.full((row_count,), "core", dtype="<U16")
    difficulty_bin = np.asarray([_difficulty_bin(c.ip_ref) for c in candidates], dtype="<U16")
    mode = np.asarray([c.mode for c in candidates], dtype="<U32")
    params0 = np.asarray([c.params_ref[0] for c in candidates], dtype=np.float32)
    ip0 = np.asarray([c.ip_ref[0] for c in candidates], dtype=np.float32)
    pfc0 = np.asarray([c.coil_witness[0, 3:] for c in candidates], dtype=np.float32)
    sol0 = np.asarray([c.coil_witness[0, :3] for c in candidates], dtype=np.float32)
    time_s = np.asarray([c.window.time_s for c in candidates], dtype=np.float64)
    np.savez_compressed(
        initial_states_out,
        schema=np.asarray("t15_simple_manifold_generated_trim50_idealized_matched_initial_states_v1"),
        shot_id=shot_id,
        source_index=source_index,
        time_s=time_s,
        ip0=ip0,
        pfc0=pfc0,
        sol0=sol0,
        params0=params0,
        split=split,
        difficulty_bin=zone,
        mode=mode,
    )
    ip_ref = np.asarray([c.ip_ref for c in candidates], dtype=np.float32)
    radii_ref = np.asarray([c.radii_ref for c in candidates], dtype=np.float32)
    coil_witness = np.asarray([c.coil_witness for c in candidates], dtype=np.float32)
    np.savez_compressed(
        targets_out,
        schema=np.asarray("t15_simple_manifold_generated_trim50_idealized_matched_targets_v1"),
        ip_ref=ip_ref,
        params_ref=np.asarray([c.params_ref for c in candidates], dtype=np.float32),
        radii_ref=radii_ref,
        coil_witness=coil_witness,
        zone=zone,
        mode=mode,
        shot_id=shot_id,
        source_index=source_index,
        replay_source_index=np.asarray([c.window.source_index for c in candidates], dtype=np.int64),
        replay_start_row=np.asarray([c.window.start_row for c in candidates], dtype=np.int64),
        time_s=time_s,
        split=split,
        state_distance_max=np.asarray([c.state_distance_max for c in candidates], dtype=np.float32),
        move_distance=np.asarray([c.move_distance for c in candidates], dtype=np.float32),
        train_shots=np.asarray(train_shots, dtype="<U8"),
        holdout_shots=np.asarray(holdout_shots, dtype="<U8"),
    )
    np.savez_compressed(
        oracle_targets_out,
        schema=np.asarray("t15_replay_window_oracle_targets_v1"),
        shot_id=shot_id,
        split=split,
        source_index=source_index,
        time_s=time_s,
        difficulty_bin=difficulty_bin,
        mode=mode,
        ip0=ip0,
        pfc0=pfc0,
        sol0=sol0,
        params0=params0,
        ip_target=ip_ref,
        boundary_radii=radii_ref,
        real_jdot_action=_normalized_jdot_action(coil_witness, limits=limits).astype(np.float32),
        oracle_ip_mean_error_a=np.zeros((row_count,), dtype=np.float32),
        oracle_ip_max_error_a=np.zeros((row_count,), dtype=np.float32),
        current_limits=np.asarray([limits.pfc_current] * 6 + [limits.sol_current] * 3, dtype=np.float32),
        derivative_limits=np.asarray([limits.pfc_deriv] * 6 + [limits.sol_deriv] * 3, dtype=np.float32),
        train_shots=np.asarray(train_shots, dtype="<U8"),
        holdout_shots=np.asarray(holdout_shots, dtype="<U8"),
    )


def _difficulty_bin(ip_ref: np.ndarray) -> str:
    delta = float(np.asarray(ip_ref, dtype=float)[-1] - np.asarray(ip_ref, dtype=float)[0])
    mag = abs(delta)
    if mag < 10_000.0:
        return "flat"
    prefix = "fast" if mag >= 40_000.0 else "medium"
    suffix = "up" if delta > 0.0 else "down"
    return f"{prefix}_{suffix}"


def _normalized_jdot_action(coil_witness: np.ndarray, *, limits: Limits) -> np.ndarray:
    """Convert SOL/PFC coil witness currents to normalized PFC/SOL Jdot actions."""
    coils = np.asarray(coil_witness, dtype=float)
    if coils.ndim != 3 or coils.shape[2] != 9:
        raise ValueError(f"coil_witness must have shape [N, T+1, 9], got {coils.shape}")
    ordered = np.concatenate([coils[:, :, 3:], coils[:, :, :3]], axis=2)
    jdot = np.diff(ordered, axis=1) / 0.001
    denom = np.asarray([limits.pfc_deriv] * 6 + [limits.sol_deriv] * 3, dtype=float).reshape(1, 1, 9)
    action = jdot / denom
    max_abs = float(np.nanmax(np.abs(action))) if action.size else 0.0
    if not np.isfinite(max_abs):
        raise ValueError("non-finite normalized witness action")
    if max_abs > 1.0001:
        raise ValueError(f"normalized witness action exceeds derivative limits: max_abs={max_abs:.6g}")
    return np.clip(action, -1.0, 1.0)


def _summary(candidates: list[Candidate], *, rejection: Counter, windows: list[ReplayWindow], args: argparse.Namespace, limits: Limits) -> dict[str, object]:
    return {
        "schema": "t15_simple_manifold_generated_trim50_idealized_matched_summary_v1",
        "boundary_param_dir": str(args.boundary_param_dir),
        "data_root": str(args.data_root),
        "machine_config": str(args.machine_config),
        "initial_states": str(args.initial_states_out),
        "targets": str(args.targets_out),
        "oracle_targets": str(args.oracle_targets_out),
        "source_replay_windows": len(windows),
        "accepted_targets": len(candidates),
        "accepted_by_mode": dict(sorted(Counter(c.mode for c in candidates).items())),
        "accepted_by_split": dict(sorted(Counter(c.window.split for c in candidates).items())),
        "accepted_by_shot": dict(sorted(Counter(c.window.shot for c in candidates).items(), key=lambda kv: int(kv[0]))),
        "rejections": dict(sorted(rejection.items())),
        "state_distance_max_p95": float(np.percentile([c.state_distance_max for c in candidates], 95.0)),
        "move_distance_p95": float(np.percentile([c.move_distance for c in candidates], 95.0)),
        "current_usage_cap": float(args.current_usage_cap),
        "join_blend_steps": int(args.join_blend_steps),
        "note": "Targets are holds and linear ramps. Only internal joins are blended; episode edges are never eased.",
        "limits": {
            "pfc_current": limits.pfc_current,
            "sol_current": limits.sol_current,
            "pfc_deriv": limits.pfc_deriv,
            "sol_deriv": limits.sol_deriv,
        },
    }


def _write_plots(candidates: list[Candidate], *, real_x: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    subset = [candidates[int(i)] for i in rng.choice(np.arange(len(candidates)), size=min(80, len(candidates)), replace=False)]
    t = np.arange(candidates[0].ip_ref.shape[0])
    labels = ("Ip [A]", "A0 [m]", "e", "delta")
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for mode in MODES:
        rows = [c for c in subset if c.mode == mode]
        for idx, c in enumerate(rows[:16]):
            x = np.column_stack([c.ip_ref, c.params_ref[:, 2], c.params_ref[:, 3] - 1.0, c.params_ref[:, 4]])
            for col, axis in enumerate(axes):
                axis.plot(t, x[:, col], alpha=0.35, linewidth=1.0, label=mode if idx == 0 and col == 0 else None)
    for axis, label in zip(axes, labels, strict=True):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_simple_manifold_trajectories.png", dpi=150)
    plt.close(fig)

    generated = np.asarray(
        [[c.ip_ref[-1] - c.ip_ref[0], c.params_ref[-1, 2] - c.params_ref[0, 2], c.params_ref[-1, 3] - c.params_ref[0, 3], c.params_ref[-1, 4] - c.params_ref[0, 4]] for c in candidates],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(real_x[:, 0] / 1000.0, real_x[:, 1], s=4, alpha=0.15, label="real states")
    ax.scatter([c.ip_ref[-1] / 1000.0 for c in candidates], [c.params_ref[-1, 2] for c in candidates], s=4, alpha=0.25, label="generated endpoints")
    ax.set_xlabel("Ip [kA]")
    ax.set_ylabel("A0 [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "endpoint_ip_a0_overlay.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(generated[:, 0] / 1000.0, generated[:, 2], s=5, alpha=0.35)
    ax.set_xlabel("generated dIp over 0.1s [kA]")
    ax.set_ylabel("generated de over 0.1s")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "generated_endpoint_delta_scatter.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
