#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_BOUNDARY_PARAM_DIR = Path("../tokamak-sim/output/t15_boundary_parameters_trim50_idealized_matched_gpu_plain_1e6")
DEFAULT_DATA_ROOT = Path("../tokamak-sim/data/t15_data_new_trim50_idealized_matched")
DEFAULT_OUT_DIR = Path("data/processed/t15_feasible_generated_trim50_idealized_0p1s")
DEFAULT_INITIAL_STATES_OUT = Path("data/processed/t15_feasible_generated_trim50_idealized_0p1s_initial_states.npz")
DEFAULT_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_feasible_generated_trim50_idealized_0p1s_targets.npz"
DEFAULT_TRAIN_SHOTS = ("3856", "3857", "3858", "3863")
DEFAULT_HOLDOUT_SHOTS = ("3864",)
PARAM_COLUMNS = ("R0", "Z0", "A0", "kappa", "delta")
X_NAMES = ("Ip", "A0", "e", "delta")

CORE_BOUNDS = {
    "Ip": (140285.0, 426401.0),
    "A0": (0.434, 0.659),
    "e": (0.134, 0.296),
    "delta": (0.070, 0.176),
}
MODERATE_BOUNDS = {
    "Ip": (125000.0, 470000.0),
    "A0": (0.420, 0.680),
    "e": (0.080, 0.360),
    "delta": (0.040, 0.230),
}
AMBITIOUS_BOUNDS = {
    "Ip": (115000.0, 510000.0),
    "A0": (0.400, 0.700),
    "e": (0.000, 0.470),
    "delta": (0.000, 0.280),
}
CORE_ENDPOINT_CAP = np.asarray([75000.0, 0.040, 0.060, 0.025], dtype=float)
REPLAY_P99_SCALE = np.asarray([70572.0, 0.0323, 0.0522, 0.0201], dtype=float)
NEAREST_DISTANCE_LIMIT = {"core": 0.6, "moderate": 1.25, "ambitious": 2.0}
BEND_PROBABILITY = np.asarray([0.005, 0.05, 0.09, 0.09], dtype=float)
BEND_LEG_CAP = np.asarray([3500.0, 0.018, 0.021, 0.022], dtype=float)
ZONE_WEIGHTS = {"core": 0.50, "moderate": 0.35, "ambitious": 0.15}


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    shot: str
    start_row: int
    source_index: int
    time_s: float
    split: str
    x: np.ndarray
    params: np.ndarray
    pfc0: np.ndarray
    sol0: np.ndarray


@dataclass(frozen=True, slots=True)
class Candidate:
    window: ReplayWindow
    zone: str
    ip_ref: np.ndarray
    params_ref: np.ndarray
    radii_ref: np.ndarray
    distance: float
    controlled_axis: str
    sign_pattern: tuple[int, int, int, int]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build coupled feasible generated T15 target windows.")
    parser.add_argument("--boundary-param-dir", type=Path, default=DEFAULT_BOUNDARY_PARAM_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-states-out", type=Path, default=DEFAULT_INITIAL_STATES_OUT)
    parser.add_argument("--targets-out", type=Path, default=DEFAULT_TARGETS_OUT)
    parser.add_argument("--train-shots", nargs="+", default=list(DEFAULT_TRAIN_SHOTS))
    parser.add_argument("--holdout-shots", nargs="+", default=list(DEFAULT_HOLDOUT_SHOTS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--target-count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    steps = int(args.steps)
    if steps != 100:
        raise SystemExit("this builder intentionally targets 0.1 s / 100-step windows")
    rng = np.random.default_rng(int(args.seed))
    train_shots = tuple(str(int(v)) for v in args.train_shots)
    holdout_shots = tuple(str(int(v)) for v in args.holdout_shots)
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
        raise SystemExit("feasible generated builder requires non-empty train and holdout replay windows")

    real_deltas = np.asarray([w.x[-1] - w.x[0] for w in train_windows], dtype=float)
    observed_signs = {_sign_pattern(delta) for delta in real_deltas}
    theta = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    real_radii = np.concatenate([_radii_from_params(w.params, theta) for w in windows], axis=0)
    radii_min = np.nanmin(real_radii, axis=0) - 0.05
    radii_max = np.nanmax(real_radii, axis=0) + 0.05

    split_counts = _split_target_counts(args.target_count, train_windows=train_windows, holdout_windows=holdout_windows)
    candidates: list[Candidate] = []
    rejection = Counter()
    for split, count in split_counts.items():
        source_windows = train_windows if split == "train" else holdout_windows
        zone_counts = _zone_counts(count)
        for zone, zone_count in zone_counts.items():
            generated = _generate_zone(
                source_windows,
                zone=zone,
                count=zone_count,
                rng=rng,
                real_deltas=real_deltas,
                observed_signs=observed_signs,
                radii_min=radii_min,
                radii_max=radii_max,
                theta=theta,
                rejection=rejection,
            )
            candidates.extend(generated)

    if not candidates:
        raise SystemExit("no feasible generated candidates were accepted")
    _write_libraries(candidates, args.initial_states_out, args.targets_out, train_shots=train_shots, holdout_shots=holdout_shots)
    summary = _summary(candidates, rejection=rejection, windows=windows, train_shots=train_shots, holdout_shots=holdout_shots, args=args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "feasible_generated_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.plots:
        _write_plots(candidates, real_deltas=real_deltas, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


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
    train_set = set(train_shots)
    holdout_set = set(holdout_shots)
    for shot, rows in rows_by_shot.items():
        split = "holdout" if shot in holdout_set else "train"
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
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(params)):
                continue
            nearest = int(np.argmin(np.abs(coil_t - float(times[0]))))
            values = coil_values[nearest]
            if values.shape[0] != 9:
                raise ValueError(f"expected 9 coil columns for shot {shot}, got {values.shape[0]}")
            windows.append(
                ReplayWindow(
                    shot=str(shot),
                    start_row=int(start),
                    source_index=int(nearest),
                    time_s=float(times[0]),
                    split=split,
                    x=x,
                    params=params,
                    pfc0=values[3:].astype(float),
                    sol0=values[:3].astype(float),
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


def _generate_zone(
    windows: list[ReplayWindow],
    *,
    zone: str,
    count: int,
    rng: np.random.Generator,
    real_deltas: np.ndarray,
    observed_signs: set[tuple[int, int, int, int]],
    radii_min: np.ndarray,
    radii_max: np.ndarray,
    theta: np.ndarray,
    rejection: Counter,
) -> list[Candidate]:
    accepted: list[Candidate] = []
    max_attempts = max(1000, int(count) * 200)
    attempts = 0
    while len(accepted) < int(count) and attempts < max_attempts:
        attempts += 1
        window = windows[int(rng.integers(0, len(windows)))]
        target_x, controlled_axis = _candidate_x(window.x, zone=zone, rng=rng)
        params = window.params.copy()
        params[:, 2] = target_x[:, 1]
        params[:, 3] = 1.0 + target_x[:, 2]
        params[:, 4] = target_x[:, 3]
        radii = _radii_from_params(params, theta)
        ok, reason, distance, signs = _feasible(
            target_x,
            params=params,
            radii=radii,
            zone=zone,
            controlled_axis=controlled_axis,
            real_deltas=real_deltas,
            observed_signs=observed_signs,
            radii_min=radii_min,
            radii_max=radii_max,
        )
        if not ok:
            rejection[f"{zone}:{reason}"] += 1
            continue
        accepted.append(
            Candidate(
                window=window,
                zone=zone,
                ip_ref=target_x[:, 0].astype(np.float32),
                params_ref=params.astype(np.float32),
                radii_ref=radii.astype(np.float32),
                distance=float(distance),
                controlled_axis=controlled_axis,
                sign_pattern=signs,
            )
        )
    if len(accepted) < int(count):
        raise RuntimeError(f"accepted only {len(accepted)} / {count} feasible {zone} targets after {attempts} attempts")
    return accepted


def _candidate_x(base: np.ndarray, *, zone: str, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    out = np.asarray(base, dtype=float).copy()
    rel = out - out[0:1]
    controlled_axis = ""
    if zone == "core":
        return _apply_optional_bends(out, rng=rng, probability_scale=1.0), controlled_axis
    if zone == "moderate":
        controlled_axis = str(rng.choice(np.asarray(["A0", "e", "delta"], dtype=object)))
        col = X_NAMES.index(controlled_axis)
        out[:, col] += _smooth_drift(controlled_axis, out[0, col], rel[-1, col], MODERATE_BOUNDS[controlled_axis], rng, scale=0.75)
        return _apply_optional_bends(out, rng=rng, probability_scale=0.75), controlled_axis
    if zone == "ambitious":
        controlled_axis = str(rng.choice(np.asarray(["e", "delta"], dtype=object)))
        out[:, 0] = out[0, 0] + 0.55 * rel[:, 0]
        out[:, 1] = out[0, 1] + 0.55 * rel[:, 1]
        col = X_NAMES.index(controlled_axis)
        out[:, col] += _smooth_drift(controlled_axis, out[0, col], rel[-1, col], AMBITIOUS_BOUNDS[controlled_axis], rng, scale=1.25)
        return _apply_optional_bends(out, rng=rng, probability_scale=0.25), controlled_axis
    raise ValueError(f"unknown feasible generated zone: {zone}")


def _apply_optional_bends(x: np.ndarray, *, rng: np.random.Generator, probability_scale: float) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    steps = out.shape[0] - 1
    if steps < 4:
        return out
    for col in range(out.shape[1]):
        probability = float(BEND_PROBABILITY[col]) * float(probability_scale)
        if float(rng.random()) >= probability:
            continue
        peak = int(rng.integers(max(2, steps // 5), min(steps - 1, 4 * steps // 5) + 1))
        direction = -1.0 if float(rng.random()) < 0.5 else 1.0
        amplitude = direction * float(rng.uniform(0.25 * BEND_LEG_CAP[col], BEND_LEG_CAP[col]))
        bump = np.zeros((steps + 1,), dtype=float)
        bump[: peak + 1] = np.linspace(0.0, amplitude, peak + 1)
        bump[peak:] = np.linspace(amplitude, 0.0, steps - peak + 1)
        out[:, col] += bump
    return out


def _smooth_drift(axis: str, start: float, base_end_delta: float, bounds: tuple[float, float], rng: np.random.Generator, *, scale: float) -> np.ndarray:
    lo, hi = bounds
    direction_options = []
    if float(start) < float(hi):
        direction_options.append(1)
    if float(start) > float(lo):
        direction_options.append(-1)
    if not direction_options:
        return np.zeros((101,), dtype=float)
    direction = int(direction_options[int(rng.integers(0, len(direction_options)))])
    room = (float(hi) - float(start) - float(base_end_delta)) if direction > 0 else (float(start) + float(base_end_delta) - float(lo))
    axis_scale = float(REPLAY_P99_SCALE[X_NAMES.index(axis)])
    max_delta = max(0.0, min(abs(room), float(scale) * axis_scale))
    if max_delta <= 1.0e-8:
        return np.zeros((101,), dtype=float)
    endpoint = float(direction) * float(rng.uniform(0.15 * max_delta, max_delta))
    start_hold = int(rng.integers(0, 35))
    ramp_steps = 100 - start_hold
    drift = np.zeros((101,), dtype=float)
    if ramp_steps <= 0:
        return drift
    drift[start_hold:] = np.linspace(0.0, endpoint, ramp_steps + 1)
    return drift


def _feasible(
    x: np.ndarray,
    *,
    params: np.ndarray,
    radii: np.ndarray,
    zone: str,
    controlled_axis: str,
    real_deltas: np.ndarray,
    observed_signs: set[tuple[int, int, int, int]],
    radii_min: np.ndarray,
    radii_max: np.ndarray,
) -> tuple[bool, str, float, tuple[int, int, int, int]]:
    bounds = {"core": CORE_BOUNDS, "moderate": MODERATE_BOUNDS, "ambitious": AMBITIOUS_BOUNDS}[zone]
    for col, name in enumerate(X_NAMES):
        lo, hi = bounds[name]
        if np.nanmin(x[:, col]) < lo - 1.0e-8 or np.nanmax(x[:, col]) > hi + 1.0e-8:
            return False, f"{name}_bounds", float("inf"), (0, 0, 0, 0)
    delta = x[-1] - x[0]
    signs = _sign_pattern(delta)
    if signs not in observed_signs and not controlled_axis:
        return False, "unobserved_sign_pattern", float("inf"), signs
    if controlled_axis:
        changed = [name for name, sign in zip(X_NAMES, signs, strict=True) if sign != 0]
        non_replay_changed = [name for name in changed if name != controlled_axis and signs not in observed_signs]
        if non_replay_changed:
            return False, "unobserved_multi_axis_sign_pattern", float("inf"), signs
    distance = _nearest_distance(delta, real_deltas)
    if distance > NEAREST_DISTANCE_LIMIT[zone]:
        return False, "nearest_distance", distance, signs
    if zone == "core" and np.any(np.abs(delta) > CORE_ENDPOINT_CAP + 1.0e-9):
        return False, "core_endpoint_cap", distance, signs
    shape_ratios = np.abs(delta[1:]) / CORE_ENDPOINT_CAP[1:]
    if int(np.count_nonzero(shape_ratios > 0.70)) > 1:
        return False, "too_many_aggressive_shape_axes", distance, signs
    if int(np.count_nonzero(shape_ratios > 0.40)) > 2:
        return False, "too_many_medium_shape_axes", distance, signs
    if zone == "moderate":
        outside_core = [
            name
            for col, name in enumerate(("A0", "e", "delta"), start=1)
            if np.nanmin(x[:, col]) < CORE_BOUNDS[name][0] - 1.0e-8 or np.nanmax(x[:, col]) > CORE_BOUNDS[name][1] + 1.0e-8
        ]
        if len(outside_core) > 1:
            return False, "moderate_multi_shape_extrapolation", distance, signs
    if zone == "ambitious":
        outside_moderate_ed = [
            name
            for col, name in ((2, "e"), (3, "delta"))
            if np.nanmin(x[:, col]) < MODERATE_BOUNDS[name][0] - 1.0e-8 or np.nanmax(x[:, col]) > MODERATE_BOUNDS[name][1] + 1.0e-8
        ]
        if len(outside_moderate_ed) > 1:
            return False, "ambitious_multi_high_shape", distance, signs
        if abs(delta[0]) > 0.55 * CORE_ENDPOINT_CAP[0] or abs(delta[1]) > 0.55 * CORE_ENDPOINT_CAP[1]:
            return False, "ambitious_ip_a0_not_reduced", distance, signs
    if np.nanmin(radii - radii_min[None, :]) < -1.0e-8 or np.nanmax(radii - radii_max[None, :]) > 1.0e-8:
        return False, "radii_envelope", distance, signs
    if not np.all(np.isfinite(params)) or not np.all(np.isfinite(radii)):
        return False, "nonfinite", distance, signs
    return True, "ok", distance, signs


def _nearest_distance(delta: np.ndarray, real_deltas: np.ndarray) -> float:
    scaled = (np.asarray(delta, dtype=float).reshape(1, 4) - real_deltas) / REPLAY_P99_SCALE.reshape(1, 4)
    return float(np.min(np.linalg.norm(scaled, axis=1)))


def _sign_pattern(delta: np.ndarray) -> tuple[int, int, int, int]:
    eps = np.asarray([500.0, 5.0e-4, 5.0e-4, 5.0e-4], dtype=float)
    signs = []
    for value, threshold in zip(np.asarray(delta, dtype=float).reshape(4), eps, strict=True):
        signs.append(1 if value > threshold else (-1 if value < -threshold else 0))
    return tuple(signs)  # type: ignore[return-value]


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


def _zone_counts(count: int) -> dict[str, int]:
    core = int(round(float(count) * ZONE_WEIGHTS["core"]))
    moderate = int(round(float(count) * ZONE_WEIGHTS["moderate"]))
    ambitious = max(0, int(count) - core - moderate)
    return {"core": core, "moderate": moderate, "ambitious": ambitious}


def _write_libraries(
    candidates: list[Candidate],
    initial_states_out: Path,
    targets_out: Path,
    *,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
) -> None:
    initial_states_out.parent.mkdir(parents=True, exist_ok=True)
    targets_out.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(candidates)
    source_index = np.arange(row_count, dtype=np.int64)
    shot_id = np.asarray([c.window.shot for c in candidates], dtype="<U8")
    split = np.asarray([c.window.split for c in candidates], dtype="<U8")
    zone = np.asarray([c.zone for c in candidates], dtype="<U16")
    params0 = np.asarray([c.params_ref[0] for c in candidates], dtype=np.float32)
    np.savez_compressed(
        initial_states_out,
        schema=np.asarray("t15_feasible_generated_trim50_idealized_initial_states_v1"),
        shot_id=shot_id,
        source_index=source_index,
        time_s=np.asarray([c.window.time_s for c in candidates], dtype=np.float64),
        ip0=np.asarray([c.ip_ref[0] for c in candidates], dtype=np.float32),
        pfc0=np.asarray([c.window.pfc0 for c in candidates], dtype=np.float32),
        sol0=np.asarray([c.window.sol0 for c in candidates], dtype=np.float32),
        params0=params0,
        split=split,
        difficulty_bin=zone,
    )
    np.savez_compressed(
        targets_out,
        schema=np.asarray("t15_feasible_generated_trim50_idealized_targets_v1"),
        ip_ref=np.asarray([c.ip_ref for c in candidates], dtype=np.float32),
        params_ref=np.asarray([c.params_ref for c in candidates], dtype=np.float32),
        radii_ref=np.asarray([c.radii_ref for c in candidates], dtype=np.float32),
        zone=zone,
        shot_id=shot_id,
        source_index=source_index,
        replay_source_index=np.asarray([c.window.source_index for c in candidates], dtype=np.int64),
        replay_start_row=np.asarray([c.window.start_row for c in candidates], dtype=np.int64),
        time_s=np.asarray([c.window.time_s for c in candidates], dtype=np.float64),
        split=split,
        controlled_axis=np.asarray([c.controlled_axis for c in candidates], dtype="<U16"),
        nearest_distance=np.asarray([c.distance for c in candidates], dtype=np.float32),
        train_shots=np.asarray(train_shots, dtype="<U8"),
        holdout_shots=np.asarray(holdout_shots, dtype="<U8"),
    )


def _summary(candidates: list[Candidate], *, rejection: Counter, windows: list[ReplayWindow], train_shots: tuple[str, ...], holdout_shots: tuple[str, ...], args: argparse.Namespace) -> dict[str, object]:
    zone_counts = Counter(c.zone for c in candidates)
    split_counts = Counter(c.window.split for c in candidates)
    shot_counts = Counter(c.window.shot for c in candidates)
    return {
        "schema": "t15_feasible_generated_trim50_idealized_summary_v1",
        "boundary_param_dir": str(args.boundary_param_dir),
        "data_root": str(args.data_root),
        "initial_states": str(args.initial_states_out),
        "targets": str(args.targets_out),
        "train_shots": list(train_shots),
        "holdout_shots": list(holdout_shots),
        "source_replay_windows": len(windows),
        "accepted_targets": len(candidates),
        "accepted_by_zone": dict(sorted(zone_counts.items())),
        "accepted_by_split": dict(sorted(split_counts.items())),
        "accepted_by_shot": dict(sorted(shot_counts.items(), key=lambda kv: int(kv[0]))),
        "rejections": dict(sorted(rejection.items())),
        "nearest_distance_mean": float(np.mean([c.distance for c in candidates])),
        "nearest_distance_p95": float(np.percentile([c.distance for c in candidates], 95.0)),
    }


def _write_plots(candidates: list[Candidate], *, real_deltas: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    subset_idx = rng.choice(np.arange(len(candidates)), size=min(60, len(candidates)), replace=False)
    by_zone = {"core": [], "moderate": [], "ambitious": []}
    for idx in subset_idx.tolist():
        by_zone[candidates[idx].zone].append(candidates[idx])
    t = np.arange(candidates[0].ip_ref.shape[0])
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    labels = ("Ip [A]", "A0 [m]", "e", "delta")
    for zone, rows in by_zone.items():
        for c in rows[:20]:
            x = np.column_stack([c.ip_ref, c.params_ref[:, 2], c.params_ref[:, 3] - 1.0, c.params_ref[:, 4]])
            for axis, label in zip(axes, labels, strict=True):
                col = labels.index(label)
                axis.plot(t, x[:, col], alpha=0.35, linewidth=1.0, label=zone if c is rows[0] else None)
    for axis, label in zip(axes, labels, strict=True):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_feasible_trajectories.png", dpi=150)
    plt.close(fig)

    deltas = np.asarray([np.asarray([c.ip_ref[-1] - c.ip_ref[0], c.params_ref[-1, 2] - c.params_ref[0, 2], c.params_ref[-1, 3] - c.params_ref[0, 3], c.params_ref[-1, 4] - c.params_ref[0, 4]]) for c in candidates], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(real_deltas[:, 0] / REPLAY_P99_SCALE[0], real_deltas[:, 2] / REPLAY_P99_SCALE[2], s=6, alpha=0.25, label="real windows")
    ax.scatter(deltas[:, 0] / REPLAY_P99_SCALE[0], deltas[:, 2] / REPLAY_P99_SCALE[2], s=6, alpha=0.35, label="generated")
    ax.set_xlabel("normalized dIp")
    ax.set_ylabel("normalized de")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "endpoint_delta_scatter.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
