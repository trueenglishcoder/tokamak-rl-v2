#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = Path("../tokamak-sim/data/t15_data_new_trim50")
DEFAULT_MACHINE_CONFIG = Path("data/processed/t15_new_trim50_plain_gpu1e6_machine_config.toml")
DEFAULT_OBSERVED_TARGETS = Path(
    "data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/"
    "t15_replay_window_oracle_targets.npz"
)
DEFAULT_OUT_DIR = Path("data/processed/t15_actuator_generated_trim50_plain_gpu1e6_0p1s")
DEFAULT_INITIAL_STATES_OUT = Path("data/processed/t15_actuator_generated_trim50_plain_gpu1e6_0p1s_initial_states.npz")
DEFAULT_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_replay_window_oracle_targets.npz"
DEFAULT_DIAGNOSTIC_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_actuator_generated_targets.npz"
DEFAULT_TRAIN_SHOTS = ("3856", "3857", "3858", "3863")
DEFAULT_HOLDOUT_SHOTS = ("3864",)
COIL_NAMES = ("PFC0", "PFC1", "PFC2", "PFC3", "PFC4", "PFC5", "SOL0", "SOL1", "SOL2")


@dataclass(frozen=True, slots=True)
class Limits:
    pfc_current: float
    sol_current: float
    pfc_deriv: float
    sol_deriv: float

    @property
    def current_vector(self) -> np.ndarray:
        return np.asarray([self.pfc_current] * 6 + [self.sol_current] * 3, dtype=float)

    @property
    def derivative_vector(self) -> np.ndarray:
        return np.asarray([self.pfc_deriv] * 6 + [self.sol_deriv] * 3, dtype=float)


@dataclass(frozen=True, slots=True)
class RealWindow:
    shot_id: str
    split: str
    start: int
    time_s: float
    ip: np.ndarray
    currents: np.ndarray


@dataclass(frozen=True, slots=True)
class JdotStyle:
    average_vectors: np.ndarray


@dataclass(frozen=True, slots=True)
class CoilCandidate:
    reset: RealWindow
    motion: RealWindow
    mode: str
    scale: float
    gains: np.ndarray
    ip0: float
    currents: np.ndarray
    action: np.ndarray


@dataclass(frozen=True, slots=True)
class ObservedEnvelope:
    ip_min: float
    ip_max: float
    radii_min: np.ndarray
    radii_max: np.ndarray
    current_min: np.ndarray
    current_max: np.ndarray
    feature_values: np.ndarray


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build actuator-first generated 0.1 s T15 targets. Piecewise-constant ladder Jdot "
            "commands are generated from real trim50 replay scale/coupling statistics, simulated "
            "through tokamak-sim, and accepted "
            "only when the resulting Ip/boundary/current trajectory stays inside the observed "
            "real trim50 replay state space."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--machine-config", type=Path, default=DEFAULT_MACHINE_CONFIG)
    parser.add_argument("--observed-targets", type=Path, default=DEFAULT_OBSERVED_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-states-out", type=Path, default=DEFAULT_INITIAL_STATES_OUT)
    parser.add_argument("--targets-out", type=Path, default=DEFAULT_TARGETS_OUT)
    parser.add_argument("--diagnostic-targets-out", type=Path, default=DEFAULT_DIAGNOSTIC_TARGETS_OUT)
    parser.add_argument("--train-shots", nargs="+", default=list(DEFAULT_TRAIN_SHOTS))
    parser.add_argument("--holdout-shots", nargs="+", default=list(DEFAULT_HOLDOUT_SHOTS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--target-count", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--angles", type=int, default=32)
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--ip-margin-a", type=float, default=15000.0)
    parser.add_argument("--radii-margin-m", type=float, default=0.05)
    parser.add_argument("--current-margin-fraction", type=float, default=0.03)
    parser.add_argument(
        "--state-feature-distance-limit",
        type=float,
        default=0.0,
        help="0 disables nearest-neighbor filtering; envelope filtering is always active.",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    if int(args.steps) != 100:
        raise SystemExit("actuator-generated builder currently targets exactly 0.1 s / 100 steps")
    if int(args.target_count) <= 0:
        raise SystemExit("--target-count must be positive")

    train_shots = tuple(str(int(v)) for v in args.train_shots)
    holdout_shots = tuple(str(int(v)) for v in args.holdout_shots)
    overlap = sorted(set(train_shots) & set(holdout_shots), key=int)
    if overlap:
        raise SystemExit("train and holdout shots overlap: " + ", ".join(overlap))

    machine_config = _repo_path(args.machine_config)
    observed_targets = _repo_path(args.observed_targets)
    data_root = _repo_path(args.data_root)
    out_dir = _repo_path(args.out_dir)
    initial_states_out = _repo_path(args.initial_states_out)
    targets_out = _repo_path(args.targets_out)
    diagnostic_targets_out = _repo_path(args.diagnostic_targets_out)

    limits = _load_limits(machine_config)
    windows = _load_real_windows(
        data_root=data_root,
        train_shots=train_shots,
        holdout_shots=holdout_shots,
        steps=int(args.steps),
    )
    if not windows:
        raise SystemExit("no real trim50 windows available for actuator generation")
    train_windows = [w for w in windows if w.split == "train"]
    holdout_windows = [w for w in windows if w.split == "holdout"]
    if not train_windows or not holdout_windows:
        raise SystemExit("actuator generation requires non-empty train and holdout windows")

    envelope = _load_observed_envelope(
        observed_targets=observed_targets,
        windows=windows,
        limits=limits,
        ip_margin_a=float(args.ip_margin_a),
        radii_margin_m=float(args.radii_margin_m),
        current_margin_fraction=float(args.current_margin_fraction),
    )

    rng = np.random.default_rng(int(args.seed))
    split_counts = _split_counts(int(args.target_count), train_windows=train_windows, holdout_windows=holdout_windows)
    candidates = {
        "train": _sample_coil_candidates(
            train_windows,
            count=split_counts["train"],
            rng=rng,
            limits=limits,
            steps=int(args.steps),
        ),
        "holdout": _sample_coil_candidates(
            holdout_windows,
            count=split_counts["holdout"],
            rng=rng,
            limits=limits,
            steps=int(args.steps),
        ),
    }
    all_candidates = [*candidates["train"], *candidates["holdout"]]

    rows, rejected = _simulate_and_filter(
        all_candidates,
        machine_config=machine_config,
        limits=limits,
        envelope=envelope,
        batch_size=int(args.batch_size),
        angles=int(args.angles),
        gpu_device=str(args.gpu_device),
        state_feature_distance_limit=float(args.state_feature_distance_limit),
    )
    if not rows:
        _write_rejections(out_dir / "actuator_generated_rejected.csv", rejected)
        raise SystemExit("no actuator-generated rollouts survived simulation filtering")

    _write_libraries(
        rows,
        initial_states_out=initial_states_out,
        targets_out=targets_out,
        diagnostic_targets_out=diagnostic_targets_out,
        limits=limits,
        train_shots=train_shots,
        holdout_shots=holdout_shots,
    )
    _write_accepted(out_dir / "actuator_generated_accepted.csv", rows)
    _write_rejections(out_dir / "actuator_generated_rejected.csv", rejected)
    summary = _summary(rows, rejected, args=args, limits=limits, envelope=envelope)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "actuator_generated_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.plots:
        _write_plots(rows, out_dir=out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _ensure_tokamak_sim_importable() -> None:
    sim_root = (ROOT.parent / "tokamak-sim").resolve()
    if str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))


def _load_limits(path: Path) -> Limits:
    if not path.exists():
        raise FileNotFoundError(f"machine config does not exist: {path}")
    text = path.read_text(encoding="utf-8")

    def value(name: str) -> float:
        import re

        match = re.search(rf"^\s*{name}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE)
        if not match:
            raise ValueError(f"{path} missing {name}")
        return float(match.group(1))

    return Limits(
        pfc_current=value("pfc_current_limit"),
        sol_current=value("sol_current_limit"),
        pfc_deriv=value("pfc_deriv_limit"),
        sol_deriv=value("sol_deriv_limit"),
    )


def _load_real_windows(
    *,
    data_root: Path,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
    steps: int,
) -> list[RealWindow]:
    out: list[RealWindow] = []
    for shot in sorted(set(train_shots) | set(holdout_shots), key=int):
        split = "holdout" if shot in holdout_shots else "train"
        ip = _load_table(data_root / "ip" / f"t15md_{shot}_ip.csv")
        coils_raw = _load_table(data_root / "coils" / f"t15md_{shot}_coils.csv")
        if ip.shape[1] < 2:
            raise ValueError(f"Ip table for shot {shot} must contain time and Ip")
        if coils_raw.shape[1] != 10:
            raise ValueError(f"coil table for shot {shot} must contain time + 9 currents")
        times = ip[:, 0]
        sol = np.stack([np.interp(times, coils_raw[:, 0], coils_raw[:, col]) for col in (1, 2, 3)], axis=1)
        pfc = np.stack([np.interp(times, coils_raw[:, 0], coils_raw[:, col]) for col in (4, 5, 6, 7, 8, 9)], axis=1)
        currents = np.concatenate([pfc, sol], axis=1)
        valid = int(ip.shape[0]) - int(steps)
        for start in range(max(valid, 0)):
            segment_times = times[start : start + int(steps) + 1]
            if segment_times.shape[0] != int(steps) + 1:
                continue
            if np.max(np.abs(np.diff(segment_times) - 0.001)) > 1.0e-6:
                continue
            out.append(
                RealWindow(
                    shot_id=str(shot),
                    split=split,
                    start=int(start),
                    time_s=float(segment_times[0]),
                    ip=np.asarray(ip[start : start + int(steps) + 1, 1], dtype=float),
                    currents=np.asarray(currents[start : start + int(steps) + 1], dtype=float),
                )
            )
    return out


def _load_table(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.loadtxt(path, delimiter=";", dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"{path} must be a 2D table with at least two rows")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path} contains non-finite values")
    if np.any(np.diff(arr[:, 0]) <= 0.0):
        raise ValueError(f"{path} time column must be strictly increasing")
    return arr


def _load_observed_envelope(
    *,
    observed_targets: Path,
    windows: list[RealWindow],
    limits: Limits,
    ip_margin_a: float,
    radii_margin_m: float,
    current_margin_fraction: float,
) -> ObservedEnvelope:
    if not observed_targets.exists():
        raise FileNotFoundError(
            f"observed real replay target NPZ is required for state-space filtering: {observed_targets}"
        )
    with np.load(observed_targets, allow_pickle=False) as data:
        required = {"ip_target", "boundary_radii"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{observed_targets} missing arrays: {', '.join(missing)}")
        ip = np.asarray(data["ip_target"], dtype=float)
        radii = np.asarray(data["boundary_radii"], dtype=float)
        feature_values = _features_from_rollouts(ip, radii)
    all_currents = np.concatenate([w.currents for w in windows], axis=0)
    span = np.nanmax(all_currents, axis=0) - np.nanmin(all_currents, axis=0)
    margin = np.maximum(float(current_margin_fraction) * span, 0.02 * limits.current_vector)
    return ObservedEnvelope(
        ip_min=float(np.nanmin(ip) - float(ip_margin_a)),
        ip_max=float(np.nanmax(ip) + float(ip_margin_a)),
        radii_min=np.nanmin(radii, axis=(0, 1)) - float(radii_margin_m),
        radii_max=np.nanmax(radii, axis=(0, 1)) + float(radii_margin_m),
        current_min=np.nanmin(all_currents, axis=0) - margin,
        current_max=np.nanmax(all_currents, axis=0) + margin,
        feature_values=feature_values,
    )


def _features_from_rollouts(ip: np.ndarray, radii: np.ndarray) -> np.ndarray:
    ip = np.asarray(ip, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if ip.ndim != 2 or radii.ndim != 3 or ip.shape[:2] != radii.shape[:2]:
        raise ValueError(f"bad feature input shapes: ip={ip.shape}, radii={radii.shape}")
    rows: list[np.ndarray] = []
    for k in range(0, ip.shape[1], 5):
        r = radii[:, k, :]
        rows.append(
            np.stack(
                [
                    ip[:, k] / 426401.0,
                    np.nanmean(r, axis=1) / 0.72,
                    np.nanmin(r, axis=1) / 0.55,
                    np.nanmax(r, axis=1) / 0.85,
                ],
                axis=1,
            )
        )
    return np.concatenate(rows, axis=0)


def _split_counts(target_count: int, *, train_windows: list[RealWindow], holdout_windows: list[RealWindow]) -> dict[str, int]:
    total = max(len(train_windows) + len(holdout_windows), 1)
    holdout = max(1, int(round(float(target_count) * len(holdout_windows) / total)))
    train = max(1, int(target_count) - holdout)
    return {"train": train, "holdout": holdout}


def _sample_coil_candidates(
    windows: list[RealWindow],
    *,
    count: int,
    rng: np.random.Generator,
    limits: Limits,
    steps: int,
) -> list[CoilCandidate]:
    style = _jdot_style_from_windows(windows=windows, steps=int(steps))
    modes = np.asarray(
        [
            "ladder_constant",
            "ladder_one_bend",
            "ladder_two_bend",
            "ladder_hold_drive",
            "ladder_drive_hold",
            "ladder_reversal",
        ],
        dtype=object,
    )
    out: list[CoilCandidate] = []
    attempts = 0
    max_attempts = max(2000, int(count) * 80)
    while len(out) < int(count) and attempts < max_attempts:
        attempts += 1
        reset = windows[int(rng.integers(0, len(windows)))]
        mode = str(modes[len(out) % len(modes)])
        motion = windows[int(rng.integers(0, len(windows)))]
        scale = float(rng.uniform(0.65, 1.20))
        gains = np.clip(rng.normal(loc=1.0, scale=0.04, size=(9,)), 0.9, 1.1)
        jdot = _generate_ladder_jdot(
            style=style,
            mode=mode,
            source_window=motion,
            rng=rng,
            limits=limits,
            steps=int(steps),
            scale=scale,
            gains=gains,
        )
        currents = reset.currents[0:1, :] + np.concatenate(
            [np.zeros((1, 9), dtype=float), np.cumsum(jdot * 0.001, axis=0)],
            axis=0,
        )
        if currents.shape != (int(steps) + 1, 9):
            continue
        if not np.all(np.isfinite(currents)):
            continue
        if np.any(np.abs(currents) > limits.current_vector[None, :]):
            continue
        action = jdot / limits.derivative_vector[None, :]
        if np.any(np.abs(action) > 1.0001):
            continue
        out.append(
            CoilCandidate(
                reset=reset,
                motion=motion,
                mode=mode,
                scale=float(scale),
                gains=np.asarray(gains, dtype=float),
                ip0=float(reset.ip[0]),
                currents=np.asarray(currents, dtype=float),
                action=np.asarray(action, dtype=float),
            )
        )
    if len(out) < int(count):
        raise RuntimeError(f"accepted only {len(out)} / {count} actuator candidates after {attempts} attempts")
    return out


def _jdot_style_from_windows(*, windows: list[RealWindow], steps: int) -> JdotStyle:
    vectors: list[np.ndarray] = []
    for window in windows:
        if window.currents.shape != (int(steps) + 1, 9):
            continue
        avg = (window.currents[-1] - window.currents[0]) / (float(steps) * 0.001)
        if not np.all(np.isfinite(avg)):
            continue
        if float(np.max(np.abs(avg))) < 1.0:
            continue
        vectors.append(np.asarray(avg, dtype=float))
    if not vectors:
        raise ValueError("could not derive any Jdot style vectors from real windows")
    return JdotStyle(average_vectors=np.stack(vectors, axis=0))


def _generate_ladder_jdot(
    *,
    style: JdotStyle,
    mode: str,
    source_window: RealWindow,
    rng: np.random.Generator,
    limits: Limits,
    steps: int,
    scale: float,
    gains: np.ndarray,
) -> np.ndarray:
    if style.average_vectors.shape[0] <= 0:
        raise ValueError("empty Jdot style")
    base = np.asarray(style.average_vectors[int(rng.integers(0, style.average_vectors.shape[0]))], dtype=float)

    # Keep the generated command on the same coarse coil-coupling manifold as
    # real replay windows, but do not copy a real step-by-step Jdot sequence.
    source_avg = (source_window.currents[-1] - source_window.currents[0]) / (float(steps) * 0.001)
    if np.all(np.isfinite(source_avg)) and float(np.max(np.abs(source_avg))) > 1.0:
        mix = float(rng.uniform(0.35, 0.75))
        base = mix * base + (1.0 - mix) * source_avg

    base = base * float(scale) * np.asarray(gains, dtype=float)
    max_action = float(np.max(np.abs(base / limits.derivative_vector)))
    if max_action > 0.82:
        base = base * (0.82 / max_action)

    factors = _ladder_factors(mode=mode, steps=int(steps), rng=rng)
    jdot = np.zeros((int(steps), 9), dtype=float)
    start = 0
    for length, factor in factors:
        end = min(int(steps), start + int(length))
        if end > start:
            jdot[start:end, :] = float(factor) * base[None, :]
        start = end
    if start < int(steps):
        jdot[start:, :] = float(factors[-1][1]) * base[None, :]

    action = jdot / limits.derivative_vector[None, :]
    max_abs = float(np.max(np.abs(action)))
    if max_abs > 0.98:
        jdot *= 0.98 / max_abs
    return jdot


def _ladder_factors(*, mode: str, steps: int, rng: np.random.Generator) -> list[tuple[int, float]]:
    if mode == "ladder_constant":
        return [(steps, float(rng.uniform(0.75, 1.15)))]
    if mode == "ladder_one_bend":
        s0 = int(rng.integers(25, 76))
        return [(s0, float(rng.uniform(0.55, 0.95))), (steps - s0, float(rng.uniform(0.90, 1.35)))]
    if mode == "ladder_two_bend":
        s0 = int(rng.integers(20, 46))
        s1 = int(rng.integers(max(s0 + 20, 50), 86))
        return [
            (s0, float(rng.uniform(0.55, 0.95))),
            (s1 - s0, float(rng.uniform(0.90, 1.35))),
            (steps - s1, float(rng.uniform(0.45, 1.05))),
        ]
    if mode == "ladder_hold_drive":
        hold = int(rng.integers(8, 31))
        return [(hold, 0.0), (steps - hold, float(rng.uniform(0.85, 1.25)))]
    if mode == "ladder_drive_hold":
        drive = int(rng.integers(60, 91))
        return [(drive, float(rng.uniform(0.85, 1.25))), (steps - drive, 0.0)]
    if mode == "ladder_reversal":
        s0 = int(rng.integers(35, 71))
        return [(s0, float(rng.uniform(0.75, 1.15))), (steps - s0, -float(rng.uniform(0.25, 0.70)))]
    raise ValueError(f"unknown ladder mode: {mode}")


def _simulate_and_filter(
    candidates: list[CoilCandidate],
    *,
    machine_config: Path,
    limits: Limits,
    envelope: ObservedEnvelope,
    batch_size: int,
    angles: int,
    gpu_device: str,
    state_feature_distance_limit: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    _ensure_tokamak_sim_importable()
    from tokamak_control.core.batched_gpu_simulator import BatchedGpuTokamakSimulator
    from tokamak_control.io.config_io import load_config

    sim_cfg = load_config(machine_config)
    theta = np.linspace(-np.pi, np.pi, int(angles), endpoint=False, dtype=float)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for start in range(0, len(candidates), int(batch_size)):
        batch = candidates[start : start + int(batch_size)]
        bsz = len(batch)
        sim = BatchedGpuTokamakSimulator(
            grid=sim_cfg.grid,
            pfc=sim_cfg.pfc,
            sol=sim_cfg.sol,
            settings=sim_cfg.physics,
            batch_size=bsz,
            angles_rad=theta,
            limiter_shape=sim_cfg.limiter_shape,
            boundary_mode=sim_cfg.boundary_mode,
            boundary_base_mode=sim_cfg.boundary_base_mode,
            boundary_legacy_precision_index2=sim_cfg.boundary_legacy_precision_index2,
            boundary_soft_level_selection=sim_cfg.boundary_soft_level_selection,
            boundary_soft_level_candidates=sim_cfg.boundary_soft_level_candidates,
            boundary_soft_level_temperature=sim_cfg.boundary_soft_level_temperature,
            boundary_soft_level_radius_weight=sim_cfg.boundary_soft_level_radius_weight,
            boundary_soft_level_missing_penalty=sim_cfg.boundary_soft_level_missing_penalty,
            boundary_soft_level_roughness_penalty=sim_cfg.boundary_soft_level_roughness_penalty,
            boundary_level_smoothing_alpha=sim_cfg.boundary_level_smoothing_alpha,
            boundary_level_search_span_fraction=sim_cfg.boundary_level_search_span_fraction,
            boundary_continuity_weight_radii=sim_cfg.boundary_continuity_weight_radii,
            boundary_continuity_weight_mean_radius=sim_cfg.boundary_continuity_weight_mean_radius,
            boundary_continuity_weight_center=sim_cfg.boundary_continuity_weight_center,
            boundary_continuity_weight_area=sim_cfg.boundary_continuity_weight_area,
            boundary_continuity_weight_level=sim_cfg.boundary_continuity_weight_level,
            gpu_device=gpu_device,
        )
        ip0 = np.asarray([c.ip0 for c in batch], dtype=float)
        current0 = np.stack([c.currents[0] for c in batch], axis=0)
        result = sim.reset(ip=ip0, pfc_currents=current0[:, : sim_cfg.pfc.n_coils], sol_currents=current0[:, sim_cfg.pfc.n_coils :])
        sim_ip = [result.state.Ip.detach().cpu().numpy().astype(float)]
        radii = [result.boundary.radii.detach().cpu().numpy().astype(float)]
        found = [result.boundary.found.detach().cpu().numpy().astype(bool)]
        for k in range(batch[0].currents.shape[0] - 1):
            next_current = np.stack([c.currents[k + 1] for c in batch], axis=0)
            result = sim.step_currents(next_current)
            sim_ip.append(result.state.Ip.detach().cpu().numpy().astype(float))
            radii.append(result.boundary.radii.detach().cpu().numpy().astype(float))
            found.append(result.boundary.found.detach().cpu().numpy().astype(bool))
        sim_ip_arr = np.stack(sim_ip, axis=1)
        radii_arr = np.stack(radii, axis=1)
        found_arr = np.stack(found, axis=1)
        for row, candidate in enumerate(batch):
            ok, reason, distance = _rollout_ok(
                ip=sim_ip_arr[row],
                radii=radii_arr[row],
                found=found_arr[row],
                currents=candidate.currents,
                envelope=envelope,
                state_feature_distance_limit=state_feature_distance_limit,
            )
            if not ok:
                rejected.append(_reject(candidate, reason, distance))
                continue
            accepted.append(
                {
                    "shot_id": candidate.reset.shot_id,
                    "split": candidate.reset.split,
                    "source_index": candidate.reset.start,
                    "time_s": candidate.reset.time_s,
                    "ip0": float(candidate.ip0),
                    "pfc0": candidate.currents[0, :6].astype(np.float32),
                    "sol0": candidate.currents[0, 6:].astype(np.float32),
                    "ip_target": sim_ip_arr[row].astype(np.float32),
                    "boundary_radii": radii_arr[row].astype(np.float32),
                    "real_jdot_action": candidate.action.astype(np.float32),
                    "difficulty_bin": _difficulty_bin(sim_ip_arr[row]),
                    "mode": candidate.mode,
                    "motion_shot_id": candidate.motion.shot_id,
                    "motion_source_index": int(candidate.motion.start),
                    "scale": float(candidate.scale),
                    "state_feature_distance": float(distance),
                    "oracle_ip_mean_error_a": 0.0,
                    "oracle_ip_max_error_a": 0.0,
                }
            )
        print(
            f"[actuator-gen-sim] processed={min(start + bsz, len(candidates))}/{len(candidates)} "
            f"accepted={len(accepted)} rejected={len(rejected)}",
            flush=True,
        )
    return accepted, rejected


def _rollout_ok(
    *,
    ip: np.ndarray,
    radii: np.ndarray,
    found: np.ndarray,
    currents: np.ndarray,
    envelope: ObservedEnvelope,
    state_feature_distance_limit: float,
) -> tuple[bool, str, float]:
    if not np.all(np.asarray(found, dtype=bool)):
        return False, "boundary_lost", float("inf")
    if not np.all(np.isfinite(ip)) or not np.all(np.isfinite(radii)) or not np.all(np.isfinite(currents)):
        return False, "nonfinite", float("inf")
    if np.nanmin(ip) < envelope.ip_min or np.nanmax(ip) > envelope.ip_max:
        return False, "ip_outside_observed_envelope", float("inf")
    if np.nanmin(radii - envelope.radii_min[None, :]) < -1.0e-8:
        return False, "radii_below_observed_envelope", float("inf")
    if np.nanmax(radii - envelope.radii_max[None, :]) > 1.0e-8:
        return False, "radii_above_observed_envelope", float("inf")
    if np.nanmin(currents - envelope.current_min[None, :]) < -1.0e-8:
        return False, "current_below_observed_envelope", float("inf")
    if np.nanmax(currents - envelope.current_max[None, :]) > 1.0e-8:
        return False, "current_above_observed_envelope", float("inf")
    if float(state_feature_distance_limit) <= 0.0:
        return True, "ok", 0.0
    features = _features_from_rollouts(ip.reshape(1, -1), radii.reshape(1, radii.shape[0], radii.shape[1]))
    diff = features[:, None, :] - envelope.feature_values[None, :, :]
    distance = float(np.sqrt(np.nanmin(np.sum(diff * diff, axis=-1))))
    if distance > float(state_feature_distance_limit):
        return False, "nearest_state_feature_distance", distance
    return True, "ok", distance


def _difficulty_bin(ip: np.ndarray) -> str:
    delta = float(np.asarray(ip, dtype=float)[-1] - np.asarray(ip, dtype=float)[0])
    mag = abs(delta)
    if mag < 10000.0:
        return "flat"
    prefix = "fast" if mag >= 40000.0 else "medium"
    suffix = "up" if delta > 0.0 else "down"
    return f"{prefix}_{suffix}"


def _write_libraries(
    rows: list[dict[str, object]],
    *,
    initial_states_out: Path,
    targets_out: Path,
    diagnostic_targets_out: Path,
    limits: Limits,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
) -> None:
    initial_states_out.parent.mkdir(parents=True, exist_ok=True)
    targets_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_targets_out.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(rows)
    if row_count <= 0:
        raise ValueError("cannot write empty actuator-generated libraries")
    source_index = np.arange(row_count, dtype=np.int64)
    np.savez_compressed(
        initial_states_out,
        schema=np.asarray("t15_actuator_generated_trim50_plain_gpu1e6_0p1s_initial_states_v1"),
        shot_id=np.asarray([r["shot_id"] for r in rows]),
        source_index=source_index,
        reset_source_index=np.asarray([r["source_index"] for r in rows], dtype=np.int64),
        time_s=np.asarray([r["time_s"] for r in rows], dtype=np.float64),
        ip0=np.asarray([r["ip0"] for r in rows], dtype=np.float32),
        pfc0=np.stack([r["pfc0"] for r in rows], axis=0).astype(np.float32),
        sol0=np.stack([r["sol0"] for r in rows], axis=0).astype(np.float32),
        split=np.asarray([r["split"] for r in rows]),
        difficulty_bin=np.asarray([r["difficulty_bin"] for r in rows]),
        mode=np.asarray([r["mode"] for r in rows]),
        motion_shot_id=np.asarray([r["motion_shot_id"] for r in rows]),
        motion_source_index=np.asarray([r["motion_source_index"] for r in rows], dtype=np.int64),
    )
    oracle_kwargs = dict(
        schema=np.asarray("t15_replay_window_oracle_targets_v1"),
        shot_id=np.asarray([r["shot_id"] for r in rows]),
        split=np.asarray([r["split"] for r in rows]),
        source_index=source_index,
        reset_source_index=np.asarray([r["source_index"] for r in rows], dtype=np.int64),
        time_s=np.asarray([r["time_s"] for r in rows], dtype=np.float64),
        difficulty_bin=np.asarray([r["difficulty_bin"] for r in rows]),
        mode=np.asarray([r["mode"] for r in rows]),
        motion_shot_id=np.asarray([r["motion_shot_id"] for r in rows]),
        motion_source_index=np.asarray([r["motion_source_index"] for r in rows], dtype=np.int64),
        scale=np.asarray([r["scale"] for r in rows], dtype=np.float32),
        state_feature_distance=np.asarray([r["state_feature_distance"] for r in rows], dtype=np.float32),
        ip0=np.asarray([r["ip0"] for r in rows], dtype=np.float32),
        pfc0=np.stack([r["pfc0"] for r in rows], axis=0).astype(np.float32),
        sol0=np.stack([r["sol0"] for r in rows], axis=0).astype(np.float32),
        ip_target=np.stack([r["ip_target"] for r in rows], axis=0).astype(np.float32),
        boundary_radii=np.stack([r["boundary_radii"] for r in rows], axis=0).astype(np.float32),
        real_jdot_action=np.stack([r["real_jdot_action"] for r in rows], axis=0).astype(np.float32),
        oracle_ip_mean_error_a=np.asarray([r["oracle_ip_mean_error_a"] for r in rows], dtype=np.float32),
        oracle_ip_max_error_a=np.asarray([r["oracle_ip_max_error_a"] for r in rows], dtype=np.float32),
        current_limits=limits.current_vector.astype(np.float32),
        derivative_limits=limits.derivative_vector.astype(np.float32),
        train_shots=np.asarray(train_shots, dtype="<U8"),
        holdout_shots=np.asarray(holdout_shots, dtype="<U8"),
    )
    np.savez_compressed(targets_out, **oracle_kwargs)
    diagnostic_kwargs = dict(oracle_kwargs)
    diagnostic_kwargs["schema"] = np.asarray("t15_actuator_generated_trim50_plain_gpu1e6_0p1s_targets_v1")
    np.savez_compressed(diagnostic_targets_out, **diagnostic_kwargs)


def _write_rejections(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["shot_id", "source_index", "time_s", "mode", "motion_shot_id", "motion_source_index", "reason", "distance"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_accepted(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row",
        "shot_id",
        "split",
        "source_index",
        "time_s",
        "mode",
        "motion_shot_id",
        "motion_source_index",
        "scale",
        "difficulty_bin",
        "ip0_a",
        "ip_end_a",
        "ip_delta_a",
        "mean_radius0_m",
        "mean_radius_end_m",
        "mean_radius_delta_m",
        "max_radius_m",
        "min_radius_m",
        "action_rms_mean",
        "action_abs_max",
        "state_feature_distance",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            ip = np.asarray(row["ip_target"], dtype=float)
            radii = np.asarray(row["boundary_radii"], dtype=float)
            action = np.asarray(row["real_jdot_action"], dtype=float)
            writer.writerow(
                {
                    "row": int(idx),
                    "shot_id": str(row["shot_id"]),
                    "split": str(row["split"]),
                    "source_index": int(row["source_index"]),
                    "time_s": f"{float(row['time_s']):.9g}",
                    "mode": str(row["mode"]),
                    "motion_shot_id": str(row["motion_shot_id"]),
                    "motion_source_index": int(row["motion_source_index"]),
                    "scale": f"{float(row['scale']):.9g}",
                    "difficulty_bin": str(row["difficulty_bin"]),
                    "ip0_a": f"{float(ip[0]):.9g}",
                    "ip_end_a": f"{float(ip[-1]):.9g}",
                    "ip_delta_a": f"{float(ip[-1] - ip[0]):.9g}",
                    "mean_radius0_m": f"{float(np.nanmean(radii[0])):.9g}",
                    "mean_radius_end_m": f"{float(np.nanmean(radii[-1])):.9g}",
                    "mean_radius_delta_m": f"{float(np.nanmean(radii[-1]) - np.nanmean(radii[0])):.9g}",
                    "max_radius_m": f"{float(np.nanmax(radii)):.9g}",
                    "min_radius_m": f"{float(np.nanmin(radii)):.9g}",
                    "action_rms_mean": f"{float(np.sqrt(np.nanmean(action * action))):.9g}",
                    "action_abs_max": f"{float(np.nanmax(np.abs(action))):.9g}",
                    "state_feature_distance": f"{float(row['state_feature_distance']):.9g}",
                }
            )


def _reject(candidate: CoilCandidate, reason: str, distance: float) -> dict[str, object]:
    return {
        "shot_id": candidate.reset.shot_id,
        "source_index": candidate.reset.start,
        "time_s": candidate.reset.time_s,
        "mode": candidate.mode,
        "motion_shot_id": candidate.motion.shot_id,
        "motion_source_index": candidate.motion.start,
        "reason": reason,
        "distance": float(distance),
    }


def _summary(
    rows: list[dict[str, object]],
    rejected: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    limits: Limits,
    envelope: ObservedEnvelope,
) -> dict[str, object]:
    splits = Counter(str(r["split"]) for r in rows)
    modes = Counter(str(r["mode"]) for r in rows)
    bins = Counter(str(r["difficulty_bin"]) for r in rows)
    reasons = Counter(str(r["reason"]) for r in rejected)
    ip = np.stack([r["ip_target"] for r in rows], axis=0)
    radii = np.stack([r["boundary_radii"] for r in rows], axis=0)
    actions = np.stack([r["real_jdot_action"] for r in rows], axis=0)
    return {
        "schema": "t15_actuator_generated_trim50_plain_gpu1e6_0p1s_summary_v1",
        "accepted_windows": int(len(rows)),
        "rejected_windows": int(len(rejected)),
        "split_counts": dict(sorted(splits.items())),
        "mode_counts": dict(sorted(modes.items())),
        "difficulty_bins": dict(sorted(bins.items())),
        "rejection_reasons": dict(sorted(reasons.items())),
        "ip_target_min_a": float(np.nanmin(ip)),
        "ip_target_max_a": float(np.nanmax(ip)),
        "boundary_radii_min_m": float(np.nanmin(radii)),
        "boundary_radii_max_m": float(np.nanmax(radii)),
        "max_abs_normalized_action": float(np.nanmax(np.abs(actions))),
        "observed_envelope": {
            "ip_min_a": float(envelope.ip_min),
            "ip_max_a": float(envelope.ip_max),
            "radii_min_m": float(np.nanmin(envelope.radii_min)),
            "radii_max_m": float(np.nanmax(envelope.radii_max)),
            "current_min_a": [float(v) for v in envelope.current_min.tolist()],
            "current_max_a": [float(v) for v in envelope.current_max.tolist()],
        },
        "limits": {
            "pfc_current": float(limits.pfc_current),
            "sol_current": float(limits.sol_current),
            "pfc_deriv": float(limits.pfc_deriv),
            "sol_deriv": float(limits.sol_deriv),
        },
        "args": {
            "target_count": int(args.target_count),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "state_feature_distance_limit": float(args.state_feature_distance_limit),
            "ip_margin_a": float(args.ip_margin_a),
            "radii_margin_m": float(args.radii_margin_m),
            "current_margin_fraction": float(args.current_margin_fraction),
        },
    }


def _write_plots(rows: list[dict[str, object]], *, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    sample_count = min(12, len(rows))
    idx = rng.choice(len(rows), size=sample_count, replace=False)
    t = np.arange(101, dtype=float) * 0.001
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
    for row_idx in idx.tolist():
        row = rows[int(row_idx)]
        label = f'{row["mode"]}:{row["shot_id"]}->{row["motion_shot_id"]}'
        ip = np.asarray(row["ip_target"], dtype=float)
        radii = np.asarray(row["boundary_radii"], dtype=float)
        action = np.asarray(row["real_jdot_action"], dtype=float)
        axes[0].plot(t, ip, lw=1.2, alpha=0.8, label=label)
        axes[1].plot(t, radii.mean(axis=1), lw=1.2, alpha=0.8)
        axes[2].plot(t, radii.max(axis=1), lw=1.2, alpha=0.8)
        axes[3].plot(t[1:], np.sqrt(np.mean(action**2, axis=1)), lw=1.2, alpha=0.8)
    axes[0].set_ylabel("Ip [A]")
    axes[1].set_ylabel("Mean radius [m]")
    axes[2].set_ylabel("Max radius [m]")
    axes[3].set_ylabel("Action RMS")
    axes[3].set_xlabel("time [s]")
    axes[0].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "sample_actuator_generated_targets.png", dpi=150)
    plt.close(fig)

    ip = np.stack([r["ip_target"] for r in rows], axis=0)
    radii = np.stack([r["boundary_radii"] for r in rows], axis=0)
    action = np.stack([r["real_jdot_action"] for r in rows], axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].hist((ip[:, -1] - ip[:, 0]) / 1000.0, bins=50)
    axes[0].set_xlabel("endpoint dIp [kA]")
    axes[1].hist((radii[:, -1].mean(axis=1) - radii[:, 0].mean(axis=1)), bins=50)
    axes[1].set_xlabel("endpoint d mean radius [m]")
    axes[2].hist(np.max(np.abs(action), axis=(1, 2)), bins=50)
    axes[2].set_xlabel("max |normalized Jdot|")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "actuator_generated_coverage_histograms.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
