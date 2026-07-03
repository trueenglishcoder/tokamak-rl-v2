#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = Path(__file__).with_name("build_t15_actuator_generated_trim50_plain_gpu1e6_0p1s.py")

DEFAULT_DATA_ROOT = Path("../tokamak-sim/data/t15_data_new_trim50")
DEFAULT_MACHINE_CONFIG = Path("data/processed/t15_new_trim50_plain_gpu1e6_machine_config.toml")
DEFAULT_OBSERVED_TARGETS = Path(
    "data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/"
    "t15_replay_window_oracle_targets.npz"
)
DEFAULT_OUT_DIR = Path("data/processed/t15_actuator_long_generated_trim50_plain_gpu1e6_0p1s")
DEFAULT_INITIAL_STATES_OUT = Path(
    "data/processed/t15_actuator_long_generated_trim50_plain_gpu1e6_0p1s_initial_states.npz"
)
DEFAULT_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_replay_window_oracle_targets.npz"
DEFAULT_DIAGNOSTIC_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_actuator_long_generated_targets.npz"
DEFAULT_PARENTS_OUT = DEFAULT_OUT_DIR / "t15_actuator_long_generated_parents.npz"
DEFAULT_TRAIN_SHOTS = ("3856", "3857", "3858", "3863")
DEFAULT_HOLDOUT_SHOTS = ("3864",)
COIL_NAMES = ("PFC0", "PFC1", "PFC2", "PFC3", "PFC4", "PFC5", "SOL0", "SOL1", "SOL2")


def _load_base_builder():
    spec = importlib.util.spec_from_file_location("_actuator_generated_0p1s_base", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import base builder: {BASE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_builder()


@dataclass(frozen=True, slots=True)
class ShotSeries:
    shot_id: str
    split: str
    time_s: np.ndarray
    ip: np.ndarray
    currents: np.ndarray


@dataclass(frozen=True, slots=True)
class ParentReset:
    shot_id: str
    split: str
    start: int
    time_s: float
    ip: np.ndarray
    currents: np.ndarray


@dataclass(frozen=True, slots=True)
class ParentCandidate:
    parent_id: int
    split: str
    reset: object
    mode: str
    scale: float
    style_source: str
    currents: np.ndarray
    action: np.ndarray


@dataclass(frozen=True, slots=True)
class JdotFeatureEnvelope:
    names: tuple[str, ...]
    min_values: np.ndarray
    max_values: np.ndarray
    samples: np.ndarray


@dataclass(frozen=True, slots=True)
class JdotStyleModel:
    split: str
    action_mean: np.ndarray
    action_chol: np.ndarray
    delta_chol: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    segment_lengths: np.ndarray
    ripple_amp: np.ndarray
    feature_envelope: JdotFeatureEnvelope
    level_bank: np.ndarray
    delta_bank: np.ndarray
    profile_mean: np.ndarray
    profile_components: np.ndarray
    profile_coeff_min: np.ndarray
    profile_coeff_max: np.ndarray
    profile_coeff_std: np.ndarray


@dataclass(frozen=True, slots=True)
class ParentRollout:
    candidate: ParentCandidate
    ip: np.ndarray
    radii: np.ndarray
    found: np.ndarray
    state_feature_distance: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build long-parent actuator-generated T15 data. The builder first creates generated "
            "1.0-1.5 s coil-current/Jdot parent rollouts, simulates Ip/boundary, filters them "
            "against the real trim50 state envelope, then cuts dense overlapping 100-step windows."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--machine-config", type=Path, default=DEFAULT_MACHINE_CONFIG)
    parser.add_argument("--observed-targets", type=Path, default=DEFAULT_OBSERVED_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-states-out", type=Path, default=DEFAULT_INITIAL_STATES_OUT)
    parser.add_argument("--targets-out", type=Path, default=DEFAULT_TARGETS_OUT)
    parser.add_argument("--diagnostic-targets-out", type=Path, default=DEFAULT_DIAGNOSTIC_TARGETS_OUT)
    parser.add_argument("--parents-out", type=Path, default=DEFAULT_PARENTS_OUT)
    parser.add_argument("--train-shots", nargs="+", default=list(DEFAULT_TRAIN_SHOTS))
    parser.add_argument("--holdout-shots", nargs="+", default=list(DEFAULT_HOLDOUT_SHOTS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--parent-count", type=int, default=100)
    parser.add_argument("--target-count", type=int, default=0, help="Compatibility alias; 0 keeps all overlapping windows.")
    parser.add_argument("--parent-min-steps", type=int, default=1000)
    parser.add_argument("--parent-max-steps", type=int, default=1500)
    parser.add_argument(
        "--parent-lengths",
        nargs="*",
        type=int,
        default=[1000, 1250, 1500],
        help="Discrete parent lengths to sample; discrete lengths allow efficient GPU batching.",
    )
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=0, help="0 keeps every accepted overlapping 100-step window.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of same-length parent trajectories per GPU batch.")
    parser.add_argument("--angles", type=int, default=32)
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--ip-margin-a", type=float, default=15000.0)
    parser.add_argument("--radii-margin-m", type=float, default=0.05)
    parser.add_argument("--current-margin-fraction", type=float, default=0.03)
    parser.add_argument("--state-feature-distance-limit", type=float, default=0.0)
    parser.add_argument("--level-scale-min", type=float, default=0.97)
    parser.add_argument("--level-scale-max", type=float, default=1.03)
    parser.add_argument(
        "--residual-action-rms",
        type=float,
        default=1.0,
        help="Multiplier for fitted small high-frequency residuals; 0 disables them.",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    if int(args.steps) != 100:
        raise SystemExit("long-parent builder currently cuts exactly 0.1 s / 100-step windows")
    if int(args.parent_count) <= 0:
        raise SystemExit("--parent-count must be positive")
    if int(args.parent_min_steps) < int(args.steps) + 1:
        raise SystemExit("--parent-min-steps must be greater than the 100-step window length")
    if int(args.parent_max_steps) < int(args.parent_min_steps):
        raise SystemExit("--parent-max-steps must be >= --parent-min-steps")
    parent_lengths = [int(v) for v in args.parent_lengths if int(args.parent_min_steps) <= int(v) <= int(args.parent_max_steps)]
    if not parent_lengths:
        raise SystemExit("--parent-lengths must include at least one value inside [parent-min-steps, parent-max-steps]")
    args.parent_lengths = parent_lengths
    if int(args.window_stride) <= 0:
        raise SystemExit("--window-stride must be positive")

    train_shots = tuple(str(int(v)) for v in args.train_shots)
    holdout_shots = tuple(str(int(v)) for v in args.holdout_shots)
    overlap = sorted(set(train_shots) & set(holdout_shots), key=int)
    if overlap:
        raise SystemExit("train and holdout shots overlap: " + ", ".join(overlap))

    data_root = _repo_path(args.data_root)
    machine_config = _repo_path(args.machine_config)
    observed_targets = _repo_path(args.observed_targets)
    out_dir = _repo_path(args.out_dir)
    initial_states_out = _repo_path(args.initial_states_out)
    targets_out = _repo_path(args.targets_out)
    diagnostic_targets_out = _repo_path(args.diagnostic_targets_out)
    parents_out = _repo_path(args.parents_out)

    limits = base._load_limits(machine_config)
    real_windows = base._load_real_windows(
        data_root=data_root,
        train_shots=train_shots,
        holdout_shots=holdout_shots,
        steps=int(args.steps),
    )
    if not real_windows:
        raise SystemExit("no real trim50 windows found")
    train_windows = [w for w in real_windows if w.split == "train"]
    holdout_windows = [w for w in real_windows if w.split == "holdout"]
    if not train_windows or not holdout_windows:
        raise SystemExit("non-empty train and holdout windows are required")

    shot_series = _load_shot_series(data_root, train_shots=train_shots, holdout_shots=holdout_shots)
    envelope = base._load_observed_envelope(
        observed_targets=observed_targets,
        windows=real_windows,
        limits=limits,
        ip_margin_a=float(args.ip_margin_a),
        radii_margin_m=float(args.radii_margin_m),
        current_margin_fraction=float(args.current_margin_fraction),
    )

    rng = np.random.default_rng(int(args.seed))
    style_models = {
        "train": _fit_jdot_style_model(
            [s for s in shot_series.values() if s.split == "train"],
            limits=limits,
            split="train",
        ),
        "holdout": _fit_jdot_style_model(
            [s for s in shot_series.values() if s.split == "holdout"],
            limits=limits,
            split="holdout",
        ),
    }
    split_counts = base._split_counts(int(args.parent_count), train_windows=train_windows, holdout_windows=holdout_windows)
    candidates, candidate_rejections = _make_parent_candidates(
        train_windows=train_windows,
        holdout_windows=holdout_windows,
        style_models=style_models,
        split_counts=split_counts,
        limits=limits,
        envelope=envelope,
        rng=rng,
        args=args,
    )
    if not candidates:
        _write_parent_rejections(out_dir / "long_parent_rejected.csv", candidate_rejections)
        raise SystemExit("no long-parent coil candidates could be generated")

    parents, parent_rejections = _simulate_and_filter_parents(
        candidates,
        machine_config=machine_config,
        envelope=envelope,
        batch_size=int(args.batch_size),
        angles=int(args.angles),
        gpu_device=str(args.gpu_device),
        state_feature_distance_limit=float(args.state_feature_distance_limit),
    )
    all_rejections = [*candidate_rejections, *parent_rejections]
    if not parents:
        _write_parent_rejections(out_dir / "long_parent_rejected.csv", all_rejections)
        raise SystemExit("no long-parent rollouts survived state-envelope filtering")

    rows, window_rejections = _cut_parent_windows(
        parents,
        window_steps=int(args.steps),
        stride=int(args.window_stride),
        envelope=envelope,
        state_feature_distance_limit=float(args.state_feature_distance_limit),
    )
    if int(args.max_windows) > 0 and len(rows) > int(args.max_windows):
        rows = _balanced_subsample_rows(rows, max_windows=int(args.max_windows), rng=rng)
    if not rows:
        _write_parent_rejections(out_dir / "long_parent_rejected.csv", all_rejections)
        _write_window_rejections(out_dir / "long_parent_window_rejected.csv", window_rejections)
        raise SystemExit("no overlapping windows survived filtering")

    base._write_libraries(
        rows,
        initial_states_out=initial_states_out,
        targets_out=targets_out,
        diagnostic_targets_out=diagnostic_targets_out,
        limits=limits,
        train_shots=train_shots,
        holdout_shots=holdout_shots,
    )
    base._write_accepted(out_dir / "long_parent_windows_accepted.csv", rows)
    _write_parent_rejections(out_dir / "long_parent_rejected.csv", all_rejections)
    _write_window_rejections(out_dir / "long_parent_window_rejected.csv", window_rejections)
    _write_parent_rollouts(parents, parents_out)

    summary = _summary(
        parents=parents,
        parent_rejections=all_rejections,
        window_rows=rows,
        window_rejections=window_rejections,
        envelope=envelope,
        limits=limits,
        args=args,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "actuator_long_generated_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.plots:
        _write_plots(parents=parents, rows=rows, out_dir=out_dir)
    _write_report(out_dir / "summary" / "actuator_long_generated_dataset_report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _load_shot_series(
    data_root: Path,
    *,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
) -> dict[str, ShotSeries]:
    out: dict[str, ShotSeries] = {}
    for shot in sorted(set(train_shots) | set(holdout_shots), key=int):
        split = "holdout" if shot in holdout_shots else "train"
        ip = base._load_table(data_root / "ip" / f"t15md_{shot}_ip.csv")
        coils_raw = base._load_table(data_root / "coils" / f"t15md_{shot}_coils.csv")
        times = ip[:, 0]
        sol = np.stack([np.interp(times, coils_raw[:, 0], coils_raw[:, col]) for col in (1, 2, 3)], axis=1)
        pfc = np.stack([np.interp(times, coils_raw[:, 0], coils_raw[:, col]) for col in (4, 5, 6, 7, 8, 9)], axis=1)
        out[str(shot)] = ShotSeries(
            shot_id=str(shot),
            split=split,
            time_s=np.asarray(times, dtype=float),
            ip=np.asarray(ip[:, 1], dtype=float),
            currents=np.concatenate([pfc, sol], axis=1).astype(float),
        )
    return out


def _make_parent_candidates(
    *,
    train_windows: list[object],
    holdout_windows: list[object],
    style_models: dict[str, JdotStyleModel],
    split_counts: dict[str, int],
    limits: object,
    envelope: object,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[list[ParentCandidate], list[dict[str, object]]]:
    candidates: list[ParentCandidate] = []
    rejected: list[dict[str, object]] = []
    parent_id = 0
    for split, windows in (("train", train_windows), ("holdout", holdout_windows)):
        count = int(split_counts[split])
        attempts = 0
        max_attempts = max(1000, count * 120)
        style = style_models[split]
        while len([c for c in candidates if c.split == split]) < count and attempts < max_attempts:
            attempts += 1
            reset_window = windows[int(rng.integers(0, len(windows)))]
            parent_steps = int(args.parent_lengths[int(rng.integers(0, len(args.parent_lengths)))])
            reset = ParentReset(
                shot_id=reset_window.shot_id,
                split=reset_window.split,
                start=reset_window.start,
                time_s=float(reset_window.time_s),
                ip=np.asarray([reset_window.ip[0]], dtype=float),
                currents=np.asarray(reset_window.currents[0:1], dtype=float),
            )
            candidate, reason = _candidate_from_style(
                parent_id=parent_id,
                reset=reset,
                parent_steps=parent_steps,
                style=style,
                limits=limits,
                envelope=envelope,
                rng=rng,
                args=args,
            )
            if candidate is None:
                rejected.append(
                    {
                        "parent_id": parent_id,
                        "split": split,
                        "reset_shot_id": reset.shot_id,
                        "reset_source_index": reset.start,
                        "style_source": style.split,
                        "steps": parent_steps,
                        "mode": "candidate",
                        "reason": reason,
                        "distance": "",
                    }
                )
                continue
            candidates.append(candidate)
            parent_id += 1
    if len(candidates) < int(args.parent_count):
        raise RuntimeError(
            f"generated only {len(candidates)} / {args.parent_count} parent candidates; "
            f"rejections={dict(Counter(str(r['reason']) for r in rejected))}"
        )
    return candidates, rejected


def _candidate_from_style(
    *,
    parent_id: int,
    reset: object,
    parent_steps: int,
    style: JdotStyleModel,
    limits: object,
    envelope: object,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[ParentCandidate | None, str]:
    scale = float(rng.uniform(float(args.level_scale_min), float(args.level_scale_max)))
    action = _generate_synthetic_ladder_jdot_action(
        style=style,
        rng=rng,
        steps=parent_steps,
        scale=scale,
        residual_multiplier=float(args.residual_action_rms),
    )
    ok, feature_reason = _jdot_features_ok(action, style.feature_envelope)
    if not ok:
        return None, feature_reason
    currents = reset.currents[0:1] + np.concatenate(
        [np.zeros((1, 9), dtype=float), np.cumsum(action * limits.derivative_vector[None, :] * 0.001, axis=0)],
        axis=0,
    )
    if not np.all(np.isfinite(currents)):
        return None, "nonfinite_current"
    if np.any(np.abs(currents) > limits.current_vector[None, :]):
        return None, "current_limit"
    if np.nanmin(currents - envelope.current_min[None, :]) < -1.0e-8:
        return None, "current_below_observed_envelope"
    if np.nanmax(currents - envelope.current_max[None, :]) > 1.0e-8:
        return None, "current_above_observed_envelope"
    if float(np.nanmax(np.abs(action))) > 1.0001:
        return None, "derivative_limit"
    return (
        ParentCandidate(
            parent_id=int(parent_id),
            split=str(reset.split),
            reset=reset,
            mode="synthetic_ladder_jdot",
            scale=float(scale),
            style_source=style.split,
            currents=np.asarray(currents, dtype=float),
            action=np.asarray(action, dtype=float),
        ),
        "ok",
    )


def _fit_jdot_style_model(series: list[ShotSeries], *, limits: object, split: str) -> JdotStyleModel:
    if not series:
        raise ValueError(f"cannot fit Jdot style model for empty split: {split}")
    block = 25
    levels_by_shot: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_levels: list[np.ndarray] = []
    all_residuals: list[np.ndarray] = []
    segment_lengths: list[int] = []
    for shot in series:
        action = np.diff(shot.currents, axis=0) / 0.001 / limits.derivative_vector[None, :]
        action = np.clip(action, -0.98, 0.98)
        if action.shape[0] < block * 2:
            continue
        all_actions.append(action)
        n_blocks = action.shape[0] // block
        blocks = action[: n_blocks * block].reshape(n_blocks, block, 9).mean(axis=1)
        levels_by_shot.append(blocks)
        all_levels.append(blocks)
        repeated = np.repeat(blocks, block, axis=0)
        all_residuals.append(action[: repeated.shape[0]] - repeated)
        if blocks.shape[0] > 1:
            changes = np.max(np.abs(np.diff(blocks, axis=0)), axis=1)
            change_points = np.where(changes > 0.025)[0] + 1
            prev = 0
            for cp in change_points.tolist() + [blocks.shape[0]]:
                length = int((cp - prev) * block)
                if length >= 25:
                    segment_lengths.append(int(np.clip(length, 40, 260)))
                prev = cp
    if not all_levels:
        raise ValueError(f"could not derive Jdot levels for split={split}")
    levels = np.concatenate(all_levels, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    residuals = np.concatenate(all_residuals, axis=0) if all_residuals else np.zeros_like(actions)
    deltas = []
    for levels_one in levels_by_shot:
        if levels_one.shape[0] > 1:
            deltas.append(np.diff(levels_one, axis=0))
    delta_values = np.concatenate(deltas, axis=0) if deltas else levels - np.mean(levels, axis=0, keepdims=True)
    if not segment_lengths:
        segment_lengths = [50, 75, 100, 125, 150, 200, 250]

    action_mean = np.mean(levels, axis=0)
    action_cov = _regularized_cov(levels, diagonal_floor=2.5e-4)
    delta_cov = _regularized_cov(delta_values, diagonal_floor=1.0e-4)
    action_min = np.maximum(np.quantile(actions, 0.005, axis=0) - 0.03, -0.92)
    action_max = np.minimum(np.quantile(actions, 0.995, axis=0) + 0.03, 0.92)
    ripple_amp = np.quantile(np.abs(residuals), 0.85, axis=0)
    ripple_amp[:6] = np.minimum(ripple_amp[:6], 0.025)
    ripple_amp[6:] = np.minimum(ripple_amp[6:], 0.090)
    ripple_amp = np.maximum(ripple_amp, np.asarray([0.004] * 6 + [0.015] * 3, dtype=float))
    feature_envelope = _fit_jdot_feature_envelope(all_actions)
    profile_mean, profile_components, profile_coeff_min, profile_coeff_max, profile_coeff_std = _fit_profile_model(
        all_actions,
        action_min=action_min,
        action_max=action_max,
    )
    return JdotStyleModel(
        split=str(split),
        action_mean=action_mean.astype(float),
        action_chol=_safe_cholesky(action_cov),
        delta_chol=_safe_cholesky(delta_cov),
        action_min=action_min.astype(float),
        action_max=action_max.astype(float),
        segment_lengths=np.asarray(segment_lengths, dtype=np.int32),
        ripple_amp=ripple_amp.astype(float),
        feature_envelope=feature_envelope,
        level_bank=levels.astype(float),
        delta_bank=delta_values.astype(float),
        profile_mean=profile_mean.astype(float),
        profile_components=profile_components.astype(float),
        profile_coeff_min=profile_coeff_min.astype(float),
        profile_coeff_max=profile_coeff_max.astype(float),
        profile_coeff_std=profile_coeff_std.astype(float),
    )


def _fit_profile_model(
    actions_by_shot: list[np.ndarray],
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    profile_steps = 1000
    profiles: list[np.ndarray] = []
    target_u = np.linspace(0.0, 1.0, profile_steps, dtype=float)
    for action in actions_by_shot:
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 100:
            continue
        source_u = np.linspace(0.0, 1.0, action.shape[0], dtype=float)
        profile = np.stack(
            [np.interp(target_u, source_u, action[:, coil]) for coil in range(9)],
            axis=1,
        )
        profiles.append(profile.astype(float))
    if not profiles:
        mean = np.repeat(np.asarray(action_min, dtype=float)[None, :], profile_steps, axis=0)
        components = np.zeros((0, profile_steps, 9), dtype=float)
        empty = np.zeros((0,), dtype=float)
        return mean, components, empty, empty, empty

    stacked = np.stack(profiles, axis=0)
    mean = np.mean(stacked, axis=0)
    if stacked.shape[0] < 2:
        components = np.zeros((0, profile_steps, 9), dtype=float)
        empty = np.zeros((0,), dtype=float)
        return np.clip(mean, action_min[None, :], action_max[None, :]), components, empty, empty, empty

    centered = (stacked - mean[None, :, :]).reshape(stacked.shape[0], -1)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(3, vt.shape[0])
    basis = vt[:n_components]
    coeff = centered @ basis.T
    coeff_min = np.min(coeff, axis=0)
    coeff_max = np.max(coeff, axis=0)
    coeff_std = np.std(coeff, axis=0, ddof=1)
    coeff_span = np.maximum(coeff_max - coeff_min, 1.0e-6)
    coeff_min = coeff_min - 0.08 * coeff_span
    coeff_max = coeff_max + 0.08 * coeff_span
    components = basis.reshape(n_components, profile_steps, 9)
    mean = np.clip(mean, action_min[None, :], action_max[None, :])
    return mean, components, coeff_min, coeff_max, coeff_std


def _fit_jdot_feature_envelope(actions_by_shot: list[np.ndarray]) -> JdotFeatureEnvelope:
    feature_rows: list[np.ndarray] = []
    for action in actions_by_shot:
        action = np.asarray(action, dtype=float)
        if action.shape[0] < 100:
            continue
        lengths = [min(700, action.shape[0]), 1000, 1250, 1500]
        for length in sorted({int(v) for v in lengths if int(v) <= action.shape[0]}):
            if length <= 0:
                continue
            max_start = action.shape[0] - length
            if max_start <= 0:
                starts = [0]
            else:
                count = min(24, max_start + 1)
                starts = np.linspace(0, max_start, count, dtype=int).tolist()
            for start in starts:
                feature_rows.append(_jdot_feature_values(action[int(start) : int(start) + length]))
    if not feature_rows:
        raise ValueError("could not build real Jdot feature envelope")
    features = np.stack(feature_rows, axis=0)
    lo = np.min(features, axis=0)
    hi = np.max(features, axis=0)
    span = np.maximum(hi - lo, 1.0e-6)
    lo = np.maximum(lo - 0.02 * span, 0.0)
    hi = hi + 0.02 * span
    return JdotFeatureEnvelope(
        names=_jdot_feature_names(),
        min_values=lo.astype(float),
        max_values=hi.astype(float),
        samples=features.astype(float),
    )


def _jdot_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for prefix in ("rms", "max_abs", "mean_abs", "mean_abs_diff", "jump_rate", "sign_rate"):
        names.extend(f"{prefix}_{coil}" for coil in COIL_NAMES)
    names.extend(
        [
            "active_count",
            "pfc_rms_mean",
            "sol_rms_mean",
            "total_rms",
            "max_simultaneous_abs_gt_0p10",
            "max_simultaneous_abs_gt_0p25",
            "max_simultaneous_abs_gt_0p50",
        ]
    )
    return tuple(names)


def _jdot_feature_values(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=float)
    if action.ndim != 2 or action.shape[1] != 9:
        raise ValueError(f"expected Jdot action [T,9], got {action.shape}")
    duration_s = max(float(action.shape[0]) * 0.001, 0.001)
    diff = np.diff(action, axis=0) if action.shape[0] > 1 else np.zeros((0, 9), dtype=float)
    rms = np.sqrt(np.mean(action**2, axis=0))
    max_abs = np.max(np.abs(action), axis=0)
    mean_abs = np.mean(np.abs(action), axis=0)
    mean_abs_diff = np.mean(np.abs(diff), axis=0) if diff.shape[0] else np.zeros((9,), dtype=float)
    jump_rate = np.sum(np.abs(diff) > 0.06, axis=0) / duration_s if diff.shape[0] else np.zeros((9,), dtype=float)
    sign_rate = _block_sign_change_rate(action, block=25, duration_s=duration_s)
    simultaneous = np.abs(action)
    extras = np.asarray(
        [
            float(np.sum(rms > 0.035)),
            float(np.mean(rms[:6])),
            float(np.mean(rms[6:])),
            float(np.sqrt(np.mean(action**2))),
            float(np.max(np.sum(simultaneous > 0.10, axis=1))),
            float(np.max(np.sum(simultaneous > 0.25, axis=1))),
            float(np.max(np.sum(simultaneous > 0.50, axis=1))),
        ],
        dtype=float,
    )
    return np.concatenate([rms, max_abs, mean_abs, mean_abs_diff, jump_rate, sign_rate, extras], axis=0).astype(float)


def _block_sign_change_rate(action: np.ndarray, *, block: int, duration_s: float) -> np.ndarray:
    n_blocks = int(action.shape[0]) // int(block)
    if n_blocks < 2:
        return np.zeros((9,), dtype=float)
    blocks = action[: n_blocks * int(block)].reshape(n_blocks, int(block), 9).mean(axis=1)
    signs = np.sign(blocks)
    signs[np.abs(blocks) < 0.025] = 0.0
    changes = np.zeros((9,), dtype=float)
    for coil in range(9):
        vals = signs[:, coil]
        prev = 0.0
        count = 0
        for value in vals:
            if value == 0.0:
                continue
            if prev != 0.0 and value != prev:
                count += 1
            prev = value
        changes[coil] = float(count) / max(float(duration_s), 0.001)
    return changes


def _jdot_features_ok(action: np.ndarray, envelope: JdotFeatureEnvelope) -> tuple[bool, str]:
    features = _jdot_feature_values(action)
    tol = 1.0e-8
    lower_mask = np.asarray([_jdot_feature_has_lower_bound(name) for name in envelope.names], dtype=bool)
    below = (features < envelope.min_values - tol) & lower_mask
    above = features > envelope.max_values + tol
    if not bool(np.any(below) or np.any(above)):
        return True, "ok"
    names = envelope.names
    if bool(np.any(above)):
        violation = np.where(above)[0]
        span = np.maximum(envelope.max_values - envelope.min_values, 1.0e-6)
        idx = int(violation[np.argmax((features[violation] - envelope.max_values[violation]) / span[violation])])
        return False, f"jdot_feature_high:{names[idx]}"
    violation = np.where(below)[0]
    span = np.maximum(envelope.max_values - envelope.min_values, 1.0e-6)
    idx = int(violation[np.argmax((envelope.min_values[violation] - features[violation]) / span[violation])])
    return False, f"jdot_feature_low:{names[idx]}"


def _jdot_feature_has_lower_bound(name: str) -> bool:
    return (
        name.startswith("rms_")
        or name.startswith("mean_abs_diff_")
        or name == "jump_rate_SOL1"
        or name in {"active_count", "pfc_rms_mean", "sol_rms_mean", "total_rms"}
    )


def _jdot_feature_score(action: np.ndarray, envelope: JdotFeatureEnvelope) -> float:
    features = _jdot_feature_values(action)
    span = np.maximum(envelope.max_values - envelope.min_values, 1.0e-6)
    tol = 1.0e-8
    lower_mask = np.asarray([_jdot_feature_has_lower_bound(name) for name in envelope.names], dtype=bool)
    low = np.where(lower_mask, np.maximum(envelope.min_values - features - tol, 0.0) / span, 0.0)
    high = np.maximum(features - envelope.max_values - tol, 0.0) / span
    return float(np.max(np.maximum(low, high)))


def _regularized_cov(values: np.ndarray, *, diagonal_floor: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 9:
        raise ValueError(f"expected [N,9] covariance values, got {values.shape}")
    if values.shape[0] < 2:
        return np.eye(9, dtype=float) * float(diagonal_floor)
    cov = np.cov(values, rowvar=False)
    diag = np.maximum(np.diag(cov), float(diagonal_floor))
    cov = 0.75 * cov + 0.25 * np.diag(diag)
    cov += np.eye(9, dtype=float) * float(diagonal_floor)
    return cov.astype(float)


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    jitter = 1.0e-8
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + np.eye(cov.shape[0]) * jitter)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.cholesky(np.diag(np.maximum(np.diag(cov), 1.0e-4)))


def _generate_synthetic_ladder_jdot_action(
    *,
    style: JdotStyleModel,
    rng: np.random.Generator,
    steps: int,
    scale: float,
    residual_multiplier: float,
) -> np.ndarray:
    best: np.ndarray | None = None
    best_score = float("inf")
    for attempt in range(240):
        spread = max(0.20, 0.48 * (0.986 ** attempt))
        residual_scale = float(residual_multiplier) * max(0.25, 0.80 * (0.990 ** attempt))
        candidate = _generate_synthetic_ladder_jdot_action_once(
            style=style,
            rng=rng,
            steps=int(steps),
            scale=float(scale),
            residual_multiplier=residual_scale,
            spread=float(spread),
        )
        score = _jdot_feature_score(candidate, style.feature_envelope)
        if score <= 0.0:
            return candidate
        if score < best_score:
            best = candidate
            best_score = score
    if best is None:
        raise RuntimeError("failed to generate a synthetic Jdot action")
    return best.astype(float)


def _generate_synthetic_ladder_jdot_action_once(
    *,
    style: JdotStyleModel,
    rng: np.random.Generator,
    steps: int,
    scale: float,
    residual_multiplier: float,
    spread: float,
) -> np.ndarray:
    if style.profile_components.shape[0] > 0:
        action = _generate_profile_jdot_action(
            style=style,
            rng=rng,
            steps=int(steps),
            scale=float(scale),
            spread=float(spread),
        )
    else:
        action = np.zeros((int(steps), 9), dtype=float)
        cursor = 0
        level = _sample_level(style, rng, scale=float(scale), spread=float(spread))
        segment_id = 0
        while cursor < int(steps):
            length = int(style.segment_lengths[int(rng.integers(0, style.segment_lengths.shape[0]))])
            length = int(np.clip(length + int(rng.integers(-20, 21)), 90, 430))
            end = min(int(steps), cursor + length)
            action[cursor:end, :] = level[None, :]
            cursor = end
            segment_id += 1
            if cursor >= int(steps):
                break
            if segment_id == 1 or float(rng.random()) < 0.10:
                level = _sample_level(style, rng, scale=float(scale), spread=float(spread))
            else:
                delta = _sample_observed_delta(style, rng) * float(rng.uniform(0.70, 1.20))
                delta += _sample_zero_mean(style.delta_chol, rng) * float(rng.uniform(0.04, 0.12)) * float(spread)
                level = np.clip(level + delta, style.action_min, style.action_max)
    action += _synthetic_residual(action.shape[0], style=style, rng=rng, multiplier=float(residual_multiplier))
    action = _match_real_jdot_activity_envelope(action, envelope=style.feature_envelope, rng=rng)
    return np.clip(action, -0.98, 0.98).astype(float)


def _generate_profile_jdot_action(
    *,
    style: JdotStyleModel,
    rng: np.random.Generator,
    steps: int,
    scale: float,
    spread: float,
) -> np.ndarray:
    profile_steps = int(style.profile_mean.shape[0])
    source_u = np.linspace(0.0, 1.0, profile_steps, dtype=float)
    target_u = np.linspace(0.0, 1.0, int(steps), dtype=float)
    gamma = float(rng.uniform(0.92, 1.08))
    warped_u = np.clip(target_u**gamma, 0.0, 1.0)

    profile = style.profile_mean.copy()
    for idx in range(style.profile_components.shape[0]):
        lo = float(style.profile_coeff_min[idx])
        hi = float(style.profile_coeff_max[idx])
        std = float(style.profile_coeff_std[idx])
        if not np.isfinite(lo + hi + std) or hi <= lo:
            continue
        coeff = float(rng.uniform(lo, hi))
        coeff += float(rng.normal(0.0, 0.08 * max(std, 1.0e-6))) * float(spread)
        coeff = float(np.clip(coeff, lo, hi))
        profile += coeff * style.profile_components[idx]

    action = np.stack(
        [np.interp(warped_u, source_u, profile[:, coil]) for coil in range(9)],
        axis=1,
    )
    coil_scale = rng.normal(1.0, 0.035 * float(spread), size=(9,))
    coil_scale = np.clip(coil_scale, 0.90, 1.10)
    action *= coil_scale[None, :] * float(scale)
    return np.clip(action, style.action_min[None, :], style.action_max[None, :])


def _sample_level(style: JdotStyleModel, rng: np.random.Generator, *, scale: float, spread: float) -> np.ndarray:
    idx = int(rng.integers(0, style.level_bank.shape[0]))
    value = style.level_bank[idx].astype(float).copy()
    value += _sample_zero_mean(style.action_chol, rng) * float(rng.uniform(0.03, 0.10)) * float(spread)
    return np.clip(value * float(scale), style.action_min, style.action_max)


def _sample_observed_delta(style: JdotStyleModel, rng: np.random.Generator) -> np.ndarray:
    if style.delta_bank.shape[0] == 0:
        return np.zeros((9,), dtype=float)
    idx = int(rng.integers(0, style.delta_bank.shape[0]))
    return style.delta_bank[idx].astype(float)


def _match_real_jdot_activity_envelope(
    action: np.ndarray,
    *,
    envelope: JdotFeatureEnvelope,
    rng: np.random.Generator,
) -> np.ndarray:
    action = np.asarray(action, dtype=float).copy()
    target = envelope.samples[int(rng.integers(0, envelope.samples.shape[0]))]
    for coil, name in enumerate(COIL_NAMES):
        rms_lo, rms_hi = _feature_range(envelope, f"rms_{name}")
        rms_idx = envelope.names.index(f"rms_{name}")
        target_rms = float(target[rms_idx]) * float(rng.uniform(0.92, 1.08))
        target_rms = float(np.clip(target_rms, rms_lo, rms_hi))
        current_rms = float(np.sqrt(np.mean(action[:, coil] ** 2)))
        if current_rms > 1.0e-9:
            action[:, coil] *= target_rms / current_rms
        mean_lo, mean_hi = _feature_range(envelope, f"mean_abs_{name}")
        current_mean = float(np.mean(np.abs(action[:, coil])))
        if current_mean > mean_hi and current_mean > 1.0e-9:
            action[:, coil] *= mean_hi / current_mean
        max_lo, max_hi = _feature_range(envelope, f"max_abs_{name}")
        current_max = float(np.max(np.abs(action[:, coil])))
        if current_max > max_hi and current_max > 1.0e-9:
            action[:, coil] *= max_hi / current_max
        current_rms = float(np.sqrt(np.mean(action[:, coil] ** 2)))
        if current_rms < rms_lo and current_rms > 1.0e-9:
            scale = min(rms_lo / current_rms, 1.25)
            action[:, coil] *= scale
        current_mean = float(np.mean(np.abs(action[:, coil])))
        if current_mean > mean_hi and current_mean > 1.0e-9:
            action[:, coil] *= mean_hi / current_mean
        current_max = float(np.max(np.abs(action[:, coil])))
        if current_max > max_hi and current_max > 1.0e-9:
            action[:, coil] *= max_hi / current_max
        # The maximum-action lower bound is diagnostic, not a hard requirement. It
        # would otherwise force every synthetic parent to hit every coil extreme.
        _ = max_lo
        _, diff_hi = _feature_range(envelope, f"mean_abs_diff_{name}")
        for _ in range(4):
            diff = np.diff(action[:, coil])
            mean_abs_diff = float(np.mean(np.abs(diff))) if diff.shape[0] else 0.0
            if mean_abs_diff <= diff_hi:
                break
            action[:, coil] = _smooth_1d_reflect(action[:, coil])
        current_mean = float(np.mean(np.abs(action[:, coil])))
        if current_mean > mean_hi and current_mean > 1.0e-9:
            action[:, coil] *= mean_hi / current_mean
    return action


def _feature_range(envelope: JdotFeatureEnvelope, name: str) -> tuple[float, float]:
    idx = envelope.names.index(name)
    return float(envelope.min_values[idx]), float(envelope.max_values[idx])


def _smooth_1d_reflect(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[0] < 3:
        return values.copy()
    padded = np.pad(values, (1, 1), mode="edge")
    return (0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]).astype(float)


def _sample_zero_mean(chol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(chol, dtype=float) @ rng.normal(size=(9,))


def _synthetic_residual(
    steps: int,
    *,
    style: JdotStyleModel,
    rng: np.random.Generator,
    multiplier: float,
) -> np.ndarray:
    if float(multiplier) <= 0.0:
        return np.zeros((int(steps), 9), dtype=float)
    t = np.arange(int(steps), dtype=float)
    residual = np.zeros((int(steps), 9), dtype=float)
    for coil in range(9):
        amp = float(style.ripple_amp[coil]) * float(multiplier) * float(rng.uniform(0.35, 1.0))
        if coil < 6:
            period = float(rng.integers(45, 121))
            noise_scale = 0.15
        else:
            period = float(rng.integers(18, 46))
            noise_scale = 0.35
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        wave = np.sin(2.0 * np.pi * t / period + phase)
        wave += 0.45 * np.sin(2.0 * np.pi * t / (period * 0.51) + 0.7 * phase)
        noise = rng.normal(0.0, amp * noise_scale, size=(int(steps),))
        residual[:, coil] = amp * wave + noise
    return residual


def _simulate_and_filter_parents(
    candidates: list[ParentCandidate],
    *,
    machine_config: Path,
    envelope: object,
    batch_size: int,
    angles: int,
    gpu_device: str,
    state_feature_distance_limit: float,
) -> tuple[list[ParentRollout], list[dict[str, object]]]:
    base._ensure_tokamak_sim_importable()
    from tokamak_control.core.batched_gpu_simulator import BatchedGpuTokamakSimulator
    from tokamak_control.io.config_io import load_config

    sim_cfg = load_config(machine_config)
    theta = np.linspace(-np.pi, np.pi, int(angles), endpoint=False, dtype=float)
    accepted: list[ParentRollout] = []
    rejected: list[dict[str, object]] = []
    groups: dict[int, list[ParentCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(int(candidate.action.shape[0]), []).append(candidate)
    processed = 0
    for steps, group in sorted(groups.items()):
        for start in range(0, len(group), int(batch_size)):
            batch = group[start : start + int(batch_size)]
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
            ip0 = np.asarray([c.reset.ip[0] for c in batch], dtype=float)
            current0 = np.stack([c.currents[0] for c in batch], axis=0)
            result = sim.reset(
                ip=ip0,
                pfc_currents=current0[:, : sim_cfg.pfc.n_coils],
                sol_currents=current0[:, sim_cfg.pfc.n_coils :],
            )
            ip_rows = [result.state.Ip.detach().cpu().numpy().astype(float)]
            radii_rows = [result.boundary.radii.detach().cpu().numpy().astype(float)]
            found_rows = [result.boundary.found.detach().cpu().numpy().astype(bool)]
            for step in range(steps):
                next_current = np.stack([c.currents[step + 1] for c in batch], axis=0)
                result = sim.step_currents(next_current)
                ip_rows.append(result.state.Ip.detach().cpu().numpy().astype(float))
                radii_rows.append(result.boundary.radii.detach().cpu().numpy().astype(float))
                found_rows.append(result.boundary.found.detach().cpu().numpy().astype(bool))
            ip_arr = np.stack(ip_rows, axis=1)
            radii_arr = np.stack(radii_rows, axis=1)
            found_arr = np.stack(found_rows, axis=1)
            for row, candidate in enumerate(batch):
                processed += 1
                ip = ip_arr[row].astype(float)
                radii = radii_arr[row].astype(float)
                found = found_arr[row].astype(bool)
                ok, reason, distance = base._rollout_ok(
                    ip=ip,
                    radii=radii,
                    found=found,
                    currents=candidate.currents,
                    envelope=envelope,
                    state_feature_distance_limit=float(state_feature_distance_limit),
                )
                if not ok:
                    rejected.append(_parent_reject(candidate, reason, distance))
                    print(
                        f"[long-parent-sim] processed={processed}/{len(candidates)} "
                        f"accepted={len(accepted)} rejected={len(rejected)} last={reason}",
                        flush=True,
                    )
                    continue
                accepted.append(
                    ParentRollout(
                        candidate=candidate,
                        ip=ip.astype(np.float32),
                        radii=radii.astype(np.float32),
                        found=found,
                        state_feature_distance=float(distance),
                    )
                )
                print(
                    f"[long-parent-sim] processed={processed}/{len(candidates)} "
                    f"accepted={len(accepted)} rejected={len(rejected)}",
                    flush=True,
                )
    return accepted, rejected


def _cut_parent_windows(
    parents: list[ParentRollout],
    *,
    window_steps: int,
    stride: int,
    envelope: object,
    state_feature_distance_limit: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for parent in parents:
        candidate = parent.candidate
        parent_steps = int(candidate.action.shape[0])
        for start in range(0, parent_steps - int(window_steps) + 1, int(stride)):
            end = start + int(window_steps)
            ip = parent.ip[start : end + 1]
            radii = parent.radii[start : end + 1]
            found = parent.found[start : end + 1]
            currents = candidate.currents[start : end + 1]
            action = candidate.action[start:end]
            ok, reason, distance = base._rollout_ok(
                ip=ip,
                radii=radii,
                found=found,
                currents=currents,
                envelope=envelope,
                state_feature_distance_limit=float(state_feature_distance_limit),
            )
            if not ok:
                rejected.append(
                    {
                        "parent_id": candidate.parent_id,
                        "split": candidate.split,
                        "window_start": start,
                        "reason": reason,
                        "distance": float(distance),
                    }
                )
                continue
            rows.append(
                {
                    "shot_id": f"gen{candidate.parent_id:04d}",
                    "split": candidate.split,
                    "source_index": int(start),
                    "time_s": float(start) * 0.001,
                    "ip0": float(ip[0]),
                    "pfc0": currents[0, :6].astype(np.float32),
                    "sol0": currents[0, 6:].astype(np.float32),
                    "ip_target": ip.astype(np.float32),
                    "boundary_radii": radii.astype(np.float32),
                    "real_jdot_action": action.astype(np.float32),
                    "difficulty_bin": base._difficulty_bin(ip),
                    "mode": candidate.mode,
                    "motion_shot_id": f"parent{candidate.parent_id:04d}",
                    "motion_source_index": int(start),
                    "scale": float(candidate.scale),
                    "state_feature_distance": float(distance),
                    "oracle_ip_mean_error_a": 0.0,
                    "oracle_ip_max_error_a": 0.0,
                }
            )
    return rows, rejected


def _balanced_subsample_rows(
    rows: list[dict[str, object]],
    *,
    max_windows: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    if len(rows) <= int(max_windows):
        return rows
    train = [r for r in rows if r["split"] == "train"]
    holdout = [r for r in rows if r["split"] == "holdout"]
    holdout_n = min(len(holdout), max(1, int(round(int(max_windows) * len(holdout) / max(len(rows), 1)))))
    train_n = int(max_windows) - holdout_n
    selected: list[dict[str, object]] = []
    for group, count in ((train, train_n), (holdout, holdout_n)):
        if count <= 0:
            continue
        idx = rng.choice(len(group), size=min(count, len(group)), replace=False)
        selected.extend(group[int(i)] for i in idx.tolist())
    return sorted(selected, key=lambda r: (str(r["split"]), str(r["shot_id"]), int(r["source_index"])))


def _write_parent_rollouts(parents: list[ParentRollout], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_steps = max(int(p.ip.shape[0]) for p in parents)
    n = len(parents)
    ip = np.full((n, max_steps), np.nan, dtype=np.float32)
    mean_radius = np.full((n, max_steps), np.nan, dtype=np.float32)
    current = np.full((n, max_steps, 9), np.nan, dtype=np.float32)
    action_rms = np.full((n, max_steps - 1), np.nan, dtype=np.float32)
    lengths = np.zeros((n,), dtype=np.int32)
    for i, parent in enumerate(parents):
        length = int(parent.ip.shape[0])
        lengths[i] = length
        ip[i, :length] = parent.ip.astype(np.float32)
        mean_radius[i, :length] = np.nanmean(parent.radii, axis=1).astype(np.float32)
        current[i, :length, :] = parent.candidate.currents.astype(np.float32)
        action_rms[i, : length - 1] = np.sqrt(np.mean(parent.candidate.action**2, axis=1)).astype(np.float32)
    np.savez_compressed(
        path,
        schema=np.asarray("t15_actuator_long_generated_parent_rollouts_v1"),
        parent_id=np.asarray([p.candidate.parent_id for p in parents], dtype=np.int64),
        split=np.asarray([p.candidate.split for p in parents]),
        mode=np.asarray([p.candidate.mode for p in parents]),
        reset_shot_id=np.asarray([p.candidate.reset.shot_id for p in parents]),
        reset_source_index=np.asarray([p.candidate.reset.start for p in parents], dtype=np.int64),
        style_source=np.asarray([p.candidate.style_source for p in parents]),
        length=lengths,
        ip=ip,
        mean_radius=mean_radius,
        current=current,
        action_rms=action_rms,
    )


def _write_parent_rejections(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "parent_id",
        "split",
        "reset_shot_id",
        "reset_source_index",
        "style_source",
        "steps",
        "mode",
        "reason",
        "distance",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_window_rejections(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["parent_id", "split", "window_start", "reason", "distance"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _parent_reject(candidate: ParentCandidate, reason: str, distance: float) -> dict[str, object]:
    return {
        "parent_id": candidate.parent_id,
        "split": candidate.split,
        "reset_shot_id": candidate.reset.shot_id,
        "reset_source_index": candidate.reset.start,
        "style_source": candidate.style_source,
        "steps": int(candidate.action.shape[0]),
        "mode": candidate.mode,
        "reason": reason,
        "distance": float(distance),
    }


def _summary(
    *,
    parents: list[ParentRollout],
    parent_rejections: list[dict[str, object]],
    window_rows: list[dict[str, object]],
    window_rejections: list[dict[str, object]],
    envelope: object,
    limits: object,
    args: argparse.Namespace,
) -> dict[str, object]:
    parent_splits = Counter(p.candidate.split for p in parents)
    window_splits = Counter(str(r["split"]) for r in window_rows)
    bins = Counter(str(r["difficulty_bin"]) for r in window_rows)
    modes = Counter(str(r["mode"]) for r in window_rows)
    parent_reasons = Counter(str(r["reason"]) for r in parent_rejections)
    window_reasons = Counter(str(r["reason"]) for r in window_rejections)
    ip = np.stack([r["ip_target"] for r in window_rows], axis=0)
    radii = np.stack([r["boundary_radii"] for r in window_rows], axis=0)
    action = np.stack([r["real_jdot_action"] for r in window_rows], axis=0)
    return {
        "schema": "t15_actuator_long_generated_trim50_plain_gpu1e6_0p1s_summary_v1",
        "accepted_parents": int(len(parents)),
        "rejected_parents": int(len(parent_rejections)),
        "accepted_windows": int(len(window_rows)),
        "rejected_windows": int(len(window_rejections)),
        "parent_split_counts": dict(sorted(parent_splits.items())),
        "window_split_counts": dict(sorted(window_splits.items())),
        "mode_counts": dict(sorted(modes.items())),
        "difficulty_bins": dict(sorted(bins.items())),
        "parent_rejection_reasons": dict(sorted(parent_reasons.items())),
        "window_rejection_reasons": dict(sorted(window_reasons.items())),
        "parent_length_steps": {
            "min": int(min(p.ip.shape[0] - 1 for p in parents)),
            "max": int(max(p.ip.shape[0] - 1 for p in parents)),
            "mean": float(np.mean([p.ip.shape[0] - 1 for p in parents])),
        },
        "ip_target_min_a": float(np.nanmin(ip)),
        "ip_target_max_a": float(np.nanmax(ip)),
        "boundary_radii_min_m": float(np.nanmin(radii)),
        "boundary_radii_max_m": float(np.nanmax(radii)),
        "max_abs_normalized_action": float(np.nanmax(np.abs(action))),
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
            "parent_count": int(args.parent_count),
            "parent_min_steps": int(args.parent_min_steps),
            "parent_max_steps": int(args.parent_max_steps),
            "parent_lengths": [int(v) for v in args.parent_lengths],
            "window_steps": int(args.steps),
            "window_stride": int(args.window_stride),
            "max_windows": int(args.max_windows),
            "seed": int(args.seed),
            "level_scale_min": float(args.level_scale_min),
            "level_scale_max": float(args.level_scale_max),
            "residual_action_rms": float(args.residual_action_rms),
        },
    }


def _write_plots(*, parents: list[ParentRollout], rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "summary"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(321)
    sample = rng.choice(len(parents), size=min(10, len(parents)), replace=False)
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=False, constrained_layout=True)
    for idx in sample.tolist():
        parent = parents[int(idx)]
        t = np.arange(parent.ip.shape[0], dtype=float) * 0.001
        currents = parent.candidate.currents
        axes[0].plot(t, parent.ip, lw=1.2, alpha=0.8, label=f"parent {parent.candidate.parent_id}")
        axes[1].plot(t, np.nanmean(parent.radii, axis=1), lw=1.2, alpha=0.8)
        axes[2].plot(t, currents[:, 0], lw=1.0, alpha=0.7)
        axes[3].plot(t, currents[:, 6], lw=1.0, alpha=0.7)
        axes[4].plot(t[:-1], np.sqrt(np.mean(parent.candidate.action**2, axis=1)), lw=1.0, alpha=0.7)
    axes[0].set_ylabel("Ip [A]")
    axes[1].set_ylabel("Mean radius [m]")
    axes[2].set_ylabel("PFC0 [A]")
    axes[3].set_ylabel("SOL0 [A]")
    axes[4].set_ylabel("Jdot action RMS")
    axes[4].set_xlabel("time [s]")
    axes[0].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.savefig(plot_dir / "sample_long_parent_rollouts.png", dpi=150)
    plt.close(fig)

    base._write_plots(rows, out_dir=plot_dir)

    ip_delta = np.asarray([float(np.asarray(r["ip_target"])[-1] - np.asarray(r["ip_target"])[0]) for r in rows])
    mean_r_delta = np.asarray(
        [
            float(np.nanmean(np.asarray(r["boundary_radii"])[-1]) - np.nanmean(np.asarray(r["boundary_radii"])[0]))
            for r in rows
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].hist(ip_delta / 1000.0, bins=80)
    axes[0].set_xlabel("100-step window dIp [kA]")
    axes[1].hist(mean_r_delta, bins=80)
    axes[1].set_xlabel("100-step window d mean radius [m]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.savefig(plot_dir / "overlapping_window_delta_histograms.png", dpi=150)
    plt.close(fig)


def _write_report(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Actuator Long-Generated Dataset",
        "",
        f"- Accepted parents: {summary['accepted_parents']}",
        f"- Rejected parents: {summary['rejected_parents']}",
        f"- Accepted overlapping 100-step windows: {summary['accepted_windows']}",
        f"- Rejected windows: {summary['rejected_windows']}",
        f"- Parent split counts: `{summary['parent_split_counts']}`",
        f"- Window split counts: `{summary['window_split_counts']}`",
        f"- Difficulty bins: `{summary['difficulty_bins']}`",
        f"- Max normalized Jdot action: `{summary['max_abs_normalized_action']:.6g}`",
        "",
        "The parent rollouts are generated first, simulated through tokamak-sim, filtered against the observed real trim50 state envelope, and only then cut into dense overlapping windows.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
