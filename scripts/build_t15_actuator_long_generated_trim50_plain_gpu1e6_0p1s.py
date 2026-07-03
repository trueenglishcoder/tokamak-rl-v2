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
class ParentCandidate:
    parent_id: int
    split: str
    reset: object
    template: ShotSeries
    template_start: int
    mode: str
    scale: float
    currents: np.ndarray
    action: np.ndarray


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
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=0, help="0 keeps every accepted overlapping 100-step window.")
    parser.add_argument("--batch-size", type=int, default=1, help="Parent simulation is intentionally one parent at a time.")
    parser.add_argument("--angles", type=int, default=32)
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--ip-margin-a", type=float, default=15000.0)
    parser.add_argument("--radii-margin-m", type=float, default=0.05)
    parser.add_argument("--current-margin-fraction", type=float, default=0.03)
    parser.add_argument("--state-feature-distance-limit", type=float, default=0.0)
    parser.add_argument("--template-scale-min", type=float, default=0.88)
    parser.add_argument("--template-scale-max", type=float, default=1.12)
    parser.add_argument("--residual-action-rms", type=float, default=0.035)
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
    split_counts = base._split_counts(int(args.parent_count), train_windows=train_windows, holdout_windows=holdout_windows)
    candidates, candidate_rejections = _make_parent_candidates(
        train_windows=train_windows,
        holdout_windows=holdout_windows,
        shot_series=shot_series,
        split_counts=split_counts,
        limits=limits,
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
    shot_series: dict[str, ShotSeries],
    split_counts: dict[str, int],
    limits: object,
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
        split_series = [s for s in shot_series.values() if s.split == split]
        while len([c for c in candidates if c.split == split]) < count and attempts < max_attempts:
            attempts += 1
            reset = windows[int(rng.integers(0, len(windows)))]
            template = split_series[int(rng.integers(0, len(split_series)))]
            parent_steps = int(rng.integers(int(args.parent_min_steps), int(args.parent_max_steps) + 1))
            if template.currents.shape[0] <= parent_steps + 1:
                continue
            template_start = int(rng.integers(0, template.currents.shape[0] - parent_steps - 1))
            candidate, reason = _candidate_from_template(
                parent_id=parent_id,
                reset=reset,
                template=template,
                template_start=template_start,
                parent_steps=parent_steps,
                limits=limits,
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
                        "template_shot_id": template.shot_id,
                        "template_start": template_start,
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


def _candidate_from_template(
    *,
    parent_id: int,
    reset: object,
    template: ShotSeries,
    template_start: int,
    parent_steps: int,
    limits: object,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[ParentCandidate | None, str]:
    template_currents = template.currents[template_start : template_start + parent_steps + 1]
    if template_currents.shape != (parent_steps + 1, 9):
        return None, "bad_template_shape"
    template_delta = template_currents - template_currents[0:1]
    mode = "template_scaled"
    scale = float(rng.uniform(float(args.template_scale_min), float(args.template_scale_max)))
    per_coil_scale = np.clip(rng.normal(loc=scale, scale=0.025, size=(9,)), 0.82, 1.18)

    residual_action = _low_frequency_ladder_action(
        rng=rng,
        steps=parent_steps,
        rms=float(args.residual_action_rms),
    )
    if float(args.residual_action_rms) > 0.0:
        mode = "template_scaled_with_ladder_residual"

    residual_current = np.concatenate(
        [np.zeros((1, 9), dtype=float), np.cumsum(residual_action * limits.derivative_vector[None, :] * 0.001, axis=0)],
        axis=0,
    )
    currents = reset.currents[0:1] + template_delta * per_coil_scale[None, :] + residual_current
    if not np.all(np.isfinite(currents)):
        return None, "nonfinite_current"
    if np.any(np.abs(currents) > limits.current_vector[None, :]):
        return None, "current_limit"
    jdot = np.diff(currents, axis=0) / 0.001
    action = jdot / limits.derivative_vector[None, :]
    if float(np.nanmax(np.abs(action))) > 1.0001:
        return None, "derivative_limit"
    return (
        ParentCandidate(
            parent_id=int(parent_id),
            split=str(reset.split),
            reset=reset,
            template=template,
            template_start=int(template_start),
            mode=mode,
            scale=float(scale),
            currents=np.asarray(currents, dtype=float),
            action=np.asarray(action, dtype=float),
        ),
        "ok",
    )


def _low_frequency_ladder_action(*, rng: np.random.Generator, steps: int, rms: float) -> np.ndarray:
    if float(rms) <= 0.0:
        return np.zeros((int(steps), 9), dtype=float)
    action = np.zeros((int(steps), 9), dtype=float)
    start = 0
    current = rng.normal(0.0, float(rms), size=(9,))
    while start < int(steps):
        length = int(rng.integers(60, 181))
        end = min(int(steps), start + length)
        if start > 0:
            jump = rng.normal(0.0, float(rms) * 0.6, size=(9,))
            current = np.clip(current + jump, -3.0 * float(rms), 3.0 * float(rms))
        action[start:end, :] = current[None, :]
        start = end
    return np.asarray(action, dtype=float)


def _simulate_and_filter_parents(
    candidates: list[ParentCandidate],
    *,
    machine_config: Path,
    envelope: object,
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
    for idx, candidate in enumerate(candidates):
        sim = BatchedGpuTokamakSimulator(
            grid=sim_cfg.grid,
            pfc=sim_cfg.pfc,
            sol=sim_cfg.sol,
            settings=sim_cfg.physics,
            batch_size=1,
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
        result = sim.reset(
            ip=np.asarray([candidate.reset.ip[0]], dtype=float),
            pfc_currents=candidate.currents[0:1, : sim_cfg.pfc.n_coils],
            sol_currents=candidate.currents[0:1, sim_cfg.pfc.n_coils :],
        )
        ip_rows = [float(result.state.Ip.detach().cpu().numpy()[0])]
        radii_rows = [result.boundary.radii.detach().cpu().numpy()[0].astype(float)]
        found_rows = [bool(result.boundary.found.detach().cpu().numpy()[0])]
        for step in range(candidate.currents.shape[0] - 1):
            result = sim.step_currents(candidate.currents[step + 1 : step + 2])
            ip_rows.append(float(result.state.Ip.detach().cpu().numpy()[0]))
            radii_rows.append(result.boundary.radii.detach().cpu().numpy()[0].astype(float))
            found_rows.append(bool(result.boundary.found.detach().cpu().numpy()[0]))
        ip = np.asarray(ip_rows, dtype=float)
        radii = np.stack(radii_rows, axis=0).astype(float)
        found = np.asarray(found_rows, dtype=bool)
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
                f"[long-parent-sim] processed={idx + 1}/{len(candidates)} "
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
            f"[long-parent-sim] processed={idx + 1}/{len(candidates)} "
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
        template_shot_id=np.asarray([p.candidate.template.shot_id for p in parents]),
        template_start=np.asarray([p.candidate.template_start for p in parents], dtype=np.int64),
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
        "template_shot_id",
        "template_start",
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
        "template_shot_id": candidate.template.shot_id,
        "template_start": candidate.template_start,
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
            "window_steps": int(args.steps),
            "window_stride": int(args.window_stride),
            "max_windows": int(args.max_windows),
            "seed": int(args.seed),
            "template_scale_min": float(args.template_scale_min),
            "template_scale_max": float(args.template_scale_max),
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
