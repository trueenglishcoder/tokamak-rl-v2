#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class WindowCandidate:
    shot_id: str
    split: str
    source_index: int
    time_s: float
    ip_target: np.ndarray
    currents: np.ndarray
    real_jdot: np.ndarray
    normalized_action: np.ndarray
    difficulty_bin: str


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build coherent 0.1s T15 replay-window oracle targets.")
    parser.add_argument("--base-config", default="configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml")
    parser.add_argument("--data-root", default="../tokamak-sim/data/t15_data_new_trim50_ip_calibrated")
    parser.add_argument("--machine-config", default="../tokamak-sim/configs/T15MD.toml")
    parser.add_argument("--target-dir", default="data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets")
    parser.add_argument("--initial-library-out", default="data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz")
    parser.add_argument("--config-out", default="configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo.yaml")
    parser.add_argument("--train-shots", nargs="+", default=["3857", "3858", "3863"])
    parser.add_argument("--holdout-shots", nargs="+", default=["3856"])
    parser.add_argument("--window-steps", type=int, default=100)
    parser.add_argument("--angles", type=int, default=32)
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--compute-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-rows-per-shot", type=int, default=0, help="0 keeps every valid window start.")
    parser.add_argument("--oracle-mean-ip-error-a", type=float, default=10000.0)
    parser.add_argument("--oracle-max-ip-error-a", type=float, default=20000.0)
    args = parser.parse_args(argv)

    if int(args.window_steps) <= 0:
        raise ValueError("--window-steps must be positive")
    if int(args.angles) <= 0:
        raise ValueError("--angles must be positive")
    train_shots = tuple(str(int(v)) for v in args.train_shots)
    holdout_shots = tuple(str(int(v)) for v in args.holdout_shots)
    overlap = sorted(set(train_shots) & set(holdout_shots))
    if overlap:
        raise ValueError(f"train and holdout shots overlap: {overlap}")

    data_root = _repo_path(args.data_root)
    target_dir = _repo_path(args.target_dir)
    initial_library_out = _repo_path(args.initial_library_out)
    machine_config = _repo_path(args.machine_config)
    base_config = _repo_path(args.base_config)
    config_out = _repo_path(args.config_out)

    _ensure_tokamak_sim_importable()
    from tokamak_control.core.batched_gpu_simulator import BatchedGpuTokamakSimulator
    from tokamak_control.core.plasma_model import PlasmaModel
    from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
    from tokamak_control.geometry.legacy_metrics import legacy_radii_at_angles
    from tokamak_control.io.config_io import load_config
    from dataclasses import replace

    sim_cfg = load_config(machine_config)
    dt = float(sim_cfg.physics.t_step)
    if not np.isclose(dt, 0.001, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"expected T15 replay-window dt=0.001, got {dt}")
    if sim_cfg.limiter_shape is None:
        raise ValueError("oracle target building requires limiter geometry")

    base = json.loads(base_config.read_text(encoding="utf-8"))
    current_limits = _current_limit_vector(base, n_pfc=sim_cfg.pfc.n_coils, n_sol=sim_cfg.sol.n_coils)
    derivative_limits = np.concatenate(
        [
            np.full((sim_cfg.pfc.n_coils,), float(sim_cfg.physics.pfc_deriv_limit), dtype=float),
            np.full((sim_cfg.sol.n_coils,), float(sim_cfg.physics.sol_deriv_limit), dtype=float),
        ]
    )
    if not np.all(np.isfinite(derivative_limits)) or np.any(derivative_limits <= 0.0):
        raise ValueError("machine config derivative limits must be finite and positive")

    candidates, pre_rejected = _load_candidates(
        data_root=data_root,
        train_shots=train_shots,
        holdout_shots=holdout_shots,
        window_steps=int(args.window_steps),
        delimiter=str(args.delimiter),
        max_rows_per_shot=int(args.max_rows_per_shot),
        current_limits=current_limits,
        derivative_limits=derivative_limits,
        dt=dt,
    )
    if not candidates:
        raise ValueError("no candidate replay windows survived preflight checks")

    target_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = target_dir / "oracle_rejected_windows.csv"
    summary_path = target_dir / "oracle_summary.json"
    oracle_path = target_dir / "t15_replay_window_oracle_targets.npz"

    if str(args.compute_backend) == "gpu":
        accepted, sim_rejected = _simulate_gpu(
            candidates,
            sim_cfg=sim_cfg,
            batch_size=int(args.batch_size),
            angles=int(args.angles),
            gpu_device=str(args.gpu_device),
            mean_ip_limit=float(args.oracle_mean_ip_error_a),
            max_ip_limit=float(args.oracle_max_ip_error_a),
            simulator_cls=BatchedGpuTokamakSimulator,
        )
    else:
        accepted, sim_rejected = _simulate_cpu(
            candidates,
            sim_cfg=sim_cfg,
            angles=int(args.angles),
            mean_ip_limit=float(args.oracle_mean_ip_error_a),
            max_ip_limit=float(args.oracle_max_ip_error_a),
            model_cls=PlasmaModel,
            replace=replace,
            find_boundary=find_plasma_boundary_with_status,
            radii_at_angles=legacy_radii_at_angles,
            boundary_error_cls=BoundaryNotFoundError,
        )
    rejected = [*pre_rejected, *sim_rejected]
    if not accepted:
        _write_rejections(rejected_path, rejected)
        raise ValueError("no replay windows survived oracle simulation/audit")

    _write_oracle_npz(oracle_path, accepted, current_limits=current_limits, derivative_limits=derivative_limits)
    _write_initial_library(initial_library_out, accepted)
    _write_config(
        base=base,
        config_out=config_out,
        machine_config=machine_config,
        target_dir=target_dir,
        initial_library=initial_library_out,
        train_output="outputs/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_oracle",
        balanced=False,
    )
    balanced_config_out = config_out.with_name(config_out.stem + "_balanced.yaml")
    _write_config(
        base=base,
        config_out=balanced_config_out,
        machine_config=machine_config,
        target_dir=target_dir,
        initial_library=initial_library_out,
        train_output="outputs/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle",
        balanced=True,
    )
    _write_rejections(rejected_path, rejected)
    summary = _summary(accepted, rejected, target_dir=target_dir, initial_library=initial_library_out, oracle_path=oracle_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "oracle": str(oracle_path),
                "initial_library": str(initial_library_out),
                "config": str(config_out),
                "balanced_config": str(balanced_config_out),
                "summary": str(summary_path),
                "accepted": summary["accepted_windows"],
                "rejected": summary["rejected_windows"],
                "difficulty_bins": summary["difficulty_bins"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _ensure_tokamak_sim_importable() -> None:
    sim_root = (ROOT.parent / "tokamak-sim").resolve()
    if str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))


def _current_limit_vector(base: dict[str, object], *, n_pfc: int, n_sol: int) -> np.ndarray:
    sim = base.get("sim", {})
    if not isinstance(sim, dict):
        raise ValueError("base config sim must be a mapping")
    limits = sim.get("current_safety_limits", {})
    if not isinstance(limits, dict):
        raise ValueError("base config current_safety_limits must be a mapping")
    pfc = limits.get("pfc_currents")
    sol = limits.get("sol_currents")
    pfc_values = _ordered_values(pfc, expected=n_pfc, name="pfc_currents")
    sol_values = _ordered_values(sol, expected=n_sol, name="sol_currents")
    return np.concatenate([pfc_values, sol_values]).astype(float)


def _ordered_values(raw: object, *, expected: int, name: str) -> np.ndarray:
    if isinstance(raw, dict):
        values = [raw[k] for k in sorted(raw)]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError(f"{name} must be a mapping or list")
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape != (int(expected),):
        raise ValueError(f"{name} must contain {expected} values, got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} values must be finite and positive")
    return arr


def _load_candidates(
    *,
    data_root: Path,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
    window_steps: int,
    delimiter: str,
    max_rows_per_shot: int,
    current_limits: np.ndarray,
    derivative_limits: np.ndarray,
    dt: float,
) -> tuple[list[WindowCandidate], list[dict[str, object]]]:
    accepted: list[WindowCandidate] = []
    rejected: list[dict[str, object]] = []
    all_shots = tuple(sorted(set(train_shots) | set(holdout_shots), key=int))
    for shot in all_shots:
        split = "holdout" if shot in holdout_shots else "train"
        ip_path = data_root / "ip" / f"t15md_{shot}_ip.csv"
        coil_path = data_root / "coils" / f"t15md_{shot}_coils.csv"
        if not ip_path.exists() or not coil_path.exists():
            rejected.append(_reject(shot, -1, float("nan"), "missing_ip_or_coil_csv"))
            continue
        ip_table = _load_table(ip_path, delimiter=delimiter)
        coil_table = _load_table(coil_path, delimiter=delimiter)
        if ip_table.shape[1] < 2:
            raise ValueError(f"{ip_path} must have time and Ip columns")
        if coil_table.shape[1] != 10:
            raise ValueError(f"{coil_path} must have time + 3 SOL + 6 PFC columns")
        valid_end = int(ip_table.shape[0]) - int(window_steps)
        starts = np.arange(0, max(valid_end, 0), dtype=np.int64)
        if int(max_rows_per_shot) > 0 and starts.size > int(max_rows_per_shot):
            keep = np.linspace(0, starts.size - 1, int(max_rows_per_shot)).round().astype(np.int64)
            starts = starts[keep]
        for start in starts:
            times = ip_table[int(start) : int(start) + int(window_steps) + 1, 0]
            ip_target = ip_table[int(start) : int(start) + int(window_steps) + 1, 1]
            if times.shape[0] != int(window_steps) + 1 or not np.all(np.isfinite(ip_target)):
                rejected.append(_reject(shot, int(start), float(ip_table[int(start), 0]), "bad_ip_window"))
                continue
            if np.max(np.abs(np.diff(times) - dt)) > 1.0e-6:
                rejected.append(_reject(shot, int(start), float(times[0]), "bad_time_step"))
                continue
            currents = _currents_at_times(coil_table, times)
            if not np.all(np.isfinite(currents)):
                rejected.append(_reject(shot, int(start), float(times[0]), "bad_current_window"))
                continue
            if np.any(np.abs(currents) > current_limits[None, :]):
                rejected.append(_reject(shot, int(start), float(times[0]), "oracle_current_violation"))
                continue
            real_jdot = np.diff(currents, axis=0) / dt
            if np.any(np.abs(real_jdot) > derivative_limits[None, :]):
                rejected.append(_reject(shot, int(start), float(times[0]), "real_jdot_exceeds_action_limits"))
                continue
            norm_action = real_jdot / derivative_limits[None, :]
            accepted.append(
                WindowCandidate(
                    shot_id=str(shot),
                    split=split,
                    source_index=int(start),
                    time_s=float(times[0]),
                    ip_target=np.asarray(ip_target, dtype=float),
                    currents=np.asarray(currents, dtype=float),
                    real_jdot=np.asarray(real_jdot, dtype=float),
                    normalized_action=np.asarray(norm_action, dtype=float),
                    difficulty_bin=_difficulty_bin(float(ip_target[-1] - ip_target[0])),
                )
            )
        print(f"[oracle-preflight] shot={shot} split={split} candidates_so_far={len(accepted)}", flush=True)
    return accepted, rejected


def _load_table(path: Path, *, delimiter: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=delimiter, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"{path} must be a 2D table with at least two rows")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path} contains non-finite values")
    if np.any(np.diff(arr[:, 0]) <= 0.0):
        raise ValueError(f"{path} time column must be strictly increasing")
    return arr


def _currents_at_times(coil_table: np.ndarray, times: np.ndarray) -> np.ndarray:
    # Project convention: coil CSV columns are time + SOL0..SOL2 + PFC0..PFC5.
    sol = np.stack([np.interp(times, coil_table[:, 0], coil_table[:, col]) for col in (1, 2, 3)], axis=1)
    pfc = np.stack([np.interp(times, coil_table[:, 0], coil_table[:, col]) for col in (4, 5, 6, 7, 8, 9)], axis=1)
    return np.concatenate([pfc, sol], axis=1)


def _difficulty_bin(delta_ip: float) -> str:
    mag = abs(float(delta_ip))
    if mag < 10000.0:
        return "flat"
    direction = "up" if float(delta_ip) > 0.0 else "down"
    if mag < 40000.0:
        return f"medium_{direction}"
    return f"fast_{direction}"


def _simulate_gpu(
    candidates: list[WindowCandidate],
    *,
    sim_cfg,
    batch_size: int,
    angles: int,
    gpu_device: str,
    mean_ip_limit: float,
    max_ip_limit: float,
    simulator_cls,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    theta = np.linspace(-np.pi, np.pi, int(angles), endpoint=False, dtype=float)
    for start in range(0, len(candidates), int(batch_size)):
        batch = candidates[start : start + int(batch_size)]
        B = len(batch)
        sim = simulator_cls(
            grid=sim_cfg.grid,
            pfc=sim_cfg.pfc,
            sol=sim_cfg.sol,
            settings=sim_cfg.physics,
            batch_size=B,
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
        ip0 = np.asarray([c.ip_target[0] for c in batch], dtype=float)
        currents0 = np.stack([c.currents[0] for c in batch], axis=0)
        result = sim.reset(ip=ip0, pfc_currents=currents0[:, : sim_cfg.pfc.n_coils], sol_currents=currents0[:, sim_cfg.pfc.n_coils :])
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
        _collect_simulated(
            batch,
            sim_ip_arr,
            radii_arr,
            found_arr,
            accepted,
            rejected,
            n_pfc=sim_cfg.pfc.n_coils,
            mean_ip_limit=mean_ip_limit,
            max_ip_limit=max_ip_limit,
        )
        print(f"[oracle-sim-gpu] processed={min(start + B, len(candidates))}/{len(candidates)} accepted={len(accepted)} rejected={len(rejected)}", flush=True)
    return accepted, rejected


def _simulate_cpu(
    candidates: list[WindowCandidate],
    *,
    sim_cfg,
    angles: int,
    mean_ip_limit: float,
    max_ip_limit: float,
    model_cls,
    replace,
    find_boundary,
    radii_at_angles,
    boundary_error_cls,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    theta = np.linspace(-np.pi, np.pi, int(angles), endpoint=False, dtype=float)
    for row, candidate in enumerate(candidates):
        pfc = sim_cfg.pfc.__class__(name=sim_cfg.pfc.name, coils=list(sim_cfg.pfc.coils), currents=candidate.currents[0, : sim_cfg.pfc.n_coils])
        sol = sim_cfg.sol.__class__(name=sim_cfg.sol.name, coils=list(sim_cfg.sol.coils), currents=candidate.currents[0, sim_cfg.pfc.n_coils :])
        model = model_cls.from_settings(
            grid=sim_cfg.grid,
            pfc=pfc,
            sol=sol,
            settings=sim_cfg.physics,
            ip0=float(candidate.ip_target[0]),
        )
        prev_poly = None
        prev_level = None
        sim_ip: list[float] = []
        radii: list[np.ndarray] = []
        found: list[bool] = []
        for k in range(candidate.currents.shape[0]):
            try:
                poly, level, _status = find_boundary(
                    model.state.psi,
                    model.grid,
                    (model.R0, model.Z0),
                    n_levels=80,
                    prev_level=prev_level,
                    prev_poly=prev_poly,
                    limiter_shape=sim_cfg.limiter_shape,
                    boundary_mode=sim_cfg.boundary_mode,
                    boundary_base_mode=sim_cfg.boundary_base_mode,
                    level_smoothing_alpha=sim_cfg.boundary_level_smoothing_alpha,
                    level_search_span_fraction=sim_cfg.boundary_level_search_span_fraction,
                    continuity_weight_radii=sim_cfg.boundary_continuity_weight_radii,
                    continuity_weight_mean_radius=sim_cfg.boundary_continuity_weight_mean_radius,
                    continuity_weight_center=sim_cfg.boundary_continuity_weight_center,
                    continuity_weight_area=sim_cfg.boundary_continuity_weight_area,
                    continuity_weight_level=sim_cfg.boundary_continuity_weight_level,
                )
                prev_poly = poly
                prev_level = level
                radii.append(np.asarray(radii_at_angles(poly, (model.R0, model.Z0), theta), dtype=float))
                found.append(True)
            except boundary_error_cls:
                radii.append(np.zeros((int(angles),), dtype=float))
                found.append(False)
            sim_ip.append(float(model.state.Ip))
            if k < candidate.currents.shape[0] - 1:
                model.step_currents(
                    pfc_currents_next=candidate.currents[k + 1, : sim_cfg.pfc.n_coils],
                    sol_currents_next=candidate.currents[k + 1, sim_cfg.pfc.n_coils :],
                )
        _collect_simulated(
            [candidate],
            np.asarray(sim_ip, dtype=float).reshape(1, -1),
            np.asarray(radii, dtype=float).reshape(1, candidate.currents.shape[0], int(angles)),
            np.asarray(found, dtype=bool).reshape(1, candidate.currents.shape[0]),
            accepted,
            rejected,
            n_pfc=sim_cfg.pfc.n_coils,
            mean_ip_limit=mean_ip_limit,
            max_ip_limit=max_ip_limit,
        )
        if (row + 1) % 100 == 0:
            print(f"[oracle-sim-cpu] processed={row + 1}/{len(candidates)} accepted={len(accepted)} rejected={len(rejected)}", flush=True)
    return accepted, rejected


def _collect_simulated(
    candidates: list[WindowCandidate],
    sim_ip: np.ndarray,
    boundary_radii: np.ndarray,
    boundary_found: np.ndarray,
    accepted: list[dict[str, object]],
    rejected: list[dict[str, object]],
    *,
    n_pfc: int,
    mean_ip_limit: float,
    max_ip_limit: float,
) -> None:
    for b, candidate in enumerate(candidates):
        found = np.asarray(boundary_found[b], dtype=bool)
        if not np.all(found):
            rejected.append(_reject(candidate.shot_id, candidate.source_index, candidate.time_s, "oracle_boundary_lost"))
            continue
        ip_err = np.abs(np.asarray(sim_ip[b], dtype=float) - np.asarray(candidate.ip_target, dtype=float))
        mean_err = float(np.mean(ip_err))
        max_err = float(np.max(ip_err))
        if mean_err > float(mean_ip_limit) or max_err > float(max_ip_limit):
            rejected.append(_reject(candidate.shot_id, candidate.source_index, candidate.time_s, f"oracle_ip_error_mean_{mean_err:.1f}_max_{max_err:.1f}"))
            continue
        radii = np.asarray(boundary_radii[b], dtype=float)
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            rejected.append(_reject(candidate.shot_id, candidate.source_index, candidate.time_s, "oracle_bad_boundary_radii"))
            continue
        accepted.append(
            {
                "shot_id": candidate.shot_id,
                "split": candidate.split,
                "source_index": candidate.source_index,
                "time_s": candidate.time_s,
                "ip0": float(candidate.ip_target[0]),
                "pfc0": candidate.currents[0, : int(n_pfc)].astype(np.float32),
                "sol0": candidate.currents[0, int(n_pfc) :].astype(np.float32),
                "ip_target": candidate.ip_target.astype(np.float32),
                "boundary_radii": radii.astype(np.float32),
                "real_jdot_action": candidate.normalized_action.astype(np.float32),
                "difficulty_bin": candidate.difficulty_bin,
                "oracle_ip_mean_error_a": mean_err,
                "oracle_ip_max_error_a": max_err,
            }
        )


def _write_oracle_npz(path: Path, rows: list[dict[str, object]], *, current_limits: np.ndarray, derivative_limits: np.ndarray) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray(["t15_replay_window_oracle_targets_v1"]),
        shot_id=np.asarray([r["shot_id"] for r in rows]),
        split=np.asarray([r["split"] for r in rows]),
        source_index=np.asarray([r["source_index"] for r in rows], dtype=np.int64),
        time_s=np.asarray([r["time_s"] for r in rows], dtype=np.float64),
        difficulty_bin=np.asarray([r["difficulty_bin"] for r in rows]),
        ip0=np.asarray([r["ip0"] for r in rows], dtype=np.float32),
        pfc0=np.stack([r["pfc0"] for r in rows], axis=0).astype(np.float32),
        sol0=np.stack([r["sol0"] for r in rows], axis=0).astype(np.float32),
        ip_target=np.stack([r["ip_target"] for r in rows], axis=0).astype(np.float32),
        boundary_radii=np.stack([r["boundary_radii"] for r in rows], axis=0).astype(np.float32),
        real_jdot_action=np.stack([r["real_jdot_action"] for r in rows], axis=0).astype(np.float32),
        oracle_ip_mean_error_a=np.asarray([r["oracle_ip_mean_error_a"] for r in rows], dtype=np.float32),
        oracle_ip_max_error_a=np.asarray([r["oracle_ip_max_error_a"] for r in rows], dtype=np.float32),
        current_limits=np.asarray(current_limits, dtype=np.float32),
        derivative_limits=np.asarray(derivative_limits, dtype=np.float32),
    )


def _write_initial_library(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shot_id=np.asarray([r["shot_id"] for r in rows]),
        source_index=np.asarray([r["source_index"] for r in rows], dtype=np.int64),
        time_s=np.asarray([r["time_s"] for r in rows], dtype=np.float64),
        ip0=np.asarray([r["ip0"] for r in rows], dtype=np.float32),
        pfc0=np.stack([r["pfc0"] for r in rows], axis=0).astype(np.float32),
        sol0=np.stack([r["sol0"] for r in rows], axis=0).astype(np.float32),
        split=np.asarray([r["split"] for r in rows]),
        difficulty_bin=np.asarray([r["difficulty_bin"] for r in rows]),
    )


def _write_config(
    *,
    base: dict[str, object],
    config_out: Path,
    machine_config: Path,
    target_dir: Path,
    initial_library: Path,
    train_output: str,
    balanced: bool,
) -> None:
    cfg = json.loads(json.dumps(base))
    cfg["name"] = (
        "t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_mpo"
        if balanced
        else "t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo"
    )

    def rel(path: Path) -> str:
        return str(Path(os.path.relpath(path.resolve(), config_out.parent.resolve())))

    cfg["sim"]["config_path"] = rel(machine_config)
    cfg["sim"]["max_episode_steps"] = 100
    cfg["sim"]["reset_source"] = "csv_initial_states"
    cfg["sim"]["csv_initial_state_library"] = rel(initial_library)
    cfg["sim"]["csv_initial_state_split"] = "train"
    cfg["sim"]["action_contract"] = "jdot_command"
    cfg["sim"].pop("delta_derivative_limits_aps", None)
    cfg["sim"]["terminate_on_boundary_loss"] = True
    cfg["sim"]["terminate_on_current_limit"] = True
    cfg["sim"]["current_hard_termination_fraction"] = 1.2
    cfg["sim"]["current_termination_grace_steps"] = 1
    cfg["sim"]["current_saturation_fraction"] = 1.0

    cfg["reference"]["duration_s"] = 0.1
    cfg["reference"]["t_step"] = 0.001
    cfg["reference"]["theta_count"] = 32
    cfg["reference"]["ip"] = {"kind": "replay_window"}
    cfg["reference"]["boundary"] = {"kind": "t15_replay_segment_conditioned", "replay_reference_dir": rel(target_dir)}

    cfg["observation"]["actor_kind"] = "controller_state_v6"
    cfg["observation"]["critic_kind"] = "compact_training_state_v2"
    cfg["observation"]["target_preview_steps"] = 10
    cfg["observation"]["target_preview_stride"] = 10
    cfg["observation"]["ip_rate_scale_aps"] = 500000.0
    cfg["observation"]["boundary_rate_scale_mps"] = 1.0

    cfg["learner"]["unroll_length"] = 100
    cfg["learner"]["min_replay_sequence_length"] = 100
    cfg["learner"]["rollout_chunk_length"] = 100
    cfg["learner"]["batch_size"] = 32
    cfg["learner"]["updates_per_rollout_chunk"] = 64
    cfg["learner"]["replay_capacity_episodes"] = 1024
    cfg["learner"]["action_samples"] = 64

    cfg["training"]["num_envs"] = 2048
    cfg["training"]["eval_max_steps"] = 100
    cfg["training"]["distributed_mode"] = "local_replay"
    cfg["training"]["production_mode"] = True
    cfg["training"]["early_stop_patience_evals"] = 0
    cfg["training"]["early_stop_min_delta"] = 0.0
    cfg["training"]["output_dir"] = "../../" + train_output

    cfg["reward"]["kind"] = "tcv_derivative"
    cfg["reward"]["terminal_reward"] = -20.0
    cfg["reward"]["reward_scale"] = 0.01
    cfg["reward"]["shape_mean_weight"] = 3.2
    cfg["reward"]["shape_max_weight"] = 0.8
    cfg["reward"]["ip_weight"] = 1.8
    cfg["reward"]["ip_scale_a"] = 25000.0
    cfg["reward"]["smoothmax_alpha"] = -5.0

    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _summary(rows: list[dict[str, object]], rejected: list[dict[str, object]], *, target_dir: Path, initial_library: Path, oracle_path: Path) -> dict[str, object]:
    split_counts = Counter(str(r["split"]) for r in rows)
    difficulty_counts = Counter(str(r["difficulty_bin"]) for r in rows)
    by_shot = Counter(str(r["shot_id"]) for r in rows)
    p90_mean_err_by_bin: dict[str, float] = {}
    for bin_name in sorted(difficulty_counts):
        values = [float(r["oracle_ip_mean_error_a"]) for r in rows if str(r["difficulty_bin"]) == bin_name]
        p90_mean_err_by_bin[bin_name] = float(np.percentile(values, 90)) if values else float("nan")
    return {
        "schema": "t15_replay_window_oracle_targets_v1",
        "target_dir": str(target_dir),
        "oracle_path": str(oracle_path),
        "initial_library": str(initial_library),
        "accepted_windows": int(len(rows)),
        "rejected_windows": int(len(rejected)),
        "split_counts": dict(sorted(split_counts.items())),
        "difficulty_bins": dict(sorted(difficulty_counts.items())),
        "accepted_by_shot": dict(sorted(by_shot.items())),
        "oracle_ip_mean_error_a_mean": float(np.mean([r["oracle_ip_mean_error_a"] for r in rows])),
        "oracle_ip_mean_error_a_p90_by_bin": p90_mean_err_by_bin,
    }


def _reject(shot_id: str, source_index: int, time_s: float, reason: str) -> dict[str, object]:
    return {"shot_id": str(shot_id), "source_index": int(source_index), "time_s": float(time_s), "reason": str(reason)}


def _write_rejections(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["shot_id", "source_index", "time_s", "reason"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
