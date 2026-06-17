from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.io.config_io import load_config

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env.batch_env import _current_limit_vector
from tokamak_rl_v2.env.t15_csv_initial_states import validate_split_nonoverlap
from tokamak_rl_v2.env.t15_reference_limits import load_reference_limits


IP_RE = re.compile(r"t15md_(\d+)_ip\.csv$")
_WORKER_SIM_CFG = None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build coherent real-T15 CSV reset-state library for plain RL.")
    ap.add_argument("--experiment-config", default="configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml")
    ap.add_argument("--data-root", default="../tokamak-sim/data/t15_data_new")
    ap.add_argument("--ip-glob", default="ip/t15md_*_ip.csv")
    ap.add_argument("--coils-dir", default="coils")
    ap.add_argument("--machine-config", default="../tokamak-sim/configs/T15MD_new_data.toml")
    ap.add_argument("--delimiter", default=";")
    ap.add_argument("--out-npz", default="data/processed/t15_csv_initial_states.npz")
    ap.add_argument("--out-json", default="data/processed/t15_csv_initial_states.json")
    ap.add_argument("--out-rejected", default="data/processed/t15_csv_initial_states_rejected.csv")
    ap.add_argument("--reference-limits", default="data/processed/t15_reference_limits.json")
    ap.add_argument("--zero-action-steps", type=int, default=10)
    ap.add_argument("--max-rows-per-shot", type=int, default=600)
    ap.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    machine_config = Path(args.machine_config).resolve()
    exp_cfg = load_experiment_config(args.experiment_config)
    sim_cfg = load_config(machine_config)
    current_limits = _current_limit_vector(exp_cfg, sim_cfg)
    reference_limits = load_reference_limits(args.reference_limits)

    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for ip_path in sorted(data_root.glob(str(args.ip_glob))):
        match = IP_RE.search(ip_path.name)
        if not match:
            continue
        shot_id = match.group(1)
        coil_path = data_root / str(args.coils_dir) / f"t15md_{shot_id}_coils.csv"
        if not coil_path.exists():
            rejected.append(_reject(shot_id, -1, float("nan"), "missing_coil_csv", float("nan"), float("nan")))
            continue
        ip = _load_two_column(ip_path, delimiter=str(args.delimiter))
        coils = _load_coils(coil_path, delimiter=str(args.delimiter))
        rows = _candidate_rows(ip, coils, max_rows=int(args.max_rows_per_shot))
        print(f"[csv-initial-states] shot={shot_id} candidates={len(rows)}", flush=True)
        for row in rows:
            row_index = int(row["source_index"])
            time_s = float(row["time_s"])
            ip0 = float(row["ip0"])
            pfc0 = np.asarray(row["pfc0"], dtype=float)
            sol0 = np.asarray(row["sol0"], dtype=float)
            dipdt = float(row["local_abs_dip_dt_a_per_s"])
            usage = float(np.max(np.abs(np.concatenate([pfc0, sol0])) / current_limits))
            reason = _basic_rejection_reason(ip0, pfc0, sol0, usage, dipdt, ip_min=float(reference_limits.ip_p01_a), ip_max=float(reference_limits.ip_p99_a))
            if reason is not None:
                rejected.append(_reject(shot_id, row_index, time_s, reason, usage, dipdt, ip0=ip0))
                continue
            candidates.append(
                _candidate(
                    shot_id=shot_id,
                    source_index=row_index,
                    time_s=time_s,
                    ip0=ip0,
                    pfc0=pfc0,
                    sol0=sol0,
                    dipdt=dipdt,
                    usage=usage,
                )
            )

    accepted = _validate_sim_candidates(
        candidates,
        rejected,
        machine_config=machine_config,
        sim_cfg=sim_cfg,
        zero_action_steps=int(args.zero_action_steps),
        workers=max(1, int(args.workers)),
        progress_every=max(1, int(args.progress_every)),
    )

    accepted = sorted(accepted, key=lambda row: (str(row["shot_id"]), float(row["time_s"]), int(row["source_index"])))
    episode_gap_s = float(exp_cfg.sim.max_episode_steps) * float(exp_cfg.reference.t_step)
    splits = _assign_splits(accepted, gap_s=episode_gap_s)
    _validate_split_gates(accepted, splits, gap_s=episode_gap_s)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        shot_id=np.asarray([str(row["shot_id"]) for row in accepted]),
        source_index=np.asarray([int(row["source_index"]) for row in accepted], dtype=np.int64),
        time_s=np.asarray([float(row["time_s"]) for row in accepted], dtype=float),
        ip0=np.asarray([float(row["ip0"]) for row in accepted], dtype=float),
        pfc0=np.stack([np.asarray(row["pfc0"], dtype=float) for row in accepted], axis=0),
        sol0=np.stack([np.asarray(row["sol0"], dtype=float) for row in accepted], axis=0),
        split=np.asarray(splits),
    )
    summary = _summary(
        accepted,
        rejected,
        data_root=data_root,
        machine_config=machine_config,
        splits=splits,
        reference_limits_path=Path(args.reference_limits).resolve(),
        split_gap_seconds=episode_gap_s,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_rejections(Path(args.out_rejected), rejected)
    print(out_npz)
    return 0


def _candidate(
    *,
    shot_id: str,
    source_index: int,
    time_s: float,
    ip0: float,
    pfc0: np.ndarray,
    sol0: np.ndarray,
    dipdt: float,
    usage: float,
) -> dict[str, object]:
    return {
        "shot_id": str(shot_id),
        "source_index": int(source_index),
        "time_s": float(time_s),
        "ip0": float(ip0),
        "pfc0": np.asarray(pfc0, dtype=float),
        "sol0": np.asarray(sol0, dtype=float),
        "local_abs_dip_dt_a_per_s": float(dipdt),
        "max_current_usage_fraction": float(usage),
    }


def _validate_sim_candidates(
    candidates: list[dict[str, object]],
    rejected: list[dict[str, object]],
    *,
    machine_config: Path,
    sim_cfg,
    zero_action_steps: int,
    workers: int,
    progress_every: int,
) -> list[dict[str, object]]:
    accepted: list[dict[str, object]] = []
    total = len(candidates)
    print(f"[csv-initial-states] simulator-validation candidates={total} workers={workers}", flush=True)
    if total == 0:
        return accepted
    if workers <= 1:
        for idx, row in enumerate(candidates, start=1):
            reason = _sim_rejection_reason(
                sim_cfg,
                ip0=float(row["ip0"]),
                pfc0=np.asarray(row["pfc0"], dtype=float),
                sol0=np.asarray(row["sol0"], dtype=float),
                zero_action_steps=zero_action_steps,
            )
            _record_sim_validation(row, reason, accepted, rejected)
            if idx % progress_every == 0 or idx == total:
                print(f"[csv-initial-states] validated={idx}/{total} accepted={len(accepted)} rejected={len(rejected)}", flush=True)
        return accepted

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(machine_config),)) as pool:
        future_to_row = {
            pool.submit(
                _sim_rejection_reason_worker,
                float(row["ip0"]),
                np.asarray(row["pfc0"], dtype=float),
                np.asarray(row["sol0"], dtype=float),
                int(zero_action_steps),
            ): row
            for row in candidates
        }
        for idx, future in enumerate(as_completed(future_to_row), start=1):
            row = future_to_row[future]
            try:
                reason = future.result()
            except Exception as exc:  # pragma: no cover - defensive multiprocessing surface
                reason = f"sim_failure:{type(exc).__name__}"
            _record_sim_validation(row, reason, accepted, rejected)
            if idx % progress_every == 0 or idx == total:
                print(f"[csv-initial-states] validated={idx}/{total} accepted={len(accepted)} rejected={len(rejected)}", flush=True)
    return accepted


def _record_sim_validation(
    row: dict[str, object],
    reason: str | None,
    accepted: list[dict[str, object]],
    rejected: list[dict[str, object]],
) -> None:
    if reason is None:
        accepted.append(row)
        return
    rejected.append(
        _reject(
            str(row["shot_id"]),
            int(row["source_index"]),
            float(row["time_s"]),
            reason,
            float(row["max_current_usage_fraction"]),
            float(row["local_abs_dip_dt_a_per_s"]),
            ip0=float(row["ip0"]),
        )
    )


def _init_worker(machine_config: str) -> None:
    global _WORKER_SIM_CFG
    _WORKER_SIM_CFG = load_config(Path(machine_config))


def _sim_rejection_reason_worker(ip0: float, pfc0: np.ndarray, sol0: np.ndarray, zero_action_steps: int) -> str | None:
    if _WORKER_SIM_CFG is None:
        raise RuntimeError("worker simulator config was not initialized")
    return _sim_rejection_reason(_WORKER_SIM_CFG, ip0=ip0, pfc0=pfc0, sol0=sol0, zero_action_steps=zero_action_steps)


def _load_two_column(path: Path, *, delimiter: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=delimiter, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns")
    arr = arr[:, :2]
    _validate_time(arr[:, 0], path)
    return arr


def _load_coils(path: Path, *, delimiter: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=delimiter, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 10:
        raise ValueError(f"{path} must contain time + 3 SOL + 6 PFC columns")
    _validate_time(arr[:, 0], path)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path} contains non-finite values")
    return arr[:, :10]


def _validate_time(t: np.ndarray, path: Path) -> None:
    if t.ndim != 1 or t.size < 3 or np.any(np.diff(t) <= 0.0):
        raise ValueError(f"{path} time column must be strictly increasing")


def _candidate_rows(ip: np.ndarray, coils: np.ndarray, *, max_rows: int) -> list[dict[str, object]]:
    start = max(float(ip[0, 0]), float(coils[0, 0]))
    end = min(float(ip[-1, 0]), float(coils[-1, 0]))
    keep = (ip[:, 0] >= start) & (ip[:, 0] <= end)
    times = ip[keep, 0]
    source_indices = np.flatnonzero(keep)
    if max_rows > 0 and times.size > max_rows:
        pick = np.linspace(0, times.size - 1, max_rows).round().astype(int)
        times = times[pick]
        source_indices = source_indices[pick]
    ip_interp = np.interp(times, ip[:, 0], ip[:, 1])
    dipdt = np.gradient(np.interp(ip[:, 0], ip[:, 0], ip[:, 1]), ip[:, 0])
    dipdt_interp = np.interp(times, ip[:, 0], np.abs(dipdt))
    sol_cols = [1, 2, 3]
    pfc_cols = [4, 5, 6, 7, 8, 9]
    sol = np.stack([np.interp(times, coils[:, 0], coils[:, col]) for col in sol_cols], axis=1)
    pfc = np.stack([np.interp(times, coils[:, 0], coils[:, col]) for col in pfc_cols], axis=1)
    out = []
    for idx in range(times.size):
        out.append(
            {
                "time_s": float(times[idx]),
                "source_index": int(source_indices[idx]),
                "ip0": float(ip_interp[idx]),
                "pfc0": pfc[idx],
                "sol0": sol[idx],
                "local_abs_dip_dt_a_per_s": float(dipdt_interp[idx]),
            }
        )
    return out


def _basic_rejection_reason(ip0: float, pfc0: np.ndarray, sol0: np.ndarray, usage: float, dipdt: float, *, ip_min: float, ip_max: float) -> str | None:
    if not np.isfinite(ip0) or ip0 <= 0.0:
        return "invalid_ip"
    if not float(ip_min) <= float(ip0) <= float(ip_max):
        return "ip_outside_reference_bounds"
    if not np.all(np.isfinite(pfc0)) or not np.all(np.isfinite(sol0)):
        return "invalid_currents"
    if not np.isfinite(usage) or usage > 1.0:
        return "current_over_safety_limit"
    if not np.isfinite(dipdt):
        return "invalid_local_dipdt"
    return None


def _sim_rejection_reason(sim_cfg, *, ip0: float, pfc0: np.ndarray, sol0: np.ndarray, zero_action_steps: int) -> str | None:
    try:
        pfc = sim_cfg.pfc.__class__(name=sim_cfg.pfc.name, coils=list(sim_cfg.pfc.coils), currents=pfc0)
        sol = sim_cfg.sol.__class__(name=sim_cfg.sol.name, coils=list(sim_cfg.sol.coils), currents=sol0)
        model = PlasmaModel.from_settings(grid=sim_cfg.grid, pfc=pfc, sol=sol, settings=replace(sim_cfg.physics, Ip0=float(ip0)))
        _assert_boundary(model, sim_cfg)
        zeros_pfc = np.zeros((sim_cfg.pfc.n_coils,), dtype=float)
        zeros_sol = np.zeros((sim_cfg.sol.n_coils,), dtype=float)
        for _ in range(max(0, int(zero_action_steps))):
            model.step(pfc_current_derivs=zeros_pfc, sol_current_derivs=zeros_sol)
            _assert_boundary(model, sim_cfg)
    except BoundaryNotFoundError:
        return "boundary_not_found"
    except Exception as exc:
        return f"sim_failure:{type(exc).__name__}"
    return None


def _assert_boundary(model: PlasmaModel, sim_cfg) -> None:
    _poly, _level, _status = find_plasma_boundary_with_status(
        model.state.psi,
        model.grid,
        (model.R0, model.Z0),
        n_levels=80,
        limiter_shape=sim_cfg.limiter_shape,
        boundary_mode=sim_cfg.boundary_mode,
    )


def _reject(shot_id: str, row_index: int, time_s: float, reason: str, usage: float, dipdt: float, *, ip0: float = float("nan")) -> dict[str, object]:
    return {
        "shot_id": str(shot_id),
        "row_index": int(row_index),
        "time_s": float(time_s),
        "reason": str(reason),
        "ip0": float(ip0),
        "max_current_usage_fraction": float(usage),
        "local_abs_dip_dt_a_per_s": float(dipdt),
    }


def _assign_splits(accepted: list[dict[str, object]], *, gap_s: float) -> np.ndarray:
    split = np.full((len(accepted),), "excluded", dtype="<U8")
    by_shot: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, row in enumerate(accepted):
        by_shot[str(row["shot_id"])].append((index, float(row["time_s"])))
    for shot_rows in by_shot.values():
        if not shot_rows:
            continue
        times = np.asarray([time_s for _idx, time_s in shot_rows], dtype=float)
        first_block = int(np.floor(float(np.nanmin(times)) / float(gap_s)))
        blocks = np.floor(times / float(gap_s)).astype(int) - first_block
        unique_blocks = np.asarray(sorted(set(int(v) for v in blocks.tolist())), dtype=int)
        candidate_blocks = list(unique_blocks[4::5])
        candidate_blocks.extend(int(v) for v in unique_blocks.tolist() if int(v) not in set(candidate_blocks))
        best: tuple[int, np.ndarray, np.ndarray] | None = None
        for block in candidate_blocks:
            holdout_mask = blocks == int(block)
            holdout_times = times[holdout_mask]
            if holdout_times.size == 0:
                continue
            distance = np.min(np.abs(times[:, None] - holdout_times[None, :]), axis=1)
            train_mask = (~holdout_mask) & (distance > float(gap_s))
            train_count = int(np.sum(train_mask))
            holdout_count = int(np.sum(holdout_mask))
            if train_count >= 80 and holdout_count >= 10 and train_count + holdout_count >= 100:
                best = (train_count + holdout_count, train_mask, holdout_mask)
                break
            score = train_count + holdout_count
            if best is None or score > best[0]:
                best = (score, train_mask, holdout_mask)
        if best is None:
            continue
        _score, train_mask, holdout_mask = best
        if int(np.sum(train_mask)) < 80 or int(np.sum(holdout_mask)) < 10 or int(np.sum(train_mask)) + int(np.sum(holdout_mask)) < 100:
            continue
        for local, (row_index, _time_s) in enumerate(shot_rows):
            if holdout_mask[local]:
                split[row_index] = "holdout"
            elif train_mask[local]:
                split[row_index] = "train"
    keep = split != "excluded"
    if not bool(np.all(keep)):
        kept = np.flatnonzero(keep)
        accepted[:] = [accepted[int(i)] for i in kept.tolist()]
        split = split[kept]
    return split


def _validate_split_gates(accepted: list[dict[str, object]], splits: np.ndarray, *, gap_s: float) -> None:
    accepted_rows = int(len(accepted))
    train_rows = int(np.sum(splits == "train"))
    holdout_rows = int(np.sum(splits == "holdout"))
    errors: list[str] = []
    if accepted_rows < 1000:
        errors.append(f"accepted_rows >= 1000 required, got {accepted_rows}")
    if train_rows < 1000:
        errors.append(f"train_rows >= 1000 required, got {train_rows}")
    if holdout_rows < 100:
        errors.append(f"holdout_rows >= 100 required, got {holdout_rows}")
    by_shot: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "train": 0, "holdout": 0})
    for row, split in zip(accepted, splits, strict=True):
        shot = str(row["shot_id"])
        by_shot[shot]["accepted"] += 1
        by_shot[shot][str(split)] += 1
    small = {
        shot: counts
        for shot, counts in sorted(by_shot.items())
        if counts["accepted"] < 100 or counts["train"] < 80 or counts["holdout"] < 10
    }
    if small:
        errors.append(f"per-shot split gates failed: {small}")
    try:
        validate_split_nonoverlap(
            np.asarray([str(row["shot_id"]) for row in accepted], dtype=str),
            np.asarray([float(row["time_s"]) for row in accepted], dtype=float),
            np.asarray(splits, dtype=str),
            min_gap_s=float(gap_s),
        )
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        raise ValueError("; ".join(errors))


def _summary(
    accepted: list[dict[str, object]],
    rejected: list[dict[str, object]],
    *,
    data_root: Path,
    machine_config: Path,
    splits: np.ndarray,
    reference_limits_path: Path,
    split_gap_seconds: float,
) -> dict[str, object]:
    accepted_by_shot = Counter(str(row["shot_id"]) for row in accepted)
    split_counts = Counter(str(value) for value in splits.tolist())
    split_by_shot: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "holdout": 0})
    for row, split in zip(accepted, splits, strict=True):
        split_by_shot[str(row["shot_id"])][str(split)] += 1
    rejected_by_reason = Counter(str(row["reason"]) for row in rejected)
    time_by_shot: dict[str, list[float]] = defaultdict(list)
    ip_by_shot: dict[str, list[float]] = defaultdict(list)
    dipdt_values = []
    ip_values = []
    for row in accepted:
        shot = str(row["shot_id"])
        time_by_shot[shot].append(float(row["time_s"]))
        ip_by_shot[shot].append(float(row["ip0"]))
        ip_values.append(float(row["ip0"]))
        dipdt_values.append(float(row["local_abs_dip_dt_a_per_s"]))
    dipdt_arr = np.asarray(dipdt_values, dtype=float)
    ip_arr = np.asarray(ip_values, dtype=float)
    time_min = {shot: float(np.min(values)) for shot, values in sorted(time_by_shot.items())}
    time_max = {shot: float(np.max(values)) for shot, values in sorted(time_by_shot.items())}
    return {
        "source_layout": "split_t15_data_new",
        "source_root": str(data_root),
        "machine_config": str(machine_config),
        "reference_limits": str(reference_limits_path),
        "split_gap_seconds": float(split_gap_seconds),
        "total_rows": int(len(accepted) + len(rejected)),
        "candidate_rows": int(len(accepted) + len(rejected)),
        "accepted_rows": int(len(accepted)),
        "train_rows": int(split_counts.get("train", 0)),
        "holdout_rows": int(split_counts.get("holdout", 0)),
        "rejected_rows": int(len(rejected)),
        "accepted_by_shot": dict(sorted(accepted_by_shot.items())),
        "split_by_shot": {shot: dict(counts) for shot, counts in sorted(split_by_shot.items())},
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        "time_min_s_by_shot": time_min,
        "time_max_s_by_shot": time_max,
        "time_s_min_by_shot": time_min,
        "time_s_max_by_shot": time_max,
        "ip_a_min_by_shot": {shot: float(np.min(values)) for shot, values in sorted(ip_by_shot.items())},
        "ip_a_max_by_shot": {shot: float(np.max(values)) for shot, values in sorted(ip_by_shot.items())},
        "ip_min_a": float(np.nanmin(ip_arr)) if ip_values else float("nan"),
        "ip_max_a": float(np.nanmax(ip_arr)) if ip_values else float("nan"),
        "abs_dip_dt_min_a_per_s": float(np.nanmin(dipdt_arr)) if dipdt_values else float("nan"),
        "abs_dip_dt_max_a_per_s": float(np.nanmax(dipdt_arr)) if dipdt_values else float("nan"),
        "abs_dip_dt_p50_a_per_s": float(np.nanpercentile(dipdt_arr, 50.0)) if dipdt_values else float("nan"),
        "abs_dip_dt_p95_a_per_s": float(np.nanpercentile(dipdt_arr, 95.0)) if dipdt_values else float("nan"),
        "abs_dip_dt_p99_a_per_s": float(np.nanpercentile(dipdt_arr, 99.0)) if dipdt_values else float("nan"),
        "local_abs_dip_dt_a_per_s_p50": float(np.nanpercentile(dipdt_arr, 50.0)) if dipdt_values else float("nan"),
        "local_abs_dip_dt_a_per_s_p95": float(np.nanpercentile(dipdt_arr, 95.0)) if dipdt_values else float("nan"),
    }


def _write_rejections(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["shot_id", "row_index", "time_s", "reason", "ip0", "max_current_usage_fraction", "local_abs_dip_dt_a_per_s"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
