#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 6-PFC t15_data_new 0.1 s replay-window RL inputs with an explicit "
            "whole-shot train/holdout split."
        )
    )
    parser.add_argument("--base-config", default="configs/experiments/t15_csv_initial_single_segment_0p1s_static_boundary_mpo.yaml")
    parser.add_argument("--data-root", default="../tokamak-sim/data/t15_data_new")
    parser.add_argument("--replay-reference-dir", default="../tokamak-sim/runs/t15md_limited_replay_dataset_sigmaL_3856_3857_3858_3863_3864")
    parser.add_argument("--machine-config", default="../tokamak-sim/configs/T15MD_new_data.toml")
    parser.add_argument("--config-out", default="configs/experiments/t15_new_replay_window_0p1s_tcvjdot_mpo.yaml")
    parser.add_argument("--library-out", default="data/processed/t15_new_replay_window_0p1s_initial_states.npz")
    parser.add_argument("--library-json", default="data/processed/t15_new_replay_window_0p1s_initial_states.json")
    parser.add_argument("--library-rejected", default="data/processed/t15_new_replay_window_0p1s_initial_states_rejected.csv")
    parser.add_argument("--summary-out", default="data/processed/t15_new_replay_window_0p1s_rl_summary.json")
    parser.add_argument("--train-shots", nargs="+", default=["3856", "3857", "3858", "3863"])
    parser.add_argument("--holdout-shots", nargs="+", default=["3864"])
    parser.add_argument("--window-steps", type=int, default=100)
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--max-rows-per-shot", type=int, default=0, help="0 keeps every valid window start.")
    args = parser.parse_args(argv)

    base_config = _resolve(args.base_config)
    data_root = _resolve(args.data_root)
    replay_reference_dir = _resolve(args.replay_reference_dir)
    machine_config = _resolve(args.machine_config)
    config_out = _resolve(args.config_out)
    library_out = _resolve(args.library_out)
    library_json = _resolve(args.library_json)
    library_rejected = _resolve(args.library_rejected)
    summary_out = _resolve(args.summary_out)
    train_shots = tuple(str(v) for v in args.train_shots)
    holdout_shots = tuple(str(v) for v in args.holdout_shots)

    overlap = sorted(set(train_shots) & set(holdout_shots))
    if overlap:
        raise ValueError(f"shots cannot be both train and holdout: {overlap}")
    if int(args.window_steps) <= 0:
        raise ValueError("--window-steps must be positive")

    cfg = _build_config(
        base_config=base_config,
        config_out=config_out,
        machine_config=machine_config,
        replay_reference_dir=replay_reference_dir,
        library_out=library_out,
    )
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    accepted, rejected = _build_library_rows(
        data_root=data_root,
        replay_reference_dir=replay_reference_dir,
        train_shots=train_shots,
        holdout_shots=holdout_shots,
        window_steps=int(args.window_steps),
        delimiter=str(args.delimiter),
        max_rows_per_shot=int(args.max_rows_per_shot),
    )
    _validate_library(accepted, train_shots=train_shots, holdout_shots=holdout_shots)

    library_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        library_out,
        shot_id=np.asarray([row["shot_id"] for row in accepted]),
        source_index=np.asarray([row["source_index"] for row in accepted], dtype=np.int64),
        time_s=np.asarray([row["time_s"] for row in accepted], dtype=float),
        ip0=np.asarray([row["ip0"] for row in accepted], dtype=float),
        pfc0=np.stack([row["pfc0"] for row in accepted], axis=0).astype(float),
        sol0=np.stack([row["sol0"] for row in accepted], axis=0).astype(float),
        split=np.asarray([row["split"] for row in accepted]),
    )

    summary = _summary(
        accepted,
        rejected,
        data_root=data_root,
        replay_reference_dir=replay_reference_dir,
        machine_config=machine_config,
        config_out=config_out,
        library_out=library_out,
        train_shots=train_shots,
        holdout_shots=holdout_shots,
        window_steps=int(args.window_steps),
    )
    library_json.parent.mkdir(parents=True, exist_ok=True)
    library_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_rejections(library_rejected, rejected)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "config": str(config_out),
                "library": str(library_out),
                "summary": str(summary_out),
                "train_shots": list(train_shots),
                "holdout_shots": list(holdout_shots),
                "train_rows": summary["train_rows"],
                "holdout_rows": summary["holdout_rows"],
            },
            indent=2,
        )
    )
    return 0


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _build_config(
    *,
    base_config: Path,
    config_out: Path,
    machine_config: Path,
    replay_reference_dir: Path,
    library_out: Path,
) -> dict[str, object]:
    cfg = json.loads(base_config.read_text(encoding="utf-8"))

    def rel(path: Path) -> str:
        return str(Path(os.path.relpath(path.resolve(), config_out.parent.resolve())))

    cfg["name"] = "t15_new_replay_window_0p1s_tcvjdot_mpo"
    cfg["sim"]["config_path"] = rel(machine_config)
    cfg["sim"]["initial_currents_path"] = None
    cfg["sim"]["max_episode_steps"] = 100
    cfg["sim"]["reset_source"] = "csv_initial_states"
    cfg["sim"]["csv_initial_state_library"] = rel(library_out)
    cfg["sim"]["csv_initial_state_split"] = "train"
    cfg["sim"]["action_contract"] = "jdot_command"
    cfg["sim"]["terminate_on_boundary_loss"] = True
    cfg["sim"]["terminate_on_current_limit"] = True
    cfg["sim"]["current_hard_termination_fraction"] = 1.2
    cfg["sim"]["current_termination_grace_steps"] = 1
    cfg["sim"]["current_saturation_fraction"] = 1.0
    cfg["sim"].pop("delta_derivative_limits_aps", None)

    cfg["reference"]["duration_s"] = 0.1
    cfg["reference"]["t_step"] = 0.001
    cfg["reference"]["theta_count"] = 32
    cfg["reference"]["ip"] = {"kind": "replay_window"}
    cfg["reference"]["boundary"] = {
        "kind": "t15_replay_segment_conditioned",
        "replay_reference_dir": rel(replay_reference_dir),
    }

    cfg["observation"]["actor_kind"] = "controller_state_v4"
    cfg["observation"]["target_preview_steps"] = 10
    cfg["observation"]["target_preview_stride"] = 10

    reward = cfg["reward"]
    reward["kind"] = "tcv_derivative"
    reward["terminal_reward"] = -20.0
    reward["terminal_remaining_cost"] = 0.0
    reward["shape_mean_weight"] = 3.2
    reward["shape_max_weight"] = 0.8
    reward["ip_weight"] = 1.8
    reward["current_weight"] = 0.75
    reward["derivative_weight"] = 0.1875
    reward["actuator_saturation_weight"] = 0.1875
    reward["reward_scale"] = 0.01
    reward["smoothmax_alpha"] = -5.0
    reward["ip_scale_a"] = 25000.0

    learner = cfg["learner"]
    learner["discount"] = 0.99
    learner["min_replay_sequence_length"] = min(int(learner.get("min_replay_sequence_length", 32)), 64)

    training = cfg["training"]
    training["steps"] = 10000000
    training["num_envs"] = 16384
    training["eval_max_steps"] = 100
    training["distributed_mode"] = "local_replay"
    training["production_mode"] = True
    training["output_dir"] = "../../outputs/t15_new_replay_window_0p1s_tcvjdot_mpo"

    return cfg


def _build_library_rows(
    *,
    data_root: Path,
    replay_reference_dir: Path,
    train_shots: Iterable[str],
    holdout_shots: Iterable[str],
    window_steps: int,
    delimiter: str,
    max_rows_per_shot: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    train_set = {str(v) for v in train_shots}
    holdout_set = {str(v) for v in holdout_shots}
    all_shots = tuple(sorted(train_set | holdout_set, key=int))

    for shot in all_shots:
        split = "holdout" if shot in holdout_set else "train"
        ip_path = data_root / "ip" / f"t15md_{shot}_ip.csv"
        coil_path = data_root / "coils" / f"t15md_{shot}_coils.csv"
        ref_path = replay_reference_dir / f"lqr_boundary_reference_{shot}_smoothed.npz"
        if not ip_path.exists():
            rejected.append(_reject(shot, -1, float("nan"), "missing_ip_csv"))
            continue
        if not coil_path.exists():
            rejected.append(_reject(shot, -1, float("nan"), "missing_coil_csv"))
            continue
        if not ref_path.exists():
            rejected.append(_reject(shot, -1, float("nan"), "missing_smoothed_boundary_reference"))
            continue

        ip = _load_table(ip_path, delimiter=delimiter)
        coils = _load_table(coil_path, delimiter=delimiter)
        ref = np.load(ref_path)
        ref_len = _reference_length(ref)
        if coils.shape[1] != 10:
            raise ValueError(f"{coil_path} must contain time + 3 SOL + 6 PFC columns, got shape {coils.shape}")
        if ip.shape[1] < 2:
            raise ValueError(f"{ip_path} must contain time + Ip columns")

        valid_end = min(ip.shape[0], ref_len) - int(window_steps)
        if valid_end <= 1:
            rejected.append(_reject(shot, -1, float(ip[0, 0]), "shot_too_short_for_window"))
            continue
        row_indices = np.arange(0, valid_end, dtype=np.int64)
        if int(max_rows_per_shot) > 0 and row_indices.size > int(max_rows_per_shot):
            keep = np.linspace(0, row_indices.size - 1, int(max_rows_per_shot)).round().astype(np.int64)
            row_indices = row_indices[keep]

        sol_cols = [1, 2, 3]
        pfc_cols = [4, 5, 6, 7, 8, 9]
        for idx in row_indices:
            time_s = float(ip[int(idx), 0])
            ip0 = float(ip[int(idx), 1])
            if not np.isfinite(time_s) or not np.isfinite(ip0) or ip0 <= 0.0:
                rejected.append(_reject(shot, int(idx), time_s, "invalid_ip"))
                continue
            sol0 = np.asarray([np.interp(time_s, coils[:, 0], coils[:, col]) for col in sol_cols], dtype=float)
            pfc0 = np.asarray([np.interp(time_s, coils[:, 0], coils[:, col]) for col in pfc_cols], dtype=float)
            if not np.all(np.isfinite(sol0)) or not np.all(np.isfinite(pfc0)):
                rejected.append(_reject(shot, int(idx), time_s, "invalid_current"))
                continue
            accepted.append(
                {
                    "shot_id": str(shot),
                    "source_index": int(idx),
                    "time_s": time_s,
                    "ip0": ip0,
                    "pfc0": pfc0,
                    "sol0": sol0,
                    "split": split,
                }
            )
        print(f"[t15-new-replay-window] shot={shot} split={split} accepted_so_far={len(accepted)}", flush=True)
    accepted.sort(key=lambda row: (int(row["shot_id"]), float(row["time_s"]), int(row["source_index"])))
    return accepted, rejected


def _load_table(path: Path, *, delimiter: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=delimiter, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D table")
    if arr.shape[0] < 3:
        raise ValueError(f"{path} is too short")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path} contains non-finite values")
    if np.any(np.diff(arr[:, 0]) <= 0.0):
        raise ValueError(f"{path} time column must be strictly increasing")
    return arr


def _reference_length(ref: np.lib.npyio.NpzFile) -> int:
    for key in ("radii", "boundary_radii", "target_radii"):
        if key in ref:
            return int(ref[key].shape[0])
    for key in ("Ip_ref", "Ip", "ip"):
        if key in ref:
            return int(ref[key].shape[0])
    raise ValueError("smoothed boundary reference has no recognizable time-series array")


def _validate_library(accepted: list[dict[str, object]], *, train_shots: Iterable[str], holdout_shots: Iterable[str]) -> None:
    split_counts = Counter(str(row["split"]) for row in accepted)
    by_shot_split: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        by_shot_split[str(row["shot_id"])].add(str(row["split"]))
    errors: list[str] = []
    if split_counts.get("train", 0) < 1000:
        errors.append(f"train rows must be >=1000, got {split_counts.get('train', 0)}")
    if split_counts.get("holdout", 0) < 100:
        errors.append(f"holdout rows must be >=100, got {split_counts.get('holdout', 0)}")
    for shot in train_shots:
        if by_shot_split.get(str(shot)) != {"train"}:
            errors.append(f"train shot {shot} does not have only train rows: {sorted(by_shot_split.get(str(shot), set()))}")
    for shot in holdout_shots:
        if by_shot_split.get(str(shot)) != {"holdout"}:
            errors.append(f"holdout shot {shot} does not have only holdout rows: {sorted(by_shot_split.get(str(shot), set()))}")
    if errors:
        raise ValueError("; ".join(errors))


def _summary(
    accepted: list[dict[str, object]],
    rejected: list[dict[str, object]],
    *,
    data_root: Path,
    replay_reference_dir: Path,
    machine_config: Path,
    config_out: Path,
    library_out: Path,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
    window_steps: int,
) -> dict[str, object]:
    split_counts = Counter(str(row["split"]) for row in accepted)
    accepted_by_shot = Counter(str(row["shot_id"]) for row in accepted)
    split_by_shot: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "holdout": 0})
    time_by_shot: dict[str, list[float]] = defaultdict(list)
    ip_by_shot: dict[str, list[float]] = defaultdict(list)
    for row in accepted:
        shot = str(row["shot_id"])
        split_by_shot[shot][str(row["split"])] += 1
        time_by_shot[shot].append(float(row["time_s"]))
        ip_by_shot[shot].append(float(row["ip0"]))
    return {
        "schema": "t15_new_replay_window_0p1s_v1",
        "data_root": str(data_root),
        "replay_reference_dir": str(replay_reference_dir),
        "machine_config": str(machine_config),
        "config_out": str(config_out),
        "library_out": str(library_out),
        "train_shots": list(train_shots),
        "holdout_shots": list(holdout_shots),
        "window_steps": int(window_steps),
        "episode_duration_s": 0.1,
        "split_policy": "explicit_whole_shot",
        "accepted_rows": int(len(accepted)),
        "train_rows": int(split_counts.get("train", 0)),
        "holdout_rows": int(split_counts.get("holdout", 0)),
        "rejected_rows": int(len(rejected)),
        "accepted_by_shot": dict(sorted(accepted_by_shot.items())),
        "split_by_shot": {shot: dict(counts) for shot, counts in sorted(split_by_shot.items())},
        "time_s_min_by_shot": {shot: float(np.min(values)) for shot, values in sorted(time_by_shot.items())},
        "time_s_max_by_shot": {shot: float(np.max(values)) for shot, values in sorted(time_by_shot.items())},
        "ip_a_min_by_shot": {shot: float(np.min(values)) for shot, values in sorted(ip_by_shot.items())},
        "ip_a_max_by_shot": {shot: float(np.max(values)) for shot, values in sorted(ip_by_shot.items())},
    }


def _reject(shot_id: str, source_index: int, time_s: float, reason: str) -> dict[str, object]:
    return {
        "shot_id": str(shot_id),
        "source_index": int(source_index),
        "time_s": float(time_s),
        "reason": str(reason),
    }


def _write_rejections(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["shot_id", "source_index", "time_s", "reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
