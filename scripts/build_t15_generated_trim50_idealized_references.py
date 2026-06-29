#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_BOUNDARY_PARAM_DIR = Path("../tokamak-sim/output/t15_boundary_parameters_trim50_idealized_low_tau_gpu_plain_1e6")
DEFAULT_DATA_ROOT = Path("../tokamak-sim/data/t15_data_new_trim50_idealized")
DEFAULT_ENVELOPE_OUT = Path("data/processed/t15_generated_trim50_idealized_envelope.json")
DEFAULT_INITIAL_STATES_OUT = Path("data/processed/t15_generated_trim50_idealized_initial_states.npz")
DEFAULT_TRAIN_SHOTS = ("3854", "3855", "3856", "3859", "3862")
DEFAULT_HOLDOUT_SHOTS = ("3863",)
PARAM_COLUMNS = ("R0", "Z0", "A0", "kappa", "delta")
DEFAULT_PADDING = 1.2
SHAPE_MAX_PADDING = 1.6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-param-dir", type=Path, default=DEFAULT_BOUNDARY_PARAM_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--envelope-out", type=Path, default=DEFAULT_ENVELOPE_OUT)
    parser.add_argument("--initial-states-out", type=Path, default=DEFAULT_INITIAL_STATES_OUT)
    parser.add_argument("--train-shots", nargs="+", default=list(DEFAULT_TRAIN_SHOTS))
    parser.add_argument("--holdout-shots", nargs="+", default=list(DEFAULT_HOLDOUT_SHOTS))
    args = parser.parse_args()

    boundary_param_dir = args.boundary_param_dir.resolve()
    data_root = args.data_root.resolve()
    train_shots = tuple(str(int(v)) for v in args.train_shots)
    holdout_shots = tuple(str(int(v)) for v in args.holdout_shots)
    wanted_shots = set(train_shots) | set(holdout_shots)

    rows = _load_boundary_rows(boundary_param_dir, wanted_shots=wanted_shots)
    if not rows:
        raise SystemExit(f"no usable boundary parameter rows found in {boundary_param_dir}")
    envelope = _build_envelope(rows, boundary_param_dir=boundary_param_dir, data_root=data_root, train_shots=train_shots, holdout_shots=holdout_shots)
    reset = _build_reset_library(rows, data_root=data_root, train_shots=train_shots, holdout_shots=holdout_shots)

    args.envelope_out.parent.mkdir(parents=True, exist_ok=True)
    args.initial_states_out.parent.mkdir(parents=True, exist_ok=True)
    args.envelope_out.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(args.initial_states_out, **reset)

    summary = {
        "envelope": str(args.envelope_out),
        "initial_states": str(args.initial_states_out),
        "rows": int(reset["ip0"].shape[0]),
        "train_rows": int(np.count_nonzero(reset["split"] == "train")),
        "holdout_rows": int(np.count_nonzero(reset["split"] == "holdout")),
        "shots": sorted(set(reset["shot_id"].astype(str).tolist()), key=int),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _load_boundary_rows(root: Path, *, wanted_shots: set[str]) -> list[dict[str, float | str]]:
    if not root.exists():
        raise FileNotFoundError(f"boundary parameter directory does not exist: {root}")
    rows: list[dict[str, float | str]] = []
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
                if shot not in wanted_shots:
                    continue
                row: dict[str, float | str] = {
                    "shot": shot,
                    "step": int(float(raw["step"])),
                    "t": float(raw["t"]),
                    "Ip": float(raw["Ip"]),
                }
                for name in PARAM_COLUMNS:
                    row[name] = float(raw[name])
                if all(np.isfinite(float(row[name])) for name in ("step", "t", "Ip", *PARAM_COLUMNS)):
                    rows.append(row)
    rows.sort(key=lambda r: (int(str(r["shot"])), float(r["t"])))
    return rows


def _build_envelope(
    rows: list[dict[str, float | str]],
    *,
    boundary_param_dir: Path,
    data_root: Path,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
) -> dict[str, object]:
    ip = _series(rows, "Ip")
    a0 = _series(rows, "A0")
    kappa = _series(rows, "kappa")
    delta = _series(rows, "delta")
    return {
        "schema": "t15_generated_trim50_idealized_envelope_v1",
        "source_boundary_param_dir": str(boundary_param_dir),
        "source_data_root": str(data_root),
        "train_shots": list(train_shots),
        "holdout_shots": list(holdout_shots),
        "ip": {
            "raw_min_a": float(np.min(ip)),
            "raw_max_a": float(np.max(ip)),
            "min_a": float(np.min(ip) * (2.0 - DEFAULT_PADDING)),
            "max_a": float(np.max(ip) * DEFAULT_PADDING),
            "abs_rate_max_aps": float(_max_abs_rate(rows, "Ip") * DEFAULT_PADDING),
        },
        "boundary": {
            "R0": {"raw_min": float(np.min(_series(rows, "R0"))), "raw_max": float(np.max(_series(rows, "R0")))},
            "Z0": {"raw_min": float(np.min(_series(rows, "Z0"))), "raw_max": float(np.max(_series(rows, "Z0")))},
            "A0": {
                "min": float(np.min(a0)),
                "max": float(np.max(a0)),
                "abs_rate_max": float(_max_abs_rate(rows, "A0") * DEFAULT_PADDING),
            },
            "elongation_excess": {
                "min": 0.0,
                "max": float((np.max(kappa) - 1.0) * SHAPE_MAX_PADDING),
                "abs_rate_max": float(_max_abs_rate(rows, "kappa") * DEFAULT_PADDING),
            },
            "delta": {
                "min": 0.0,
                "max": float(np.max(delta) * SHAPE_MAX_PADDING),
                "abs_rate_max": float(_max_abs_rate(rows, "delta") * DEFAULT_PADDING),
            },
        },
    }


def _build_reset_library(
    rows: list[dict[str, float | str]],
    *,
    data_root: Path,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
) -> dict[str, np.ndarray]:
    train_set = set(train_shots)
    holdout_set = set(holdout_shots)
    coils_by_shot = {shot: _load_coils(data_root / "coils" / f"t15md_{shot}_coils.csv") for shot in sorted(train_set | holdout_set)}
    shot_id: list[str] = []
    source_index: list[int] = []
    time_s: list[float] = []
    ip0: list[float] = []
    pfc0: list[np.ndarray] = []
    sol0: list[np.ndarray] = []
    params0: list[np.ndarray] = []
    split: list[str] = []
    for row in rows:
        shot = str(row["shot"])
        coil_t, coil_values = coils_by_shot[shot]
        nearest = int(np.argmin(np.abs(coil_t - float(row["t"]))))
        values = coil_values[nearest]
        if values.shape[0] != 9:
            raise ValueError(f"expected 9 coil columns for shot {shot}, got {values.shape[0]}")
        shot_id.append(shot)
        source_index.append(nearest)
        time_s.append(float(row["t"]))
        ip0.append(float(row["Ip"]))
        sol0.append(values[:3].astype(float))
        pfc0.append(values[3:].astype(float))
        params0.append(np.asarray([float(row[name]) for name in PARAM_COLUMNS], dtype=float))
        if shot in holdout_set:
            split.append("holdout")
        elif shot in train_set:
            split.append("train")
        else:
            raise ValueError(f"shot {shot} is not in train or holdout split")
    return {
        "schema": np.asarray("t15_generated_trim50_idealized_initial_states_v1"),
        "shot_id": np.asarray(shot_id, dtype="<U8"),
        "source_index": np.asarray(source_index, dtype=np.int64),
        "time_s": np.asarray(time_s, dtype=float),
        "ip0": np.asarray(ip0, dtype=float),
        "pfc0": np.asarray(pfc0, dtype=float),
        "sol0": np.asarray(sol0, dtype=float),
        "params0": np.asarray(params0, dtype=float),
        "split": np.asarray(split, dtype="<U8"),
    }


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


def _series(rows: Iterable[dict[str, float | str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _max_abs_rate(rows: list[dict[str, float | str]], key: str) -> float:
    best = 0.0
    by_shot: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        by_shot.setdefault(str(row["shot"]), []).append(row)
    for shot_rows in by_shot.values():
        shot_rows.sort(key=lambda r: float(r["t"]))
        values = _series(shot_rows, key)
        times = _series(shot_rows, "t")
        dt = np.diff(times)
        dv = np.diff(values)
        mask = np.isfinite(dt) & np.isfinite(dv) & (dt > 0.0)
        if np.any(mask):
            best = max(best, float(np.max(np.abs(dv[mask] / dt[mask]))))
    if best <= 0.0:
        raise ValueError(f"could not compute positive rate for {key}")
    return best


if __name__ == "__main__":
    raise SystemExit(main())
