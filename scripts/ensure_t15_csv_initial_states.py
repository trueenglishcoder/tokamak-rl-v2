#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import sys
from pathlib import Path

import numpy as np

from build_t15_csv_initial_state_library import main as build_initial_states
from tokamak_rl_v2.env.t15_csv_initial_states import CsvInitialStateLibrary
from tokamak_rl_v2.env.t15_reference_limits import load_reference_limits
from tokamak_rl_v2.training.policy_pipeline import _validate_initial_state_summary


def _valid(path: Path, summary_path: Path, reference_limits_path: Path) -> bool:
    if not path.exists() or not summary_path.exists():
        return False
    try:
        _validate_initial_state_summary(summary_path)
        limits = load_reference_limits(reference_limits_path)
        with np.load(path, allow_pickle=False) as data:
            expected = {"shot_id", "source_index", "time_s", "ip0", "pfc0", "sol0", "split"}
            if set(data.files) != expected:
                raise ValueError(f"arrays must be exactly {sorted(expected)}, got {sorted(data.files)}")
            n_pfc = int(np.asarray(data["pfc0"]).shape[1])
            n_sol = int(np.asarray(data["sol0"]).shape[1])
            ip0 = np.asarray(data["ip0"], dtype=float).reshape(-1)
            if ip0.size == 0:
                raise ValueError("ip0 array is empty")
            if float(np.nanmin(ip0)) < float(limits.ip_p01_a) or float(np.nanmax(ip0)) > float(limits.ip_p99_a):
                raise ValueError(
                    "reset Ip is outside current production reference bounds "
                    f"[{float(limits.ip_p01_a):.6g}, {float(limits.ip_p99_a):.6g}]"
                )
        CsvInitialStateLibrary(path, n_pfc=n_pfc, n_sol=n_sol, split="train")
        CsvInitialStateLibrary(path, n_pfc=n_pfc, n_sol=n_sol, split="holdout")
    except Exception as exc:
        print(f"{path} is stale or invalid: {exc}", file=sys.stderr, flush=True)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or rebuild the cheap processed T15 CSV reset-state artifact.")
    parser.add_argument("--path", default="data/processed/t15_new_trim50_plain_gpu1e6_csv_initial_states.npz")
    parser.add_argument("--summary", default="data/processed/t15_new_trim50_plain_gpu1e6_csv_initial_states.json")
    parser.add_argument("--rejected", default="data/processed/t15_new_trim50_plain_gpu1e6_csv_initial_states_rejected.csv")
    parser.add_argument("--reference-limits", default="data/processed/t15_reference_limits.json")
    parser.add_argument("--experiment-config", default="configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-rows-per-shot", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.path)
    summary = Path(args.summary)
    rejected = Path(args.rejected)
    reference_limits = Path(args.reference_limits)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not args.force and _valid(path, summary, reference_limits):
            print(f"{path} is valid", flush=True)
            return 0
        rc = build_initial_states(
            [
                "--experiment-config",
                str(args.experiment_config),
                "--out-npz",
                str(path),
                "--out-json",
                str(summary),
                "--out-rejected",
                str(rejected),
                "--reference-limits",
                str(reference_limits),
                "--workers",
                str(int(args.workers)),
                "--max-rows-per-shot",
                str(int(args.max_rows_per_shot)),
            ]
        )
        if int(rc) != 0:
            return int(rc)
        if not _valid(path, summary, reference_limits):
            print(f"{path} is still invalid after rebuild", file=sys.stderr, flush=True)
            return 2
        print(f"{path} rebuilt and valid", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
