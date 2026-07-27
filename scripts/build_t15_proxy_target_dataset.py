#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokamak_control.io.config_io import load_config
from tokamak_rl_v2.data.target_trajectories import build_target_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build target-only T15 proxy Ip/boundary trajectory windows.")
    parser.add_argument("--seed-target-library", type=Path, default=Path("data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/t15_replay_window_oracle_targets.npz"))
    parser.add_argument("--initial-library", type=Path, default=Path("data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz"))
    parser.add_argument("--machine-envelope", type=Path, default=Path("configs/machine/t15_proxy_v1.yaml"))
    parser.add_argument("--sim-config", type=Path, default=Path("../tokamak-sim/configs/T15MD.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/t15_proxy_target_v1"))
    parser.add_argument("--theta-count", type=int, default=32)
    parser.add_argument("--train-parents", type=int, default=48)
    parser.add_argument("--holdout-parents", type=int, default=8)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--window-steps", type=int, default=100)
    parser.add_argument("--window-stride-steps", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)

    sim_config = _repo_path(args.sim_config)
    cfg = load_config(sim_config)
    if cfg.limiter_shape is None:
        raise ValueError(f"sim config has no limiter geometry: {sim_config}")
    summary = build_target_dataset(
        target_seed_path=_repo_path(args.seed_target_library),
        initial_library_path=_repo_path(args.initial_library),
        machine_envelope_path=_repo_path(args.machine_envelope),
        out_dir=_repo_path(args.out_dir),
        limiter_shape=np.asarray(cfg.limiter_shape, dtype=float),
        boundary_center=(float(cfg.physics.R0), float(cfg.physics.Z0)),
        theta_count=int(args.theta_count),
        train_parents=int(args.train_parents),
        holdout_parents=int(args.holdout_parents),
        min_steps=int(args.min_steps),
        max_steps=int(args.max_steps),
        window_steps=int(args.window_steps),
        window_stride_steps=int(args.window_stride_steps),
        dt=float(args.dt),
        seed=int(args.seed),
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def _repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
