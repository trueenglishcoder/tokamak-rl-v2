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
from tokamak_rl_v2.data.target_trajectories import audit_target_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit balanced target-only T15 proxy trajectory dataset geometry and coverage.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/t15_proxy_target_balanced_v1"))
    parser.add_argument("--sim-config", type=Path, default=Path("../tokamak-sim/configs/T15MD.toml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/t15_proxy_target_balanced_v1"))
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args(argv)

    cfg = load_config(_repo_path(args.sim_config))
    if cfg.limiter_shape is None:
        raise ValueError("sim config has no limiter geometry")
    summary = audit_target_dataset(
        dataset_dir=_repo_path(args.dataset_dir),
        limiter_shape=np.asarray(cfg.limiter_shape, dtype=float),
        boundary_center=(float(cfg.physics.R0), float(cfg.physics.Z0)),
        dt=float(args.dt),
        out_dir=_repo_path(args.out_dir),
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
