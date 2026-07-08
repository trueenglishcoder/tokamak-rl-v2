#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokamak_control.io.config_io import load_config
from tokamak_rl_v2.data.target_preview import write_target_preview
from tokamak_rl_v2.data.target_trajectories import build_target_dataset, parse_difficulty_fractions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a small preview of balanced target-only T15 proxy trajectories.")
    parser.add_argument("--seed-target-library", type=Path, default=Path("data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/t15_replay_window_oracle_targets.npz"))
    parser.add_argument("--initial-library", type=Path, default=Path("data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz"))
    parser.add_argument("--machine-envelope", type=Path, default=Path("configs/machine/t15_proxy_v1.yaml"))
    parser.add_argument("--sim-config", type=Path, default=Path("../tokamak-sim/configs/T15MD_new_data.toml"))
    parser.add_argument("--preview-dir", type=Path, default=Path("data/previews/t15_proxy_target_balanced_v1_preview"))
    parser.add_argument("--theta-count", type=int, default=32)
    parser.add_argument("--train-parents", type=int, default=4)
    parser.add_argument("--holdout-parents", type=int, default=2)
    parser.add_argument("--min-steps", type=int, default=180)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--window-steps", type=int, default=100)
    parser.add_argument("--window-stride-steps", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-train-windows", type=int, default=240)
    parser.add_argument("--target-holdout-windows", type=int, default=96)
    parser.add_argument("--target-difficulty-fractions", type=str, default="hold=0.20,slow=0.35,medium=0.30,fast=0.15")
    parser.add_argument("--example-count", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    preview_dir = _repo_path(args.preview_dir)
    dataset_dir = preview_dir / "dataset"
    if preview_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"preview directory exists; pass --overwrite to replace it: {preview_dir}")
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    sim_config = _repo_path(args.sim_config)
    cfg = load_config(sim_config)
    if cfg.limiter_shape is None:
        raise ValueError(f"sim config has no limiter geometry: {sim_config}")

    summary = build_target_dataset(
        target_seed_path=_repo_path(args.seed_target_library),
        initial_library_path=_repo_path(args.initial_library),
        machine_envelope_path=_repo_path(args.machine_envelope),
        out_dir=dataset_dir,
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
        target_train_windows=int(args.target_train_windows),
        target_holdout_windows=int(args.target_holdout_windows),
        target_difficulty_fractions=parse_difficulty_fractions(args.target_difficulty_fractions),
    )
    preview = write_target_preview(
        dataset_dir=dataset_dir,
        out_dir=preview_dir,
        example_count=int(args.example_count),
        title="T15 proxy balanced target-only preview",
    )
    out = {
        "dataset_summary": summary.to_dict(),
        "preview": preview.to_dict(),
    }
    (preview_dir / "preview_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"preview_html={preview.html_path}")
    print(f"preview_index={preview.index_path}")
    print(f"preview_ready={preview_dir / 'PREVIEW_READY'}")
    return 0


def _repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
