#!/bin/bash
#SBATCH --job-name=tokamak-build-ip15-equilibrium-lcfs
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/tokamak-build-ip15-equilibrium-lcfs-%j.out
#SBATCH --error=slurm_logs/tokamak-build-ip15-equilibrium-lcfs-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs data/processed

export REPLAY_ROOT=${REPLAY_ROOT:-runs/t15md_trim50_ip15_equilibrium_lcfs_top5_replay_gpu_full_${SLURM_JOB_ID}}
export TARGET_DIR=${TARGET_DIR:-data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_targets}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_initial_states.npz}
export CONFIG_OUT=${CONFIG_OUT:-configs/experiments/t15_ip15_equilibrium_lcfs_replay_window_0p1s_tcvjdot_mpo.yaml}

printf 'replay_root=%s\n' "$REPLAY_ROOT"
printf 'target_dir=%s\n' "$TARGET_DIR"
printf 'reset_library=%s\n' "$RESET_LIBRARY"
printf 'config_out=%s\n' "$CONFIG_OUT"

srun \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
    set -euo pipefail
    mkdir -p /tmp/nvidia-lib
    ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
    ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
    ln -sf /host-libs/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so.1
    ln -sf /tmp/nvidia-lib/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so
    export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}

    cd /workspace/tokamak-sim
    python3 scripts/run_t15md_limited_replay_dataset.py \
      --shots 3856 3857 3858 3863 3864 \
      --config configs/T15MD.toml \
      --data-root data/t15_data_new_trim50_ip_calibrated \
      --initial-prefix T15MD_ip15_equilibrium_lcfs \
      --out "$REPLAY_ROOT" \
      --angles 32 \
      --boundary-mode equilibrium_lcfs \
      --compute-backend gpu \
      --gpu-device cuda:0 \
      --no-video

    python3 - <<"PYREPLAYCHECK"
import os
from pathlib import Path

import numpy as np

root = Path(os.environ["REPLAY_ROOT"])
expected_shots = {"3856", "3857", "3858", "3863", "3864"}
required = {
    "version",
    "radii_true",
    "boundary_poly_true",
    "boundary_topology_code",
    "boundary_axis",
    "boundary_x_points",
    "boundary_limiter_contacts",
    "boundary_fixed_angle_valid",
    "boundary_fixed_angle_counts",
    "boundary_quality",
}
artifacts_by_shot: dict[str, list[Path]] = {shot: [] for shot in expected_shots}

for path in sorted(root.glob("t15md_limited_replay_*/**/run*.npz")):
    shot = next(
        (
            part.rsplit("_", 1)[-1]
            for part in path.parts
            if part.startswith("t15md_limited_replay_")
        ),
        None,
    )
    if shot in artifacts_by_shot:
        artifacts_by_shot[shot].append(path)

artifact_counts = {shot: len(paths) for shot, paths in artifacts_by_shot.items()}
if any(count != 1 for count in artifact_counts.values()):
    raise RuntimeError(
        "expected exactly one fresh replay artifact per shot, "
        f"got {artifact_counts} under {root}"
    )

for shot in sorted(expected_shots, key=int):
    path = artifacts_by_shot[shot][0]
    with np.load(path, allow_pickle=False) as run:
        missing = sorted(required - set(run.files))
        if missing:
            raise RuntimeError(f"{path}: missing schema-v4 keys {missing}")

        version = np.asarray(run["version"]).reshape(-1)
        if version.size != 1 or int(version[0]) != 4:
            raise RuntimeError(
                f"{path}: expected version=4, got {version.tolist()}"
            )

        radii = np.asarray(run["radii_true"], dtype=float)
        polygon = np.asarray(run["boundary_poly_true"], dtype=float)
        topology = np.asarray(run["boundary_topology_code"], dtype=np.int64).reshape(-1)
        axis = np.asarray(run["boundary_axis"], dtype=float)
        x_points = np.asarray(run["boundary_x_points"], dtype=float)
        contacts = np.asarray(run["boundary_limiter_contacts"], dtype=float)
        fixed_valid = np.asarray(
            run["boundary_fixed_angle_valid"],
            dtype=bool,
        ).reshape(-1)
        counts = np.asarray(
            run["boundary_fixed_angle_counts"],
            dtype=np.int64,
        )
        quality = np.asarray(run["boundary_quality"], dtype=float)

        if radii.ndim != 2 or radii.shape[1] != 32:
            raise RuntimeError(
                f"{path}: expected radii_true shape (steps, 32), got {radii.shape}"
            )
        steps = int(radii.shape[0])
        expected_shapes = {
            "boundary_fixed_angle_valid": (steps,),
            "boundary_fixed_angle_counts": (steps, 32),
            "boundary_topology_code": (steps,),
            "boundary_axis": (steps, 2),
            "boundary_quality": (steps, 6),
        }
        actual_shapes = {
            "boundary_fixed_angle_valid": fixed_valid.shape,
            "boundary_fixed_angle_counts": counts.shape,
            "boundary_topology_code": topology.shape,
            "boundary_axis": axis.shape,
            "boundary_quality": quality.shape,
        }
        mismatched = {
            key: {"expected": expected_shapes[key], "actual": actual_shapes[key]}
            for key in expected_shapes
            if actual_shapes[key] != expected_shapes[key]
        }
        if mismatched:
            raise RuntimeError(f"{path}: LCFS channel shape mismatch {mismatched}")
        if polygon.ndim != 3 or polygon.shape[0] != steps or polygon.shape[2] != 2:
            raise RuntimeError(f"{path}: invalid boundary_poly_true shape {polygon.shape}")
        if x_points.ndim != 3 or x_points.shape[0] != steps or x_points.shape[2] != 2:
            raise RuntimeError(f"{path}: invalid boundary_x_points shape {x_points.shape}")
        if contacts.ndim != 3 or contacts.shape[0] != steps or contacts.shape[2] != 2:
            raise RuntimeError(
                f"{path}: invalid boundary_limiter_contacts shape {contacts.shape}"
            )

        radii_ok = np.isfinite(radii).all(axis=1) & (radii > 0.0).all(axis=1)
        polygon_finite = np.isfinite(polygon).all(axis=2)
        polygon_points = polygon_finite.sum(axis=1)
        polygon_closed = np.linalg.norm(polygon[:, 0, :] - polygon[:, -1, :], axis=1) <= 1.0e-8
        topology_ok = np.isin(topology, np.asarray([1, 2, 3, 4]))
        counts_ok = (counts == 1).all(axis=1)
        axis_ok = np.isfinite(axis).all(axis=1)
        quality_ok = np.isfinite(quality).all(axis=1)
        x_count = np.isfinite(x_points).all(axis=2).sum(axis=1)
        contact_count = np.isfinite(contacts).all(axis=2).sum(axis=1)
        topology_detail_ok = (
            ((topology == 1) & (x_count == 0) & (contact_count >= 1))
            | ((topology == 2) & (x_count == 1))
            | ((topology == 3) & (x_count == 2))
            | ((topology == 4) & (x_count >= 3))
        )

        invalid = np.flatnonzero(
            ~fixed_valid
            | ~radii_ok
            | (polygon_points < 4)
            | ~polygon_closed
            | ~topology_ok
            | ~counts_ok
            | ~axis_ok
            | ~quality_ok
            | ~topology_detail_ok
        )
        if invalid.size:
            preview = invalid[:20].tolist()
            raise RuntimeError(
                f"{path}: inconsistent full-GPU LCFS artifacts at steps {preview}; "
                f"invalid_count={int(invalid.size)}"
            )

        print(
            "validated_full_gpu_replay",
            shot,
            "steps",
            steps,
            "path",
            path,
        )
PYREPLAYCHECK

    cd /workspace/tokamak-rl-v2
    python3 scripts/build_t15_new_replay_window_oracle_targets.py \
      --base-config configs/experiments/t15_ip15_equilibrium_lcfs_replay_window_0p1s_tcvjdot_mpo_balanced.yaml \
      --data-root ../tokamak-sim/data/t15_data_new_trim50_ip_calibrated \
      --machine-config ../tokamak-sim/configs/T15MD.toml \
      --target-dir "$TARGET_DIR" \
      --initial-library-out "$RESET_LIBRARY" \
      --config-out "$CONFIG_OUT" \
      --window-steps 100 \
      --angles 32 \
      --compute-backend gpu \
      --gpu-device cuda:0 \
      --train-shots 3856 3857 3858 3863 \
      --holdout-shots 3864

    python3 - <<"PYCHECK"
import json
import os
from pathlib import Path

summary_path = Path(os.environ["TARGET_DIR"]) / "oracle_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_shots = {"3856", "3857", "3858", "3863", "3864"}
actual_shots = set(summary["accepted_by_shot"])
if actual_shots != expected_shots:
    raise SystemExit(
        "oracle dataset shots mismatch: "
        f"expected={sorted(expected_shots)} actual={sorted(actual_shots)}"
    )
if int(summary["split_counts"].get("holdout", 0)) <= 0:
    raise SystemExit("oracle dataset has no holdout windows")
if int(summary["split_counts"].get("train", 0)) <= 0:
    raise SystemExit("oracle dataset has no train windows")
print(json.dumps(summary, indent=2, sort_keys=True))
PYCHECK
  '
