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

export REPLAY_ROOT=${REPLAY_ROOT:-../tokamak-sim/runs/t15md_trim50_ip15_equilibrium_lcfs_top5_replay}
export TARGET_DIR=${TARGET_DIR:-data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_targets}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_initial_states.npz}
export CONFIG_OUT=${CONFIG_OUT:-configs/experiments/t15_ip15_equilibrium_lcfs_replay_window_0p1s_tcvjdot_mpo.yaml}

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
      --out runs/t15md_trim50_ip15_equilibrium_lcfs_top5_replay \
      --angles 32 \
      --boundary-mode equilibrium_lcfs \
      --compute-backend gpu \
      --gpu-device cuda:0 \
      --no-video

    python3 - <<"PYREPLAYCHECK"
from pathlib import Path

import numpy as np

root = Path("runs/t15md_trim50_ip15_equilibrium_lcfs_top5_replay")
required = {
    "artifact_version",
    "radii_true",
    "boundary_found",
    "boundary_topology_code",
    "boundary_fixed_angle_valid",
    "boundary_fixed_angle_counts",
}

artifacts = []

for path in sorted(root.rglob("*.npz")):
    try:
        with np.load(path, allow_pickle=False) as run:
            if not required.issubset(set(run.files)):
                continue

            version = np.asarray(run["artifact_version"]).reshape(-1)
            if version.size != 1 or int(version[0]) != 4:
                raise RuntimeError(
                    f"{path}: expected artifact_version=4, "
                    f"got {version.tolist()}"
                )

            radii = np.asarray(run["radii_true"])
            found = np.asarray(run["boundary_found"])
            topology = np.asarray(run["boundary_topology_code"])
            valid = np.asarray(run["boundary_fixed_angle_valid"])
            counts = np.asarray(run["boundary_fixed_angle_counts"])

            if radii.ndim != 2 or radii.shape[1] != 32:
                raise RuntimeError(
                    f"{path}: expected radii_true shape (steps, 32), "
                    f"got {radii.shape}"
                )

            steps = radii.shape[0]
            for name, value in (
                ("boundary_found", found),
                ("boundary_topology_code", topology),
                ("boundary_fixed_angle_valid", valid),
                ("boundary_fixed_angle_counts", counts),
            ):
                if value.shape[0] != steps:
                    raise RuntimeError(
                        f"{path}: {name} length {value.shape[0]} "
                        f"does not match {steps} replay steps"
                    )

            artifacts.append(path)

    except (OSError, ValueError):
        continue

if len(artifacts) < 5:
    raise RuntimeError(
        "Expected at least five schema-v4 replay artifacts, "
        f"found {len(artifacts)}"
    )

print("validated_replay_artifacts")
for path in artifacts:
    print(path)
PYREPLAYCHECK

    cd /workspace/tokamak-rl-v2
    python3 scripts/build_t15_new_replay_window_oracle_targets.py \
      --base-config configs/experiments/t15_ip15_equilibrium_lcfs_replay_window_0p1s_tcvjdot_mpo_balanced.yaml \
      --data-root ../tokamak-sim/data/t15_data_new_trim50_ip_calibrated \
      --machine-config ../tokamak-sim/configs/T15MD.toml \
      --replay-root ../tokamak-sim/runs/t15md_trim50_ip15_equilibrium_lcfs_top5_replay \
      --target-dir data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_targets \
      --initial-library-out data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_initial_states.npz \
      --config-out configs/experiments/t15_ip15_equilibrium_lcfs_replay_window_0p1s_tcvjdot_mpo.yaml \
      --window-steps 100 \
      --angles 32 \
      --train-shots 3856 3857 3858 3863 \
      --holdout-shots 3864

    python3 - <<"PYCHECK"
import json
from pathlib import Path

summary_path = Path("data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_targets/oracle_summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_shots = {"3856", "3857", "3858", "3863", "3864"}
actual_shots = set(summary["accepted_by_shot"])
if actual_shots != expected_shots:
    raise SystemExit(f"oracle dataset shots mismatch: expected={sorted(expected_shots)} actual={sorted(actual_shots)}")
if int(summary["split_counts"].get("holdout", 0)) <= 0:
    raise SystemExit("oracle dataset has no holdout windows")
if int(summary["split_counts"].get("train", 0)) <= 0:
    raise SystemExit("oracle dataset has no train windows")
print(json.dumps(summary, indent=2, sort_keys=True))
PYCHECK
  '
