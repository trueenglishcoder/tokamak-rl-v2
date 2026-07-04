#!/bin/bash
#SBATCH --job-name=t15-long-target-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/t15-long-target-build-%j.out
#SBATCH --error=slurm_logs/t15-long-target-build-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs

export PARENT_COUNT=${PARENT_COUNT:-12}
export PARENT_MIN_STEPS=${PARENT_MIN_STEPS:-1000}
export PARENT_MAX_STEPS=${PARENT_MAX_STEPS:-1200}
export PARENT_STRIDE=${PARENT_STRIDE:-1}
export PERTURB_FRACTION=${PERTURB_FRACTION:-0.035}
export STATE_DISTANCE_LIMIT=${STATE_DISTANCE_LIMIT:-0.18}
export TARGET_DIR=${TARGET_DIR:-data/processed/t15_long_target_generated_trim50_plain_gpu1e6_oracle_window_0p1s}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_long_target_generated_trim50_plain_gpu1e6_oracle_window_0p1s_initial_states.npz}
export TARGETS=${TARGETS:-${TARGET_DIR}/t15_long_target_generated_targets.npz}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export PARENTS=${PARENTS:-${TARGET_DIR}/t15_long_target_generated_parents.npz}

srun \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
    set -euo pipefail
    mkdir -p /tmp/nvidia-lib
    ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
    ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
    export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2/scripts:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    cd /workspace/tokamak-rl-v2

    python3 scripts/build_t15_long_target_generated_trim50_idealized_matched_0p1s.py \
      --parent-count "${PARENT_COUNT}" \
      --parent-min-steps "${PARENT_MIN_STEPS}" \
      --parent-max-steps "${PARENT_MAX_STEPS}" \
      --parent-stride "${PARENT_STRIDE}" \
      --perturb-fraction "${PERTURB_FRACTION}" \
      --state-distance-limit "${STATE_DISTANCE_LIMIT}" \
      --out-dir "${TARGET_DIR}" \
      --initial-states-out "${RESET_LIBRARY}" \
      --targets-out "${TARGETS}" \
      --oracle-targets-out "${ORACLE_TARGETS}" \
      --parents-out "${PARENTS}"

    echo
    echo "===== GENERATED DATASET REPORT ====="
    cat "${TARGET_DIR}/summary/long_target_generated_dataset_report.md"
    echo
    echo "===== GENERATED FILES ====="
    find "${TARGET_DIR}" -maxdepth 3 -type f | sort
  '
