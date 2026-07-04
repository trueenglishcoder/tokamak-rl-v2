#!/bin/bash
#SBATCH --job-name=t15-synlong30-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/t15-synlong30-build-%j.out
#SBATCH --error=slurm_logs/t15-synlong30-build-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs

export REAL_RESET_LIBRARY=${REAL_RESET_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export REAL_ORACLE_TARGET_DIR=${REAL_ORACLE_TARGET_DIR:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export REAL_ORACLE_TARGETS=${REAL_ORACLE_TARGETS:-${REAL_ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export ORACLE_TARGET_DIR=${ORACLE_TARGET_DIR:-data/processed/t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export SYNTHETIC_TRAIN_PARENTS=${SYNTHETIC_TRAIN_PARENTS:-30}
export SYNTHETIC_HOLDOUT_PARENTS=${SYNTHETIC_HOLDOUT_PARENTS:-1}
export SYNTHETIC_MIN_STEPS=${SYNTHETIC_MIN_STEPS:-1000}
export SYNTHETIC_MAX_STEPS=${SYNTHETIC_MAX_STEPS:-1500}
export SYNTHETIC_SEED=${SYNTHETIC_SEED:-37}
export SYNTHETIC_WIGGLE_ROOM=${SYNTHETIC_WIGGLE_ROOM:-1.15}

if [[ ! -f "${REAL_RESET_LIBRARY}" ]]; then
  echo "missing canonical real reset library: ${REAL_RESET_LIBRARY}" >&2
  exit 2
fi
if [[ ! -f "${REAL_ORACLE_TARGETS}" ]]; then
  echo "missing canonical real oracle targets: ${REAL_ORACLE_TARGETS}" >&2
  exit 2
fi

rm -f "${RESET_LIBRARY}"
rm -rf "${ORACLE_TARGET_DIR}"
mkdir -p "${ORACLE_TARGET_DIR}"

srun \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace \
  bash -lc '
    set -euo pipefail
    cd /workspace/tokamak-rl-v2
    python3 scripts/build_t15_synthetic_long30_trim50_plain_gpu1e6_oracle_windows.py \
      --oracle-target "${REAL_ORACLE_TARGETS}" \
      --initial-library "${REAL_RESET_LIBRARY}" \
      --target-dir "${ORACLE_TARGET_DIR}" \
      --initial-library-out "${RESET_LIBRARY}" \
      --train-parents "${SYNTHETIC_TRAIN_PARENTS}" \
      --holdout-parents "${SYNTHETIC_HOLDOUT_PARENTS}" \
      --min-steps "${SYNTHETIC_MIN_STEPS}" \
      --max-steps "${SYNTHETIC_MAX_STEPS}" \
      --seed "${SYNTHETIC_SEED}" \
      --wiggle-room "${SYNTHETIC_WIGGLE_ROOM}"
  '

test -f "${RESET_LIBRARY}"
test -f "${ORACLE_TARGETS}"

echo "synthetic_long30_reset_library=${RESET_LIBRARY}"
echo "synthetic_long30_oracle_targets=${ORACLE_TARGETS}"
echo "synthetic_long30_report=${ORACLE_TARGET_DIR}/synthetic_long30_report.md"
