#!/bin/bash
#SBATCH --job-name=t15-synth-emp60-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/t15-synth-emp60-build-%j.out
#SBATCH --error=slurm_logs/t15-synth-emp60-build-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs

export REAL_RESET_LIBRARY=${REAL_RESET_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export REAL_ORACLE_TARGET_DIR=${REAL_ORACLE_TARGET_DIR:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export REAL_ORACLE_TARGETS=${REAL_ORACLE_TARGETS:-${REAL_ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export SYNTH_OUT_DIR=${SYNTH_OUT_DIR:-data/processed/t15_synth_empirical_long60}
export SYNTH_TRAIN_PARENTS=${SYNTH_TRAIN_PARENTS:-60}
export SYNTH_HOLDOUT_PARENTS=${SYNTH_HOLDOUT_PARENTS:-6}
export SYNTH_PREVIEW_EXAMPLES=${SYNTH_PREVIEW_EXAMPLES:-6}
export SYNTH_MIN_STEPS=${SYNTH_MIN_STEPS:-1000}
export SYNTH_MAX_STEPS=${SYNTH_MAX_STEPS:-1500}
export SYNTH_WIGGLE_ROOM=${SYNTH_WIGGLE_ROOM:-1.15}
export SYNTH_SEED=${SYNTH_SEED:-61}

if [[ ! -f "${REAL_RESET_LIBRARY}" ]]; then
  echo "missing canonical real reset library: ${REAL_RESET_LIBRARY}" >&2
  exit 2
fi
if [[ ! -f "${REAL_ORACLE_TARGETS}" ]]; then
  echo "missing canonical real oracle targets: ${REAL_ORACLE_TARGETS}" >&2
  exit 2
fi

rm -rf "${SYNTH_OUT_DIR}"
mkdir -p "${SYNTH_OUT_DIR}"

srun \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace \
  bash -lc '
    set -euo pipefail
    cd /workspace/tokamak-rl-v2
    python3 scripts/build_t15_synth_empirical_long60.py \
      --oracle-target "${REAL_ORACLE_TARGETS}" \
      --initial-library "${REAL_RESET_LIBRARY}" \
      --out-dir "${SYNTH_OUT_DIR}" \
      --train-parents "${SYNTH_TRAIN_PARENTS}" \
      --holdout-parents "${SYNTH_HOLDOUT_PARENTS}" \
      --preview-examples "${SYNTH_PREVIEW_EXAMPLES}" \
      --min-steps "${SYNTH_MIN_STEPS}" \
      --max-steps "${SYNTH_MAX_STEPS}" \
      --wiggle-room "${SYNTH_WIGGLE_ROOM}" \
      --seed "${SYNTH_SEED}"
  '

test -f "${SYNTH_OUT_DIR}/t15_replay_window_oracle_targets.npz"
test -f "${SYNTH_OUT_DIR}/t15_synth_empirical_long60_initial_states.npz"
test -f "${SYNTH_OUT_DIR}/t15_synth_empirical_long60_report.md"

echo "synth_empirical_long60_dir=${SYNTH_OUT_DIR}"
echo "synth_empirical_long60_targets=${SYNTH_OUT_DIR}/t15_replay_window_oracle_targets.npz"
echo "synth_empirical_long60_initial=${SYNTH_OUT_DIR}/t15_synth_empirical_long60_initial_states.npz"
echo "synth_empirical_long60_report=${SYNTH_OUT_DIR}/t15_synth_empirical_long60_report.md"
