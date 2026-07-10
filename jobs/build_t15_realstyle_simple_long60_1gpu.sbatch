#!/bin/bash
#SBATCH --job-name=t15-realstyle60-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/%u/tokamak/tokamak-rl-v2/slurm_logs/t15-realstyle60-build-%j.out
#SBATCH --error=/scratch/%u/tokamak/tokamak-rl-v2/slurm_logs/t15-realstyle60-build-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs

export REAL_RESET_LIBRARY=${REAL_RESET_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export REAL_ORACLE_TARGET_DIR=${REAL_ORACLE_TARGET_DIR:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export REAL_ORACLE_TARGETS=${REAL_ORACLE_TARGETS:-${REAL_ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export REALSTYLE_OUT_DIR=${REALSTYLE_OUT_DIR:-data/processed/t15_realstyle_simple_long60}
export REALSTYLE_INITIAL_OUT=${REALSTYLE_INITIAL_OUT:-data/processed/t15_realstyle_simple_long60_initial_states.npz}
export PREVIEW_ONLY=${PREVIEW_ONLY:-0}
export PREVIEW_EXAMPLES=${PREVIEW_EXAMPLES:-6}
export TRAIN_PARENTS=${TRAIN_PARENTS:-60}
export HOLDOUT_PARENTS=${HOLDOUT_PARENTS:-6}
export MIN_STEPS=${MIN_STEPS:-500}
export MAX_STEPS=${MAX_STEPS:-1500}
export RESET_MAX_SOURCE_INDEX=${RESET_MAX_SOURCE_INDEX:-500}
export WIGGLE_ROOM=${WIGGLE_ROOM:-1.15}
export SEED=${SEED:-71}

if [[ ! -f "${REAL_RESET_LIBRARY}" ]]; then
  echo "missing canonical real reset library: ${REAL_RESET_LIBRARY}" >&2
  exit 2
fi
if [[ ! -f "${REAL_ORACLE_TARGETS}" ]]; then
  echo "missing canonical real oracle targets: ${REAL_ORACLE_TARGETS}" >&2
  exit 2
fi

rm -rf "${REALSTYLE_OUT_DIR}"
rm -f "${REALSTYLE_INITIAL_OUT}"
mkdir -p "${REALSTYLE_OUT_DIR}"

srun \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace \
  bash -lc '
    set -euo pipefail
    cd /workspace/tokamak-rl-v2
    PREVIEW_ARG=()
    if [[ "${PREVIEW_ONLY}" == "1" ]]; then
      PREVIEW_ARG=(--preview-only)
    fi
    python3 scripts/build_t15_realstyle_simple_long60.py \
      --oracle-target "${REAL_ORACLE_TARGETS}" \
      --initial-library "${REAL_RESET_LIBRARY}" \
      --out-dir "${REALSTYLE_OUT_DIR}" \
      --initial-library-out "${REALSTYLE_INITIAL_OUT}" \
      --preview-examples "${PREVIEW_EXAMPLES}" \
      --train-parents "${TRAIN_PARENTS}" \
      --holdout-parents "${HOLDOUT_PARENTS}" \
      --min-steps "${MIN_STEPS}" \
      --max-steps "${MAX_STEPS}" \
      --reset-max-source-index "${RESET_MAX_SOURCE_INDEX}" \
      --wiggle-room "${WIGGLE_ROOM}" \
      --seed "${SEED}" \
      "${PREVIEW_ARG[@]}"
  '

test -f "${REALSTYLE_OUT_DIR}/report.md"
test -f "${REALSTYLE_OUT_DIR}/summary.json"
if [[ "${PREVIEW_ONLY}" != "1" ]]; then
  test -f "${REALSTYLE_OUT_DIR}/t15_replay_window_oracle_targets.npz"
  test -f "${REALSTYLE_INITIAL_OUT}"
fi

echo "realstyle_simple_long60_dir=${REALSTYLE_OUT_DIR}"
echo "realstyle_simple_long60_report=${REALSTYLE_OUT_DIR}/report.md"
echo "realstyle_simple_long60_summary=${REALSTYLE_OUT_DIR}/summary.json"
if [[ "${PREVIEW_ONLY}" != "1" ]]; then
  echo "realstyle_simple_long60_targets=${REALSTYLE_OUT_DIR}/t15_replay_window_oracle_targets.npz"
  echo "realstyle_simple_long60_initial=${REALSTYLE_INITIAL_OUT}"
fi
