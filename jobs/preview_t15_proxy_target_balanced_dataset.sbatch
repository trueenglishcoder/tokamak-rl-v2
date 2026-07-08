#!/bin/bash
#SBATCH --job-name=t15-proxy-target-balanced-preview
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/t15-proxy-target-balanced-preview-%j.out
#SBATCH --error=slurm_logs/t15-proxy-target-balanced-preview-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs

export SEED_TARGET_LIBRARY=${SEED_TARGET_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/t15_replay_window_oracle_targets.npz}
export INITIAL_LIBRARY=${INITIAL_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export MACHINE_ENVELOPE=${MACHINE_ENVELOPE:-configs/machine/t15_proxy_v1.yaml}
export SIM_CONFIG=${SIM_CONFIG:-../tokamak-sim/configs/T15MD_new_data.toml}
export TARGET_PREVIEW_DIR=${TARGET_PREVIEW_DIR:-data/previews/t15_proxy_target_balanced_v1_preview}
export TARGET_PREVIEW_TRAIN_PARENTS=${TARGET_PREVIEW_TRAIN_PARENTS:-4}
export TARGET_PREVIEW_HOLDOUT_PARENTS=${TARGET_PREVIEW_HOLDOUT_PARENTS:-2}
export TARGET_PREVIEW_MIN_STEPS=${TARGET_PREVIEW_MIN_STEPS:-180}
export TARGET_PREVIEW_MAX_STEPS=${TARGET_PREVIEW_MAX_STEPS:-240}
export TARGET_WINDOW_STEPS=${TARGET_WINDOW_STEPS:-100}
export TARGET_WINDOW_STRIDE_STEPS=${TARGET_WINDOW_STRIDE_STEPS:-1}
export TARGET_SEED=${TARGET_SEED:-123}
export TARGET_PREVIEW_TRAIN_WINDOWS=${TARGET_PREVIEW_TRAIN_WINDOWS:-240}
export TARGET_PREVIEW_HOLDOUT_WINDOWS=${TARGET_PREVIEW_HOLDOUT_WINDOWS:-96}
export TARGET_DIFFICULTY_FRACTIONS=${TARGET_DIFFICULTY_FRACTIONS:-hold=0.20,slow=0.35,medium=0.30,fast=0.15}
export TARGET_PREVIEW_EXAMPLES=${TARGET_PREVIEW_EXAMPLES:-8}

if [[ ! -f "${SEED_TARGET_LIBRARY}" ]]; then
  echo "missing seed target library: ${SEED_TARGET_LIBRARY}" >&2
  exit 2
fi
if [[ ! -f "${INITIAL_LIBRARY}" ]]; then
  echo "missing initial-state library: ${INITIAL_LIBRARY}" >&2
  exit 2
fi
if [[ ! -f "${MACHINE_ENVELOPE}" ]]; then
  echo "missing machine envelope: ${MACHINE_ENVELOPE}" >&2
  exit 2
fi
if [[ ! -f "${SIM_CONFIG}" ]]; then
  echo "missing tokamak-sim config: ${SIM_CONFIG}" >&2
  exit 2
fi
if [[ "${TARGET_WINDOW_STRIDE_STEPS}" != "1" ]]; then
  echo "balanced target dataset requires TARGET_WINDOW_STRIDE_STEPS=1 for fully overlapping windows" >&2
  exit 2
fi

rm -rf "${TARGET_PREVIEW_DIR}"

srun \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace \
  bash -lc '
    set -euo pipefail
    cd /workspace/tokamak-rl-v2
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    python3 scripts/preview_t15_proxy_target_balanced_dataset.py \
      --seed-target-library "${SEED_TARGET_LIBRARY}" \
      --initial-library "${INITIAL_LIBRARY}" \
      --machine-envelope "${MACHINE_ENVELOPE}" \
      --sim-config "${SIM_CONFIG}" \
      --preview-dir "${TARGET_PREVIEW_DIR}" \
      --train-parents "${TARGET_PREVIEW_TRAIN_PARENTS}" \
      --holdout-parents "${TARGET_PREVIEW_HOLDOUT_PARENTS}" \
      --min-steps "${TARGET_PREVIEW_MIN_STEPS}" \
      --max-steps "${TARGET_PREVIEW_MAX_STEPS}" \
      --window-steps "${TARGET_WINDOW_STEPS}" \
      --window-stride-steps "${TARGET_WINDOW_STRIDE_STEPS}" \
      --seed "${TARGET_SEED}" \
      --target-train-windows "${TARGET_PREVIEW_TRAIN_WINDOWS}" \
      --target-holdout-windows "${TARGET_PREVIEW_HOLDOUT_WINDOWS}" \
      --target-difficulty-fractions "${TARGET_DIFFICULTY_FRACTIONS}" \
      --example-count "${TARGET_PREVIEW_EXAMPLES}" \
      --overwrite
  '

test -f "${TARGET_PREVIEW_DIR}/PREVIEW_READY"
test -f "${TARGET_PREVIEW_DIR}/preview.html"
test -f "${TARGET_PREVIEW_DIR}/preview_index.json"
test -f "${TARGET_PREVIEW_DIR}/dataset/summary.json"

echo "target_preview_dir=${TARGET_PREVIEW_DIR}"
echo "target_preview_html=${TARGET_PREVIEW_DIR}/preview.html"
echo "target_preview_index=${TARGET_PREVIEW_DIR}/preview_index.json"
echo "target_preview_summary=${TARGET_PREVIEW_DIR}/preview_summary.json"
