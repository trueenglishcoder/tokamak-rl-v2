#!/bin/bash
#SBATCH --job-name=t15-proxy-target-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/t15-proxy-target-build-%j.out
#SBATCH --error=slurm_logs/t15-proxy-target-build-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs

export SEED_TARGET_LIBRARY=${SEED_TARGET_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/t15_replay_window_oracle_targets.npz}
export INITIAL_LIBRARY=${INITIAL_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export MACHINE_ENVELOPE=${MACHINE_ENVELOPE:-configs/machine/t15_proxy_v1.yaml}
export SIM_CONFIG=${SIM_CONFIG:-../tokamak-sim/configs/T15MD.toml}
export TARGET_OUT_DIR=${TARGET_OUT_DIR:-data/processed/t15_proxy_target_v1}
export TARGET_PREVIEW_DIR=${TARGET_PREVIEW_DIR:-data/previews/t15_proxy_target_v1_preview}
export REQUIRE_TARGET_PREVIEW=${REQUIRE_TARGET_PREVIEW:-1}
export TARGET_TRAIN_PARENTS=${TARGET_TRAIN_PARENTS:-48}
export TARGET_HOLDOUT_PARENTS=${TARGET_HOLDOUT_PARENTS:-8}
export TARGET_MIN_STEPS=${TARGET_MIN_STEPS:-1000}
export TARGET_MAX_STEPS=${TARGET_MAX_STEPS:-1500}
export TARGET_WINDOW_STEPS=${TARGET_WINDOW_STEPS:-100}
export TARGET_WINDOW_STRIDE_STEPS=${TARGET_WINDOW_STRIDE_STEPS:-1}
export TARGET_SEED=${TARGET_SEED:-123}

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
  echo "target dataset v1 requires TARGET_WINDOW_STRIDE_STEPS=1 for fully overlapping windows" >&2
  exit 2
fi
if [[ "${REQUIRE_TARGET_PREVIEW}" == "1" && ! -f "${TARGET_PREVIEW_DIR}/PREVIEW_READY" ]]; then
  echo "missing target preview marker: ${TARGET_PREVIEW_DIR}/PREVIEW_READY" >&2
  echo "Run this before the full build: sbatch jobs/preview_t15_proxy_target_dataset.sbatch" >&2
  echo "Set REQUIRE_TARGET_PREVIEW=0 only for deliberate non-interactive rebuilds." >&2
  exit 2
fi

if [[ -f "${TARGET_PREVIEW_DIR}/preview.html" ]]; then
  echo "using_target_preview=${TARGET_PREVIEW_DIR}/preview.html"
fi

rm -rf "${TARGET_OUT_DIR}"
mkdir -p "${TARGET_OUT_DIR}"

srun \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace \
  bash -lc '
    set -euo pipefail
    cd /workspace/tokamak-rl-v2
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    python3 scripts/build_t15_proxy_target_dataset.py \
      --seed-target-library "${SEED_TARGET_LIBRARY}" \
      --initial-library "${INITIAL_LIBRARY}" \
      --machine-envelope "${MACHINE_ENVELOPE}" \
      --sim-config "${SIM_CONFIG}" \
      --out-dir "${TARGET_OUT_DIR}" \
      --train-parents "${TARGET_TRAIN_PARENTS}" \
      --holdout-parents "${TARGET_HOLDOUT_PARENTS}" \
      --min-steps "${TARGET_MIN_STEPS}" \
      --max-steps "${TARGET_MAX_STEPS}" \
      --window-steps "${TARGET_WINDOW_STEPS}" \
      --window-stride-steps "${TARGET_WINDOW_STRIDE_STEPS}" \
      --seed "${TARGET_SEED}"
    python3 scripts/audit_t15_proxy_target_dataset.py \
      --dataset-dir "${TARGET_OUT_DIR}" \
      --sim-config "${SIM_CONFIG}" \
      --out-dir "data/diagnostics/t15_proxy_target_v1"
  '

test -f "${TARGET_OUT_DIR}/t15_target_trajectory_targets.npz"
test -f "${TARGET_OUT_DIR}/t15_proxy_target_v1_initial_states.npz"
test -f "${TARGET_OUT_DIR}/summary.json"
test -f "data/diagnostics/t15_proxy_target_v1/target_feasibility_summary.json"

echo "target_dataset_dir=${TARGET_OUT_DIR}"
echo "target_dataset_targets=${TARGET_OUT_DIR}/t15_target_trajectory_targets.npz"
echo "target_dataset_initial=${TARGET_OUT_DIR}/t15_proxy_target_v1_initial_states.npz"
echo "target_dataset_summary=${TARGET_OUT_DIR}/summary.json"
echo "target_dataset_audit=data/diagnostics/t15_proxy_target_v1/target_feasibility_summary.json"
