#!/bin/bash
#SBATCH --job-name=t15-actgen-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/t15-actgen-build-%j.out
#SBATCH --error=slurm_logs/t15-actgen-build-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml}
export TRIM_MACHINE_CONFIG_SOURCE=${TRIM_MACHINE_CONFIG_SOURCE:-../tokamak-sim/configs/T15MD_new_data.toml}
export TRIM_DATA_ROOT=${TRIM_DATA_ROOT:-../tokamak-sim/data/t15_data_new_trim50}

export OBSERVED_RESET_LIBRARY=${OBSERVED_RESET_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export OBSERVED_TARGET_DIR=${OBSERVED_TARGET_DIR:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export OBSERVED_TARGETS=${OBSERVED_TARGETS:-${OBSERVED_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export OBSERVED_MACHINE_CONFIG=${OBSERVED_MACHINE_CONFIG:-data/processed/t15_new_trim50_plain_gpu1e6_machine_config.toml}
export OBSERVED_CONFIG_OUT=${OBSERVED_CONFIG_OUT:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_config.yaml}

export GENERATED_TARGET_DIR=${GENERATED_TARGET_DIR:-data/processed/t15_actuator_generated_trim50_plain_gpu1e6_0p1s}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_actuator_generated_trim50_plain_gpu1e6_0p1s_initial_states.npz}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${GENERATED_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export DIAGNOSTIC_TARGETS=${DIAGNOSTIC_TARGETS:-${GENERATED_TARGET_DIR}/t15_actuator_generated_targets.npz}
export ACTUATOR_TARGET_COUNT=${ACTUATOR_TARGET_COUNT:-12000}
export ACTUATOR_BUILD_BATCH_SIZE=${ACTUATOR_BUILD_BATCH_SIZE:-256}

for path in "${SOURCE_CONFIG}" "${TRIM_MACHINE_CONFIG_SOURCE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required file: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${TRIM_DATA_ROOT}/ip" || ! -d "${TRIM_DATA_ROOT}/coils" ]]; then
  echo "missing trim50 data root: ${TRIM_DATA_ROOT}" >&2
  exit 2
fi

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
    cd /workspace/tokamak-rl-v2

    python3 scripts/write_trim50_plain_gpu1e6_machine_config.py \
      --source "${TRIM_MACHINE_CONFIG_SOURCE}" \
      --out "${OBSERVED_MACHINE_CONFIG}"

    if [[ ! -f "${OBSERVED_RESET_LIBRARY}" || ! -f "${OBSERVED_TARGETS}" || ! -d "${OBSERVED_TARGET_DIR}" ]]; then
      python3 scripts/build_t15_new_replay_window_oracle_targets.py \
        --base-config "${SOURCE_CONFIG}" \
        --data-root "${TRIM_DATA_ROOT}" \
        --machine-config "${OBSERVED_MACHINE_CONFIG}" \
        --target-dir "${OBSERVED_TARGET_DIR}" \
        --initial-library-out "${OBSERVED_RESET_LIBRARY}" \
        --config-out "${OBSERVED_CONFIG_OUT}" \
        --compute-backend gpu \
        --gpu-device cuda:0 \
        --batch-size 256 \
        --window-steps 100 \
        --angles 32 \
        --train-shots 3856 3857 3858 3863 \
        --holdout-shots 3864
    fi

    python3 scripts/build_t15_actuator_generated_trim50_plain_gpu1e6_0p1s.py \
      --data-root "${TRIM_DATA_ROOT}" \
      --machine-config "${OBSERVED_MACHINE_CONFIG}" \
      --observed-targets "${OBSERVED_TARGETS}" \
      --out-dir "${GENERATED_TARGET_DIR}" \
      --initial-states-out "${RESET_LIBRARY}" \
      --targets-out "${ORACLE_TARGETS}" \
      --diagnostic-targets-out "${DIAGNOSTIC_TARGETS}" \
      --target-count "${ACTUATOR_TARGET_COUNT}" \
      --batch-size "${ACTUATOR_BUILD_BATCH_SIZE}" \
      --gpu-device cuda:0 \
      --plots

    python3 scripts/summarize_t15_actuator_generated_dataset.py \
      "${GENERATED_TARGET_DIR}" \
      --initial-states "${RESET_LIBRARY}" \
      --out-dir "${GENERATED_TARGET_DIR}/summary"
  '

echo "dataset_dir=${GENERATED_TARGET_DIR}"
echo "initial_states=${RESET_LIBRARY}"
echo "summary=${GENERATED_TARGET_DIR}/summary/actuator_generated_dataset_report.md"
