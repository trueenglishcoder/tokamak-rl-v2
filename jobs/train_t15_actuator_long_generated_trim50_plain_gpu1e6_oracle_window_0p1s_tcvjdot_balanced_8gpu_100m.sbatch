#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-actlong
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-actuator-long-generated-8gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-actuator-long-generated-8gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export BUILDER_SCRIPT=${BUILDER_SCRIPT:-scripts/build_t15_actuator_long_generated_trim50_plain_gpu1e6_0p1s.py}
export GENERATED_TARGET_DIR=${GENERATED_TARGET_DIR:-data/processed/t15_actuator_long_generated_trim50_plain_gpu1e6_0p1s}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_actuator_long_generated_trim50_plain_gpu1e6_0p1s_initial_states.npz}
export ORACLE_TARGET_DIR=${ORACLE_TARGET_DIR:-${GENERATED_TARGET_DIR}}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export DIAGNOSTIC_TARGETS=${DIAGNOSTIC_TARGETS:-${GENERATED_TARGET_DIR}/t15_actuator_long_generated_targets.npz}

export CONFIG_NAME=${CONFIG_NAME:-t15_actuator_long_generated_trim50_plain_gpu1e6_oracle_window_0p1s_tcvjdot_balanced_8gpu_100m}
export RUN_NAME=${RUN_NAME:-t15_actuator_long_generated_trim50_plain_gpu1e6_oracle_window_0p1s_tcvjdot_balanced_8gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-actuator-long-generated-oracle-8gpu-100m-${SLURM_JOB_ID:-manual}}

if [[ ! -f "${RESET_LIBRARY}" || ! -f "${ORACLE_TARGETS}" || ! -f "${DIAGNOSTIC_TARGETS}" ]]; then
  echo "missing prebuilt long-parent actuator-generated dataset." >&2
  echo "Build and inspect it first:" >&2
  echo "  sbatch jobs/build_t15_actuator_long_generated_trim50_plain_gpu1e6_0p1s_1gpu.sbatch" >&2
  echo "Expected artifacts:" >&2
  echo "  ${RESET_LIBRARY}" >&2
  echo "  ${ORACLE_TARGETS}" >&2
  echo "  ${DIAGNOSTIC_TARGETS}" >&2
  exit 2
fi

bash jobs/train_t15_actuator_generated_trim50_plain_gpu1e6_oracle_window_0p1s_tcvjdot_balanced_8gpu_100m.sbatch
