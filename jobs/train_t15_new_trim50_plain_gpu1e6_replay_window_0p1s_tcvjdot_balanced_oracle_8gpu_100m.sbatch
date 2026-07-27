#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-trim50-gpu1e6
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-trim50-gpu1e6-8gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-trim50-gpu1e6-8gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml}
export TRIM_MACHINE_CONFIG_SOURCE=${TRIM_MACHINE_CONFIG_SOURCE:-../tokamak-sim/configs/T15MD.toml}
export TRIM_DATA_ROOT=${TRIM_DATA_ROOT:-../tokamak-sim/data/t15_data_new_trim50_ip_calibrated}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export ORACLE_TARGET_DIR=${ORACLE_TARGET_DIR:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export ORACLE_MACHINE_CONFIG=${ORACLE_MACHINE_CONFIG:-../tokamak-sim/configs/T15MD.toml}
export ORACLE_CONFIG_OUT=${ORACLE_CONFIG_OUT:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_config.yaml}
export CONFIG_NAME=${CONFIG_NAME:-t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m}
export RUN_NAME=${RUN_NAME:-t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-trim50-gpu1e6-balanced-8gpu-100m-${SLURM_JOB_ID:-manual}}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-100000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}
export NUM_ENVS=${NUM_ENVS:-2048}
export GPUS_PER_NODE=8
export WORLD_SIZE=8
export WANDB_MODE=${WANDB_MODE:-online}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$(( 20000 + (${SLURM_JOB_ID:-1593} % 40000) ))}
export HOLDOUT_EVAL_EPISODES=${HOLDOUT_EVAL_EPISODES:-1239}

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "missing trim50 training config: ${SOURCE_CONFIG}" >&2
  echo "run the trim50 oracle target build job first" >&2
  exit 2
fi
if [[ ! -f "${TRIM_MACHINE_CONFIG_SOURCE}" ]]; then
  echo "missing T15MD config source: ${TRIM_MACHINE_CONFIG_SOURCE}" >&2
  exit 2
fi
if [[ ! -d "${TRIM_DATA_ROOT}" ]]; then
  echo "missing trim50 data root: ${TRIM_DATA_ROOT}" >&2
  exit 2
fi

if [[ ! -f "${RESET_LIBRARY}" || ! -f "${ORACLE_TARGETS}" || ! -d "${ORACLE_TARGET_DIR}" ]]; then
  echo "trim50 oracle reset/target libraries are missing; building them inside the container"
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

