#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-reward-cal-array
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --array=1-12%8
#SBATCH --output=slurm_logs/tokamak-rl-v2-reward-cal-array-%A_%a.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-reward-cal-array-%A_%a.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_hold_reset_boundary_ip_stage1_gpu.yaml}
export CALIBRATION_OUTPUT=${CALIBRATION_OUTPUT:-outputs/t15_physical_reward_calibration_${SLURM_ARRAY_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-reward-calibration-${SLURM_ARRAY_JOB_ID:-manual}}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}

export NUM_ENVS=${NUM_ENVS:-32}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-500000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-125000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-125000}
export MAX_CANDIDATES=${MAX_CANDIDATES:-12}

if [ "${SLURM_ARRAY_TASK_ID}" -gt "${MAX_CANDIDATES}" ]; then
  echo "array task ${SLURM_ARRAY_TASK_ID} is above MAX_CANDIDATES=${MAX_CANDIDATES}; exiting"
  exit 0
fi

srun \
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
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}

    cd /workspace/tokamak-rl-v2
    mkdir -p slurm_logs outputs

    python3 - <<PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYCUDA

    echo "policy_config=${POLICY_CONFIG}"
    echo "calibration_output=${CALIBRATION_OUTPUT}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "num_envs=${NUM_ENVS}"
    echo "train_env_steps=${TRAIN_ENV_STEPS}"
    echo "eval_env_steps=${EVAL_ENV_STEPS}"
    echo "checkpoint_env_steps=${CHECKPOINT_ENV_STEPS}"
    echo "max_candidates=${MAX_CANDIDATES}"

    python3 scripts/calibrate_physical_reward.py \
      --base-config "${POLICY_CONFIG}" \
      --output-root "${CALIBRATION_OUTPUT}" \
      --candidate-index "${SLURM_ARRAY_TASK_ID}" \
      --train-env-steps "${TRAIN_ENV_STEPS}" \
      --eval-env-steps "${EVAL_ENV_STEPS}" \
      --checkpoint-env-steps "${CHECKPOINT_ENV_STEPS}" \
      --num-envs "${NUM_ENVS}" \
      --max-candidates "${MAX_CANDIDATES}" \
      --device cuda:0 \
      --sim-compute-backend gpu \
      --sim-gpu-device cuda:0 \
      --batch-size 256 \
      --unroll-length 64 \
      --replay-capacity-episodes 128 \
      --rollout-chunk-length 64 \
      --updates-per-rollout-chunk 24 \
      --action-samples 20 \
      --actor-update-chunk-size 2048 \
      --wandb \
      --wandb-project "${WANDB_PROJECT_NAME}" \
      --wandb-mode "${WANDB_MODE}"

    echo "reward calibration candidate ${SLURM_ARRAY_TASK_ID} completed at $(date -Is)"
  '
