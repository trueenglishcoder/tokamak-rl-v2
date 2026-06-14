#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-reward-cal
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/tokamak-rl-v2-reward-cal-1gpu-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-reward-cal-1gpu-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_hold_reset_boundary_ip_stage1_gpu.yaml}
export CALIBRATION_OUTPUT=${CALIBRATION_OUTPUT:-outputs/t15_physical_reward_calibration_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-reward-calibration-${SLURM_JOB_ID:-manual}}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}

export NUM_ENVS=${NUM_ENVS:-32}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-2000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-250000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-250000}
export MAX_CANDIDATES=${MAX_CANDIDATES:-12}
export STOP_ON_PASS=${STOP_ON_PASS:-1}

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
    echo "num_envs=${NUM_ENVS}"
    echo "train_env_steps=${TRAIN_ENV_STEPS}"
    echo "eval_env_steps=${EVAL_ENV_STEPS}"
    echo "checkpoint_env_steps=${CHECKPOINT_ENV_STEPS}"
    echo "max_candidates=${MAX_CANDIDATES}"
    echo "stop_on_pass=${STOP_ON_PASS}"

    STOP_FLAG=""
    if [ "${STOP_ON_PASS}" = "1" ]; then
      STOP_FLAG="--stop-on-pass"
    fi

    python3 scripts/calibrate_physical_reward.py \
      --base-config "${POLICY_CONFIG}" \
      --output-root "${CALIBRATION_OUTPUT}" \
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
      --wandb-mode "${WANDB_MODE}" \
      ${STOP_FLAG}

    echo "reward calibration completed at $(date -Is)"
  '
