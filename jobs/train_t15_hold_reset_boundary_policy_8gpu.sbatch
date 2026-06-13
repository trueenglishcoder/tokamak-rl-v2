#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-policy
#SBATCH --partition=batch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/tokamak-rl-v2-policy-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-policy-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_hold_reset_boundary_ip_stage1_gpu.yaml}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/t15_hold_reset_boundary_policy_${SLURM_JOB_ID:-manual}}
export RUN_NAME=${RUN_NAME:-t15_hold_reset_boundary_policy_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-policy-${SLURM_JOB_ID:-manual}}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}

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
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
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

    python3 scripts/train_policy_pipeline.py \
      --config ${POLICY_CONFIG} \
      --output-dir ${TRAIN_OUTPUT} \
      ${TRAIN_STEPS:+--steps ${TRAIN_STEPS}} \
      --device cuda:0 \
      --sim-compute-backend gpu \
      --sim-gpu-device cuda:0 \
      --num-envs 224 \
      --actor-workers 7 \
      --actor-devices cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
      --batch-size 256 \
      --unroll-length 64 \
      --replay-capacity-episodes 256 \
      --rollout-chunk-length 64 \
      --updates-per-rollout-chunk 24 \
      --hidden-dim 256 \
      --critic-hidden-dim 256 \
      --critic-mlp-hidden-dim 256 \
      --action-samples 20 \
      --actor-update-chunk-size 2048 \
      --wandb \
      --wandb-project ${WANDB_PROJECT_NAME} \
      --wandb-name ${RUN_NAME} \
      --wandb-group ${RUN_NAME} \
      --wandb-mode ${WANDB_MODE} \
      --allow-failed-gates

    echo "train_policy_pipeline completed at $(date -Is)"
  '
