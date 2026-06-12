#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-train
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/tokamak-rl-v2-train-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-train-%j.err

set -euo pipefail
cd /scratch/$USER/tokamak

export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/t15_static_boundary_full_${SLURM_JOB_ID:-manual}}
export RUN_NAME=${RUN_NAME:-t15_static_boundary_full_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-train-${SLURM_JOB_ID:-manual}}

srun --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
    set -euo pipefail

    mkdir -p /tmp/nvidia-lib
    ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
    ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
    ln -sf /host-libs/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so.1
    ln -sf /tmp/nvidia-lib/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so

    export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}

    cd /workspace/tokamak-rl-v2
    mkdir -p slurm_logs outputs

    python - <<PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYCUDA

    python scripts/train.py \
      --config configs/experiments/t15_static_boundary_gpu.yaml \
      --device cuda:0 \
      --sim-compute-backend gpu \
      --sim-gpu-device cuda:0 \
      --num-envs 96 \
      --actor-workers 3 \
      --actor-devices cuda:1,cuda:2,cuda:3 \
      --output-dir ${TRAIN_OUTPUT} \
      --wandb \
      --wandb-project ${WANDB_PROJECT_NAME} \
      --wandb-name ${RUN_NAME}
  '
