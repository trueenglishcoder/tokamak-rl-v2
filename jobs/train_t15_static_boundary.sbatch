#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/tokamak-rl-v2-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-%j.err

set -euo pipefail
cd /scratch/$USER/tokamak

srun --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
    mkdir -p /tmp/nvidia-lib
    ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
    ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
    ln -sf /host-libs/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so.1
    ln -sf /tmp/nvidia-lib/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so
    export LD_LIBRARY_PATH=/tmp/nvidia-lib:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:$PYTHONPATH
    cd /workspace/tokamak-rl-v2
    python scripts/train.py --config configs/experiments/t15_static_boundary_gpu.yaml --device cuda --wandb --wandb-project tokamak-rl-v2 --wandb-name t15_static_boundary_gpu
  '
