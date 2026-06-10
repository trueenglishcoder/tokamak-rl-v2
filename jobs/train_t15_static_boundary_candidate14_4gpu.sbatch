#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-c14
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/tokamak-rl-v2-c14-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-c14-%j.err

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

    export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}

    cd /workspace/tokamak-rl-v2

    python -c "import torch; print(\"cuda_available\", torch.cuda.is_available()); print(\"device_count\", torch.cuda.device_count()); print(\"devices\", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

    python scripts/train.py \
      --config configs/experiments/t15_static_boundary_gpu.yaml \
      --device cuda:0 \
      --sim-compute-backend gpu \
      --sim-gpu-device cuda:0 \
      --num-envs 96 \
      --actor-workers 3 \
      --actor-devices cuda:1,cuda:2,cuda:3 \
      --output-dir outputs/train_t15_static_boundary_candidate14_4gpu \
      --wandb \
      --wandb-project tokamak-rl-v2 \
      --wandb-name train_t15_static_boundary_candidate14_4gpu
  '
