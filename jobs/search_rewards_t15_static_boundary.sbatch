#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-rewards
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/tokamak-rl-v2-reward-search-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-reward-search-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

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

    export LD_LIBRARY_PATH=/tmp/nvidia-lib:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:$PYTHONPATH

    cd /workspace/tokamak-rl-v2

    python - <<PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("device_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PYCUDA

    python scripts/search_rewards.py \
      --config configs/experiments/t15_static_boundary_gpu.yaml \
      --strategy successive_halving \
      --output-dir outputs/reward_search_t15_static_boundary_intentional_v2 \
      --stage-steps 10000,50000,150000 \
      --stage-keep 12,4,2 \
      --stage-eval-episodes 16,32,64 \
      --baseline-policy no_control \
      --refine-top-k 4 \
      --refine-midpoints \
      --num-envs 64 \
      --batch-size 256 \
      --unroll-length 64 \
      --replay-capacity-episodes 64 \
      --rollout-chunk-length 64 \
      --updates-per-rollout-chunk 16 \
      --hidden-dim 256 \
      --critic-hidden-dim 256 \
      --critic-mlp-hidden-dim 256 \
      --action-samples 20 \
      --actor-update-chunk-size 1024 \
      --eval-max-steps 1000 \
      --device cuda \
      --sim-compute-backend gpu \
      --sim-gpu-device cuda:0 \
      --shape-good-values 0.003,0.005 \
      --shape-bad-values 0.04,0.06,0.08 \
      --ip-good-values 500,1000 \
      --ip-bad-values 15000,25000,40000 \
      --current-good-values 100 \
      --current-bad-values 50000 \
      --terminal-reward-values -5 \
      --reward-scale-values 0.1 \
      --wandb \
      --wandb-project tokamak-rl-v2 \
      --wandb-name reward_search_t15_static_boundary_intentional_v2 \
      --wandb-group reward_search_intentional \
      --wandb-mode online
  '
