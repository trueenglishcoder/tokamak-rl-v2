#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-qsearch
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/tokamak-rl-v2-quality-search-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-quality-search-%j.err

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

    export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    export RUN_NAME=reward_search_t15_static_boundary_tracking_primary_2h_${SLURM_JOB_ID:-manual}
    export WANDB_PROJECT_NAME=tokamak-rl-v2-${SLURM_JOB_ID:-manual}

    cd /workspace/tokamak-rl-v2

    python - <<PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYCUDA

    python scripts/search_rewards.py \
      --config configs/experiments/t15_static_boundary_gpu.yaml \
      --strategy successive_halving \
      --output-dir outputs/${RUN_NAME} \
      --stage-steps 12000,24000,36000 \
      --stage-keep 2,1,1 \
      --stage-eval-episodes 4,8,12 \
      --baseline-policy no_control \
      --require-no-control-improvement \
      --parallel-candidates 4 \
      --gpu-devices 0,1,2,3 \
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
      --actor-update-chunk-size 2048 \
      --eval-max-steps 500 \
      --device cuda \
      --sim-compute-backend gpu \
      --sim-gpu-device cuda:0 \
      --shape-good-values 0.003 \
      --shape-bad-values 0.04,0.08 \
      --ip-good-values 500 \
      --ip-bad-values 15000,40000 \
      --current-good-values 100 \
      --current-bad-values 50000 \
      --terminal-reward-values -5 \
      --reward-scale-values 0.1 \
      --wandb \
      --wandb-project ${WANDB_PROJECT_NAME} \
      --wandb-name ${RUN_NAME} \
      --wandb-group ${RUN_NAME} \
      --wandb-mode online
  '
