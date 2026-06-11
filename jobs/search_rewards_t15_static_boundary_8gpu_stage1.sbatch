#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-rs8-s1
#SBATCH --partition=batch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/tokamak-rl-v2-rs8-stage1-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-rs8-stage1-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

export SEARCH_ROOT=${SEARCH_ROOT:-outputs/reward_search_t15_static_boundary_8gpu}
export RUN_NAME=${RUN_NAME:-reward_search_t15_static_boundary_8gpu_stage1_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-reward-8gpu-stage1-${SLURM_JOB_ID:-manual}}

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

    python - <<PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYCUDA

    python scripts/search_rewards.py \
      --config configs/experiments/t15_static_boundary_gpu.yaml \
      --strategy successive_halving \
      --stage-only stage_01_short \
      --output-dir ${SEARCH_ROOT} \
      --stage-steps 20000,60000,120000 \
      --stage-keep 6,2,1 \
      --stage-eval-episodes 4,8,16 \
      --stage-max-shape-degradation-m 0.025,0.020,0.015 \
      --stage-min-ip-improvement-a 2000,8000,15000 \
      --baseline-policy no_control \
      --parallel-candidates 8 \
      --gpu-devices 0,1,2,3,4,5,6,7 \
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
      --ip-bad-values 25000,40000 \
      --shape-weight-values 2.0,4.0 \
      --ip-weight-values 1.5,3.0 \
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
