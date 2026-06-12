#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-rs-control
#SBATCH --partition=batch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --time=06:30:00
#SBATCH --output=slurm_logs/tokamak-rl-v2-rs-control-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-rs-control-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

export SEARCH_ROOT=outputs/reward_search_t15_static_boundary_control_discovery_${SLURM_JOB_ID:-manual}
export RUN_NAME=reward_search_t15_static_boundary_control_discovery_${SLURM_JOB_ID:-manual}
export WANDB_PROJECT_NAME=tokamak-rl-v2-reward-control-discovery-${SLURM_JOB_ID:-manual}

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

    python - <<PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYCUDA

    python scripts/search_rewards.py \
      --config configs/experiments/t15_static_boundary_gpu.yaml \
      --strategy successive_halving \
      --candidate-preset control_discovery \
      --ranking-profile control_discovery \
      --output-dir ${SEARCH_ROOT} \
      --stage-steps 30000,100000,250000 \
      --stage-keep 8,3,1 \
      --stage-eval-episodes 12,24,48 \
      --stage-max-shape-error-m 0.20,0.17,0.145 \
      --stage-max-shape-degradation-m 0.08,0.06,0.045 \
      --stage-min-ip-improvement-a 10000,30000,50000 \
      --stage-max-ip-error-a 150000,120000,100000 \
      --stage-min-action-rms 0.004,0.006,0.008 \
      --stage-max-action-rms 0.70,0.65,0.60 \
      --stage-min-policy-weight-max 0.055,0.060,0.065 \
      --stage-min-sampled-q-spread 0.000001,0.000003,0.000005 \
      --stage-min-actor-param-delta-norm 0.000001,0.000003,0.000005 \
      --baseline-policy no_control \
      --parallel-candidates 8 \
      --gpu-devices 0,1,2,3,4,5,6,7 \
      --num-envs 64 \
      --batch-size 256 \
      --unroll-length 64 \
      --replay-capacity-episodes 128 \
      --rollout-chunk-length 64 \
      --updates-per-rollout-chunk 24 \
      --hidden-dim 256 \
      --critic-hidden-dim 256 \
      --critic-mlp-hidden-dim 256 \
      --action-samples 20 \
      --actor-update-chunk-size 2048 \
      --eval-max-steps 500 \
      --device cuda \
      --sim-compute-backend gpu \
      --sim-gpu-device cuda:0 \
      --wandb \
      --wandb-project ${WANDB_PROJECT_NAME} \
      --wandb-name ${RUN_NAME} \
      --wandb-group ${RUN_NAME} \
      --wandb-mode online
  '
