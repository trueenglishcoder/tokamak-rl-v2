#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-shot-frag
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/tokamak-rl-v2-shot-frag-1gpu-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-shot-frag-1gpu-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_shot_fragment_boundary_ip_500_gpu.yaml}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/t15_shot_fragment_policy_1gpu_1000_${SLURM_JOB_ID:-manual}}
export RUN_NAME=${RUN_NAME:-t15_shot_fragment_policy_1gpu_1000_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-shot-fragment-1gpu-${SLURM_JOB_ID:-manual}}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}

export NUM_ENVS=${NUM_ENVS:-32}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-1000000}
export TRAIN_DECISION_STEPS=$(( (TRAIN_ENV_STEPS + NUM_ENVS - 1) / NUM_ENVS ))
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-100000}
export EVAL_DECISION_STEPS=$(( (EVAL_ENV_STEPS + NUM_ENVS - 1) / NUM_ENVS ))
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-100000}
export CHECKPOINT_DECISION_STEPS=$(( (CHECKPOINT_ENV_STEPS + NUM_ENVS - 1) / NUM_ENVS ))
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-256}

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
    echo "train_env_steps=${TRAIN_ENV_STEPS}"
    echo "num_envs=${NUM_ENVS}"
    echo "train_decision_steps=${TRAIN_DECISION_STEPS}"
    echo "eval_env_steps=${EVAL_ENV_STEPS}"
    echo "eval_decision_steps=${EVAL_DECISION_STEPS}"
    echo "save_checkpoints=${SAVE_CHECKPOINTS}"
    echo "checkpoint_env_steps=${CHECKPOINT_ENV_STEPS}"
    echo "checkpoint_decision_steps=${CHECKPOINT_DECISION_STEPS}"

    CHECKPOINT_ARGS=()
    if [[ "${SAVE_CHECKPOINTS}" == "1" || "${SAVE_CHECKPOINTS}" == "true" ]]; then
      CHECKPOINT_ARGS=(--save-checkpoints --checkpoint-interval-steps "${CHECKPOINT_DECISION_STEPS}")
    fi

    python3 scripts/train_policy_pipeline.py \
      --config ${POLICY_CONFIG} \
      --output-dir ${TRAIN_OUTPUT} \
      --steps ${TRAIN_DECISION_STEPS} \
      --device cuda:0 \
      --sim-compute-backend gpu \
      --sim-gpu-device cuda:0 \
      --num-envs ${NUM_ENVS} \
      --actor-workers 1 \
      --batch-size 256 \
      --replay-capacity-episodes ${REPLAY_CAPACITY_EPISODES} \
      --hidden-dim 256 \
      --critic-hidden-dim 256 \
      --critic-mlp-hidden-dim 256 \
      --action-samples 20 \
      --actor-update-chunk-size 2048 \
      "${CHECKPOINT_ARGS[@]}" \
      --eval-interval-steps ${EVAL_DECISION_STEPS} \
      --wandb \
      --wandb-project ${WANDB_PROJECT_NAME} \
      --wandb-name ${RUN_NAME} \
      --wandb-group ${RUN_NAME} \
      --wandb-mode ${WANDB_MODE} \
      --controller-rollout-steps 1000 \
      --allow-failed-gates

    echo "train_policy_pipeline completed at $(date -Is)"
  '
