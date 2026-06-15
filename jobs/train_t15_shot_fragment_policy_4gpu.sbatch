#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-shot-frag-4gpu
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/tokamak-rl-v2-shot-frag-4gpu-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-shot-frag-4gpu-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_shot_fragment_boundary_ip_500_gpu.yaml}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/t15_shot_fragment_policy_4gpu_${SLURM_JOB_ID:-manual}}
export RUN_NAME=${RUN_NAME:-t15_shot_fragment_policy_4gpu_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-shot-fragment-4gpu-${SLURM_JOB_ID:-manual}}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}

# Distributed training counts --steps as environment/global steps.
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-10000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}

# GPU 0 runs the learner/evaluation. GPUs 1-3 run simulator actor workers.
export NUM_ENVS=${NUM_ENVS:-96}
export ACTOR_WORKERS=${ACTOR_WORKERS:-3}
export ACTOR_DEVICES=${ACTOR_DEVICES:-cuda:1,cuda:2,cuda:3}
export LEARNER_DEVICE=${LEARNER_DEVICE:-cuda:0}
export LEARNER_SIM_GPU_DEVICE=${LEARNER_SIM_GPU_DEVICE:-cuda:0}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-384}

if (( NUM_ENVS % ACTOR_WORKERS != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by ACTOR_WORKERS=${ACTOR_WORKERS}" >&2
  exit 2
fi

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

    cd /workspace/tokamak-rl-v2
    mkdir -p slurm_logs outputs

    python3 - <<PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYCUDA

    echo "policy_config=${POLICY_CONFIG}"
    echo "train_output=${TRAIN_OUTPUT}"
    echo "train_env_steps=${TRAIN_ENV_STEPS}"
    echo "eval_env_steps=${EVAL_ENV_STEPS}"
    echo "num_envs=${NUM_ENVS}"
    echo "actor_workers=${ACTOR_WORKERS}"
    echo "actor_devices=${ACTOR_DEVICES}"
    echo "learner_device=${LEARNER_DEVICE}"
    echo "replay_capacity_episodes=${REPLAY_CAPACITY_EPISODES}"
    echo "save_checkpoints=${SAVE_CHECKPOINTS}"
    echo "checkpoint_env_steps=${CHECKPOINT_ENV_STEPS}"

    CHECKPOINT_ARGS=()
    if [[ "${SAVE_CHECKPOINTS}" == "1" || "${SAVE_CHECKPOINTS}" == "true" ]]; then
      CHECKPOINT_ARGS=(--save-checkpoints --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}")
    fi

    python3 scripts/train_policy_pipeline.py \
      --config ${POLICY_CONFIG} \
      --output-dir ${TRAIN_OUTPUT} \
      --steps ${TRAIN_ENV_STEPS} \
      --device ${LEARNER_DEVICE} \
      --sim-compute-backend gpu \
      --sim-gpu-device ${LEARNER_SIM_GPU_DEVICE} \
      --num-envs ${NUM_ENVS} \
      --actor-workers ${ACTOR_WORKERS} \
      --actor-devices ${ACTOR_DEVICES} \
      --batch-size 256 \
      --unroll-length 128 \
      --replay-capacity-episodes ${REPLAY_CAPACITY_EPISODES} \
      --rollout-chunk-length 128 \
      --updates-per-rollout-chunk 32 \
      --hidden-dim 256 \
      --critic-hidden-dim 256 \
      --critic-mlp-hidden-dim 256 \
      --action-samples 20 \
      --actor-update-chunk-size 2048 \
      "${CHECKPOINT_ARGS[@]}" \
      --eval-interval-steps ${EVAL_ENV_STEPS} \
      --wandb \
      --wandb-project ${WANDB_PROJECT_NAME} \
      --wandb-name ${RUN_NAME} \
      --wandb-group ${RUN_NAME} \
      --wandb-mode ${WANDB_MODE} \
      --controller-rollout-steps 500 \
      --allow-failed-gates

    echo "train_policy_pipeline completed at $(date -Is)"
  '
