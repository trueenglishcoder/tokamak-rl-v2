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

# local_replay counts --steps as global environment steps.
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-1000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-100000}
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-100000}

# Same effective workload as the 1-GPU job, sharded over 4 ranks:
# 1-GPU: 32 envs, batch 256, replay 256, chunk 64, updates/chunk 16.
# 4-GPU: 8 envs/rank, batch 64/rank, replay 64/rank, synchronized gradients.
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export NUM_ENVS=${NUM_ENVS:-32}
export BATCH_SIZE=${BATCH_SIZE:-64}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-64}
export ROLLOUT_CHUNK_LENGTH=${ROLLOUT_CHUNK_LENGTH:-64}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-16}

if (( NUM_ENVS % NPROC_PER_NODE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
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
    echo "nproc_per_node=${NPROC_PER_NODE}"
    echo "batch_size_per_rank=${BATCH_SIZE}"
    echo "replay_capacity_episodes=${REPLAY_CAPACITY_EPISODES}"
    echo "rollout_chunk_length=${ROLLOUT_CHUNK_LENGTH}"
    echo "updates_per_rollout_chunk=${UPDATES_PER_ROLLOUT_CHUNK}"
    echo "save_checkpoints=${SAVE_CHECKPOINTS}"
    echo "checkpoint_env_steps=${CHECKPOINT_ENV_STEPS}"

    CHECKPOINT_ARGS=()
    if [[ "${SAVE_CHECKPOINTS}" == "1" || "${SAVE_CHECKPOINTS}" == "true" ]]; then
      CHECKPOINT_ARGS=(--save-checkpoints --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}")
    fi

    torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} scripts/train_policy_pipeline.py \
      --config ${POLICY_CONFIG} \
      --output-dir ${TRAIN_OUTPUT} \
      --steps ${TRAIN_ENV_STEPS} \
      --device auto \
      --sim-compute-backend gpu \
      --num-envs ${NUM_ENVS} \
      --distributed-mode local_replay \
      --batch-size ${BATCH_SIZE} \
      --replay-capacity-episodes ${REPLAY_CAPACITY_EPISODES} \
      --rollout-chunk-length ${ROLLOUT_CHUNK_LENGTH} \
      --updates-per-rollout-chunk ${UPDATES_PER_ROLLOUT_CHUNK} \
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
      --controller-rollout-steps 1000 \
      --allow-failed-gates

    echo "train_policy_pipeline completed at $(date -Is)"
  '
