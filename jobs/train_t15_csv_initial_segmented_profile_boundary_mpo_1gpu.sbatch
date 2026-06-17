#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-csv-mpo
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/tokamak-rl-v2-csv-mpo-1gpu-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-csv-mpo-1gpu-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/t15_csv_initial_segmented_profile_boundary_mpo_1gpu_${SLURM_JOB_ID:-manual}}
export RUN_NAME=${RUN_NAME:-t15_csv_initial_segmented_profile_boundary_mpo_1gpu_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-csv-segmented-profile-1gpu-${SLURM_JOB_ID:-manual}}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}

export NUM_ENVS=${NUM_ENVS:-256}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-20000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-100000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-500000}
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-512}
export BATCH_SIZE=${BATCH_SIZE:-256}
export ROLLOUT_CHUNK_LENGTH=${ROLLOUT_CHUNK_LENGTH:-64}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-32}
export PREPARE_DATA=${PREPARE_DATA:-0}
export PREPARE_WORKERS=${PREPARE_WORKERS:-8}
export PREPARE_MAX_ROWS_PER_SHOT=${PREPARE_MAX_ROWS_PER_SHOT:-600}

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
    mkdir -p slurm_logs outputs data/processed

    python3 - <<PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYCUDA

    echo "policy_config=${POLICY_CONFIG}"
    echo "train_output=${TRAIN_OUTPUT}"
    echo "train_env_steps=${TRAIN_ENV_STEPS}"
    echo "num_envs=${NUM_ENVS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "replay_capacity_episodes=${REPLAY_CAPACITY_EPISODES}"
    echo "rollout_chunk_length=${ROLLOUT_CHUNK_LENGTH}"
    echo "updates_per_rollout_chunk=${UPDATES_PER_ROLLOUT_CHUNK}"
    echo "eval_env_steps=${EVAL_ENV_STEPS}"
    echo "save_checkpoints=${SAVE_CHECKPOINTS}"
    echo "checkpoint_env_steps=${CHECKPOINT_ENV_STEPS}"
    echo "prepare_data=${PREPARE_DATA}"
    echo "prepare_workers=${PREPARE_WORKERS}"
    echo "prepare_max_rows_per_shot=${PREPARE_MAX_ROWS_PER_SHOT}"

    if [[ ! -f data/processed/t15_reference_limits.json || ! -f data/processed/t15_csv_initial_states.npz || "${PREPARE_DATA}" == "1" || "${PREPARE_DATA}" == "true" ]]; then
      python3 scripts/build_t15_reference_limits.py
      python3 scripts/build_t15_csv_initial_state_library.py \
        --experiment-config "${POLICY_CONFIG}" \
        --workers "${PREPARE_WORKERS}" \
        --max-rows-per-shot "${PREPARE_MAX_ROWS_PER_SHOT}"
    fi

    CHECKPOINT_ARGS=()
    if [[ "${SAVE_CHECKPOINTS}" == "1" || "${SAVE_CHECKPOINTS}" == "true" ]]; then
      CHECKPOINT_ARGS=(--save-checkpoints --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}")
    else
      CHECKPOINT_ARGS=(--no-save-checkpoints)
    fi

    python3 scripts/train_policy_pipeline.py \
      --config "${POLICY_CONFIG}" \
      --output-dir "${TRAIN_OUTPUT}" \
      --steps "${TRAIN_ENV_STEPS}" \
      --device cuda:0 \
      --sim-compute-backend gpu \
      --sim-gpu-device cuda:0 \
      --num-envs "${NUM_ENVS}" \
      --distributed-mode local_replay \
      --actor-workers 1 \
      --batch-size "${BATCH_SIZE}" \
      --replay-capacity-episodes "${REPLAY_CAPACITY_EPISODES}" \
      --rollout-chunk-length "${ROLLOUT_CHUNK_LENGTH}" \
      --updates-per-rollout-chunk "${UPDATES_PER_ROLLOUT_CHUNK}" \
      --hidden-dim 256 \
      --critic-hidden-dim 256 \
      --critic-mlp-hidden-dim 256 \
      --action-samples 20 \
      --actor-update-chunk-size 2048 \
      "${CHECKPOINT_ARGS[@]}" \
      --eval-interval-steps "${EVAL_ENV_STEPS}" \
      --eval-episodes 128 \
      --eval-max-steps 500 \
      --controller-rollout-steps 0 \
      --wandb \
      --wandb-project "${WANDB_PROJECT_NAME}" \
      --wandb-name "${RUN_NAME}" \
      --wandb-group "${RUN_NAME}" \
      --wandb-mode "${WANDB_MODE}"

    echo "train_policy_pipeline completed at $(date -Is)"
  '
