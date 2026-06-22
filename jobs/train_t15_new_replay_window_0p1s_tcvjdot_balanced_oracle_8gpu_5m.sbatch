#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-rw-balanced-5m
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-rw-oracle-balanced-8gpu-5m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-rw-oracle-balanced-8gpu-5m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export REWARD_PRESET=balanced
export RUN_NAME="t15_new_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_5m_${SLURM_JOB_ID:-manual}"
export TRAIN_OUTPUT="outputs/${RUN_NAME}"
export GENERATED_CONFIG="${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json"
export WANDB_PROJECT_NAME="tokamak-rl-v2-tcvjdot-rw-oracle-balanced-8gpu-5m-${SLURM_JOB_ID:-manual}"
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-5000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}
export NUM_ENVS=${NUM_ENVS:-2048}
export GPUS_PER_NODE=8
export WORLD_SIZE=8
export WANDB_MODE=${WANDB_MODE:-online}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29593}

if [[ ! -f data/processed/t15_new_replay_window_0p1s_oracle_initial_states.npz ]]; then
  echo "missing oracle reset library; run jobs/build_t15_new_replay_window_oracle_targets_1gpu.sbatch first" >&2
  exit 2
fi
if [[ ! -f data/processed/t15_new_replay_window_0p1s_oracle_targets/t15_replay_window_oracle_targets.npz ]]; then
  echo "missing oracle targets; run jobs/build_t15_new_replay_window_oracle_targets_1gpu.sbatch first" >&2
  exit 2
fi

mkdir -p "$(dirname "${GENERATED_CONFIG}")"
python3 scripts/make_t15_new_replay_window_oracle_training_config.py \
  --output-config "${GENERATED_CONFIG}" \
  --output-dir "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}" \
  --reward-preset "${REWARD_PRESET}" \
  --steps "${TRAIN_ENV_STEPS}" \
  --num-envs "${NUM_ENVS}" \
  --eval-interval-steps "${EVAL_ENV_STEPS}" \
  --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_name=${RUN_NAME}"
echo "train_output=${TRAIN_OUTPUT}"
echo "generated_config=${GENERATED_CONFIG}"
echo "wandb_project=${WANDB_PROJECT_NAME}"
echo "train_env_steps=${TRAIN_ENV_STEPS}"
echo "num_envs=${NUM_ENVS}"
echo "world_size=${WORLD_SIZE}"
echo "local_envs_per_rank=$(( NUM_ENVS / WORLD_SIZE ))"

if (( NUM_ENVS % WORLD_SIZE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export PYTHONFAULTHANDLER=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    cd /workspace/tokamak-rl-v2

    python3 - <<PYCUDA
import torch
count = torch.cuda.device_count()
print("cuda_available", torch.cuda.is_available(), "device_count", count)
print("devices", [torch.cuda.get_device_name(i) for i in range(count)])
if count < 8:
    raise SystemExit(f"expected 8 visible CUDA devices, got {count}")
PYCUDA

    python3 -m torch.distributed.run \
      --nnodes 1 \
      --nproc_per_node 8 \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      scripts/train_policy_pipeline.py \
        --config "/workspace/tokamak-rl-v2/${GENERATED_CONFIG}" \
        --output-dir "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}" \
        --steps "${TRAIN_ENV_STEPS}" \
        --device cuda:0 \
        --sim-compute-backend gpu \
        --sim-gpu-device cuda \
        --num-envs "${NUM_ENVS}" \
        --distributed-mode local_replay \
        --actor-workers 1 \
        --batch-size 32 \
        --unroll-length 100 \
        --min-replay-sequence-length 100 \
        --replay-capacity-episodes 1024 \
        --rollout-chunk-length 100 \
        --updates-per-rollout-chunk 64 \
        --hidden-dim 256 \
        --critic-hidden-dim 256 \
        --critic-mlp-hidden-dim 256 \
        --action-samples 20 \
        --actor-update-chunk-size 2048 \
        --save-checkpoints \
        --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}" \
        --eval-interval-steps "${EVAL_ENV_STEPS}" \
        --eval-episodes 128 \
        --eval-max-steps 100 \
        --controller-rollout-steps 0 \
        --wandb \
        --wandb-project "${WANDB_PROJECT_NAME}" \
        --wandb-name "${RUN_NAME}" \
        --wandb-group "${RUN_NAME}" \
        --wandb-mode "${WANDB_MODE}" \
        --wandb-optional \
        --wandb-metric-preset focused
  '

if [[ -f "${TRAIN_OUTPUT}/checkpoints/latest.pt" ]]; then
  srun \
    --ntasks=1 \
    --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
    --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
    bash -lc '
      set -euo pipefail
      mkdir -p /tmp/nvidia-lib
      ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
      ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
      export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
      export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
      cd /workspace/tokamak-rl-v2
      python3 scripts/evaluate_replay_window_oracle_baselines.py \
        --config "/workspace/tokamak-rl-v2/${GENERATED_CONFIG}" \
        --checkpoint "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/checkpoints/latest.pt" \
        --output-dir "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/oracle_baseline_eval" \
        --episodes 256 \
        --steps 100 \
        --split holdout \
        --device cuda:0
    '
fi
