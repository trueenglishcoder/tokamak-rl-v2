#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-fixed-hold-12gpu
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --output=slurm_logs/tokamak-rl-v2-fixed-hold-12gpu-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-fixed-hold-12gpu-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs data/processed

export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_csv_hold_ip_fixed_horizon.yaml}
export RUN_PRESET=${RUN_PRESET:-diagnostic}
export NUM_ENVS=${NUM_ENVS:-3072}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}
export BATCH_SIZE=${BATCH_SIZE:-256}
export ROLLOUT_CHUNK_LENGTH=${ROLLOUT_CHUNK_LENGTH:-64}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-64}
export GPUS_PER_NODE=${GPUS_PER_NODE:-6}
export MAX_REQUEUES=${MAX_REQUEUES:-2}
case "${RUN_PRESET}" in
  diagnostic)
    export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-20000000}
    export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-1000000}
    export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}
    export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
    ;;
  custom)
    export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:?RUN_PRESET=custom requires TRAIN_ENV_STEPS}
    export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-1000000}
    export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-${EVAL_ENV_STEPS}}
    export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
    ;;
  *)
    echo "RUN_PRESET must be diagnostic or custom; got ${RUN_PRESET}" >&2
    exit 2
    ;;
esac

export NNODES=${SLURM_JOB_NUM_NODES:-2}
export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29611}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/t15_csv_hold_ip_fixed_horizon_12gpu_${SLURM_JOB_ID:-manual}}
export RUN_NAME=${RUN_NAME:-t15_csv_fixed_horizon_hold_ip_12gpu_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-fixed-horizon-diagnostics-${SLURM_JOB_ID:-manual}}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-60}

if (( NUM_ENVS % WORLD_SIZE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi

EPISODE_STEPS=$(python3 - "${POLICY_CONFIG}" <<'PYSTEPS'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
print(int(cfg["sim"]["max_episode_steps"]))
PYSTEPS
)
MIN_LEARNING_ENV_STEPS=$(( NUM_ENVS * EPISODE_STEPS + NUM_ENVS ))
if (( TRAIN_ENV_STEPS < MIN_LEARNING_ENV_STEPS )); then
  echo "TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS} is too short; need at least ${MIN_LEARNING_ENV_STEPS} global env steps for one full ${EPISODE_STEPS}-step episode plus replay updates" >&2
  exit 2
fi

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "node_list=${SLURM_JOB_NODELIST:-manual}"
echo "restart_count=${SLURM_RESTART_COUNT:-0}"
echo "policy_config=${POLICY_CONFIG}"
echo "train_output=${TRAIN_OUTPUT}"
echo "run_name=${RUN_NAME}"
echo "wandb_project=${WANDB_PROJECT_NAME}"
echo "world_size=${WORLD_SIZE}"
echo "num_envs=${NUM_ENVS}"
echo "local_envs_per_rank=$(( NUM_ENVS / WORLD_SIZE ))"
echo "episode_steps=${EPISODE_STEPS}"
echo "train_env_steps=${TRAIN_ENV_STEPS}"
echo "eval_env_steps=${EVAL_ENV_STEPS}"
echo "checkpoint_env_steps=${CHECKPOINT_ENV_STEPS}"

set +e
srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --gres="gpu:${GPUS_PER_NODE}" \
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
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

    cd /workspace/tokamak-rl-v2
    mkdir -p slurm_logs outputs data/processed
    python3 scripts/apply_t15_actuator_limits.py /workspace/tokamak-sim/configs/T15MD_new_data.toml
    python3 scripts/ensure_t15_reference_limits.py
    python3 scripts/ensure_t15_csv_initial_states.py --experiment-config "${POLICY_CONFIG}" --workers 8 --max-rows-per-shot 600

    python3 - <<PYCUDA
import os
import torch
count = torch.cuda.device_count()
expected = int(os.environ["GPUS_PER_NODE"])
print("node_rank", os.environ.get("SLURM_NODEID", "0"), "cuda_available", torch.cuda.is_available(), "device_count", count)
print("devices", [torch.cuda.get_device_name(i) for i in range(count)])
if count < expected:
    raise SystemExit(f"expected at least {expected} visible CUDA devices, got {count}")
PYCUDA

    CHECKPOINT_ARGS=()
    if [[ "${SAVE_CHECKPOINTS}" == "1" || "${SAVE_CHECKPOINTS}" == "true" ]]; then
      CHECKPOINT_ARGS=(--save-checkpoints --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}")
    else
      CHECKPOINT_ARGS=(--no-save-checkpoints)
    fi
    RESUME_ARGS=()
    if [[ -f "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/checkpoints/latest.pt" ]]; then
      RESUME_ARGS=(--resume-checkpoint "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/checkpoints/latest.pt")
      echo "node_rank=${SLURM_NODEID:-0} warm_resume_checkpoint=/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/checkpoints/latest.pt"
    fi

    python3 -m torch.distributed.run \
      --nnodes "${NNODES}" \
      --nproc_per_node "${GPUS_PER_NODE}" \
      --node_rank "${SLURM_NODEID:-0}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      tokamak_rl_v2/training/cli.py \
        --config "/workspace/tokamak-rl-v2/${POLICY_CONFIG}" \
        --output-dir "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}" \
        --steps "${TRAIN_ENV_STEPS}" \
        --device cuda \
        --sim-compute-backend gpu \
        --sim-gpu-device cuda \
        --num-envs "${NUM_ENVS}" \
        --distributed-mode local_replay \
        --actor-workers 1 \
        --batch-size "${BATCH_SIZE}" \
        --replay-capacity-episodes "${REPLAY_CAPACITY_EPISODES}" \
        --rollout-chunk-length "${ROLLOUT_CHUNK_LENGTH}" \
        --updates-per-rollout-chunk "${UPDATES_PER_ROLLOUT_CHUNK}" \
        "${CHECKPOINT_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        --eval-interval-steps "${EVAL_ENV_STEPS}" \
        --eval-episodes 128 \
        --eval-max-steps "${EPISODE_STEPS}" \
        --wandb \
        --wandb-project "${WANDB_PROJECT_NAME}" \
        --wandb-name "${RUN_NAME}" \
        --wandb-mode "${WANDB_MODE}"

    echo "node_rank=${SLURM_NODEID:-0} completed at $(date -Is)"
  '
run_rc=$?
set -e

if (( run_rc != 0 )); then
  if [[ -f "${TRAIN_OUTPUT}/checkpoints/latest.pt" && "${SLURM_RESTART_COUNT:-0}" -lt "${MAX_REQUEUES}" ]]; then
    echo "srun failed with rc=${run_rc}; requeueing ${SLURM_JOB_ID} from latest checkpoint"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  fi
  echo "srun failed with rc=${run_rc}; no eligible warm-resume requeue remains" >&2
  exit "${run_rc}"
fi

echo "12-GPU fixed-horizon hold-Ip diagnostic completed at $(date -Is)"
