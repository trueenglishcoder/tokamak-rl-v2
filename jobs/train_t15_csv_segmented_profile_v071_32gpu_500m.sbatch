#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-v071-32gpu
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-v071-32gpu-500m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-v071-32gpu-500m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs data/processed

export RUN_PRESET=${RUN_PRESET:-full}
export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml}
export NUM_ENVS=${NUM_ENVS:-8192}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-512}
export BATCH_SIZE=${BATCH_SIZE:-256}
export ROLLOUT_CHUNK_LENGTH=${ROLLOUT_CHUNK_LENGTH:-64}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-64}
case "${RUN_PRESET}" in
  full)
    export TRAIN_ENV_STEPS=500000000
    export EVAL_ENV_STEPS=5000000
    export CHECKPOINT_ENV_STEPS=5000000
    export MILESTONE_CHECKPOINT_ENV_STEPS=10000000
    export EVAL_CHECKPOINT_TOP_K=10
    export KEEP_LATEST_CHECKPOINT=1
    export SAVE_CHECKPOINTS=1
    export REWARD_SWEEP_MODE=0
    ;;
  smoke)
    export TRAIN_ENV_STEPS=10000000
    export EVAL_ENV_STEPS=5000000
    export CHECKPOINT_ENV_STEPS=5000000
    export MILESTONE_CHECKPOINT_ENV_STEPS=5000000
    export EVAL_CHECKPOINT_TOP_K=2
    export KEEP_LATEST_CHECKPOINT=1
    export SAVE_CHECKPOINTS=1
    export REWARD_SWEEP_MODE=1
    ;;
  custom)
    export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:?RUN_PRESET=custom requires TRAIN_ENV_STEPS}
    export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:?RUN_PRESET=custom requires EVAL_ENV_STEPS}
    export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-${EVAL_ENV_STEPS}}
    export MILESTONE_CHECKPOINT_ENV_STEPS=${MILESTONE_CHECKPOINT_ENV_STEPS:-0}
    export EVAL_CHECKPOINT_TOP_K=${EVAL_CHECKPOINT_TOP_K:-0}
    export KEEP_LATEST_CHECKPOINT=${KEEP_LATEST_CHECKPOINT:-1}
    export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
    export REWARD_SWEEP_MODE=${REWARD_SWEEP_MODE:-0}
    ;;
  *)
    echo "RUN_PRESET must be full, smoke, or custom; got ${RUN_PRESET}" >&2
    exit 2
    ;;
esac
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${SLURM_JOB_NUM_NODES:-4}
export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29571}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/t15_csv_segmented_profile_v071_32gpu_500m_${SLURM_JOB_ID:-manual}}
export RUN_NAME=${RUN_NAME:-t15_csv_segmented_profile_v071_32gpu_500m_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-v071-32gpu-${RUN_PRESET}}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-60}

if [[ ! -f data/processed/t15_reference_limits.json || ! -f data/processed/t15_csv_initial_states.npz ]]; then
  echo "missing processed CSV production artifacts; build data/processed/t15_reference_limits.json and data/processed/t15_csv_initial_states.npz before launching" >&2
  exit 2
fi

export GENERATED_CONFIG="${TRAIN_OUTPUT}/generated_configs/t15_csv_segmented_profile_v071_32gpu_500m.json"
mkdir -p "$(dirname "${GENERATED_CONFIG}")"

python3 - "${POLICY_CONFIG}" "${GENERATED_CONFIG}" "${TRAIN_OUTPUT}" <<'PYCFG'
import json
import pathlib
import sys

base_path = pathlib.Path(sys.argv[1])
generated_path = pathlib.Path(sys.argv[2])
train_output = sys.argv[3]

cfg = json.loads(base_path.read_text(encoding="utf-8"))
cfg["name"] = "t15_csv_initial_segmented_profile_boundary_mpo_v071_32gpu_500m"
cfg.setdefault("sim", {})["config_path"] = "/workspace/tokamak-sim/configs/T15MD_new_data.toml"
cfg["sim"]["csv_initial_state_library"] = "/workspace/tokamak-rl-v2/data/processed/t15_csv_initial_states.npz"
cfg.setdefault("reference", {}).setdefault("ip", {})["limits_path"] = "/workspace/tokamak-rl-v2/data/processed/t15_reference_limits.json"
cfg.setdefault("reward", {}).update(
    {
        "shape_mean_scale_m": 0.03,
        "shape_max_scale_m": 0.08,
        "ip_scale_a": 25000.0,
        "reward_scale": 1.0,
        "terminal_reward": -20.0,
        "shape_mean_weight": 6.0,
        "shape_max_weight": 2.0,
        "ip_weight": 3.5,
        "current_weight": 8.0,
        "current_soft_fraction": 0.85,
        "derivative_weight": 0.75,
        "action_weight": 0.04,
        "delta_action_weight": 0.075,
    }
)
training = cfg.setdefault("training", {})
training["steps"] = int(__import__("os").environ["TRAIN_ENV_STEPS"])
training["num_envs"] = int(__import__("os").environ["NUM_ENVS"])
training["output_dir"] = f"/workspace/tokamak-rl-v2/{train_output}"
training["save_checkpoints"] = __import__("os").environ["SAVE_CHECKPOINTS"].lower() in {"1", "true", "yes", "on"}
training["checkpoint_interval_steps"] = int(__import__("os").environ["CHECKPOINT_ENV_STEPS"])
training["eval_interval_steps"] = int(__import__("os").environ["EVAL_ENV_STEPS"])
training["eval_checkpoint_top_k"] = int(__import__("os").environ["EVAL_CHECKPOINT_TOP_K"])
training["milestone_checkpoint_interval_steps"] = int(__import__("os").environ["MILESTONE_CHECKPOINT_ENV_STEPS"])
training["keep_latest_checkpoint"] = __import__("os").environ["KEEP_LATEST_CHECKPOINT"].lower() in {"1", "true", "yes", "on"}
training["distributed_mode"] = "local_replay"
training["actor_workers"] = 1
learner = cfg.setdefault("learner", {})
learner["replay_capacity_episodes"] = int(__import__("os").environ["REPLAY_CAPACITY_EPISODES"])
learner["batch_size"] = int(__import__("os").environ["BATCH_SIZE"])
learner["rollout_chunk_length"] = int(__import__("os").environ["ROLLOUT_CHUNK_LENGTH"])
learner["updates_per_rollout_chunk"] = int(__import__("os").environ["UPDATES_PER_ROLLOUT_CHUNK"])

generated_path.parent.mkdir(parents=True, exist_ok=True)
generated_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(generated_path)
PYCFG

cat > "${TRAIN_OUTPUT}/v071_reward.json" <<'JSON'
{
  "source": "physical reward sweep recommendation",
  "variant": "v071_s1_i2_c3_a2",
  "reward": {
    "shape_mean_weight": 6.0,
    "shape_max_weight": 2.0,
    "ip_weight": 3.5,
    "current_weight": 8.0,
    "current_soft_fraction": 0.85,
    "derivative_weight": 0.75,
    "action_weight": 0.04,
    "delta_action_weight": 0.075
  }
}
JSON

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_preset=${RUN_PRESET}"
echo "node_list=${SLURM_JOB_NODELIST:-manual}"
echo "master_addr=${MASTER_ADDR}"
echo "master_port=${MASTER_PORT}"
echo "nnodes=${NNODES}"
echo "gpus_per_node=${GPUS_PER_NODE}"
echo "world_size=${WORLD_SIZE}"
echo "policy_config=${GENERATED_CONFIG}"
echo "train_output=${TRAIN_OUTPUT}"
echo "run_name=${RUN_NAME}"
echo "wandb_project=${WANDB_PROJECT_NAME}"
echo "train_env_steps=${TRAIN_ENV_STEPS}"
echo "num_envs=${NUM_ENVS}"
echo "local_envs_per_rank=$(( NUM_ENVS / WORLD_SIZE ))"
echo "eval_env_steps=${EVAL_ENV_STEPS}"
echo "checkpoint_env_steps=${CHECKPOINT_ENV_STEPS}"
echo "milestone_checkpoint_env_steps=${MILESTONE_CHECKPOINT_ENV_STEPS}"
echo "eval_checkpoint_top_k=${EVAL_CHECKPOINT_TOP_K}"
echo "keep_latest_checkpoint=${KEEP_LATEST_CHECKPOINT}"
echo "save_checkpoints=${SAVE_CHECKPOINTS}"
echo "batch_size=${BATCH_SIZE}"
echo "replay_capacity_episodes=${REPLAY_CAPACITY_EPISODES}"
echo "rollout_chunk_length=${ROLLOUT_CHUNK_LENGTH}"
echo "updates_per_rollout_chunk=${UPDATES_PER_ROLLOUT_CHUNK}"
echo "reward_sweep_mode=${REWARD_SWEEP_MODE}"

if (( NUM_ENVS % WORLD_SIZE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi

MIN_LEARNING_ENV_STEPS=$(( NUM_ENVS * 500 + NUM_ENVS ))
if (( TRAIN_ENV_STEPS < MIN_LEARNING_ENV_STEPS )); then
  echo "TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS} is too short for 32-GPU learning; need at least ${MIN_LEARNING_ENV_STEPS} global env steps to complete one 500-step episode and create replay updates" >&2
  exit 2
fi

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

    NODE_RANK=${SLURM_NODEID:-0}
    CHECKPOINT_ARGS=()
    if [[ "${SAVE_CHECKPOINTS}" == "1" || "${SAVE_CHECKPOINTS}" == "true" ]]; then
      CHECKPOINT_ARGS=(--save-checkpoints --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}")
    else
      CHECKPOINT_ARGS=(--no-save-checkpoints)
    fi
    MODE_ARGS=()
    if [[ "${REWARD_SWEEP_MODE}" == "1" || "${REWARD_SWEEP_MODE}" == "true" ]]; then
      MODE_ARGS=(--reward-sweep-mode)
    fi

    python3 -m torch.distributed.run \
      --nnodes "${NNODES}" \
      --nproc_per_node "${GPUS_PER_NODE}" \
      --node_rank "${NODE_RANK}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      scripts/train_policy_pipeline.py \
        --config "/workspace/tokamak-rl-v2/${GENERATED_CONFIG}" \
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
        "${MODE_ARGS[@]}" \
        --wandb \
        --wandb-project "${WANDB_PROJECT_NAME}" \
        --wandb-name "${RUN_NAME}" \
        --wandb-group "${RUN_NAME}" \
        --wandb-mode "${WANDB_MODE}"

    echo "node_rank=${NODE_RANK} completed at $(date -Is)"
  '

echo "32-GPU v071 training job completed at $(date -Is)"
