#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-f002-tcvd
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --output=slurm_logs/tokamak-rl-v2-f002-tcv-delta-12gpu-20m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-f002-tcv-delta-12gpu-20m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs data/processed

export RUN_PRESET=${RUN_PRESET:-full}
export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml}
export NUM_ENVS=${NUM_ENVS:-3072}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}
export BATCH_SIZE=${BATCH_SIZE:-256}
export ROLLOUT_CHUNK_LENGTH=${ROLLOUT_CHUNK_LENGTH:-64}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-64}
export MAX_REQUEUES=${MAX_REQUEUES:-3}

case "${RUN_PRESET}" in
  full)
    export TRAIN_ENV_STEPS=20000000
    export EVAL_ENV_STEPS=1000000
    export CHECKPOINT_ENV_STEPS=5000000
    export MILESTONE_CHECKPOINT_ENV_STEPS=5000000
    export EVAL_CHECKPOINT_TOP_K=10
    export KEEP_LATEST_CHECKPOINT=1
    export SAVE_CHECKPOINTS=1
    ;;
  custom)
    export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:?RUN_PRESET=custom requires TRAIN_ENV_STEPS}
    export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:?RUN_PRESET=custom requires EVAL_ENV_STEPS}
    export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-5000000}
    export MILESTONE_CHECKPOINT_ENV_STEPS=${MILESTONE_CHECKPOINT_ENV_STEPS:-5000000}
    export EVAL_CHECKPOINT_TOP_K=${EVAL_CHECKPOINT_TOP_K:-10}
    export KEEP_LATEST_CHECKPOINT=${KEEP_LATEST_CHECKPOINT:-1}
    export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
    ;;
  *)
    echo "RUN_PRESET must be full or custom; got ${RUN_PRESET}" >&2
    exit 2
    ;;
esac

export GPUS_PER_NODE=${GPUS_PER_NODE:-6}
export NNODES=${SLURM_JOB_NUM_NODES:-2}
export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29584}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/t15_csv_segmented_profile_f002_tcv_delta_12gpu_20m_${SLURM_JOB_ID:-manual}}
export RUN_NAME=${RUN_NAME:-t15_csv_segmented_profile_f002_tcv_delta_12gpu_20m_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-f002-tcv-delta-12gpu-20m}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-60}

if [[ ! -f data/processed/t15_csv_initial_states.npz ]]; then
  echo "missing processed CSV reset artifact; build data/processed/t15_csv_initial_states.npz before launching" >&2
  exit 2
fi

export GENERATED_CONFIG="${TRAIN_OUTPUT}/generated_configs/t15_csv_segmented_profile_f002_tcv_delta_12gpu_20m.json"
mkdir -p "$(dirname "${GENERATED_CONFIG}")"

python3 - "${POLICY_CONFIG}" "${GENERATED_CONFIG}" "${TRAIN_OUTPUT}" <<'PYCFG'
import json
import os
import pathlib
import sys

base_path = pathlib.Path(sys.argv[1])
generated_path = pathlib.Path(sys.argv[2])
train_output = sys.argv[3]

cfg = json.loads(base_path.read_text(encoding="utf-8"))
cfg["name"] = "t15_csv_initial_segmented_profile_boundary_mpo_f002_tcv_delta_12gpu_20m"

sim = cfg.setdefault("sim", {})
sim["config_path"] = "/workspace/tokamak-sim/configs/T15MD_new_data.toml"
sim["max_episode_steps"] = 2000
sim["csv_initial_state_library"] = "/workspace/tokamak-rl-v2/data/processed/t15_csv_initial_states.npz"
sim["action_contract"] = "delta_jdot"
sim["delta_derivative_scale_aps"] = 500000.0
sim["terminate_on_boundary_loss"] = True
sim["terminate_on_current_limit"] = True
sim["current_termination_over_limit_a"] = 0.0
sim["current_termination_grace_steps"] = 1
sim["current_hard_termination_fraction"] = 1.20
sim["current_saturation_fraction"] = 1.0

reference = cfg.setdefault("reference", {})
reference["duration_s"] = 2.0
ip_reference = reference.setdefault("ip", {})
ip_reference["limits_path"] = "/workspace/tokamak-rl-v2/data/processed/t15_reference_limits.json"
ip_reference.update(
    {
        "kind": "segmented_profile",
        "start_mode": "reset_ip",
        "ramp_rate_reference": "robust_mean",
        "ramp_up_rate_min_fraction": 0.3,
        "ramp_up_rate_fraction": 0.55,
        "ramp_down_rate_min_fraction": 0.3,
        "ramp_down_rate_fraction": 0.55,
        "segment_min_steps": 300,
        "segment_max_steps": 800,
        "segment_count_min": 3,
        "segment_count_max": 5,
        "hold_min_steps": 300,
        "hold_max_steps": 800,
        "final_hold_min_steps": 0,
        "smooth_ramps": True,
        "max_delta_fraction": 0.60,
    }
)

observation = cfg.setdefault("observation", {})
observation["target_preview_steps"] = 10
observation["target_preview_stride"] = 50

cfg.setdefault("reward", {}).update(
    {
        "kind": "tcv_derivative",
        "shape_mean_scale_m": 0.03,
        "shape_max_scale_m": 0.08,
        "ip_scale_a": 15000.0,
        "boundary_missing_error_m": 1.0,
        "boundary_missing_weight": 20.0,
        "reward_scale": 0.01,
        "smoothmax_alpha": -5.0,
        "terminal_reward": -5.0,
        "terminal_remaining_cost": 0.0,
        "current_soft_fraction": 0.9,
        "current_bad_fraction": 1.0,
        "derivative_soft_fraction": 0.9,
        "derivative_bad_fraction": 1.1,
        "current_usage_weight": 0.0,
        "derivative_usage_weight": 0.0,
        "action_weight": 0.0,
        "delta_action_weight": 0.0,
        "shape_mean_weight": 3.2,
        "shape_max_weight": 0.8,
        "ip_weight": 1.8,
        "current_weight": 0.75,
        "derivative_weight": 0.1875,
        "actuator_saturation_weight": 0.1875,
    }
)

training = cfg.setdefault("training", {})
training["steps"] = int(os.environ["TRAIN_ENV_STEPS"])
training["num_envs"] = int(os.environ["NUM_ENVS"])
training["output_dir"] = f"/workspace/tokamak-rl-v2/{train_output}"
training["save_checkpoints"] = os.environ["SAVE_CHECKPOINTS"].lower() in {"1", "true", "yes", "on"}
training["checkpoint_interval_steps"] = int(os.environ["CHECKPOINT_ENV_STEPS"])
training["eval_interval_steps"] = int(os.environ["EVAL_ENV_STEPS"])
training["eval_max_steps"] = int(cfg["sim"]["max_episode_steps"])
training["eval_checkpoint_top_k"] = int(os.environ["EVAL_CHECKPOINT_TOP_K"])
training["milestone_checkpoint_interval_steps"] = int(os.environ["MILESTONE_CHECKPOINT_ENV_STEPS"])
training["keep_latest_checkpoint"] = os.environ["KEEP_LATEST_CHECKPOINT"].lower() in {"1", "true", "yes", "on"}
training["distributed_mode"] = "local_replay"
training["actor_workers"] = 1

learner = cfg.setdefault("learner", {})
learner["discount"] = 0.9995
learner["replay_capacity_episodes"] = int(os.environ["REPLAY_CAPACITY_EPISODES"])
learner["batch_size"] = int(os.environ["BATCH_SIZE"])
learner["rollout_chunk_length"] = int(os.environ["ROLLOUT_CHUNK_LENGTH"])
learner["updates_per_rollout_chunk"] = int(os.environ["UPDATES_PER_ROLLOUT_CHUNK"])

checks = {
    "reward.kind": cfg["reward"]["kind"] == "tcv_derivative",
    "reward.shape_mean_weight": cfg["reward"]["shape_mean_weight"] == 3.2,
    "reward.shape_max_weight": cfg["reward"]["shape_max_weight"] == 0.8,
    "reward.ip_weight": cfg["reward"]["ip_weight"] == 1.8,
    "reward.current_weight": cfg["reward"]["current_weight"] == 0.75,
    "reward.derivative_weight": cfg["reward"]["derivative_weight"] == 0.1875,
    "reward.actuator_saturation_weight": cfg["reward"]["actuator_saturation_weight"] == 0.1875,
    "sim.action_contract": cfg["sim"]["action_contract"] == "delta_jdot",
    "sim.terminate_on_boundary_loss": cfg["sim"]["terminate_on_boundary_loss"] is True,
    "sim.terminate_on_current_limit": cfg["sim"]["terminate_on_current_limit"] is True,
    "sim.current_termination_over_limit_a": float(cfg["sim"]["current_termination_over_limit_a"]) == 0.0,
    "sim.current_termination_grace_steps": int(cfg["sim"]["current_termination_grace_steps"]) == 1,
    "sim.current_hard_termination_fraction": float(cfg["sim"]["current_hard_termination_fraction"]) == 1.20,
    "sim.max_episode_steps": int(cfg["sim"]["max_episode_steps"]) == 2000,
    "reference.duration_s": float(cfg["reference"]["duration_s"]) == 2.0,
    "observation.target_preview_steps": int(cfg["observation"]["target_preview_steps"]) == 10,
    "observation.target_preview_stride": int(cfg["observation"]["target_preview_stride"]) == 50,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"generated f002 config failed sanity checks: {failed}")

generated_path.parent.mkdir(parents=True, exist_ok=True)
generated_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(generated_path)
PYCFG

cat > "${TRAIN_OUTPUT}/f002_reward.json" <<'JSON'
{
  "source": "tcv_delta_jdot two-phase reward search",
  "variant": "pass2_focused/f002_sf0_if1_af0",
  "reward": {
    "kind": "tcv_derivative",
    "shape_mean_weight": 3.2,
    "shape_max_weight": 0.8,
    "ip_weight": 1.8,
    "current_weight": 0.75,
    "derivative_weight": 0.1875,
    "actuator_saturation_weight": 0.1875,
    "reward_scale": 0.01,
    "terminal_reward": -5.0
  },
  "sim": {
    "action_contract": "delta_jdot",
    "delta_derivative_scale_aps": 500000.0,
    "terminate_on_boundary_loss": true,
    "terminate_on_current_limit": true,
    "current_termination_over_limit_a": 0.0,
    "current_termination_grace_steps": 1,
    "current_hard_termination_fraction": 1.20
  }
}
JSON

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_preset=${RUN_PRESET}"
echo "node_list=${SLURM_JOB_NODELIST:-manual}"
echo "restart_count=${SLURM_RESTART_COUNT:-0}"
echo "max_requeues=${MAX_REQUEUES}"
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

if (( NUM_ENVS % WORLD_SIZE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi

EPISODE_STEPS=$(python3 - "${GENERATED_CONFIG}" <<'PYSTEPS'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
print(int(cfg["sim"]["max_episode_steps"]))
PYSTEPS
)
echo "episode_steps=${EPISODE_STEPS}"
export EPISODE_STEPS

MIN_LEARNING_ENV_STEPS=$(( NUM_ENVS * EPISODE_STEPS + NUM_ENVS ))
if (( TRAIN_ENV_STEPS < MIN_LEARNING_ENV_STEPS )); then
  echo "TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS} is too short for 12-GPU learning; need at least ${MIN_LEARNING_ENV_STEPS} global env steps to complete one ${EPISODE_STEPS}-step episode and create replay updates" >&2
  exit 2
fi

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
    RESUME_ARGS=()
    if [[ -f "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/checkpoints/latest.pt" ]]; then
      RESUME_ARGS=(--resume-checkpoint "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/checkpoints/latest.pt")
      echo "node_rank=${NODE_RANK} warm_resume_checkpoint=/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/checkpoints/latest.pt"
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
        "${RESUME_ARGS[@]}" \
        --eval-interval-steps "${EVAL_ENV_STEPS}" \
        --eval-episodes 128 \
        --eval-max-steps "${EPISODE_STEPS}" \
        --controller-rollout-steps 0 \
        --wandb \
        --wandb-project "${WANDB_PROJECT_NAME}" \
        --wandb-name "${RUN_NAME}" \
        --wandb-group "${RUN_NAME}" \
        --wandb-mode "${WANDB_MODE}" \
        --wandb-optional \
        --wandb-metric-preset focused

    echo "node_rank=${NODE_RANK} completed at $(date -Is)"
  '
run_rc=$?
set -e

if (( run_rc != 0 )); then
  validation_status=$(python3 - "${TRAIN_OUTPUT}/policy_validation.json" <<'PYSTATUS'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit(0)
try:
    print(str(json.loads(path.read_text(encoding="utf-8")).get("status", "")))
except Exception:
    print("")
PYSTATUS
)
  if [[ "${validation_status}" == "failed_gates" ]]; then
    echo "training completed and failed policy gates; treating as completed Slurm job, not infrastructure failure"
    exit 0
  fi
  if [[ -f "${TRAIN_OUTPUT}/checkpoints/latest.pt" && "${SLURM_RESTART_COUNT:-0}" -lt "${MAX_REQUEUES}" ]]; then
    echo "srun failed with rc=${run_rc}; requeueing ${SLURM_JOB_ID} from latest checkpoint"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  fi
  echo "srun failed with rc=${run_rc}; no eligible warm-resume requeue remains" >&2
  exit "${run_rc}"
fi

echo "12-GPU f002 TCV delta-Jdot 20M training job completed at $(date -Is)"
