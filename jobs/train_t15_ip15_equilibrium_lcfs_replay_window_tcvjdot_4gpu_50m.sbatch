#!/bin/bash
#SBATCH --job-name=tokamak-ip15-equilibrium-lcfs-50m
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-ip15-equilibrium-lcfs-4gpu-50m-%j.out
#SBATCH --error=slurm_logs/tokamak-ip15-equilibrium-lcfs-4gpu-50m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_ip15_equilibrium_lcfs_replay_window_0p1s_tcvjdot_mpo_balanced.yaml}
export MACHINE_CONFIG_SOURCE=${MACHINE_CONFIG_SOURCE:-../tokamak-sim/configs/T15MD.toml}
export DATA_ROOT=${DATA_ROOT:-../tokamak-sim/data/t15_data_new_trim50_ip_calibrated}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_initial_states.npz}
export ORACLE_TARGET_DIR=${ORACLE_TARGET_DIR:-data/processed/t15_ip15_equilibrium_lcfs_replay_window_0p1s_oracle_targets}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export ORACLE_SUMMARY=${ORACLE_SUMMARY:-${ORACLE_TARGET_DIR}/oracle_summary.json}
export CONFIG_NAME=${CONFIG_NAME:-t15_ip15_equilibrium_lcfs_replay_window_tcvjdot_4gpu_50m}
export RUN_NAME=${RUN_NAME:-t15_ip15_equilibrium_lcfs_replay_window_tcvjdot_4gpu_50m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export GENERATED_MACHINE_CONFIG=${GENERATED_MACHINE_CONFIG:-${TRAIN_OUTPUT}/generated_configs/T15MD.toml}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-ip15-equilibrium-lcfs-4gpu-50m-${SLURM_JOB_ID:-manual}}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-50000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}
export NUM_ENVS=${NUM_ENVS:-1024}
export GPUS_PER_NODE=4
export WORLD_SIZE=4
export UPDATES_PER_ROLLOUT_CHUNK=64
export BATCH_SIZE=64
export REPLAY_CAPACITY_EPISODES=2048
export ROLLOUT_CHUNK_LENGTH=200
export HOLDOUT_EVAL_EPISODES=${HOLDOUT_EVAL_EPISODES:-}
export WANDB_MODE=${WANDB_MODE:-online}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-1593} % 40000)))}

for required in "${SOURCE_CONFIG}" "${MACHINE_CONFIG_SOURCE}" "${RESET_LIBRARY}" "${ORACLE_TARGETS}" "${ORACLE_SUMMARY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required experiment artifact: ${required}" >&2
    echo "run jobs/build_t15_ip15_equilibrium_lcfs_replay_window_oracle_targets_1gpu.sbatch first" >&2
    exit 2
  fi
done
if [[ ! -d "${DATA_ROOT}" || ! -d "${ORACLE_TARGET_DIR}" ]]; then
  echo "missing calibrated data or oracle target directory" >&2
  exit 2
fi
if (( NUM_ENVS % WORLD_SIZE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi

if [[ -z "${HOLDOUT_EVAL_EPISODES:-}" ]]; then
  HOLDOUT_EVAL_EPISODES="$(python3 - "${ORACLE_SUMMARY}" <<'PYHOLDOUT'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
count = int(summary.get("split_counts", {}).get("holdout", 0))
if count <= 0:
    raise RuntimeError("oracle summary has no holdout windows")
print(count)
PYHOLDOUT
)"
fi
export HOLDOUT_EVAL_EPISODES

mkdir -p "$(dirname "${GENERATED_CONFIG}")"
cp "${MACHINE_CONFIG_SOURCE}" "${GENERATED_MACHINE_CONFIG}"

python3 - <<PYCFG
import json
from pathlib import Path

source = Path("${SOURCE_CONFIG}")
out = Path("${GENERATED_CONFIG}")
cfg = json.loads(source.read_text(encoding="utf-8"))

cfg["name"] = "${CONFIG_NAME}"
cfg["sim"]["config_path"] = "/workspace/tokamak-rl-v2/${GENERATED_MACHINE_CONFIG}"
cfg["sim"]["csv_initial_state_library"] = "/workspace/tokamak-rl-v2/${RESET_LIBRARY}"
cfg["reference"]["boundary"]["replay_reference_dir"] = "/workspace/tokamak-rl-v2/${ORACLE_TARGET_DIR}"
cfg["training"]["steps"] = int("${TRAIN_ENV_STEPS}")
cfg["training"]["num_envs"] = int("${NUM_ENVS}")
cfg["training"]["output_dir"] = "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}"
cfg["training"]["eval_interval_steps"] = int("${EVAL_ENV_STEPS}")
cfg["training"]["checkpoint_interval_steps"] = int("${CHECKPOINT_ENV_STEPS}")
cfg["training"]["milestone_checkpoint_interval_steps"] = int("${CHECKPOINT_ENV_STEPS}")
cfg["training"]["eval_checkpoint_top_k"] = 10
cfg["training"]["keep_latest_checkpoint"] = True
cfg["training"]["distributed_mode"] = "local_replay"
cfg["training"]["production_mode"] = True
cfg["training"]["eval_episodes"] = int("${HOLDOUT_EVAL_EPISODES}")
cfg["training"]["eval_max_steps"] = 100
cfg["training"]["save_checkpoints"] = True
cfg["training"]["early_stop_patience_evals"] = 0
cfg["training"]["early_stop_min_delta"] = 0.0
cfg["learner"]["batch_size"] = int("${BATCH_SIZE}")
cfg["learner"]["replay_capacity_episodes"] = int("${REPLAY_CAPACITY_EPISODES}")
cfg["learner"]["rollout_chunk_length"] = int("${ROLLOUT_CHUNK_LENGTH}")
cfg["learner"]["updates_per_rollout_chunk"] = int("${UPDATES_PER_ROLLOUT_CHUNK}")

expected_reward = {
    "shape_mean_weight": 3.2,
    "shape_max_weight": 0.8,
    "ip_weight": 1.8,
    "current_weight": 0.75,
    "derivative_weight": 0.1875,
    "actuator_saturation_weight": 0.1875,
    "boundary_missing_weight": 20.0,
    "reward_scale": 0.01,
}
for key, expected in expected_reward.items():
    actual = float(cfg["reward"][key])
    if actual != expected:
        raise SystemExit(f"successful-run reward weight changed: {key} expected={expected} actual={actual}")
if int(cfg["training"]["steps"]) != 50_000_000:
    raise SystemExit("this experiment must run for exactly 50M environment steps")
if int(cfg["training"]["num_envs"]) != 1024:
    raise SystemExit("the 4-GPU scaled contract requires 1024 environments")
if int(cfg["learner"]["batch_size"]) != 64:
    raise SystemExit("the 4-GPU scaled contract requires batch_size=64 per rank")
if int(cfg["learner"]["replay_capacity_episodes"]) != 2048:
    raise SystemExit("the 4-GPU scaled contract requires replay_capacity_episodes=2048 per rank")
if int(cfg["learner"]["rollout_chunk_length"]) != 200:
    raise SystemExit("the 4-GPU scaled contract requires rollout_chunk_length=200")
if int(cfg["learner"]["updates_per_rollout_chunk"]) != 64:
    raise SystemExit("the 4-GPU scaled contract requires 64 updates per rollout chunk")

world_size = 4
local_envs = int(cfg["training"]["num_envs"]) // world_size
if local_envs != 256:
    raise SystemExit(f"expected 256 environments per GPU, got {local_envs}")
if int(cfg["training"]["num_envs"]) * int(cfg["learner"]["rollout_chunk_length"]) != 204_800:
    raise SystemExit("global transitions per update block must remain 204800")
if world_size * int(cfg["learner"]["batch_size"]) != 256:
    raise SystemExit("effective distributed batch must remain 256 sequences")
if world_size * int(cfg["learner"]["replay_capacity_episodes"]) != 8192:
    raise SystemExit("aggregate replay capacity must remain 8192 episodes")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(out)
PYCFG

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_name=${RUN_NAME}"
echo "generated_config=${GENERATED_CONFIG}"
echo "train_env_steps=${TRAIN_ENV_STEPS}"
echo "num_envs=${NUM_ENVS}"
echo "world_size=${WORLD_SIZE}"
echo "local_envs_per_rank=$((NUM_ENVS / WORLD_SIZE))"
echo "batch_size_per_rank=${BATCH_SIZE}"
echo "effective_distributed_batch=$((BATCH_SIZE * WORLD_SIZE))"
echo "rollout_chunk_length=${ROLLOUT_CHUNK_LENGTH}"
echo "transitions_per_update_block=$((NUM_ENVS * ROLLOUT_CHUNK_LENGTH))"
echo "replay_capacity_per_rank=${REPLAY_CAPACITY_EPISODES}"
echo "aggregate_replay_capacity=$((REPLAY_CAPACITY_EPISODES * WORLD_SIZE))"
echo "holdout_eval_episodes=${HOLDOUT_EVAL_EPISODES}"

srun \
  --ntasks=1 \
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

    python3 - <<PYCHECK
from dataclasses import replace
from pathlib import Path
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv

cfg = load_experiment_config("/workspace/tokamak-rl-v2/${GENERATED_CONFIG}")
required = [
    cfg.sim.config_path,
    cfg.sim.csv_initial_state_library,
    cfg.reference.boundary.replay_reference_dir,
]
missing = [str(path) for path in required if path is None or not Path(path).exists()]
if missing:
    raise SystemExit("generated training config has missing paths: " + ", ".join(missing))

env_cfg = replace(cfg, sim=replace(cfg.sim, gpu_device="cuda:0"))
env = TokamakMagneticControlEnv(env_cfg, batch_size=8, device="cuda:0", seed=int(cfg.training.seed))
observation = env.reset()
if str(env.cfg.boundary_mode) != "equilibrium_lcfs":
    raise SystemExit(f"expected equilibrium_lcfs, got {env.cfg.boundary_mode}")
if observation.shape[0] != 8:
    raise SystemExit(f"GPU boundary smoke test returned wrong batch shape: {tuple(observation.shape)}")
print("preflight_boundary_mode", env.cfg.boundary_mode)
print("preflight_observation_shape", tuple(observation.shape))
print("preflight_action_dim", env.action_dim)
del env, observation
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PYCHECK

    python3 - <<"PYCUDA"
import torch
count = torch.cuda.device_count()
print("cuda_available", torch.cuda.is_available(), "device_count", count)
print("devices", [torch.cuda.get_device_name(index) for index in range(count)])
if count < 4:
    raise SystemExit(f"expected 4 visible CUDA devices, got {count}")
PYCUDA

    python3 -m torch.distributed.run \
      --nnodes 1 \
      --node_rank 0 \
      --nproc_per_node "${GPUS_PER_NODE}" \
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
        --batch-size "${BATCH_SIZE}" \
        --unroll-length 100 \
        --min-replay-sequence-length 100 \
        --replay-capacity-episodes "${REPLAY_CAPACITY_EPISODES}" \
        --rollout-chunk-length "${ROLLOUT_CHUNK_LENGTH}" \
        --updates-per-rollout-chunk "${UPDATES_PER_ROLLOUT_CHUNK}" \
        --hidden-dim 256 \
        --critic-hidden-dim 256 \
        --critic-mlp-hidden-dim 256 \
        --action-samples 64 \
        --actor-update-chunk-size 2048 \
        --save-checkpoints \
        --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}" \
        --eval-interval-steps "${EVAL_ENV_STEPS}" \
        --eval-episodes "${HOLDOUT_EVAL_EPISODES}" \
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

if [[ -d "${TRAIN_OUTPUT}/exports/best_actor" ]]; then
  srun \
    --ntasks=1 \
    --gres=gpu:1 \
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
        --export-dir "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/exports/best_actor" \
        --output-dir "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/oracle_baseline_eval_best_actor" \
        --all-windows \
        --steps 100 \
        --split holdout \
        --device cuda:0
    ' || echo "post-training oracle evaluation failed; training artifacts remain valid" >&2
fi
