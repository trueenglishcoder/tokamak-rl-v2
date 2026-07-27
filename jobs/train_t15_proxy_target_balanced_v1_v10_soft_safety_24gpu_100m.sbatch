#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-t15-proxy-target-balanced-v1-v10-soft-safety-24gpu100m
#SBATCH --partition=batch
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-t15-proxy-target-balanced-v1-v10-soft-safety-24gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-t15-proxy-target-balanced-v1-v10-soft-safety-24gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_proxy_target_balanced_v1_v10_soft_safety_24gpu_100m.yaml}
export TRIM_MACHINE_CONFIG_SOURCE=${TRIM_MACHINE_CONFIG_SOURCE:-../tokamak-sim/configs/T15MD.toml}
export TARGET_DATASET_DIR=${TARGET_DATASET_DIR:-data/processed/t15_proxy_target_balanced_v1}
export RESET_LIBRARY=${RESET_LIBRARY:-${TARGET_DATASET_DIR}/t15_proxy_target_v1_initial_states.npz}
export TARGETS=${TARGETS:-${TARGET_DATASET_DIR}/t15_target_trajectory_targets.npz}
export CONFIG_NAME=${CONFIG_NAME:-t15_proxy_target_balanced_v1_v10_soft_safety_24gpu_100m}
export RUN_NAME=${RUN_NAME:-t15_proxy_target_balanced_v1_v10_soft_safety_24gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-t15-proxy-target-balanced-v1-v10-soft-safety-24gpu-100m-${SLURM_JOB_ID:-manual}}

export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-100000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${SLURM_NNODES:-3}
export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
export NUM_ENVS=${NUM_ENVS:-6144}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-64}
export WANDB_MODE=${WANDB_MODE:-online}
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}
else
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
export MASTER_PORT=${MASTER_PORT:-$(( 20000 + (${SLURM_JOB_ID:-2468} % 40000) ))}

if (( WORLD_SIZE != 24 )); then
  echo "this job expects WORLD_SIZE=24, got WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi
if (( NUM_ENVS % WORLD_SIZE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi
if (( NUM_ENVS / WORLD_SIZE != 256 )); then
  echo "warning: local envs per rank is $(( NUM_ENVS / WORLD_SIZE )); intended target is 256" >&2
fi
if (( UPDATES_PER_ROLLOUT_CHUNK != 64 )); then
  echo "warning: UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK}; intended diagnostic target is 64" >&2
fi

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "missing target-only source config: ${SOURCE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${TRIM_MACHINE_CONFIG_SOURCE}" ]]; then
  echo "missing tokamak-sim machine config: ${TRIM_MACHINE_CONFIG_SOURCE}" >&2
  exit 2
fi
if [[ ! -f "${RESET_LIBRARY}" ]]; then
  echo "missing target-only reset library: ${RESET_LIBRARY}" >&2
  echo "build it first with jobs/build_t15_proxy_target_balanced_dataset.sbatch" >&2
  exit 2
fi
if [[ ! -f "${TARGETS}" ]]; then
  echo "missing target-only reference targets: ${TARGETS}" >&2
  echo "build it first with jobs/build_t15_proxy_target_balanced_dataset.sbatch" >&2
  exit 2
fi

HOLDOUT_EVAL_EPISODES=$(srun \
  --nodes=1 \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace \
  bash -lc '
    set -euo pipefail
    cd /workspace/tokamak-rl-v2
    python3 scripts/count_npz_string_value.py "${RESET_LIBRARY}" --array split --value holdout
  ')
export HOLDOUT_EVAL_EPISODES
if [[ "${HOLDOUT_EVAL_EPISODES}" -le 0 ]]; then
  echo "target-only reset library has no holdout rows" >&2
  exit 2
fi

srun \
  --nodes=1 \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace \
  bash -lc '
    set -euo pipefail
    cd /workspace/tokamak-rl-v2
    python3 - <<PYVALID
from pathlib import Path
import numpy as np

reset_path = Path("${RESET_LIBRARY}")
target_path = Path("${TARGETS}")
with np.load(reset_path, allow_pickle=False) as reset, np.load(target_path, allow_pickle=False) as target:
    reset_required = {"shot_id", "source_index", "time_s", "ip0", "pfc0", "sol0", "split", "difficulty_bin"}
    target_required = {"shot_id", "source_index", "time_s", "difficulty_bin", "ip_target", "boundary_radii", "split", "target_only"}
    missing_reset = sorted(reset_required - set(reset.files))
    missing_target = sorted(target_required - set(target.files))
    if missing_reset or missing_target:
        raise SystemExit(f"target-only dataset missing arrays: reset={missing_reset} target={missing_target}")
    if "real_jdot_action" in target.files:
        raise SystemExit("target-only dataset must not contain real_jdot_action")
    n_reset = int(np.asarray(reset["ip0"]).shape[0])
    n_target = int(np.asarray(target["ip_target"]).shape[0])
    if n_reset <= 0 or n_target <= 0 or n_reset != n_target:
        raise SystemExit(f"target-only row count mismatch: reset={n_reset} target={n_target}")
    ip_shape = np.asarray(target["ip_target"]).shape
    radii_shape = np.asarray(target["boundary_radii"]).shape
    if ip_shape[1] != 101 or radii_shape[1:3] != (101, 32):
        raise SystemExit(f"target-only shape mismatch: ip={ip_shape} radii={radii_shape}")
    source_index = np.asarray(target["source_index"], dtype=np.int64)
    if not np.any(np.diff(np.sort(np.unique(source_index))) == 1):
        raise SystemExit("target-only dataset does not show one-step overlapping window starts")
    split = np.asarray(reset["split"]).astype(str)
    train = int(np.sum(split == "train"))
    holdout = int(np.sum(split == "holdout"))
    if train <= 0 or holdout <= 0:
        raise SystemExit(f"target-only dataset needs train and holdout rows, got train={train} holdout={holdout}")
    print(f"target_only_validation_ok rows={n_reset} train={train} holdout={holdout}")
PYVALID
  '

mkdir -p "$(dirname "${GENERATED_CONFIG}")"
python3 scripts/write_trim50_plain_gpu1e6_machine_config.py \
  --source "${TRIM_MACHINE_CONFIG_SOURCE}" \
  --out "$(dirname "${GENERATED_CONFIG}")/T15MD.toml"

python3 - <<PYCFG
import json
from pathlib import Path

source = Path("${SOURCE_CONFIG}")
out = Path("${GENERATED_CONFIG}")
cfg = json.loads(source.read_text(encoding="utf-8"))

cfg["name"] = "${CONFIG_NAME}"
cfg["sim"]["config_path"] = "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/generated_configs/T15MD.toml"
cfg["sim"]["csv_initial_state_library"] = "/workspace/tokamak-rl-v2/${RESET_LIBRARY}"
cfg["reference"]["boundary"]["replay_reference_dir"] = "/workspace/tokamak-rl-v2/${TARGET_DATASET_DIR}"
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
cfg["training"]["eval_max_steps"] = 100
cfg["training"]["save_checkpoints"] = True
cfg["training"]["early_stop_patience_evals"] = 0
cfg["training"]["early_stop_min_delta"] = 0.0
cfg["learner"]["updates_per_rollout_chunk"] = int("${UPDATES_PER_ROLLOUT_CHUNK}")

checks = {
    "target_only_reset_library": "t15_proxy_target_balanced_v1" in str(cfg["sim"]["csv_initial_state_library"]),
    "target_only_reference_dir": "t15_proxy_target_balanced_v1" in str(cfg["reference"]["boundary"]["replay_reference_dir"]),
    "replay_window_ip_loader": cfg["reference"]["ip"]["kind"] == "replay_window",
    "target_boundary_loader": cfg["reference"]["boundary"]["kind"] == "t15_replay_segment_conditioned",
    "actor_obs_v8_safety_margin": cfg["observation"]["actor_kind"] == "controller_state_v8_safety_margin",
    "steps_100m": int(cfg["training"]["steps"]) == 100000000,
    "num_envs_6144": int(cfg["training"]["num_envs"]) == 6144,
    "updates_per_rollout_chunk_64": int(cfg["learner"]["updates_per_rollout_chunk"]) == 64,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"target-only generated config failed invariant checks: {failed}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(out)
PYCFG

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_name=${RUN_NAME}"
echo "train_output=${TRAIN_OUTPUT}"
echo "generated_config=${GENERATED_CONFIG}"
echo "wandb_project=${WANDB_PROJECT_NAME}"
echo "num_envs=${NUM_ENVS}"
echo "world_size=${WORLD_SIZE}"
echo "local_envs_per_rank=$(( NUM_ENVS / WORLD_SIZE ))"
echo "updates_per_rollout_chunk=${UPDATES_PER_ROLLOUT_CHUNK}"
echo "holdout_eval_episodes=${HOLDOUT_EVAL_EPISODES}"
echo "master_addr=${MASTER_ADDR}"
echo "master_port=${MASTER_PORT}"

srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
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
from pathlib import Path
from dataclasses import replace
import os
import torch
from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv

cfg = load_experiment_config("/workspace/tokamak-rl-v2/${GENERATED_CONFIG}")
print("preflight hostname", os.uname().nodename, "slurm_nodeid", os.environ.get("SLURM_NODEID"))
print("preflight reset_library", cfg.sim.csv_initial_state_library)
print("preflight target_reference_dir", cfg.reference.boundary.replay_reference_dir)
print("preflight steps", cfg.training.steps, "num_envs", cfg.training.num_envs)
print("preflight updates_per_rollout_chunk", cfg.learner.updates_per_rollout_chunk)
print("preflight actor_kind", cfg.observation.actor_kind)
required = [cfg.sim.config_path, cfg.sim.csv_initial_state_library, cfg.reference.boundary.replay_reference_dir]
missing = [str(p) for p in required if p is None or not Path(p).exists()]
if missing:
    raise SystemExit("generated target-only config has missing paths: " + ", ".join(missing))
if cfg.observation.actor_kind != "controller_state_v8_safety_margin":
    raise SystemExit(f"target-only safety run requires actor_kind=controller_state_v8_safety_margin, got {cfg.observation.actor_kind}")
if int(cfg.training.steps) != 100000000 or int(cfg.training.num_envs) != 6144:
    raise SystemExit("target-only safety run expected 100M steps and 6144 envs")
if int(cfg.learner.updates_per_rollout_chunk) != 64:
    raise SystemExit("target-only safety run expected updates_per_rollout_chunk=64")

env_cfg = replace(cfg, sim=replace(cfg.sim, gpu_device="cuda:0"))
env = TokamakMagneticControlEnv(env_cfg, batch_size=1, device="cuda:0", seed=int(cfg.training.seed))
obs = env.reset()
critic_obs = env.critic_obs()
print("preflight env_reset_ok", "obs_dim", tuple(obs.shape), "critic_obs_dim", tuple(critic_obs.shape), "action_dim", env.action_dim)
del env, obs, critic_obs
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PYCHECK

    python3 - <<PYCUDA
import os
import torch
count = torch.cuda.device_count()
expected = int(os.environ.get("GPUS_PER_NODE", "8"))
print("cuda_available", torch.cuda.is_available(), "device_count", count)
print("devices", [torch.cuda.get_device_name(i) for i in range(count)])
if count < expected:
    raise SystemExit(f"expected at least {expected} visible CUDA devices on each node, got {count}")
PYCUDA

    NODE_RANK=${SLURM_NODEID:-0}
    python3 -m torch.distributed.run \
      --nnodes "${NNODES}" \
      --node_rank "${NODE_RANK}" \
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
        --batch-size 32 \
        --unroll-length 100 \
        --min-replay-sequence-length 100 \
        --replay-capacity-episodes 1024 \
        --rollout-chunk-length 100 \
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
