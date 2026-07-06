#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-trim50-gpu1e6-24gpu
#SBATCH --partition=batch
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-trim50-gpu1e6-24gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-trim50-gpu1e6-24gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml}
export TRIM_MACHINE_CONFIG_SOURCE=${TRIM_MACHINE_CONFIG_SOURCE:-../tokamak-sim/configs/T15MD_new_data.toml}
export TRIM_DATA_ROOT=${TRIM_DATA_ROOT:-../tokamak-sim/data/t15_data_new_trim50}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export ORACLE_TARGET_DIR=${ORACLE_TARGET_DIR:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export ORACLE_MACHINE_CONFIG=${ORACLE_MACHINE_CONFIG:-data/processed/t15_new_trim50_plain_gpu1e6_machine_config.toml}
export ORACLE_CONFIG_OUT=${ORACLE_CONFIG_OUT:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_config.yaml}

export CONFIG_NAME=${CONFIG_NAME:-t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_24gpu_100m}
export RUN_NAME=${RUN_NAME:-t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_24gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-trim50-gpu1e6-balanced-24gpu-100m-${SLURM_JOB_ID:-manual}}

export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-100000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}

# 8-GPU reference job used 2048 envs, 8 ranks, 256 envs/rank, 64 updates/chunk.
# This 24-GPU scaling job keeps 256 envs/rank and keeps the same update density by
# scaling updates/chunk from 64 to 192.
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${SLURM_NNODES:-3}
export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
export NUM_ENVS=${NUM_ENVS:-6144}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-192}

export WANDB_MODE=${WANDB_MODE:-online}
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}
else
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
export MASTER_PORT=${MASTER_PORT:-$(( 20000 + (${SLURM_JOB_ID:-1593} % 40000) ))}
export HOLDOUT_EVAL_EPISODES=${HOLDOUT_EVAL_EPISODES:-1239}

if (( WORLD_SIZE != 24 )); then
  echo "this 24-GPU scaling job expected WORLD_SIZE=24, got WORLD_SIZE=${WORLD_SIZE} from NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
  exit 2
fi
if (( NUM_ENVS % WORLD_SIZE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi
if (( UPDATES_PER_ROLLOUT_CHUNK != 192 )); then
  echo "warning: UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK}; reference-density 24-GPU scaling uses 192" >&2
fi

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "missing trim50 training config: ${SOURCE_CONFIG}" >&2
  echo "run the trim50 oracle target build job first" >&2
  exit 2
fi
if [[ ! -f "${TRIM_MACHINE_CONFIG_SOURCE}" ]]; then
  echo "missing trim50 machine config source: ${TRIM_MACHINE_CONFIG_SOURCE}" >&2
  exit 2
fi
if [[ ! -d "${TRIM_DATA_ROOT}" ]]; then
  echo "missing trim50 data root: ${TRIM_DATA_ROOT}" >&2
  exit 2
fi

if [[ ! -f "${RESET_LIBRARY}" || ! -f "${ORACLE_TARGETS}" || ! -d "${ORACLE_TARGET_DIR}" ]]; then
  echo "trim50 oracle reset/target libraries are missing; building them inside the container"
  srun \
    --nodes=1 \
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
      export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
      cd /workspace/tokamak-rl-v2
      python3 scripts/write_trim50_plain_gpu1e6_machine_config.py \
        --source "${TRIM_MACHINE_CONFIG_SOURCE}" \
        --out "${ORACLE_MACHINE_CONFIG}"
      python3 scripts/build_t15_new_replay_window_oracle_targets.py \
        --base-config "${SOURCE_CONFIG}" \
        --data-root "${TRIM_DATA_ROOT}" \
        --machine-config "${ORACLE_MACHINE_CONFIG}" \
        --target-dir "${ORACLE_TARGET_DIR}" \
        --initial-library-out "${RESET_LIBRARY}" \
        --config-out "${ORACLE_CONFIG_OUT}" \
        --compute-backend gpu \
        --gpu-device cuda:0 \
        --batch-size 256 \
        --window-steps 100 \
        --angles 32 \
        --train-shots 3856 3857 3858 3863 \
        --holdout-shots 3864
    '
fi

if [[ ! -f "${RESET_LIBRARY}" ]]; then
  echo "missing trim50 oracle reset library after build: ${RESET_LIBRARY}" >&2
  exit 2
fi
if [[ ! -f "${ORACLE_TARGETS}" ]]; then
  echo "missing trim50 oracle targets after build: ${ORACLE_TARGETS}" >&2
  exit 2
fi
if [[ ! -d "${ORACLE_TARGET_DIR}" ]]; then
  echo "missing trim50 oracle target directory after build: ${ORACLE_TARGET_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname "${GENERATED_CONFIG}")"
python3 scripts/write_trim50_plain_gpu1e6_machine_config.py \
  --source "${TRIM_MACHINE_CONFIG_SOURCE}" \
  --out "$(dirname "${GENERATED_CONFIG}")/T15MD_new_data.toml"

python3 - <<PYCFG
import json
from pathlib import Path

source = Path("${SOURCE_CONFIG}")
out = Path("${GENERATED_CONFIG}")
cfg = json.loads(source.read_text(encoding="utf-8"))

cfg["name"] = "${CONFIG_NAME}"
cfg["sim"]["config_path"] = "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/generated_configs/T15MD_new_data.toml"
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
cfg["training"]["eval_max_steps"] = 100
cfg["training"]["save_checkpoints"] = True
cfg["training"]["early_stop_patience_evals"] = 0
cfg["training"]["early_stop_min_delta"] = 0.0
cfg["learner"]["updates_per_rollout_chunk"] = int("${UPDATES_PER_ROLLOUT_CHUNK}")

boundary_override_keys = sorted(
    key for key in cfg["sim"]
    if str(key).startswith("boundary_soft_level_") or str(key) == "boundary_smooth_selected_level"
)
if boundary_override_keys:
    raise SystemExit(
        "trim50 training config must not set RL-side boundary soft/smoothing overrides: "
        + ", ".join(boundary_override_keys)
    )

checks = {
    "trim50_config_copy": cfg["sim"]["config_path"].endswith("/generated_configs/T15MD_new_data.toml"),
    "trim50_reset_library": "trim50_plain_gpu1e6" in str(cfg["sim"]["csv_initial_state_library"]),
    "trim50_reference_dir": "trim50_plain_gpu1e6" in str(cfg["reference"]["boundary"]["replay_reference_dir"]),
    "replay_window_ip": cfg["reference"]["ip"]["kind"] == "replay_window",
    "replay_window_boundary": cfg["reference"]["boundary"]["kind"] == "t15_replay_segment_conditioned",
    "jdot_command": cfg["sim"]["action_contract"] == "jdot_command",
    "episode_steps_100": int(cfg["sim"]["max_episode_steps"]) == 100,
    "duration_0p1": abs(float(cfg["reference"]["duration_s"]) - 0.1) < 1e-12,
    "actor_obs_v6": cfg["observation"]["actor_kind"] == "controller_state_v6",
    "critic_obs_v2": cfg["observation"]["critic_kind"] == "compact_training_state_v2",
    "reward_tcv": cfg["reward"]["kind"] == "tcv_derivative",
    "updates_per_rollout_chunk_scaled": int(cfg["learner"]["updates_per_rollout_chunk"]) == int("${UPDATES_PER_ROLLOUT_CHUNK}"),
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"trim50 training config failed invariant checks: {failed}")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(out)
PYCFG

python3 scripts/assert_config_only_allowed_diffs.py \
  --source "${SOURCE_CONFIG}" \
  --generated "${GENERATED_CONFIG}" \
  --allow name \
  --allow sim.config_path \
  --allow sim.csv_initial_state_library \
  --allow reference.boundary.replay_reference_dir \
  --allow training.steps \
  --allow training.num_envs \
  --allow training.output_dir \
  --allow training.eval_interval_steps \
  --allow training.checkpoint_interval_steps \
  --allow training.milestone_checkpoint_interval_steps \
  --allow training.eval_checkpoint_top_k \
  --allow training.keep_latest_checkpoint \
  --allow training.distributed_mode \
  --allow training.production_mode \
  --allow training.eval_max_steps \
  --allow training.save_checkpoints \
  --allow training.early_stop_patience_evals \
  --allow training.early_stop_min_delta \
  --allow learner.updates_per_rollout_chunk

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_name=${RUN_NAME}"
echo "train_output=${TRAIN_OUTPUT}"
echo "source_config=${SOURCE_CONFIG}"
echo "generated_config=${GENERATED_CONFIG}"
echo "wandb_project=${WANDB_PROJECT_NAME}"
echo "train_env_steps=${TRAIN_ENV_STEPS}"
echo "num_envs=${NUM_ENVS}"
echo "nnodes=${NNODES}"
echo "gpus_per_node=${GPUS_PER_NODE}"
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
import math
import os
import torch
from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv

cfg = load_experiment_config("/workspace/tokamak-rl-v2/${GENERATED_CONFIG}")
print("preflight hostname", os.uname().nodename, "slurm_nodeid", os.environ.get("SLURM_NODEID"))
print("preflight config_path", cfg.sim.config_path)
print("preflight reset_library", cfg.sim.csv_initial_state_library)
print("preflight replay_reference_dir", cfg.reference.boundary.replay_reference_dir)
print("preflight action_contract", cfg.sim.action_contract)
print("preflight steps", cfg.training.steps, "num_envs", cfg.training.num_envs)
print("preflight updates_per_rollout_chunk", cfg.learner.updates_per_rollout_chunk)

required = [
    cfg.sim.config_path,
    cfg.sim.csv_initial_state_library,
    cfg.reference.boundary.replay_reference_dir,
]
missing = [str(p) for p in required if p is None or not Path(p).exists()]
if missing:
    raise SystemExit("generated trim50 training config has missing paths inside container: " + ", ".join(missing))

env_cfg = replace(cfg, sim=replace(cfg.sim, gpu_device="cuda:0"))
env = TokamakMagneticControlEnv(env_cfg, batch_size=1, device="cuda:0", seed=int(cfg.training.seed))
print("preflight boundary_mode", env.cfg.boundary_mode)
print("preflight boundary_base_mode", env.cfg.boundary_base_mode)
print("preflight boundary_legacy_precision_index2", env.cfg.boundary_legacy_precision_index2)
print("preflight boundary_smooth_selected_level", env.cfg.boundary_smooth_selected_level)
print("preflight boundary_soft_level_selection", env.cfg.boundary_soft_level_selection)
if not math.isclose(float(env.cfg.boundary_legacy_precision_index2), 1.0e-6, rel_tol=0.0, abs_tol=1.0e-12):
    raise SystemExit(
        "trim50 plain-gpu1e6 training must use boundary.legacy_precision_index2=1e-6; "
        f"loaded {env.cfg.boundary_legacy_precision_index2}"
    )
if str(env.cfg.boundary_mode) == "tracked_flux_contour":
    raise SystemExit("trim50 plain-gpu1e6 training must not use tracked_flux_contour boundary mode")
if bool(env.cfg.boundary_smooth_selected_level):
    raise SystemExit("trim50 plain-gpu1e6 training must not use boundary.smooth_selected_level")
if bool(env.cfg.boundary_soft_level_selection):
    raise SystemExit("trim50 plain-gpu1e6 training must not use boundary.soft_level_selection")
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

if [[ -d "${TRAIN_OUTPUT}/exports/best_actor" ]]; then
  srun \
    --nodes=1 \
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
        --export-dir "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/exports/best_actor" \
        --output-dir "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/oracle_baseline_eval_best_actor" \
        --all-windows \
        --steps 100 \
        --split holdout \
        --device cuda:0
    ' || echo "post-training oracle best-actor diagnostic failed; training artifacts remain valid" >&2
else
  echo "best actor export not found, skipping oracle best-actor diagnostic: ${TRAIN_OUTPUT}/exports/best_actor" >&2
fi
