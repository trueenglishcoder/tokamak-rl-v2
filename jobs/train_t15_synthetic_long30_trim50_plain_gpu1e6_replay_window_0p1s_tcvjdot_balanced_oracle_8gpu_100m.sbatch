#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-synlong30
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-synlong30-8gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-synlong30-8gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export ORACLE_TARGET_DIR=${ORACLE_TARGET_DIR:-data/processed/t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export CONFIG_NAME=${CONFIG_NAME:-t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m}
export RUN_NAME=${RUN_NAME:-t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-synlong30-trim50-gpu1e6-balanced-8gpu-100m-${SLURM_JOB_ID:-manual}}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-100000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}
export NUM_ENVS=${NUM_ENVS:-2048}

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "missing synthetic long30 training config: ${SOURCE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${RESET_LIBRARY}" ]]; then
  echo "missing synthetic long30 reset library: ${RESET_LIBRARY}" >&2
  echo "build it first with jobs/build_t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_1gpu.sbatch" >&2
  exit 2
fi
if [[ ! -f "${ORACLE_TARGETS}" ]]; then
  echo "missing synthetic long30 oracle targets: ${ORACLE_TARGETS}" >&2
  echo "build them first with jobs/build_t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_1gpu.sbatch" >&2
  exit 2
fi
if [[ ! -d "${ORACLE_TARGET_DIR}" ]]; then
  echo "missing synthetic long30 oracle target directory: ${ORACLE_TARGET_DIR}" >&2
  exit 2
fi

HOLDOUT_EVAL_EPISODES=$(python3 scripts/count_npz_string_value.py "${RESET_LIBRARY}" --array split --value holdout)
export HOLDOUT_EVAL_EPISODES
if [[ "${HOLDOUT_EVAL_EPISODES}" -le 0 ]]; then
  echo "synthetic long30 reset library has no holdout windows" >&2
  exit 2
fi

srun \
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
target_path = Path("${ORACLE_TARGETS}")
with np.load(reset_path, allow_pickle=False) as reset, np.load(target_path, allow_pickle=False) as target:
    reset_required = {"shot_id", "source_index", "time_s", "ip0", "pfc0", "sol0", "split", "difficulty_bin"}
    target_required = {
        "shot_id",
        "source_index",
        "time_s",
        "difficulty_bin",
        "ip_target",
        "boundary_radii",
        "real_jdot_action",
        "current_limits",
        "derivative_limits",
    }
    missing_reset = sorted(reset_required - set(reset.files))
    missing_target = sorted(target_required - set(target.files))
    if missing_reset or missing_target:
        raise SystemExit(f"synthetic long30 dataset missing arrays: reset={missing_reset} target={missing_target}")
    n_reset = int(np.asarray(reset["ip0"]).shape[0])
    n_target = int(np.asarray(target["ip_target"]).shape[0])
    if n_reset <= 0 or n_target <= 0 or n_reset != n_target:
        raise SystemExit(f"synthetic long30 row count mismatch: reset={n_reset} target={n_target}")
    ip_shape = np.asarray(target["ip_target"]).shape
    radii_shape = np.asarray(target["boundary_radii"]).shape
    action_shape = np.asarray(target["real_jdot_action"]).shape
    if ip_shape[1] != 101 or radii_shape[1:3] != (101, 32) or action_shape[1:] != (100, 9):
        raise SystemExit(f"synthetic long30 shape mismatch: ip={ip_shape} radii={radii_shape} action={action_shape}")
    split = np.asarray(reset["split"]).astype(str)
    train = int(np.sum(split == "train"))
    holdout = int(np.sum(split == "holdout"))
    if train <= 0 or holdout <= 0:
        raise SystemExit(f"synthetic long30 needs train and holdout rows, got train={train} holdout={holdout}")
    print(f"synthetic_long30_validation_ok rows={n_reset} train={train} holdout={holdout}")
PYVALID
  '

echo "synthetic_long30_reset_library=${RESET_LIBRARY}"
echo "synthetic_long30_oracle_targets=${ORACLE_TARGETS}"
echo "synthetic_long30_report=${ORACLE_TARGET_DIR}/synthetic_long30_report.md"
echo "synthetic_long30_holdout_eval_episodes=${HOLDOUT_EVAL_EPISODES}"

exec bash jobs/train_t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch
