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

export REAL_SOURCE_CONFIG=${REAL_SOURCE_CONFIG:-configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml}
export REAL_RESET_LIBRARY=${REAL_RESET_LIBRARY:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export REAL_ORACLE_TARGET_DIR=${REAL_ORACLE_TARGET_DIR:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export REAL_ORACLE_TARGETS=${REAL_ORACLE_TARGETS:-${REAL_ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export ORACLE_TARGET_DIR=${ORACLE_TARGET_DIR:-data/processed/t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export CONFIG_NAME=${CONFIG_NAME:-t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m}
export RUN_NAME=${RUN_NAME:-t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-synlong30-trim50-gpu1e6-balanced-8gpu-100m-${SLURM_JOB_ID:-manual}}
export SYNTHETIC_TRAIN_PARENTS=${SYNTHETIC_TRAIN_PARENTS:-30}
export SYNTHETIC_HOLDOUT_PARENTS=${SYNTHETIC_HOLDOUT_PARENTS:-1}
export SYNTHETIC_MIN_STEPS=${SYNTHETIC_MIN_STEPS:-1000}
export SYNTHETIC_MAX_STEPS=${SYNTHETIC_MAX_STEPS:-1500}
export SYNTHETIC_SEED=${SYNTHETIC_SEED:-37}
export SYNTHETIC_WIGGLE_ROOM=${SYNTHETIC_WIGGLE_ROOM:-1.15}
export SYNTHETIC_MAX_ATTEMPTS_PER_PARENT=${SYNTHETIC_MAX_ATTEMPTS_PER_PARENT:-2000}

export TRIM_MACHINE_CONFIG_SOURCE=${TRIM_MACHINE_CONFIG_SOURCE:-../tokamak-sim/configs/T15MD_new_data.toml}
export TRIM_DATA_ROOT=${TRIM_DATA_ROOT:-../tokamak-sim/data/t15_data_new_trim50}
export ORACLE_MACHINE_CONFIG=${ORACLE_MACHINE_CONFIG:-data/processed/t15_new_trim50_plain_gpu1e6_machine_config.toml}
export ORACLE_CONFIG_OUT=${ORACLE_CONFIG_OUT:-data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_config.yaml}

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "missing synthetic long30 training config: ${SOURCE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${REAL_SOURCE_CONFIG}" ]]; then
  echo "missing canonical real trim50 training config: ${REAL_SOURCE_CONFIG}" >&2
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

if [[ ! -f "${REAL_RESET_LIBRARY}" || ! -f "${REAL_ORACLE_TARGETS}" || ! -d "${REAL_ORACLE_TARGET_DIR}" ]]; then
  echo "canonical real trim50 oracle libraries are missing; rebuilding them inside the container"
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
      export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
      cd /workspace/tokamak-rl-v2

      python3 scripts/write_trim50_plain_gpu1e6_machine_config.py \
        --source "${TRIM_MACHINE_CONFIG_SOURCE}" \
        --out "${ORACLE_MACHINE_CONFIG}"

      python3 scripts/build_t15_new_replay_window_oracle_targets.py \
        --base-config "${REAL_SOURCE_CONFIG}" \
        --data-root "${TRIM_DATA_ROOT}" \
        --machine-config "${ORACLE_MACHINE_CONFIG}" \
        --target-dir "${REAL_ORACLE_TARGET_DIR}" \
        --initial-library-out "${REAL_RESET_LIBRARY}" \
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

if [[ ! -f "${RESET_LIBRARY}" || ! -f "${ORACLE_TARGETS}" || ! -d "${ORACLE_TARGET_DIR}" ]]; then
  echo "synthetic long30 oracle libraries are missing; building them inside the container"
  srun \
    --ntasks=1 \
    --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
    --container-mounts /scratch/$USER/tokamak:/workspace \
    bash -lc '
      set -euo pipefail
      cd /workspace/tokamak-rl-v2
      python3 scripts/build_t15_synthetic_long30_trim50_plain_gpu1e6_oracle_windows.py \
        --oracle-target "${REAL_ORACLE_TARGETS}" \
        --initial-library "${REAL_RESET_LIBRARY}" \
        --target-dir "${ORACLE_TARGET_DIR}" \
        --initial-library-out "${RESET_LIBRARY}" \
        --train-parents "${SYNTHETIC_TRAIN_PARENTS}" \
        --holdout-parents "${SYNTHETIC_HOLDOUT_PARENTS}" \
        --min-steps "${SYNTHETIC_MIN_STEPS}" \
        --max-steps "${SYNTHETIC_MAX_STEPS}" \
        --seed "${SYNTHETIC_SEED}" \
        --wiggle-room "${SYNTHETIC_WIGGLE_ROOM}" \
        --max-attempts-per-parent "${SYNTHETIC_MAX_ATTEMPTS_PER_PARENT}"
    '
fi

if [[ ! -f "${RESET_LIBRARY}" ]]; then
  echo "missing synthetic long30 reset library after build: ${RESET_LIBRARY}" >&2
  exit 2
fi
if [[ ! -f "${ORACLE_TARGETS}" ]]; then
  echo "missing synthetic long30 oracle targets after build: ${ORACLE_TARGETS}" >&2
  exit 2
fi

if [[ -z "${HOLDOUT_EVAL_EPISODES:-}" ]]; then
  HOLDOUT_EVAL_EPISODES=$(python3 scripts/count_npz_string_value.py "${RESET_LIBRARY}" --array split --value holdout)
  export HOLDOUT_EVAL_EPISODES
fi
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
