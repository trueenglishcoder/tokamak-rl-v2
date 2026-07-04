#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-long-target
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-long-target-8gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-long-target-8gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

# This launcher intentionally delegates to the old-good 8-GPU replay-window
# oracle trainer. The only intended difference is the target/reset library.
OLD_GOOD_SOURCE_CONFIG=configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml
export SOURCE_CONFIG=${SOURCE_CONFIG:-${OLD_GOOD_SOURCE_CONFIG}}
export TRIM_MACHINE_CONFIG_SOURCE=${TRIM_MACHINE_CONFIG_SOURCE:-../tokamak-sim/configs/T15MD_new_data.toml}
export TRIM_DATA_ROOT=${TRIM_DATA_ROOT:-../tokamak-sim/data/t15_data_new_trim50_idealized_matched}

export BUILDER_SCRIPT=${BUILDER_SCRIPT:-scripts/build_t15_long_target_generated_trim50_idealized_matched_0p1s.py}
export GENERATED_TARGET_DIR=${GENERATED_TARGET_DIR:-data/processed/t15_long_target_generated_trim50_plain_gpu1e6_oracle_window_0p1s}
export GENERATED_TARGETS=${GENERATED_TARGETS:-${GENERATED_TARGET_DIR}/t15_long_target_generated_targets.npz}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_long_target_generated_trim50_plain_gpu1e6_oracle_window_0p1s_initial_states.npz}
export ORACLE_TARGET_DIR=${ORACLE_TARGET_DIR:-${GENERATED_TARGET_DIR}}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export PARENTS=${PARENTS:-${GENERATED_TARGET_DIR}/t15_long_target_generated_parents.npz}

export PARENT_COUNT=${PARENT_COUNT:-12}
export PARENT_MIN_STEPS=${PARENT_MIN_STEPS:-1000}
export PARENT_MAX_STEPS=${PARENT_MAX_STEPS:-1200}
export PARENT_STRIDE=${PARENT_STRIDE:-1}
export PERTURB_FRACTION=${PERTURB_FRACTION:-0.035}
export STATE_DISTANCE_LIMIT=${STATE_DISTANCE_LIMIT:-0.18}

export CONFIG_NAME=${CONFIG_NAME:-t15_long_target_generated_trim50_plain_gpu1e6_oracle_window_0p1s_tcvjdot_balanced_8gpu_100m}
export RUN_NAME=${RUN_NAME:-t15_long_target_generated_trim50_plain_gpu1e6_oracle_window_0p1s_tcvjdot_balanced_8gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-long-target-generated-8gpu-100m-${SLURM_JOB_ID:-manual}}
export HOLDOUT_EVAL_EPISODES=${HOLDOUT_EVAL_EPISODES:-1239}

if [[ "${SOURCE_CONFIG}" != "${OLD_GOOD_SOURCE_CONFIG}" ]]; then
  echo "this launch path must use the old-good source config exactly: ${OLD_GOOD_SOURCE_CONFIG}" >&2
  echo "got SOURCE_CONFIG=${SOURCE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "missing old-good source config: ${SOURCE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${BUILDER_SCRIPT}" ]]; then
  echo "missing long-target builder: ${BUILDER_SCRIPT}" >&2
  exit 2
fi
if [[ ! -d "${TRIM_DATA_ROOT}/ip" || ! -d "${TRIM_DATA_ROOT}/coils" ]]; then
  echo "missing matched idealized trim50 data root: ${TRIM_DATA_ROOT}" >&2
  echo "build it in tokamak-sim with scripts/build_t15_trim50_idealized_matched_pipeline.sh first" >&2
  exit 2
fi
if [[ ! -d "../tokamak-sim/output/t15_boundary_parameters_trim50_idealized_matched_gpu_plain_1e6" ]]; then
  echo "missing matched idealized boundary parameters: ../tokamak-sim/output/t15_boundary_parameters_trim50_idealized_matched_gpu_plain_1e6" >&2
  exit 2
fi

if [[ ! -f "${RESET_LIBRARY}" || ! -f "${GENERATED_TARGETS}" || ! -f "${ORACLE_TARGETS}" || ! -f "${PARENTS}" ]]; then
  echo "long-target generated replay-window/oracle libraries are missing; building them inside the container"
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
      export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2/scripts:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
      cd /workspace/tokamak-rl-v2

      python3 "${BUILDER_SCRIPT}" \
        --parent-count "${PARENT_COUNT}" \
        --parent-min-steps "${PARENT_MIN_STEPS}" \
        --parent-max-steps "${PARENT_MAX_STEPS}" \
        --parent-stride "${PARENT_STRIDE}" \
        --perturb-fraction "${PERTURB_FRACTION}" \
        --state-distance-limit "${STATE_DISTANCE_LIMIT}" \
        --out-dir "${GENERATED_TARGET_DIR}" \
        --initial-states-out "${RESET_LIBRARY}" \
        --targets-out "${GENERATED_TARGETS}" \
        --oracle-targets-out "${ORACLE_TARGETS}" \
        --parents-out "${PARENTS}"
    '
fi

if [[ ! -f "${RESET_LIBRARY}" ]]; then
  echo "missing generated reset library after build: ${RESET_LIBRARY}" >&2
  exit 2
fi
if [[ ! -f "${GENERATED_TARGETS}" ]]; then
  echo "missing generated diagnostic targets after build: ${GENERATED_TARGETS}" >&2
  exit 2
fi
if [[ ! -f "${ORACLE_TARGETS}" ]]; then
  echo "missing generated oracle targets after build: ${ORACLE_TARGETS}" >&2
  exit 2
fi

srun \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    cd /workspace/tokamak-rl-v2
    python3 - <<PY
import json
import numpy as np
from pathlib import Path
from tokamak_rl_v2.env.references import T15ReplayBoundaryLibrary

source = json.loads(Path("${SOURCE_CONFIG}").read_text(encoding="utf-8"))
if source["reference"]["ip"]["kind"] != "replay_window":
    raise SystemExit("source config is not the old-good replay_window Ip config")
if source["reference"]["boundary"]["kind"] != "t15_replay_segment_conditioned":
    raise SystemExit("source config is not the old-good replay-window boundary config")

with np.load("${ORACLE_TARGETS}", allow_pickle=False) as data:
    required = {"shot_id", "source_index", "time_s", "ip_target", "boundary_radii", "real_jdot_action", "difficulty_bin"}
    missing = sorted(required - set(data.files))
    if missing:
        raise SystemExit("generated oracle target is missing arrays: " + ", ".join(missing))
    if data["ip_target"].shape[1] != 101:
        raise SystemExit(f"expected generated ip_target shape [N,101], got {data['ip_target'].shape}")
    if data["boundary_radii"].shape[1:] != (101, 32):
        raise SystemExit(f"expected generated boundary_radii shape [N,101,32], got {data['boundary_radii'].shape}")
    if data["real_jdot_action"].shape[1:] != (100, 9):
        raise SystemExit(f"expected generated real_jdot_action shape [N,100,9], got {data['real_jdot_action'].shape}")
    keys = list(zip(data["shot_id"].astype(str).tolist(), data["source_index"].astype(int).tolist()))
    if len(keys) != len(set(keys)):
        raise SystemExit("generated oracle target has duplicate (shot_id, source_index) keys")
    print("generated_oracle_rows", int(data["ip_target"].shape[0]))

with np.load("${RESET_LIBRARY}", allow_pickle=False) as data:
    split = data["split"].astype(str)
    print("generated_reset_rows", int(split.shape[0]), "holdout", int((split == "holdout").sum()))

lib = T15ReplayBoundaryLibrary("${ORACLE_TARGET_DIR}", theta_count=32)
print("oracle_loader_shots", len(lib.shot_ids))
PY
  '

echo "delegating to old-good replay-window/oracle trainer"
echo "RUN_NAME=${RUN_NAME}"
echo "HOLDOUT_EVAL_EPISODES=${HOLDOUT_EVAL_EPISODES}"

bash jobs/train_t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch
