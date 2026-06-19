#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-reward36-sat
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --array=0-11%12
#SBATCH --output=slurm_logs/tokamak-rl-v2-csv-reward36-saturation-1gpu-%A_%a.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-csv-reward36-saturation-1gpu-%A_%a.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-1000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-100000}
export NUM_ENVS=${NUM_ENVS:-256}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}
export BATCH_SIZE=${BATCH_SIZE:-256}
export ROLLOUT_CHUNK_LENGTH=${ROLLOUT_CHUNK_LENGTH:-64}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-32}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}
export SWEEP_ROOT_PREFIX=${SWEEP_ROOT_PREFIX:-outputs/t15_reward_sweep36_saturation_1m}

srun \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
    set -euo pipefail
    unset RUN_NAME
    unset TRAIN_OUTPUT
    unset WANDB_PROJECT_NAME

    mkdir -p /tmp/nvidia-lib
    ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
    ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
    ln -sf /host-libs/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so.1
    ln -sf /tmp/nvidia-lib/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so

    export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}

    cd /workspace/tokamak-rl-v2
    mkdir -p slurm_logs outputs data/processed
    python3 scripts/apply_t15_actuator_limits.py /workspace/tokamak-sim/configs/T15MD_new_data.toml
    python3 scripts/ensure_t15_reference_limits.py
    python3 scripts/ensure_t15_csv_initial_states.py \
      --experiment-config "${POLICY_CONFIG}" \
      --workers 8 \
      --max-rows-per-shot 600

    python3 - <<\PYPREFLIGHT
import pathlib
import re
import sys
import torch

required = [pathlib.Path("data/processed/t15_reference_limits.json"), pathlib.Path("data/processed/t15_csv_initial_states.npz")]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print("missing processed CSV production artifacts: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(2)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("no visible CUDA device for reward sweep task", file=sys.stderr)
    raise SystemExit(2)
toml = pathlib.Path("/workspace/tokamak-sim/configs/T15MD_new_data.toml").read_text(encoding="utf-8")
for key, expected in {"pfc_deriv_limit": 5070038.4, "sol_deriv_limit": 20950244.0}.items():
    match = re.search(rf"^{key}\s*=\s*([0-9.eE+-]+)", toml, flags=re.MULTILINE)
    if not match or abs(float(match.group(1)) - expected) > 1e-6:
        print(f"{key} was not patched to corrected legal value {expected}", file=sys.stderr)
        raise SystemExit(2)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYPREFLIGHT

    ARRAY_JOB=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
    ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-0}
    SWEEP_ROOT=${SWEEP_ROOT_PREFIX}_${ARRAY_JOB}
    SWEEP_WANDB_PROJECT=tokamak-rl-v2-reward36-saturation-${ARRAY_JOB}
    MANIFEST=${SWEEP_ROOT}/variants.json

    mkdir -p "${SWEEP_ROOT}" "${SWEEP_ROOT}/generated_configs"
    if [[ ! -f "${MANIFEST}" ]]; then
      echo "missing saturation reward-sweep manifest: ${MANIFEST}; launch through scripts/submit_saturation_reward_sweep.py" >&2
      exit 2
    fi

    echo "policy_config=${POLICY_CONFIG}"
    echo "sweep_root=${SWEEP_ROOT}"
    echo "manifest=${MANIFEST}"
    echo "wandb_project=${SWEEP_WANDB_PROJECT}"
    echo "array_job=${ARRAY_JOB}"
    echo "array_task=${ARRAY_TASK}"
    echo "train_env_steps=${TRAIN_ENV_STEPS}"
    echo "num_envs=${NUM_ENVS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "replay_capacity_episodes=${REPLAY_CAPACITY_EPISODES}"
    echo "rollout_chunk_length=${ROLLOUT_CHUNK_LENGTH}"
    echo "updates_per_rollout_chunk=${UPDATES_PER_ROLLOUT_CHUNK}"
    echo "eval_env_steps=${EVAL_ENV_STEPS}"
    echo "save_checkpoints=0"

    for LOCAL_RUN_INDEX in 0 1 2; do
      VARIANT_INDEX=$(( ARRAY_TASK * 3 + LOCAL_RUN_INDEX ))
      python3 scripts/run_reward_sweep_candidate.py \
        --manifest "${MANIFEST}" \
        --variant-index "${VARIANT_INDEX}" \
        --base-config "${POLICY_CONFIG}" \
        --sweep-root "${SWEEP_ROOT}" \
        --train-env-steps "${TRAIN_ENV_STEPS}" \
        --eval-env-steps "${EVAL_ENV_STEPS}" \
        --num-envs "${NUM_ENVS}" \
        --batch-size "${BATCH_SIZE}" \
        --replay-capacity-episodes "${REPLAY_CAPACITY_EPISODES}" \
        --rollout-chunk-length "${ROLLOUT_CHUNK_LENGTH}" \
        --updates-per-rollout-chunk "${UPDATES_PER_ROLLOUT_CHUNK}" \
        --wandb-project "${SWEEP_WANDB_PROJECT}" \
        --wandb-mode "${WANDB_MODE}"
    done

    echo "saturation reward sweep one-pass array task completed at $(date -Is)"
  '
