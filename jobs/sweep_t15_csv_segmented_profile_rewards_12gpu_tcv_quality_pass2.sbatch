#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvd-p2
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-11%12
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcv-derivative-half-slope-pass2-1gpu-%A_%a.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcv-derivative-half-slope-pass2-1gpu-%A_%a.err

set -euo pipefail

cd /scratch/$USER/tokamak

: "${PASS1_ROOT:?Set PASS1_ROOT to outputs/t15_reward_sweep48_tcv_derivative_half_slope_2m5m_<PASS1_JOBID>}"

export POLICY_CONFIG=configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-5000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export NUM_ENVS=${NUM_ENVS:-256}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}
export BATCH_SIZE=${BATCH_SIZE:-256}
export ROLLOUT_CHUNK_LENGTH=${ROLLOUT_CHUNK_LENGTH:-64}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-32}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}

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

    ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-0}
    PASS1_NAME=$(basename "${PASS1_ROOT}")
    SWEEP_ROOT=${PASS1_ROOT}/pass2_focused
    SWEEP_WANDB_PROJECT=tokamak-rl-v2-tcv-derivative-half-slope-pass2-${PASS1_NAME}
    MANIFEST=${SWEEP_ROOT}/variants.json

    mkdir -p "${SWEEP_ROOT}" "${SWEEP_ROOT}/generated_configs"
    if [[ ! -f "${MANIFEST}" ]]; then
      echo "missing pass2 reward-sweep manifest: ${MANIFEST}; pass1 aggregation must create it" >&2
      exit 2
    fi

    echo "policy_config=${POLICY_CONFIG}"
    echo "pass1_root=${PASS1_ROOT}"
    echo "sweep_root=${SWEEP_ROOT}"
    echo "manifest=${MANIFEST}"
    echo "wandb_project=${SWEEP_WANDB_PROJECT}"
    echo "array_task=${ARRAY_TASK}"
    echo "variant_index=${ARRAY_TASK}"
    echo "train_env_steps=${TRAIN_ENV_STEPS}"
    echo "num_envs=${NUM_ENVS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "replay_capacity_episodes=${REPLAY_CAPACITY_EPISODES}"
    echo "rollout_chunk_length=${ROLLOUT_CHUNK_LENGTH}"
    echo "updates_per_rollout_chunk=${UPDATES_PER_ROLLOUT_CHUNK}"
    echo "eval_env_steps=${EVAL_ENV_STEPS}"
    echo "save_checkpoints=0"

    python3 scripts/run_reward_sweep_candidate.py \
      --manifest "${MANIFEST}" \
      --variant-index "${ARRAY_TASK}" \
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

    echo "tcv-derivative reward sweep pass2 array task completed at $(date -Is)"
  '
