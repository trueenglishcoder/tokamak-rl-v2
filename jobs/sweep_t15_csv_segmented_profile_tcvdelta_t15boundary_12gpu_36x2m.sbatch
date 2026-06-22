#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvd-t15b-sweep
#SBATCH --partition=batch
#SBATCH --array=0-11%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvd-t15b-sweep-%A_%a.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvd-t15b-sweep-%A_%a.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export ROOT=${ROOT:-outputs/t15_reward_sweep36_tcvdelta_t15boundary_2m_${SLURM_ARRAY_JOB_ID}}
export MANIFEST=${MANIFEST:-${ROOT}/variants.json}
export BASE_CONFIG=${BASE_CONFIG:-configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-2000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-200000}
export NUM_ENVS=${NUM_ENVS:-256}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-288}
export BATCH_SIZE=${BATCH_SIZE:-256}
export ROLLOUT_CHUNK_LENGTH=${ROLLOUT_CHUNK_LENGTH:-64}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-32}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvdelta-t15boundary-36x2m-${SLURM_ARRAY_JOB_ID}}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-60}

if [[ ! -f "${MANIFEST}" ]]; then
  echo "missing manifest ${MANIFEST}; use scripts/submit_tcvdelta_t15boundary_reward_sweep.py" >&2
  exit 2
fi
if [[ ! -f data/processed/t15_csv_initial_states.npz ]]; then
  echo "missing data/processed/t15_csv_initial_states.npz" >&2
  exit 2
fi
if [[ ! -f data/processed/t15_reference_limits.json ]]; then
  echo "missing data/processed/t15_reference_limits.json" >&2
  exit 2
fi
if [[ ! -d ../tokamak-sim/runs/t15md_limited_replay_dataset ]]; then
  echo "missing ../tokamak-sim/runs/t15md_limited_replay_dataset" >&2
  exit 2
fi
python3 - <<'PYPREFLIGHT'
from pathlib import Path

reset_library = Path("data/processed/t15_csv_initial_states.npz")
replay_dir = Path("../tokamak-sim/runs/t15md_limited_replay_dataset")
required = {"3854", "3855", "3856", "3857", "3858", "3859", "3862", "3863", "3864"}
if not reset_library.exists():
    raise SystemExit(f"missing {reset_library}")
smoothed = sorted(replay_dir.glob("lqr_boundary_reference_*_smoothed.npz"))
if not smoothed:
    raise SystemExit(
        "missing smoothed replay boundary references in "
        f"{replay_dir}; run tokamak-sim/scripts/smooth_lqr_boundary_references.py"
    )
available = {
    path.name.removeprefix("lqr_boundary_reference_").removesuffix("_smoothed.npz")
    for path in smoothed
}
missing = sorted(required - available)
if missing:
    raise SystemExit(
        "smoothed replay boundary references do not cover reset shots: "
        + ", ".join(missing)
    )
print(f"smoothed_replay_boundary_references={len(smoothed)}")
PYPREFLIGHT

echo "root=${ROOT}"
echo "manifest=${MANIFEST}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "wandb_project=${WANDB_PROJECT_NAME}"
echo "train_env_steps=${TRAIN_ENV_STEPS}"
echo "eval_env_steps=${EVAL_ENV_STEPS}"
echo "num_envs=${NUM_ENVS}"

for LOCAL_INDEX in 0 1 2; do
  VARIANT_INDEX=$(( SLURM_ARRAY_TASK_ID * 3 + LOCAL_INDEX ))
  if (( VARIANT_INDEX >= 36 )); then
    continue
  fi

  VARIANT_NAME=$(python3 scripts/tcvdelta_t15boundary_sweep.py variant-name --manifest "${MANIFEST}" --index "${VARIANT_INDEX}")
  CANDIDATE_DIR="${ROOT}/${VARIANT_NAME}"
  GENERATED_CONFIG="${CANDIDATE_DIR}/config.json"
  export VARIANT_INDEX VARIANT_NAME CANDIDATE_DIR GENERATED_CONFIG
  mkdir -p "${CANDIDATE_DIR}"

  python3 scripts/tcvdelta_t15boundary_sweep.py generate-config \
    --base-config "${BASE_CONFIG}" \
    --manifest "${MANIFEST}" \
    --index "${VARIANT_INDEX}" \
    --output-dir "/workspace/tokamak-rl-v2/${CANDIDATE_DIR}" \
    --config-out "${GENERATED_CONFIG}" \
    --train-steps "${TRAIN_ENV_STEPS}" \
    --eval-steps "${EVAL_ENV_STEPS}" \
    --num-envs "${NUM_ENVS}" \
    --replay-capacity-episodes "${REPLAY_CAPACITY_EPISODES}" \
    --rl-root "/workspace/tokamak-rl-v2" \
    --sim-root "/workspace/tokamak-sim"

  echo "starting variant_index=${VARIANT_INDEX} variant_name=${VARIANT_NAME} at $(date -Is)"

  set +e
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
      ln -sf /host-libs/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so.1
      ln -sf /tmp/nvidia-lib/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so

      export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
      export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

      cd /workspace/tokamak-rl-v2
      python3 - <<PYCUDA
import torch
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("expected one visible CUDA device")
PYCUDA

      python3 scripts/train_policy_pipeline.py \
        --config "/workspace/tokamak-rl-v2/${GENERATED_CONFIG}" \
        --output-dir "/workspace/tokamak-rl-v2/${CANDIDATE_DIR}" \
        --steps "${TRAIN_ENV_STEPS}" \
        --device cuda:0 \
        --sim-compute-backend gpu \
        --sim-gpu-device cuda:0 \
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
        --no-save-checkpoints \
        --eval-interval-steps "${EVAL_ENV_STEPS}" \
        --eval-episodes 128 \
        --eval-max-steps 2000 \
        --controller-rollout-steps 0 \
        --reward-sweep-mode \
        --no-export \
        --wandb \
        --wandb-project "${WANDB_PROJECT_NAME}" \
        --wandb-name "${VARIANT_NAME}" \
        --wandb-group "${ROOT##*/}" \
        --wandb-mode "${WANDB_MODE}" \
        --wandb-optional \
        --wandb-metric-preset sweep
    '
  RUN_RC=$?
  set -e

  rm -rf "${CANDIDATE_DIR}/checkpoints" "${CANDIDATE_DIR}/exports"

  if (( RUN_RC != 0 )); then
    python3 - "${CANDIDATE_DIR}" "${RUN_RC}" "${VARIANT_INDEX}" "${VARIANT_NAME}" <<'PYFAIL'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
rc = int(sys.argv[2])
variant_index = int(sys.argv[3])
variant_name = sys.argv[4]
path = out / "policy_validation.json"
if not path.exists():
    path.write_text(
        json.dumps(
            {
                "status": "sweep_failed_training",
                "training_status": "slurm_candidate_failed",
                "return_code": rc,
                "variant_index": variant_index,
                "variant_name": variant_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
PYFAIL
    echo "variant ${VARIANT_NAME} failed with rc=${RUN_RC}; continuing"
  else
    echo "variant ${VARIANT_NAME} completed at $(date -Is)"
  fi
done

echo "array task ${SLURM_ARRAY_TASK_ID} completed at $(date -Is)"
