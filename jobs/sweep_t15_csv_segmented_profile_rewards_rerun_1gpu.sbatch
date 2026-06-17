#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-reward288-rerun
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/tokamak-rl-v2-csv-reward288-rerun-1gpu-%A_%a.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-csv-reward288-rerun-1gpu-%A_%a.err

set -euo pipefail

cd /scratch/$USER/tokamak

: "${RERUN_MANIFEST:?Set RERUN_MANIFEST to outputs/.../selection/missing_or_failed_variants.json}"
export POLICY_CONFIG=configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-1000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-100000}
export NUM_ENVS=${NUM_ENVS:-256}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-512}
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

    python3 - <<\PYPREFLIGHT
import pathlib
import re
import sys
import torch

required = [
    pathlib.Path("data/processed/t15_reference_limits.json"),
    pathlib.Path("data/processed/t15_csv_initial_states.npz"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print("missing processed CSV production artifacts: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(2)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("no visible CUDA device for reward sweep rerun task", file=sys.stderr)
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
    RUN_ENV_FILE=$(mktemp)
    python3 - "${RERUN_MANIFEST}" "${ARRAY_TASK}" > "${RUN_ENV_FILE}" <<\PYENV
import json
import pathlib
import shlex
import sys

path = pathlib.Path(sys.argv[1])
index = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
variants = payload.get("variants") or []
if index < 0 or index >= len(variants):
    raise SystemExit(f"rerun task index {index} out of range for {len(variants)} variants")
item = variants[index]
print("SOURCE_ROOT=" + shlex.quote(str(item["root"])))
print("SOURCE_MANIFEST=" + shlex.quote(str(item["manifest"])))
print("VARIANT_INDEX=" + shlex.quote(str(item["variant_index"])))
print("VARIANT_FOLDER=" + shlex.quote(str(item["folder"])))
PYENV
    source "${RUN_ENV_FILE}"
    rm -f "${RUN_ENV_FILE}"

    RERUN_NAME=$(basename "$(dirname "${RERUN_MANIFEST}")")
    SWEEP_WANDB_PROJECT=tokamak-rl-v2-reward288-legal-rerun-${RERUN_NAME}-${ARRAY_JOB}

    echo "rerun_manifest=${RERUN_MANIFEST}"
    echo "source_root=${SOURCE_ROOT}"
    echo "source_manifest=${SOURCE_MANIFEST}"
    echo "variant_index=${VARIANT_INDEX}"
    echo "variant_folder=${VARIANT_FOLDER}"
    echo "wandb_project=${SWEEP_WANDB_PROJECT}"

    python3 scripts/run_reward_sweep_candidate.py \
      --manifest "${SOURCE_MANIFEST}" \
      --variant-index "${VARIANT_INDEX}" \
      --base-config "${POLICY_CONFIG}" \
      --sweep-root "${SOURCE_ROOT}" \
      --train-env-steps "${TRAIN_ENV_STEPS}" \
      --eval-env-steps "${EVAL_ENV_STEPS}" \
      --num-envs "${NUM_ENVS}" \
      --batch-size "${BATCH_SIZE}" \
      --replay-capacity-episodes "${REPLAY_CAPACITY_EPISODES}" \
      --rollout-chunk-length "${ROLLOUT_CHUNK_LENGTH}" \
      --updates-per-rollout-chunk "${UPDATES_PER_ROLLOUT_CHUNK}" \
      --wandb-project "${SWEEP_WANDB_PROJECT}" \
      --wandb-mode "${WANDB_MODE}"

    echo "reward sweep rerun array task completed at $(date -Is)"
  '
