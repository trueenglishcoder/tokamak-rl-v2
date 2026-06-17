#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-reward144
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --array=0-47%48
#SBATCH --output=slurm_logs/tokamak-rl-v2-csv-reward144-1gpu-%A_%a.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-csv-reward144-1gpu-%A_%a.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml
export TRAIN_ENV_STEPS=1000000
export EVAL_ENV_STEPS=100000
export CHECKPOINT_ENV_STEPS=500000
export SAVE_CHECKPOINTS=0
export NUM_ENVS=256
export REPLAY_CAPACITY_EPISODES=512
export BATCH_SIZE=256
export ROLLOUT_CHUNK_LENGTH=64
export UPDATES_PER_ROLLOUT_CHUNK=32
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

    if [[ ! -f data/processed/t15_reference_limits.json || ! -f data/processed/t15_csv_initial_states.npz ]]; then
      echo "missing processed CSV production artifacts; run the production data preparation job before the reward sweep" >&2
      exit 2
    fi

    ARRAY_JOB=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
    ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-0}
    SWEEP_ROOT=outputs/t15_reward_sweep144_legal_1m_${ARRAY_JOB}
    SWEEP_WANDB_PROJECT=tokamak-rl-v2-reward144-legal-1m-${ARRAY_JOB}
    MANIFEST=${SWEEP_ROOT}/variants.json

    mkdir -p "${SWEEP_ROOT}" "${SWEEP_ROOT}/generated_configs"
    MANIFEST_TMP=${MANIFEST}.${ARRAY_TASK}.$$.tmp
    python3 scripts/build_reward_sweep_manifest.py --output "${MANIFEST_TMP}" >/dev/null
    mv "${MANIFEST_TMP}" "${MANIFEST}"

    python3 - <<\PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYCUDA

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
    echo "save_checkpoints=${SAVE_CHECKPOINTS}"

    for LOCAL_RUN_INDEX in 0 1 2; do
      VARIANT_INDEX=$(( ARRAY_TASK * 3 + LOCAL_RUN_INDEX ))
      RUN_ENV_FILE=$(mktemp)
      python3 - "${VARIANT_INDEX}" "${MANIFEST}" "${POLICY_CONFIG}" "${SWEEP_ROOT}" "${TRAIN_ENV_STEPS}" "${EVAL_ENV_STEPS}" "${SAVE_CHECKPOINTS}" "${REPLAY_CAPACITY_EPISODES}" "${BATCH_SIZE}" "${NUM_ENVS}" "${ROLLOUT_CHUNK_LENGTH}" "${UPDATES_PER_ROLLOUT_CHUNK}" > "${RUN_ENV_FILE}" <<\PYCFG
import json
import pathlib
import sys

variant_index = int(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
base_path = pathlib.Path(sys.argv[3])
sweep_root = pathlib.Path(sys.argv[4])
train_steps = int(sys.argv[5])
eval_steps = int(sys.argv[6])
save_checkpoints = str(sys.argv[7]).lower() in {"1", "true", "yes", "on"}
replay_capacity = int(sys.argv[8])
batch_size = int(sys.argv[9])
num_envs = int(sys.argv[10])
rollout_chunk = int(sys.argv[11])
updates_per_chunk = int(sys.argv[12])

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
variants = manifest["variants"]
if variant_index < 0 or variant_index >= len(variants):
    raise SystemExit(f"VARIANT_OUT_OF_RANGE=1\nVARIANT_INDEX={variant_index}")
variant = variants[variant_index]
folder = str(variant["folder"])
output_dir = sweep_root / folder
generated_config = sweep_root / "generated_configs" / f"{folder}.json"

cfg = json.loads(base_path.read_text(encoding="utf-8"))
repo_root = pathlib.Path.cwd()
base_name = cfg.get("name", base_path.stem)
cfg["name"] = f"{base_name}_{folder}"
cfg.setdefault("sim", {})["config_path"] = str(pathlib.Path("/workspace/tokamak-sim/configs/T15MD_new_data.toml"))
cfg["sim"]["csv_initial_state_library"] = str(repo_root / "data/processed/t15_csv_initial_states.npz")
cfg.setdefault("reference", {}).setdefault("ip", {})["limits_path"] = str(repo_root / "data/processed/t15_reference_limits.json")
cfg.setdefault("reward", {}).update(variant["reward"])
cfg.setdefault("training", {})["steps"] = train_steps
cfg["training"]["output_dir"] = str(output_dir)
cfg["training"]["save_checkpoints"] = save_checkpoints
cfg["training"]["checkpoint_interval_steps"] = max(1, train_steps)
cfg["training"]["eval_interval_steps"] = max(1, eval_steps)
cfg["training"]["num_envs"] = num_envs
cfg.setdefault("learner", {})["replay_capacity_episodes"] = replay_capacity
cfg["learner"]["batch_size"] = batch_size
cfg["learner"]["rollout_chunk_length"] = rollout_chunk
cfg["learner"]["updates_per_rollout_chunk"] = updates_per_chunk

output_dir.mkdir(parents=True, exist_ok=True)
generated_config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
(output_dir / "reward_variant.json").write_text(json.dumps({
    "variant_index": variant_index,
    "variant": variant,
    "base_config": str(base_path),
    "generated_config": str(generated_config),
    "output_dir": str(output_dir),
}, indent=2), encoding="utf-8")

print(f"VARIANT_INDEX={variant_index}")
variant_name = variant["name"]
print(f"VARIANT_NAME={variant_name}")
print(f"VARIANT_FOLDER={folder}")
print(f"GENERATED_CONFIG={generated_config}")
print(f"TRAIN_OUTPUT={output_dir}")
print("VARIANT_OUT_OF_RANGE=0")
PYCFG
      source "${RUN_ENV_FILE}"
      rm -f "${RUN_ENV_FILE}"

      if [[ "${VARIANT_OUT_OF_RANGE}" == "1" ]]; then
        echo "variant_index=${VARIANT_INDEX} out of range; skipping"
        continue
      fi

      echo "variant_index=${VARIANT_INDEX}"
      echo "variant_name=${VARIANT_NAME}"
      echo "variant_folder=${VARIANT_FOLDER}"
      echo "generated_config=${GENERATED_CONFIG}"
      echo "train_output=${TRAIN_OUTPUT}"
      echo "run_name=${VARIANT_FOLDER}"

      CHECKPOINT_ARGS=()
      if [[ "${SAVE_CHECKPOINTS}" == "1" || "${SAVE_CHECKPOINTS}" == "true" ]]; then
        CHECKPOINT_ARGS=(--save-checkpoints --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}")
      else
        CHECKPOINT_ARGS=(--no-save-checkpoints)
      fi

      set +e
      python3 scripts/train_policy_pipeline.py \
        --config "${GENERATED_CONFIG}" \
        --output-dir "${TRAIN_OUTPUT}" \
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
        "${CHECKPOINT_ARGS[@]}" \
        --eval-interval-steps "${EVAL_ENV_STEPS}" \
        --eval-episodes 128 \
        --eval-max-steps 500 \
        --controller-rollout-steps 0 \
        --reward-sweep-mode \
        --wandb \
        --wandb-project "${SWEEP_WANDB_PROJECT}" \
        --wandb-name "${VARIANT_FOLDER}" \
        --wandb-group "${SWEEP_WANDB_PROJECT}" \
        --wandb-mode "${WANDB_MODE}"
      RUN_STATUS=$?
      set -e

      echo "variant_index=${VARIANT_INDEX} run_status=${RUN_STATUS}"
      if [[ "${RUN_STATUS}" != "0" ]]; then
        python3 - "${TRAIN_OUTPUT}" "${RUN_STATUS}" <<\PYFAIL
import json
import pathlib
import sys
out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
path = out / "policy_validation.json"
if not path.exists():
    path.write_text(json.dumps({
        "status": "sweep_failed_training",
        "reason": "train_policy_pipeline.py returned nonzero",
        "return_code": int(sys.argv[2]),
        "output_dir": str(out),
    }, indent=2), encoding="utf-8")
PYFAIL
      fi
    done

    echo "reward sweep array task completed at $(date -Is)"
  '
