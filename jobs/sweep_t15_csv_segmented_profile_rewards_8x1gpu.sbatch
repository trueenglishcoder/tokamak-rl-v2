#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-csv-reward-sweep
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --array=0-7%8
#SBATCH --output=slurm_logs/tokamak-rl-v2-csv-reward-sweep-1gpu-%A_%a.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-csv-reward-sweep-1gpu-%A_%a.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml}
export REWARD_VARIANTS=${REWARD_VARIANTS:-configs/reward_sweeps/t15_csv_segmented_profile_reward_8x1m.json}
export VARIANT_INDEX=${SLURM_ARRAY_TASK_ID:-0}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-1000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-100000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-500000}
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
export NUM_ENVS=${NUM_ENVS:-256}
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-512}
export BATCH_SIZE=${BATCH_SIZE:-256}
export ROLLOUT_CHUNK_LENGTH=${ROLLOUT_CHUNK_LENGTH:-64}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-32}
export SWEEP_RUN_PREFIX=${SWEEP_RUN_PREFIX:-t15_csv_reward}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}

srun \
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
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}

    cd /workspace/tokamak-rl-v2
    mkdir -p slurm_logs outputs configs/experiments data/processed

    if [[ ! -f data/processed/t15_reference_limits.json || ! -f data/processed/t15_csv_initial_states.npz ]]; then
      echo "missing processed CSV production artifacts; run the normal production preparation once before the reward sweep" >&2
      exit 2
    fi

    python3 - <<\PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYCUDA

    GENERATED_CONFIG=$(python3 - <<\PYCFG
import json
import os
import pathlib
import re

base_path = pathlib.Path(os.environ["POLICY_CONFIG"])
manifest_path = pathlib.Path(os.environ["REWARD_VARIANTS"])
variant_index = int(os.environ["VARIANT_INDEX"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
variants = manifest["variants"]
if variant_index < 0 or variant_index >= len(variants):
    raise SystemExit(f"variant index {variant_index} out of range for {len(variants)} variants")
variant = variants[variant_index]
variant_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(variant["name"]))

array_job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "manual"
array_task = os.environ.get("SLURM_ARRAY_TASK_ID", str(variant_index))
run_prefix = os.environ.get("SWEEP_RUN_PREFIX", "t15_csv_reward")
run_name = f"{run_prefix}_{variant_index}_{variant_name}_1m_{array_job}_{array_task}"
run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)

cfg = json.loads(base_path.read_text(encoding="utf-8"))
cfg["name"] = str(cfg.get("name", base_path.stem)) + "_" + variant_name
cfg.setdefault("reward", {}).update(variant["reward"])

generated_path = pathlib.Path("configs/experiments") / f".generated_{run_name}.json"
generated_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

output_dir = pathlib.Path(os.environ.get("TRAIN_OUTPUT") or f"outputs/{run_name}")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "reward_variant.json").write_text(json.dumps({
    "variant_index": variant_index,
    "variant": variant,
    "base_config": str(base_path),
    "generated_config": str(generated_path),
    "run_name": run_name,
}, indent=2), encoding="utf-8")

print(generated_path)
PYCFG
)

    VARIANT_NAME=$(python3 - <<\PYVAR
import json, os
variant=json.loads(open(os.environ["REWARD_VARIANTS"], encoding="utf-8").read())["variants"][int(os.environ["VARIANT_INDEX"])]
print(variant["name"])
PYVAR
)

    ARRAY_JOB=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
    ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-${VARIANT_INDEX}}
    RUN_NAME=${SWEEP_RUN_PREFIX}_${VARIANT_INDEX}_${VARIANT_NAME}_1m_${ARRAY_JOB}_${ARRAY_TASK}
    TRAIN_OUTPUT=outputs/${RUN_NAME}
    WANDB_PROJECT_NAME=${SWEEP_WANDB_PROJECT_NAME:-tokamak-rl-v2-reward-direction-8x1m-${ARRAY_JOB}}

    echo "policy_config=${GENERATED_CONFIG}"
    echo "base_policy_config=${POLICY_CONFIG}"
    echo "reward_variants=${REWARD_VARIANTS}"
    echo "variant_index=${VARIANT_INDEX}"
    echo "variant_name=${VARIANT_NAME}"
    echo "train_output=${TRAIN_OUTPUT}"
    echo "run_name=${RUN_NAME}"
    echo "wandb_project=${WANDB_PROJECT_NAME}"
    echo "train_env_steps=${TRAIN_ENV_STEPS}"
    echo "num_envs=${NUM_ENVS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "replay_capacity_episodes=${REPLAY_CAPACITY_EPISODES}"
    echo "rollout_chunk_length=${ROLLOUT_CHUNK_LENGTH}"
    echo "updates_per_rollout_chunk=${UPDATES_PER_ROLLOUT_CHUNK}"
    echo "eval_env_steps=${EVAL_ENV_STEPS}"
    echo "save_checkpoints=${SAVE_CHECKPOINTS}"
    echo "checkpoint_env_steps=${CHECKPOINT_ENV_STEPS}"

    CHECKPOINT_ARGS=()
    if [[ "${SAVE_CHECKPOINTS}" == "1" || "${SAVE_CHECKPOINTS}" == "true" ]]; then
      CHECKPOINT_ARGS=(--save-checkpoints --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}")
    else
      CHECKPOINT_ARGS=(--no-save-checkpoints)
    fi

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
      --wandb \
      --wandb-project "${WANDB_PROJECT_NAME}" \
      --wandb-name "${RUN_NAME}" \
      --wandb-group "${WANDB_PROJECT_NAME}" \
      --wandb-mode "${WANDB_MODE}"

    echo "reward sweep task completed at $(date -Is)"
  '
