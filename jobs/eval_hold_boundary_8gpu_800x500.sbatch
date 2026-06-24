#!/bin/bash
#SBATCH --job-name=hold-boundary-eval
#SBATCH --partition=batch
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/hold-boundary-eval-%A_%a.out
#SBATCH --error=slurm_logs/hold-boundary-eval-%A_%a.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

DEFAULT_RUN="t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m_200731"
SOURCE_CONFIG="${SOURCE_CONFIG:-outputs/${DEFAULT_RUN}/generated_configs/${DEFAULT_RUN}.json}"
EXPORT_DIR="${EXPORT_DIR:-outputs/${DEFAULT_RUN}/exports/best_actor}"
EPISODES_PER_SHARD="${EPISODES_PER_SHARD:-100}"
STEPS="${STEPS:-500}"
EXPECTED_SHOT="${EXPECTED_SHOT:-3864}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
EVAL_ROOT="${EVAL_ROOT:-outputs/hold_boundary_eval_${ARRAY_ID}}"
SHARD_DIR="${EVAL_ROOT}/shard_${TASK_ID}"
BASE_INDEX="${BASE_INDEX:-$((TASK_ID * EPISODES_PER_SHARD))}"
SEED="${SEED:-$((386400 + TASK_ID * 1000))}"

export SOURCE_CONFIG EXPORT_DIR EPISODES_PER_SHARD STEPS EXPECTED_SHOT TASK_ID ARRAY_ID EVAL_ROOT SHARD_DIR BASE_INDEX SEED

if [[ ! -f "$SOURCE_CONFIG" ]]; then
  echo "missing SOURCE_CONFIG=$SOURCE_CONFIG" >&2
  exit 2
fi
if [[ ! -d "$EXPORT_DIR" ]]; then
  echo "missing EXPORT_DIR=$EXPORT_DIR" >&2
  exit 2
fi
if [[ ! -f "$EXPORT_DIR/actor.pt" ]]; then
  echo "missing exported actor: $EXPORT_DIR/actor.pt" >&2
  exit 2
fi

mkdir -p "$SHARD_DIR"

echo "array_id=$ARRAY_ID"
echo "task_id=$TASK_ID"
echo "source_config=$SOURCE_CONFIG"
echo "export_dir=$EXPORT_DIR"
echo "eval_root=$EVAL_ROOT"
echo "shard_dir=$SHARD_DIR"
echo "episodes_per_shard=$EPISODES_PER_SHARD"
echo "steps=$STEPS"
echo "base_index=$BASE_INDEX"
echo "seed=$SEED"
echo "expected_shot=$EXPECTED_SHOT"

srun --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
set -euo pipefail
mkdir -p /tmp/nvidia-lib
ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
export MPLCONFIGDIR=/tmp/matplotlib
cd /workspace/tokamak-rl-v2

python3 scripts/evaluate_hold_boundary_task.py \
  --config "/workspace/tokamak-rl-v2/${SOURCE_CONFIG}" \
  --export-dir "/workspace/tokamak-rl-v2/${EXPORT_DIR}" \
  --output-dir "/workspace/tokamak-rl-v2/${SHARD_DIR}" \
  --episodes "${EPISODES_PER_SHARD}" \
  --steps "${STEPS}" \
  --split holdout \
  --expected-shot "${EXPECTED_SHOT}" \
  --seed "${SEED}" \
  --base-index "${BASE_INDEX}" \
  --device cuda:0
'
