#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-reward-agg1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_logs/tokamak-rl-v2-csv-reward-agg1-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-csv-reward-agg1-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

: "${ROOT:?Set ROOT to outputs/t15_reward_sweep72_legal_1m_<PASS1_JOBID>}"

srun \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    cd /workspace/tokamak-rl-v2

    echo "root=${ROOT}"
    echo "pass1_root=${ROOT}/pass1_broad"
    echo "out_dir=${ROOT}/selection/pass1"
    PROFILE=$(python3 - <<\PYPROFILE
import json
import os
import pathlib

manifest = pathlib.Path(os.environ["ROOT"]) / "pass1_broad/variants.json"
payload = json.loads(manifest.read_text(encoding="utf-8"))
print(payload.get("profile") or "legal")
PYPROFILE
)
    echo "profile=${PROFILE}"

    python3 scripts/summarize_reward_sweep_physical.py \
      "${ROOT}/pass1_broad" \
      --out-dir "${ROOT}/selection/pass1" \
      --top 25

    python3 - <<\PYCENTER
import os
import pathlib
import sys

path = pathlib.Path(os.environ["ROOT"]) / "selection/pass1/physical_best_candidate.json"
sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from build_reward_sweep_manifest import load_center_reward

try:
    center = load_center_reward(path)
except Exception as exc:
    print(f"pass1 aggregation could not extract a focused-sweep center from {path}: {exc}", file=sys.stderr)
    raise SystemExit(3)
print(f"pass1_center_reward={center}")
PYCENTER

    mkdir -p "${ROOT}/pass2_focused/generated_configs"
    if [[ "${PROFILE}" == "saturation" || "${PROFILE}" == "tcv_quality" ]]; then
      PASS2_RUNS_PER_TASK=1
      PASS2_ARRAY_TASKS=12
    else
      PASS2_RUNS_PER_TASK=3
      PASS2_ARRAY_TASKS=12
    fi
    echo "pass2_runs_per_task=${PASS2_RUNS_PER_TASK}"
    echo "pass2_array_tasks=${PASS2_ARRAY_TASKS}"

    python3 scripts/build_reward_sweep_manifest.py \
      --pass focused \
      --profile "${PROFILE}" \
      --center "${ROOT}/selection/pass1/physical_best_candidate.json" \
      --runs-per-array-task "${PASS2_RUNS_PER_TASK}" \
      --array-task-count "${PASS2_ARRAY_TASKS}" \
      --output "${ROOT}/pass2_focused/variants.json"

    python3 scripts/build_reward_sweep_rerun_manifest.py \
      "${ROOT}" \
      --out "${ROOT}/selection/missing_or_failed_variants.json"

    echo "pass1 aggregation completed at $(date -Is)"
  '
