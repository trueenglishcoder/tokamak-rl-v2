#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-reward288-agg1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_logs/tokamak-rl-v2-csv-reward288-agg1-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-csv-reward288-agg1-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

: "${ROOT:?Set ROOT to outputs/t15_reward_sweep288_legal_1m_<PASS1_JOBID>}"

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

    python3 scripts/summarize_reward_sweep_physical.py \
      "${ROOT}/pass1_broad" \
      --out-dir "${ROOT}/selection/pass1" \
      --top 25

    python3 - <<\PYCENTER
import json
import os
import pathlib
import sys

path = pathlib.Path(os.environ["ROOT"]) / "selection/pass1/physical_best_candidate.json"
payload = json.loads(path.read_text(encoding="utf-8"))
best = payload.get("best_candidate")
best = best if isinstance(best, dict) else {}
required = ["shape_mean_weight", "shape_max_weight", "ip_weight", "current_weight", "derivative_weight"]
missing = [key for key in required if key not in best]
if missing:
    print(f"pass1 produced no usable focused-sweep center; missing reward fields: {missing}", file=sys.stderr)
    raise SystemExit(3)
PYCENTER

    python3 scripts/build_reward_sweep_rerun_manifest.py \
      "${ROOT}" \
      --out "${ROOT}/selection/missing_or_failed_variants.json"

    echo "pass1 aggregation completed at $(date -Is)"
  '
