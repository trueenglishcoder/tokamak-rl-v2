#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-reward36-sat-agg
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/tokamak-rl-v2-csv-reward36-saturation-aggregate-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-csv-reward36-saturation-aggregate-%j.err

set -euo pipefail

: "${ROOT:?Set ROOT to the one-pass reward sweep root}"

cd /scratch/$USER/tokamak

srun \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    cd /workspace/tokamak-rl-v2

    ROOT="${ROOT}"
    OUT_DIR="${ROOT}/selection"
    mkdir -p "${OUT_DIR}"

    echo "root=${ROOT}"
    echo "out_dir=${OUT_DIR}"

    python3 scripts/summarize_reward_sweep_physical.py \
      "${ROOT}" \
      --out-dir "${OUT_DIR}" \
      --top 25

    echo
    echo "===== physical_selection_report.md ====="
    cat "${OUT_DIR}/physical_selection_report.md"

    echo
    echo "===== physical_best_candidate.json ====="
    cat "${OUT_DIR}/physical_best_candidate.json"
  '
