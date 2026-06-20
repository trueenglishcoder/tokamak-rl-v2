#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-reward-agg-final
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_logs/tokamak-rl-v2-csv-reward-final-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-csv-reward-final-%j.err

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
    echo "out_dir=${ROOT}/selection"

    python3 scripts/summarize_two_pass_reward_sweep_physical.py \
      "${ROOT}" \
      --out-dir "${ROOT}/selection"

    python3 scripts/build_reward_sweep_rerun_manifest.py \
      "${ROOT}" \
      --out "${ROOT}/selection/missing_or_failed_variants.json"

    cat "${ROOT}/selection/final_reward_selection_report.md"
    cat "${ROOT}/selection/final_reward_recommendation.json"

    echo "final reward selection completed at $(date -Is)"
  '
