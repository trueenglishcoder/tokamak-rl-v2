#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvd-0p1s-agg
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvd-0p1s-agg-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvd-0p1s-agg-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs

ROOT=${ROOT:?ROOT must point to the sweep output root}
python3 scripts/summarize_tcvdelta_single_segment_0p1s_reward_sweep.py "${ROOT}" --out-dir "${ROOT}/selection"

echo "summary=${ROOT}/selection/reward_search_report.md"
cat "${ROOT}/selection/reward_search_report.md"
