#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-gen500
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-gen-trim50-0p5s-16gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-gen-trim50-0p5s-16gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_generated_trim50_idealized_matched_0p5s_tcvjdot_balanced_mpo.yaml}
export RUN_LABEL=${RUN_LABEL:-t15_generated_trim50_idealized_matched_0p5s_tcvjdot_balanced}

exec bash jobs/train_t15_generated_trim50_idealized_matched_0p1s_tcvjdot_balanced_16gpu_100m.sbatch
