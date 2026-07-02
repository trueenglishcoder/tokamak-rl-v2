#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-pert8
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-perturbed-matched-trim50-8gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-perturbed-matched-trim50-8gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_replay_window_perturbed_trim50_idealized_matched_0p1s_tcvjdot_balanced_mpo.yaml}
export BUILDER_SCRIPT=${BUILDER_SCRIPT:-scripts/build_t15_replay_window_perturbed_trim50_idealized_0p1s.py}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_replay_window_perturbed_trim50_idealized_matched_0p1s_initial_states.npz}
export TARGET_LIBRARY=${TARGET_LIBRARY:-data/processed/t15_replay_window_perturbed_trim50_idealized_matched_0p1s/t15_feasible_generated_trim50_idealized_0p1s_targets.npz}
export RUN_LABEL=${RUN_LABEL:-t15_replay_window_perturbed_trim50_idealized_matched_0p1s_tcvjdot_balanced}
export TRAINING_FAMILY_TOKEN=${TRAINING_FAMILY_TOKEN:-replay_window_perturbed_trim50_idealized_matched}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-perturbed-matched-window-trim50-8gpu-100m-${SLURM_JOB_ID:-manual}}

bash jobs/train_t15_simple_manifold_generated_trim50_idealized_matched_0p1s_tcvjdot_balanced_8gpu_100m.sbatch
