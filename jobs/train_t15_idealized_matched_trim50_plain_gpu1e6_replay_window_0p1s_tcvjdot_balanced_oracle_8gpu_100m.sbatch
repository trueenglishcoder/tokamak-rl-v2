#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-ideal-oracle
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-idealized-oracle-8gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-idealized-oracle-8gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2

# This is deliberately the same oracle-window training path as the working
# trim50 run. Only the CSV data root and output names change.
export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml}
export TRIM_MACHINE_CONFIG_SOURCE=${TRIM_MACHINE_CONFIG_SOURCE:-../tokamak-sim/configs/T15MD_new_data.toml}
export TRIM_DATA_ROOT=${TRIM_DATA_ROOT:-../tokamak-sim/data/t15_data_new_trim50_idealized_matched}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_idealized_matched_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz}
export ORACLE_TARGET_DIR=${ORACLE_TARGET_DIR:-data/processed/t15_idealized_matched_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets}
export ORACLE_TARGETS=${ORACLE_TARGETS:-${ORACLE_TARGET_DIR}/t15_replay_window_oracle_targets.npz}
export ORACLE_MACHINE_CONFIG=${ORACLE_MACHINE_CONFIG:-data/processed/t15_idealized_matched_trim50_plain_gpu1e6_machine_config.toml}
export ORACLE_CONFIG_OUT=${ORACLE_CONFIG_OUT:-data/processed/t15_idealized_matched_trim50_plain_gpu1e6_replay_window_0p1s_oracle_config.yaml}
export RUN_NAME=${RUN_NAME:-t15_idealized_matched_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-idealized-matched-oracle-8gpu-100m-${SLURM_JOB_ID:-manual}}

if [[ ! -d "${TRIM_DATA_ROOT}/ip" || ! -d "${TRIM_DATA_ROOT}/coils" ]]; then
  echo "missing idealized matched trim50 data root: ${TRIM_DATA_ROOT}" >&2
  echo "build it in tokamak-sim with scripts/build_t15_trim50_idealized_matched_pipeline.sh first" >&2
  exit 2
fi

bash jobs/train_t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch
