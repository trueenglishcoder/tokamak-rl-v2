#!/bin/bash
set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2

export CALIBRATION_OUTPUT=${CALIBRATION_OUTPUT:-outputs/t15_physical_reward_calibration_$(date +%Y%m%d_%H%M%S)}
export MAX_CANDIDATES=${MAX_CANDIDATES:-12}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-500000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-125000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-125000}
export NUM_ENVS=${NUM_ENVS:-32}

array_job=$(sbatch --parsable \
  --export=ALL,CALIBRATION_OUTPUT="${CALIBRATION_OUTPUT}",MAX_CANDIDATES="${MAX_CANDIDATES}",TRAIN_ENV_STEPS="${TRAIN_ENV_STEPS}",EVAL_ENV_STEPS="${EVAL_ENV_STEPS}",CHECKPOINT_ENV_STEPS="${CHECKPOINT_ENV_STEPS}",NUM_ENVS="${NUM_ENVS}" \
  jobs/calibrate_t15_physical_reward_array_1gpu.sbatch)

collector_job=$(sbatch --parsable \
  --dependency=afterany:${array_job} \
  --export=ALL,CALIBRATION_OUTPUT="${CALIBRATION_OUTPUT}",MAX_CANDIDATES="${MAX_CANDIDATES}" \
  jobs/collect_t15_physical_reward_calibration.sbatch)

echo "calibration_output=${CALIBRATION_OUTPUT}"
echo "array_job=${array_job}"
echo "collector_job=${collector_job}"
echo "summary will be written to ${CALIBRATION_OUTPUT}/calibration_summary.csv"
