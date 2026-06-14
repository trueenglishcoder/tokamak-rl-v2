#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-reward-collect
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/tokamak-rl-v2-reward-collect-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-reward-collect-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_hold_reset_boundary_ip_stage1_gpu.yaml}
export CALIBRATION_OUTPUT=${CALIBRATION_OUTPUT:?CALIBRATION_OUTPUT must point to the array output directory}
export MAX_CANDIDATES=${MAX_CANDIDATES:-12}

srun \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    cd /workspace/tokamak-rl-v2
    python3 scripts/calibrate_physical_reward.py \
      --base-config "${POLICY_CONFIG}" \
      --output-root "${CALIBRATION_OUTPUT}" \
      --max-candidates "${MAX_CANDIDATES}" \
      --summary-only
    echo "reward calibration summary collected at $(date -Is)"
  '
