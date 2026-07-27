#!/bin/bash
#SBATCH --job-name=t15-proxy-target-balanced-audit
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/t15-proxy-target-balanced-audit-%j.out
#SBATCH --error=slurm_logs/t15-proxy-target-balanced-audit-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs

export TARGET_OUT_DIR=${TARGET_OUT_DIR:-data/processed/t15_proxy_target_balanced_v1}
export SIM_CONFIG=${SIM_CONFIG:-../tokamak-sim/configs/T15MD.toml}
export AUDIT_OUT_DIR=${AUDIT_OUT_DIR:-data/diagnostics/t15_proxy_target_balanced_v1}

if [[ ! -f "${TARGET_OUT_DIR}/t15_target_trajectory_targets.npz" ]]; then
  echo "missing balanced target-only dataset: ${TARGET_OUT_DIR}/t15_target_trajectory_targets.npz" >&2
  exit 2
fi
if [[ ! -f "${TARGET_OUT_DIR}/t15_proxy_target_v1_initial_states.npz" ]]; then
  echo "missing balanced target-only initial-state library: ${TARGET_OUT_DIR}/t15_proxy_target_v1_initial_states.npz" >&2
  exit 2
fi

srun \
  --ntasks=1 \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace \
  bash -lc '
    set -euo pipefail
    cd /workspace/tokamak-rl-v2
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    python3 scripts/audit_t15_proxy_target_balanced_dataset.py \
      --dataset-dir "${TARGET_OUT_DIR}" \
      --sim-config "${SIM_CONFIG}" \
      --out-dir "${AUDIT_OUT_DIR}"
  '

test -f "${AUDIT_OUT_DIR}/target_feasibility_summary.json"
test -f "${AUDIT_OUT_DIR}/target_feasibility_report.md"
echo "target_dataset_audit=${AUDIT_OUT_DIR}/target_feasibility_summary.json"
