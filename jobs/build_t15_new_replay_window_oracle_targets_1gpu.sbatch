#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-build-rw-oracle
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/tokamak-rl-v2-build-rw-oracle-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-build-rw-oracle-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs data/processed

srun \
  --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
    set -euo pipefail
    mkdir -p /tmp/nvidia-lib
    ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
    ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
    ln -sf /host-libs/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so.1
    ln -sf /tmp/nvidia-lib/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so
    export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    cd /workspace/tokamak-rl-v2

    python3 scripts/build_t15_new_replay_window_oracle_targets.py \
      --compute-backend gpu \
      --gpu-device cuda:0 \
      --train-shots 3856 3857 3858 3863 \
      --holdout-shots 3864
  '
