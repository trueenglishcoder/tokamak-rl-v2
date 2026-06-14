#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-hinf-bootstrap
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/tokamak-rl-v2-hinf-bootstrap-1gpu-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-hinf-bootstrap-1gpu-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak

export POLICY_CONFIG=${POLICY_CONFIG:-configs/experiments/t15_shot_fragment_boundary_ip_500_gpu.yaml}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/t15_hinf_bootstrap_policy_1gpu_${SLURM_JOB_ID:-manual}}
export RUN_NAME=${RUN_NAME:-t15_hinf_bootstrap_policy_1gpu_${SLURM_JOB_ID:-manual}}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-hinf-bootstrap-1gpu-${SLURM_JOB_ID:-manual}}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-30}

export NUM_ENVS=${NUM_ENVS:-32}
export BOOTSTRAP_DECISION_STEPS=${BOOTSTRAP_DECISION_STEPS:-500}
export BOOTSTRAP_RESET_INTERVAL_STEPS=${BOOTSTRAP_RESET_INTERVAL_STEPS:-250}
export BOOTSTRAP_LOG_INTERVAL=${BOOTSTRAP_LOG_INTERVAL:-10}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-1000000}
export TRAIN_DECISION_STEPS=$(( (TRAIN_ENV_STEPS + NUM_ENVS - 1) / NUM_ENVS ))
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-125000}
export EVAL_DECISION_STEPS=$(( (EVAL_ENV_STEPS + NUM_ENVS - 1) / NUM_ENVS ))
export SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-0}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-250000}
export CHECKPOINT_DECISION_STEPS=$(( (CHECKPOINT_ENV_STEPS + NUM_ENVS - 1) / NUM_ENVS ))
export REPLAY_CAPACITY_EPISODES=${REPLAY_CAPACITY_EPISODES:-512}

export HINF_Q_ERROR=${HINF_Q_ERROR:-3000000.0}
export HINF_Q_IP=${HINF_Q_IP:-10000000.0}
export HINF_GAMMA=${HINF_GAMMA:-7905.694150420949}
export HINF_U_CLIP=${HINF_U_CLIP:-2000000.0}
export HINF_J_CURR=${HINF_J_CURR:-0.0}

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
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}

    cd /workspace/tokamak-rl-v2
    mkdir -p slurm_logs outputs

    python3 - <<PYCUDA
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PYCUDA

    echo "policy_config=${POLICY_CONFIG}"
    echo "train_output=${TRAIN_OUTPUT}"
    echo "num_envs=${NUM_ENVS}"
    echo "bootstrap_decision_steps=${BOOTSTRAP_DECISION_STEPS}"
    echo "bootstrap_reset_interval_steps=${BOOTSTRAP_RESET_INTERVAL_STEPS}"
    echo "bootstrap_log_interval=${BOOTSTRAP_LOG_INTERVAL}"
    echo "train_env_steps=${TRAIN_ENV_STEPS}"
    echo "train_decision_steps=${TRAIN_DECISION_STEPS}"
    echo "eval_env_steps=${EVAL_ENV_STEPS}"
    echo "eval_decision_steps=${EVAL_DECISION_STEPS}"
    echo "save_checkpoints=${SAVE_CHECKPOINTS}"
    echo "checkpoint_env_steps=${CHECKPOINT_ENV_STEPS}"
    echo "checkpoint_decision_steps=${CHECKPOINT_DECISION_STEPS}"
    echo "hinf_q_error=${HINF_Q_ERROR}"
    echo "hinf_q_ip=${HINF_Q_IP}"
    echo "hinf_gamma=${HINF_GAMMA}"
    echo "hinf_u_clip=${HINF_U_CLIP}"

    CHECKPOINT_ARGS=()
    if [[ "${SAVE_CHECKPOINTS}" == "1" || "${SAVE_CHECKPOINTS}" == "true" ]]; then
      CHECKPOINT_ARGS=(--save-checkpoints --checkpoint-interval-steps "${CHECKPOINT_DECISION_STEPS}")
    fi

    python3 scripts/train_hinf_bootstrap_pipeline.py \
      --config ${POLICY_CONFIG} \
      --output-dir ${TRAIN_OUTPUT} \
      --steps ${TRAIN_DECISION_STEPS} \
      --device cuda:0 \
      --sim-compute-backend gpu \
      --sim-gpu-device cuda:0 \
      --num-envs ${NUM_ENVS} \
      --actor-workers 1 \
      --batch-size 256 \
      --unroll-length 128 \
      --replay-capacity-episodes ${REPLAY_CAPACITY_EPISODES} \
      --rollout-chunk-length 128 \
      --updates-per-rollout-chunk 32 \
      --hidden-dim 256 \
      --critic-hidden-dim 256 \
      --critic-mlp-hidden-dim 256 \
      --actor-initial-std 0.12 \
      --action-samples 20 \
      --actor-update-chunk-size 2048 \
      "${CHECKPOINT_ARGS[@]}" \
      --eval-interval-steps ${EVAL_DECISION_STEPS} \
      --bootstrap-steps ${BOOTSTRAP_DECISION_STEPS} \
      --bootstrap-reset-interval-steps ${BOOTSTRAP_RESET_INTERVAL_STEPS} \
      --bootstrap-log-interval ${BOOTSTRAP_LOG_INTERVAL} \
      --bootstrap-lr 0.0001 \
      --bootstrap-target-std 0.08 \
      --hinf-q-error ${HINF_Q_ERROR} \
      --hinf-q-ip ${HINF_Q_IP} \
      --hinf-gamma ${HINF_GAMMA} \
      --hinf-u-clip ${HINF_U_CLIP} \
      --hinf-j-curr ${HINF_J_CURR} \
      --wandb \
      --wandb-project ${WANDB_PROJECT_NAME} \
      --wandb-name ${RUN_NAME} \
      --wandb-group ${RUN_NAME} \
      --wandb-mode ${WANDB_MODE} \
      --controller-rollout-steps 500 \
      --allow-failed-gates

    echo "train_hinf_bootstrap_pipeline completed at $(date -Is)"
  '
