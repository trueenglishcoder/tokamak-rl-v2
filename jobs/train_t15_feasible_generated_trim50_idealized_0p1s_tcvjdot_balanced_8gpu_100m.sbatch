#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-feas8
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-feasible-trim50-8gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-feasible-trim50-8gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_feasible_generated_trim50_idealized_0p1s_tcvjdot_balanced_mpo.yaml}
export BUILDER_SCRIPT=${BUILDER_SCRIPT:-scripts/build_t15_feasible_generated_trim50_idealized_0p1s.py}
export MACHINE_CONFIG_SOURCE=${MACHINE_CONFIG_SOURCE:-../tokamak-sim/runs/t15md_limited_replay_dataset_trim50_idealized_matched_gpu_plain_1e6/T15MD_new_data_legacy_contour_limited_replay.toml}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_feasible_generated_trim50_idealized_0p1s_initial_states.npz}
export TARGET_LIBRARY=${TARGET_LIBRARY:-data/processed/t15_feasible_generated_trim50_idealized_0p1s/t15_feasible_generated_trim50_idealized_0p1s_targets.npz}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-100000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export WORLD_SIZE=${WORLD_SIZE:-8}
export LOCAL_ENVS_PER_RANK=${LOCAL_ENVS_PER_RANK:-256}
export NUM_ENVS=${NUM_ENVS:-$(( WORLD_SIZE * LOCAL_ENVS_PER_RANK ))}
export GLOBAL_LEARNER_BATCH=${GLOBAL_LEARNER_BATCH:-256}
export BATCH_SIZE=${BATCH_SIZE:-$(( GLOBAL_LEARNER_BATCH / WORLD_SIZE ))}
export UPDATES_PER_ROLLOUT_CHUNK=${UPDATES_PER_ROLLOUT_CHUNK:-64}
export RUN_LABEL=${RUN_LABEL:-t15_feasible_generated_trim50_idealized_0p1s_tcvjdot_balanced}
export RUN_NAME=${RUN_NAME:-${RUN_LABEL}_8gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-feasible-generated-trim50-8gpu-100m-${SLURM_JOB_ID:-manual}}
export WANDB_MODE=${WANDB_MODE:-online}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$(( 20000 + (${SLURM_JOB_ID:-1731} % 40000) ))}

for path in "${SOURCE_CONFIG}" "${BUILDER_SCRIPT}" "${MACHINE_CONFIG_SOURCE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required feasible-generated training artifact: ${path}" >&2
    exit 2
  fi
done
if (( NUM_ENVS % WORLD_SIZE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi
if (( BATCH_SIZE * WORLD_SIZE != GLOBAL_LEARNER_BATCH )); then
  echo "BATCH_SIZE=${BATCH_SIZE} and WORLD_SIZE=${WORLD_SIZE} do not preserve global learner batch ${GLOBAL_LEARNER_BATCH}" >&2
  exit 2
fi

if [[ ! -f "${RESET_LIBRARY}" || ! -f "${TARGET_LIBRARY}" ]]; then
  echo "feasible generated reset/target libraries are missing; building them inside the container"
  srun --ntasks=1 \
    --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
    --container-mounts /scratch/$USER/tokamak:/workspace \
    bash -lc '
      set -euo pipefail
      export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
      cd /workspace/tokamak-rl-v2
      python3 "${BUILDER_SCRIPT}" \
        --initial-states-out "${RESET_LIBRARY}" \
        --targets-out "${TARGET_LIBRARY}" \
        --out-dir "$(dirname "${TARGET_LIBRARY}")"
    '
fi

for path in "${RESET_LIBRARY}" "${TARGET_LIBRARY}"; do
  if [[ ! -f "${path}" ]]; then
    echo "failed to create feasible-generated training artifact: ${path}" >&2
    exit 2
  fi
done

if [[ -z "${HOLDOUT_EVAL_EPISODES:-}" ]]; then
  HOLDOUT_EVAL_EPISODES=$(
    srun --ntasks=1 \
      --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
      --container-mounts /scratch/$USER/tokamak:/workspace \
      bash -lc '
        set -euo pipefail
        export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
        cd /workspace/tokamak-rl-v2
        python3 scripts/count_npz_string_value.py "${RESET_LIBRARY}" --array split --value holdout
      '
  )
  export HOLDOUT_EVAL_EPISODES
fi
if [[ ! "${HOLDOUT_EVAL_EPISODES}" =~ ^[0-9]+$ ]] || (( HOLDOUT_EVAL_EPISODES <= 0 )); then
  echo "invalid HOLDOUT_EVAL_EPISODES=${HOLDOUT_EVAL_EPISODES}; expected positive holdout count" >&2
  exit 2
fi

mkdir -p "$(dirname "${GENERATED_CONFIG}")"
cp "${MACHINE_CONFIG_SOURCE}" "$(dirname "${GENERATED_CONFIG}")/T15MD_new_data_trim50_idealized_gpu_plain_1e6.toml"

python3 - <<PYCFG
import json
from pathlib import Path

source = Path("${SOURCE_CONFIG}")
out = Path("${GENERATED_CONFIG}")
cfg = json.loads(source.read_text(encoding="utf-8"))
cfg["name"] = "${RUN_LABEL}_8gpu_100m"
cfg["sim"]["config_path"] = "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/generated_configs/T15MD_new_data_trim50_idealized_gpu_plain_1e6.toml"
cfg["sim"]["csv_initial_state_library"] = "/workspace/tokamak-rl-v2/${RESET_LIBRARY}"
cfg["reference"]["ip"]["feasible_reference_dir"] = "/workspace/tokamak-rl-v2/$(dirname "${TARGET_LIBRARY}")"
cfg["reference"]["boundary"]["feasible_reference_dir"] = "/workspace/tokamak-rl-v2/$(dirname "${TARGET_LIBRARY}")"
cfg["training"]["steps"] = int("${TRAIN_ENV_STEPS}")
cfg["training"]["num_envs"] = int("${NUM_ENVS}")
cfg["training"]["output_dir"] = "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}"
cfg["training"]["eval_interval_steps"] = int("${EVAL_ENV_STEPS}")
cfg["training"]["checkpoint_interval_steps"] = int("${CHECKPOINT_ENV_STEPS}")
cfg["training"]["milestone_checkpoint_interval_steps"] = int("${CHECKPOINT_ENV_STEPS}")
cfg["training"]["eval_checkpoint_top_k"] = 10
cfg["training"]["keep_latest_checkpoint"] = True
cfg["training"]["distributed_mode"] = "local_replay"
cfg["training"]["production_mode"] = True
cfg["training"]["eval_max_steps"] = int(cfg["sim"]["max_episode_steps"])
cfg["training"]["save_checkpoints"] = True
cfg["training"]["early_stop_patience_evals"] = 0
cfg["learner"]["batch_size"] = int("${BATCH_SIZE}")
cfg["learner"]["updates_per_rollout_chunk"] = int("${UPDATES_PER_ROLLOUT_CHUNK}")

checks = {
    "feasible_ip": cfg["reference"]["ip"]["kind"] == "feasible_generated_window",
    "feasible_boundary": cfg["reference"]["boundary"]["kind"] == "feasible_generated_window",
    "same_target_dir": cfg["reference"]["ip"]["feasible_reference_dir"] == cfg["reference"]["boundary"]["feasible_reference_dir"],
    "jdot_command": cfg["sim"]["action_contract"] == "jdot_command",
    "steps_100": int(cfg["sim"]["max_episode_steps"]) == 100,
    "actor_obs_v6": cfg["observation"]["actor_kind"] == "controller_state_v6",
    "critic_obs_v2": cfg["observation"]["critic_kind"] == "compact_training_state_v2",
    "reward_tcv": cfg["reward"]["kind"] == "tcv_derivative",
    "local_replay": cfg["training"]["distributed_mode"] == "local_replay",
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"feasible generated training config failed invariant checks: {failed}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(out)
PYCFG

export EPISODE_STEPS=100

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_name=${RUN_NAME}"
echo "train_output=${TRAIN_OUTPUT}"
echo "source_config=${SOURCE_CONFIG}"
echo "builder_script=${BUILDER_SCRIPT}"
echo "generated_config=${GENERATED_CONFIG}"
echo "target_library=${TARGET_LIBRARY}"
echo "wandb_project=${WANDB_PROJECT_NAME}"
echo "train_env_steps=${TRAIN_ENV_STEPS}"
echo "num_envs=${NUM_ENVS}"
echo "world_size=${WORLD_SIZE}"
echo "local_envs_per_rank=$(( NUM_ENVS / WORLD_SIZE ))"
echo "batch_size_per_rank=${BATCH_SIZE}"
echo "updates_per_rollout_chunk=${UPDATES_PER_ROLLOUT_CHUNK}"
echo "holdout_eval_episodes=${HOLDOUT_EVAL_EPISODES}"

srun --container-image /scratch/$USER/tokamak/tokamak-rl-v2.sqsh \
  --container-mounts /scratch/$USER/tokamak:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs \
  bash -lc '
    set -euo pipefail
    mkdir -p /tmp/nvidia-lib
    ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
    ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
    ln -sf /host-libs/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so.1
    ln -sf /tmp/nvidia-lib/libnvidia-ml.so.1 /tmp/nvidia-lib/libnvidia-ml.so
    export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    cd /workspace/tokamak-rl-v2

    python3 - <<PYCHECK
import numpy as np
import torch
from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv

cfg = load_experiment_config("/workspace/tokamak-rl-v2/${GENERATED_CONFIG}")
with np.load("/workspace/tokamak-rl-v2/${RESET_LIBRARY}", allow_pickle=False) as d:
    if "params0" not in d.files or "difficulty_bin" not in d.files:
        raise SystemExit("feasible reset library must include params0 and difficulty_bin")
    print("reset rows", int(d["ip0"].shape[0]), "holdout", int(np.count_nonzero(d["split"].astype(str) == "holdout")))
with np.load("/workspace/tokamak-rl-v2/${TARGET_LIBRARY}", allow_pickle=False) as d:
    print("target rows", int(d["ip_ref"].shape[0]), "zones", sorted(set(d["zone"].astype(str).tolist())))
env = TokamakMagneticControlEnv(cfg, batch_size=2, device="cuda:0", seed=int(cfg.training.seed))
env.set_curriculum_progress(0.75)
obs = env.reset()
print("preflight env_reset_ok", tuple(obs.shape), tuple(env.reference.ip.shape), tuple(env.reference.radii.shape), "action_dim", env.action_dim)
del env, obs
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PYCHECK

    python3 -m torch.distributed.run \
      --nnodes 1 \
      --nproc_per_node "${GPUS_PER_NODE}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      scripts/train_policy_pipeline.py \
        --config "/workspace/tokamak-rl-v2/${GENERATED_CONFIG}" \
        --output-dir "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}" \
        --steps "${TRAIN_ENV_STEPS}" \
        --device cuda:0 \
        --sim-compute-backend gpu \
        --sim-gpu-device cuda \
        --num-envs "${NUM_ENVS}" \
        --distributed-mode local_replay \
        --actor-workers 1 \
        --batch-size "${BATCH_SIZE}" \
        --unroll-length 100 \
        --min-replay-sequence-length 100 \
        --replay-capacity-episodes 1024 \
        --rollout-chunk-length 100 \
        --updates-per-rollout-chunk "${UPDATES_PER_ROLLOUT_CHUNK}" \
        --hidden-dim 256 \
        --critic-hidden-dim 256 \
        --critic-mlp-hidden-dim 256 \
        --action-samples 64 \
        --actor-update-chunk-size 2048 \
        --save-checkpoints \
        --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}" \
        --eval-interval-steps "${EVAL_ENV_STEPS}" \
        --eval-episodes "${HOLDOUT_EVAL_EPISODES}" \
        --eval-max-steps "${EPISODE_STEPS}" \
        --controller-rollout-steps 0 \
        --wandb \
        --wandb-project "${WANDB_PROJECT_NAME}" \
        --wandb-name "${RUN_NAME}" \
        --wandb-group "${RUN_NAME}" \
        --wandb-mode "${WANDB_MODE}" \
        --wandb-optional \
        --wandb-metric-preset focused
  '
