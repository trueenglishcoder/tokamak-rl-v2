#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-gen-trim50
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-gen-trim50-8gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-gen-trim50-8gpu-100m-%j.err

set -euo pipefail

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_generated_trim50_idealized_0p1s_tcvjdot_balanced_mpo.yaml}
export GENERATED_MACHINE_CONFIG_SOURCE=${GENERATED_MACHINE_CONFIG_SOURCE:-../tokamak-sim/runs/t15md_limited_replay_dataset_trim50_idealized_low_tau_gpu_plain_1e6_video/T15MD_new_data_legacy_contour_limited_replay.toml}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_generated_trim50_idealized_initial_states.npz}
export ENVELOPE_JSON=${ENVELOPE_JSON:-data/processed/t15_generated_trim50_idealized_envelope.json}
export RUN_NAME=${RUN_NAME:-t15_generated_trim50_idealized_0p1s_tcvjdot_balanced_8gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-generated-trim50-8gpu-100m-${SLURM_JOB_ID:-manual}}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-100000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}
export NUM_ENVS=${NUM_ENVS:-2048}
export GPUS_PER_NODE=8
export WORLD_SIZE=8
export WANDB_MODE=${WANDB_MODE:-online}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$(( 20000 + (${SLURM_JOB_ID:-1731} % 40000) ))}

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "missing generated training config: ${SOURCE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${GENERATED_MACHINE_CONFIG_SOURCE}" ]]; then
  echo "missing plain GPU 1e-6 machine config: ${GENERATED_MACHINE_CONFIG_SOURCE}" >&2
  exit 2
fi
if [[ ! -f "${RESET_LIBRARY}" ]]; then
  echo "missing generated reset library: ${RESET_LIBRARY}" >&2
  echo "run scripts/build_t15_generated_trim50_idealized_references.py first" >&2
  exit 2
fi
if [[ ! -f "${ENVELOPE_JSON}" ]]; then
  echo "missing generated envelope: ${ENVELOPE_JSON}" >&2
  echo "run scripts/build_t15_generated_trim50_idealized_references.py first" >&2
  exit 2
fi
if (( NUM_ENVS % WORLD_SIZE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi

if [[ -z "${HOLDOUT_EVAL_EPISODES:-}" ]]; then
  HOLDOUT_EVAL_EPISODES=$(python3 scripts/count_npz_string_value.py "${RESET_LIBRARY}" --array split --value holdout)
  export HOLDOUT_EVAL_EPISODES
fi
if [[ ! "${HOLDOUT_EVAL_EPISODES}" =~ ^[0-9]+$ ]] || (( HOLDOUT_EVAL_EPISODES <= 0 )); then
  echo "invalid HOLDOUT_EVAL_EPISODES=${HOLDOUT_EVAL_EPISODES}; expected a positive holdout count from ${RESET_LIBRARY}" >&2
  exit 2
fi

mkdir -p "$(dirname "${GENERATED_CONFIG}")"
cp "${GENERATED_MACHINE_CONFIG_SOURCE}" "$(dirname "${GENERATED_CONFIG}")/T15MD_new_data_trim50_idealized_gpu_plain_1e6.toml"
python3 - <<PYCFG
import json
from pathlib import Path

source = Path("${SOURCE_CONFIG}")
out = Path("${GENERATED_CONFIG}")
cfg = json.loads(source.read_text(encoding="utf-8"))

cfg["name"] = "t15_generated_trim50_idealized_0p1s_tcvjdot_balanced_8gpu_100m"
cfg["sim"]["config_path"] = "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/generated_configs/T15MD_new_data_trim50_idealized_gpu_plain_1e6.toml"
cfg["sim"]["csv_initial_state_library"] = "/workspace/tokamak-rl-v2/${RESET_LIBRARY}"
cfg["reference"]["ip"]["limits_path"] = "/workspace/tokamak-rl-v2/${ENVELOPE_JSON}"
cfg["reference"]["boundary"]["envelope_path"] = "/workspace/tokamak-rl-v2/${ENVELOPE_JSON}"
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
cfg["training"]["eval_max_steps"] = 100
cfg["training"]["save_checkpoints"] = True
cfg["training"]["early_stop_patience_evals"] = 0
cfg["training"]["early_stop_min_delta"] = 0.0

checks = {
    "generated_config_copy": cfg["sim"]["config_path"].endswith("/generated_configs/T15MD_new_data_trim50_idealized_gpu_plain_1e6.toml"),
    "generated_reset_library": "t15_generated_trim50_idealized_initial_states" in str(cfg["sim"]["csv_initial_state_library"]),
    "generated_envelope_ip": "t15_generated_trim50_idealized_envelope" in str(cfg["reference"]["ip"]["limits_path"]),
    "generated_envelope_boundary": "t15_generated_trim50_idealized_envelope" in str(cfg["reference"]["boundary"]["envelope_path"]),
    "generated_ip": cfg["reference"]["ip"]["kind"] == "generated_segment_profile",
    "generated_boundary": cfg["reference"]["boundary"]["kind"] == "generated_parameter_profile",
    "jdot_command": cfg["sim"]["action_contract"] == "jdot_command",
    "episode_steps_100": int(cfg["sim"]["max_episode_steps"]) == 100,
    "duration_0p1": abs(float(cfg["reference"]["duration_s"]) - 0.1) < 1e-12,
    "actor_obs_v6": cfg["observation"]["actor_kind"] == "controller_state_v6",
    "critic_obs_v2": cfg["observation"]["critic_kind"] == "compact_training_state_v2",
    "reward_tcv": cfg["reward"]["kind"] == "tcv_derivative",
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"generated training config failed invariant checks: {failed}")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(out)
PYCFG

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_name=${RUN_NAME}"
echo "train_output=${TRAIN_OUTPUT}"
echo "source_config=${SOURCE_CONFIG}"
echo "generated_config=${GENERATED_CONFIG}"
echo "wandb_project=${WANDB_PROJECT_NAME}"
echo "train_env_steps=${TRAIN_ENV_STEPS}"
echo "num_envs=${NUM_ENVS}"
echo "world_size=${WORLD_SIZE}"
echo "local_envs_per_rank=$(( NUM_ENVS / WORLD_SIZE ))"
echo "holdout_eval_episodes=${HOLDOUT_EVAL_EPISODES}"

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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    cd /workspace/tokamak-rl-v2

    python3 - <<PYCHECK
from pathlib import Path
from dataclasses import replace
import math
import numpy as np
import torch
from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv

cfg = load_experiment_config("/workspace/tokamak-rl-v2/${GENERATED_CONFIG}")
print("preflight config_path", cfg.sim.config_path)
print("preflight reset_library", cfg.sim.csv_initial_state_library)
print("preflight generated_envelope", cfg.reference.ip.limits_path, cfg.reference.boundary.envelope_path)
with np.load(cfg.sim.csv_initial_state_library, allow_pickle=False) as d:
    if "params0" not in d.files:
        raise SystemExit("generated reset library is missing params0")
    print("preflight rows", int(d["ip0"].shape[0]), "holdout", int(np.count_nonzero(d["split"].astype(str) == "holdout")))
required = [cfg.sim.config_path, cfg.sim.csv_initial_state_library, cfg.reference.ip.limits_path, cfg.reference.boundary.envelope_path]
missing = [str(p) for p in required if p is None or not Path(p).exists()]
if missing:
    raise SystemExit("generated training config has missing paths inside container: " + ", ".join(missing))

env_cfg = replace(cfg, sim=replace(cfg.sim, gpu_device="cuda:0"))
env = TokamakMagneticControlEnv(env_cfg, batch_size=1, device="cuda:0", seed=int(cfg.training.seed))
print("preflight boundary_mode", env.cfg.boundary_mode)
print("preflight boundary_base_mode", env.cfg.boundary_base_mode)
print("preflight boundary_legacy_precision_index2", env.cfg.boundary_legacy_precision_index2)
print("preflight boundary_smooth_selected_level", env.cfg.boundary_smooth_selected_level)
print("preflight boundary_soft_level_selection", env.cfg.boundary_soft_level_selection)
if not math.isclose(float(env.cfg.boundary_legacy_precision_index2), 1.0e-6, rel_tol=0.0, abs_tol=1.0e-12):
    raise SystemExit("generated trim50 training must use boundary.legacy_precision_index2=1e-6")
if str(env.cfg.boundary_mode) == "tracked_flux_contour":
    raise SystemExit("generated trim50 training must not use tracked_flux_contour boundary mode")
if bool(env.cfg.boundary_smooth_selected_level):
    raise SystemExit("generated trim50 training must not use boundary.smooth_selected_level")
if bool(env.cfg.boundary_soft_level_selection):
    raise SystemExit("generated trim50 training must not use boundary.soft_level_selection")
obs = env.reset()
critic_obs = env.critic_obs()
print("preflight env_reset_ok", "obs_dim", tuple(obs.shape), "critic_obs_dim", tuple(critic_obs.shape), "action_dim", env.action_dim)
del env, obs, critic_obs
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PYCHECK

    python3 - <<PYCUDA
import torch
count = torch.cuda.device_count()
print("cuda_available", torch.cuda.is_available(), "device_count", count)
print("devices", [torch.cuda.get_device_name(i) for i in range(count)])
if count < 8:
    raise SystemExit(f"expected 8 visible CUDA devices, got {count}")
PYCUDA

    python3 -m torch.distributed.run \
      --nnodes 1 \
      --nproc_per_node 8 \
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
        --batch-size 32 \
        --unroll-length 100 \
        --min-replay-sequence-length 100 \
        --replay-capacity-episodes 1024 \
        --rollout-chunk-length 100 \
        --updates-per-rollout-chunk 64 \
        --hidden-dim 256 \
        --critic-hidden-dim 256 \
        --critic-mlp-hidden-dim 256 \
        --action-samples 64 \
        --actor-update-chunk-size 2048 \
        --save-checkpoints \
        --checkpoint-interval-steps "${CHECKPOINT_ENV_STEPS}" \
        --eval-interval-steps "${EVAL_ENV_STEPS}" \
        --eval-episodes "${HOLDOUT_EVAL_EPISODES}" \
        --eval-max-steps 100 \
        --controller-rollout-steps 0 \
        --wandb \
        --wandb-project "${WANDB_PROJECT_NAME}" \
        --wandb-name "${RUN_NAME}" \
        --wandb-group "${RUN_NAME}" \
        --wandb-mode "${WANDB_MODE}" \
        --wandb-optional \
        --wandb-metric-preset focused
  '
