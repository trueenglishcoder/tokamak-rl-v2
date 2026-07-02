#!/bin/bash
#SBATCH --job-name=tokamak-rl-v2-tcvj-simple8
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/tokamak-rl-v2-tcvj-simple-matched-trim50-8gpu-100m-%j.out
#SBATCH --error=slurm_logs/tokamak-rl-v2-tcvj-simple-matched-trim50-8gpu-100m-%j.err

set -euo pipefail
trap 'echo "training launch failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs outputs

export SOURCE_CONFIG=${SOURCE_CONFIG:-configs/experiments/t15_simple_manifold_generated_trim50_idealized_matched_0p1s_tcvjdot_balanced_mpo.yaml}
export BUILDER_SCRIPT=${BUILDER_SCRIPT:-scripts/build_t15_simple_manifold_generated_trim50_idealized_0p1s.py}
export MACHINE_CONFIG_SOURCE=${MACHINE_CONFIG_SOURCE:-../tokamak-sim/runs/t15md_limited_replay_dataset_trim50_idealized_matched_gpu_plain_1e6/T15MD_new_data_legacy_contour_limited_replay.toml}
export RESET_LIBRARY=${RESET_LIBRARY:-data/processed/t15_simple_manifold_generated_trim50_idealized_matched_0p1s_initial_states.npz}
export TARGET_LIBRARY=${TARGET_LIBRARY:-data/processed/t15_simple_manifold_generated_trim50_idealized_matched_0p1s/t15_feasible_generated_trim50_idealized_0p1s_targets.npz}
export TRAIN_ENV_STEPS=${TRAIN_ENV_STEPS:-100000000}
export EVAL_ENV_STEPS=${EVAL_ENV_STEPS:-500000}
export CHECKPOINT_ENV_STEPS=${CHECKPOINT_ENV_STEPS:-1000000}
export NUM_ENVS=${NUM_ENVS:-2048}
export GPUS_PER_NODE=8
export WORLD_SIZE=8
export BATCH_SIZE=32
export UPDATES_PER_ROLLOUT_CHUNK=64
export RUN_LABEL=${RUN_LABEL:-t15_simple_manifold_generated_trim50_idealized_matched_0p1s_tcvjdot_balanced}
export TRAINING_FAMILY_TOKEN=${TRAINING_FAMILY_TOKEN:-trim50_idealized_matched}
export RUN_NAME=${RUN_NAME:-${RUN_LABEL}_8gpu_100m_${SLURM_JOB_ID:-manual}}
export TRAIN_OUTPUT=${TRAIN_OUTPUT:-outputs/${RUN_NAME}}
export GENERATED_CONFIG=${GENERATED_CONFIG:-${TRAIN_OUTPUT}/generated_configs/${RUN_NAME}.json}
export WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-tokamak-rl-v2-tcvjdot-simple-manifold-matched-trim50-8gpu-100m-${SLURM_JOB_ID:-manual}}
export WANDB_MODE=${WANDB_MODE:-online}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$(( 20000 + (${SLURM_JOB_ID:-1731} % 40000) ))}

for path in "${SOURCE_CONFIG}" "${BUILDER_SCRIPT}" "${MACHINE_CONFIG_SOURCE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required simple-manifold training artifact: ${path}" >&2
    exit 2
  fi
done
if (( NUM_ENVS % WORLD_SIZE != 0 )); then
  echo "NUM_ENVS=${NUM_ENVS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi

if [[ ! -f "${RESET_LIBRARY}" || ! -f "${TARGET_LIBRARY}" ]]; then
  echo "simple-manifold generated reset/target libraries are missing; building them inside the container"
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
    echo "failed to create simple-manifold training artifact: ${path}" >&2
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
cp "${MACHINE_CONFIG_SOURCE}" "$(dirname "${GENERATED_CONFIG}")/T15MD_new_data.toml"

python3 - <<PYCFG
import json
from pathlib import Path

source = Path("${SOURCE_CONFIG}")
out = Path("${GENERATED_CONFIG}")
cfg = json.loads(source.read_text(encoding="utf-8"))

cfg["name"] = "${RUN_LABEL}_8gpu_100m"
cfg["sim"]["config_path"] = "/workspace/tokamak-rl-v2/${TRAIN_OUTPUT}/generated_configs/T15MD_new_data.toml"
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
cfg["training"]["eval_max_steps"] = 100
cfg["training"]["save_checkpoints"] = True
cfg["training"]["early_stop_patience_evals"] = 0
cfg["training"]["early_stop_min_delta"] = 0.0

family_token = "${TRAINING_FAMILY_TOKEN}"
checks = {
    "generated_config_copy": cfg["sim"]["config_path"].endswith("/generated_configs/T15MD_new_data.toml"),
    "generated_reset_library": family_token in str(cfg["sim"]["csv_initial_state_library"]),
    "generated_reference_dir": family_token in str(cfg["reference"]["boundary"]["feasible_reference_dir"]),
    "feasible_window_ip": cfg["reference"]["ip"]["kind"] == "feasible_generated_window",
    "feasible_window_boundary": cfg["reference"]["boundary"]["kind"] == "feasible_generated_window",
    "same_generated_reference_dir": cfg["reference"]["ip"]["feasible_reference_dir"] == cfg["reference"]["boundary"]["feasible_reference_dir"],
    "jdot_command": cfg["sim"]["action_contract"] == "jdot_command",
    "episode_steps_100": int(cfg["sim"]["max_episode_steps"]) == 100,
    "duration_0p1": abs(float(cfg["reference"]["duration_s"]) - 0.1) < 1e-12,
    "reference_seed_matches_successful_oracle": int(cfg["reference"]["seed"]) == 11,
    "actor_obs_v6": cfg["observation"]["actor_kind"] == "controller_state_v6",
    "critic_obs_v2": cfg["observation"]["critic_kind"] == "compact_training_state_v2",
    "reward_tcv": cfg["reward"]["kind"] == "tcv_derivative",
    "batch_size_matches_successful_oracle": int(cfg["learner"]["batch_size"]) == 32,
    "updates_match_successful_oracle": int(cfg["learner"]["updates_per_rollout_chunk"]) == 64,
    "num_envs_matches_successful_oracle": int(cfg["training"]["num_envs"]) == 2048,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"simple-manifold 8gpu training config failed invariant checks: {failed}")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(out)
PYCFG

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
import json
import math
import numpy as np
import torch
from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.training.policy_pipeline import _preflight_artifact_failure

cfg = load_experiment_config("/workspace/tokamak-rl-v2/${GENERATED_CONFIG}")
artifact_failure = _preflight_artifact_failure(cfg)
if artifact_failure is not None:
    raise SystemExit("trainer artifact preflight failed before distributed launch: " + json.dumps(artifact_failure, sort_keys=True))
required = [
    cfg.sim.config_path,
    cfg.sim.csv_initial_state_library,
    cfg.reference.ip.feasible_reference_dir,
    cfg.reference.boundary.feasible_reference_dir,
]
missing = [str(p) for p in required if p is None or not Path(p).exists()]
if missing:
    raise SystemExit("simple-manifold generated training config has missing paths inside container: " + ", ".join(missing))

with np.load("/workspace/tokamak-rl-v2/${RESET_LIBRARY}", allow_pickle=False) as d:
    if "params0" not in d.files or "difficulty_bin" not in d.files:
        raise SystemExit("simple-manifold reset library must include params0 and difficulty_bin")
    reset_zones = sorted(set(d["difficulty_bin"].astype(str).tolist()))
    print("preflight reset rows", int(d["ip0"].shape[0]), "holdout", int(np.count_nonzero(d["split"].astype(str) == "holdout")), "zones", reset_zones)
    if reset_zones != ["core"]:
        raise SystemExit(f"simple-manifold reset library should only use core curriculum bucket, got {reset_zones}")
with np.load("/workspace/tokamak-rl-v2/${TARGET_LIBRARY}", allow_pickle=False) as d:
    target_zones = sorted(set(d["zone"].astype(str).tolist()))
    modes = sorted(set(d["mode"].astype(str).tolist())) if "mode" in d.files else []
    print("preflight target rows", int(d["ip_ref"].shape[0]), "zones", target_zones, "modes", modes)
    if target_zones != ["core"]:
        raise SystemExit(f"simple-manifold target library should only use core curriculum bucket, got {target_zones}")

env = TokamakMagneticControlEnv(cfg, batch_size=2, device="cuda:0", seed=int(cfg.training.seed))
env.set_curriculum_progress(0.75)
obs = env.reset()
print("preflight boundary_mode", env.cfg.boundary_mode)
print("preflight boundary_base_mode", env.cfg.boundary_base_mode)
print("preflight boundary_legacy_precision_index2", env.cfg.boundary_legacy_precision_index2)
print("preflight env_reset_ok", tuple(obs.shape), tuple(env.reference.ip.shape), tuple(env.reference.radii.shape), "action_dim", env.action_dim)
if not math.isclose(float(env.cfg.boundary_legacy_precision_index2), 1.0e-6, rel_tol=0.0, abs_tol=1.0e-12):
    raise SystemExit(
        "simple-manifold training must use boundary.legacy_precision_index2=1e-6; "
        f"loaded {env.cfg.boundary_legacy_precision_index2}"
    )
if str(env.cfg.boundary_mode) == "tracked_flux_contour":
    raise SystemExit("simple-manifold training must not use tracked_flux_contour boundary mode")
if bool(env.cfg.boundary_smooth_selected_level):
    raise SystemExit("simple-manifold training must not use boundary.smooth_selected_level")
if bool(env.cfg.boundary_soft_level_selection):
    raise SystemExit("simple-manifold training must not use boundary.soft_level_selection")
del env, obs
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
