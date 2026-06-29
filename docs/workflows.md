# Workflows

This file records maintained workflows only. Historical reward searches and old
delta-Jdot learned-policy jobs were removed from active launch paths.

## Local Checks

```bash
cd ~/tokamak/tokamak-rl-v2
PYTHONPATH=.:../tokamak-sim python3 -m pytest -q \
  tests/test_production_cleanup.py \
  tests/test_replay_window_oracle_contracts.py \
  tests/test_hold_boundary_eval.py
```

Run all core contracts when changing env/reward/replay/export code:

```bash
PYTHONPATH=.:../tokamak-sim python3 -m pytest -q tests/test_core_contracts.py
```

## Build Trim50 Oracle Targets

```bash
cd /scratch/$USER/tokamak/tokamak-sim
git pull --ff-only origin main

cd /scratch/$USER/tokamak/tokamak-rl-v2
git pull --ff-only origin main
mkdir -p slurm_logs

BUILD_JOB=$(sbatch --parsable jobs/build_t15_new_replay_window_oracle_targets_1gpu.sbatch)
echo "BUILD_JOB=$BUILD_JOB"
squeue -j "$BUILD_JOB"
```

Expected generated files:

```text
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/t15_replay_window_oracle_targets.npz
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/oracle_summary.json
configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml
```

## Train The Final Policy

```bash
cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs

TRAIN_JOB=$(sbatch --parsable jobs/train_t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch)
echo "TRAIN_JOB=$TRAIN_JOB"
squeue -j "$TRAIN_JOB"
```

The job performs preflight checks for the trim50 machine config, reset library,
oracle targets, action contract, observation schemas, and boundary extraction
settings before training starts.

## Monitor A Run

```bash
squeue -u "$USER"
sacct -j "$TRAIN_JOB" --format=JobID,JobName%42,State,ExitCode,Elapsed,NodeList,Reason -X
tail -f slurm_logs/*"${TRAIN_JOB}"*.out
tail -f slurm_logs/*"${TRAIN_JOB}"*.err
```

## Package A Run Without Checkpoints

```bash
cd /scratch/$USER/tokamak/tokamak-rl-v2

RUN=t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m_<jobid>
python3 scripts/copy_run_without_checkpoints.py \
  --run "outputs/${RUN}" \
  --out "_server_outputs/${RUN}_no_checkpoints_for_codex"
```

## Evaluate An Exported Actor

Replay-window evaluation:

```bash
cd /scratch/$USER/tokamak/tokamak-rl-v2

python3 scripts/evaluate_replay_window_oracle_baselines.py \
  --config outputs/<run>/generated_configs/<run>.json \
  --export-dir outputs/<run>/exports/best_actor \
  --output-dir outputs/<run>/oracle_eval_holdout \
  --all-windows \
  --steps 100 \
  --split holdout \
  --device cuda:0
```

Hold-boundary diagnostic:

```bash
EVAL_JOB=$(sbatch --parsable jobs/eval_hold_boundary_cut900_seg300_8gpu_800x500.sbatch)
python3 scripts/summarize_hold_boundary_eval.py \
  "outputs/hold_boundary_eval_cut900_seg300_${EVAL_JOB}" \
  --out-dir "outputs/hold_boundary_eval_cut900_seg300_${EVAL_JOB}/summary"
```

## Important Metrics

Use physical metrics first:

```text
eval/shape_error_mean_m_late
eval/shape_error_max_m_late
eval/ip_error_a_late
eval/mean_episode_completion
eval/current_over_limit_*_late
eval/action_saturation_fraction_late
eval/action_rms_late
```

Training losses, Q values, and actor loss are diagnostics, not the primary
acceptance criteria.
