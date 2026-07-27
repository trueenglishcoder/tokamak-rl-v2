# Current RL Pipeline

This repo is currently organized around one supported learned-policy path:

```text
T15 6-PFC trim50 CSV data
+ plain GPU fixed-angle replay boundary references, legacy_precision_index2=1e-6
+ 0.1 s replay-window episodes from real T15 segments
+ oracle targets from exact T15 current replay
+ learned action contract jdot_command
+ actor observation controller_state_v6
+ critic observation compact_training_state_v2
+ TCV-derivative reward
+ 8-GPU balanced-oracle MPO training
```

Older delta-Jdot learned-policy jobs, static-boundary searches, 2 s T15-boundary
searches, 0.2 s antidrift runs, and 4-PFC launch scripts are not active.

## Inputs

Required ignored/local artifacts:

```text
../tokamak-sim/data/t15_data_new_trim50/
../tokamak-sim/runs/t15md_trim50_plain_gpu1e6_top5_replay/
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/
```

Train shots are `3855, 3863, 3856, 3854`. Shot `3859` is held out.

## Contract

- The actor outputs normalized absolute `Jdot` commands.
- The env clips to derivative limits, computes `J_next = J_now + dt * Jdot`,
  and calls `tokamak-sim` with absolute next currents.
- Replay stores the requested normalized `Jdot` command.
- Exported learned controllers use `absolute_jdot_command_v1`.
- Old delta-Jdot learned checkpoints and exports are invalid.

## Canonical Files

```text
configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml
jobs/build_t15_new_replay_window_oracle_targets_1gpu.sbatch
jobs/train_t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch
scripts/build_t15_new_replay_window_oracle_targets.py
scripts/export_policy.py
scripts/evaluate_replay_window_oracle_baselines.py
scripts/evaluate_hold_boundary_task.py
```

## Canonical Server Flow

```bash
cd /scratch/$USER/tokamak/tokamak-sim
git pull --ff-only origin main

cd /scratch/$USER/tokamak/tokamak-rl-v2
git pull --ff-only origin main
mkdir -p slurm_logs

sbatch jobs/build_t15_new_replay_window_oracle_targets_1gpu.sbatch
sbatch jobs/train_t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch
```

The 100M job writes a run-local generated config and a copied
`T15MD.toml` under its output directory. That generated config is the
source of truth for exports and post-run analysis.

## Evaluation

Use `scripts/evaluate_replay_window_oracle_baselines.py` for all-window train or
holdout replay-window evaluation of an exported actor. Use
`scripts/evaluate_hold_boundary_task.py` for synthetic hold-boundary diagnostics.

Physical metrics matter most:

```text
shape_error_mean_m_late
shape_error_max_m_late
ip_error_a_late
mean_episode_completion
current_over_limit_*_late
action_saturation_fraction_late
action_rms_late
```

MPO losses and actor loss are debugging signals, not acceptance metrics.
