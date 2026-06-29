# Repository Layout

This document describes the maintained source tree after the trim50 cleanup.
Generated runs, W&B caches, and downloaded bundles are local artifacts rather
than source.

## Versioned Source Areas

```text
README.md              Project overview and common commands
CURRENT_PIPELINE.md    Canonical active RL pipeline
configs/               Experiment configs
docs/                  Operational docs and presentation notes
jobs/                  Slurm job wrappers
scripts/               Python entry points and offline builders
tests/                 Fast local tests and contract checks
tokamak_rl_v2/         Importable package
```

## Active Experiment Config

The maintained production config is:

```text
configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml
```

It describes the final working path:

```text
T15 6-PFC trim50 data
plain GPU fixed-angle boundary references, legacy_precision_index2=1e-6
0.1 s replay-window episodes
jdot_command learned-policy action contract
controller_state_v6 actor observations
compact_training_state_v2 critic observations
TCV-derivative reward
```

## Active Job Wrappers

Maintained Slurm jobs are:

```text
jobs/build_t15_new_replay_window_oracle_targets_1gpu.sbatch
jobs/train_t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch
jobs/eval_hold_boundary_8gpu_800x500.sbatch
jobs/eval_hold_boundary_cut900_seg300_8gpu_800x500.sbatch
```

Retired reward sweeps, delta-Jdot learned-policy jobs, static-boundary
single-segment jobs, 0.2 s antidrift jobs, 2 s replay-boundary jobs, and 4-PFC
training launchers are intentionally not active.

## Package Layout

```text
tokamak_rl_v2/config/        dataclasses and strict config loading
tokamak_rl_v2/env/           batched env, references, reset/window libraries
tokamak_rl_v2/networks/      actor and critic
tokamak_rl_v2/rewards/       TCV-derivative reward
tokamak_rl_v2/training/      trainer, MPO, replay, pipeline, distributed mode
tokamak_rl_v2/export/        deterministic actor export
```

## Runtime Data

The final path reads local/generated data:

```text
../tokamak-sim/data/t15_data_new_trim50/
../tokamak-sim/runs/t15md_trim50_plain_gpu_1e6_setup/
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/
```

These are produced by replay/oracle-target builders and are intentionally not
raw source. T15 CSV source data stays protected in `tokamak-sim/data/`.

## Local Artifact Areas

```text
outputs/       training outputs, exports, metrics, checkpoints
slurm_logs/    Slurm stdout/stderr
wandb/         local W&B cache
.venv/         local virtual environment
.pytest_cache/ pytest cache
__pycache__/   bytecode cache
```

## Server Paths

Cluster working tree:

```text
/scratch/$USER/tokamak/tokamak-rl-v2
```

Container mount:

```text
/scratch/$USER/tokamak -> /workspace
```

Inside jobs:

```text
/workspace/tokamak-rl-v2
/workspace/tokamak-sim
```
