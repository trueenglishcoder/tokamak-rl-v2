# Repository Layout

This document describes the source tree and the main local artifact folders.

## Versioned Source Areas

```text
README.md          Project overview and common commands
pyproject.toml     Package metadata and console entry points
configs/           Experiment configs
docs/              Operational docs and presentation assets
jobs/              Slurm job wrappers
scripts/           Python entry points and offline builders
tests/             Fast local tests and core contracts
tokamak_rl_v2/     Importable package
```

## Current Experiment Configs

Tracked experiment configs are:

```text
configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml
configs/experiments/t15_csv_easy_segmented_fixed_horizon.yaml
configs/experiments/t15_csv_hold_ip_fixed_horizon.yaml
configs/experiments/t15_hold_reset_boundary_ip_hold_gpu.yaml
configs/experiments/t15_static_boundary.yaml
configs/experiments/t15_static_boundary_gpu.yaml
```

The maintained production config is:

```text
configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml
```

## Current Job Wrappers

Tracked Slurm jobs are:

```text
jobs/train_t15_csv_segmented_profile_tcvdelta_t15boundary_12gpu_20m.sbatch
```

The maintained production job is:

```text
jobs/train_t15_csv_segmented_profile_tcvdelta_t15boundary_12gpu_20m.sbatch
```

## Package Layout

Main package areas:

```text
tokamak_rl_v2/config/        dataclasses and strict config loading
tokamak_rl_v2/env/           batched environment, references, CSV reset libraries
tokamak_rl_v2/networks/      actor and critic
tokamak_rl_v2/rewards/       TCV-derivative and legacy reward modules
tokamak_rl_v2/training/      trainer, MPO, replay, pipeline, distributed mode
tokamak_rl_v2/export/        deterministic actor export
```

## Processed Runtime Data

Production runtime reads:

```text
data/processed/t15_csv_initial_states.npz
data/processed/t15_csv_initial_states.json
data/processed/t15_reference_limits.json
../tokamak-sim/runs/t15md_limited_replay_dataset/
```

These are versioned runtime-facing artifacts produced by offline builder
scripts plus local replay outputs. The environment uses them at runtime instead
of reading raw T15 CSVs.

## Local Artifact Areas

These folders are local outputs, not source:

```text
outputs/           training outputs, CSVs, exports, validation JSON, checkpoints
slurm_logs/        Slurm stdout/stderr
wandb/             local W&B cache
.venv/             local virtual environment
.pytest_cache/     pytest cache
__pycache__/       bytecode cache
```

## Typical Server Paths

Working tree on the cluster:

```text
/scratch/$USER/tokamak/tokamak-rl-v2
```

Container mount:

```text
/scratch/$USER/tokamak -> /workspace
```

Inside the job container:

```text
/workspace/tokamak-rl-v2
/workspace/tokamak-sim
```
