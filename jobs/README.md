# Slurm Jobs

This directory intentionally contains only supported production jobs and a small
set of diagnostics.

## Production

```text
build_t15_new_replay_window_oracle_targets_1gpu.sbatch
train_t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch
```

These use:

```text
trim50 real T15 data
plain GPU fixed-angle boundary extraction, legacy_precision_index2=1e-6
0.1 s replay-window/oracle targets from real T15 windows
jdot_command learned-policy action contract
controller_state_v6 actor input
compact_training_state_v2 critic input
TCV-derivative reward
```

## Supported Synthetic Experiment

```text
train_t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch
```

This is the current synthetic-long experiment. It first builds 30 new
1.0-1.5 s synthetic training parents plus one synthetic holdout parent inside
the Slurm container, cuts them into overlapping 100-step replay windows, and
then delegates to the same 8-GPU training script used by the real trim50
production path. The only intended differences are dataset paths, run name, and
W&B project name.

## Diagnostics

```text
eval_hold_boundary_8gpu_800x500.sbatch
eval_hold_boundary_cut900_seg300_8gpu_800x500.sbatch
```

Old generated/idealized/actuator/perturbed/simple-manifold/long-target jobs
were removed from the active tree because they repeatedly produced confusing,
non-production launch paths. New synthetic work should stay behind the explicit
`synthetic_long30` naming unless it is promoted or retired.

## Command Hygiene

- Use `squeue -u "$USER"` for status checks.
- Do not run Python tools that import `numpy` on the login node. Use Slurm
  container jobs or explicit `srun --container-image ...`.
- Jobs should preflight required inputs and print the exact output directory
  before training starts.
