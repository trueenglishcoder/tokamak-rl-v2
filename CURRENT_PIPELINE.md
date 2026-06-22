# Current RL Pipeline

This repo's active path is the repaired TCV-delta T15-boundary production path.

## Plant And Action Contract

- The actor outputs requested normalized `delta_Jdot`.
- The environment accumulates `delta_Jdot` into an applied derivative command,
  converts that to absolute next coil currents, and sends those currents to the
  current `tokamak-sim` plant.
- Replay stores the actor-requested delta action, because that is what the actor
  controls.
- The environment state keeps both previous requested delta action and previous
  accumulated derivative command. Checkpoints missing either state are invalid.

## References

- Ip targets remain generated `segmented_profile` programs:
  2000 steps, half-slope T15-scale ramps, no back-to-back ramps, and sparse
  500 ms target preview.
- Boundary targets use `reference.boundary.kind=t15_replay_segment_conditioned`.
- The boundary library comes from:
  `../tokamak-sim/runs/t15md_limited_replay_dataset/`.
- For each CSV reset, the matching shot's smoothed replay boundary segment is
  selected by reset time/source index and shifted so step 0 equals the reset
  boundary. Boundary references are not looked up by instantaneous Ip.

## Reward

- The active reward kind is `tcv_derivative`.
- Normal per-step reward is `reward_scale * quality`.
- Terminal reward is the source-style scaled value:
  `reward_scale * terminal_reward`.
- Disk CSV/JSON metrics remain the source of truth. W&B is only monitoring.

## Canonical Launch

Use only:

```bash
sbatch jobs/train_t15_csv_segmented_profile_tcvdelta_t15boundary_12gpu_20m.sbatch
```

The job creates a fresh W&B project name using the Slurm job id and logs a focused
metric preset.
