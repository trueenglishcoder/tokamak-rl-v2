# Artifacts And Metrics

This document explains what the maintained production path writes and how to
read the main outputs.

## Output Directory

A normal production run writes something like:

```text
outputs/t15_csv_initial_segmented_profile_boundary_mpo_1gpu_<jobid>
```

Typical contents:

```text
config_snapshot.json
losses.csv
reward_components.csv
eval_history.csv
replay_health.csv
metrics.json
policy_validation.json
closed_loop_rollout_report.json
exports/best_actor/
exports/final_actor/
checkpoints/                 optional
```

## `config_snapshot.json`

Resolved config after CLI overrides. Use this when comparing runs, because it is
the exact configuration seen by the trainer.

## `losses.csv`

Written when learner updates happen.

Important columns:

- `critic_loss`
- `actor_loss`
- `mean_kl`
- `std_kl`
- `q_mean`
- `target_q_mean`
- `actor_mle_loss`
- `actor_param_delta_norm`
- `sampled_q_spread`
- `policy_weight_max`
- `policy_weight_entropy`
- `mpo_temperature`
- `mean_kl_penalty`
- `std_kl_penalty`
- replay-health fields

Notes:

- `actor_loss` may be negative; that is not automatically a bug.
- Q-values may be negative because rewards are negative physical costs.
- MPO diagnostics are logs, not production pass/fail gates.

## `reward_components.csv`

Per-logging-interval reward diagnostics.

Common production columns:

- `shape_error_mean_m`
- `shape_error_max_m`
- `ip_error_a`
- `current_over_limit_a`
- `current_usage_fraction`
- `current_margin_fraction`
- `derivative_usage`
- `max_abs_action`
- `action_rms`
- `delta_action_rms`
- `physical_cost`
- `shape_mean_loss`
- `shape_max_loss`
- `ip_loss`
- `current_loss`
- `derivative_loss`
- `action_loss`
- `delta_action_loss`
- `boundary_found`
- termination flags

These are the most direct view of what the controller is actually doing.

## `eval_history.csv`

Periodic deterministic actor evaluation on fixed seeds.

In production mode this always uses the full configured episode horizon:

```text
sim.max_episode_steps
```

Important fields:

- boundary retention
- current-over-limit metrics
- shape error
- Ip error
- mean and minimum episode completion
- late-episode metrics

## `replay_health.csv`

Replay and learner-readiness diagnostics:

- replay size
- completed episode count
- full-sequence eligible episode count
- minimum-sequence eligible episode count
- mean/min/max episode length
- learner no-update warning

If replay never becomes ready, the pipeline should fail fast with
`failed_replay_health`.

## `metrics.json`

Trainer summary written at the end of training or fail-fast exit.

Useful fields:

- `status`
- `steps`
- `env_steps`
- `updates`
- `best_eval`
- `best_eval_details`
- `distributed_mode`
- `total_training_envs`
- `save_checkpoints`

## `policy_validation.json`

This is the main decision file.

It contains:

- final status
- selected checkpoint
- selected export directory
- actor holdout evaluation
- no-control baseline when applicable
- tail training-loss summary
- exported-controller rollout report
- final gates

For production, pass/fail is based on physical behavior:

- episode completion
- boundary retention
- current-limit respect
- shape accuracy
- Ip accuracy and improvement
- exported-controller full-episode rollout

Not on MPO diagnostic thresholds.

## `closed_loop_rollout_report.json`

Full-episode validation of the exported controller through
`tokamak-sim`’s learned-controller runtime.

Important fields:

- `mean_episode_completion`
- `min_episode_completion`
- `boundary_found_mean`
- `boundary_found_late_min`
- `shape_error_mean_m`
- `shape_error_late_m`
- `ip_error_a`
- `ip_error_late_a`
- `current_over_limit_a_max`
- `current_over_limit_a_late_max`
- `action_rms`

## `exports/`

Each export directory contains the deterministic actor bundle:

```text
actor.pt
policy_weights.npz
controller_schema.json
normalization.json
metadata.json
```

The production actor schema is:

```text
controller_state_v3
```

## Checkpoints

When checkpoint saving is enabled:

```text
checkpoints/latest.pt
checkpoints/best.pt
checkpoints/final.pt
```

Manual export from a checkpoint is supported, but only for
`controller_state_v3` checkpoints.
