# Artifacts And Metrics

This document explains outputs from the maintained trim50 replay-window
training path.

## Output Directory

The canonical 100M job writes:

```text
outputs/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m_<jobid>/
```

Typical contents:

```text
config_snapshot.json
generated_configs/
losses.csv
reward_components.csv
eval_history.csv
replay_health.csv
metrics.json
policy_validation.json
closed_loop_rollout_report.json
exports/best_actor/
exports/final_actor/
checkpoints/
```

## Config Snapshots

Use `config_snapshot.json` and `generated_configs/<run>.json` when comparing
runs. They are the exact resolved configuration seen by the trainer.

The job also copies the active `tokamak-sim` trim50 machine config into the
run-local `generated_configs/` directory for export/evaluation reproducibility.

## `eval_history.csv`

Periodic deterministic actor evaluation. For the active config this evaluates
100-step replay-window episodes.

High-signal columns:

```text
shape_error_mean_m_late
shape_error_max_m_late
ip_error_a_late
mean_episode_completion
full_episode_success
boundary_found_late_min
current_over_limit_*_late
action_saturation_fraction_late
action_rms_late
selection_score
```

Physical metrics are the primary acceptance signal. Actor loss, critic loss, and
Q values are diagnostics.

## `reward_components.csv`

Reward diagnostics written during training:

```text
shape_error_mean_m
shape_error_max_m
ip_error_a
current_usage_fraction
current_over_limit_a
derivative_usage
action_rms
delta_action_rms
physical_cost
shape_mean_loss
shape_max_loss
ip_loss
current_loss
derivative_loss
actuator_saturation_loss
boundary_found
terminal_* diagnostics
```

For `jdot_command`, derivative effort is based on the applied absolute Jdot
command. Saturation loss measures requested normalized Jdot minus applied
clipped normalized Jdot.

## `losses.csv`

Learner diagnostics:

```text
critic_loss
actor_loss
q_mean
target_q_mean
sampled_q_spread
policy_weight_max
policy_weight_entropy
mpo_temperature
mean_kl
std_kl
```

Negative actor loss is not itself a failure. Rising physical metrics are what
matter.

## `replay_health.csv`

Replay readiness and episode-length diagnostics:

```text
replay_length
completed_episodes
eligible_sequences
mean_episode_length
min_episode_length
max_episode_length
```

The final pipeline uses 100-step episodes, so stable full-horizon completion is
expected when the controller is healthy.

## `metrics.json` And `policy_validation.json`

`metrics.json` is the training summary. `policy_validation.json` is the main
decision file and records selected checkpoint/export paths plus validation
results.

Useful fields:

```text
status
steps/env_steps
best_eval
best_eval_details
selected_checkpoint
selected_export_dir
closed_loop_rollout_report
```

## Exports

The active learned-controller export is:

```text
exports/best_actor/
```

Expected files:

```text
actor.pt
policy_weights.npz
controller_schema.json
normalization.json
metadata.json
```

The active schema is:

```text
observation_kind = controller_state_v6
action_contract = absolute_jdot_command_v1
```

Old delta-Jdot exports are intentionally rejected by `tokamak-sim`.

## Checkpoints

Checkpoints are large and not needed for ordinary analysis if `exports/best_actor`
exists. Keep them on the server unless continuing training or re-exporting.
