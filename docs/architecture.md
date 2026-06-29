# Architecture

This document describes the maintained runtime architecture of
`tokamak-rl-v2` after the trim50 cleanup.

## System Boundary

`tokamak-sim` provides:

```text
machine/config loading
CPU and GPU plant stepping
fixed-angle GPU boundary measurements for training
learned-controller runtime loading
presentation/replay artifact generation
```

`tokamak-rl-v2` provides:

```text
batched RL environment
reset/window libraries
replay-window Ip and boundary references
TCV-derivative reward
actor and critic networks
episode replay
MPO learner
training, export, and evaluation scripts
```

The exported bundle contains only the deterministic actor path plus schema and
normalization metadata. The critic and learner stay on the training side.

## Maintained Production Path

```text
real trim50 T15 CSVs
-> replay exact coil currents through tokamak-sim
-> record plain GPU fixed-angle boundary references at legacy_precision_index2=1e-6
-> build 0.1 s replay-window oracle targets
-> sample coherent reset/window from train shots
-> reset batched GPU plant
-> actor outputs normalized absolute Jdot command
-> env clips Jdot and computes absolute next currents
-> tokamak-sim advances plant
-> env measures fixed-angle boundary on GPU
-> TCV-derivative reward
-> episode replay
-> MPO update
-> deterministic actor export
```

The canonical production entry point is:

```text
scripts/train_policy_pipeline.py
```

with:

```text
configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml
```

## Environment

Main class:

```text
tokamak_rl_v2.env.batch_env.TokamakMagneticControlEnv
```

The environment owns:

```text
batched plant state
current reset/window metadata
Ip and boundary target batch
previous applied normalized Jdot action
current-over-limit counters
reward and termination state
```

Each lane is independent. Batching is for throughput only.

## Reset And Reference Path

For production:

```text
sample one row from t15_new_trim50_plain_gpu1e6 oracle initial-state library
load matching replay-window target by shot/time
reset plant with Ip0, PFC0, SOL0
set Ip reference from the real window
set boundary reference from the replayed boundary window
```

Runtime training does not synthesize Ip trajectories for the final path. Ip and
boundary targets come from the same real shot/time window.

## Action Contract

Active learned-policy contract:

```text
sim.action_contract = jdot_command
export action_contract = absolute_jdot_command_v1
```

Actor output `a_t in [-1, 1]` means:

```text
Jdot_cmd = a_t * derivative_limit
J_next = J_now + dt * Jdot_cmd
```

The plant API still receives absolute next currents via `step_currents`.

Zaitsev/LQR diagnostic controllers may still use delta-Jdot internally, but
learned policies do not.

## Observations

Actor schema:

```text
controller_state_v6
```

Critic schema:

```text
compact_training_state_v2
```

The actor sees controller-available scalar/vector state: Ip, target preview,
currents, current derivatives, boundary radii, boundary target, boundary error,
previous applied action, and status features.

The final critic path is compact and does not require full `psi_flat`.

## Reward

The active reward is `tcv_derivative`.

It combines:

```text
boundary mean/max error quality
Ip error quality
current-limit quality
Jdot/actuator quality
requested-vs-applied clipping quality
boundary-present quality
terminal replacement on operational failure
```

Normal reward is `reward_scale * quality`. Terminal reward is the configured
scaled terminal value.

## Evaluation

Production uses full configured episodes for:

```text
periodic actor eval
final holdout replay-window eval
exported actor checks
hold-boundary diagnostics
```

Use physical metrics first:

```text
shape_error_mean_m_late
shape_error_max_m_late
ip_error_a_late
mean_episode_completion
current_over_limit_*_late
action_saturation_fraction_late
```
