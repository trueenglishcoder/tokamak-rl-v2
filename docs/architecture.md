# Architecture

This document describes the maintained runtime architecture of
`tokamak-rl-v2`.

## System Boundary

`tokamak-sim` provides:

- machine configuration loading
- CPU `PlasmaModel`
- GPU `BatchedGpuTokamakSimulator`
- boundary finding from `psi`
- limiter geometry
- learned-controller runtime loading

`tokamak-rl-v2` provides:

- batched RL environment
- reset-state libraries and runtime wrappers
- Ip and boundary reference generation
- TCV-derivative quality reward
- actor and critic networks
- replay
- MPO learner
- training / evaluation / export pipeline

The exported bundle contains only the deterministic actor path plus schema and
normalization metadata. The critic and learner stay on the training side.

## Maintained Production Path

The production path is:

```text
CSV reset row
-> reset tokamak-sim plant
-> read simulator-found boundary
-> generate segmented_profile Ip target
-> use matching-shot smoothed replay boundary segment anchored to reset boundary
-> actor samples requested normalized delta-Jdot
-> environment accumulates derivative command and sends absolute next currents
-> simulator advances plant
-> TCV-derivative reward
-> sequence replay
-> recurrent critic + MPO update
-> deterministic actor export
-> full-episode holdout actor evaluation
-> full-episode exported-controller validation
```

Production starts at:

```text
scripts/train_policy_pipeline.py
```

with:

```text
configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml
```

## Environment

Main class:

```text
tokamak_rl_v2.env.batch_env.TokamakMagneticControlEnv
```

The environment owns:

- batched plant state
- current reset metadata
- reference batch
- previous action
- current-over-limit counters
- reset-time `psi` normalization statistics

Each lane is independent. The environment is batched only for throughput.

### Reset Path

For production:

- sample one coherent row from `t15_csv_initial_states.npz`
- reset the plant with that row’s `Ip0 + PFC0 + SOL0`
- compute the physical boundary after reset
- generate a shot-matched replay-boundary target through
  `t15_replay_segment_conditioned`

Runtime training does not read raw T15 CSVs directly.

### Time-Base Invariants

Two time-base checks are enforced:

```text
reference.t_step == tokamak-sim physics.t_step
reference.duration_s == sim.max_episode_steps * reference.t_step   (production)
```

That keeps the generated target horizon aligned with the simulator horizon.

## References

Boundary target:

- `t15_replay_segment_conditioned`

Ip target:

- `segmented_profile`

`segmented_profile` generates bounded segment programs over:

- `hold`
- `ramp_up`
- `ramp_down`

The generated target:

- starts at reset Ip
- stays strictly positive
- stays inside aggregate real-data bounds
- obeys signed ramp-rate limits
- contains at least one hold
- contains at least one nonzero ramp

If `smooth_ramps=true`, ramp segments are cosine-eased while preserving rate
limits through a safety factor.

## Observations

Actor observation kind:

```text
controller_state_v3
```

Critic observation kind:

```text
privileged_training_state_v1
```

Actor observations include only controller-available features, such as:

- step fraction
- Ip / Ip target / Ip error
- active currents
- active current derivatives
- measured boundary radii
- reference radii
- boundary error
- boundary-found flag
- previous action
- target preview

Critic observations include the actor observation plus training-only privilege:

- normalized `psi_flat`
- current usage fraction
- current margin fraction
- derivative usage

## Reward

The maintained reward is `tcv_derivative`.

It combines:

- TCV-style boundary quality
- TCV-style Ip quality
- current operational-limit quality
- derivative/actuator quality
- requested-vs-applied delta rejection quality
- boundary-present quality
- terminal reward replacement for operational failures

Normal reward is `reward_scale * quality`. Terminal reward is the scaled
source-style value `reward_scale * terminal_reward`.

## Learning Stack

Networks:

- feedforward Gaussian actor
- recurrent Q critic

Replay:

- episode-aware sequence replay
- actor observations
- critic observations
- action
- reward
- discount
- next actor observations
- next critic observations
- terminal masks

Learner:

- Maximum a Posteriori Policy Optimisation (MPO)

Production pass/fail does not depend on MPO diagnostic thresholds. Metrics such
as `sampled_q_spread` and `policy_weight_max` are still logged for inspection.

## Evaluation And Validation

Production mode always uses the full configured episode horizon for:

- periodic actor evaluation
- final holdout actor evaluation
- exported-controller validation

Controller validation runs the exported bundle through
`tokamak-sim`’s `LearnedMagneticController` path, not just the PyTorch actor in
isolation.
