# Configuration

Experiment files live under `configs/experiments/` and are loaded by
`tokamak_rl_v2.config.loader.load_experiment_config`.

Parsing is strict. Unknown keys are rejected at the top level and in the main
sections:

```text
sim
reference
reference.ip
reference.boundary
observation
reward
randomization
network
learner
training
```

The active production config is:

```text
configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml
```

## Production Contract

The maintained learned-policy path requires:

```text
sim.reset_source = csv_initial_states
sim.action_contract = jdot_command
reference.ip.kind = replay_window
reference.boundary.kind = t15_replay_segment_conditioned
observation.actor_kind = controller_state_v6
observation.critic_kind = compact_training_state_v2
reward.kind = tcv_derivative
training.production_mode = true
```

The time base is checked:

```text
reference.duration_s == sim.max_episode_steps * reference.t_step
reference.t_step == tokamak-sim physics.t_step
```

For the final trim50 path this is `0.1 s = 100 * 0.001 s`.

## `sim`

Important active fields:

```yaml
sim:
  config_path: "../../../tokamak-sim/runs/t15md_trim50_plain_gpu_1e6_setup/T15MD_new_data_trim50_plain_gpu_1e6_3856.toml"
  compute_backend: gpu
  gpu_device: cuda:0
  angles: 32
  max_episode_steps: 100
  reset_source: csv_initial_states
  csv_initial_state_library: "../../data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz"
  csv_initial_state_split: train
  action_contract: jdot_command
  terminate_on_boundary_loss: true
  terminate_on_current_limit: true
  current_termination_over_limit_a: 5000.0
  current_termination_grace_steps: 1
  current_hard_termination_fraction: 1.2
```

`jdot_command` means the actor directly requests normalized absolute coil-current
derivative command. The env clips it to derivative limits, computes
`J_next = J_now + dt * Jdot`, and calls `tokamak-sim` with absolute next
currents.

## `reference`

The final training task uses real replay windows, not generated Ip programs:

```yaml
reference:
  duration_s: 0.1
  t_step: 0.001
  theta_count: 32
  ip:
    kind: replay_window
  boundary:
    kind: t15_replay_segment_conditioned
    replay_reference_dir: "../../data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets"
```

The oracle-target builder samples coherent 0.1 s windows from real T15 trim50
shots. Ip and boundary targets come from the same shot/time window. Training
shots are `3856`, `3857`, `3858`, and `3863`; shot `3864` is holdout.

Other reference kinds remain in code for diagnostics and tests, but they are not
the active production training target.

## `observation`

Active observation schema:

```yaml
observation:
  actor_kind: controller_state_v6
  critic_kind: compact_training_state_v2
  target_preview_steps: 10
  target_preview_stride: 10
```

The actor sees controller-available state: Ip, target preview, coil currents,
coil derivatives, fixed-angle boundary radii, boundary targets, previous
applied action, and status scalars.

The critic uses the compact privileged training schema. It does not require full
`psi` in the final high-throughput training path.

## `reward`

The active reward is `tcv_derivative` with the final balanced weights:

```yaml
reward:
  kind: tcv_derivative
  shape_mean_weight: 3.2
  shape_max_weight: 0.8
  ip_weight: 1.8
  current_weight: 0.75
  derivative_weight: 0.1875
  actuator_saturation_weight: 0.1875
  shape_mean_scale_m: 0.03
  shape_max_scale_m: 0.08
  ip_scale_a: 25000.0
  terminal_reward: -20.0
  reward_scale: 0.01
  smoothmax_alpha: -5.0
```

Normal reward is `reward_scale * quality`. Terminal replacement uses the
configured terminal reward with the same reward scale.

## `learner`

The final run uses full-episode sequences:

```yaml
learner:
  unroll_length: 100
  min_replay_sequence_length: 100
  rollout_chunk_length: 100
  updates_per_rollout_chunk: 64
  action_samples: 64
```

## `training`

The canonical 100M training job sets:

```yaml
training:
  steps: 100000000
  num_envs: 2048
  distributed_mode: local_replay
  production_mode: true
  eval_interval_steps: 500000
  checkpoint_interval_steps: 1000000
  milestone_checkpoint_interval_steps: 1000000
  eval_checkpoint_top_k: 10
```

Production configs must be launched through `scripts/train_policy_pipeline.py`;
the plain trainer rejects `production_mode=true`.
