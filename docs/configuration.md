# Configuration

Experiment files live under `configs/experiments/` and are loaded by
`tokamak_rl_v2.config.loader.load_experiment_config`.

Parsing is strict. Unknown keys are rejected at the top level and in the main
sections:

- `sim`
- `reference`
- `reference.ip`
- `reference.boundary`
- `observation`
- `randomization`
- `network`
- `learner`
- `training`

The production config is:

```text
configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml
```

## Top-Level Shape

```yaml
name: t15_example

sim: {}
reference: {}
observation: {}
reward: {}
randomization: {}
network: {}
learner: {}
training: {}
```

Legacy level-based advancement blocks are not supported.

## Production Contract

When `training.production_mode: true`, the loader requires:

```yaml
sim:
  reset_source: csv_initial_states
  csv_initial_state_split: train

reference:
  boundary:
    kind: hold_reset_boundary
  ip:
    kind: segmented_profile

training:
  production_mode: true
```

Production also enforces:

```text
reference.duration_s == sim.max_episode_steps * reference.t_step
```

and runtime enforces:

```text
reference.t_step == tokamak-sim physics.t_step
```

## `sim`

The `sim` section configures the plant wrapper.

Important fields:

```yaml
sim:
  config_path: "../../../tokamak-sim/configs/T15MD_new_data.toml"
  initial_currents_path: null
  compute_backend: gpu
  gpu_device: cuda:0
  angles: 32
  max_episode_steps: 500
  reset_source: csv_initial_states
  csv_initial_state_library: "../../data/processed/t15_csv_initial_states.npz"
  csv_initial_state_split: train
  current_limit_scale: 1.0
  derivative_limit_scale: 1.0
  action_scale: 1.0
  terminate_on_boundary_loss: true
  terminate_on_current_limit: false
  current_termination_over_limit_a: 5000.0
  current_termination_grace_steps: 8
  current_hard_termination_fraction: 1.05
```

`reset_source` supports:

- `initial_ranges`
- `csv_initial_states`

Production uses only `csv_initial_states`.

### Current Safety Limits

`current_safety_limits` provides the coil-current magnitudes used for:

- current normalization
- current usage and margin diagnostics
- current-over-limit diagnostics

There is no hidden projection layer in the maintained path.

## `reference`

The production reference path is:

```yaml
reference:
  duration_s: 0.5
  t_step: 0.001
  theta_count: 32
  seed: 11
  ip:
    kind: segmented_profile
    limits_path: "../../data/processed/t15_reference_limits.json"
    start_mode: reset_ip
    segment_min_steps: 50
    segment_max_steps: 300
    segment_count_min: 3
    segment_count_max: 8
    hold_probability: 0.35
    plateau_min_fraction: 0.25
    plateau_max_fraction: 1.0
    end_min_fraction: 0.25
    end_max_fraction: 1.0
    ramp_up_rate_fraction: 0.25
    ramp_down_rate_fraction: 0.25
    hold_min_steps: 50
    hold_max_steps: 250
    final_hold_min_steps: 50
    smooth_ramps: true
    max_delta_fraction: 0.60
  boundary:
    kind: hold_reset_boundary
```

### `reference.ip.kind`

Supported kinds in the current code are:

- `segmented`
- `hold_reset`
- `segmented_profile`

The production kind is `segmented_profile`.

`segmented_profile`:

- starts at reset Ip
- stays strictly positive
- stays inside aggregate real-data bounds
- obeys separate positive and negative ramp-rate limits
- contains at least one hold segment
- contains at least one nonzero ramp

`smooth_ramps` is the production-facing smoothing switch:

- `false`: linear ramps
- `true`: cosine-eased monotone ramps

The old seconds-based smoothing field is rejected.

## `observation`

The maintained observation kinds are:

```yaml
observation:
  actor_kind: controller_state_v2
  critic_kind: privileged_training_state_v1
  target_preview_steps: 8
  target_preview_stride: 10
```

Production actor observations exclude `psi_flat`. Production critic
observations include normalized `psi_flat` plus current/derivative privilege.

## `reward`

Production uses one dense negative physical-cost reward only.

Important fields:

```yaml
reward:
  shape_mean_scale_m: 0.03
  shape_max_scale_m: 0.08
  ip_scale_a: 25000.0
  boundary_missing_error_m: 0.10
  shape_mean_weight: 4.0
  shape_max_weight: 1.0
  ip_weight: 3.0
  current_weight: 2.0
  derivative_weight: 0.5
  action_weight: 0.02
  delta_action_weight: 0.05
  current_soft_fraction: 0.90
  derivative_soft_fraction: 0.90
  terminal_reward: -20.0
  reward_scale: 1.0
```

Removed legacy reward keys are rejected.

## `learner`

The learner section controls replay and MPO settings:

```yaml
learner:
  discount: 0.99
  unroll_length: 64
  batch_size: 256
  replay_capacity_episodes: 4096
  action_samples: 20
  min_replay_sequence_length: 64
  actor_update_chunk_size: 2048
  rollout_chunk_length: 64
  updates_per_rollout_chunk: 32
```

## `training`

Important fields:

```yaml
training:
  steps: 20000000
  num_envs: 256
  device: auto
  output_dir: "../../outputs/t15_csv_initial_segmented_profile_boundary_mpo"
  save_checkpoints: true
  checkpoint_interval_steps: 500000
  eval_interval_steps: 100000
  eval_episodes: 128
  eval_max_steps: 500
  actor_workers: 1
  distributed_mode: local_replay
  production_mode: true
```

Production runs must go through `scripts/train_policy_pipeline.py`. The plain
trainer CLI rejects `production_mode=true`.
