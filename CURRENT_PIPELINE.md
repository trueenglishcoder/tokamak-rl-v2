# Current RL Pipeline

```text
T15 6-PFC calibrated Ip +15% trim50 data
+ suchkov_spline_contour boundary mode
+ 0.1 s replay-window episodes
+ oracle targets from exact T15 current replay
+ actor observation controller_state_v6
+ critic observation compact_training_state_v2
+ TCV-derivative reward
+ 8-GPU balanced-oracle MPO training for 50M steps
```

## Baseline and Experiment

The successful 100M baseline is retained unchanged. It used
`legacy_contour_limited`, 2048 environments, 8 GPUs, and the reward weights now
copied into the new experiment.

The active experiment changes only:

```text
Ip source data: calibrated +15%
boundary mode: suchkov_spline_contour
training length: 50M environment steps
```

## Inputs and Split

```text
../tokamak-sim/data/t15_data_new_trim50_ip_calibrated/
../tokamak-sim/runs/t15md_trim50_ip15_suchkov_top5_replay/
data/processed/t15_ip15_suchkov_replay_window_0p1s_oracle_initial_states.npz
data/processed/t15_ip15_suchkov_replay_window_0p1s_oracle_targets/
```

Train shots are `3856`, `3857`, `3858`, and `3863`. Shot `3864` is holdout.

## Server Flow

```bash
cd /scratch/$USER/tokamak/tokamak-sim
git pull --ff-only origin main

cd /scratch/$USER/tokamak/tokamak-rl-v2
git pull --ff-only origin main

sbatch jobs/build_t15_ip15_suchkov_replay_window_oracle_targets_1gpu.sbatch
sbatch jobs/train_t15_ip15_suchkov_replay_window_tcvjdot_8gpu_50m.sbatch
```
