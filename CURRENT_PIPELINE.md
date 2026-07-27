# Current RL Pipeline

```text
T15 6-PFC calibrated (+15%) trim50 CSV data
+ spline_contour boundary mode
+ 0.1 s replay-window episodes from real T15 segments
+ oracle targets from exact T15 current replay
+ learned action contract jdot_command
+ actor observation controller_state_v6
+ critic observation compact_training_state_v2
+ TCV-derivative reward
+ 8-GPU balanced-oracle MPO training
```

## Single source of truth

All configuration flows from one file:
```text
../tokamak-sim/configs/T15MD.toml
```
sigma = 4472135, inductance_L = 2.68e-7, actuator_tau = 0.0, R_wall = 0.0
boundary.mode = spline_contour

No machine config generation. No config rewriting.

## Inputs

```text
../tokamak-sim/data/t15_data_new_trim50_ip_calibrated/
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/
```

Train shots are `3857, 3858, 3863`. Shot `3856` is held out.

## Canonical Server Flow

```bash
cd /scratch/$USER/tokamak/tokamak-sim && git pull --ff-only origin main
cd /scratch/$USER/tokamak/tokamak-rl-v2 && git pull --ff-only origin main

sbatch jobs/build_t15_new_replay_window_oracle_targets_1gpu.sbatch
sbatch jobs/train_t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch
```

## Evaluation

```bash
python3 scripts/evaluate_replay_window_oracle_baselines.py ...
python3 scripts/evaluate_hold_boundary_task.py ...
```
