# tokamak-rl-v2

Reinforcement-learning training stack for `tokamak-sim`.

The active supported path is the T15 trim50 replay-window policy:

```text
6-PFC T15 trim50 data
+ plain GPU fixed-angle replay references with legacy_precision_index2=1e-6
+ 0.1 s real replay-window episodes
+ jdot_command actor contract
+ TCV-derivative reward
+ feed-forward actor, recurrent critic, MPO, replay
+ deterministic actor export for tokamak-sim
```

See [CURRENT_PIPELINE.md](CURRENT_PIPELINE.md) for the exact contract and file
names.

## Setup

```bash
cd ~/tokamak/tokamak-rl-v2
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 -m pip install -e ../tokamak-sim
```

On the Slurm server, use the prepared container image:

```text
/scratch/$USER/tokamak/tokamak-rl-v2.sqsh
```

## Active Config

```text
configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml
```

This config is intentionally narrow:

- `reference.ip.kind = replay_window`
- `reference.boundary.kind = t15_replay_segment_conditioned`
- `sim.action_contract = jdot_command`
- `observation.actor_kind = controller_state_v6`
- `observation.critic_kind = compact_training_state_v2`
- `reward.kind = tcv_derivative`
- `sim.max_episode_steps = 100`

## Build Oracle Targets

```bash
cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs
sbatch jobs/build_t15_new_replay_window_oracle_targets_1gpu.sbatch
```

This writes:

```text
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz
data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/
```

## Train

```bash
cd /scratch/$USER/tokamak/tokamak-rl-v2
mkdir -p slurm_logs
sbatch jobs/train_t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m.sbatch
```

The job runs one 8-GPU local-replay training run for 100M global env steps and
writes outputs under:

```text
outputs/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_balanced_oracle_8gpu_100m_<jobid>/
```

## Export And Evaluate

Typical outputs:

```text
config_snapshot.json
eval_history.csv
losses.csv
metrics.json
policy_validation.json
exports/best_actor/
checkpoints/
```

Manual export:

```bash
PYTHONPATH=.:../tokamak-sim python3 scripts/export_policy.py \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --out outputs/<run>/exports/manual_best_actor
```

Replay-window evaluation:

```bash
PYTHONPATH=.:../tokamak-sim python3 scripts/evaluate_replay_window_oracle_baselines.py \
  --config outputs/<run>/generated_configs/<run>.json \
  --export-dir outputs/<run>/exports/best_actor \
  --output-dir outputs/<run>/oracle_eval_holdout \
  --all-windows \
  --steps 100 \
  --split holdout \
  --device cuda:0
```

Hold-boundary diagnostics:

```bash
sbatch jobs/eval_hold_boundary_cut900_seg300_8gpu_800x500.sbatch
```

## Tests

Focused checks:

```bash
cd ~/tokamak/tokamak-rl-v2
PYTHONPATH=.:../tokamak-sim python3 -m pytest -q \
  tests/test_production_cleanup.py \
  tests/test_replay_window_oracle_contracts.py \
  tests/test_hold_boundary_eval.py
```

Core contracts:

```bash
PYTHONPATH=.:../tokamak-sim python3 -m pytest -q tests/test_core_contracts.py
```

## Notes

Old delta-Jdot learned-policy sweeps, static-boundary reward searches, 2 s
T15-boundary jobs, 0.2 s antidrift jobs, and 4-PFC training/search launchers have
been retired from the active repo. The 4-PFC data itself remains protected for
future work.
