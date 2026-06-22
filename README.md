# tokamak-rl-v2

`tokamak-rl-v2` is the reinforcement-learning training stack that sits on top
of `tokamak-sim`.

The maintained production path is intentionally narrow:

```text
CSV initial-state resets
+ aggregate CSV reference limits
+ generated segmented_profile Ip targets
+ T15 replay-derived segment-conditioned boundary targets
+ TCV-derivative positive quality reward
+ delta-Jdot actor contract
+ feedforward actor + recurrent critic + MPO + replay
+ deterministic actor export
+ full-episode actor eval + full-episode exported-controller validation
```

This repository does not own the plant physics. It owns the RL environment,
reference generation, reward, replay, learner, training pipeline, export, and
validation logic.

## Production Entry Point

The production config is:

```text
configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml
```

The production entrypoint is:

```text
scripts/train_policy_pipeline.py
```

Production mode is strict:

- it requires `sim.reset_source = csv_initial_states`
- it requires `reference.ip.kind = segmented_profile`
- it requires `reference.boundary.kind = t15_replay_segment_conditioned`
- it requires `reward.kind = tcv_derivative`
- it requires per-coil `delta_derivative_limits_aps`
- it rejects `--allow-failed-gates`
- it rejects controller-rollout bypasses
- it evaluates and validates on the full configured episode horizon

The plain trainer CLI is still available for non-production experiments, but it
refuses `production_mode=true` configs.

## Setup

```bash
cd ~/tokamak/tokamak-rl-v2
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 -m pip install -e ../tokamak-sim
```

If you already use the simulator virtual environment, you can run tests through
`../tokamak-sim/.venv/bin/python`.

## Fast Local Checks

```bash
cd ~/tokamak/tokamak-rl-v2
PYTHONPATH=.:../tokamak-sim ../tokamak-sim/.venv/bin/python -m pytest -q \
  tests/test_production_cleanup.py \
  tests/test_t15_csv_initial_states.py \
  tests/test_t15_reference_limits.py
```

Core contracts:

```bash
cd ~/tokamak/tokamak-rl-v2
PYTHONPATH=.:../tokamak-sim ../tokamak-sim/.venv/bin/python -m pytest -q tests/test_core_contracts.py
```

## Production Training

Local launch:

```bash
cd ~/tokamak/tokamak-rl-v2

PYTHONPATH=.:../tokamak-sim python3 scripts/train_policy_pipeline.py \
  --config configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml \
  --output-dir outputs/t15_csv_initial_segmented_profile_boundary_mpo_local \
  --steps 1000000 \
  --num-envs 256 \
  --device cuda:0 \
  --sim-compute-backend gpu \
  --sim-gpu-device cuda:0 \
  --wandb \
  --wandb-project tokamak-rl-v2-local \
  --wandb-name t15_csv_initial_segmented_profile_boundary_mpo_local
```

Slurm launch:

```bash
cd /scratch/$USER/tokamak/tokamak-rl-v2
sbatch jobs/train_t15_csv_segmented_profile_tcvdelta_t15boundary_12gpu_20m.sbatch
```

That job writes to:

```text
outputs/t15_csv_segmented_profile_tcvdelta_t15boundary_12gpu_20m_<jobid>
slurm_logs/tokamak-rl-v2-tcvd-t15boundary-12gpu-20m-<jobid>.out
slurm_logs/tokamak-rl-v2-tcvd-t15boundary-12gpu-20m-<jobid>.err
```

## Production Runtime Contract

At runtime, training reads only processed artifacts:

```text
data/processed/t15_csv_initial_states.npz
data/processed/t15_reference_limits.json
../tokamak-sim/runs/t15md_limited_replay_dataset/
```

The environment does not read raw T15 CSV traces during rollout. Raw CSVs are
used only by offline builders and replay runs that produce the processed reset,
Ip-limit, and boundary-reference artifacts.

Boundary targets come from smoothed T15 replay boundary segments matched by shot
id and reset time, then shifted so step 0 equals the reset boundary. Ip targets
are generated online as bounded `segmented_profile` programs that start at reset
Ip, stay positive, stay inside aggregate limits, and obey signed ramp-rate
limits.

## Outputs

A normal production run writes:

```text
config_snapshot.json
losses.csv
reward_components.csv
eval_history.csv
metrics.json
policy_validation.json
closed_loop_rollout_report.json
exports/best_actor/
exports/final_actor/
checkpoints/                 optional
```

The final decision is in `policy_validation.json`. Physical success is based on
full-episode actor behavior and full-episode exported-controller behavior, not
on MPO diagnostic thresholds.

## Manual Export

Manual checkpoint export is supported through:

```bash
PYTHONPATH=.:../tokamak-sim python3 scripts/export_policy.py \
  --checkpoint outputs/run/checkpoints/best.pt \
  --out outputs/run/exports/manual_best_actor
```

Manual export accepts only checkpoints with the production actor schema:

```text
controller_state_v3
```

## More Detail

- [Configuration](docs/configuration.md)
- [Workflows](docs/workflows.md)
- [Architecture](docs/architecture.md)
- [Repository Layout](docs/repository-layout.md)
- [Artifacts And Metrics](docs/artifacts-and-metrics.md)
