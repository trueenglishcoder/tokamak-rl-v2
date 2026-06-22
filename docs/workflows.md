# Workflows

This document records the maintained local and Slurm workflows.

## Local Setup

```bash
cd ~/tokamak/tokamak-rl-v2
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 -m pip install -e ../tokamak-sim
```

Or use the simulator environment directly:

```bash
cd ~/tokamak/tokamak-rl-v2
PYTHONPATH=.:../tokamak-sim ../tokamak-sim/.venv/bin/python -m pytest -q tests/test_core_contracts.py
```

## Fast Local Validation

Production cleanup checks:

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

## Non-Production Tiny Smoke

Use a non-production config for tiny local smoke checks:

```bash
cd ~/tokamak/tokamak-rl-v2

PYTHONPATH=.:../tokamak-sim python3 scripts/train_policy_pipeline.py \
  --config configs/experiments/t15_static_boundary.yaml \
  --output-dir /tmp/tokamak_rl_v2_smoke \
  --steps 2 \
  --num-envs 2 \
  --device cpu \
  --sim-compute-backend cpu \
  --batch-size 2 \
  --unroll-length 2 \
  --replay-capacity-episodes 8 \
  --rollout-chunk-length 1 \
  --updates-per-rollout-chunk 1 \
  --eval-episodes 1 \
  --eval-max-steps 5 \
  --eval-interval-steps 1 \
  --controller-rollout-steps 2 \
  --allow-failed-gates \
  --wandb-mode disabled
```

Use this only as a wiring check. It is not a policy-quality run.

## Production Local Run

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

Production notes:

- do not use `scripts/train.py`
- do not pass `--allow-failed-gates`
- do not pass `--skip-controller-rollout-gate`
- `--controller-rollout-steps 0` means full episode

## Push Local Changes

```bash
cd ~/tokamak/tokamak-rl-v2
git status
git add README.md docs/architecture.md docs/artifacts-and-metrics.md docs/configuration.md docs/repository-layout.md docs/workflows.md
git add tokamak_rl_v2 tests
git commit -m "Update production contract"
git push origin main
```

## Pull On The Server

When you are already logged into the Slurm server:

```bash
cd /scratch/$USER/tokamak/tokamak-rl-v2
git pull origin main
```

## Launch The Production Slurm Job

```bash
cd /scratch/$USER/tokamak/tokamak-rl-v2
sbatch jobs/train_t15_csv_segmented_profile_tcvdelta_t15boundary_12gpu_20m.sbatch
```

That job:

1. checks that processed reset, Ip-limit, and replay-boundary artifacts exist,
2. writes a per-job generated config under the output folder,
3. launches production training through `scripts/train_policy_pipeline.py`,
4. uses a fresh W&B project name containing the Slurm job id,
5. logs focused W&B metrics while keeping full disk CSV/JSON outputs.

## Monitor A Slurm Run

Queue state:

```bash
squeue -u $USER
```

Accounting:

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList,Reason -X
```

Logs:

```bash
cd /scratch/$USER/tokamak/tokamak-rl-v2
tail -f slurm_logs/*<jobid>*.out
tail -f slurm_logs/*<jobid>*.err
```

## Inspect The Latest Run

```bash
cd /scratch/$USER/tokamak/tokamak-rl-v2

export OUT=$(ls -td outputs/t15_csv_initial_segmented_profile_boundary_mpo_1gpu_* outputs/t15_csv_initial_segmented_profile_boundary_mpo_* 2>/dev/null | head -n 1)
echo "OUT=$OUT"

python3 - <<'PY'
import json, os, pathlib

out = pathlib.Path(os.environ["OUT"])
p = json.load(open(out / "policy_validation.json"))

print("STATUS:", p.get("status"))
print("CHECKPOINT:", p.get("checkpoint"))
print("EXPORT_DIR:", p.get("export_dir"))
print("ACTOR_EVAL:", p.get("actor_eval"))
print("CONTROLLER:", p.get("controller_rollout"))
print("GATES:")
for gate in p.get("gates", []):
    print(" ", gate.get("name"), gate.get("passed"), gate.get("value"))
PY
```

## Cancel A Job

```bash
scancel <jobid>
```
