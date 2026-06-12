# tokamak-rl-v2

Magnetic-control training stack for `tokamak-sim` using the published TCV training structure where it applies to this simpler plant: feedforward stochastic actor, recurrent Q critic, Maximum a Posteriori Policy Optimisation, episode-aware sequence replay, target-reference observations, dense physical rewards, deterministic mean-policy export, and actor/learner execution.

The first production objective is T15 initial-boundary hold with segmented Ip tracking from replay-bounded handover-like initial conditions. The actor observation schema is `joint_state_v1`: current/target Ip, active currents and derivatives, full psi grid, reconstructed boundary radii, target radii, boundary-radii error, boundary-found flag, and configurable target preview.

## TCV-style Training Path

The maintained path is a fixed-objective actor-critic pipeline:

- many `tokamak-sim` environment copies collect closed-loop rollout trajectories;
- trajectories are stored in episode-aware sequence replay;
- a recurrent Q critic is trained from observed rewards and next-state targets;
- a feedforward Gaussian actor is improved with MPO using critic-scored action samples;
- deterministic evaluation uses the actor mean action on fixed seeds;
- only the compact actor is exported for `tokamak-sim` controller rollout validation.

The reward is part of the fixed training objective. It is treated as a physical cost signal for shape error, Ip tracking, current limits, derivative usage, action magnitude, and action smoothness.

## Local Smoke

```bash
python scripts/train.py   --config configs/experiments/t15_static_boundary.yaml   --steps 1000   --num-envs 8   --device auto
```

For the paper-like policy path, use the gated pipeline. It runs reset sanity, no-control baseline, training, deterministic actor evaluation, MPO health gates, export validation, and a `LearnedMagneticController` rollout check:

```bash
python scripts/train_policy_pipeline.py \
  --config configs/experiments/t15_hold_reset_boundary_ip_stage1_gpu.yaml \
  --steps 10000 \
  --num-envs 16 \
  --device cuda:0 \
  --sim-compute-backend gpu \
  --sim-gpu-device cuda:0
```

## Maintained Server Jobs

Use these jobs on the Slurm/enroot server:

```bash
sbatch jobs/train_t15_hold_reset_boundary_policy_8gpu.sbatch
sbatch jobs/train_t15_static_boundary.sbatch
```

The repository should contain only fixed-objective training, evaluation, export, and controller-validation paths.

Previous checkpoints produced before the MPO learner repair are intentionally incompatible and should not be resumed.
