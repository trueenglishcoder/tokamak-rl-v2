# tokamak-rl-v2

Magnetic-control training stack for `tokamak-sim` using the published TCV training structure where it applies to this simpler plant: feedforward stochastic actor, recurrent Q critic, Maximum a Posteriori Policy Optimisation, episode-aware sequence replay, target-reference observations, quality-transform rewards, deterministic mean-policy export, and actor/learner execution.

The default production objective is T15 static-boundary control from replay-bounded handover-like initial conditions. The actor observation schema is `joint_state_v1`: current/target Ip, active currents and derivatives, full psi grid, reconstructed boundary radii, target radii, boundary-radii error, boundary-found flag, and configurable target preview.

## Local Smoke

```bash
python scripts/train.py   --config configs/experiments/t15_static_boundary.yaml   --steps 1000   --num-envs 8   --device auto
```

## Maintained Server Jobs

Use these jobs on the Slurm/enroot server:

```bash
sbatch jobs/search_rewards_t15_static_boundary_control_discovery_8gpu.sbatch
sbatch jobs/train_t15_static_boundary.sbatch
```

Older reward-search and candidate-specific job files are intentionally disabled so stale narrow searches cannot be launched accidentally.

## Search Gate

Reward search is not trusted unless candidate logs show all of these:

- `tail100.policy_weight_max` above the uniform-action baseline.
- finite, nonzero `tail100.sampled_q_spread`.
- finite, nonzero `tail100.actor_param_delta_norm`.
- physical evaluation metrics: boundary found, shape error, Ip error, current-limit violation, action RMS, and no-control improvement.

Previous checkpoints produced before the MPO learner repair are intentionally incompatible and should not be resumed.
