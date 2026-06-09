# tokamak-rl-v2

Fresh magnetic-control training stack for `tokamak-sim`. The simulator plant is simpler than the published TCV FGE plant, but the training machinery follows the published magnetic-control recipe: feedforward stochastic actor, recurrent Q critic, Maximum a Posteriori Policy Optimisation, FIFO sequence replay, hidden domain randomization, target-reference observations, quality-transform rewards, deterministic mean-policy export, and actor/learner execution.

The default production objective is T15 static-boundary control from replay-bounded handover-like initial conditions.

```bash
python scripts/train.py --config configs/experiments/t15_static_boundary.yaml --steps 1000 --num-envs 8 --device auto
```
