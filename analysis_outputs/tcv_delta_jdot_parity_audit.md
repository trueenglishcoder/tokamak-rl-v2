# Delta-Jdot TCV-Parity Audit

Date: 2026-06-20

## Scope

Read pass completed over tracked project files before repair:

- `tokamak-rl-v2`: all tracked files in the repository.
- `tokamak-sim`: controller/export/runtime files affecting learned-controller parity:
  - `tokamak_control/control/learned_magnetic_controller.py`
  - `tokamak_control/control/registry.py`
  - `tokamak_control/cli/run_simulation.py`
  - `tokamak_control/control/base.py`
  - `tests/test_learned_magnetic_controller.py`

Audit count: 117 files, 30520 lines.

## Issues Found And Changed

| Area | Audit result | Repair |
| --- | --- | --- |
| RL delta-Jdot scaling | Issue: `delta_jdot` used one scalar `delta_derivative_scale_aps=500000`, so all coils had the same allowed derivative-command increment. | Added `sim.delta_derivative_limits_aps` with per-coil PFC/SOL limits derived from real T15 data plus 20%. |
| Action contract naming | Issue: scalar and per-coil delta-Jdot exports could share an action contract name. | New per-coil exports use `delta_jdot_derivative_command_v2`; scalar fallback remains legacy `delta_jdot_derivative_command_v1`. |
| Replay / critic action | No issue after prior delta-Jdot repair: replay stores `BatchStep.requested_action`, which is the actor-controlled requested normalized delta. | Left this semantics unchanged and pinned by tests. |
| Plant command | No issue in accumulation logic: requested delta accumulates into normalized derivative command and clips to derivative-command limits before stepping the plant. | Changed scaling source from scalar to per-coil limits. |
| TCV derivative reward actuator term | Issue: the derivative/voltage-equivalent component could be interpreted as accumulated derivative command rather than realized delta-Jdot. | `tcv_derivative` now uses realized normalized delta-Jdot (`applied_delta_action`) for the derivative actuator component. |
| TCV terminal semantics | No issue in current implementation: terminal step returns terminal reward replacement for `tcv_derivative`. | Left unchanged and covered by tests. |
| Current-aware projection | No issue for source-locked TCV mode: loader rejects `tcv_derivative` unless `current_saturation_fraction=1.0`, so no hidden current projection is active. | Left legacy saturation available only outside `tcv_derivative`. |
| Exported controller | Issue: exported controller only had scalar delta scale for delta-Jdot bundles. | Added `delta_derivative_limits_aps` support for v2 bundles and kept v1 scalar compatibility for old bundles. |
| Training jobs / sweep manifests | Issue: TCV delta searches and selected long-run jobs still generated scalar-only delta settings. | Added per-coil delta-Jdot limits to TCV derivative manifests and TCV delta job-generated configs. |
| Actuator legality analysis | Issue: real-data analysis reported current and derivative limits but not change-in-derivative limits. | Added `Jdot` and `delta_Jdot` analysis with `1.2 * max(abs(delta_Jdot))` recommendations. |

## Real T15 Delta-Jdot Limits

Computed from `tokamak-sim/data/t15_data_new/coils/t15md_*_coils.csv`, filtering intervals with `dt < 0.5 ms`, using `Jdot = dI/dt`, then `delta_Jdot = Jdot[t] - Jdot[t-1]`, and finally multiplying by `1.2`.

```text
PFC: [163347.0, 310755.0, 87838.08, 153214.2, 404364.0, 1191036.96]
SOL: [1437338.8, 5889842.0, 1946208.8]
```

## Files Changed In This Pass

- `configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml`
- `jobs/train_t15_csv_segmented_profile_f002_tcv_delta_12gpu_20m.sbatch`
- `jobs/train_t15_csv_segmented_profile_tcvdelta_balanced_12gpu_50m.sbatch`
- `scripts/analyze_t15_actuator_legality.py`
- `scripts/build_reward_sweep_manifest.py`
- `tests/test_core_contracts.py`
- `tests/test_reward_sweep.py`
- `tokamak_rl_v2/config/loader.py`
- `tokamak_rl_v2/config/schema.py`
- `tokamak_rl_v2/env/batch_env.py`
- `tokamak_rl_v2/rewards/physical.py`
- `../tokamak-sim/tokamak_control/control/learned_magnetic_controller.py`
- `../tokamak-sim/tests/test_learned_magnetic_controller.py`

## Remaining Intentional Legacy Paths

- Non-TCV `physical_cost` and saturation modes still exist for old experiments and diagnostics.
- `delta_jdot_derivative_command_v1` remains readable for old scalar-delta exports.
- `delta_derivative_scale_aps` remains as a legacy fallback, but active TCV derivative profiles now emit `delta_derivative_limits_aps`.
