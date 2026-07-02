# Target Library Comparison

## Libraries

- Reference: `feasible` rows=12000, steps=100, schema=`t15_feasible_generated_trim50_idealized_targets_v1`
- Candidate: `simple (mode=ramp)` rows=2400, steps=100, schema=`t15_simple_manifold_generated_trim50_idealized_targets_v1`

## High-Signal Metrics

| metric | reference mean | reference p90 | candidate mean | candidate p90 | p90 ratio |
|---|---:|---:|---:|---:|---:|
| `endpoint_abs_dip_a` | 28109.4315 | 59048.2094 | 19840.5311 | 55841.3906 | 0.9457 |
| `endpoint_mean_abs_dradii_m` | 0.0215 | 0.0320 | 0.0214 | 0.0293 | 0.9161 |
| `ip_rate_abs_max_aps` | 3.381e+05 | 6.255e+05 | 1.984e+05 | 5.584e+05 | 0.8927 |
| `boundary_step_mean_abs_m_max` | 0.0021 | 0.0081 | 0.000215 | 0.0002944 | 0.0364 |
| `boundary_step_any_angle_abs_m_max` | 0.0031 | 0.0114 | 0.0002789 | 0.0003845 | 0.0337 |
| `current_usage_fraction_max` |  |  | 0.6475 | 0.7107 |  |
| `action_rms` |  |  | 0.1305 | 0.2602 |  |
| `action_step_jump_abs_max` |  |  | 2.904e-05 | 4.773e-05 |  |
| `jdot_rms_aps` |  |  | 2.218e+06 | 3.747e+06 |  |
| `jdot_step_jump_abs_max_aps` |  |  | 594.6354 | 1000.0000 |  |
| `nearest_reference_window_distance` |  |  | 0.0404 | 0.0677 |  |

## Largest Distribution Changes

| metric | candidate/reference p90 | reference p90 | candidate p90 |
|---|---:|---:|---:|
| `R0_step_abs_max` | 0 | 0.0014 | 0 |
| `Z0_step_abs_max` | 0 | 0.0004812 | 0 |
| `endpoint_abs_dR0` | 0 | 0.0137 | 0 |
| `endpoint_abs_dZ0` | 0 | 0.0042 | 0 |
| `boundary_step_any_angle_abs_m_max` | 0.0337 | 0.0114 | 0.0003845 |
| `boundary_step_mean_abs_m_max` | 0.0364 | 0.0081 | 0.0002944 |
| `kappa_step_abs_max` | 0.0387 | 0.0064 | 0.0002472 |
| `delta_step_abs_max` | 0.0399 | 0.0039 | 0.000157 |
| `A0_step_abs_max` | 0.0459 | 0.0051 | 0.0002331 |
| `endpoint_abs_dkappa` | 0.6018 | 0.0411 | 0.0247 |
| `endpoint_abs_ddelta` | 0.8024 | 0.0196 | 0.0157 |
| `endpoint_max_abs_dradii_m` | 0.8662 | 0.0439 | 0.0381 |
| `boundary_step_mean_abs_m_mean` | 0.8813 | 0.0003324 | 0.000293 |
| `ip_rate_abs_max_aps` | 0.8927 | 6.255e+05 | 5.584e+05 |
| `ip_step_abs_max_a` | 0.8927 | 625.5312 | 558.4188 |
| `boundary_mean_radius_range_m` | 0.9037 | 0.0324 | 0.0293 |
| `endpoint_mean_abs_dradii_m` | 0.9161 | 0.0320 | 0.0293 |
| `endpoint_abs_dA0` | 0.9215 | 0.0253 | 0.0233 |
| `endpoint_signed_dip_a` | 0.9440 | 58700.6797 | 55411.3312 |
| `endpoint_abs_dip_a` | 0.9457 | 59048.2094 | 55841.3906 |

## Notes

- `nearest_reference_window_distance` is computed from endpoint `Ip` and all 32 endpoint boundary-radii deltas, normalized by the reference p99 scales.
- For oracle replay-window libraries, currents are reconstructed from initial currents plus stored normalized `real_jdot_action`.
- For generated libraries with `coil_witness`, current and Jdot metrics describe the witness/open-loop trajectory used to generate the target, not necessarily what the learned policy did.
- This is a data-distribution audit. It does not run LQR or closed-loop policy simulations.
