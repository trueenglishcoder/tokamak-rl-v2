# Target Library Comparison

## Libraries

- Reference: `feasible` rows=12000, steps=100, schema=`t15_feasible_generated_trim50_idealized_targets_v1`
- Candidate: `simple` rows=12000, steps=100, schema=`t15_simple_manifold_generated_trim50_idealized_targets_v1`

## High-Signal Metrics

| metric | reference mean | reference p90 | candidate mean | candidate p90 | p90 ratio |
|---|---:|---:|---:|---:|---:|
| `endpoint_abs_dip_a` | 28109.4315 | 59048.2094 | 13324.8063 | 53935.4906 | 0.9134 |
| `endpoint_mean_abs_dradii_m` | 0.0215 | 0.0320 | 0.0174 | 0.0289 | 0.9047 |
| `ip_rate_abs_max_aps` | 3.381e+05 | 6.255e+05 | 1.829e+05 | 5.721e+05 | 0.9146 |
| `boundary_step_mean_abs_m_max` | 0.0021 | 0.0081 | 0.0002765 | 0.000584 | 0.0721 |
| `boundary_step_any_angle_abs_m_max` | 0.0031 | 0.0114 | 0.000353 | 0.0007337 | 0.0643 |
| `current_usage_fraction_max` |  |  | 0.6407 | 0.7107 |  |
| `action_rms` |  |  | 0.1121 | 0.2594 |  |
| `action_step_jump_abs_max` |  |  | 0.0284 | 0.0824 |  |
| `jdot_rms_aps` |  |  | 1.985e+06 | 3.819e+06 |  |
| `jdot_step_jump_abs_max_aps` |  |  | 5.529e+05 | 1.692e+06 |  |
| `nearest_reference_window_distance` |  |  | 0.0767 | 0.2235 |  |

## Largest Distribution Changes

| metric | candidate/reference p90 | reference p90 | candidate p90 |
|---|---:|---:|---:|
| `R0_step_abs_max` | 0 | 0.0014 | 0 |
| `Z0_step_abs_max` | 0 | 0.0004812 | 0 |
| `endpoint_abs_dR0` | 0 | 0.0137 | 0 |
| `endpoint_abs_dZ0` | 0 | 0.0042 | 0 |
| `kappa_step_abs_max` | 0.0415 | 0.0064 | 0.000265 |
| `boundary_step_any_angle_abs_m_max` | 0.0643 | 0.0114 | 0.0007337 |
| `delta_step_abs_max` | 0.0706 | 0.0039 | 0.0002777 |
| `boundary_step_mean_abs_m_max` | 0.0721 | 0.0081 | 0.000584 |
| `A0_step_abs_max` | 0.0921 | 0.0051 | 0.0004678 |
| `endpoint_abs_dkappa` | 0.4209 | 0.0411 | 0.0173 |
| `endpoint_abs_ddelta` | 0.7692 | 0.0196 | 0.0150 |
| `endpoint_max_abs_dradii_m` | 0.8350 | 0.0439 | 0.0367 |
| `boundary_step_mean_abs_m_mean` | 0.8703 | 0.0003324 | 0.0002893 |
| `boundary_mean_radius_range_m` | 0.8924 | 0.0324 | 0.0289 |
| `endpoint_mean_abs_dradii_m` | 0.9047 | 0.0320 | 0.0289 |
| `endpoint_abs_dA0` | 0.9101 | 0.0253 | 0.0230 |
| `endpoint_signed_dip_a` | 0.9112 | 58700.6797 | 53487.1031 |
| `endpoint_abs_dip_a` | 0.9134 | 59048.2094 | 53935.4906 |
| `ip_rate_abs_max_aps` | 0.9146 | 6.255e+05 | 5.721e+05 |
| `ip_step_abs_max_a` | 0.9146 | 625.5312 | 572.1344 |

## Notes

- `nearest_reference_window_distance` is computed from endpoint `Ip` and all 32 endpoint boundary-radii deltas, normalized by the reference p99 scales.
- For oracle replay-window libraries, currents are reconstructed from initial currents plus stored normalized `real_jdot_action`.
- For generated libraries with `coil_witness`, current and Jdot metrics describe the witness/open-loop trajectory used to generate the target, not necessarily what the learned policy did.
- This is a data-distribution audit. It does not run LQR or closed-loop policy simulations.
