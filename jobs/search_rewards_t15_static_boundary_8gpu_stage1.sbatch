#!/bin/bash
set -euo pipefail
cat >&2 <<'MSG'
This job is obsolete and intentionally disabled.

Use one of the maintained jobs instead:
  sbatch jobs/search_rewards_t15_static_boundary_control_discovery_8gpu.sbatch
  sbatch jobs/train_t15_static_boundary.sbatch

Disabled file: jobs/search_rewards_t15_static_boundary_8gpu_stage1.sbatch
MSG
exit 2
