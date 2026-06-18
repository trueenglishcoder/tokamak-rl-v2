#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

try:
    from scripts.build_reward_sweep_manifest import PROFILE_CURRENT_CONSTRAINT
    from scripts.submit_two_pass_reward_sweep import submit_chain
except ModuleNotFoundError:  # pragma: no cover - used when run as python3 scripts/...
    from build_reward_sweep_manifest import PROFILE_CURRENT_CONSTRAINT
    from submit_two_pass_reward_sweep import submit_chain


def main() -> int:
    payload = submit_chain(
        pass1_job=Path("jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_current_constraint_pass1.sbatch"),
        pass1_aggregate_job=Path("jobs/aggregate_t15_reward_sweep_pass1.sbatch"),
        pass2_job=Path("jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_current_constraint_pass2.sbatch"),
        final_aggregate_job=Path("jobs/aggregate_t15_reward_sweep_final.sbatch"),
        root_prefix="outputs/t15_reward_sweep72_current_constraint_1m",
        profile=PROFILE_CURRENT_CONSTRAINT,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
