#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_reward_sweep_manifest import PROFILE_TCV_QUALITY
    from scripts.submit_two_pass_reward_sweep import submit_chain
except ModuleNotFoundError:  # pragma: no cover - used when run as python3 scripts/...
    from build_reward_sweep_manifest import PROFILE_TCV_QUALITY
    from submit_two_pass_reward_sweep import submit_chain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit a 36+12 TCV-style quality reward sweep.")
    parser.add_argument(
        "--pass1-job",
        type=Path,
        default=Path("jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_tcv_quality_pass1.sbatch"),
    )
    parser.add_argument(
        "--pass1-aggregate-job",
        type=Path,
        default=Path("jobs/aggregate_t15_reward_sweep_pass1.sbatch"),
    )
    parser.add_argument(
        "--pass2-job",
        type=Path,
        default=Path("jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_tcv_quality_pass2.sbatch"),
    )
    parser.add_argument(
        "--final-aggregate-job",
        type=Path,
        default=Path("jobs/aggregate_t15_reward_sweep_final.sbatch"),
    )
    parser.add_argument("--root-prefix", default="outputs/t15_reward_sweep48_tcv_quality_half_slope_2m5m")
    args = parser.parse_args(argv)
    payload = submit_chain(
        pass1_job=args.pass1_job,
        pass1_aggregate_job=args.pass1_aggregate_job,
        pass2_job=args.pass2_job,
        final_aggregate_job=args.final_aggregate_job,
        root_prefix=args.root_prefix,
        profile=PROFILE_TCV_QUALITY,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
