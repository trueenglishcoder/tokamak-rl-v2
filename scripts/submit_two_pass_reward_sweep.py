#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _job_id(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise RuntimeError("sbatch returned an empty job id")
    return text.split(";", 1)[0]


def _submit(args: list[str]) -> str:
    result = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return _job_id(result.stdout)


def submit_chain(
    *,
    pass1_job: Path,
    pass1_aggregate_job: Path,
    pass2_job: Path,
    final_aggregate_job: Path,
) -> dict[str, Any]:
    Path("slurm_logs").mkdir(parents=True, exist_ok=True)
    pass1_jobid = _submit(["sbatch", "--parsable", str(pass1_job)])
    root = f"outputs/t15_reward_sweep288_legal_1m_{pass1_jobid}"
    center_json = f"{root}/selection/pass1/physical_best_candidate.json"

    pass1_aggregate_jobid = _submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{pass1_jobid}",
            f"--export=ALL,PASS1_JOBID={pass1_jobid},ROOT={root}",
            str(pass1_aggregate_job),
        ]
    )
    pass2_jobid = _submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{pass1_aggregate_jobid}",
            f"--export=ALL,PASS1_ROOT={root},CENTER_JSON={center_json}",
            str(pass2_job),
        ]
    )
    final_aggregate_jobid = _submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{pass2_jobid}",
            f"--export=ALL,PASS1_JOBID={pass1_jobid},PASS2_JOBID={pass2_jobid},ROOT={root}",
            str(final_aggregate_job),
        ]
    )

    payload = {
        "root": root,
        "center_json": center_json,
        "pass1_jobid": pass1_jobid,
        "pass1_aggregate_jobid": pass1_aggregate_jobid,
        "pass2_jobid": pass2_jobid,
        "final_aggregate_jobid": final_aggregate_jobid,
        "jobs": {
            "pass1": str(pass1_job),
            "pass1_aggregate": str(pass1_aggregate_job),
            "pass2": str(pass2_job),
            "final_aggregate": str(final_aggregate_job),
        },
    }

    out = Path(root) / "selection" / "submission_chain.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit the full two-pass legal-actuator reward sweep dependency chain.")
    parser.add_argument("--pass1-job", type=Path, default=Path("jobs/sweep_t15_csv_segmented_profile_rewards_96gpu_pass1_broad.sbatch"))
    parser.add_argument("--pass1-aggregate-job", type=Path, default=Path("jobs/aggregate_t15_reward_sweep_pass1.sbatch"))
    parser.add_argument("--pass2-job", type=Path, default=Path("jobs/sweep_t15_csv_segmented_profile_rewards_96gpu_pass2_focused.sbatch"))
    parser.add_argument("--final-aggregate-job", type=Path, default=Path("jobs/aggregate_t15_reward_sweep_final.sbatch"))
    args = parser.parse_args(argv)
    payload = submit_chain(
        pass1_job=args.pass1_job,
        pass1_aggregate_job=args.pass1_aggregate_job,
        pass2_job=args.pass2_job,
        final_aggregate_job=args.final_aggregate_job,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
