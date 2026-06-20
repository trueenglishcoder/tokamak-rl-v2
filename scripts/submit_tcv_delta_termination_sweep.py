#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from scripts.build_reward_sweep_manifest import PROFILE_TCV_DELTA_TERMINATION_F002, build_manifest
except ModuleNotFoundError:  # pragma: no cover - used when run as python3 scripts/...
    from build_reward_sweep_manifest import PROFILE_TCV_DELTA_TERMINATION_F002, build_manifest


def _job_id(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise RuntimeError("sbatch returned an empty job id")
    return text.split(";", 1)[0]


def _submit(args: list[str]) -> str:
    result = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return _job_id(result.stdout)


def _run(args: list[str]) -> None:
    subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_manifest(root: str) -> str:
    manifest_path = Path(root) / "variants.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        "broad",
        profile=PROFILE_TCV_DELTA_TERMINATION_F002,
        runs_per_array_task=1,
        array_task_count=12,
    )
    if int(manifest["variant_count"]) != 12:
        raise RuntimeError(f"termination manifest must contain 12 variants, got {manifest['variant_count']}")
    folders = [str(variant["folder"]) for variant in manifest["variants"]]
    if folders[0] != "t000_f110_g001" or folders[-1] != "t011_f130_g025":
        raise RuntimeError(f"unexpected termination variant folder range: {folders[0]} ... {folders[-1]}")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(manifest_path)


def submit_termination_sweep(
    *,
    sweep_job: Path,
    aggregate_job: Path,
    root_prefix: str,
) -> dict[str, Any]:
    Path("slurm_logs").mkdir(parents=True, exist_ok=True)
    sweep_jobid = _submit(["sbatch", "--parsable", "--hold", f"--export=ALL,SWEEP_ROOT_PREFIX={root_prefix}", str(sweep_job)])
    root = f"{root_prefix}_{sweep_jobid}"
    manifest = _write_manifest(root)
    aggregate_jobid = _submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{sweep_jobid}",
            f"--export=ALL,ROOT={root}",
            str(aggregate_job),
        ]
    )
    payload = {
        "root": root,
        "profile": PROFILE_TCV_DELTA_TERMINATION_F002,
        "manifest": manifest,
        "sweep_jobid": sweep_jobid,
        "aggregate_jobid": aggregate_jobid,
        "jobs": {
            "sweep": str(sweep_job),
            "aggregate": str(aggregate_job),
        },
    }
    out = Path(root) / "selection" / "submission_chain.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _run(["scontrol", "release", sweep_jobid])
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit a 12-run f002 TCV delta-Jdot current-termination sweep.")
    parser.add_argument(
        "--sweep-job",
        type=Path,
        default=Path("jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_tcv_delta_termination_f002.sbatch"),
    )
    parser.add_argument(
        "--aggregate-job",
        type=Path,
        default=Path("jobs/aggregate_t15_reward_sweep_onepass.sbatch"),
    )
    parser.add_argument("--root-prefix", default="outputs/t15_reward_sweep12_tcv_delta_termination_f002_5m")
    args = parser.parse_args(argv)
    payload = submit_termination_sweep(sweep_job=args.sweep_job, aggregate_job=args.aggregate_job, root_prefix=args.root_prefix)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
