#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from scripts.build_reward_sweep_manifest import PROFILE_TCV_DELTA_NO_TERMINATION, build_manifest
except ModuleNotFoundError:  # pragma: no cover - used when run as python3 scripts/...
    from build_reward_sweep_manifest import PROFILE_TCV_DELTA_NO_TERMINATION, build_manifest


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
        profile=PROFILE_TCV_DELTA_NO_TERMINATION,
        runs_per_array_task=3,
        array_task_count=12,
    )
    if int(manifest["variant_count"]) != 36:
        raise RuntimeError(f"no-termination manifest must contain 36 variants, got {manifest['variant_count']}")
    folders = [str(variant["folder"]) for variant in manifest["variants"]]
    if folders[0] != "n000_s0_i0_a0" or folders[-1] != "n035_s2_i2_a3":
        raise RuntimeError(f"unexpected no-termination variant folder range: {folders[0]} ... {folders[-1]}")
    if {variant["sim"]["terminate_on_boundary_loss"] for variant in manifest["variants"]} != {False}:
        raise RuntimeError("no-termination variants must disable boundary termination")
    if {variant["sim"]["terminate_on_current_limit"] for variant in manifest["variants"]} != {False}:
        raise RuntimeError("no-termination variants must disable current termination")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(manifest_path)


def submit_no_termination_sweep(
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
        "profile": PROFILE_TCV_DELTA_NO_TERMINATION,
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
    parser = argparse.ArgumentParser(description="Submit a 36-run no-termination TCV delta-Jdot reward sweep.")
    parser.add_argument(
        "--sweep-job",
        type=Path,
        default=Path("jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_tcv_delta_no_termination_onepass.sbatch"),
    )
    parser.add_argument(
        "--aggregate-job",
        type=Path,
        default=Path("jobs/aggregate_t15_reward_sweep_onepass.sbatch"),
    )
    parser.add_argument("--root-prefix", default="outputs/t15_reward_sweep36_tcv_delta_noterm_2m")
    args = parser.parse_args(argv)
    payload = submit_no_termination_sweep(
        sweep_job=args.sweep_job,
        aggregate_job=args.aggregate_job,
        root_prefix=args.root_prefix,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
