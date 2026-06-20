#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from scripts.build_reward_sweep_manifest import PROFILE_TCV_DELTA_F002_SEMANTICS, build_manifest
except ModuleNotFoundError:  # pragma: no cover - used when run as python3 scripts/...
    from build_reward_sweep_manifest import PROFILE_TCV_DELTA_F002_SEMANTICS, build_manifest


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
        profile=PROFILE_TCV_DELTA_F002_SEMANTICS,
        runs_per_array_task=3,
        array_task_count=12,
    )
    if int(manifest["variant_count"]) != 36:
        raise RuntimeError(f"f002 semantics manifest must contain 36 variants, got {manifest['variant_count']}")
    folders = [str(variant["folder"]) for variant in manifest["variants"]]
    if folders[0] != "s000_t0_q0_r0" or folders[-1] != "s035_t2_q2_r3":
        raise RuntimeError(f"unexpected f002 semantics folder range: {folders[0]} ... {folders[-1]}")
    if {variant["reward"]["shape_mean_weight"] for variant in manifest["variants"]} != {3.2}:
        raise RuntimeError("f002 semantics variants must keep shape_mean_weight fixed at 3.2")
    if {variant["reward"]["terminal_reward"] for variant in manifest["variants"]} != {-10.0, -20.0, -50.0}:
        raise RuntimeError("f002 semantics variants must vary terminal_reward over {-10,-20,-50}")
    if {variant["sim"]["terminate_on_boundary_loss"] for variant in manifest["variants"]} != {True}:
        raise RuntimeError("f002 semantics variants must enable boundary termination")
    if {variant["sim"]["terminate_on_current_limit"] for variant in manifest["variants"]} != {True}:
        raise RuntimeError("f002 semantics variants must enable current termination")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(manifest_path)


def submit_f002_semantics_sweep(
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
        "profile": PROFILE_TCV_DELTA_F002_SEMANTICS,
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
    parser = argparse.ArgumentParser(description="Submit a 36-run f002-locked TCV delta-Jdot semantics sweep.")
    parser.add_argument(
        "--sweep-job",
        type=Path,
        default=Path("jobs/sweep_t15_csv_segmented_profile_rewards_12gpu_tcv_delta_f002_semantics_onepass.sbatch"),
    )
    parser.add_argument(
        "--aggregate-job",
        type=Path,
        default=Path("jobs/aggregate_t15_reward_sweep_onepass.sbatch"),
    )
    parser.add_argument("--root-prefix", default="outputs/t15_reward_sweep36_tcv_delta_f002_semantics_5m")
    args = parser.parse_args(argv)
    payload = submit_f002_semantics_sweep(
        sweep_job=args.sweep_job,
        aggregate_job=args.aggregate_job,
        root_prefix=args.root_prefix,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
