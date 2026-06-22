#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokamak_rl_v2.sweeps.tcvdelta_single_segment_0p1s import write_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit the 36x1M TCV-Jdot 0.1 s single-segment reward sweep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--job", default="jobs/sweep_t15_csv_single_segment_0p1s_static_boundary_tcvdelta_12gpu_36x1m.sbatch")
    parser.add_argument("--aggregate-job", default="jobs/aggregate_tcvdelta_single_segment_0p1s_reward_sweep.sbatch")
    args = parser.parse_args(argv)

    job = ROOT / args.job
    aggregate_job = ROOT / args.aggregate_job
    if not job.exists():
        raise SystemExit(f"missing sweep job: {job}")
    if not aggregate_job.exists():
        raise SystemExit(f"missing aggregate job: {aggregate_job}")
    _preflight_inputs()
    (ROOT / "slurm_logs").mkdir(exist_ok=True)
    if args.dry_run:
        fake_jobid = "DRYRUN"
        root = ROOT / f"outputs/t15_reward_sweep36_tcvjdot_single_segment_0p1s_static_boundary_1m_{fake_jobid}"
        manifest = root / "variants.json"
        write_manifest(manifest)
        print(json.dumps({"dry_run": True, "root": str(root.relative_to(ROOT)), "manifest": str(manifest.relative_to(ROOT))}, indent=2))
        return 0

    sweep_jobid = _check_output(["sbatch", "--hold", "--parsable", str(job.relative_to(ROOT))])
    root = ROOT / f"outputs/t15_reward_sweep36_tcvjdot_single_segment_0p1s_static_boundary_1m_{sweep_jobid}"
    manifest = root / "variants.json"
    write_manifest(manifest)
    aggregate_jobid = _check_output(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{sweep_jobid}",
            f"--export=ALL,ROOT={root.relative_to(ROOT)}",
            str(aggregate_job.relative_to(ROOT)),
        ]
    )
    subprocess.run(["scontrol", "release", sweep_jobid], check=True)
    print(
        json.dumps(
            {
                "root": str(root.relative_to(ROOT)),
                "manifest": str(manifest.relative_to(ROOT)),
                "sweep_jobid": sweep_jobid,
                "aggregate_jobid": aggregate_jobid,
                "jobs": {
                    "sweep": str(job.relative_to(ROOT)),
                    "aggregate": str(aggregate_job.relative_to(ROOT)),
                },
            },
            indent=2,
        )
    )
    return 0


def _preflight_inputs() -> None:
    required_files = [
        ROOT / "configs/experiments/t15_csv_initial_single_segment_0p1s_static_boundary_mpo.yaml",
        ROOT / "data/processed/t15_csv_initial_states.npz",
        ROOT / "data/processed/t15_reference_limits.json",
    ]
    missing = [str(path.relative_to(ROOT.parent)) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit("missing required sweep inputs:\n  " + "\n  ".join(missing))


def _check_output(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout.strip().splitlines()[-1].strip()


if __name__ == "__main__":
    raise SystemExit(main())
