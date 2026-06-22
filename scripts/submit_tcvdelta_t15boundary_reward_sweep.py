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

from tokamak_rl_v2.sweeps.tcvdelta_t15boundary import write_manifest


REQUIRED_T15_BOUNDARY_SHOTS = ("3854", "3855", "3856", "3857", "3858", "3859", "3862", "3863", "3864")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit the 36x2M TCV-delta T15-boundary reward sweep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--job", default="jobs/sweep_t15_csv_segmented_profile_tcvdelta_t15boundary_12gpu_36x2m.sbatch")
    parser.add_argument("--aggregate-job", default="jobs/aggregate_tcvdelta_t15boundary_reward_sweep.sbatch")
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
        root = ROOT / f"outputs/t15_reward_sweep36_tcvdelta_t15boundary_2m_{fake_jobid}"
        manifest = root / "variants.json"
        write_manifest(manifest)
        print(json.dumps({"dry_run": True, "root": str(root.relative_to(ROOT)), "manifest": str(manifest.relative_to(ROOT))}, indent=2))
        return 0

    sweep_jobid = _check_output(["sbatch", "--hold", "--parsable", str(job.relative_to(ROOT))])
    root = ROOT / f"outputs/t15_reward_sweep36_tcvdelta_t15boundary_2m_{sweep_jobid}"
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
    reset_library = ROOT / "data/processed/t15_csv_initial_states.npz"
    replay_dir = ROOT.parent / "tokamak-sim/runs/t15md_limited_replay_dataset"
    required_files = [
        reset_library,
        ROOT / "data/processed/t15_reference_limits.json",
    ]
    missing = [str(path.relative_to(ROOT.parent)) for path in required_files if not path.exists()]
    if not replay_dir.is_dir():
        missing.append(str(replay_dir.relative_to(ROOT.parent)))
    if missing:
        raise SystemExit(
            "missing required sweep inputs:\n  "
            + "\n  ".join(missing)
            + "\nGenerate/copy the T15 limited replay boundary dataset before submitting."
        )
    _preflight_replay_boundary_coverage(reset_library, replay_dir)


def _preflight_replay_boundary_coverage(reset_library: Path, replay_dir: Path) -> None:
    if not reset_library.exists():
        raise SystemExit(f"missing {reset_library.relative_to(ROOT.parent)}")
    smoothed = sorted(replay_dir.glob("lqr_boundary_reference_*_smoothed.npz"))
    if not smoothed:
        raise SystemExit(
            f"missing smoothed replay boundary references in {replay_dir.relative_to(ROOT.parent)}\n"
            "Run tokamak-sim/scripts/smooth_lqr_boundary_references.py after generating the replay dataset."
        )
    available = {
        path.name.removeprefix("lqr_boundary_reference_").removesuffix("_smoothed.npz")
        for path in smoothed
    }
    wanted = set(REQUIRED_T15_BOUNDARY_SHOTS)
    missing = sorted(wanted - available)
    if missing:
        raise SystemExit(
            "smoothed replay boundary references do not cover reset shots:\n  "
            + "\n  ".join(missing)
            + f"\nDirectory: {replay_dir.relative_to(ROOT.parent)}"
        )


def _check_output(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout.strip().splitlines()[-1].strip()


if __name__ == "__main__":
    raise SystemExit(main())
