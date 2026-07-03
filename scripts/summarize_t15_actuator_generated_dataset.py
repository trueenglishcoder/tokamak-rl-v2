#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize an actuator-generated T15 0.1 s dataset.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--initial-states", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.dataset_dir.expanduser().resolve()
    initial_path = (
        args.initial_states.expanduser().resolve()
        if args.initial_states is not None
        else root.parent / f"{root.name}_initial_states.npz"
    )
    oracle_path = root / "t15_replay_window_oracle_targets.npz"
    diagnostic_path = root / "t15_actuator_generated_targets.npz"
    summary_path = root / "actuator_generated_summary.json"
    accepted_path = root / "actuator_generated_accepted.csv"
    rejected_path = root / "actuator_generated_rejected.csv"
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir is not None else root / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [initial_path, oracle_path, diagnostic_path, summary_path, accepted_path, rejected_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing actuator-generated dataset artifacts:\n" + "\n".join(missing))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    accepted = _read_csv(accepted_path)
    rejected = _read_csv(rejected_path)
    oracle_stats = _oracle_stats(oracle_path)
    reset_stats = _reset_stats(initial_path)

    mode_counts = Counter(row["mode"] for row in accepted)
    split_counts = Counter(row["split"] for row in accepted)
    difficulty_counts = Counter(row["difficulty_bin"] for row in accepted)
    rejection_counts = Counter(row["reason"] for row in rejected if row.get("reason"))

    report = _render_report(
        root=root,
        initial_path=initial_path,
        oracle_stats=oracle_stats,
        reset_stats=reset_stats,
        summary=summary,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        split_counts=split_counts,
        mode_counts=mode_counts,
        difficulty_counts=difficulty_counts,
        rejection_counts=rejection_counts,
    )
    report_path = out_dir / "actuator_generated_dataset_report.md"
    report_path.write_text(report, encoding="utf-8")
    (out_dir / "actuator_generated_dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(root),
                "initial_states": str(initial_path),
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "split_counts": dict(sorted(split_counts.items())),
                "mode_counts": dict(sorted(mode_counts.items())),
                "difficulty_counts": dict(sorted(difficulty_counts.items())),
                "rejection_counts": dict(sorted(rejection_counts.items())),
                "oracle_stats": oracle_stats,
                "reset_stats": reset_stats,
                "builder_summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(report_path)
    print(report)
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _oracle_stats(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        ip = np.asarray(data["ip_target"], dtype=float)
        radii = np.asarray(data["boundary_radii"], dtype=float)
        action = np.asarray(data["real_jdot_action"], dtype=float)
        return {
            "rows": int(ip.shape[0]),
            "ip_shape": list(ip.shape),
            "boundary_radii_shape": list(radii.shape),
            "real_jdot_action_shape": list(action.shape),
            "ip_min_a": float(np.nanmin(ip)),
            "ip_max_a": float(np.nanmax(ip)),
            "endpoint_dip_min_a": float(np.nanmin(ip[:, -1] - ip[:, 0])),
            "endpoint_dip_max_a": float(np.nanmax(ip[:, -1] - ip[:, 0])),
            "radii_min_m": float(np.nanmin(radii)),
            "radii_max_m": float(np.nanmax(radii)),
            "mean_radius_delta_min_m": float(np.nanmin(np.nanmean(radii[:, -1], axis=1) - np.nanmean(radii[:, 0], axis=1))),
            "mean_radius_delta_max_m": float(np.nanmax(np.nanmean(radii[:, -1], axis=1) - np.nanmean(radii[:, 0], axis=1))),
            "action_abs_max": float(np.nanmax(np.abs(action))),
            "action_rms_mean": float(np.sqrt(np.nanmean(action * action))),
        }


def _reset_stats(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        split = np.asarray(data["split"]).astype(str)
        shot = np.asarray(data["shot_id"]).astype(str)
        return {
            "rows": int(split.shape[0]),
            "split_counts": {str(name): int(np.sum(split == name)) for name in sorted(set(split.tolist()))},
            "shot_counts": {str(name): int(np.sum(shot == name)) for name in sorted(set(shot.tolist()), key=int)},
        }


def _table(counter: Counter[str]) -> str:
    if not counter:
        return "_none_\n"
    lines = ["| value | count |", "|---|---:|"]
    for key, value in sorted(counter.items()):
        lines.append(f"| `{key}` | {int(value)} |")
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _render_report(
    *,
    root: Path,
    initial_path: Path,
    oracle_stats: dict[str, object],
    reset_stats: dict[str, object],
    summary: dict[str, object],
    accepted_count: int,
    rejected_count: int,
    split_counts: Counter[str],
    mode_counts: Counter[str],
    difficulty_counts: Counter[str],
    rejection_counts: Counter[str],
) -> str:
    rejection_total = max(int(accepted_count) + int(rejected_count), 1)
    rejection_fraction = float(rejected_count) / float(rejection_total)
    lines = [
        "# Actuator-Generated Dataset Report",
        "",
        f"- Dataset dir: `{root}`",
        f"- Initial states: `{initial_path}`",
        f"- Accepted trajectories: `{accepted_count}`",
        f"- Rejected candidates: `{rejected_count}`",
        f"- Rejection fraction: `{rejection_fraction:.3%}`",
        f"- Oracle rows: `{oracle_stats['rows']}`",
        f"- Reset rows: `{reset_stats['rows']}`",
        "",
        "## Shapes And Ranges",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "ip_shape",
        "boundary_radii_shape",
        "real_jdot_action_shape",
        "ip_min_a",
        "ip_max_a",
        "endpoint_dip_min_a",
        "endpoint_dip_max_a",
        "radii_min_m",
        "radii_max_m",
        "mean_radius_delta_min_m",
        "mean_radius_delta_max_m",
        "action_abs_max",
        "action_rms_mean",
    ):
        lines.append(f"| `{key}` | `{_fmt(oracle_stats[key])}` |")
    lines += [
        "",
        "## Accepted By Split",
        "",
        _table(split_counts),
        "## Accepted By Mode",
        "",
        _table(mode_counts),
        "## Accepted By Difficulty",
        "",
        _table(difficulty_counts),
        "## Rejected By Reason",
        "",
        _table(rejection_counts),
        "## Plots",
        "",
        f"- `{root / 'sample_actuator_generated_targets.png'}`",
        f"- `{root / 'actuator_generated_coverage_histograms.png'}`",
        "",
        "## Builder Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
