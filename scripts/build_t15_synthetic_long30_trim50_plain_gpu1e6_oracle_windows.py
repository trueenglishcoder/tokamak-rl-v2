#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_t15_synthetic_long_preview import (
    RealSpace,
    _generate_previews,
    _load_real_space,
    _repo_path,
    _require_inputs,
    _write_plots,
)


ROOT = Path(__file__).resolve().parents[1]
WINDOW_STEPS = 100
DT = 0.001
TRAIN_PARENT_ID_BASE = 900000
HOLDOUT_PARENT_ID_BASE = 910000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a training-ready replay-window/oracle dataset from new long synthetic "
            "T15 trajectories constrained by the safe coupled space of the successful "
            "real trim50 replay-window library."
        )
    )
    parser.add_argument(
        "--oracle-target",
        type=Path,
        default=Path(
            "data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets/"
            "t15_replay_window_oracle_targets.npz"
        ),
    )
    parser.add_argument(
        "--initial-library",
        type=Path,
        default=Path("data/processed/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz"),
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/processed/t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets"),
    )
    parser.add_argument(
        "--initial-library-out",
        type=Path,
        default=Path("data/processed/t15_synthetic_long30_trim50_plain_gpu1e6_replay_window_0p1s_oracle_initial_states.npz"),
    )
    parser.add_argument("--train-parents", type=int, default=30)
    parser.add_argument("--holdout-parents", type=int, default=1)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--wiggle-room", type=float, default=1.15)
    parser.add_argument("--radii-margin-m", type=float, default=0.025)
    parser.add_argument("--current-envelope-margin", type=float, default=0.08)
    parser.add_argument("--max-cloud-rows", type=int, default=20000)
    parser.add_argument("--knn", type=int, default=24)
    parser.add_argument("--max-attempts-per-parent", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--plot-examples", type=int, default=8)
    args = parser.parse_args(argv)

    target_path = _repo_path(args.oracle_target)
    source_initial_path = _repo_path(args.initial_library)
    target_dir = _repo_path(args.target_dir)
    initial_out = _repo_path(args.initial_library_out)
    _require_inputs(target_path=target_path, initial_path=source_initial_path)

    if int(args.train_parents) <= 0:
        raise ValueError("--train-parents must be positive")
    if int(args.holdout_parents) <= 0:
        raise ValueError("--holdout-parents must be positive")
    if int(args.min_steps) < WINDOW_STEPS:
        raise ValueError(f"--min-steps must be at least {WINDOW_STEPS}")
    if int(args.max_steps) < int(args.min_steps):
        raise ValueError("--max-steps must be >= --min-steps")

    target_dir.mkdir(parents=True, exist_ok=True)
    initial_out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    print("[synthetic-long30] loading train safe space", flush=True)
    train_space = _load_space(args=args, target_path=target_path, initial_path=source_initial_path, split="train", rng=rng)
    print("[synthetic-long30] loading holdout safe space", flush=True)
    holdout_space = _load_space(args=args, target_path=target_path, initial_path=source_initial_path, split="holdout", rng=rng)

    train_parents, train_rejects = _make_parents(
        label="train",
        real=train_space,
        count=int(args.train_parents),
        args=args,
        rng=rng,
    )
    holdout_parents, holdout_rejects = _make_parents(
        label="holdout",
        real=holdout_space,
        count=int(args.holdout_parents),
        args=args,
        rng=rng,
    )

    rows: list[dict[str, Any]] = []
    parent_summaries: list[dict[str, Any]] = []
    for parent_idx, parent in enumerate(train_parents):
        parent_id = TRAIN_PARENT_ID_BASE + parent_idx
        parent_rows = _windows_from_parent(parent_id=parent_id, split="train", parent=parent, real=train_space)
        rows.extend(parent_rows)
        parent_summaries.append(_parent_summary(parent_id=parent_id, split="train", parent=parent, window_count=len(parent_rows)))
    for parent_idx, parent in enumerate(holdout_parents):
        parent_id = HOLDOUT_PARENT_ID_BASE + parent_idx
        parent_rows = _windows_from_parent(parent_id=parent_id, split="holdout", parent=parent, real=holdout_space)
        rows.extend(parent_rows)
        parent_summaries.append(_parent_summary(parent_id=parent_id, split="holdout", parent=parent, window_count=len(parent_rows)))

    if not rows:
        raise RuntimeError("synthetic long30 builder produced zero replay windows")

    oracle_path = target_dir / "t15_replay_window_oracle_targets.npz"
    _write_oracle_npz(oracle_path, rows, current_limits=train_space.current_limits, derivative_limits=train_space.derivative_limits)
    _write_initial_library(initial_out, rows)

    plot_dir = target_dir / "parent_previews"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_items = train_parents[: max(0, int(args.plot_examples))]
    if plot_items:
        _write_plots(out_dir=plot_dir, trajectories=plot_items, real=train_space, dt=DT)

    summary = _summary(
        args=args,
        source_target=target_path,
        source_initial=source_initial_path,
        oracle_path=oracle_path,
        initial_out=initial_out,
        rows=rows,
        parents=parent_summaries,
        train_rejects=train_rejects,
        holdout_rejects=holdout_rejects,
        train_space=train_space,
    )
    (target_dir / "synthetic_long30_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(target_dir / "synthetic_long30_report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def _load_space(
    *,
    args: argparse.Namespace,
    target_path: Path,
    initial_path: Path,
    split: str,
    rng: np.random.Generator,
) -> RealSpace:
    return _load_real_space(
        target_path=target_path,
        initial_path=initial_path,
        dt=DT,
        pca_components=int(args.pca_components),
        wiggle_room=float(args.wiggle_room),
        radii_margin_m=float(args.radii_margin_m),
        current_envelope_margin=float(args.current_envelope_margin),
        max_cloud_rows=int(args.max_cloud_rows),
        safe_split=str(split),
        rng=rng,
    )


def _make_parents(
    *,
    label: str,
    real: RealSpace,
    count: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[dict[str, np.ndarray | int]], dict[str, int]]:
    print(f"[synthetic-long30] generating {count} {label} parent trajectories", flush=True)
    parents, rejects = _generate_previews(
        real=real,
        examples=int(count),
        min_steps=int(args.min_steps),
        max_steps=int(args.max_steps),
        dt=DT,
        wiggle_room=float(args.wiggle_room),
        knn=int(args.knn),
        max_attempts=int(args.max_attempts_per_parent) * int(count),
        progress_every=int(args.progress_every),
        rng=rng,
    )
    if len(parents) != int(count):
        raise RuntimeError(
            f"generated only {len(parents)} / {count} {label} synthetic parents; "
            f"reject_counts={rejects}"
        )
    return parents, rejects


def _windows_from_parent(
    *,
    parent_id: int,
    split: str,
    parent: dict[str, np.ndarray | int],
    real: RealSpace,
) -> list[dict[str, Any]]:
    steps = int(parent["steps"])
    ip = np.asarray(parent["ip"], dtype=np.float64).reshape(-1)
    radii = np.asarray(parent["radii"], dtype=np.float64)
    currents = np.asarray(parent["currents"], dtype=np.float64)
    if ip.shape[0] != steps + 1 or radii.shape[0] != steps + 1 or currents.shape[0] != steps + 1:
        raise ValueError(f"parent {parent_id} has inconsistent point count")
    if radii.shape[1] < 32:
        raise ValueError(f"parent {parent_id} has only {radii.shape[1]} boundary angles")
    if currents.shape[1] != real.current_limits.shape[0]:
        raise ValueError(f"parent {parent_id} current dimension mismatch")

    jdot = np.diff(currents, axis=0) / DT
    action = jdot / real.derivative_limits.reshape(1, -1)
    if not np.all(np.isfinite(action)):
        raise ValueError(f"parent {parent_id} produced non-finite actions")
    if np.any(np.abs(action) > 1.0 + 1.0e-6):
        raise ValueError(f"parent {parent_id} produced actions outside normalized derivative limits")

    rows: list[dict[str, Any]] = []
    for start in range(0, ip.shape[0] - WINDOW_STEPS):
        end = start + WINDOW_STEPS
        ip_target = ip[start : end + 1]
        current0 = currents[start]
        rows.append(
            {
                "shot_id": int(parent_id),
                "split": str(split),
                "source_index": int(start),
                "time_s": float(start * DT),
                "ip0": float(ip_target[0]),
                "pfc0": current0[:6].astype(np.float32),
                "sol0": current0[6:].astype(np.float32),
                "ip_target": ip_target.astype(np.float32),
                "boundary_radii": radii[start : end + 1, :32].astype(np.float32),
                "real_jdot_action": action[start:end].astype(np.float32),
                "difficulty_bin": _difficulty_bin(float(ip_target[-1] - ip_target[0])),
                "oracle_ip_mean_error_a": 0.0,
                "oracle_ip_max_error_a": 0.0,
            }
        )
    return rows


def _difficulty_bin(delta_ip: float) -> str:
    mag = abs(float(delta_ip))
    if mag < 10000.0:
        return "flat"
    direction = "up" if float(delta_ip) > 0.0 else "down"
    if mag < 40000.0:
        return f"medium_{direction}"
    return f"fast_{direction}"


def _write_oracle_npz(path: Path, rows: list[dict[str, Any]], *, current_limits: np.ndarray, derivative_limits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(["t15_replay_window_oracle_targets_v1"]),
        shot_id=np.asarray([r["shot_id"] for r in rows], dtype=np.int64),
        split=np.asarray([r["split"] for r in rows]),
        source_index=np.asarray([r["source_index"] for r in rows], dtype=np.int64),
        time_s=np.asarray([r["time_s"] for r in rows], dtype=np.float64),
        difficulty_bin=np.asarray([r["difficulty_bin"] for r in rows]),
        ip0=np.asarray([r["ip0"] for r in rows], dtype=np.float32),
        pfc0=np.stack([r["pfc0"] for r in rows], axis=0).astype(np.float32),
        sol0=np.stack([r["sol0"] for r in rows], axis=0).astype(np.float32),
        ip_target=np.stack([r["ip_target"] for r in rows], axis=0).astype(np.float32),
        boundary_radii=np.stack([r["boundary_radii"] for r in rows], axis=0).astype(np.float32),
        real_jdot_action=np.stack([r["real_jdot_action"] for r in rows], axis=0).astype(np.float32),
        oracle_ip_mean_error_a=np.asarray([r["oracle_ip_mean_error_a"] for r in rows], dtype=np.float32),
        oracle_ip_max_error_a=np.asarray([r["oracle_ip_max_error_a"] for r in rows], dtype=np.float32),
        current_limits=np.asarray(current_limits, dtype=np.float32),
        derivative_limits=np.asarray(derivative_limits, dtype=np.float32),
    )


def _write_initial_library(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shot_id=np.asarray([r["shot_id"] for r in rows], dtype=np.int64),
        source_index=np.asarray([r["source_index"] for r in rows], dtype=np.int64),
        time_s=np.asarray([r["time_s"] for r in rows], dtype=np.float64),
        ip0=np.asarray([r["ip0"] for r in rows], dtype=np.float32),
        pfc0=np.stack([r["pfc0"] for r in rows], axis=0).astype(np.float32),
        sol0=np.stack([r["sol0"] for r in rows], axis=0).astype(np.float32),
        split=np.asarray([r["split"] for r in rows]),
        difficulty_bin=np.asarray([r["difficulty_bin"] for r in rows]),
    )


def _parent_summary(
    *,
    parent_id: int,
    split: str,
    parent: dict[str, np.ndarray | int],
    window_count: int,
) -> dict[str, Any]:
    currents = np.asarray(parent["currents"], dtype=np.float64)
    jdot = np.asarray(parent["jdot"], dtype=np.float64)
    return {
        "parent_id": int(parent_id),
        "split": str(split),
        "steps": int(parent["steps"]),
        "windows": int(window_count),
        "ip_min": float(np.min(np.asarray(parent["ip"], dtype=np.float64))),
        "ip_max": float(np.max(np.asarray(parent["ip"], dtype=np.float64))),
        "radii_min": float(np.min(np.asarray(parent["radii"], dtype=np.float64))),
        "radii_max": float(np.max(np.asarray(parent["radii"], dtype=np.float64))),
        "current_abs_max": float(np.max(np.abs(currents))),
        "jdot_abs_max": float(np.max(np.abs(jdot))),
    }


def _summary(
    *,
    args: argparse.Namespace,
    source_target: Path,
    source_initial: Path,
    oracle_path: Path,
    initial_out: Path,
    rows: list[dict[str, Any]],
    parents: list[dict[str, Any]],
    train_rejects: dict[str, int],
    holdout_rejects: dict[str, int],
    train_space: RealSpace,
) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for row in rows:
        split_counts[str(row["split"])] = split_counts.get(str(row["split"]), 0) + 1
        difficulty_counts[str(row["difficulty_bin"])] = difficulty_counts.get(str(row["difficulty_bin"]), 0) + 1
    actions = np.stack([r["real_jdot_action"] for r in rows], axis=0)
    pfc0 = np.stack([r["pfc0"] for r in rows], axis=0)
    sol0 = np.stack([r["sol0"] for r in rows], axis=0)
    current0 = np.concatenate([pfc0, sol0], axis=1)
    return {
        "schema": "t15_synthetic_long30_oracle_window_summary_v1",
        "source_oracle_target": str(source_target),
        "source_initial_library": str(source_initial),
        "oracle_path": str(oracle_path),
        "initial_library": str(initial_out),
        "train_parents": int(args.train_parents),
        "holdout_parents": int(args.holdout_parents),
        "parent_steps": {"min": int(args.min_steps), "max": int(args.max_steps)},
        "window_steps": WINDOW_STEPS,
        "dt": DT,
        "accepted_windows": int(len(rows)),
        "split_counts": dict(sorted(split_counts.items())),
        "difficulty_bins": dict(sorted(difficulty_counts.items())),
        "train_parent_reject_counts": dict(sorted(train_rejects.items())),
        "holdout_parent_reject_counts": dict(sorted(holdout_rejects.items())),
        "parent_summaries": parents,
        "action_abs_max": float(np.max(np.abs(actions))),
        "action_abs_mean": float(np.mean(np.abs(actions))),
        "current_usage_fraction_max": float(np.max(np.abs(current0) / train_space.current_limits.reshape(1, -1))),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# T15 Synthetic Long30 Replay-Window Dataset",
        "",
        f"- Accepted windows: {summary['accepted_windows']}",
        f"- Train parents: {summary['train_parents']}",
        f"- Holdout parents: {summary['holdout_parents']}",
        f"- Parent step range: {summary['parent_steps']['min']}..{summary['parent_steps']['max']}",
        f"- Window steps: {summary['window_steps']}",
        f"- Max normalized action: {summary['action_abs_max']:.4f}",
        f"- Mean absolute normalized action: {summary['action_abs_mean']:.4f}",
        f"- Max reset current usage: {summary['current_usage_fraction_max']:.4f}",
        "",
        "## Splits",
        "",
        "| split | windows |",
        "|---|---:|",
    ]
    for split, count in summary["split_counts"].items():
        lines.append(f"| `{split}` | {count} |")
    lines.extend(["", "## Difficulty Bins", "", "| bin | windows |", "|---|---:|"])
    for bin_name, count in summary["difficulty_bins"].items():
        lines.append(f"| `{bin_name}` | {count} |")
    lines.extend(["", "## Parent Generation Rejections", "", "| split | reason | count |", "|---|---|---:|"])
    wrote = False
    for split_key in ("train_parent_reject_counts", "holdout_parent_reject_counts"):
        split = "train" if split_key.startswith("train") else "holdout"
        for reason, count in summary[split_key].items():
            lines.append(f"| `{split}` | `{reason}` | {count} |")
            wrote = True
    if not wrote:
        lines.append("| none | none | 0 |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
