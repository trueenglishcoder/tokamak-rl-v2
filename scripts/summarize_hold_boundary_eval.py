#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

SUMMARY_METRICS = (
    "mean_episode_completion",
    "min_episode_completion",
    "full_episode_success",
    "shape_error_mean_m_late",
    "shape_error_max_m_late",
    "ip_error_a_late",
    "current_usage_fraction_late",
    "current_over_limit_a_late_max",
    "current_over_limit_fraction_late",
    "current_over_limit_5ka_fraction_late",
    "current_over_limit_1pct_fraction_late",
    "action_rms_late",
    "action_saturation_fraction_late",
    "terminated_boundary",
    "terminated_current",
    "boundary_found_late_min",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate hold_boundary_eval shard outputs.")
    ap.add_argument("root", help="Evaluation root containing shard_* directories")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"hold_boundary_eval root does not exist: {root}")
    out_dir = Path(args.out_dir) if args.out_dir else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(p for p in root.glob("shard_*") if p.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_* directories found under {root}")

    shard_summaries = []
    combined_by_policy: dict[str, list[dict[str, object]]] = {"policy": [], "no_control": []}
    for shard in shard_dirs:
        summary_path = shard / "hold_boundary_eval_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        shard_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_summaries.append(
            {
                "shard": shard.name,
                "summary_path": str(summary_path),
                "episodes": shard_summary.get("episodes"),
                "steps": shard_summary.get("steps"),
                "policies": shard_summary.get("policies", {}),
            }
        )
        for policy in tuple(combined_by_policy):
            csv_path = shard / f"hold_boundary_eval_{policy}_windows.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"missing shard window CSV: {csv_path}")
            combined_by_policy[policy].extend(_read_csv(csv_path, shard=shard.name))

    aggregate: dict[str, Any] = {
        "root": str(root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "shard_count": len(shard_dirs),
        "shards": shard_summaries,
        "policies": {},
    }
    for policy, rows in combined_by_policy.items():
        aggregate["policies"][policy] = _summary_from_windows(rows)
        _write_csv(out_dir / f"hold_boundary_eval_{policy}_windows_combined.csv", rows)

    (out_dir / "hold_boundary_eval_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    summary_rows = [
        {"policy": policy, **{metric: aggregate["policies"][policy].get(metric, float("nan")) for metric in SUMMARY_METRICS}}
        for policy in ("policy", "no_control")
    ]
    _write_csv(out_dir / "hold_boundary_eval_summary.csv", summary_rows)
    _write_report(aggregate, out_dir / "hold_boundary_eval_report.md")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


def _read_csv(path: Path, *, shard: str) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            out: dict[str, object] = {"shard": shard}
            out.update(row)
            rows.append(out)
        return rows


def _summary_from_windows(rows: list[dict[str, object]]) -> dict[str, float]:
    out: dict[str, float] = {"episodes": float(len(rows))}
    if not rows:
        return out
    for key in (
        "episode_completion",
        "full_episode_success",
        "shape_error_mean_m_late",
        "shape_error_max_m_late",
        "ip_error_a_late",
        "current_usage_fraction_late",
        "action_rms_late",
        "action_saturation_fraction_late",
        "terminated_boundary",
        "terminated_current",
    ):
        values = _col(rows, key)
        out[_summary_key(key)] = float(np.nanmean(values)) if values.size else float("nan")
    for key in ("episode_completion", "boundary_found_late"):
        values = _col(rows, key)
        out[f"{_summary_key(key)}_min"] = float(np.nanmin(values)) if values.size else float("nan")
    current_over = _col(rows, "current_over_limit_a_late")
    usage = _col(rows, "current_usage_fraction_late")
    out["current_over_limit_a_late_max"] = float(np.nanmax(current_over)) if current_over.size else float("nan")
    out["current_over_limit_fraction_late"] = float(np.nanmean(current_over > 0.0)) if current_over.size else float("nan")
    out["current_over_limit_5ka_fraction_late"] = float(np.nanmean(current_over > 5000.0)) if current_over.size else float("nan")
    out["current_over_limit_1pct_fraction_late"] = float(np.nanmean(usage > 1.01)) if usage.size else float("nan")
    return out


def _summary_key(key: str) -> str:
    return {
        "episode_completion": "mean_episode_completion",
        "full_episode_success": "full_episode_success",
        "terminated_boundary": "terminated_boundary",
        "terminated_current": "terminated_current",
    }.get(key, key)


def _col(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(aggregate: dict[str, Any], path: Path) -> None:
    policies = aggregate.get("policies", {})
    lines = [
        "# hold_boundary_eval",
        "",
        f"Root: `{aggregate.get('root')}`",
        f"Shards: {aggregate.get('shard_count')}",
        "",
        "| policy | " + " | ".join(SUMMARY_METRICS) + " |",
        "|---|" + "|".join("---" for _ in SUMMARY_METRICS) + "|",
    ]
    for policy in ("policy", "no_control"):
        metrics = policies.get(policy, {})
        lines.append("| " + policy + " | " + " | ".join(_fmt(metrics.get(metric)) for metric in SUMMARY_METRICS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    if abs(number) >= 1000.0 or (0.0 < abs(number) < 1.0e-3):
        return f"{number:.4g}"
    return f"{number:.5f}"


if __name__ == "__main__":
    raise SystemExit(main())
