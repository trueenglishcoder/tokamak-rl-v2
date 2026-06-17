#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from scripts.summarize_reward_sweep_physical import (
    SUMMARY_FIELDS,
    _format_float,
    _row_for_json,
    _write_csv,
    pareto_front,
    rank_rows,
    regime_summary,
    summarize,
)
from scripts.build_reward_sweep_rerun_manifest import build as build_missing_manifest


THRESHOLDS = {
    "top": 25,
    "min_completion": 0.95,
    "min_boundary_late": 0.999,
    "max_terminated_boundary": 0.001,
    "max_current_over_limit_a_max": 250000.0,
    "max_current_over_limit_fraction_late": 0.5,
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_rows(rows: list[dict[str, Any]], *, sweep_pass: str, source_root: Path) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["sweep_pass"] = sweep_pass
        out["source_root"] = str(source_root)
        for key in ("selection_valid", "valid_actor_eval", "on_pareto_front"):
            out[key] = _bool_value(out.get(key))
        try:
            out["variant_index"] = int(float(out.get("variant_index", -1)))
        except (TypeError, ValueError):
            out["variant_index"] = -1
        normalized.append(out)
    return normalized


def _run_pass_summary(pass_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    summarize(
        pass_root,
        out_dir,
        top=int(THRESHOLDS["top"]),
        min_completion=float(THRESHOLDS["min_completion"]),
        min_boundary_late=float(THRESHOLDS["min_boundary_late"]),
        max_terminated_boundary=float(THRESHOLDS["max_terminated_boundary"]),
        max_current_over_limit_a_max=float(THRESHOLDS["max_current_over_limit_a_max"]),
        max_current_over_limit_fraction_late=float(THRESHOLDS["max_current_over_limit_fraction_late"]),
    )
    return _read_csv(out_dir / "physical_sweep_summary.csv")


def _reward_from_row(row: dict[str, Any]) -> dict[str, float]:
    keys = [
        "shape_mean_weight",
        "shape_max_weight",
        "ip_weight",
        "current_weight",
        "current_soft_fraction",
        "current_bad_fraction",
        "derivative_weight",
        "derivative_soft_fraction",
        "derivative_bad_fraction",
        "action_weight",
        "delta_action_weight",
    ]
    reward: dict[str, float] = {}
    for key in keys:
        value = row.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            reward[key] = number
    return reward


def _short(row: dict[str, Any] | None) -> str:
    if row is None:
        return "none"
    return (
        f"`{row['folder']}` from `{row['sweep_pass']}` "
        f"(score={_format_float(row.get('physical_priority_score'))}, "
        f"completion={_format_float(row.get('mean_episode_completion'))}, "
        f"boundary_late={_format_float(row.get('boundary_found_late_min'))}, "
        f"shape={_format_float(row.get('shape_error_mean_m_late'))} m, "
        f"Ip={_format_float(row.get('ip_error_a_late'))} A, "
        f"current_max={_format_float(row.get('current_over_limit_a_max'))} A)"
    )


def summarize_two_pass(root: Path, out_dir: Path) -> dict[str, Any]:
    pass1_root = root / "pass1_broad"
    pass2_root = root / "pass2_focused"
    if not (pass1_root / "variants.json").exists():
        raise ValueError(f"Missing pass1 manifest: {pass1_root / 'variants.json'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pass1_dir = out_dir / "pass1"
    pass2_dir = out_dir / "pass2"
    pass1_rows = _normalize_rows(_run_pass_summary(pass1_root, pass1_dir), sweep_pass="pass1_broad", source_root=pass1_root)
    shutil.copyfile(pass1_dir / "physical_sweep_summary.csv", out_dir / "pass1_physical_summary.csv")
    shutil.copyfile(pass1_dir / "physical_best_candidate.json", out_dir / "pass1_physical_best_candidate.json")

    pass2_rows: list[dict[str, Any]] = []
    if (pass2_root / "variants.json").exists():
        pass2_rows = _normalize_rows(_run_pass_summary(pass2_root, pass2_dir), sweep_pass="pass2_focused", source_root=pass2_root)
        shutil.copyfile(pass2_dir / "physical_sweep_summary.csv", out_dir / "pass2_physical_summary.csv")
    else:
        _write_csv(out_dir / "pass2_physical_summary.csv", [])

    combined = rank_rows(pass1_rows + pass2_rows)
    front = pareto_front(combined)
    regimes = regime_summary(combined, int(THRESHOLDS["top"]))

    fields = ["sweep_pass", "source_root", *SUMMARY_FIELDS]
    _write_csv(out_dir / "combined_physical_summary.csv", combined, fields)
    _write_csv(out_dir / "combined_pareto_front.csv", front, fields)
    _write_csv(out_dir / "combined_regime_summary.csv", regimes)

    missing = build_missing_manifest(root)
    (out_dir / "missing_or_failed_variants.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")

    valid = [row for row in combined if row["selection_valid"]]
    pass2_valid = [row for row in valid if row.get("sweep_pass") == "pass2_focused"]
    recommendation = pass2_valid[0] if pass2_valid else (valid[0] if valid else None)
    backup = None
    if recommendation is not None:
        for row in valid:
            if row["folder"] != recommendation["folder"] or row["sweep_pass"] != recommendation["sweep_pass"]:
                backup = row
                break

    payload = {
        "root": str(root),
        "recommended_candidate": _row_for_json(recommendation),
        "recommended_reward": _reward_from_row(recommendation or {}),
        "backup_candidate": _row_for_json(backup),
        "backup_reward": _reward_from_row(backup or {}),
        "valid_candidates": len(valid),
        "total_candidates": len(combined),
        "missing_or_failed_count": missing["missing_or_failed_count"],
        "thresholds": {key: value for key, value in THRESHOLDS.items() if key != "top"},
        "ranking_note": "Physical held-out metrics only; reward, mean_return, and physical_cost are not ranking inputs.",
    }
    (out_dir / "final_reward_recommendation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Legal-Actuator Reward Selection",
        "",
        "This report ranks candidates by held-out physical actor behavior. It does not rank by reward value, mean return, or physical cost.",
        "",
        f"- Total candidates found: `{len(combined)}`.",
        f"- Valid candidates after hard filters: `{len(valid)}`.",
        f"- Missing or failed candidates: `{missing['missing_or_failed_count']}`.",
        f"- Recommended reward for long training: {_short(recommendation)}.",
        f"- Backup reward if the main candidate starts abusing current limits: {_short(backup)}.",
        "",
        "## Top Combined Candidates",
        "",
        "| Rank | Pass | Variant | Completion | Boundary Late | Shape Late m | Ip Late A | Current Max A |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in valid[: int(THRESHOLDS["top"])]:
        lines.append(
            f"| {row['physical_rank']} | `{row['sweep_pass']}` | `{row['folder']}` | "
            f"{_format_float(row.get('mean_episode_completion'))} | "
            f"{_format_float(row.get('boundary_found_late_min'))} | "
            f"{_format_float(row.get('shape_error_mean_m_late'))} | "
            f"{_format_float(row.get('ip_error_a_late'))} | "
            f"{_format_float(row.get('current_over_limit_a_max'))} |"
        )
    lines.extend(["", "## Pareto Front", ""])
    lines.append("| Pareto Rank | Pass | Variant | Shape Late m | Ip Late A | Current Max A | Current Fraction Late |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for row in front[: int(THRESHOLDS["top"])]:
        lines.append(
            f"| {row.get('pareto_rank', '')} | `{row['sweep_pass']}` | `{row['folder']}` | "
            f"{_format_float(row.get('shape_error_mean_m_late'))} | "
            f"{_format_float(row.get('ip_error_a_late'))} | "
            f"{_format_float(row.get('current_over_limit_a_max'))} | "
            f"{_format_float(row.get('current_over_limit_fraction_late'))} |"
        )
    (out_dir / "final_reward_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a two-pass legal-actuator reward sweep.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or (args.root / "selection")
    summarize_two_pass(args.root, out_dir)
    print(out_dir / "final_reward_selection_report.md")
    print(out_dir / "final_reward_recommendation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
