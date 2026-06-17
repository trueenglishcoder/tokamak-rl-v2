#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCORE_COLUMNS = [
    "shape_error_mean_m_late",
    "shape_error_max_m_late",
    "ip_error_a_late",
    "current_over_limit_a_late",
    "current_over_limit_fraction_late",
    "mean_episode_completion",
    "boundary_found_late_min",
]


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _metric(row: dict[str, str], *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        value = _float(row, key)
        if math.isfinite(value):
            return value
    return default


def score_eval_row(row: dict[str, str]) -> float:
    shape_mean = _metric(row, "shape_error_mean_m_late", "shape_error_mean_m", default=1.0)
    shape_max = _metric(row, "shape_error_max_m_late", "shape_error_max_m", default=1.0)
    ip_error = _metric(row, "ip_error_a_late", "ip_error_a", default=1.0e9)
    current_over = _metric(row, "current_over_limit_a_late", "current_over_limit_a_late_max", "current_over_limit_a_max", "current_over_limit_a", default=1.0e9)
    current_fraction = _metric(row, "current_over_limit_fraction_late", "current_over_limit_fraction", default=1.0)
    completion = _metric(row, "mean_episode_completion", "episode_progress", default=0.0)
    boundary_late = _metric(row, "boundary_found_late_min", "boundary_found_min", "boundary_found", default=0.0)
    return float(
        2.0 * shape_mean / 0.03
        + 1.0 * shape_max / 0.08
        + 2.0 * ip_error / 25000.0
        + 3.0 * current_over / 20000.0
        + 2.0 * current_fraction
        + 20.0 * max(0.0, 0.95 - completion)
        + 100.0 * max(0.0, 0.999 - boundary_late)
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("inf")


def _format_float(value: object) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return "inf"
    if abs(v) >= 1000.0 or (abs(v) < 0.001 and v != 0.0):
        return f"{v:.6g}"
    return f"{v:.6f}".rstrip("0").rstrip(".")


def summarize_variant(root: Path, variant: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    folder = str(variant.get("folder") or f"v{int(variant['index']):03d}_{variant['name']}")
    run_dir = root / folder
    validation = _read_json(run_dir / "policy_validation.json")
    metrics = _read_json(run_dir / "metrics.json")
    rows = _read_csv(run_dir / "eval_history.csv")
    status = str(validation.get("status") or metrics.get("status") or ("missing" if not run_dir.exists() else "no_eval"))
    failure: dict[str, Any] | None = None

    scored_rows: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        score = score_eval_row(row)
        if math.isfinite(score):
            scored_rows.append((score, row))

    if scored_rows:
        best_score, best_row = min(scored_rows, key=lambda item: item[0])
        best_step = int(_metric(best_row, "env_step", "step", default=0.0))
        tail_score = _mean(score for score, _row in scored_rows[-3:])
        final_score, final_row = scored_rows[-1]
        selected_score = 0.4 * best_score + 0.6 * tail_score
    else:
        best_score = tail_score = final_score = selected_score = float("inf")
        best_step = 0
        final_row = {}
        failure = {
            "variant_index": int(variant["index"]),
            "name": str(variant["name"]),
            "folder": folder,
            "status": status,
            "reason": "missing or empty eval_history.csv",
            "path": str(run_dir),
        }

    reward = variant.get("reward", {}) if isinstance(variant.get("reward"), dict) else {}
    summary = {
        "rank": "",
        "variant_index": int(variant["index"]),
        "name": str(variant["name"]),
        "folder": folder,
        "status": status,
        "selected_score": selected_score,
        "best_score": best_score,
        "best_step": best_step,
        "tail_score": tail_score,
        "final_score": final_score,
        "final_step": int(_metric(final_row, "env_step", "step", default=0.0)),
        "final_mean_episode_completion": _metric(final_row, "mean_episode_completion", "episode_progress", default=float("nan")),
        "final_boundary_found_late_min": _metric(final_row, "boundary_found_late_min", "boundary_found_min", "boundary_found", default=float("nan")),
        "final_shape_error_mean_m_late": _metric(final_row, "shape_error_mean_m_late", "shape_error_mean_m", default=float("nan")),
        "final_shape_error_max_m_late": _metric(final_row, "shape_error_max_m_late", "shape_error_max_m", default=float("nan")),
        "final_ip_error_a_late": _metric(final_row, "ip_error_a_late", "ip_error_a", default=float("nan")),
        "final_current_over_limit_a_late": _metric(final_row, "current_over_limit_a_late", "current_over_limit_a_late_max", "current_over_limit_a_max", "current_over_limit_a", default=float("nan")),
        "final_current_over_limit_fraction_late": _metric(final_row, "current_over_limit_fraction_late", "current_over_limit_fraction", default=float("nan")),
        "shape_mean_weight": reward.get("shape_mean_weight", ""),
        "shape_max_weight": reward.get("shape_max_weight", ""),
        "ip_weight": reward.get("ip_weight", ""),
        "current_weight": reward.get("current_weight", ""),
        "current_soft_fraction": reward.get("current_soft_fraction", ""),
        "current_bad_fraction": reward.get("current_bad_fraction", ""),
        "derivative_weight": reward.get("derivative_weight", ""),
        "derivative_soft_fraction": reward.get("derivative_soft_fraction", ""),
        "derivative_bad_fraction": reward.get("derivative_bad_fraction", ""),
        "action_weight": reward.get("action_weight", ""),
        "delta_action_weight": reward.get("delta_action_weight", ""),
    }
    return summary, failure


def aggregate(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "variants.json")
    variants = manifest.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"{root / 'variants.json'} does not contain a nonempty variants list")

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        summary, failure = summarize_variant(root, variant)
        summaries.append(summary)
        if failure is not None:
            failures.append(failure)

    summaries.sort(key=lambda row: (float(row["selected_score"]), int(row["variant_index"])))
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return {"manifest": manifest, "summaries": summaries, "failures": failures}


def write_outputs(root: Path, result: dict[str, Any]) -> None:
    summaries: list[dict[str, Any]] = result["summaries"]
    failures: list[dict[str, Any]] = result["failures"]
    summary_path = root / "reward_sweep_summary.csv"
    if summaries:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)
    failures_path = root / "reward_sweep_failures.csv"
    failure_fields = ["variant_index", "name", "folder", "status", "reason", "path"]
    with failures_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=failure_fields)
        writer.writeheader()
        for row in failures:
            writer.writerow({key: row.get(key, "") for key in failure_fields})

    top = summaries[:20]
    md_lines = [
        "# Reward Sweep Top 20",
        "",
        "| Rank | Variant | Score | Best | Tail | Completion | Boundary Late | Shape Late m | Ip Late A | Current Late A |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top:
        md_lines.append(
            "| {rank} | `{name}` | {selected_score} | {best_score} | {tail_score} | {completion} | {boundary} | {shape} | {ip} | {current} |".format(
                rank=row["rank"],
                name=row["folder"],
                selected_score=_format_float(row["selected_score"]),
                best_score=_format_float(row["best_score"]),
                tail_score=_format_float(row["tail_score"]),
                completion=_format_float(row["final_mean_episode_completion"]),
                boundary=_format_float(row["final_boundary_found_late_min"]),
                shape=_format_float(row["final_shape_error_mean_m_late"]),
                ip=_format_float(row["final_ip_error_a_late"]),
                current=_format_float(row["final_current_over_limit_a_late"]),
            )
        )
    (root / "reward_sweep_top20.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    best = summaries[0] if summaries else {}
    (root / "reward_sweep_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate and rank a reward direction sweep from local output files.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate(args.root)
    write_outputs(args.root, result)
    print(args.root / "reward_sweep_summary.csv")
    print(args.root / "reward_sweep_top20.md")
    print(args.root / "reward_sweep_best.json")
    print(args.root / "reward_sweep_failures.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
