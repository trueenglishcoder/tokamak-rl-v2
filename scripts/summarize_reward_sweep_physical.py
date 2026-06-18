#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LOWER_BETTER = {
    "current_over_limit_fraction_late",
    "current_over_limit_a_max",
    "current_usage_fraction_late_max",
    "terminated_boundary",
    "termination_failure_fraction",
    "shape_error_mean_m_late",
    "shape_error_max_m_late",
    "ip_error_a_late",
    "action_rms_late",
    "delta_action_rms_late",
}
HIGHER_BETTER = {"mean_episode_completion", "full_episode_success", "boundary_found_late_min"}

SUMMARY_FIELDS = [
    "physical_rank",
    "pareto_rank",
    "variant_index",
    "folder",
    "name",
    "status",
    "valid_actor_eval",
    "selection_valid",
    "selection_reason",
    "on_pareto_front",
    "uses_padded_metrics",
    "physical_priority_score",
    "mean_episode_completion",
    "full_episode_success",
    "termination_failure_fraction",
    "boundary_found_late_min",
    "terminated_boundary",
    "shape_error_mean_m_late",
    "shape_error_max_m_late",
    "ip_error_a_late",
    "current_over_limit_a_late",
    "current_over_limit_a_max",
    "current_over_limit_fraction_late",
    "current_usage_fraction_late_max",
    "action_rms_late",
    "delta_action_rms_late",
    "shape_regime",
    "ip_regime",
    "current_regime",
    "derivative_regime",
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
    "terminate_on_current_limit",
    "current_termination_over_limit_a",
    "current_termination_grace_steps",
    "current_hard_termination_fraction",
    "eval_rows",
    "tail_completion",
    "tail_boundary_late_min",
    "tail_shape_error_mean_m_late",
    "tail_ip_error_a_late",
    "tail_current_over_limit_fraction_late",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite(value: Any, default: float = float("nan")) -> float:
    if value in ("", None):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _metric(source: dict[str, Any], *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        value = _finite(source.get(key))
        if math.isfinite(value):
            return value
    return default


def _mean(values: Iterable[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _median(values: Iterable[float]) -> float:
    vals = sorted(value for value in values if math.isfinite(value))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _format_float(value: Any) -> str:
    number = _finite(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 1000.0 or (0.0 < abs(number) < 0.001):
        return f"{number:.6g}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _variant_from_reward_file(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    variant = raw.get("variant")
    return variant if isinstance(variant, dict) else {}


def _discover_variants(root: Path) -> list[dict[str, Any]]:
    manifest = _read_json(root / "variants.json")
    variants = manifest.get("variants")
    if isinstance(variants, list) and variants:
        return [variant for variant in variants if isinstance(variant, dict)]

    discovered: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name[:1] in {"v", "b", "f"}):
        variant = _variant_from_reward_file(run_dir / "reward_variant.json")
        if not variant:
            variant = {"folder": run_dir.name, "name": run_dir.name, "index": len(discovered)}
        discovered.append(variant)
    return discovered


def _folder_for_variant(variant: dict[str, Any]) -> str:
    folder = variant.get("folder")
    if folder:
        return str(folder)
    index = int(_finite(variant.get("index"), default=0.0))
    name = str(variant.get("name") or f"variant_{index:03d}")
    prefix = str(variant.get("folder_prefix") or variant.get("prefix") or "v")
    return f"{prefix}{index:03d}_{name}"


def _tail_metric(rows: list[dict[str, str]], *keys: str) -> float:
    values = [_metric(row, *keys) for row in rows[-3:]]
    return _mean(values)


def _selection_reason(
    metrics: dict[str, Any],
    *,
    min_completion: float,
    min_boundary_late: float,
    max_terminated_boundary: float,
    max_current_over_limit_a_max: float,
    max_current_over_limit_fraction_late: float,
) -> tuple[bool, str]:
    if not metrics["valid_actor_eval"]:
        return False, "missing_actor_eval"
    checks = [
        ("low_full_episode_success", metrics["full_episode_success"] >= min_completion),
        ("low_completion", metrics["mean_episode_completion"] >= min_completion),
        ("lost_late_boundary", metrics["boundary_found_late_min"] >= min_boundary_late),
        ("boundary_termination", metrics["terminated_boundary"] <= max_terminated_boundary),
        ("large_current_max", metrics["current_over_limit_a_max"] <= max_current_over_limit_a_max),
        ("large_current_fraction", metrics["current_over_limit_fraction_late"] <= max_current_over_limit_fraction_late),
    ]
    for reason, passed in checks:
        if not bool(passed):
            return False, reason
    return True, "ok"


def _physical_priority_score(row: dict[str, Any]) -> float:
    completion_gap = max(0.0, 1.0 - _finite(row["mean_episode_completion"], 0.0))
    full_success_gap = max(0.0, 1.0 - _finite(row.get("full_episode_success"), _finite(row["mean_episode_completion"], 0.0)))
    boundary_gap = max(0.0, 1.0 - _finite(row["boundary_found_late_min"], 0.0))
    current_fraction = _finite(row["current_over_limit_fraction_late"], 1.0)
    current_max = _finite(row["current_over_limit_a_max"], 1.0e9)
    current_usage_max = _finite(row["current_usage_fraction_late_max"], 10.0)
    shape_mean = _finite(row["shape_error_mean_m_late"], 10.0)
    shape_max = _finite(row["shape_error_max_m_late"], 10.0)
    ip_error = _finite(row["ip_error_a_late"], 1.0e9)
    action = _finite(row["action_rms_late"], 10.0)
    delta_action = _finite(row["delta_action_rms_late"], 10.0)
    terminated_boundary = _finite(row["terminated_boundary"], 1.0)
    termination_failure = _finite(row.get("termination_failure_fraction"), full_success_gap)
    return float(
        500.0 * completion_gap
        + 500.0 * full_success_gap
        + 1000.0 * boundary_gap
        + 200.0 * termination_failure
        + 100.0 * terminated_boundary
        + 60.0 * current_fraction
        + 6.0 * current_max / 20000.0
        + 20.0 * max(0.0, current_usage_max - 1.0)
        + 2.0 * shape_mean / 0.03
        + 1.0 * shape_max / 0.08
        + 2.0 * ip_error / 25000.0
        + 0.5 * action / 0.25
        + 0.5 * delta_action / 0.05
    )


def summarize_variant(
    root: Path,
    variant: dict[str, Any],
    *,
    min_completion: float,
    min_boundary_late: float,
    max_terminated_boundary: float,
    max_current_over_limit_a_max: float,
    max_current_over_limit_fraction_late: float,
) -> dict[str, Any]:
    folder = _folder_for_variant(variant)
    run_dir = root / folder
    reward_file_variant = _variant_from_reward_file(run_dir / "reward_variant.json")
    if reward_file_variant:
        variant = {**variant, **reward_file_variant}
    reward = variant.get("reward") if isinstance(variant.get("reward"), dict) else {}
    sim = variant.get("sim") if isinstance(variant.get("sim"), dict) else {}
    validation = _read_json(run_dir / "policy_validation.json")
    actor_eval = validation.get("actor_eval")
    actor_eval = actor_eval if isinstance(actor_eval, dict) else {}
    eval_rows = _read_csv(run_dir / "eval_history.csv")
    uses_padded = any(str(key).startswith("padded_") for key in actor_eval)
    completion_metric = _metric(actor_eval, "mean_episode_completion", "episode_progress", default=0.0)
    full_success_metric = _metric(actor_eval, "full_episode_success", "mean_episode_completion", "episode_progress", default=0.0)
    termination_failure_metric = _metric(actor_eval, "termination_failure_fraction", default=max(0.0, 1.0 - full_success_metric))

    row: dict[str, Any] = {
        "physical_rank": "",
        "pareto_rank": "",
        "variant_index": int(_finite(variant.get("index"), default=-1.0)),
        "folder": folder,
        "name": str(variant.get("name") or folder),
        "status": str(validation.get("status") or ("missing" if not run_dir.exists() else "missing_policy_validation")),
        "valid_actor_eval": bool(actor_eval),
        "selection_valid": False,
        "selection_reason": "",
        "on_pareto_front": False,
        "uses_padded_metrics": uses_padded,
        "physical_priority_score": float("inf"),
        "mean_episode_completion": completion_metric,
        "full_episode_success": full_success_metric,
        "termination_failure_fraction": termination_failure_metric,
        "boundary_found_late_min": _metric(actor_eval, "padded_boundary_found_late_min", "boundary_found_late_min", "boundary_found_min", "boundary_found", default=0.0),
        "terminated_boundary": _metric(actor_eval, "terminated_boundary_late", "terminated_boundary", default=1.0),
        "shape_error_mean_m_late": _metric(actor_eval, "padded_shape_error_mean_m_late", "shape_error_mean_m_late", "shape_error_mean_m", default=float("nan")),
        "shape_error_max_m_late": _metric(actor_eval, "padded_shape_error_max_m_late", "shape_error_max_m_late", "shape_error_max_m", default=float("nan")),
        "ip_error_a_late": _metric(actor_eval, "padded_ip_error_a_late", "ip_error_a_late", "ip_error_a", default=float("nan")),
        "current_over_limit_a_late": _metric(actor_eval, "padded_current_over_limit_a_late", "current_over_limit_a_late", "current_over_limit_a", default=float("nan")),
        "current_over_limit_a_max": _metric(actor_eval, "padded_current_over_limit_a_late_max", "padded_current_over_limit_a_max", "current_over_limit_a_late_max", "current_over_limit_a_max", "current_over_limit_a", default=float("nan")),
        "current_over_limit_fraction_late": _metric(actor_eval, "padded_current_over_limit_fraction_late", "current_over_limit_fraction_late", "current_over_limit_fraction", default=float("nan")),
        "current_usage_fraction_late_max": _metric(actor_eval, "padded_current_usage_fraction_late_max", "padded_current_usage_fraction_max", "current_usage_fraction_late_max", "current_usage_fraction_max", "current_usage_fraction", default=float("nan")),
        "action_rms_late": _metric(actor_eval, "action_rms_late", "action_rms", default=float("nan")),
        "delta_action_rms_late": _metric(actor_eval, "delta_action_rms_late", "delta_action_rms", default=float("nan")),
        "shape_regime": str(variant.get("shape_regime") or ""),
        "ip_regime": str(variant.get("ip_regime") or ""),
        "current_regime": str(variant.get("current_regime") or ""),
        "derivative_regime": str(variant.get("derivative_regime") or variant.get("actuator_regime") or ""),
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
        "terminate_on_current_limit": sim.get("terminate_on_current_limit", ""),
        "current_termination_over_limit_a": sim.get("current_termination_over_limit_a", ""),
        "current_termination_grace_steps": sim.get("current_termination_grace_steps", ""),
        "current_hard_termination_fraction": sim.get("current_hard_termination_fraction", ""),
        "eval_rows": len(eval_rows),
        "tail_completion": _tail_metric(eval_rows, "mean_episode_completion", "episode_progress"),
        "tail_boundary_late_min": _tail_metric(eval_rows, "padded_boundary_found_late_min", "boundary_found_late_min", "boundary_found_min", "boundary_found"),
        "tail_shape_error_mean_m_late": _tail_metric(eval_rows, "padded_shape_error_mean_m_late", "shape_error_mean_m_late", "shape_error_mean_m"),
        "tail_ip_error_a_late": _tail_metric(eval_rows, "padded_ip_error_a_late", "ip_error_a_late", "ip_error_a"),
        "tail_current_over_limit_fraction_late": _tail_metric(eval_rows, "padded_current_over_limit_fraction_late", "current_over_limit_fraction_late", "current_over_limit_fraction"),
    }
    row["selection_valid"], row["selection_reason"] = _selection_reason(
        row,
        min_completion=min_completion,
        min_boundary_late=min_boundary_late,
        max_terminated_boundary=max_terminated_boundary,
        max_current_over_limit_a_max=max_current_over_limit_a_max,
        max_current_over_limit_fraction_late=max_current_over_limit_fraction_late,
    )
    row["physical_priority_score"] = _physical_priority_score(row)
    return row


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    strictly_better = False
    for key in HIGHER_BETTER:
        av = _finite(a.get(key))
        bv = _finite(b.get(key))
        if not math.isfinite(av) or not math.isfinite(bv):
            return False
        if av < bv:
            return False
        strictly_better = strictly_better or av > bv
    for key in LOWER_BETTER:
        av = _finite(a.get(key))
        bv = _finite(b.get(key))
        if not math.isfinite(av) or not math.isfinite(bv):
            return False
        if av > bv:
            return False
        strictly_better = strictly_better or av < bv
    return strictly_better


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    valid = [row for row in rows if row["selection_valid"]]
    for candidate in valid:
        if not any(_dominates(other, candidate) for other in valid if other is not candidate):
            front.append(candidate)
    front.sort(key=lambda row: (_finite(row["physical_priority_score"], float("inf")), int(row["variant_index"])))
    for rank, row in enumerate(front, start=1):
        row["pareto_rank"] = rank
        row["on_pareto_front"] = True
    return front


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (not row["selection_valid"], _finite(row["physical_priority_score"], float("inf")), int(row["variant_index"])))
    for rank, row in enumerate(ranked, start=1):
        row["physical_rank"] = rank
    return ranked


def _regime_key(row: dict[str, Any], kind: str) -> tuple[str, str]:
    key = {
        "shape": "shape_regime",
        "ip": "ip_regime",
        "current": "current_regime",
        "derivative": "derivative_regime",
    }[kind]
    return kind, str(row.get(key) or "")


def regime_summary(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for kind in ("shape", "ip", "current", "derivative"):
            groups[_regime_key(row, kind)].append(row)

    top_folders = {row["folder"] for row in rank_rows(rows)[:top_n]}
    summaries: list[dict[str, Any]] = []
    for (kind, regime), group in sorted(groups.items()):
        valid = [row for row in group if row["selection_valid"]]
        summaries.append(
            {
                "regime_kind": kind,
                "regime": regime,
                "run_count": len(group),
                "selection_valid_count": len(valid),
                "pareto_count": sum(bool(row["on_pareto_front"]) for row in group),
                f"top{top_n}_count": sum(row["folder"] in top_folders for row in group),
                "median_physical_priority_score": _median(_finite(row["physical_priority_score"]) for row in valid),
                "median_completion": _median(_finite(row["mean_episode_completion"]) for row in valid),
                "median_boundary_late_min": _median(_finite(row["boundary_found_late_min"]) for row in valid),
                "median_shape_error_mean_m_late": _median(_finite(row["shape_error_mean_m_late"]) for row in valid),
                "median_ip_error_a_late": _median(_finite(row["ip_error_a_late"]) for row in valid),
                "median_current_over_limit_a_max": _median(_finite(row["current_over_limit_a_max"]) for row in valid),
                "median_current_over_limit_fraction_late": _median(_finite(row["current_over_limit_fraction_late"]) for row in valid),
            }
        )
    summaries.sort(key=lambda row: (row["regime_kind"], -int(row["selection_valid_count"]), _finite(row["median_physical_priority_score"], float("inf")), row["regime"]))
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _row_for_json(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = None
        else:
            result[key] = value
    return result


def _short_row(row: dict[str, Any] | None) -> str:
    if row is None:
        return "none"
    return (
        f"`{row['folder']}` "
        f"(score={_format_float(row['physical_priority_score'])}, "
        f"completion={_format_float(row['mean_episode_completion'])}, "
        f"boundary_late={_format_float(row['boundary_found_late_min'])}, "
        f"shape_late={_format_float(row['shape_error_mean_m_late'])} m, "
        f"Ip_late={_format_float(row['ip_error_a_late'])} A, "
        f"current_max={_format_float(row['current_over_limit_a_max'])} A, "
        f"current_frac_late={_format_float(row['current_over_limit_fraction_late'])})"
    )


def write_report(out_dir: Path, rows: list[dict[str, Any]], front: list[dict[str, Any]], regimes: list[dict[str, Any]], top_n: int) -> None:
    valid = [row for row in rows if row["selection_valid"]]
    best = valid[0] if valid else None
    best_pareto = front[0] if front else None
    backup = None
    if valid:
        for row in valid:
            if best is None or row["folder"] != best["folder"]:
                backup = row
                break
    best_regime = min(
        (row for row in regimes if row["selection_valid_count"]),
        key=lambda row: (-int(row[f"top{top_n}_count"]), -int(row["pareto_count"]), _finite(row["median_physical_priority_score"], float("inf"))),
        default=None,
    )
    lines = [
        "# Physical Reward Sweep Selection",
        "",
        "Selection is based on held-out actor physical metrics from `policy_validation.json -> actor_eval`.",
        "Reward value, `physical_cost`, `mean_return`, and the older sweep `selected_score` are not used for ranking.",
        "",
        "## Result",
        "",
        f"- Valid candidates after hard physical filters: `{len(valid)}` / `{len(rows)}`.",
        f"- Best individual candidate: {_short_row(best)}.",
        f"- Best Pareto-front candidate: {_short_row(best_pareto)}.",
        f"- Recommended long-run reward: `{best['folder']}`." if best else "- Recommended long-run reward: none.",
        f"- Backup reward if current abuse grows: `{backup['folder']}`." if backup else "- Backup reward if current abuse grows: none.",
        "",
        "## Best Regime Cluster",
        "",
    ]
    if best_regime:
        lines.append(
            f"- `{best_regime['regime_kind']}={best_regime['regime']}`: "
            f"valid={best_regime['selection_valid_count']}, pareto={best_regime['pareto_count']}, "
            f"top{top_n}={best_regime[f'top{top_n}_count']}, "
            f"median_score={_format_float(best_regime['median_physical_priority_score'])}."
        )
    else:
        lines.append("- No valid regime cluster found.")
    lines.extend(["", "## Top Physical Candidates", ""])
    lines.append("| Rank | Variant | Completion | Boundary Late | Shape Late m | Ip Late A | Current Max A | Current Fraction Late |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in valid[:top_n]:
        lines.append(
            f"| {row['physical_rank']} | `{row['folder']}` | "
            f"{_format_float(row['mean_episode_completion'])} | "
            f"{_format_float(row['boundary_found_late_min'])} | "
            f"{_format_float(row['shape_error_mean_m_late'])} | "
            f"{_format_float(row['ip_error_a_late'])} | "
            f"{_format_float(row['current_over_limit_a_max'])} | "
            f"{_format_float(row['current_over_limit_fraction_late'])} |"
        )
    lines.extend(["", "## Pareto Front", ""])
    lines.append("| Pareto Rank | Variant | Shape Late m | Ip Late A | Current Max A | Current Fraction Late |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
    for row in front[:top_n]:
        lines.append(
            f"| {row['pareto_rank']} | `{row['folder']}` | "
            f"{_format_float(row['shape_error_mean_m_late'])} | "
            f"{_format_float(row['ip_error_a_late'])} | "
            f"{_format_float(row['current_over_limit_a_max'])} | "
            f"{_format_float(row['current_over_limit_fraction_late'])} |"
        )
    (out_dir / "physical_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
    root: Path,
    out_dir: Path,
    *,
    top: int,
    min_completion: float,
    min_boundary_late: float,
    max_terminated_boundary: float,
    max_current_over_limit_a_max: float,
    max_current_over_limit_fraction_late: float,
) -> dict[str, Any]:
    variants = _discover_variants(root)
    if not variants:
        raise ValueError(f"No variants found under {root}")
    rows = [
        summarize_variant(
            root,
            variant,
            min_completion=min_completion,
            min_boundary_late=min_boundary_late,
            max_terminated_boundary=max_terminated_boundary,
            max_current_over_limit_a_max=max_current_over_limit_a_max,
            max_current_over_limit_fraction_late=max_current_over_limit_fraction_late,
        )
        for variant in variants
    ]
    ranked = rank_rows(rows)
    front = pareto_front(ranked)
    regimes = regime_summary(ranked, top)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "physical_sweep_summary.csv", ranked, SUMMARY_FIELDS)
    _write_csv(out_dir / "physical_top_runs.csv", [row for row in ranked if row["selection_valid"]][:top], SUMMARY_FIELDS)
    _write_csv(out_dir / "physical_pareto_front.csv", front, SUMMARY_FIELDS)
    _write_csv(out_dir / "physical_regime_summary.csv", regimes)
    best = next((row for row in ranked if row["selection_valid"]), None)
    if best is None:
        best = next((row for row in ranked if row["valid_actor_eval"] and math.isfinite(_finite(row["physical_priority_score"], float("inf")))), None)
    if best is None and ranked:
        best = ranked[0]
    best_payload = {
        "source_root": str(root),
        "best_candidate": _row_for_json(best),
        "best_pareto_candidate": _row_for_json(front[0] if front else None),
        "best_candidate_passed_hard_filters": bool(best and best["selection_valid"]),
        "valid_candidates": sum(bool(row["selection_valid"]) for row in ranked),
        "total_candidates": len(ranked),
        "thresholds": {
            "min_completion": min_completion,
            "min_boundary_late": min_boundary_late,
            "max_terminated_boundary": max_terminated_boundary,
            "max_current_over_limit_a_max": max_current_over_limit_a_max,
            "max_current_over_limit_fraction_late": max_current_over_limit_fraction_late,
        },
    }
    (out_dir / "physical_best_candidate.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    write_report(out_dir, ranked, front, regimes, top)
    return best_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select reward-sweep candidates using held-out physical actor metrics.")
    parser.add_argument("root", type=Path, help="Sweep root containing variants.json and vNNN_* run folders.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to ROOT/physical_reward_selection.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-completion", type=float, default=0.95)
    parser.add_argument("--min-boundary-late", type=float, default=0.999)
    parser.add_argument("--max-terminated-boundary", type=float, default=0.001)
    parser.add_argument("--max-current-over-limit-a-max", type=float, default=250000.0)
    parser.add_argument("--max-current-over-limit-fraction-late", type=float, default=0.5)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or (args.root / "physical_reward_selection")
    summarize(
        args.root,
        out_dir,
        top=args.top,
        min_completion=args.min_completion,
        min_boundary_late=args.min_boundary_late,
        max_terminated_boundary=args.max_terminated_boundary,
        max_current_over_limit_a_max=args.max_current_over_limit_a_max,
        max_current_over_limit_fraction_late=args.max_current_over_limit_fraction_late,
    )
    for name in [
        "physical_sweep_summary.csv",
        "physical_top_runs.csv",
        "physical_pareto_front.csv",
        "physical_regime_summary.csv",
        "physical_best_candidate.json",
        "physical_selection_report.md",
    ]:
        print(out_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
