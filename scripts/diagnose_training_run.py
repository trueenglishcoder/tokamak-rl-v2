from __future__ import annotations

import argparse
import csv
import json
import math
import re
from bisect import bisect_left
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "diagnostics" / "training_run_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _load_artifacts(run_dir=run_dir, job_id=args.job_id, slurm_log_dir=Path(args.slurm_log_dir) if args.slurm_log_dir else None)
    report = _build_report(run_dir=run_dir, artifacts=artifacts)

    (out_dir / "training_diagnosis.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (out_dir / "training_diagnosis.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_correlation_csv(out_dir / "episode_length_correlations.csv", report)

    print(out_dir / "training_diagnosis.md")
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Diagnose why a training run stopped and which logged metrics move with "
            "replay_mean_episode_length. Uses only stdlib so it can run on the login node."
        )
    )
    ap.add_argument("run_dir", help="Training output directory, e.g. outputs/<run_name>.")
    ap.add_argument("--job-id", default=None, help="Optional Slurm job id for log matching.")
    ap.add_argument("--slurm-log-dir", default="slurm_logs", help="Directory containing Slurm .out/.err logs.")
    ap.add_argument("--out-dir", default=None)
    return ap


def _load_artifacts(*, run_dir: Path, job_id: str | None, slurm_log_dir: Path | None) -> dict[str, Any]:
    csvs = {
        "losses": _read_csv(run_dir / "losses.csv"),
        "replay_health": _read_csv(run_dir / "replay_health.csv"),
        "reward_components": _read_csv(run_dir / "reward_components.csv"),
        "eval_history": _read_csv(run_dir / "eval_history.csv"),
    }
    logs = _read_slurm_logs(slurm_log_dir=slurm_log_dir, job_id=job_id)
    config = _load_config(run_dir)
    return {
        "metrics": _read_json(run_dir / "metrics.json"),
        "policy_validation": _read_json(run_dir / "policy_validation.json"),
        "config": config,
        "csvs": csvs,
        "slurm_logs": logs,
    }


def _build_report(*, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    csvs = artifacts["csvs"]
    length_rows = _best_episode_length_rows(csvs)
    length_summary = _episode_length_summary(length_rows)
    final_steps = _final_steps(csvs)
    stop_summary = _stop_summary(
        metrics=artifacts["metrics"],
        policy_validation=artifacts["policy_validation"],
        config=artifacts["config"],
        final_steps=final_steps,
        logs=artifacts["slurm_logs"],
    )
    correlations = {
        name: _correlations_against_episode_length(rows=rows, length_rows=length_rows)
        for name, rows in csvs.items()
        if rows
    }
    trends = {name: _trend_summary(rows) for name, rows in csvs.items() if rows}
    suspects = _rank_suspects(correlations)
    return {
        "run_dir": str(run_dir),
        "stop_summary": stop_summary,
        "episode_length_summary": length_summary,
        "final_steps_by_file": final_steps,
        "trends": trends,
        "episode_length_correlations": correlations,
        "top_episode_length_correlates": suspects,
        "artifact_status": {
            "metrics_json": artifacts["metrics"].get("status", "ok"),
            "policy_validation_json": artifacts["policy_validation"].get("status", "ok"),
            **{f"{name}_csv_rows": len(rows) for name, rows in csvs.items()},
            "slurm_log_files": [str(item["path"]) for item in artifacts["slurm_logs"]],
        },
    }


def _stop_summary(
    *,
    metrics: dict[str, Any],
    policy_validation: dict[str, Any],
    config: dict[str, Any],
    final_steps: dict[str, int],
    logs: list[dict[str, Any]],
) -> dict[str, Any]:
    configured_steps = _nested_int(config, ("training", "steps"))
    metrics_steps = _int(metrics.get("steps"))
    metrics_env_steps = _int(metrics.get("env_steps"))
    train_result = policy_validation.get("train_result") if isinstance(policy_validation, dict) else None
    validation_train_steps = _int(train_result.get("steps")) if isinstance(train_result, dict) else None
    validation_train_env_steps = _int(train_result.get("env_steps")) if isinstance(train_result, dict) else None
    observed_final = max([0, *(step for step in final_steps.values() if step is not None)])
    log_flags = _log_flags(logs)
    ratio = _safe_ratio(metrics_steps or validation_train_steps or observed_final, configured_steps)
    verdict = _stop_verdict(
        configured_steps=configured_steps,
        metrics=metrics,
        policy_validation=policy_validation,
        observed_final=observed_final,
        ratio=ratio,
        log_flags=log_flags,
    )
    return {
        "configured_steps": configured_steps,
        "metrics_status": metrics.get("status"),
        "metrics_steps": metrics_steps,
        "metrics_env_steps": metrics_env_steps,
        "metrics_early_stop_step": metrics.get("early_stop_step"),
        "policy_validation_status": policy_validation.get("status"),
        "policy_validation_reason": policy_validation.get("reason") or policy_validation.get("error"),
        "policy_validation_train_steps": validation_train_steps,
        "policy_validation_train_env_steps": validation_train_env_steps,
        "observed_final_logged_step": observed_final,
        "completed_fraction": ratio,
        "slurm_log_flags": log_flags,
        "verdict": verdict,
    }


def _stop_verdict(
    *,
    configured_steps: int | None,
    metrics: dict[str, Any],
    policy_validation: dict[str, Any],
    observed_final: int,
    ratio: float | None,
    log_flags: dict[str, bool],
) -> str:
    if log_flags.get("nccl_watchdog"):
        return "Training was killed by NCCL watchdog or distributed-process failure before reaching the configured step count."
    if log_flags.get("oom"):
        return "Training likely died from an out-of-memory condition."
    if log_flags.get("sigterm") or "interrupted" in str(policy_validation.get("status", "")).lower():
        return "Training received a termination signal before normal completion."
    if log_flags.get("timeout"):
        return "Training likely hit a scheduler or launcher timeout."
    status = str(metrics.get("status", "")).lower()
    if status == "early_stopped":
        return "Trainer early-stopped according to metrics.json."
    if configured_steps and observed_final and ratio is not None and ratio < 0.95:
        if status == "completed":
            return "Trainer reported completed before the configured step target; inspect generated config/step accounting."
        return "Run stopped before the configured step target; inspect Slurm state and policy_validation.json."
    if status == "completed":
        return "Trainer reports normal completion."
    return "Stop reason is not explicit in available artifacts."


def _episode_length_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    pairs = [(step, value) for step, value in _series(rows, "replay_mean_episode_length") if math.isfinite(value)]
    if not pairs:
        return {"status": "missing_replay_mean_episode_length"}
    values = [value for _, value in pairs]
    first = _window_mean(values, "first")
    middle = _window_mean(values, "middle")
    last = _window_mean(values, "last")
    min_step, min_value = min(pairs, key=lambda item: item[1])
    max_step, max_value = max(pairs, key=lambda item: item[1])
    return {
        "status": "ok",
        "rows": len(pairs),
        "first_mean": first,
        "middle_mean": middle,
        "last_mean": last,
        "first_to_last_delta": None if first is None or last is None else last - first,
        "min": min_value,
        "min_step": min_step,
        "max": max_value,
        "max_step": max_step,
    }


def _correlations_against_episode_length(*, rows: list[dict[str, str]], length_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not rows or not length_rows:
        return []
    length_pairs = [(step, value) for step, value in _series(length_rows, "replay_mean_episode_length") if math.isfinite(value)]
    if len(length_pairs) < 3:
        return []
    length_steps = [step for step, _ in length_pairs]
    length_values = [value for _, value in length_pairs]
    out: list[dict[str, Any]] = []
    keys = sorted({key for row in rows for key in row if key not in {"step", "env_step", "global_step", "decision_step"}})
    for key in keys:
        if key == "replay_mean_episode_length":
            continue
        xs: list[float] = []
        ys: list[float] = []
        pairs = _series(rows, key)
        for step, value in pairs:
            if not math.isfinite(value):
                continue
            length = _nearest_value(step, length_steps, length_values)
            if length is None or not math.isfinite(length):
                continue
            xs.append(value)
            ys.append(length)
        corr = _corr(xs, ys)
        if corr is None:
            continue
        first = _window_mean(xs, "first")
        last = _window_mean(xs, "last")
        out.append(
            {
                "column": key,
                "corr_with_replay_mean_episode_length": corr,
                "samples": len(xs),
                "first_mean": first,
                "last_mean": last,
                "first_to_last_delta": None if first is None or last is None else last - first,
            }
        )
    out.sort(key=lambda item: abs(float(item["corr_with_replay_mean_episode_length"])), reverse=True)
    return out[:30]


def _rank_suspects(correlations: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    important_tokens = (
        "terminated",
        "error",
        "loss",
        "cost",
        "over_limit",
        "saturation",
        "usage",
        "action",
        "q_",
        "q",
        "boundary",
        "ip",
        "current",
        "derivative",
    )
    rows: list[dict[str, Any]] = []
    for source, items in correlations.items():
        for item in items:
            col = str(item["column"])
            if not any(token in col for token in important_tokens):
                continue
            row = dict(item)
            row["source"] = source
            rows.append(row)
    rows.sort(key=lambda item: abs(float(item["corr_with_replay_mean_episode_length"])), reverse=True)
    return rows[:25]


def _trend_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"status": "missing"}
    keys = sorted({key for row in rows for key in row if key not in {"step", "env_step", "global_step", "decision_step"}})
    out: dict[str, Any] = {"status": "ok", "rows": len(rows)}
    for key in keys:
        values = [value for _, value in _series(rows, key) if math.isfinite(value)]
        if not values:
            continue
        first = _window_mean(values, "first")
        middle = _window_mean(values, "middle")
        last = _window_mean(values, "last")
        if first is None or last is None:
            continue
        out[key] = {
            "first_mean": first,
            "middle_mean": middle,
            "last_mean": last,
            "first_to_last_delta": last - first,
        }
    return out


def _write_correlation_csv(path: Path, report: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for source, items in report.get("episode_length_correlations", {}).items():
        for item in items:
            rows.append({"source": source, **item})
    fields = ["source", "column", "corr_with_replay_mean_episode_length", "samples", "first_mean", "last_mean", "first_to_last_delta"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _markdown_report(report: dict[str, Any]) -> str:
    stop = report["stop_summary"]
    length = report["episode_length_summary"]
    lines = [
        "# Training Run Diagnosis",
        "",
        "## Stop Summary",
        "",
        f"- Verdict: {stop.get('verdict')}",
        f"- Configured steps: {_fmt(stop.get('configured_steps'))}",
        f"- metrics.json status: {stop.get('metrics_status')} at steps={_fmt(stop.get('metrics_steps'))}, env_steps={_fmt(stop.get('metrics_env_steps'))}",
        f"- policy_validation status: {stop.get('policy_validation_status')} ({stop.get('policy_validation_reason')})",
        f"- Final logged step: {_fmt(stop.get('observed_final_logged_step'))}",
        f"- Completed fraction: {_fmt_pct(stop.get('completed_fraction'))}",
        f"- Slurm/log flags: {json.dumps(stop.get('slurm_log_flags', {}), sort_keys=True)}",
        "",
        "## Replay Episode Length",
        "",
    ]
    if length.get("status") == "ok":
        lines.extend(
            [
                f"- First window mean: {_fmt(length.get('first_mean'))}",
                f"- Middle window mean: {_fmt(length.get('middle_mean'))}",
                f"- Last window mean: {_fmt(length.get('last_mean'))}",
                f"- First-to-last delta: {_fmt(length.get('first_to_last_delta'))}",
                f"- Min: {_fmt(length.get('min'))} at step {_fmt(length.get('min_step'))}",
                f"- Max: {_fmt(length.get('max'))} at step {_fmt(length.get('max_step'))}",
            ]
        )
    else:
        lines.append(f"- {length.get('status')}")
    lines.extend(["", "## Strongest Correlates With Episode Length", ""])
    suspects = report.get("top_episode_length_correlates", [])
    if not suspects:
        lines.append("- No usable correlations found.")
    else:
        lines.append("| source | column | corr | first | last | delta |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for item in suspects[:15]:
            lines.append(
                "| {source} | `{column}` | {corr} | {first} | {last} | {delta} |".format(
                    source=item.get("source"),
                    column=item.get("column"),
                    corr=_fmt(item.get("corr_with_replay_mean_episode_length")),
                    first=_fmt(item.get("first_mean")),
                    last=_fmt(item.get("last_mean")),
                    delta=_fmt(item.get("first_to_last_delta")),
                )
            )
    lines.extend(["", "## Eval Metric Drift", ""])
    eval_trends = report.get("trends", {}).get("eval_history", {})
    wanted = [
        "mean_episode_completion",
        "shape_error_mean_m_late",
        "shape_error_max_m_late",
        "ip_error_a_late",
        "current_over_limit_fraction_late",
        "current_over_limit_a_late_max",
        "action_saturation_fraction_late",
        "action_rms_late",
        "selection_score",
    ]
    lines.append("| metric | first | last | delta |")
    lines.append("|---|---:|---:|---:|")
    for key in wanted:
        item = eval_trends.get(key)
        if not isinstance(item, dict):
            continue
        lines.append(
            f"| `{key}` | {_fmt(item.get('first_mean'))} | {_fmt(item.get('last_mean'))} | {_fmt(item.get('first_to_last_delta'))} |"
        )
    lines.extend(["", "## Artifact Status", "", "```json", json.dumps(report.get("artifact_status", {}), indent=2), "```", ""])
    return "\n".join(lines)


def _best_episode_length_rows(csvs: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    for name in ("replay_health", "losses"):
        rows = csvs.get(name, [])
        if rows and "replay_mean_episode_length" in rows[0]:
            return rows
    return []


def _load_config(run_dir: Path) -> dict[str, Any]:
    for candidate in [run_dir / "config_snapshot.json", *sorted((run_dir / "generated_configs").glob("*.json"))]:
        data = _read_json(candidate)
        if data.get("status") != "missing":
            return data
    return {"status": "missing"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "path": str(path), "error": repr(exc)}
    return value if isinstance(value, dict) else {"value": value}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_slurm_logs(*, slurm_log_dir: Path | None, job_id: str | None) -> list[dict[str, Any]]:
    if slurm_log_dir is None or job_id is None:
        return []
    root = slurm_log_dir
    if not root.is_absolute():
        root = Path.cwd() / root
    paths = sorted(root.glob(f"*{job_id}*.out")) + sorted(root.glob(f"*{job_id}*.err"))
    out: list[dict[str, Any]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            out.append({"path": str(path), "error": repr(exc), "tail": ""})
            continue
        lines = text.splitlines()
        out.append({"path": str(path), "line_count": len(lines), "tail": "\n".join(lines[-240:])})
    return out


def _log_flags(logs: list[dict[str, Any]]) -> dict[str, bool]:
    text = "\n".join(str(item.get("tail", "")) for item in logs).lower()
    return {
        "nccl_watchdog": "processgroupnccl" in text or "nccl watchdog" in text or "watchdog got stuck" in text,
        "sigterm": "sigterm" in text or "received signal 15" in text or "received sigterm" in text,
        "oom": "out of memory" in text or "cuda oom" in text or "oom-kill" in text or "oom_kill" in text,
        "timeout": "time limit" in text or "timeout" in text or "timed out" in text,
        "child_failed": "childfailederror" in text or "child failed" in text,
        "traceback": "traceback (most recent call last)" in text,
        "cancelled": "cancelled" in text or "canceled" in text,
    }


def _final_steps(csvs: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, rows in csvs.items():
        steps = [_step(row) for row in rows]
        finite = [step for step in steps if step is not None]
        if finite:
            out[name] = max(finite)
    return out


def _series(rows: list[dict[str, str]], key: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        step = _step(row)
        if step is None:
            step = index
        out.append((step, _float(row.get(key))))
    return out


def _step(row: dict[str, str]) -> int | None:
    for key in ("step", "env_step", "global_step", "decision_step"):
        if key in row:
            value = _int(row.get(key))
            if value is not None:
                return value
    return None


def _nearest_value(step: int, steps: list[int], values: list[float]) -> float | None:
    if not steps:
        return None
    idx = bisect_left(steps, step)
    candidates: list[int] = []
    if idx < len(steps):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best = min(candidates, key=lambda i: abs(steps[i] - step))
    return values[best]


def _window_mean(values: list[float], which: str) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    n = len(finite)
    width = max(1, n // 5)
    if which == "first":
        sample = finite[:width]
    elif which == "middle":
        start = max(0, n // 2 - width // 2)
        sample = finite[start : start + width]
    elif which == "last":
        sample = finite[-width:]
    else:
        sample = finite
    return sum(sample) / len(sample)


def _corr(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 1.0e-24 or vy <= 1.0e-24:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def _nested_int(data: dict[str, Any], path: tuple[str, ...]) -> int | None:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return _int(cur)


def _safe_ratio(numer: int | None, denom: int | None) -> float | None:
    if numer is None or denom in (None, 0):
        return None
    return float(numer) / float(denom)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = re.sub(r"[^0-9+\-.eE]", "", value)
        return int(float(value))
    except Exception:
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return "n/a"
    if abs(x) >= 10000 or (0 < abs(x) < 0.001):
        return f"{x:.4g}"
    return f"{x:.4f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
