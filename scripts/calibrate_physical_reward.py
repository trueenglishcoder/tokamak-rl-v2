from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    reward: dict[str, float]


TARGETED_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("base_s2_i2_c8_m075", {}),
    Candidate("s3_i2_c12_m070", {"shape_weight": 3.0, "current_weight": 12.0, "current_margin_start_fraction": 0.70}),
    Candidate("s4_i2_c12_m070", {"shape_weight": 4.0, "current_weight": 12.0, "current_margin_start_fraction": 0.70}),
    Candidate("s4_i15_c12_m070", {"shape_weight": 4.0, "ip_weight": 1.5, "current_weight": 12.0, "current_margin_start_fraction": 0.70}),
    Candidate("s3_i25_c12_m070", {"shape_weight": 3.0, "ip_weight": 2.5, "current_weight": 12.0, "current_margin_start_fraction": 0.70}),
    Candidate("s4_i2_c16_m065", {"shape_weight": 4.0, "current_weight": 16.0, "current_margin_start_fraction": 0.65}),
    Candidate("s5_i15_c16_m065", {"shape_weight": 5.0, "ip_weight": 1.5, "current_weight": 16.0, "current_margin_start_fraction": 0.65}),
    Candidate("s5_i2_c16_m065", {"shape_weight": 5.0, "current_weight": 16.0, "current_margin_start_fraction": 0.65}),
    Candidate("s4_i2_c20_m060", {"shape_weight": 4.0, "current_weight": 20.0, "current_margin_start_fraction": 0.60}),
    Candidate("s5_i15_c20_m060", {"shape_weight": 5.0, "ip_weight": 1.5, "current_weight": 20.0, "current_margin_start_fraction": 0.60}),
    Candidate("s4_i2_bad025_c16_m065", {"shape_weight": 4.0, "shape_bad_m": 0.025, "current_weight": 16.0, "current_margin_start_fraction": 0.65}),
    Candidate("s5_i2_bad025_c20_m060", {"shape_weight": 5.0, "shape_bad_m": 0.025, "current_weight": 20.0, "current_margin_start_fraction": 0.60}),
)


SUMMARY_FIELDS = (
    "rank",
    "candidate",
    "status",
    "passed",
    "score",
    "output_dir",
    "export_dir",
    "checkpoint",
    "shape_weight",
    "ip_weight",
    "current_weight",
    "current_margin_start_fraction",
    "shape_bad_m",
    "shape_error_mean_m",
    "shape_error_mean_m_max",
    "ip_error_a",
    "ip_improvement_a",
    "ip_improvement_fraction",
    "current_over_limit_a_max",
    "current_usage_fraction_max",
    "boundary_found",
    "action_rms",
    "physical_cost",
    "mean_return",
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base_config = Path(args.base_config).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "configs"
    run_root = output_root / "runs"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    candidates = _indexed_candidates(args)
    if not candidates:
        raise SystemExit("no reward candidates selected")

    print(f"base_config={base_config}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"candidate_count={len(candidates)}", flush=True)
    print(f"train_env_steps={args.train_env_steps}", flush=True)
    print(f"num_envs={args.num_envs}", flush=True)
    print(f"decision_steps={_decision_steps(args.train_env_steps, args.num_envs)}", flush=True)

    if args.summary_only:
        rows = [_summarize_candidate(candidate=candidate, output_dir=_candidate_output_dir(run_root, index)) for index, candidate in candidates]
        ranked = _rank_rows(rows)
        _write_summary(output_root, ranked)
        best = ranked[0] if ranked else {}
        (output_root / "best_candidate.json").write_text(json.dumps(_jsonable(best), indent=2), encoding="utf-8")
        print("\n=== best candidate ===", flush=True)
        _print_one_line(best)
        print(f"summary_csv={output_root / 'calibration_summary.csv'}", flush=True)
        print(f"summary_json={output_root / 'calibration_summary.json'}", flush=True)
        print(f"best_candidate_json={output_root / 'best_candidate.json'}", flush=True)
        return 0 if bool(best.get("passed", False)) else int(args.exit_code_on_no_pass)

    rows: list[dict[str, Any]] = []
    for item_i, (index, candidate) in enumerate(candidates, start=1):
        print(f"\n=== candidate {index}/{len(candidates)}: {candidate.name} ===", flush=True)
        config_path = _candidate_config_path(config_dir, index, candidate)
        candidate_output = _candidate_output_dir(run_root, index)
        if not args.skip_existing or not (candidate_output / "policy_validation.json").exists():
            _write_candidate_config(base_config, config_path, candidate)
            _run_candidate(args, config_path=config_path, output_dir=candidate_output, candidate=candidate)
        else:
            print(f"skipping existing validation: {candidate_output}", flush=True)
        row = _summarize_candidate(candidate=candidate, output_dir=candidate_output)
        rows.append(row)
        _write_candidate_summary(output_root, index, candidate, row)
        if args.candidate_index is None:
            _write_summary(output_root, rows)
        _print_one_line(row)
        if args.stop_on_pass and bool(row.get("passed", False)):
            print(f"stopping because candidate passed gates: {candidate.name}", flush=True)
            break

    ranked = _rank_rows(rows)
    _write_summary(output_root, ranked)
    best = ranked[0] if ranked else {}
    (output_root / "best_candidate.json").write_text(json.dumps(_jsonable(best), indent=2), encoding="utf-8")
    print("\n=== best candidate ===", flush=True)
    _print_one_line(best)
    print(f"summary_csv={output_root / 'calibration_summary.csv'}", flush=True)
    print(f"summary_json={output_root / 'calibration_summary.json'}", flush=True)
    print(f"best_candidate_json={output_root / 'best_candidate.json'}", flush=True)
    return 0 if bool(best.get("passed", False)) else int(args.exit_code_on_no_pass)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Deterministic physical-reward calibration runner for T15 hold-boundary control.")
    ap.add_argument("--base-config", default="configs/experiments/t15_hold_reset_boundary_ip_stage1_gpu.yaml")
    ap.add_argument("--output-root", default="outputs/t15_reward_calibration")
    ap.add_argument("--candidate-json", default=None, help="Optional JSON file with [{'name': str, 'reward': {...}}, ...].")
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--candidate-index", type=int, default=None, help="Run only one 1-based candidate index. Useful for Slurm arrays.")
    ap.add_argument("--summary-only", action="store_true", help="Do not train; rank existing candidate outputs.")
    ap.add_argument("--train-env-steps", type=int, default=2_000_000)
    ap.add_argument("--eval-env-steps", type=int, default=250_000)
    ap.add_argument("--checkpoint-env-steps", type=int, default=250_000)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sim-compute-backend", choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--sim-gpu-device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--unroll-length", type=int, default=64)
    ap.add_argument("--replay-capacity-episodes", type=int, default=128)
    ap.add_argument("--rollout-chunk-length", type=int, default=64)
    ap.add_argument("--updates-per-rollout-chunk", type=int, default=24)
    ap.add_argument("--action-samples", type=int, default=20)
    ap.add_argument("--actor-update-chunk-size", type=int, default=2048)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--eval-max-steps", type=int, default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="tokamak-rl-v2-reward-calibration")
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    ap.add_argument("--stop-on-pass", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--rerun-existing", action="store_false", dest="skip_existing")
    ap.add_argument("--exit-code-on-no-pass", type=int, default=0)
    return ap


def _indexed_candidates(args: argparse.Namespace) -> list[tuple[int, Candidate]]:
    candidates = _load_candidates(args)
    if args.max_candidates is not None:
        candidates = candidates[: max(0, int(args.max_candidates))]
    indexed = list(enumerate(candidates, start=1))
    if args.candidate_index is not None:
        index = int(args.candidate_index)
        if index < 1 or index > len(indexed):
            raise ValueError(f"--candidate-index must be between 1 and {len(indexed)}, got {index}")
        indexed = [indexed[index - 1]]
    return indexed


def _load_candidates(args: argparse.Namespace) -> list[Candidate]:
    if args.candidate_json is None:
        return list(TARGETED_CANDIDATES)
    raw = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("--candidate-json must contain a list")
    candidates: list[Candidate] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {index} must be an object")
        name = str(item.get("name", f"candidate_{index:02d}"))
        reward = item.get("reward", {})
        if not isinstance(reward, dict):
            raise ValueError(f"candidate {name} reward must be an object")
        candidates.append(Candidate(name=_safe_name(name), reward={str(k): float(v) for k, v in reward.items()}))
    return candidates


def _write_candidate_config(base_config: Path, config_path: Path, candidate: Candidate) -> None:
    data = json.loads(base_config.read_text(encoding="utf-8"))
    data["name"] = f"{data.get('name', 't15')}_{candidate.name}"
    sim = data.get("sim", {})
    if isinstance(sim, dict):
        for key in ("config_path", "initial_currents_path"):
            raw = sim.get(key)
            if raw:
                path = Path(str(raw))
                sim[key] = str(path if path.is_absolute() else (base_config.parent / path).resolve())
    reward = dict(data.get("reward", {}))
    reward.update(candidate.reward)
    data["reward"] = reward
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run_candidate(args: argparse.Namespace, *, config_path: Path, output_dir: Path, candidate: Candidate) -> None:
    decision_steps = _decision_steps(args.train_env_steps, args.num_envs)
    eval_steps = max(1, _decision_steps(args.eval_env_steps, args.num_envs))
    checkpoint_steps = max(1, _decision_steps(args.checkpoint_env_steps, args.num_envs))
    cmd = [
        sys.executable,
        "scripts/train_policy_pipeline.py",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--steps",
        str(decision_steps),
        "--device",
        str(args.device),
        "--sim-compute-backend",
        str(args.sim_compute_backend),
        "--sim-gpu-device",
        str(args.sim_gpu_device),
        "--num-envs",
        str(args.num_envs),
        "--actor-workers",
        "1",
        "--batch-size",
        str(args.batch_size),
        "--unroll-length",
        str(args.unroll_length),
        "--replay-capacity-episodes",
        str(args.replay_capacity_episodes),
        "--rollout-chunk-length",
        str(args.rollout_chunk_length),
        "--updates-per-rollout-chunk",
        str(args.updates_per_rollout_chunk),
        "--hidden-dim",
        "256",
        "--critic-hidden-dim",
        "256",
        "--critic-mlp-hidden-dim",
        "256",
        "--action-samples",
        str(args.action_samples),
        "--actor-update-chunk-size",
        str(args.actor_update_chunk_size),
        "--checkpoint-interval-steps",
        str(checkpoint_steps),
        "--eval-interval-steps",
        str(eval_steps),
        "--allow-failed-gates",
    ]
    if args.eval_episodes is not None:
        cmd.extend(["--eval-episodes", str(args.eval_episodes)])
    if args.eval_max_steps is not None:
        cmd.extend(["--eval-max-steps", str(args.eval_max_steps)])
    if args.wandb and args.wandb_mode != "disabled":
        cmd.extend(
            [
                "--wandb",
                "--wandb-project",
                str(args.wandb_project),
                "--wandb-name",
                candidate.name,
                "--wandb-group",
                Path(args.output_root).name,
                "--wandb-mode",
                str(args.wandb_mode),
            ]
        )
    print("command=" + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _candidate_config_path(config_dir: Path, index: int, candidate: Candidate) -> Path:
    return config_dir / f"{index:02d}_{candidate.name}.json"


def _candidate_output_dir(run_root: Path, index: int) -> Path:
    return run_root / f"{index:02d}"


def _write_candidate_summary(output_root: Path, index: int, candidate: Candidate, row: dict[str, Any]) -> None:
    summary_dir = output_root / "candidate_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"{index:02d}_{candidate.name}.json"
    path.write_text(json.dumps(_jsonable(row), indent=2), encoding="utf-8")


def _summarize_candidate(*, candidate: Candidate, output_dir: Path) -> dict[str, Any]:
    validation_path = output_dir / "policy_validation.json"
    if not validation_path.exists():
        return {
            "candidate": candidate.name,
            "status": "missing_validation",
            "passed": False,
            "score": -float("inf"),
            "output_dir": str(output_dir),
            **candidate.reward,
        }
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    actor = report.get("actor_eval", {}) or {}
    baseline = report.get("no_control", {}) or {}
    gates = report.get("gates", []) or []
    passed = bool(gates) and all(bool(g.get("passed", False)) for g in gates if isinstance(g, dict))
    baseline_ip = _float(baseline.get("ip_error_a"))
    actor_ip = _float(actor.get("ip_error_a"))
    ip_improvement = baseline_ip - actor_ip if _finite(baseline_ip) and _finite(actor_ip) else float("nan")
    ip_improvement_fraction = ip_improvement / max(abs(baseline_ip), 1.0e-12) if _finite(ip_improvement) and _finite(baseline_ip) else float("nan")
    row: dict[str, Any] = {
        "candidate": candidate.name,
        "status": report.get("status", "unknown"),
        "passed": passed,
        "score": _calibration_score(actor=actor, baseline=baseline),
        "output_dir": str(output_dir),
        "export_dir": report.get("export_dir"),
        "checkpoint": report.get("checkpoint"),
        "ip_improvement_a": ip_improvement,
        "ip_improvement_fraction": ip_improvement_fraction,
    }
    config_reward = _reward_from_snapshot(output_dir)
    for key in ("shape_weight", "ip_weight", "current_weight", "current_margin_start_fraction", "shape_bad_m"):
        if key in config_reward:
            row[key] = config_reward[key]
        elif key in candidate.reward:
            row[key] = candidate.reward[key]
    for key in (
        "shape_error_mean_m",
        "shape_error_mean_m_max",
        "ip_error_a",
        "current_over_limit_a_max",
        "current_usage_fraction_max",
        "boundary_found",
        "action_rms",
        "physical_cost",
        "mean_return",
    ):
        if key in actor:
            row[key] = actor[key]
    return row


def _calibration_score(*, actor: dict[str, Any], baseline: dict[str, Any]) -> float:
    shape = _float(actor.get("shape_error_mean_m"))
    ip = _float(actor.get("ip_error_a"))
    baseline_ip = _float(baseline.get("ip_error_a"))
    current_over = _float(actor.get("current_over_limit_a_max", actor.get("current_over_limit_a", 0.0)))
    boundary = _float(actor.get("boundary_found", 0.0))
    action_rms = _float(actor.get("action_rms", 0.0))
    physical = _float(actor.get("physical_cost", float("nan")))

    score = 0.0
    if _finite(shape):
        score -= 100.0 * shape / 0.03
        score -= 500.0 * max(0.0, shape - 0.03) / 0.03
    else:
        score -= 10_000.0
    if _finite(ip):
        score -= 40.0 * ip / 20_000.0
    else:
        score -= 10_000.0
    if _finite(baseline_ip) and _finite(ip):
        improvement = max(0.0, baseline_ip - ip)
        score += 100.0 * improvement / max(abs(baseline_ip), 1.0)
        score += 40.0 * min(improvement, 20_000.0) / 20_000.0
    if _finite(physical):
        score -= 20.0 * physical
    if _finite(current_over) and current_over > 0.0:
        score -= 100_000.0 + min(current_over, 1_000_000.0)
    if _finite(boundary):
        score -= 100_000.0 * max(0.0, 0.999 - boundary)
    else:
        score -= 100_000.0
    if not (_finite(action_rms) and 0.005 <= action_rms < 0.5):
        score -= 10_000.0
    return float(score)


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (_bool(row.get("passed")), _float(row.get("score"))), reverse=True)
    out: list[dict[str, Any]] = []
    for index, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = index
        out.append(item)
    return out


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    ranked = _rank_rows(rows)
    csv_path = output_root / "calibration_summary.csv"
    json_path = output_root / "calibration_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in SUMMARY_FIELDS})
    json_path.write_text(json.dumps(_jsonable(ranked), indent=2), encoding="utf-8")


def _reward_from_snapshot(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "config_snapshot.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    reward = data.get("reward", {})
    return reward if isinstance(reward, dict) else {}


def _print_one_line(row: dict[str, Any]) -> None:
    if not row:
        print("no candidates completed", flush=True)
        return
    print(
        "candidate={candidate} passed={passed} score={score:.3f} "
        "shape={shape_error_mean_m} ip={ip_error_a} ip_improve={ip_improvement_a} "
        "current_over_max={current_over_limit_a_max} boundary={boundary_found} export={export_dir}".format(
            candidate=row.get("candidate"),
            passed=row.get("passed"),
            score=_float(row.get("score")),
            shape_error_mean_m=_fmt(row.get("shape_error_mean_m")),
            ip_error_a=_fmt(row.get("ip_error_a")),
            ip_improvement_a=_fmt(row.get("ip_improvement_a")),
            current_over_limit_a_max=_fmt(row.get("current_over_limit_a_max")),
            boundary_found=_fmt(row.get("boundary_found")),
            export_dir=row.get("export_dir"),
        ),
        flush=True,
    )


def _decision_steps(env_steps: int, num_envs: int) -> int:
    return max(1, (int(env_steps) + int(num_envs) - 1) // int(num_envs))


def _safe_name(value: str) -> str:
    safe = []
    for char in value:
        safe.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(safe).strip("_") or "candidate"


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _bool(value: Any) -> bool:
    return bool(value)


def _fmt(value: Any) -> str:
    number = _float(value)
    if math.isfinite(number):
        return f"{number:.6g}"
    return "nan"


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
