from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


PROFILE = "tcvdelta_t15boundary_36x2m"
CANDIDATES_PER_TASK = 3
ARRAY_TASKS = 12
VARIANT_COUNT = 36

SURVIVAL_REGIMES: tuple[tuple[str, dict[str, float]], ...] = (
    ("t0", {"terminal_reward": -5.0}),
    ("t1", {"terminal_reward": -20.0}),
    ("t2", {"terminal_reward": -80.0}),
)

TRACKING_REGIMES: tuple[tuple[str, dict[str, float]], ...] = (
    ("q0", {"shape_mean_weight": 2.0, "shape_max_weight": 0.50, "ip_weight": 1.0}),
    ("q1", {"shape_mean_weight": 3.2, "shape_max_weight": 0.80, "ip_weight": 1.8}),
    ("q2", {"shape_mean_weight": 5.0, "shape_max_weight": 1.25, "ip_weight": 3.0}),
)

ACTUATOR_REGIMES: tuple[tuple[str, dict[str, float]], ...] = (
    ("a0", {"current_weight": 0.50, "derivative_weight": 0.125, "actuator_saturation_weight": 0.125}),
    ("a1", {"current_weight": 0.75, "derivative_weight": 0.1875, "actuator_saturation_weight": 0.1875}),
    ("a2", {"current_weight": 1.50, "derivative_weight": 0.375, "actuator_saturation_weight": 0.375}),
    ("a3", {"current_weight": 3.00, "derivative_weight": 0.750, "actuator_saturation_weight": 0.750}),
)

FIXED_REWARD: dict[str, float | str] = {
    "kind": "tcv_derivative",
    "reward_scale": 0.01,
    "smoothmax_alpha": -5.0,
    "shape_mean_scale_m": 0.03,
    "shape_max_scale_m": 0.08,
    "ip_scale_a": 25000.0,
    "boundary_missing_error_m": 1.0,
    "boundary_missing_weight": 20.0,
    "current_soft_fraction": 0.90,
    "current_bad_fraction": 1.00,
    "derivative_soft_fraction": 0.90,
    "derivative_bad_fraction": 1.10,
    "action_weight": 0.0,
    "delta_action_weight": 0.0,
    "terminal_remaining_cost": 0.0,
    "current_usage_weight": 0.0,
    "derivative_usage_weight": 0.0,
}


def build_manifest() -> dict[str, object]:
    variants: list[dict[str, object]] = []
    index = 0
    for t_name, t_values in SURVIVAL_REGIMES:
        for q_name, q_values in TRACKING_REGIMES:
            for a_name, a_values in ACTUATOR_REGIMES:
                reward = dict(FIXED_REWARD)
                reward.update(t_values)
                reward.update(q_values)
                reward.update(a_values)
                name = f"s{index:03d}_{t_name}_{q_name}_{a_name}"
                variants.append(
                    {
                        "index": index,
                        "name": name,
                        "task_id": index // CANDIDATES_PER_TASK,
                        "local_index": index % CANDIDATES_PER_TASK,
                        "regimes": {"survival": t_name, "tracking": q_name, "actuator": a_name},
                        "reward": reward,
                    }
                )
                index += 1
    if len(variants) != VARIANT_COUNT:
        raise AssertionError(f"expected {VARIANT_COUNT} variants, got {len(variants)}")
    names = [str(v["name"]) for v in variants]
    if len(set(names)) != len(names):
        raise AssertionError("variant names are not unique")
    return {
        "profile": PROFILE,
        "variant_count": VARIANT_COUNT,
        "array_tasks": ARRAY_TASKS,
        "candidates_per_task": CANDIDATES_PER_TASK,
        "variants": variants,
    }


def write_manifest(path: Path) -> dict[str, object]:
    manifest = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    variants = manifest.get("variants")
    if not isinstance(variants, list) or len(variants) != VARIANT_COUNT:
        raise ValueError(f"{path} must contain {VARIANT_COUNT} variants")
    return manifest


def variant_by_index(manifest: Mapping[str, object], index: int) -> Mapping[str, object]:
    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise ValueError("manifest variants must be a list")
    for variant in variants:
        if isinstance(variant, Mapping) and int(variant.get("index", -1)) == int(index):
            return variant
    raise KeyError(f"variant index not found: {index}")


def generate_candidate_config(
    *,
    base_config: Path,
    manifest: Mapping[str, object],
    variant_index: int,
    output_dir: str,
    train_steps: int,
    eval_steps: int,
    num_envs: int,
    replay_capacity_episodes: int,
    rl_root: str | None = None,
    sim_root: str | None = None,
) -> dict[str, object]:
    variant = variant_by_index(manifest, variant_index)
    reward = variant.get("reward")
    if not isinstance(reward, Mapping):
        raise ValueError(f"variant {variant_index} has no reward mapping")
    cfg = json.loads(base_config.read_text(encoding="utf-8"))
    name = str(variant["name"])
    cfg["name"] = f"t15_csv_segmented_profile_tcvdelta_t15boundary_sweep_{name}"
    cfg["reward"].update(dict(reward))
    cfg["training"]["steps"] = int(train_steps)
    cfg["training"]["num_envs"] = int(num_envs)
    cfg["training"]["output_dir"] = str(output_dir)
    cfg["training"]["save_checkpoints"] = False
    cfg["training"]["checkpoint_interval_steps"] = int(train_steps)
    cfg["training"]["milestone_checkpoint_interval_steps"] = 0
    cfg["training"]["eval_checkpoint_top_k"] = 0
    cfg["training"]["eval_interval_steps"] = int(eval_steps)
    cfg["training"]["eval_max_steps"] = int(cfg["sim"]["max_episode_steps"])
    cfg["training"]["distributed_mode"] = "single"
    cfg["training"]["actor_workers"] = 1
    cfg["learner"]["replay_capacity_episodes"] = int(replay_capacity_episodes)

    if rl_root is not None:
        cfg["sim"]["csv_initial_state_library"] = f"{rl_root}/data/processed/t15_csv_initial_states.npz"
        cfg["reference"]["ip"]["limits_path"] = f"{rl_root}/data/processed/t15_reference_limits.json"
    if sim_root is not None:
        cfg["sim"]["config_path"] = f"{sim_root}/configs/T15MD_new_data.toml"
        cfg["reference"]["boundary"]["replay_reference_dir"] = f"{sim_root}/runs/t15md_limited_replay_dataset"

    validate_candidate_config_dict(cfg)
    return cfg


def validate_candidate_config_dict(cfg: Mapping[str, Any]) -> None:
    checks = {
        "reference.boundary.kind": cfg["reference"]["boundary"]["kind"] == "t15_replay_segment_conditioned",
        "reference.ip.kind": cfg["reference"]["ip"]["kind"] == "segmented_profile",
        "sim.max_episode_steps": int(cfg["sim"]["max_episode_steps"]) == 2000,
        "reward.kind": cfg["reward"]["kind"] == "tcv_derivative",
        "sim.action_contract": cfg["sim"]["action_contract"] == "delta_jdot",
        "sim.delta_derivative_limits_aps": bool(cfg["sim"].get("delta_derivative_limits_aps")),
        "sim.terminate_on_boundary_loss": cfg["sim"]["terminate_on_boundary_loss"] is True,
        "sim.terminate_on_current_limit": cfg["sim"]["terminate_on_current_limit"] is True,
        "sim.current_hard_termination_fraction": float(cfg["sim"]["current_hard_termination_fraction"]) == 1.20,
        "sim.current_termination_grace_steps": int(cfg["sim"]["current_termination_grace_steps"]) == 1,
        "sim.current_saturation_fraction": float(cfg["sim"]["current_saturation_fraction"]) == 1.0,
        "reward.reward_scale": float(cfg["reward"]["reward_scale"]) == 0.01,
        "reward.smoothmax_alpha": float(cfg["reward"]["smoothmax_alpha"]) == -5.0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("candidate config failed invariant checks: " + ", ".join(failed))


def write_candidate_config(config: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def summarize_root(root: Path, *, out_dir: Path | None = None) -> dict[str, object]:
    manifest = load_manifest(root / "variants.json")
    out_dir = out_dir or (root / "selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for variant in manifest["variants"]:  # type: ignore[index]
        assert isinstance(variant, Mapping)
        name = str(variant["name"])
        run_dir = root / name
        row: dict[str, object] = {
            "index": int(variant["index"]),
            "name": name,
            "task_id": int(variant["task_id"]),
            "local_index": int(variant["local_index"]),
            "survival": str(variant["regimes"]["survival"]),  # type: ignore[index]
            "tracking": str(variant["regimes"]["tracking"]),  # type: ignore[index]
            "actuator": str(variant["regimes"]["actuator"]),  # type: ignore[index]
            "run_dir": str(run_dir),
        }
        row.update({f"reward_{k}": v for k, v in dict(variant["reward"]).items()})  # type: ignore[arg-type]
        validation = _read_json(run_dir / "policy_validation.json")
        if validation:
            row["status"] = validation.get("status", "")
            row["training_status"] = validation.get("training_status", "")
        else:
            row["status"] = "missing_validation"
        final_metrics = _read_json(run_dir / "metrics.json")
        if final_metrics:
            for key, value in _flatten_numeric(final_metrics).items():
                row[f"metrics_{key}"] = value
        eval_row = _last_csv_row(run_dir / "eval_history.csv")
        if eval_row:
            for key, value in eval_row.items():
                row[f"eval_{key}"] = value
            row["has_actor_eval"] = 1
        else:
            row["has_actor_eval"] = 0
        row["physical_priority_score"] = _physical_priority_score(row)
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda row: float(row["physical_priority_score"]), reverse=True)
    _write_summary_csv(out_dir / "reward_search_summary.csv", rows_sorted)
    best = next((row for row in rows_sorted if int(row.get("has_actor_eval", 0)) == 1), rows_sorted[0] if rows_sorted else {})
    (out_dir / "best_available_candidate.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    _write_report(out_dir / "reward_search_report.md", rows_sorted)
    return {"root": str(root), "out_dir": str(out_dir), "best": best, "rows": len(rows)}


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _last_csv_row(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    if not rows:
        return {}
    out: dict[str, object] = {}
    for key, value in rows[-1].items():
        out[key] = _maybe_float(value)
    return out


def _maybe_float(value: object) -> object:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return value
    return out if math.isfinite(out) else value


def _flatten_numeric(data: Mapping[str, object], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_numeric(value, name))
        else:
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                out[name] = number
    return out


def _metric(row: Mapping[str, object], *names: str, default: float) -> float:
    for name in names:
        value = row.get(name)
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return default


def _physical_priority_score(row: Mapping[str, object]) -> float:
    completion = _metric(row, "eval_mean_episode_completion", "metrics_final.mean_episode_completion", default=0.0)
    full_success = _metric(row, "eval_full_episode_success", "metrics_final.full_episode_success", default=completion)
    min_completion = _metric(row, "eval_min_episode_completion", default=completion)
    terminated_boundary = _metric(row, "eval_terminated_boundary", "eval_terminated_boundary_max", default=0.0)
    terminated_current = _metric(row, "eval_terminated_current", "eval_terminated_current_max", default=0.0)
    boundary_late = _metric(row, "eval_padded_boundary_found_late_min", "eval_boundary_found_late_min", default=0.0)
    current_5ka = _metric(row, "eval_padded_current_over_limit_5ka_fraction_late", "eval_current_over_limit_5ka_fraction_late", default=1.0)
    current_1pct = _metric(row, "eval_padded_current_over_limit_1pct_fraction_late", "eval_current_over_limit_1pct_fraction_late", default=1.0)
    current_over = _metric(row, "eval_padded_current_over_limit_a_late_max", "eval_current_over_limit_a_late_max", "eval_current_over_limit_a_max", default=1.0e6)
    shape = _metric(row, "eval_padded_shape_error_mean_m_late", "eval_shape_error_mean_m_late", default=1.0)
    ip = _metric(row, "eval_padded_ip_error_a_late", "eval_ip_error_a_late", default=1.0e6)
    saturation = _metric(row, "eval_action_saturation_fraction_late", default=1.0)
    delta_rms = _metric(row, "eval_delta_action_rms_late", "eval_delta_action_rms", default=1.0)
    objective = (
        250.0 * max(1.0 - completion, 0.0)
        + 250.0 * max(1.0 - full_success, 0.0)
        + 200.0 * max(0.95 - min_completion, 0.0)
        + 800.0 * max(0.999 - boundary_late, 0.0)
        + 80.0 * max(terminated_boundary, 0.0)
        + 80.0 * max(terminated_current, 0.0)
        + 3.0 * current_5ka
        + 2.0 * current_1pct
        + current_over / 20000.0
        + 2.0 * shape / 0.03
        + 2.0 * ip / 25000.0
        + 0.5 * saturation
        + 0.25 * delta_rms / 0.1
    )
    return float(-objective)


def _write_summary_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_report(path: Path, rows: list[Mapping[str, object]]) -> None:
    lines = ["# TCV-Delta T15-Boundary Reward Search", ""]
    completed = sum(1 for row in rows if int(row.get("has_actor_eval", 0)) == 1)
    lines.append(f"- total candidates: {len(rows)}")
    lines.append(f"- candidates with actor eval: {completed}")
    if rows:
        best = rows[0]
        lines.extend(
            [
                "",
                "## Best Available",
                "",
                f"- name: `{best.get('name', '')}`",
                f"- score: `{best.get('physical_priority_score', '')}`",
                f"- regimes: `{best.get('survival', '')}/{best.get('tracking', '')}/{best.get('actuator', '')}`",
                f"- completion: `{best.get('eval_mean_episode_completion', '')}`",
                f"- boundary late min: `{best.get('eval_boundary_found_late_min', best.get('eval_padded_boundary_found_late_min', ''))}`",
                f"- Ip late error A: `{best.get('eval_ip_error_a_late', best.get('eval_padded_ip_error_a_late', ''))}`",
                f"- current >5kA late fraction: `{best.get('eval_current_over_limit_5ka_fraction_late', best.get('eval_padded_current_over_limit_5ka_fraction_late', ''))}`",
            ]
        )
    lines.extend(["", "## Top 10", ""])
    for row in rows[:10]:
        lines.append(
            f"- `{row.get('name', '')}` score `{row.get('physical_priority_score', '')}` "
            f"completion `{row.get('eval_mean_episode_completion', '')}` "
            f"boundary `{row.get('eval_boundary_found_late_min', row.get('eval_padded_boundary_found_late_min', ''))}` "
            f"ip `{row.get('eval_ip_error_a_late', row.get('eval_padded_ip_error_a_late', ''))}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TCV-delta T15-boundary sweep utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_manifest = sub.add_parser("manifest")
    p_manifest.add_argument("--out", required=True)
    p_name = sub.add_parser("variant-name")
    p_name.add_argument("--manifest", required=True)
    p_name.add_argument("--index", type=int, required=True)
    p_cfg = sub.add_parser("generate-config")
    p_cfg.add_argument("--base-config", required=True)
    p_cfg.add_argument("--manifest", required=True)
    p_cfg.add_argument("--index", type=int, required=True)
    p_cfg.add_argument("--output-dir", required=True)
    p_cfg.add_argument("--config-out", required=True)
    p_cfg.add_argument("--train-steps", type=int, required=True)
    p_cfg.add_argument("--eval-steps", type=int, required=True)
    p_cfg.add_argument("--num-envs", type=int, required=True)
    p_cfg.add_argument("--replay-capacity-episodes", type=int, default=288)
    p_cfg.add_argument("--rl-root", default=None)
    p_cfg.add_argument("--sim-root", default=None)
    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("root")
    p_sum.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    if args.cmd == "manifest":
        write_manifest(Path(args.out))
        return 0
    if args.cmd == "variant-name":
        manifest = load_manifest(Path(args.manifest))
        print(str(variant_by_index(manifest, args.index)["name"]))
        return 0
    if args.cmd == "generate-config":
        manifest = load_manifest(Path(args.manifest))
        cfg = generate_candidate_config(
            base_config=Path(args.base_config),
            manifest=manifest,
            variant_index=int(args.index),
            output_dir=str(args.output_dir),
            train_steps=int(args.train_steps),
            eval_steps=int(args.eval_steps),
            num_envs=int(args.num_envs),
            replay_capacity_episodes=int(args.replay_capacity_episodes),
            rl_root=args.rl_root,
            sim_root=args.sim_root,
        )
        write_candidate_config(cfg, Path(args.config_out))
        return 0
    if args.cmd == "summarize":
        summarize_root(Path(args.root), out_dir=None if args.out_dir is None else Path(args.out_dir))
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())

