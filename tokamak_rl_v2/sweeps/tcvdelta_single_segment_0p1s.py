from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from tokamak_rl_v2.sweeps.tcvdelta_t15boundary import (
    ACTUATOR_REGIMES,
    ARRAY_TASKS,
    CANDIDATES_PER_TASK,
    FIXED_REWARD,
    SURVIVAL_REGIMES,
    TRACKING_REGIMES,
    VARIANT_COUNT,
    _flatten_numeric,
    _last_csv_row,
    _physical_priority_score,
    _read_json,
    _write_summary_csv,
)


PROFILE = "tcvjdot_single_segment_0p1s_36x1m"


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
    cfg["name"] = f"t15_csv_single_segment_0p1s_static_boundary_tcvjdot_sweep_{name}"
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
    cfg["training"]["distributed_mode"] = "local_replay"
    cfg["training"]["actor_workers"] = 1
    cfg["learner"]["replay_capacity_episodes"] = int(replay_capacity_episodes)

    if rl_root is not None:
        cfg["sim"]["csv_initial_state_library"] = f"{rl_root}/data/processed/t15_csv_initial_states.npz"
        cfg["reference"]["ip"]["limits_path"] = f"{rl_root}/data/processed/t15_reference_limits.json"
    if sim_root is not None:
        cfg["sim"]["config_path"] = f"{sim_root}/configs/T15MD_new_data.toml"

    validate_candidate_config_dict(cfg)
    return cfg


def validate_candidate_config_dict(cfg: Mapping[str, Any]) -> None:
    preview_steps = int(cfg["observation"]["target_preview_steps"])
    preview_stride = int(cfg["observation"]["target_preview_stride"])
    episode_steps = int(cfg["sim"]["max_episode_steps"])
    checks = {
        "reference.boundary.kind": cfg["reference"]["boundary"]["kind"] == "hold_reset_boundary",
        "reference.ip.kind": cfg["reference"]["ip"]["kind"] == "single_segment_profile",
        "reference.duration_s": math.isclose(float(cfg["reference"]["duration_s"]), 0.1, rel_tol=0.0, abs_tol=1.0e-12),
        "reference.t_step": math.isclose(float(cfg["reference"]["t_step"]), 0.001, rel_tol=0.0, abs_tol=1.0e-12),
        "sim.max_episode_steps": episode_steps == 100,
        "training.eval_max_steps": int(cfg["training"]["eval_max_steps"]) == 100,
        "observation.preview": preview_steps == 10 and preview_stride == 10 and preview_steps * preview_stride <= episode_steps,
        "reward.kind": cfg["reward"]["kind"] == "tcv_derivative",
        "observation.actor_kind": cfg["observation"]["actor_kind"] == "controller_state_v4",
        "sim.action_contract": cfg["sim"]["action_contract"] == "jdot_command",
        "sim.no_delta_derivative_limits": "delta_derivative_limits_aps" not in cfg["sim"],
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
        row["status"] = validation.get("status", "missing_validation") if validation else "missing_validation"
        if validation:
            row["training_status"] = validation.get("training_status", "")
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
    best = next((row for row in rows_sorted if int(row.get("has_actor_eval", 0)) == 1), {})
    if not best and rows_sorted:
        best = {
            "status": "no_actor_eval",
            "message": "No candidate produced actor evaluation data; inspect candidate logs before ranking.",
            "total_candidates": len(rows_sorted),
        }
    (out_dir / "best_available_candidate.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    _write_report(out_dir / "reward_search_report.md", rows_sorted)
    return {"root": str(root), "out_dir": str(out_dir), "best": best, "rows": len(rows)}


def _write_report(path: Path, rows: list[Mapping[str, object]]) -> None:
    lines = ["# TCV-Jdot 0.1 s Single-Segment Reward Search", ""]
    completed = sum(1 for row in rows if int(row.get("has_actor_eval", 0)) == 1)
    lines.append(f"- total candidates: {len(rows)}")
    lines.append(f"- candidates with actor eval: {completed}")
    eval_rows = [row for row in rows if int(row.get("has_actor_eval", 0)) == 1]
    if eval_rows:
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
    elif rows:
        lines.extend(
            [
                "",
                "## Best Available",
                "",
                "No candidate produced actor evaluation data. Inspect Slurm logs and candidate validation files.",
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
    parser = argparse.ArgumentParser(description="TCV-Jdot 0.1 s single-segment sweep utilities")
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
