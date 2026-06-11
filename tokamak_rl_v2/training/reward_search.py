from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.config.schema import ExperimentConfig, RewardConfig
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.training.trainer import Trainer


REWARD_FIELDS = [
    "shape_good_m",
    "shape_bad_m",
    "ip_good_a",
    "ip_bad_a",
    "shape_weight",
    "ip_weight",
    "current_good_a",
    "current_bad_a",
    "derivative_good",
    "derivative_bad",
    "action_penalty_weight",
    "delta_action_penalty_weight",
    "terminal_reward",
    "reward_scale",
]
STAGE_NAMES = ["stage_01_short", "stage_02_medium", "stage_03_final"]
WORKER_ARG_NAMES = [
    "config",
    "output_dir",
    "num_envs",
    "device",
    "sim_compute_backend",
    "sim_gpu_device",
    "batch_size",
    "unroll_length",
    "replay_capacity_episodes",
    "hidden_dim",
    "critic_hidden_dim",
    "critic_mlp_hidden_dim",
    "rollout_chunk_length",
    "updates_per_rollout_chunk",
    "action_samples",
    "actor_update_chunk_size",
    "checkpoint_interval_steps",
    "eval_interval_steps",
    "eval_episodes",
    "eval_max_steps",
    "actor_workers",
    "wandb",
    "wandb_project",
    "wandb_name",
    "wandb_group",
    "wandb_mode",
    "require_no_control_improvement",
    "stage_max_shape_degradation_m",
    "stage_min_ip_improvement_a",
]


@dataclass(frozen=True, slots=True)
class RewardCandidate:
    index: int
    reward: RewardConfig
    source: str = "grid"
    parent: int | None = None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_task is not None:
        return _run_worker_task(Path(args.worker_task))
    if args.config is None:
        raise ValueError("--config is required")
    base = _apply_overrides(load_experiment_config(args.config), args)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    value_grid = _reward_value_grid(base.reward, args)
    rewards = list(_candidate_rewards_from_values(base.reward, value_grid))
    if args.strategy == "grid" and args.search_seed is not None:
        rng = np.random.default_rng(int(args.search_seed))
        order = rng.permutation(len(rewards))
        rewards = [rewards[int(i)] for i in order]
    if args.max_candidates is not None:
        rewards = rewards[: int(args.max_candidates)]
    candidates = [RewardCandidate(index=i, reward=reward) for i, reward in enumerate(rewards)]
    manifest = {
        "strategy": args.strategy,
        "stage_only": args.stage_only,
        "config": str(Path(args.config).resolve()),
        "candidate_count": len(candidates),
        "base_reward": asdict(base.reward),
        "reward_fields": REWARD_FIELDS,
        "reward_values": {name: values for name, values in value_grid},
        "baseline_policy": args.baseline_policy if args.strategy == "successive_halving" else None,
        "stage_steps": _int_values(args.stage_steps) if args.strategy == "successive_halving" else None,
        "stage_keep": _int_values(args.stage_keep) if args.strategy == "successive_halving" else None,
        "stage_eval_episodes": _int_values(args.stage_eval_episodes) if args.strategy == "successive_halving" else None,
        "refine_top_k": int(args.refine_top_k),
        "refine_midpoints": bool(args.refine_midpoints),
        "require_no_control_improvement": bool(args.require_no_control_improvement),
        "parallel_candidates": int(args.parallel_candidates),
        "gpu_devices": _gpu_devices(args),
        "baseline_results": args.baseline_results,
        "previous_stage_results": args.previous_stage_results,
        "stage_input_count": args.stage_input_count,
        "stage_max_shape_degradation_m": _optional_float_values(args.stage_max_shape_degradation_m),
        "stage_min_ip_improvement_a": _optional_float_values(args.stage_min_ip_improvement_a),
    }
    (out / "search_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.stage_only != "all":
        if args.strategy != "successive_halving":
            raise ValueError("--stage-only requires --strategy successive_halving")
        return _run_one_requested_stage(base=base, args=args, out=out, candidates=candidates)
    if args.strategy == "successive_halving":
        return _run_successive_halving(base=base, args=args, out=out, candidates=candidates, value_grid=value_grid)
    return _run_grid_search(base=base, args=args, out=out, candidates=candidates)


def _run_grid_search(*, base: ExperimentConfig, args: argparse.Namespace, out: Path, candidates: Sequence[RewardCandidate]) -> int:
    rows: list[dict[str, object]] = []
    for position, candidate in enumerate(candidates):
        cand_dir = out / f"candidate_{candidate.index:04d}"
        row = _candidate_row(candidate, stage="grid", candidate_dir=cand_dir)
        print(f"[{position + 1}/{len(candidates)}] reward={_reward_label(candidate.reward)}", flush=True)
        if args.dry_run:
            row.update({"status": "dry_run", "score": ""})
            rows.append(row)
            _write_results(out / "results.csv", rows)
            continue
        try:
            trained = _train_candidate(base=base, args=args, candidate=candidate, candidate_dir=cand_dir, steps=_steps_or_default(args, base), eval_episodes=int(base.training.eval_episodes), stage_name="grid", stage_index=0)
            row.update(trained)
        except Exception as exc:
            row.update({"status": "error", "score": "", "error": repr(exc)})
            cand_dir.mkdir(parents=True, exist_ok=True)
            (cand_dir / "error.txt").write_text(repr(exc), encoding="utf-8")
            if not args.continue_on_error:
                rows.append(row)
                _write_results(out / "results.csv", rows)
                raise
        rows.append(row)
        _write_results(out / "results.csv", rows)
    _write_results(out / "results.csv", rows)
    return 0


def _run_successive_halving(*, base: ExperimentConfig, args: argparse.Namespace, out: Path, candidates: Sequence[RewardCandidate], value_grid: Sequence[tuple[str, list[float]]]) -> int:
    stage_steps = _int_values(args.stage_steps)
    stage_keep = _int_values(args.stage_keep)
    stage_eval_episodes = _int_values(args.stage_eval_episodes)
    if len(stage_steps) != 3 or len(stage_keep) != 3 or len(stage_eval_episodes) != 3:
        raise ValueError("successive_halving requires exactly three values for --stage-steps, --stage-keep, and --stage-eval-episodes")
    if args.baseline_policy != "no_control":
        raise ValueError("only --baseline-policy no_control is supported; no analytic controller baseline is allowed")
    baseline_dir = out / "stage_00_baseline"
    baseline_rows: list[dict[str, object]] = []
    baseline_by_key: dict[tuple[float, ...], dict[str, object]] = {}
    all_rows: list[dict[str, object]] = []
    promotion_rows: list[dict[str, object]] = []
    if args.dry_run:
        dry_rows = [_candidate_row(candidate, stage="stage_00_baseline", candidate_dir=baseline_dir / f"candidate_{candidate.index:04d}") | {"status": "dry_run"} for candidate in candidates]
        _write_results(baseline_dir / "results.csv", dry_rows)
        _write_results(out / "baseline_results.csv", dry_rows)
        _write_results(out / "results.csv", dry_rows)
        return 0
    for position, candidate in enumerate(candidates):
        print(f"[baseline {position + 1}/{len(candidates)}] reward={_reward_label(candidate.reward)}", flush=True)
        row = _evaluate_baseline_candidate(base=base, args=args, candidate=candidate, candidate_dir=baseline_dir / f"candidate_{candidate.index:04d}", eval_episodes=stage_eval_episodes[0])
        baseline_rows.append(row)
        baseline_by_key[_reward_key(candidate.reward)] = row
        _write_results(baseline_dir / "results.csv", baseline_rows)
        _write_results(out / "baseline_results.csv", baseline_rows)
    active = list(candidates)
    next_candidate_index = (max((c.index for c in candidates), default=-1) + 1)
    final_ranked: list[dict[str, object]] = []
    for stage_idx, (stage_name, steps, keep, eval_episodes) in enumerate(zip(STAGE_NAMES, stage_steps, stage_keep, stage_eval_episodes), start=1):
        stage_dir = out / stage_name
        stage_rows = _run_stage_candidates(
            base=base,
            args=args,
            candidates=active,
            stage_dir=stage_dir,
            stage_name=stage_name,
            stage_index=stage_idx,
            steps=int(steps),
            eval_episodes=int(eval_episodes),
            baseline_by_key=baseline_by_key,
            aggregate_results_path=out / "results.csv",
            aggregate_rows=all_rows,
        )
        all_rows.extend(stage_rows)
        _write_results(out / "results.csv", all_rows)
        ranked = _rank_rows(stage_rows)
        final_ranked = ranked
        eligible = [row for row in ranked if row.get("status") == "ok" and row.get("promotion_reason") == "eligible"]
        if not eligible:
            _write_results(out / "promotion_history.csv", promotion_rows)
            raise RuntimeError(f"no eligible reward candidates after {stage_name}; inspect {stage_dir / 'results.csv'}")
        promoted_rows = eligible[: min(int(keep), len(eligible))]
        promoted_ids = {int(row["candidate"]) for row in promoted_rows}
        for rank, row in enumerate(ranked, start=1):
            promo = {"stage": stage_name, "candidate": row.get("candidate"), "rank": rank, "promoted": int(int(row.get("candidate", -1)) in promoted_ids), "promotion_reason": row.get("promotion_reason", ""), "eval.mean_return": row.get("eval.mean_return", ""), "eval.shape_error_mean_m": row.get("eval.shape_error_mean_m", ""), "eval.ip_error_a": row.get("eval.ip_error_a", ""), "eval.current_over_limit_a": row.get("eval.current_over_limit_a", ""), "eval.boundary_found": row.get("eval.boundary_found", ""), "eval.shape_improvement_over_no_control_m": row.get("eval.shape_improvement_over_no_control_m", ""), "eval.ip_improvement_over_no_control_a": row.get("eval.ip_improvement_over_no_control_a", ""), "eval.improvement_over_no_control": row.get("eval.improvement_over_no_control", "")}
            promotion_rows.append(promo)
        _write_results(out / "promotion_history.csv", promotion_rows)
        promoted_candidates = [_candidate_by_index(active, int(row["candidate"])) for row in promoted_rows]
        if stage_idx == 1 and args.refine_midpoints:
            refined = _refine_candidates(promoted_candidates[: int(args.refine_top_k)], value_grid=value_grid, existing={_reward_key(c.reward) for c in list(candidates) + promoted_candidates}, next_index=next_candidate_index)
            next_candidate_index += len(refined)
            for candidate in refined:
                print(f"[baseline refined {candidate.index}] reward={_reward_label(candidate.reward)}", flush=True)
                row = _evaluate_baseline_candidate(base=base, args=args, candidate=candidate, candidate_dir=baseline_dir / f"candidate_{candidate.index:04d}", eval_episodes=stage_eval_episodes[0])
                baseline_rows.append(row)
                baseline_by_key[_reward_key(candidate.reward)] = row
            _write_results(baseline_dir / "results.csv", baseline_rows)
            _write_results(out / "baseline_results.csv", baseline_rows)
            active = promoted_candidates + refined
        else:
            active = promoted_candidates
    finalists = final_ranked[: min(stage_keep[-1], len(final_ranked))]
    _write_results(out / "finalists.csv", finalists)
    if finalists:
        best = finalists[0]
        best_reward = _reward_from_row(best)
        (out / "best_reward.json").write_text(json.dumps(asdict(best_reward), indent=2), encoding="utf-8")
        (out / "best_reward.yaml").write_text("".join(f"{k}: {v}\n" for k, v in asdict(best_reward).items()), encoding="utf-8")
    _write_results(out / "results.csv", all_rows)
    return 0


def _run_one_requested_stage(*, base: ExperimentConfig, args: argparse.Namespace, out: Path, candidates: Sequence[RewardCandidate]) -> int:
    stage_steps = _int_values(args.stage_steps)
    stage_keep = _int_values(args.stage_keep)
    stage_eval_episodes = _int_values(args.stage_eval_episodes)
    if len(stage_steps) != 3 or len(stage_keep) != 3 or len(stage_eval_episodes) != 3:
        raise ValueError("staged reward search requires exactly three values for --stage-steps, --stage-keep, and --stage-eval-episodes")
    if args.baseline_policy != "no_control":
        raise ValueError("only --baseline-policy no_control is supported; no analytic controller baseline is allowed")
    if args.stage_only == "stage_00_baseline":
        baseline_rows = _evaluate_baseline_candidates(base=base, args=args, out=out, candidates=candidates, eval_episodes=stage_eval_episodes[0])
        _write_results(out / "results.csv", baseline_rows)
        return 0

    baseline_rows = _load_or_build_baseline(base=base, args=args, out=out, candidates=candidates, eval_episodes=stage_eval_episodes[0])
    baseline_by_key = {_reward_key(_candidate_from_row(row).reward): row for row in baseline_rows}
    stage_name = str(args.stage_only)
    stage_idx = STAGE_NAMES.index(stage_name) + 1
    active = _stage_input_candidates(args=args, all_candidates=candidates, stage_idx=stage_idx, previous_keep=stage_keep[stage_idx - 2] if stage_idx > 1 else None)
    stage_dir = out / stage_name
    stage_rows = _run_stage_candidates(
        base=base,
        args=args,
        candidates=active,
        stage_dir=stage_dir,
        stage_name=stage_name,
        stage_index=stage_idx,
        steps=int(stage_steps[stage_idx - 1]),
        eval_episodes=int(stage_eval_episodes[stage_idx - 1]),
        baseline_by_key=baseline_by_key,
        aggregate_results_path=out / "results.csv",
        aggregate_rows=[],
    )
    _write_results(stage_dir / "results.csv", stage_rows)
    _write_results(out / "results.csv", stage_rows)
    ranked = _rank_rows(stage_rows)
    eligible_count = len([row for row in ranked if row.get("status") == "ok" and row.get("promotion_reason") == "eligible"])
    keep = min(int(stage_keep[stage_idx - 1]), eligible_count)
    _write_promotion_artifacts(out=out, stage_name=stage_name, ranked=ranked, keep=keep)
    if keep == 0:
        raise RuntimeError(f"no eligible reward candidates after {stage_name}; inspect {stage_dir / 'results.csv'}")
    if stage_idx == len(STAGE_NAMES):
        finalists = ranked[: min(int(stage_keep[-1]), len(ranked))]
        _write_results(out / "finalists.csv", finalists)
        if finalists:
            best_reward = _reward_from_row(finalists[0])
            (out / "best_reward.json").write_text(json.dumps(asdict(best_reward), indent=2), encoding="utf-8")
            (out / "best_reward.yaml").write_text("".join(f"{k}: {v}\n" for k, v in asdict(best_reward).items()), encoding="utf-8")
    return 0


def _evaluate_baseline_candidates(*, base: ExperimentConfig, args: argparse.Namespace, out: Path, candidates: Sequence[RewardCandidate], eval_episodes: int) -> list[dict[str, object]]:
    baseline_dir = out / "stage_00_baseline"
    baseline_rows: list[dict[str, object]] = []
    for position, candidate in enumerate(candidates):
        print(f"[baseline {position + 1}/{len(candidates)}] reward={_reward_label(candidate.reward)}", flush=True)
        row = _evaluate_baseline_candidate(base=base, args=args, candidate=candidate, candidate_dir=baseline_dir / f"candidate_{candidate.index:04d}", eval_episodes=eval_episodes)
        baseline_rows.append(row)
        _write_results(baseline_dir / "results.csv", baseline_rows)
        _write_results(out / "baseline_results.csv", baseline_rows)
    return baseline_rows


def _load_or_build_baseline(*, base: ExperimentConfig, args: argparse.Namespace, out: Path, candidates: Sequence[RewardCandidate], eval_episodes: int) -> list[dict[str, object]]:
    baseline_path = Path(args.baseline_results).resolve() if args.baseline_results else out / "baseline_results.csv"
    if baseline_path.exists():
        rows = _read_results(baseline_path)
        _write_results(out / "baseline_results.csv", rows)
        return rows
    return _evaluate_baseline_candidates(base=base, args=args, out=out, candidates=candidates, eval_episodes=eval_episodes)


def _stage_input_candidates(*, args: argparse.Namespace, all_candidates: Sequence[RewardCandidate], stage_idx: int, previous_keep: int | None) -> list[RewardCandidate]:
    if stage_idx == 1:
        return list(all_candidates)
    if not args.previous_stage_results:
        raise ValueError(f"--previous-stage-results is required for {STAGE_NAMES[stage_idx - 1]}")
    previous_rows = _read_results(Path(args.previous_stage_results).resolve())
    ranked = _rank_rows(previous_rows)
    requested = int(args.stage_input_count or previous_keep or len(ranked))
    selected = [row for row in ranked if row.get("status") == "ok" and row.get("promotion_reason") == "eligible"][:requested]
    if not selected:
        raise RuntimeError(f"no eligible previous-stage candidates in {args.previous_stage_results}")
    return [_candidate_from_row(row) for row in selected]


def _write_promotion_artifacts(*, out: Path, stage_name: str, ranked: Sequence[dict[str, object]], keep: int) -> None:
    promoted_ids = {int(row["candidate"]) for row in ranked[: int(keep)] if row.get("status") == "ok" and row.get("promotion_reason") == "eligible"}
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(ranked, start=1):
        rows.append({
            "stage": stage_name,
            "candidate": row.get("candidate"),
            "rank": rank,
            "promoted": int(int(row.get("candidate", -1)) in promoted_ids),
            "promotion_reason": row.get("promotion_reason", ""),
            "eval.mean_return": row.get("eval.mean_return", ""),
            "eval.shape_error_mean_m": row.get("eval.shape_error_mean_m", ""),
            "eval.ip_error_a": row.get("eval.ip_error_a", ""),
            "eval.current_over_limit_a": row.get("eval.current_over_limit_a", ""),
            "eval.boundary_found": row.get("eval.boundary_found", ""),
            "eval.shape_improvement_over_no_control_m": row.get("eval.shape_improvement_over_no_control_m", ""),
            "eval.ip_improvement_over_no_control_a": row.get("eval.ip_improvement_over_no_control_a", ""),
            "eval.improvement_over_no_control": row.get("eval.improvement_over_no_control", ""),
        })
    _write_results(out / "promotion_history.csv", rows)
    _write_results(out / stage_name / "promoted_candidates.csv", [row for row in ranked if int(row.get("candidate", -1)) in promoted_ids])


def _run_stage_candidates(
    *,
    base: ExperimentConfig,
    args: argparse.Namespace,
    candidates: Sequence[RewardCandidate],
    stage_dir: Path,
    stage_name: str,
    stage_index: int,
    steps: int,
    eval_episodes: int,
    baseline_by_key: dict[tuple[float, ...], dict[str, object]],
    aggregate_results_path: Path,
    aggregate_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if int(args.parallel_candidates) <= 1:
        return _run_stage_candidates_sequential(
            base=base,
            args=args,
            candidates=candidates,
            stage_dir=stage_dir,
            stage_name=stage_name,
            stage_index=stage_index,
            steps=steps,
            eval_episodes=eval_episodes,
            baseline_by_key=baseline_by_key,
        )
    return _run_stage_candidates_parallel(
        args=args,
        candidates=candidates,
        stage_dir=stage_dir,
        stage_name=stage_name,
        stage_index=stage_index,
        steps=steps,
        eval_episodes=eval_episodes,
        baseline_by_key=baseline_by_key,
        aggregate_results_path=aggregate_results_path,
        aggregate_rows=aggregate_rows,
    )


def _run_stage_candidates_sequential(
    *,
    base: ExperimentConfig,
    args: argparse.Namespace,
    candidates: Sequence[RewardCandidate],
    stage_dir: Path,
    stage_name: str,
    stage_index: int,
    steps: int,
    eval_episodes: int,
    baseline_by_key: dict[tuple[float, ...], dict[str, object]],
) -> list[dict[str, object]]:
    stage_rows: list[dict[str, object]] = []
    for position, candidate in enumerate(candidates):
        print(f"[{stage_name} {position + 1}/{len(candidates)}] reward={_reward_label(candidate.reward)}", flush=True)
        row = _run_one_stage_candidate(
            base=base,
            args=args,
            candidate=candidate,
            candidate_dir=stage_dir / f"candidate_{candidate.index:04d}",
            stage_name=stage_name,
            stage_index=stage_index,
            steps=steps,
            eval_episodes=eval_episodes,
            baseline_by_key=baseline_by_key,
        )
        stage_rows.append(row)
        _write_results(stage_dir / "results.csv", stage_rows)
    return stage_rows


def _run_one_stage_candidate(
    *,
    base: ExperimentConfig,
    args: argparse.Namespace,
    candidate: RewardCandidate,
    candidate_dir: Path,
    stage_name: str,
    stage_index: int,
    steps: int,
    eval_episodes: int,
    baseline_by_key: dict[tuple[float, ...], dict[str, object]],
) -> dict[str, object]:
    row = _candidate_row(candidate, stage=stage_name, candidate_dir=candidate_dir)
    try:
        trained = _train_candidate(
            base=base,
            args=args,
            candidate=candidate,
            candidate_dir=candidate_dir,
            steps=steps,
            eval_episodes=eval_episodes,
            stage_name=stage_name,
            stage_index=stage_index,
            baseline_row=baseline_by_key.get(_reward_key(candidate.reward)),
            promotion_policy=_promotion_policy_for_stage(args, stage_index),
        )
        row.update(trained)
    except Exception as exc:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        row.update({"status": "error", "score": "", "error": repr(exc), "promotion_reason": "error"})
        (candidate_dir / "error.txt").write_text(repr(exc), encoding="utf-8")
        if not args.continue_on_error:
            raise
    return row


def _run_stage_candidates_parallel(
    *,
    args: argparse.Namespace,
    candidates: Sequence[RewardCandidate],
    stage_dir: Path,
    stage_name: str,
    stage_index: int,
    steps: int,
    eval_episodes: int,
    baseline_by_key: dict[tuple[float, ...], dict[str, object]],
    aggregate_results_path: Path,
    aggregate_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    gpu_devices = _gpu_devices(args)
    parallel = int(args.parallel_candidates)
    if parallel < 1:
        raise ValueError("--parallel-candidates must be >= 1")
    if len(gpu_devices) < parallel:
        raise ValueError("--gpu-devices must provide at least --parallel-candidates entries")
    pending = list(enumerate(candidates, start=1))
    stage_rows: list[dict[str, object]] = []
    running: list[dict[str, object]] = []
    free_gpus = list(gpu_devices[:parallel])
    stage_dir.mkdir(parents=True, exist_ok=True)
    while pending or running:
        while pending and free_gpus:
            position, candidate = pending.pop(0)
            gpu = free_gpus.pop(0)
            print(f"[{stage_name} {position}/{len(candidates)} gpu={gpu}] reward={_reward_label(candidate.reward)}", flush=True)
            running.append(_launch_candidate_worker(args=args, candidate=candidate, stage_dir=stage_dir, stage_name=stage_name, stage_index=stage_index, steps=steps, eval_episodes=eval_episodes, gpu_device=gpu, baseline_row=baseline_by_key.get(_reward_key(candidate.reward))))
        still_running: list[dict[str, object]] = []
        for item in running:
            proc: subprocess.Popen[bytes] = item["process"]  # type: ignore[assignment]
            returncode = proc.poll()
            if returncode is None:
                still_running.append(item)
                continue
            stdout_f = item.get("stdout_f")
            stderr_f = item.get("stderr_f")
            if stdout_f is not None:
                stdout_f.close()
            if stderr_f is not None:
                stderr_f.close()
            gpu = str(item["gpu_device"])
            free_gpus.append(gpu)
            row = _read_worker_result(args=args, item=item, returncode=returncode, baseline_by_key=baseline_by_key)
            stage_rows.append(row)
            _write_results(stage_dir / "results.csv", stage_rows)
            _write_results(aggregate_results_path, [*aggregate_rows, *stage_rows])
            if returncode != 0 and not args.continue_on_error:
                for other in still_running:
                    other_proc: subprocess.Popen[bytes] = other["process"]  # type: ignore[assignment]
                    other_proc.terminate()
                raise RuntimeError(f"candidate worker failed for candidate {item['candidate_index']} with exit code {returncode}; see {item['stderr_path']}")
        running = still_running
        if running and not free_gpus:
            time.sleep(2.0)
    return stage_rows


def _launch_candidate_worker(
    *,
    args: argparse.Namespace,
    candidate: RewardCandidate,
    stage_dir: Path,
    stage_name: str,
    stage_index: int,
    steps: int,
    eval_episodes: int,
    gpu_device: str,
    baseline_row: dict[str, object] | None,
) -> dict[str, object]:
    candidate_dir = stage_dir / f"candidate_{candidate.index:04d}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    task_path = candidate_dir / "worker_task.json"
    result_path = candidate_dir / "worker_result.json"
    stdout_path = candidate_dir / "worker.out"
    stderr_path = candidate_dir / "worker.err"
    worker_args = _worker_args(args, gpu_device=gpu_device)
    payload = {
        "config": args.config,
        "args": worker_args,
        "candidate": {"index": candidate.index, "reward": asdict(candidate.reward), "source": candidate.source, "parent": candidate.parent},
        "candidate_dir": str(candidate_dir),
        "stage_name": stage_name,
        "stage_index": int(stage_index),
        "steps": int(steps),
        "eval_episodes": int(eval_episodes),
        "result_path": str(result_path),
        "baseline_row": baseline_row,
        "promotion_policy": _promotion_policy_for_stage(args, stage_index),
    }
    task_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stdout_f = stdout_path.open("wb")
    stderr_f = stderr_path.open("wb")
    env = os.environ.copy()
    cmd = [sys.executable, "-m", "tokamak_rl_v2.training.reward_search", "--worker-task", str(task_path)]
    process = subprocess.Popen(cmd, cwd=Path.cwd(), env=env, stdout=stdout_f, stderr=stderr_f)
    return {"process": process, "gpu_device": gpu_device, "candidate_index": candidate.index, "candidate_reward": candidate.reward, "result_path": result_path, "stdout_path": stdout_path, "stderr_path": stderr_path, "stdout_f": stdout_f, "stderr_f": stderr_f}


def _worker_args(args: argparse.Namespace, *, gpu_device: str) -> dict[str, object]:
    out = {name: getattr(args, name, None) for name in WORKER_ARG_NAMES}
    out["device"] = f"cuda:{gpu_device}" if str(gpu_device).isdigit() else str(gpu_device)
    out["sim_gpu_device"] = f"cuda:{gpu_device}" if str(gpu_device).isdigit() else str(gpu_device)
    out["sim_compute_backend"] = args.sim_compute_backend
    out["actor_workers"] = 1
    return out


def _read_worker_result(*, args: argparse.Namespace, item: dict[str, object], returncode: int, baseline_by_key: dict[tuple[float, ...], dict[str, object]]) -> dict[str, object]:
    result_path = Path(item["result_path"])  # type: ignore[arg-type]
    reward = item["candidate_reward"]
    if result_path.exists():
        row = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        row = {"candidate": item["candidate_index"], "status": "error", "score": "", "error": f"worker exited {returncode} without result", "worker.stderr": str(item["stderr_path"])}
    if isinstance(reward, RewardConfig):
        _attach_baseline_improvement(row, baseline_row=baseline_by_key.get(_reward_key(reward)), require_no_control_improvement=bool(args.require_no_control_improvement), promotion_policy=_promotion_policy_for_stage(args, int(row.get("stage_index", 0) or 0)))
    row["worker_returncode"] = int(returncode)
    row["worker_stdout"] = str(item["stdout_path"])
    row["worker_stderr"] = str(item["stderr_path"])
    return row


def _attach_baseline_improvement(row: dict[str, object], *, baseline_row: dict[str, object] | None, require_no_control_improvement: bool, promotion_policy: Mapping[str, float | None] | None = None) -> None:
    baseline_row = baseline_row or {}
    baseline_return = _row_float(baseline_row, "baseline.mean_return", default=float("nan"))
    eval_return = _row_float(row, "eval.mean_return", default=float("nan"))
    row["eval.improvement_over_no_control"] = eval_return - baseline_return if math.isfinite(eval_return) and math.isfinite(baseline_return) else float("nan")
    baseline_shape = _row_float(baseline_row, "baseline.shape_error_mean_m", default=float("nan"))
    eval_shape = _row_float(row, "eval.shape_error_mean_m", default=float("nan"))
    row["eval.shape_improvement_over_no_control_m"] = baseline_shape - eval_shape if math.isfinite(eval_shape) and math.isfinite(baseline_shape) else float("nan")
    baseline_ip = _row_float(baseline_row, "baseline.ip_error_a", default=float("nan"))
    eval_ip = _row_float(row, "eval.ip_error_a", default=float("nan"))
    row["eval.ip_improvement_over_no_control_a"] = baseline_ip - eval_ip if math.isfinite(eval_ip) and math.isfinite(baseline_ip) else float("nan")
    row["search.require_no_control_improvement"] = int(bool(require_no_control_improvement))
    for key, value in dict(promotion_policy or {}).items():
        row[f"search.{key}"] = "" if value is None else float(value)
    row["promotion_reason"] = _promotion_reason(row)


def _run_worker_task(task_path: Path) -> int:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    args = argparse.Namespace(**task["args"])
    args.worker_task = None
    args.config = task["config"]
    base = _apply_overrides(load_experiment_config(args.config), args)
    candidate_data = task["candidate"]
    candidate = RewardCandidate(
        index=int(candidate_data["index"]),
        reward=RewardConfig(**candidate_data["reward"]),
        source=str(candidate_data.get("source", "grid")),
        parent=candidate_data.get("parent"),
    )
    candidate_dir = Path(task["candidate_dir"])
    result_path = Path(task["result_path"])
    row = _candidate_row(candidate, stage=str(task["stage_name"]), candidate_dir=candidate_dir)
    try:
        trained = _train_candidate(base=base, args=args, candidate=candidate, candidate_dir=candidate_dir, steps=int(task["steps"]), eval_episodes=int(task["eval_episodes"]), stage_name=str(task["stage_name"]), stage_index=int(task["stage_index"]), baseline_row=task.get("baseline_row"), promotion_policy=task.get("promotion_policy"))
        row.update(trained)
        result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        row.update({"status": "error", "score": "", "error": repr(exc), "promotion_reason": "error"})
        (candidate_dir / "error.txt").write_text(repr(exc), encoding="utf-8")
        result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        raise


def _train_candidate(*, base: ExperimentConfig, args: argparse.Namespace, candidate: RewardCandidate, candidate_dir: Path, steps: int, eval_episodes: int, stage_name: str, stage_index: int, baseline_row: dict[str, object] | None = None, promotion_policy: Mapping[str, float | None] | None = None) -> dict[str, object]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "reward_config.json").write_text(json.dumps(asdict(candidate.reward), indent=2), encoding="utf-8")
    training = replace(base.training, output_dir=candidate_dir, eval_interval_steps=max(1, int(steps)), eval_episodes=int(eval_episodes), eval_max_steps=int(base.training.eval_max_steps), checkpoint_interval_steps=max(1, min(int(base.training.checkpoint_interval_steps), int(steps))))
    cfg = replace(base, reward=candidate.reward, training=training)
    wandb_run = _start_wandb(args=args, cfg=cfg, candidate=candidate, stage_name=stage_name, stage_index=stage_index)
    try:
        result = Trainer(cfg, steps=int(steps), num_envs=args.num_envs, device=args.device, output_dir=candidate_dir, wandb_run=wandb_run).train()
        summary = _summarize_reward_components(candidate_dir / "reward_components.csv")
        eval_metrics = _best_eval_metrics(result)
        score = float(eval_metrics.get("mean_return", result.get("best_eval", -float("inf"))))
        row: dict[str, object] = {"status": "ok", "score": score, **{f"metric.{k}": v for k, v in result.items() if not isinstance(v, dict)}, **{f"eval.{k}": v for k, v in eval_metrics.items()}, **summary}
        row["stage_index"] = int(stage_index)
        _attach_baseline_improvement(row, baseline_row=baseline_row, require_no_control_improvement=bool(getattr(args, "require_no_control_improvement", False)), promotion_policy=promotion_policy)
        if wandb_run is not None:
            wandb_run.summary["search/promotion_reason"] = row["promotion_reason"]
            for key in ("eval.improvement_over_no_control", "eval.shape_improvement_over_no_control_m", "eval.ip_improvement_over_no_control_a"):
                if key in row and _is_number(row[key]):
                    wandb_run.summary[key.replace(".", "/")] = float(row[key])
            for key, value in eval_metrics.items():
                wandb_run.summary[f"eval/{key}"] = value
        return row
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _evaluate_baseline_candidate(*, base: ExperimentConfig, args: argparse.Namespace, candidate: RewardCandidate, candidate_dir: Path, eval_episodes: int) -> dict[str, object]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "reward_config.json").write_text(json.dumps(asdict(candidate.reward), indent=2), encoding="utf-8")
    cfg = replace(base, reward=candidate.reward, training=replace(base.training, output_dir=candidate_dir, eval_episodes=int(eval_episodes)))
    row = _candidate_row(candidate, stage="stage_00_baseline", candidate_dir=candidate_dir)
    wandb_run = _start_wandb(args=args, cfg=cfg, candidate=candidate, stage_name="stage_00_baseline", stage_index=0)
    try:
        metrics = _evaluate_no_control(config=cfg, args=args, episodes=int(eval_episodes), max_steps=int(base.training.eval_max_steps))
        row.update({"status": "ok", "score": float(metrics.get("mean_return", -float("inf"))), **{f"baseline.{k}": v for k, v in metrics.items()}})
        if wandb_run is not None:
            wandb_run.log({"search/stage": 0, "search/candidate": candidate.index, **{f"baseline/{k}": v for k, v in metrics.items()}}, step=0)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return row


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Search T15 static-boundary reward transform settings.")
    ap.add_argument("--worker-task", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--config", required=False)
    ap.add_argument("--output-dir", default="outputs/reward_search_t15_static_boundary")
    ap.add_argument("--strategy", choices=("grid", "successive_halving"), default="grid")
    ap.add_argument("--stage-only", choices=("all", "stage_00_baseline", "stage_01_short", "stage_02_medium", "stage_03_final"), default="all")
    ap.add_argument("--baseline-results", default=None)
    ap.add_argument("--previous-stage-results", default=None)
    ap.add_argument("--stage-input-count", type=int, default=None)
    ap.add_argument("--stage-steps", default="25000,100000,300000")
    ap.add_argument("--stage-keep", default="12,4,2")
    ap.add_argument("--stage-eval-episodes", default="16,32,64")
    ap.add_argument("--stage-max-shape-degradation-m", default=None)
    ap.add_argument("--stage-min-ip-improvement-a", default=None)
    ap.add_argument("--baseline-policy", choices=("no_control",), default="no_control")
    ap.add_argument("--refine-top-k", type=int, default=4)
    ap.add_argument("--refine-midpoints", action="store_true")
    ap.add_argument("--parallel-candidates", type=int, default=1)
    ap.add_argument("--gpu-devices", default=None)
    ap.add_argument("--require-no-control-improvement", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sim-compute-backend", choices=("cpu", "gpu"), default=None)
    ap.add_argument("--sim-gpu-device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--unroll-length", type=int, default=None)
    ap.add_argument("--replay-capacity-episodes", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--critic-hidden-dim", type=int, default=None)
    ap.add_argument("--critic-mlp-hidden-dim", type=int, default=None)
    ap.add_argument("--rollout-chunk-length", type=int, default=None)
    ap.add_argument("--updates-per-rollout-chunk", type=int, default=None)
    ap.add_argument("--action-samples", type=int, default=None)
    ap.add_argument("--actor-update-chunk-size", type=int, default=None)
    ap.add_argument("--checkpoint-interval-steps", type=int, default=None)
    ap.add_argument("--eval-interval-steps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--eval-max-steps", type=int, default=None)
    ap.add_argument("--actor-workers", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--search-seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--shape-good-values", default=None)
    ap.add_argument("--shape-bad-values", default=None)
    ap.add_argument("--ip-good-values", default=None)
    ap.add_argument("--ip-bad-values", default=None)
    ap.add_argument("--shape-weight-values", default=None)
    ap.add_argument("--ip-weight-values", default=None)
    ap.add_argument("--current-good-values", default=None)
    ap.add_argument("--current-bad-values", default=None)
    ap.add_argument("--terminal-reward-values", default=None)
    ap.add_argument("--reward-scale-values", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="tokamak-rl-v2")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return ap


def _apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.sim_compute_backend is not None or args.sim_gpu_device is not None:
        cfg = replace(cfg, sim=replace(cfg.sim, compute_backend=args.sim_compute_backend or cfg.sim.compute_backend, gpu_device=args.sim_gpu_device or cfg.sim.gpu_device))
    if any(v is not None for v in (args.batch_size, args.unroll_length, args.replay_capacity_episodes, args.rollout_chunk_length, args.updates_per_rollout_chunk, args.action_samples, args.actor_update_chunk_size)):
        cfg = replace(cfg, learner=replace(cfg.learner, batch_size=args.batch_size or cfg.learner.batch_size, unroll_length=args.unroll_length or cfg.learner.unroll_length, replay_capacity_episodes=args.replay_capacity_episodes or cfg.learner.replay_capacity_episodes, rollout_chunk_length=args.rollout_chunk_length or cfg.learner.rollout_chunk_length, updates_per_rollout_chunk=args.updates_per_rollout_chunk or cfg.learner.updates_per_rollout_chunk, action_samples=args.action_samples or cfg.learner.action_samples, actor_update_chunk_size=args.actor_update_chunk_size or cfg.learner.actor_update_chunk_size))
    if any(v is not None for v in (args.hidden_dim, args.critic_hidden_dim, args.critic_mlp_hidden_dim)):
        cfg = replace(cfg, network=replace(cfg.network, hidden_dim=args.hidden_dim or cfg.network.hidden_dim, critic_hidden_dim=args.critic_hidden_dim or cfg.network.critic_hidden_dim, critic_mlp_hidden_dim=args.critic_mlp_hidden_dim or cfg.network.critic_mlp_hidden_dim))
    if any(v is not None for v in (args.checkpoint_interval_steps, args.eval_interval_steps, args.eval_episodes, args.eval_max_steps, args.actor_workers)):
        cfg = replace(cfg, training=replace(cfg.training, checkpoint_interval_steps=args.checkpoint_interval_steps or cfg.training.checkpoint_interval_steps, eval_interval_steps=args.eval_interval_steps or cfg.training.eval_interval_steps, eval_episodes=args.eval_episodes or cfg.training.eval_episodes, eval_max_steps=args.eval_max_steps or cfg.training.eval_max_steps, actor_workers=args.actor_workers or cfg.training.actor_workers))
    return cfg


def _reward_value_grid(base: RewardConfig, args: argparse.Namespace) -> list[tuple[str, list[float]]]:
    raw_values = [
        ("shape_good_m", args.shape_good_values),
        ("shape_bad_m", args.shape_bad_values),
        ("ip_good_a", args.ip_good_values),
        ("ip_bad_a", args.ip_bad_values),
        ("shape_weight", args.shape_weight_values),
        ("ip_weight", args.ip_weight_values),
        ("current_good_a", args.current_good_values),
        ("current_bad_a", args.current_bad_values),
        ("terminal_reward", args.terminal_reward_values),
        ("reward_scale", args.reward_scale_values),
    ]
    return [(name, _float_values(raw, getattr(base, name))) for name, raw in raw_values]


def _candidate_rewards(base: RewardConfig, args: argparse.Namespace) -> Iterable[RewardConfig]:
    yield from _candidate_rewards_from_values(base, _reward_value_grid(base, args))


def _candidate_rewards_from_values(base: RewardConfig, values: Sequence[tuple[str, list[float]]]) -> Iterable[RewardConfig]:
    seen: set[tuple[float, ...]] = set()
    for combo in itertools.product(*(vals for _name, vals in values)):
        data = asdict(base)
        for (name, _vals), value in zip(values, combo):
            data[name] = float(value)
        _validate_reward_candidate(data)
        reward = RewardConfig(**data)
        key = _reward_key(reward)
        if key not in seen:
            seen.add(key)
            yield reward


def _gpu_devices(args: argparse.Namespace) -> list[str]:
    raw = args.gpu_devices
    if raw is None or str(raw).strip() == "":
        return [str(i) for i in range(max(1, int(getattr(args, "parallel_candidates", 1))))]
    out = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not out:
        raise ValueError("--gpu-devices must not be empty")
    return out


def _float_values(raw: str | None, default: float) -> list[float]:
    if raw is None or str(raw).strip() == "":
        return [float(default)]
    out = []
    for value in str(raw).split(","):
        if value.strip():
            parsed = float(value.strip())
            if parsed not in out:
                out.append(parsed)
    if not out:
        raise ValueError("value list must not be empty")
    return out


def _int_values(raw: str) -> list[int]:
    out = [int(v.strip()) for v in str(raw).split(",") if v.strip()]
    if not out:
        raise ValueError("integer value list must not be empty")
    if any(v <= 0 for v in out):
        raise ValueError("integer value list entries must be positive")
    return out


def _optional_float_values(raw: str | None) -> list[float] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return _float_values(raw, 0.0)


def _stage_float_value(raw: str | None, stage_index: int) -> float | None:
    values = _optional_float_values(raw)
    if not values:
        return None
    if len(values) != len(STAGE_NAMES):
        raise ValueError(f"stage threshold lists must contain {len(STAGE_NAMES)} values")
    return float(values[int(stage_index) - 1])


def _promotion_policy_for_stage(args: argparse.Namespace, stage_index: int) -> dict[str, float | None]:
    if int(stage_index) < 1:
        return {"max_shape_degradation_m": None, "min_ip_improvement_a": None}
    return {
        "max_shape_degradation_m": _stage_float_value(args.stage_max_shape_degradation_m, stage_index),
        "min_ip_improvement_a": _stage_float_value(args.stage_min_ip_improvement_a, stage_index),
    }


def _validate_reward_candidate(data: dict[str, float]) -> None:
    if data["shape_bad_m"] <= data["shape_good_m"]:
        raise ValueError("shape_bad_m must be greater than shape_good_m")
    if data["ip_bad_a"] <= data["ip_good_a"]:
        raise ValueError("ip_bad_a must be greater than ip_good_a")
    if data["current_bad_a"] <= data["current_good_a"]:
        raise ValueError("current_bad_a must be greater than current_good_a")
    if data["shape_weight"] <= 0.0:
        raise ValueError("shape_weight must be positive")
    if data["ip_weight"] <= 0.0:
        raise ValueError("ip_weight must be positive")
    if data["reward_scale"] <= 0.0:
        raise ValueError("reward_scale must be positive")


def _rank_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        copied = dict(row)
        copied.setdefault("promotion_reason", _promotion_reason(copied))
        annotated.append(copied)
    return sorted(annotated, key=_ranking_key)


def _ranking_key(row: dict[str, object]) -> tuple[float, ...]:
    status_bad = 0.0 if row.get("status") == "ok" else 1.0
    rejected = 0.0 if row.get("promotion_reason") == "eligible" else 1.0
    shape = _row_float(row, "eval.shape_error_mean_m")
    ip = _row_float(row, "eval.ip_error_a")
    improvement = _row_float(row, "eval.improvement_over_no_control", default=-float("inf"))
    shape_improvement = _row_float(row, "eval.shape_improvement_over_no_control_m", default=-float("inf"))
    ip_improvement = _row_float(row, "eval.ip_improvement_over_no_control_a", default=-float("inf"))
    delta_quality = _row_float(row, "eval.delta_action_quality", default=-float("inf"))
    action_quality = _row_float(row, "eval.action_quality", default=-float("inf"))
    mean_return = _row_float(row, "eval.mean_return", default=-float("inf"))
    return (status_bad, rejected, shape, -ip_improvement, ip, -shape_improvement, -improvement, -delta_quality, -action_quality if math.isfinite(action_quality) else float("inf"), -mean_return)


def _promotion_reason(row: dict[str, object]) -> str:
    if row.get("status") not in (None, "ok"):
        return str(row.get("status", "error"))
    boundary = _row_float(row, "eval.boundary_found", default=float("nan"))
    if not math.isfinite(boundary) or boundary < 0.995:
        return "rejected_boundary_not_reliable"
    current = _row_float(row, "eval.current_over_limit_a", default=float("nan"))
    if not math.isfinite(current) or current > 1.0e-6:
        return "rejected_current_limit"
    shape = _row_float(row, "eval.shape_error_mean_m", default=float("nan"))
    ip = _row_float(row, "eval.ip_error_a", default=float("nan"))
    if not math.isfinite(shape) or not math.isfinite(ip):
        return "rejected_missing_physical_metrics"
    require_no_control = bool(int(_row_float(row, "search.require_no_control_improvement", default=0.0)))
    max_shape_degradation = _row_float(row, "search.max_shape_degradation_m", default=float("nan"))
    min_ip_improvement = _row_float(row, "search.min_ip_improvement_a", default=float("nan"))
    shape_improvement = _row_float(row, "eval.shape_improvement_over_no_control_m", default=float("nan"))
    ip_improvement = _row_float(row, "eval.ip_improvement_over_no_control_a", default=float("nan"))
    if math.isfinite(max_shape_degradation) and math.isfinite(shape_improvement) and shape_improvement < -float(max_shape_degradation):
        return "rejected_shape_degradation_over_limit"
    if math.isfinite(min_ip_improvement) and math.isfinite(ip_improvement) and ip_improvement < float(min_ip_improvement):
        return "rejected_ip_improvement_below_minimum"
    if require_no_control:
        if math.isfinite(shape_improvement) and shape_improvement <= 0.0:
            return "rejected_shape_not_better_than_no_control"
        if math.isfinite(ip_improvement) and ip_improvement <= 0.0:
            return "rejected_ip_not_better_than_no_control"
    return "eligible"


def _refine_candidates(top_candidates: Sequence[RewardCandidate], *, value_grid: Sequence[tuple[str, list[float]]], existing: set[tuple[float, ...]], next_index: int) -> list[RewardCandidate]:
    if len(top_candidates) < 2:
        return []
    values_by_name = {name: sorted(set(float(v) for v in values)) for name, values in value_grid}
    variable_fields = [name for name in REWARD_FIELDS if len({float(getattr(candidate.reward, name)) for candidate in top_candidates}) > 1]
    refined: list[RewardCandidate] = []
    for candidate in top_candidates:
        for name in variable_fields:
            values = values_by_name.get(name, [])
            current = float(getattr(candidate.reward, name))
            if current not in values:
                continue
            pos = values.index(current)
            neighbors = []
            if pos > 0:
                neighbors.append(values[pos - 1])
            if pos + 1 < len(values):
                neighbors.append(values[pos + 1])
            for neighbor in neighbors:
                data = asdict(candidate.reward)
                data[name] = 0.5 * (current + float(neighbor))
                try:
                    _validate_reward_candidate(data)
                except ValueError:
                    continue
                reward = RewardConfig(**data)
                key = _reward_key(reward)
                if key in existing:
                    continue
                existing.add(key)
                refined.append(RewardCandidate(index=next_index + len(refined), reward=reward, source="midpoint_refinement", parent=candidate.index))
    return refined


def _best_eval_metrics(result: dict[str, object]) -> dict[str, float]:
    raw = result.get("best_eval_details", {})
    if isinstance(raw, dict) and raw:
        return {str(k): float(v) for k, v in raw.items() if _is_number(v)}
    best = result.get("best_eval", -float("inf"))
    return {"mean_return": float(best) if _is_number(best) else -float("inf")}


def _evaluate_no_control(*, config: ExperimentConfig, args: argparse.Namespace, episodes: int, max_steps: int) -> dict[str, float]:
    device = _resolve_torch_device(args.device or config.training.device)
    batch_size = max(1, min(int(episodes), int(args.num_envs or config.training.num_envs)))
    env = TokamakMagneticControlEnv(config, batch_size=batch_size, device=device, seed=int(config.training.seed) + 100000)
    returns: list[float] = []
    component_values: dict[str, list[float]] = {}
    remaining = int(episodes)
    while remaining > 0:
        obs = env.reset()
        total = torch.zeros((env.batch_size,), dtype=torch.float32, device=device)
        for _ in range(int(max_steps)):
            action = torch.zeros((env.batch_size, env.action_dim), dtype=torch.float32, device=device)
            out = env.step(action)
            total += out.reward
            comps = out.info.get("reward_components", {}) if isinstance(out.info, dict) else {}
            if isinstance(comps, dict):
                for name, value in comps.items():
                    arr = np.asarray(value, dtype=float).reshape(-1)
                    if arr.size:
                        component_values.setdefault(str(name), []).extend(arr[np.isfinite(arr)].astype(float).tolist())
            obs = out.obs
            if bool(torch.all(out.terminated | out.truncated).item()):
                break
        returns.extend(total.detach().cpu().numpy().astype(float).tolist())
        remaining -= env.batch_size
    selected_returns = np.asarray(returns[: int(episodes)], dtype=float)
    metrics: dict[str, float] = {"mean_return": float(np.nanmean(selected_returns)) if selected_returns.size else float("nan")}
    for name, values in component_values.items():
        arr = np.asarray(values, dtype=float)
        if arr.size:
            metrics[name] = float(np.nanmean(arr))
    return metrics


def _resolve_torch_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return device


def _start_wandb(*, args: argparse.Namespace, cfg: ExperimentConfig, candidate: RewardCandidate, stage_name: str, stage_index: int):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    import wandb
    name_base = args.wandb_name or Path(args.output_dir).name
    run = wandb.init(
        project=args.wandb_project,
        name=f"{name_base}_{stage_name}_candidate_{candidate.index:04d}",
        group=args.wandb_group or Path(args.output_dir).name,
        mode=args.wandb_mode,
        config={"experiment": cfg.name, "search": {"stage": stage_name, "stage_index": int(stage_index), "candidate": candidate.index, "candidate_source": candidate.source, "candidate_parent": candidate.parent}, "reward": asdict(candidate.reward)},
        reinit=True,
    )
    run.log({"search/stage": int(stage_index), "search/candidate": int(candidate.index)}, step=0)
    return run


def _candidate_row(candidate: RewardCandidate, *, stage: str, candidate_dir: Path) -> dict[str, object]:
    return {"candidate": candidate.index, "stage": stage, "status": "pending", "candidate_source": candidate.source, "candidate_parent": "" if candidate.parent is None else candidate.parent, "candidate_dir": str(candidate_dir), **{f"reward.{k}": v for k, v in asdict(candidate.reward).items()}}


def _candidate_by_index(candidates: Sequence[RewardCandidate], index: int) -> RewardCandidate:
    for candidate in candidates:
        if candidate.index == int(index):
            return candidate
    raise KeyError(f"unknown candidate index: {index}")


def _reward_from_row(row: dict[str, object]) -> RewardConfig:
    data = {field: float(row[f"reward.{field}"]) for field in REWARD_FIELDS}
    return RewardConfig(**data)


def _candidate_from_row(row: dict[str, object]) -> RewardCandidate:
    return RewardCandidate(index=int(row["candidate"]), reward=_reward_from_row(row), source=str(row.get("candidate_source", "resume")), parent=_optional_int(row.get("candidate_parent")))


def _reward_key(reward: RewardConfig) -> tuple[float, ...]:
    return tuple(float(getattr(reward, name)) for name in REWARD_FIELDS)


def _steps_or_default(args: argparse.Namespace, base: ExperimentConfig) -> int:
    return int(base.training.steps if args.steps is None else args.steps)


def _row_float(row: dict[str, object], key: str, *, default: float = float("inf")) -> float:
    try:
        value = row.get(key, default)
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def _is_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _summarize_reward_components(path: Path, *, tail_rows: int = 100) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    subset = rows[-min(int(tail_rows), len(rows)):]
    out: dict[str, float] = {}
    for key in rows[0]:
        if key == "step":
            continue
        vals = []
        for row in subset:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        if vals:
            out[f"tail100.{key}"] = float(np.nanmean(vals))
    return out


def _write_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_results(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"results file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _reward_label(reward: RewardConfig) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in asdict(reward).items())


if __name__ == "__main__":
    raise SystemExit(main())
