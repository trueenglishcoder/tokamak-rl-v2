from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.config.loader import _validate_experiment_config
from tokamak_rl_v2.config.schema import ExperimentConfig
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.training.cli import _device_list
from tokamak_rl_v2.training.trainer import Trainer, _eval_max_steps_for_config, filter_wandb_metrics


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.eval_seed_offset) < 0 or int(args.holdout_eval_seed_offset) < 0:
        raise ValueError("evaluation seed offsets must be non-negative")
    if int(args.eval_seed_offset) == int(args.holdout_eval_seed_offset):
        raise ValueError("holdout_eval_seed_offset must differ from eval_seed_offset")
    cfg = _apply_overrides(load_experiment_config(args.config), args)
    _validate_experiment_config(cfg)
    if bool(cfg.training.production_mode) and bool(args.skip_controller_rollout_gate) and not bool(args.reward_sweep_mode):
        raise ValueError("production_mode rejects --skip-controller-rollout-gate")
    if bool(cfg.training.production_mode):
        controller_rollout_steps = int(args.controller_rollout_steps)
        if controller_rollout_steps not in (0, int(cfg.sim.max_episode_steps)):
            raise ValueError("production_mode requires --controller-rollout-steps to be 0 or exactly sim.max_episode_steps")
    output_dir = Path(args.output_dir or cfg.training.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.training.production_mode) and bool(args.allow_failed_gates):
        raise ValueError("production_mode rejects --allow-failed-gates")
    wandb_run = None
    gate_profile = _gate_profile_for_config(cfg)
    production_mode = bool(cfg.training.production_mode)
    rank0 = _distributed_rank() == 0
    runtime_device = _rank_runtime_device(args.device or cfg.training.device, cfg)
    runtime_cfg = _rank_runtime_config(cfg, runtime_device)
    previous_signal_handlers = _install_shutdown_signal_handlers()
    try:
        artifact_failure = _preflight_artifact_failure(runtime_cfg)
        if artifact_failure is not None:
            if rank0:
                report = {
                    "status": artifact_failure["status"],
                    "config": str(Path(args.config).resolve()),
                    "output_dir": str(output_dir),
                    "artifact_preflight": artifact_failure,
                    "gates": [{"name": artifact_failure["name"], "passed": False, "value": artifact_failure["path"], "threshold": "exists"}],
                }
                _write_json(output_dir / "policy_validation.json", report)
            return 2 if not args.allow_failed_gates else 0

        reset_report: dict[str, float] = {}
        baseline: Mapping[str, float] = {}
        holdout_baseline: Mapping[str, float] = {}
        baseline_difficulty: dict[str, float] = {}
        selection_seed_offset = int(args.eval_seed_offset)
        holdout_seed_offset = int(args.holdout_eval_seed_offset)
        if not production_mode:
            reset_report = run_reset_sanity(runtime_cfg, device=runtime_device, num_envs=args.num_envs)
            reset_gate = reset_report["max_abs_boundary_radii_error_m"] <= float(args.reset_error_tolerance_m) and reset_report["boundary_found_mean"] >= float(args.min_boundary_found)
            if not reset_gate:
                if rank0:
                    report = {
                        "status": "failed_reset_sanity",
                        "reset_sanity": reset_report,
                        "gates": [{"name": "reset_sanity", "passed": False}],
                    }
                    _write_json(output_dir / "policy_validation.json", report)
                return 2 if not args.allow_failed_gates else 0

        trainer = Trainer(
            cfg,
            steps=args.steps,
            num_envs=args.num_envs,
            device=args.device,
            output_dir=output_dir,
            wandb_run=None,
            resume_checkpoint=args.resume_checkpoint,
            export_policy=not bool(args.no_export),
            wandb_metric_preset=args.wandb_metric_preset,
        )
        wandb_run = _start_wandb(args, cfg, output_dir=output_dir)
        trainer.wandb_run = wandb_run
        trainer._configure_wandb_metrics()
        if not production_mode and rank0:
            _wandb_log(wandb_run, "pipeline/reset", reset_report, step=0)
        if not production_mode:
            baseline = trainer.evaluate_detailed(episodes=int(cfg.training.eval_episodes), max_steps=_eval_max_steps_for_config(cfg), policy="no_control", seed_offset=selection_seed_offset)
            holdout_baseline = trainer.evaluate_detailed(episodes=int(cfg.training.eval_episodes), max_steps=_eval_max_steps_for_config(cfg), policy="no_control", seed_offset=holdout_seed_offset)
            if rank0:
                _write_baseline_report(
                    output_dir,
                    reset_sanity=reset_report,
                    no_control_selection=baseline,
                    no_control=holdout_baseline,
                    selection_seed_offset=selection_seed_offset,
                    holdout_seed_offset=holdout_seed_offset,
                )
                _wandb_log(wandb_run, "pipeline/no_control_selection", baseline, step=0)
                _wandb_log(wandb_run, "pipeline/no_control_holdout", holdout_baseline, step=0)
            baseline_difficulty = _baseline_difficulty_report(
                holdout_baseline,
                min_ip_error_a=float(gate_profile["min_baseline_ip_error_a"]),
                min_ip_error_late_a=float(gate_profile["min_baseline_ip_error_late_a"]),
            )
            if rank0:
                _wandb_log(wandb_run, "pipeline/no_control_difficulty", baseline_difficulty, step=0)
            if bool(gate_profile["enforce_baseline_difficulty"]) and not bool(baseline_difficulty["passed"]):
                if rank0:
                    report = {
                        "status": "failed_baseline_difficulty",
                        "config": str(Path(args.config).resolve()),
                        "output_dir": str(output_dir),
                        "evaluation_seed_offsets": {
                            "selection": selection_seed_offset,
                            "holdout": holdout_seed_offset,
                        },
                        "reset_sanity": reset_report,
                        "no_control_selection": baseline,
                        "no_control": holdout_baseline,
                        "baseline_difficulty": baseline_difficulty,
                        "gates": [
                            {
                                "name": "baseline_difficulty",
                                "passed": False,
                                "value": {
                                    "ip_error_a": holdout_baseline.get("ip_error_a"),
                                    "ip_error_a_late": holdout_baseline.get("ip_error_a_late"),
                                },
                                "threshold": {
                                    "ip_error_a": f">= {float(gate_profile['min_baseline_ip_error_a']):g}",
                                    "ip_error_a_late": f">= {float(gate_profile['min_baseline_ip_error_late_a']):g}",
                                },
                            }
                        ],
                    }
                    _write_json(output_dir / "policy_validation.json", report)
                    _wandb_log(wandb_run, "pipeline", {"passed": 0.0, "failed_baseline_difficulty": 1.0}, step=0)
                return 2 if not args.allow_failed_gates else 0
        train_result = trainer.train()
        if cfg.training.distributed_mode == "local_replay" and not rank0:
            return 0
        train_status = str(train_result.get("status", "completed")) if isinstance(train_result, Mapping) else "completed"
        if train_status.startswith("failed_"):
            if rank0:
                status = "sweep_failed_training" if bool(args.reward_sweep_mode) else train_status
                report = {
                    "status": status,
                    "training_status": train_status,
                    "config": str(Path(args.config).resolve()),
                    "output_dir": str(output_dir),
                    "reset_sanity": reset_report,
                    "no_control_selection": baseline,
                    "no_control": holdout_baseline,
                    "train_result": train_result,
                    "gates": [{"name": train_status, "passed": False, "value": train_result.get("failure_details", {}) if isinstance(train_result, Mapping) else {}, "threshold": "training fail-fast must pass"}],
                }
                _write_json(output_dir / "policy_validation.json", report)
                _wandb_log(wandb_run, "pipeline", {"passed": 0.0, train_status: 1.0}, step=_train_env_step(train_result, cfg))
            if bool(args.reward_sweep_mode):
                return 0
            return 2 if not args.allow_failed_gates else 0

        selected_checkpoint = _selected_checkpoint(output_dir)
        if selected_checkpoint is not None and selected_checkpoint.name == "best.pt":
            _load_actor_weights(trainer, selected_checkpoint)
        elif selected_checkpoint is None:
            trainer.restore_best_actor()
        try:
            actor_eval = trainer.evaluate_detailed(episodes=int(cfg.training.eval_episodes), max_steps=_eval_max_steps_for_config(cfg), policy="actor", seed_offset=holdout_seed_offset)
        except Exception as exc:
            if rank0 and bool(args.reward_sweep_mode):
                report = {
                    "status": "sweep_failed_eval",
                    "config": str(Path(args.config).resolve()),
                    "output_dir": str(output_dir),
                    "evaluation_seed_offsets": {
                        "selection": selection_seed_offset,
                        "holdout": holdout_seed_offset,
                    },
                    "train_result": train_result,
                    "error": repr(exc),
                    "gates": [{"name": "actor_eval", "passed": False, "value": repr(exc), "threshold": "eval must complete"}],
                }
                _write_json(output_dir / "policy_validation.json", report)
                _wandb_log(wandb_run, "pipeline", {"sweep_failed_eval": 1.0}, step=_train_env_step(train_result, cfg))
                return 0
            raise
        train_env_step = _train_env_step(train_result, cfg)
        _wandb_log(wandb_run, "pipeline/actor_eval_holdout", actor_eval, step=train_env_step)
        losses = summarize_training_losses(output_dir / "losses.csv")
        selected_export = _selected_export_dir(output_dir)
        if bool(args.reward_sweep_mode):
            gate_report = evaluate_policy_gates(
                actor_eval=actor_eval,
                no_control=holdout_baseline,
                tail_losses=losses,
                action_samples=int(cfg.learner.action_samples),
                min_boundary_found=float(args.min_boundary_found),
                max_current_over_limit_a=float(args.max_current_over_limit_a),
                max_shape_error_m=float(args.max_shape_error_m),
                min_ip_improvement_frac=float(args.min_ip_improvement_frac),
                min_ip_improvement_a=float(args.min_ip_improvement_a),
                max_ip_error_a=gate_profile["max_ip_error_a"],
                max_ip_error_late_a=gate_profile["max_ip_error_late_a"],
                min_action_rms=float(args.min_action_rms),
                max_action_rms=float(args.max_action_rms),
                min_mean_episode_completion=float(gate_profile["min_mean_episode_completion"]),
                min_episode_completion=float(gate_profile["min_episode_completion"]),
                min_baseline_ip_error_a=float(gate_profile["min_baseline_ip_error_a"]),
                min_baseline_ip_error_late_a=float(gate_profile["min_baseline_ip_error_late_a"]),
                always_require_ip_improvement=bool(gate_profile["always_require_ip_improvement"]),
                min_policy_weight_extra=float(args.min_policy_weight_extra),
                min_sampled_q_spread=float(args.min_sampled_q_spread),
                include_mpo_gates=False,
                require_controller_rollout=False,
                controller_rollout={},
                max_controller_shape_error_m=float(args.max_controller_shape_error_m),
                max_controller_ip_error_a=float(gate_profile["max_controller_ip_error_a"]),
            )
            report = {
                "status": "sweep_completed",
                "gate_passed": bool(gate_report["passed"]),
                "config": str(Path(args.config).resolve()),
                "output_dir": str(output_dir),
                "evaluation_seed_offsets": {
                    "selection": selection_seed_offset,
                    "holdout": holdout_seed_offset,
                },
                "checkpoint": None if selected_checkpoint is None else str(selected_checkpoint),
                "export_dir": None if selected_export is None else str(selected_export),
                "reset_sanity": reset_report,
                "no_control_selection": baseline,
                "no_control": holdout_baseline,
                "baseline_difficulty": baseline_difficulty,
                "train_result": train_result,
                "actor_eval": actor_eval,
                "tail_losses": losses,
                "gates": gate_report["checks"],
            }
            _write_json(output_dir / "policy_validation.json", report)
            _wandb_log(wandb_run, "pipeline/tail_losses", losses, step=train_env_step)
            _wandb_log(wandb_run, "pipeline/gates", _gate_metrics(gate_report["checks"]), step=train_env_step)
            _wandb_log(wandb_run, "pipeline", {"sweep_completed": 1.0, "gate_passed": 1.0 if gate_report["passed"] else 0.0}, step=train_env_step)
            return 0
        controller_steps = int(args.controller_rollout_steps) if int(args.controller_rollout_steps) > 0 else int(cfg.sim.max_episode_steps)
        rollout_report = validate_exported_controller(selected_export, cfg, steps=controller_steps) if selected_export is not None else {"status": "missing_export"}

        gate_report = evaluate_policy_gates(
            actor_eval=actor_eval,
            no_control=holdout_baseline,
            tail_losses=losses,
            action_samples=int(cfg.learner.action_samples),
            min_boundary_found=float(args.min_boundary_found),
            max_current_over_limit_a=float(args.max_current_over_limit_a),
            max_shape_error_m=float(args.max_shape_error_m),
            min_ip_improvement_frac=float(args.min_ip_improvement_frac),
            min_ip_improvement_a=float(args.min_ip_improvement_a),
            max_ip_error_a=gate_profile["max_ip_error_a"],
            max_ip_error_late_a=gate_profile["max_ip_error_late_a"],
            min_action_rms=float(args.min_action_rms),
            max_action_rms=float(args.max_action_rms),
            min_mean_episode_completion=float(gate_profile["min_mean_episode_completion"]),
            min_episode_completion=float(gate_profile["min_episode_completion"]),
            min_baseline_ip_error_a=float(gate_profile["min_baseline_ip_error_a"]),
            min_baseline_ip_error_late_a=float(gate_profile["min_baseline_ip_error_late_a"]),
            always_require_ip_improvement=bool(gate_profile["always_require_ip_improvement"]),
            min_policy_weight_extra=float(args.min_policy_weight_extra),
            min_sampled_q_spread=float(args.min_sampled_q_spread),
            include_mpo_gates=not production_mode,
            require_controller_rollout=not bool(args.skip_controller_rollout_gate),
            controller_rollout=rollout_report,
            max_controller_shape_error_m=float(args.max_controller_shape_error_m),
            max_controller_ip_error_a=float(gate_profile["max_controller_ip_error_a"]),
        )
        report = {
            "status": "passed" if gate_report["passed"] else "failed_gates",
            "config": str(Path(args.config).resolve()),
            "output_dir": str(output_dir),
            "evaluation_seed_offsets": {
                "selection": selection_seed_offset,
                "holdout": holdout_seed_offset,
            },
            "checkpoint": None if selected_checkpoint is None else str(selected_checkpoint),
            "export_dir": None if selected_export is None else str(selected_export),
            "reset_sanity": reset_report,
            "no_control_selection": baseline,
            "no_control": holdout_baseline,
            "baseline_difficulty": baseline_difficulty,
            "train_result": train_result,
            "actor_eval": actor_eval,
            "tail_losses": losses,
            "controller_rollout": rollout_report,
            "gates": gate_report["checks"],
        }
        _write_json(output_dir / "policy_validation.json", report)
        _write_json(output_dir / "closed_loop_rollout_report.json", rollout_report)
        _wandb_log(wandb_run, "pipeline/tail_losses", losses, step=train_env_step)
        _wandb_log(wandb_run, "pipeline/controller_rollout", rollout_report, step=train_env_step)
        _wandb_log(wandb_run, "pipeline/gates", _gate_metrics(gate_report["checks"]), step=train_env_step)
        return 0 if gate_report["passed"] or args.allow_failed_gates else 2
    except KeyboardInterrupt as exc:
        _write_json(
            output_dir / "policy_validation.json",
            {
                "status": "interrupted",
                "reason": str(exc) or "training interrupted",
                "output_dir": str(output_dir),
            },
        )
        _wandb_log(wandb_run, "pipeline", {"interrupted": 1.0}, step=0)
        return 130
    finally:
        _restore_signal_handlers(previous_signal_handlers)
        if wandb_run is not None:
            wandb_run.finish()


def run_reset_sanity(config: ExperimentConfig, *, device: str | None = None, num_envs: int | None = None) -> dict[str, float]:
    dev = _resolve_device(device or config.training.device)
    batch_size = max(1, min(int(num_envs or config.training.num_envs), 8))
    env = TokamakMagneticControlEnv(config, batch_size=batch_size, device=dev, seed=int(config.training.seed) + 900000)
    obs = env.reset()
    schema = env.export_schema()
    slices = schema["feature_slices"]
    err0, err1 = slices["boundary_radii_error"]
    found0, found1 = slices["boundary_found"]
    errors = obs[:, int(err0) : int(err1)].detach().cpu().numpy().astype(float)
    found = obs[:, int(found0) : int(found1)].detach().cpu().numpy().astype(float).reshape(-1)
    return {
        "batch_size": float(batch_size),
        "max_abs_boundary_radii_error_m": float(np.nanmax(np.abs(errors))) if errors.size else float("nan"),
        "mean_abs_boundary_radii_error_m": float(np.nanmean(np.abs(errors))) if errors.size else float("nan"),
        "boundary_found_mean": float(np.nanmean(found)) if found.size else float("nan"),
        "boundary_found_min": float(np.nanmin(found)) if found.size else float("nan"),
    }


def _write_baseline_report(
    output_dir: Path,
    *,
    reset_sanity: Mapping[str, float],
    no_control_selection: Mapping[str, float],
    no_control: Mapping[str, float],
    selection_seed_offset: int,
    holdout_seed_offset: int,
) -> None:
    _write_json(
        output_dir / "no_control_baseline.json",
        {
            "status": "ready_for_training",
            "evaluation_seed_offsets": {
                "selection": int(selection_seed_offset),
                "holdout": int(holdout_seed_offset),
            },
            "reset_sanity": dict(reset_sanity),
            "no_control_selection": dict(no_control_selection),
            "no_control": dict(no_control),
        },
    )


def _install_shutdown_signal_handlers() -> dict[signal.Signals, signal.Handlers]:
    previous: dict[signal.Signals, signal.Handlers] = {}

    def request_shutdown(signum, _frame) -> None:
        name = signal.Signals(signum).name
        raise KeyboardInterrupt(f"received {name}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, request_shutdown)
    return previous


def _restore_signal_handlers(previous: Mapping[signal.Signals, signal.Handlers]) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def evaluate_policy_gates(
    *,
    actor_eval: Mapping[str, float],
    no_control: Mapping[str, float],
    tail_losses: Mapping[str, float],
    action_samples: int,
    min_boundary_found: float,
    max_current_over_limit_a: float,
    max_shape_error_m: float,
    min_ip_improvement_frac: float,
    min_ip_improvement_a: float,
    max_ip_error_a: float | None,
    max_ip_error_late_a: float | None,
    min_action_rms: float,
    max_action_rms: float,
    min_mean_episode_completion: float,
    min_episode_completion: float,
    min_baseline_ip_error_a: float,
    min_baseline_ip_error_late_a: float,
    always_require_ip_improvement: bool,
    min_policy_weight_extra: float,
    min_sampled_q_spread: float,
    include_mpo_gates: bool,
    require_controller_rollout: bool,
    controller_rollout: Mapping[str, object],
    max_controller_shape_error_m: float,
    max_controller_ip_error_a: float,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, *, value: object, threshold: object, details: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value, "threshold": threshold, "details": details})

    boundary_found = _metric(actor_eval, "boundary_found")
    add("boundary_found", _finite(boundary_found) and boundary_found >= min_boundary_found, value=boundary_found, threshold=f">= {min_boundary_found:g}")
    boundary_found_late = _metric(actor_eval, "boundary_found_late_min", default=_metric(actor_eval, "boundary_found_min", default=boundary_found))
    add("boundary_found_late", _finite(boundary_found_late) and boundary_found_late >= min_boundary_found, value=boundary_found_late, threshold=f"late/min >= {min_boundary_found:g}")

    current_over_mean = _metric(actor_eval, "current_over_limit_a")
    current_over_max = _metric(actor_eval, "current_over_limit_a_max", default=current_over_mean)
    add("current_limit", _finite(current_over_max) and current_over_max <= max_current_over_limit_a, value={"max_a": current_over_max, "mean_a": current_over_mean, "fraction": _metric(actor_eval, "current_over_limit_fraction")}, threshold=f"max <= {max_current_over_limit_a:g} A")
    current_over_late = _metric(actor_eval, "current_over_limit_a_late_max", default=current_over_max)
    add("current_limit_late", _finite(current_over_late) and current_over_late <= max_current_over_limit_a, value={"late_max_a": current_over_late, "late_fraction": _metric(actor_eval, "current_over_limit_fraction_late")}, threshold=f"late/max <= {max_current_over_limit_a:g} A")

    shape_error = _metric(actor_eval, "shape_error_mean_m")
    add("shape_error_mean", _finite(shape_error) and shape_error <= max_shape_error_m, value=shape_error, threshold=f"<= {max_shape_error_m:g} m")
    shape_error_late = _metric(actor_eval, "shape_error_mean_m_late", default=shape_error)
    shape_drift = _metric(actor_eval, "shape_error_mean_m_late_minus_early")
    add("shape_error_late", _finite(shape_error_late) and shape_error_late <= max_shape_error_m, value={"late_m": shape_error_late, "late_minus_early_m": shape_drift}, threshold=f"late <= {max_shape_error_m:g} m")

    baseline_ip = _metric(no_control, "ip_error_a")
    actor_ip = _metric(actor_eval, "ip_error_a")
    baseline_ip_late = _metric(no_control, "ip_error_a_late", default=baseline_ip)
    actor_ip_late = _metric(actor_eval, "ip_error_a_late", default=actor_ip)
    if float(min_baseline_ip_error_a) > 0.0 or float(min_baseline_ip_error_late_a) > 0.0:
        add(
            "task_difficulty",
            _finite(baseline_ip)
            and baseline_ip >= float(min_baseline_ip_error_a)
            and _finite(baseline_ip_late)
            and baseline_ip_late >= float(min_baseline_ip_error_late_a),
            value={"baseline_a": baseline_ip, "baseline_late_a": baseline_ip_late},
            threshold={
                "baseline_a": f">= {float(min_baseline_ip_error_a):g} A",
                "baseline_late_a": f">= {float(min_baseline_ip_error_late_a):g} A",
            },
        )
    if max_ip_error_a is not None:
        add("ip_error_mean", _finite(actor_ip) and actor_ip <= float(max_ip_error_a), value=actor_ip, threshold=f"<= {float(max_ip_error_a):g} A")
    if max_ip_error_late_a is not None:
        add("ip_error_late", _finite(actor_ip_late) and actor_ip_late <= float(max_ip_error_late_a), value=actor_ip_late, threshold=f"late <= {float(max_ip_error_late_a):g} A")

    ip_improvement = baseline_ip - actor_ip if _finite(baseline_ip) and _finite(actor_ip) else float("nan")
    ip_improvement_frac = ip_improvement / max(abs(baseline_ip), 1.0e-12) if _finite(ip_improvement) and _finite(baseline_ip) else float("nan")
    if always_require_ip_improvement or (_finite(baseline_ip) and baseline_ip >= min_ip_improvement_a):
        add(
            "ip_improvement",
            ip_improvement >= min_ip_improvement_a and ip_improvement_frac >= min_ip_improvement_frac,
            value={"absolute_a": ip_improvement, "fraction": ip_improvement_frac, "baseline_a": baseline_ip, "actor_a": actor_ip},
            threshold=f">= {min_ip_improvement_a:g} A and >= {min_ip_improvement_frac:g}",
        )
    else:
        add(
            "ip_improvement",
            True,
            value={"absolute_a": ip_improvement, "fraction": ip_improvement_frac, "baseline_a": baseline_ip, "actor_a": actor_ip},
            threshold=f"skipped unless no-control Ip error >= {min_ip_improvement_a:g} A",
            details="baseline error is below the absolute gate",
        )

    ip_late_improvement = baseline_ip_late - actor_ip_late if _finite(baseline_ip_late) and _finite(actor_ip_late) else float("nan")
    ip_late_improvement_frac = ip_late_improvement / max(abs(baseline_ip_late), 1.0e-12) if _finite(ip_late_improvement) and _finite(baseline_ip_late) else float("nan")
    if always_require_ip_improvement or (_finite(baseline_ip_late) and baseline_ip_late >= min_ip_improvement_a):
        add(
            "ip_improvement_late",
            ip_late_improvement >= min_ip_improvement_a and ip_late_improvement_frac >= min_ip_improvement_frac,
            value={"absolute_a": ip_late_improvement, "fraction": ip_late_improvement_frac, "baseline_a": baseline_ip_late, "actor_a": actor_ip_late, "late_minus_early_a": _metric(actor_eval, "ip_error_a_late_minus_early")},
            threshold=f"late >= {min_ip_improvement_a:g} A and >= {min_ip_improvement_frac:g}",
        )
    else:
        add(
            "ip_improvement_late",
            True,
            value={"absolute_a": ip_late_improvement, "fraction": ip_late_improvement_frac, "baseline_a": baseline_ip_late, "actor_a": actor_ip_late},
            threshold=f"skipped unless late no-control Ip error >= {min_ip_improvement_a:g} A",
        )

    action_rms = _metric(actor_eval, "action_rms")
    add("action_rms_min", _finite(action_rms) and action_rms >= min_action_rms, value=action_rms, threshold=f">= {min_action_rms:g}")
    add("action_rms_max", _finite(action_rms) and action_rms < max_action_rms, value=action_rms, threshold=f"< {max_action_rms:g}")

    mean_completion = _metric(actor_eval, "mean_episode_completion", default=_metric(actor_eval, "episode_progress"))
    min_completion = _metric(actor_eval, "min_episode_completion", default=mean_completion)
    add(
        "episode_completion",
        _finite(mean_completion)
        and mean_completion >= min_mean_episode_completion
        and _finite(min_completion)
        and min_completion >= min_episode_completion,
        value={"mean": mean_completion, "min": min_completion, "mean_steps": _metric(actor_eval, "mean_episode_steps"), "min_steps": _metric(actor_eval, "min_episode_steps")},
        threshold=f"mean >= {min_mean_episode_completion:g}, min >= {min_episode_completion:g}",
    )

    if include_mpo_gates:
        policy_weight_max = _metric(tail_losses, "tail100.policy_weight_max")
        uniform = 1.0 / max(int(action_samples), 1)
        policy_weight_threshold = uniform + float(min_policy_weight_extra)
        add("mpo_policy_weights_nonuniform", _finite(policy_weight_max) and policy_weight_max > policy_weight_threshold, value=policy_weight_max, threshold=f"> {policy_weight_threshold:g}")

        q_spread = _metric(tail_losses, "tail100.sampled_q_spread")
        add("mpo_sampled_q_spread", _finite(q_spread) and q_spread >= min_sampled_q_spread, value=q_spread, threshold=f">= {min_sampled_q_spread:g}")

    if require_controller_rollout:
        status = str(controller_rollout.get("status", "missing"))
        add("controller_rollout", status == "ok", value=status, threshold="ok")
        controller_mean_completion = _metric(controller_rollout, "mean_episode_completion")
        controller_min_completion = _metric(controller_rollout, "min_episode_completion")
        add("controller_episode_completion", status == "ok" and _finite(controller_mean_completion) and controller_mean_completion >= min_mean_episode_completion and _finite(controller_min_completion) and controller_min_completion >= min_episode_completion, value={"mean": controller_mean_completion, "min": controller_min_completion}, threshold=f"mean >= {min_mean_episode_completion:g}, min >= {min_episode_completion:g}")
        controller_boundary = _metric(controller_rollout, "boundary_found_mean")
        controller_boundary_late = _metric(controller_rollout, "boundary_found_late_min", default=controller_boundary)
        add("controller_boundary_found", status == "ok" and _finite(controller_boundary) and controller_boundary >= min_boundary_found and _finite(controller_boundary_late) and controller_boundary_late >= min_boundary_found, value={"mean": controller_boundary, "late_min": controller_boundary_late}, threshold=f"mean/late >= {min_boundary_found:g}")
        controller_current = _metric(controller_rollout, "current_over_limit_a_max")
        controller_current_late = _metric(controller_rollout, "current_over_limit_a_late_max", default=controller_current)
        add("controller_current_limit", status == "ok" and _finite(controller_current) and controller_current <= max_current_over_limit_a and _finite(controller_current_late) and controller_current_late <= max_current_over_limit_a, value={"max": controller_current, "late_max": controller_current_late}, threshold=f"max/late <= {max_current_over_limit_a:g} A")
        controller_shape = _metric(controller_rollout, "shape_error_mean_m")
        controller_shape_late = _metric(controller_rollout, "shape_error_late_m", default=controller_shape)
        add("controller_shape_error", status == "ok" and _finite(controller_shape) and controller_shape <= max_controller_shape_error_m and _finite(controller_shape_late) and controller_shape_late <= max_controller_shape_error_m, value={"mean_m": controller_shape, "late_m": controller_shape_late}, threshold=f"mean/late <= {max_controller_shape_error_m:g} m")
        controller_ip = _metric(controller_rollout, "ip_error_a")
        controller_ip_late = _metric(controller_rollout, "ip_error_late_a", default=controller_ip)
        add("controller_ip_error", status == "ok" and _finite(controller_ip) and controller_ip <= max_controller_ip_error_a and _finite(controller_ip_late) and controller_ip_late <= max_controller_ip_error_a, value={"mean_a": controller_ip, "late_a": controller_ip_late}, threshold=f"mean/late <= {max_controller_ip_error_a:g} A")

    return {"passed": all(bool(check["passed"]) for check in checks), "checks": checks}


def summarize_training_losses(path: Path, *, tail_rows: int = 100) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    subset = rows[-min(int(tail_rows), len(rows)) :]
    out: dict[str, float] = {}
    for key in rows[0]:
        if key == "step":
            continue
        vals: list[float] = []
        for row in subset:
            raw = row.get(key, "")
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                vals.append(value)
        if vals:
            out[f"tail100.{key}"] = float(np.nanmean(np.asarray(vals, dtype=float)))
    return out


class _ArrayReferenceScenario:
    def __init__(self, *, ip: np.ndarray, radii: np.ndarray, dt: float) -> None:
        self.ip = np.asarray(ip, dtype=float).reshape(-1)
        self.radii = np.asarray(radii, dtype=float)
        self.dt = max(float(dt), 1.0e-12)
        if self.radii.ndim != 2 or self.radii.shape[0] != self.ip.shape[0]:
            raise ValueError("reference scenario expects radii [steps, angles] aligned with Ip")

    def _index(self, t: float) -> int:
        idx = int(round(float(t) / self.dt))
        return int(np.clip(idx, 0, self.ip.shape[0] - 1))

    def Ip_ref(self, t: float) -> float:
        return float(self.ip[self._index(t)])

    def ref_radii(self, angles: np.ndarray, t: float) -> np.ndarray:
        ref = self.radii[self._index(t)]
        if ref.shape != np.asarray(angles).reshape(-1).shape:
            raise ValueError("reference scenario angle count does not match exported controller request")
        return np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)


def validate_exported_controller(export_dir: Path | None, config: ExperimentConfig, *, steps: int) -> dict[str, object]:
    if export_dir is None:
        return {"status": "missing_export"}
    try:
        from tokamak_control.control.learned_magnetic_controller import LearnedMagneticController
        from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
        from tokamak_control.geometry.legacy_metrics import legacy_radii_at_angles

        rollout_config = replace(config, sim=replace(config.sim, compute_backend="cpu", gpu_device="cuda:0", csv_initial_state_split="holdout"))
        env = TokamakMagneticControlEnv(rollout_config, batch_size=1, device="cpu", seed=int(config.training.seed) + 910000)
        controller = LearnedMagneticController(export_dir=export_dir)
        current_limits = env.current_limits.detach().cpu().numpy().astype(float)
        episode_count = max(1, int(config.training.eval_episodes))
        step_count = max(1, int(steps))
        action_rms_all: list[float] = []
        completion: list[float] = []
        boundary_found_all: list[float] = []
        shape_error_all: list[float] = []
        ip_error_all: list[float] = []
        current_over_all: list[float] = []
        current_usage_all: list[float] = []
        boundary_late: list[float] = []
        shape_late: list[float] = []
        ip_late: list[float] = []
        current_late: list[float] = []
        current_usage_late: list[float] = []

        for _episode in range(episode_count):
            controller.reset()
            env.reset()
            if env.reference is None:
                raise RuntimeError("controller validation environment did not create references")
            model = env._cpu_models[0]
            angles = np.asarray(env.angles, dtype=float)
            center = (model.R0, model.Z0)
            ref_ip = env.reference.ip[0].detach().cpu().numpy().astype(float)
            ref_radii_series = env.reference.radii[0, :, : int(config.sim.angles)].detach().cpu().numpy().astype(float)
            scenario = _ArrayReferenceScenario(ip=ref_ip, radii=ref_radii_series, dt=float(model.t_step))
            first_failure_step: int | None = None
            ep_boundary: list[float] = []
            ep_shape: list[float] = []
            ep_ip: list[float] = []
            ep_current: list[float] = []
            ep_current_usage: list[float] = []
            prev_boundary_poly: np.ndarray | None = None
            prev_boundary_level: float | None = None
            for step_index in range(step_count):
                ref_index = min(step_index, ref_ip.shape[0] - 1)
                ref_radii = np.nan_to_num(ref_radii_series[ref_index], nan=0.0, posinf=0.0, neginf=0.0)
                ip_target = float(ref_ip[ref_index])
                psi = model.compute_psi()
                try:
                    poly, level, _status = find_plasma_boundary_with_status(
                        psi,
                        model.grid,
                        center,
                        n_levels=80,
                        prev_level=prev_boundary_level,
                        prev_poly=prev_boundary_poly,
                        limiter_shape=env.cfg.limiter_shape,
                        boundary_mode=env.cfg.boundary_mode,
                        boundary_base_mode=env.cfg.boundary_base_mode,
                        level_smoothing_alpha=env.cfg.boundary_level_smoothing_alpha,
                        level_search_span_fraction=env.cfg.boundary_level_search_span_fraction,
                        continuity_weight_radii=env.cfg.boundary_continuity_weight_radii,
                        continuity_weight_mean_radius=env.cfg.boundary_continuity_weight_mean_radius,
                        continuity_weight_center=env.cfg.boundary_continuity_weight_center,
                        continuity_weight_area=env.cfg.boundary_continuity_weight_area,
                        continuity_weight_level=env.cfg.boundary_continuity_weight_level,
                    )
                    prev_boundary_poly = np.asarray(poly, dtype=float)
                    prev_boundary_level = float(level)
                    found = 1.0
                    radii = legacy_radii_at_angles(poly, center, angles)
                    shape = float(np.nanmean(np.abs(np.nan_to_num(radii) - ref_radii)))
                except BoundaryNotFoundError:
                    poly = None
                    prev_boundary_poly = None
                    prev_boundary_level = None
                    found = 0.0
                    shape = float(config.reward.boundary_missing_error_m)
                ip_err = abs(float(model.state.Ip) - ip_target)
                action = controller.compute_control(
                    model=model,
                    psi=psi,
                    boundary_poly=poly,
                    center=center,
                    measure_angles=angles,
                    ref_radii=ref_radii,
                    Ip_ref=ip_target,
                    scenario=scenario,
                    max_episode_steps=int(config.sim.max_episode_steps),
                )
                pfc_next = np.asarray(action.pfc_currents_next, dtype=float)
                sol_next = np.asarray(action.sol_currents_next, dtype=float)
                if not np.all(np.isfinite(pfc_next)) or not np.all(np.isfinite(sol_next)):
                    raise ValueError("controller emitted non-finite next-current commands")
                prev_currents = np.concatenate([
                    np.asarray(model.state.pfc_currents, dtype=float),
                    np.asarray(model.state.sol_currents, dtype=float),
                ])
                next_currents = np.concatenate([pfc_next, sol_next])
                deriv = (next_currents - prev_currents) / max(float(model.t_step), 1.0e-12)
                action_rms_all.append(float(np.sqrt(np.mean(np.square(deriv)))) if deriv.size else 0.0)
                model.step_currents(pfc_currents_next=pfc_next, sol_currents_next=sol_next)
                currents = np.concatenate([np.asarray(model.state.pfc_currents, dtype=float), np.asarray(model.state.sol_currents, dtype=float)])
                current_over = float(np.nanmax(np.maximum(np.abs(currents) - current_limits, 0.0)))
                current_usage = float(np.nanmax(np.abs(currents) / np.maximum(current_limits, 1.0e-12)))
                if first_failure_step is None and (found < 0.5 or current_over > 0.0):
                    first_failure_step = step_index + 1
                ep_boundary.append(found)
                ep_shape.append(shape)
                ep_ip.append(ip_err)
                ep_current.append(current_over)
                ep_current_usage.append(current_usage)
            if first_failure_step is None:
                first_failure_step = step_count
            completion.append(float(first_failure_step) / float(step_count))
            late_start = max(0, int(0.8 * step_count))
            boundary_found_all.extend(ep_boundary)
            shape_error_all.extend(ep_shape)
            ip_error_all.extend(ep_ip)
            current_over_all.extend(ep_current)
            current_usage_all.extend(ep_current_usage)
            boundary_late.extend(ep_boundary[late_start:])
            shape_late.extend(ep_shape[late_start:])
            ip_late.extend(ep_ip[late_start:])
            current_late.extend(ep_current[late_start:])
            current_usage_late.extend(ep_current_usage[late_start:])

        completion_arr = np.asarray(completion, dtype=float)
        return {
            "status": "ok",
            "export_dir": str(export_dir),
            "episodes": int(episode_count),
            "steps": int(step_count),
            "reference_kind": {
                "boundary": str(config.reference.boundary.kind),
                "ip": str(config.reference.ip.kind),
            },
            "mean_episode_completion": float(np.nanmean(completion_arr)),
            "min_episode_completion": float(np.nanmin(completion_arr)),
            "physical_derivative_rms": float(np.nanmean(np.asarray(action_rms_all, dtype=float))),
            "action_rms": float(np.nanmean(np.asarray(action_rms_all, dtype=float))),
            "boundary_found_mean": float(np.nanmean(np.asarray(boundary_found_all, dtype=float))),
            "boundary_found_late_min": float(np.nanmin(np.asarray(boundary_late, dtype=float))) if boundary_late else float("nan"),
            "shape_error_mean_m": float(np.nanmean(np.asarray(shape_error_all, dtype=float))),
            "shape_error_late_m": float(np.nanmean(np.asarray(shape_late, dtype=float))) if shape_late else float("nan"),
            "ip_error_a": float(np.nanmean(np.asarray(ip_error_all, dtype=float))),
            "ip_error_late_a": float(np.nanmean(np.asarray(ip_late, dtype=float))) if ip_late else float("nan"),
            "current_over_limit_a_max": float(np.nanmax(np.asarray(current_over_all, dtype=float))) if current_over_all else 0.0,
            "current_over_limit_a_late_max": float(np.nanmax(np.asarray(current_late, dtype=float))) if current_late else 0.0,
            "current_over_limit_fraction": float(np.nanmean(np.asarray(current_over_all, dtype=float) > 0.0)) if current_over_all else 0.0,
            "current_over_limit_fraction_late": float(np.nanmean(np.asarray(current_late, dtype=float) > 0.0)) if current_late else 0.0,
            "current_over_limit_5ka_fraction": float(np.nanmean(np.asarray(current_over_all, dtype=float) > 5000.0)) if current_over_all else 0.0,
            "current_over_limit_5ka_fraction_late": float(np.nanmean(np.asarray(current_late, dtype=float) > 5000.0)) if current_late else 0.0,
            "current_over_limit_1pct_fraction": float(np.nanmean(np.asarray(current_usage_all, dtype=float) > 1.01)) if current_usage_all else 0.0,
            "current_over_limit_1pct_fraction_late": float(np.nanmean(np.asarray(current_usage_late, dtype=float) > 1.01)) if current_usage_late else 0.0,
        }
    except Exception as exc:
        return {"status": "error", "export_dir": str(export_dir), "error": repr(exc)}


def _apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.sim_compute_backend is not None or args.sim_gpu_device is not None:
        cfg = replace(
            cfg,
            sim=replace(
                cfg.sim,
                compute_backend=args.sim_compute_backend if args.sim_compute_backend is not None else cfg.sim.compute_backend,
                gpu_device=args.sim_gpu_device if args.sim_gpu_device is not None else cfg.sim.gpu_device,
            ),
        )
    if any(v is not None for v in (args.batch_size, args.unroll_length, args.replay_capacity_episodes, args.min_replay_sequence_length, args.rollout_chunk_length, args.updates_per_rollout_chunk, args.action_samples, args.actor_update_chunk_size)):
        cfg = replace(
            cfg,
            learner=replace(
                cfg.learner,
                batch_size=args.batch_size if args.batch_size is not None else cfg.learner.batch_size,
                unroll_length=args.unroll_length if args.unroll_length is not None else cfg.learner.unroll_length,
                replay_capacity_episodes=args.replay_capacity_episodes if args.replay_capacity_episodes is not None else cfg.learner.replay_capacity_episodes,
                min_replay_sequence_length=args.min_replay_sequence_length if args.min_replay_sequence_length is not None else cfg.learner.min_replay_sequence_length,
                rollout_chunk_length=args.rollout_chunk_length if args.rollout_chunk_length is not None else cfg.learner.rollout_chunk_length,
                updates_per_rollout_chunk=args.updates_per_rollout_chunk if args.updates_per_rollout_chunk is not None else cfg.learner.updates_per_rollout_chunk,
                action_samples=args.action_samples if args.action_samples is not None else cfg.learner.action_samples,
                actor_update_chunk_size=args.actor_update_chunk_size if args.actor_update_chunk_size is not None else cfg.learner.actor_update_chunk_size,
            ),
        )
    if any(v is not None for v in (args.hidden_dim, args.critic_hidden_dim, args.critic_mlp_hidden_dim)):
        cfg = replace(
            cfg,
            network=replace(
                cfg.network,
                hidden_dim=args.hidden_dim if args.hidden_dim is not None else cfg.network.hidden_dim,
                critic_hidden_dim=args.critic_hidden_dim if args.critic_hidden_dim is not None else cfg.network.critic_hidden_dim,
                critic_mlp_hidden_dim=args.critic_mlp_hidden_dim if args.critic_mlp_hidden_dim is not None else cfg.network.critic_mlp_hidden_dim,
            ),
        )
    if any(v is not None for v in (args.steps, args.num_envs, args.device, args.output_dir, args.save_checkpoints, args.checkpoint_interval_steps, args.eval_interval_steps, args.eval_episodes, args.eval_max_steps, args.actor_workers, args.actor_devices, args.distributed_mode)):
        steps = int(args.steps) if args.steps is not None else int(cfg.training.steps)
        checkpoint_interval = args.checkpoint_interval_steps if args.checkpoint_interval_steps is not None else cfg.training.checkpoint_interval_steps
        eval_interval = args.eval_interval_steps if args.eval_interval_steps is not None else cfg.training.eval_interval_steps
        checkpoint_interval = min(int(checkpoint_interval), max(1, steps))
        eval_interval = min(int(eval_interval), max(1, steps))
        cfg = replace(
            cfg,
            training=replace(
                cfg.training,
                steps=steps,
                num_envs=args.num_envs if args.num_envs is not None else cfg.training.num_envs,
                device=args.device if args.device is not None else cfg.training.device,
                output_dir=Path(args.output_dir).resolve() if args.output_dir is not None else cfg.training.output_dir,
                save_checkpoints=args.save_checkpoints if args.save_checkpoints is not None else cfg.training.save_checkpoints,
                checkpoint_interval_steps=checkpoint_interval,
                eval_interval_steps=eval_interval,
                eval_episodes=args.eval_episodes if args.eval_episodes is not None else cfg.training.eval_episodes,
                eval_max_steps=args.eval_max_steps if args.eval_max_steps is not None else cfg.training.eval_max_steps,
                actor_workers=args.actor_workers if args.actor_workers is not None else cfg.training.actor_workers,
                actor_devices=_device_list(args.actor_devices) if args.actor_devices is not None else cfg.training.actor_devices,
                distributed_mode=args.distributed_mode if args.distributed_mode is not None else cfg.training.distributed_mode,
            ),
        )
    return _cap_training_intervals(cfg)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Gated training pipeline for the first deployable T15 RL controller.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume-checkpoint", default=None)
    ap.add_argument("--sim-compute-backend", choices=("cpu", "gpu"), default=None)
    ap.add_argument("--sim-gpu-device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--unroll-length", type=int, default=None)
    ap.add_argument("--replay-capacity-episodes", type=int, default=None)
    ap.add_argument("--min-replay-sequence-length", type=int, default=None)
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
    ap.add_argument("--eval-seed-offset", type=int, default=100000)
    ap.add_argument("--holdout-eval-seed-offset", type=int, default=200000)
    ap.add_argument("--actor-workers", type=int, default=None)
    ap.add_argument("--actor-devices", default=None)
    ap.add_argument("--distributed-mode", choices=("single", "local_replay"), default=None)
    ap.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--reset-error-tolerance-m", type=float, default=1.0e-6)
    ap.add_argument("--min-boundary-found", type=float, default=0.999)
    ap.add_argument("--max-current-over-limit-a", type=float, default=0.0)
    ap.add_argument("--max-shape-error-m", type=float, default=0.03)
    ap.add_argument("--min-ip-improvement-frac", type=float, default=0.25)
    ap.add_argument("--min-ip-improvement-a", type=float, default=20000.0)
    ap.add_argument("--min-action-rms", type=float, default=0.005)
    ap.add_argument("--max-action-rms", type=float, default=0.5)
    ap.add_argument("--min-episode-completion", type=float, default=0.95)
    ap.add_argument("--max-controller-shape-error-m", type=float, default=0.03)
    ap.add_argument("--max-controller-ip-error-a", type=float, default=40000.0)
    ap.add_argument("--min-policy-weight-extra", type=float, default=1.0e-4)
    ap.add_argument("--min-sampled-q-spread", type=float, default=1.0e-8)
    ap.add_argument("--controller-rollout-steps", type=int, default=0)
    ap.add_argument("--skip-controller-rollout-gate", action="store_true")
    ap.add_argument("--allow-failed-gates", action="store_true")
    ap.add_argument("--reward-sweep-mode", action="store_true")
    ap.add_argument("--no-export", action="store_true", help="Disable trainer actor exports. Used for reward sweeps.")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="tokamak-rl-v2-policy")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    ap.add_argument("--wandb-optional", action="store_true", help="Continue if W&B init fails.")
    ap.add_argument("--wandb-metric-preset", choices=("full", "focused", "sweep"), default="full")
    return ap


def _cap_training_intervals(cfg: ExperimentConfig) -> ExperimentConfig:
    steps = int(cfg.training.steps)
    if steps <= 0:
        return cfg
    checkpoint_interval = min(int(cfg.training.checkpoint_interval_steps), steps)
    eval_interval = min(int(cfg.training.eval_interval_steps), steps)
    if checkpoint_interval == int(cfg.training.checkpoint_interval_steps) and eval_interval == int(cfg.training.eval_interval_steps):
        return cfg
    return replace(cfg, training=replace(cfg.training, checkpoint_interval_steps=checkpoint_interval, eval_interval_steps=eval_interval))


def _gate_profile_for_config(cfg: ExperimentConfig) -> dict[str, float | bool | None]:
    is_production_csv = bool(cfg.training.production_mode) and str(cfg.sim.reset_source) == "csv_initial_states" and str(cfg.reference.ip.kind) == "segmented_profile"
    if is_production_csv:
        return {
            "enforce_baseline_difficulty": False,
            "min_baseline_ip_error_a": 0.0,
            "min_baseline_ip_error_late_a": 0.0,
            "always_require_ip_improvement": False,
            "max_ip_error_a": 25000.0,
            "max_ip_error_late_a": 25000.0,
            "min_mean_episode_completion": 0.95,
            "min_episode_completion": 0.90,
            "max_controller_ip_error_a": 25000.0,
        }
    return {
        "enforce_baseline_difficulty": False,
        "min_baseline_ip_error_a": 0.0,
        "min_baseline_ip_error_late_a": 0.0,
        "always_require_ip_improvement": False,
        "max_ip_error_a": None,
        "max_ip_error_late_a": None,
        "min_mean_episode_completion": 0.95,
        "min_episode_completion": 0.95,
        "max_controller_ip_error_a": 40000.0,
    }


def _baseline_difficulty_report(
    metrics: Mapping[str, object],
    *,
    min_ip_error_a: float,
    min_ip_error_late_a: float,
) -> dict[str, float]:
    ip_error_a = _metric(metrics, "ip_error_a")
    ip_error_a_late = _metric(metrics, "ip_error_a_late", default=ip_error_a)
    passed = (
        _finite(ip_error_a)
        and ip_error_a >= float(min_ip_error_a)
        and _finite(ip_error_a_late)
        and ip_error_a_late >= float(min_ip_error_late_a)
    )
    return {
        "passed": 1.0 if passed else 0.0,
        "ip_error_a": ip_error_a,
        "ip_error_a_late": ip_error_a_late,
        "min_ip_error_a": float(min_ip_error_a),
        "min_ip_error_a_late": float(min_ip_error_late_a),
    }


def _preflight_artifact_failure(cfg: ExperimentConfig) -> dict[str, object] | None:
    if str(cfg.sim.reset_source) == "csv_initial_states":
        path = cfg.sim.csv_initial_state_library
        if path is None or not Path(path).exists():
            return {"status": "failed_initial_state_library", "name": "initial_state_library", "path": "" if path is None else str(path)}
        summary_path = Path(path).with_suffix(".json")
        try:
            _validate_initial_state_summary(summary_path)
            from tokamak_rl_v2.env.t15_csv_initial_states import CsvInitialStateLibrary

            with np.load(path, allow_pickle=False) as data:
                expected_arrays = {"shot_id", "source_index", "time_s", "ip0", "pfc0", "sol0", "split"}
                if set(data.files) != expected_arrays:
                    raise ValueError(f"initial-state library arrays must be exactly {sorted(expected_arrays)}, got {sorted(data.files)}")
                n_pfc = int(np.asarray(data["pfc0"]).shape[1])
                n_sol = int(np.asarray(data["sol0"]).shape[1])
                ip0 = np.asarray(data["ip0"], dtype=float).reshape(-1)
            CsvInitialStateLibrary(path, n_pfc=n_pfc, n_sol=n_sol, split="train")
            CsvInitialStateLibrary(path, n_pfc=n_pfc, n_sol=n_sol, split="holdout")
        except Exception as exc:
            return {"status": "failed_initial_state_library", "name": "initial_state_library", "path": str(path), "reason": repr(exc)}
    else:
        ip0 = np.asarray([], dtype=float)
    if str(cfg.reference.ip.kind) == "segmented_profile":
        path = cfg.reference.ip.limits_path
        if path is None or not Path(path).exists():
            return {"status": "failed_reference_limits", "name": "reference_limits", "path": "" if path is None else str(path)}
        try:
            _validate_reference_limits_summary(Path(path))
            from tokamak_rl_v2.env.t15_reference_limits import load_reference_limits

            limits = load_reference_limits(Path(path))
            if str(cfg.reference.ip.ramp_rate_reference) == "robust_mean" and (
                limits.positive_ramp_mean_a_per_s is None or limits.negative_ramp_abs_mean_a_per_s is None
            ):
                raise ValueError("reference limits missing robust ramp mean fields; rebuild t15_reference_limits.json")
            if ip0.size and (float(np.nanmin(ip0)) < float(limits.ip_p01_a) or float(np.nanmax(ip0)) > float(limits.ip_p99_a)):
                raise ValueError("initial-state library contains reset Ip outside production reference bounds")
        except Exception as exc:
            return {"status": "failed_reference_limits", "name": "reference_limits", "path": str(path), "reason": repr(exc)}
    return None


def _validate_initial_state_summary(path: Path) -> None:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("initial-state summary must be a JSON object")
    required = {
        "accepted_rows": 1000,
        "train_rows": 1000,
        "holdout_rows": 100,
    }
    for key, minimum in required.items():
        value = int(raw.get(key, 0))
        if value < minimum:
            raise ValueError(f"{key} must be >= {minimum}, got {value}")
    split_by_shot = raw.get("split_by_shot", {})
    accepted_by_shot = raw.get("accepted_by_shot", {})
    if not isinstance(split_by_shot, dict) or not isinstance(accepted_by_shot, dict) or not split_by_shot:
        raise ValueError("initial-state summary must contain accepted_by_shot and split_by_shot")
    split_policy = str(raw.get("split_policy", ""))
    if split_policy in {"explicit_whole_shot", "whole_shot"}:
        train_shots = []
        holdout_shots = []
        for shot, accepted in accepted_by_shot.items():
            counts = split_by_shot.get(str(shot), {})
            train = int(counts.get("train", 0))
            holdout = int(counts.get("holdout", 0))
            if int(accepted) < 100:
                raise ValueError(f"shot {shot} has too few accepted rows: {accepted}")
            if train > 0 and holdout > 0:
                raise ValueError(f"whole-shot split requires shot {shot} to be in exactly one split, got train={train} holdout={holdout}")
            if train > 0:
                train_shots.append(str(shot))
            if holdout > 0:
                holdout_shots.append(str(shot))
        if not train_shots or not holdout_shots:
            raise ValueError("whole-shot split requires at least one train shot and one holdout shot")
        return
    for shot, accepted in accepted_by_shot.items():
        counts = split_by_shot.get(str(shot), {})
        if int(accepted) < 100 or int(counts.get("train", 0)) < 80 or int(counts.get("holdout", 0)) < 10:
            raise ValueError(f"shot {shot} does not satisfy accepted/train/holdout split gates")


def _validate_reference_limits_summary(path: Path) -> None:
    from tokamak_rl_v2.env.t15_reference_limits import load_reference_limits

    load_reference_limits.cache_clear()
    limits = load_reference_limits(path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "ip_min_a",
        "ip_max_a",
        "ip_p01_a",
        "ip_p99_a",
        "positive_dipdt_p95_a_per_s",
        "positive_dipdt_p99_a_per_s",
        "negative_dipdt_abs_p95_a_per_s",
        "negative_dipdt_abs_p99_a_per_s",
        "shot_ids",
        "sample_count",
    ):
        if key not in raw:
            raise ValueError(f"reference limits missing {key}")
    if limits.sample_count < 1000:
        raise ValueError(f"reference limits sample_count must be >= 1000, got {limits.sample_count}")


def _selected_checkpoint(output_dir: Path) -> Path | None:
    best = output_dir / "checkpoints" / "best.pt"
    if best.exists():
        return best
    final = output_dir / "checkpoints" / "final.pt"
    return final if final.exists() else None


def _selected_export_dir(output_dir: Path) -> Path | None:
    best = output_dir / "exports" / "best_actor"
    if best.is_dir():
        return best
    final = output_dir / "exports" / "final_actor"
    return final if final.is_dir() else None


def _load_actor_weights(trainer: Trainer, checkpoint: Path) -> None:
    data = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    if not isinstance(data, dict) or "actor_state_dict" not in data:
        raise ValueError(f"checkpoint does not contain actor_state_dict: {checkpoint}")
    trainer.actor.load_state_dict(data["actor_state_dict"])


def _metric(metrics: Mapping[str, float], key: str, default: float = float("nan")) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return default


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(value)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        if dev.index is None:
            return torch.device("cuda:0")
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is not visible: {value}; visible device count is {torch.cuda.device_count()}")
    return dev


def _distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _distributed_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", str(_distributed_rank())))


def _rank_runtime_device(value: str | None, cfg: ExperimentConfig) -> str:
    raw = str(value or cfg.training.device)
    if cfg.training.distributed_mode != "local_replay":
        return raw
    if raw == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{_distributed_local_rank()}"


def _rank_runtime_config(cfg: ExperimentConfig, runtime_device: str) -> ExperimentConfig:
    if cfg.training.distributed_mode != "local_replay" or cfg.sim.compute_backend != "gpu":
        return cfg
    return replace(cfg, sim=replace(cfg.sim, gpu_device=runtime_device))


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(data)), indent=2), encoding="utf-8")


def _start_wandb(args: argparse.Namespace, cfg: ExperimentConfig, *, output_dir: Path):
    if not bool(args.wandb) or args.wandb_mode == "disabled":
        return None
    if cfg.training.distributed_mode == "local_replay" and _distributed_rank() != 0:
        return None
    try:
        import wandb

        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or cfg.name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            config={
                "experiment": cfg.name,
                "policy_pipeline": str(cfg.reference.boundary.kind),
                "reference_boundary_kind": str(cfg.reference.boundary.kind),
                "reference_ip_kind": str(cfg.reference.ip.kind),
                "sim_action_contract": str(cfg.sim.action_contract),
                "sim_compute_backend": str(cfg.sim.compute_backend),
                "distributed_mode": str(cfg.training.distributed_mode),
                "output_dir": str(output_dir),
                "eval_seed_offset": int(args.eval_seed_offset),
                "holdout_eval_seed_offset": int(args.holdout_eval_seed_offset),
                "wandb_metric_preset": str(args.wandb_metric_preset),
            },
        )
    except Exception as exc:
        if bool(getattr(args, "wandb_optional", False)):
            print(f"warning: W&B init failed and --wandb-optional is set; continuing without W&B: {exc}", file=sys.stderr)
            return None
        raise


def _wandb_log(wandb_run, prefix: str, metrics: Mapping[str, object], *, step: int) -> None:
    if wandb_run is None:
        return
    payload: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            payload[f"{prefix}/{key}"] = numeric
    preset = "full"
    try:
        preset = str(wandb_run.config.get("wandb_metric_preset", "full"))
    except Exception:
        preset = "full"
    payload = filter_wandb_metrics(payload, preset)
    if payload:
        try:
            wandb_run.log({"global_step": int(step), **payload}, step=int(step))
        except Exception as exc:
            print(f"warning: W&B log failed; continuing with disk outputs: {exc}", file=sys.stderr)


def _train_env_step(train_result: Mapping[str, object], cfg: ExperimentConfig) -> int:
    raw = train_result.get("env_steps", None)
    if raw is not None:
        return int(raw)
    steps = int(train_result.get("steps", cfg.training.steps))
    if int(cfg.training.actor_workers) > 1:
        return steps
    return steps * int(cfg.training.num_envs)


def _gate_metrics(checks: object) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(checks, list):
        return out
    passed = 0
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        name = str(check.get("name", "unknown"))
        ok = 1.0 if bool(check.get("passed", False)) else 0.0
        out[f"{name}_passed"] = ok
        passed += int(ok)
    if checks:
        out["passed_fraction"] = float(passed) / float(len(checks))
        out["passed"] = 1.0 if passed == len(checks) else 0.0
    return out


if __name__ == "__main__":
    raise SystemExit(main())
