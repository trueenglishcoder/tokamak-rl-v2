from __future__ import annotations

import argparse
import csv
import json
import math
import signal
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
from tokamak_rl_v2.training.trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.eval_seed_offset) < 0 or int(args.holdout_eval_seed_offset) < 0:
        raise ValueError("evaluation seed offsets must be non-negative")
    if int(args.eval_seed_offset) == int(args.holdout_eval_seed_offset):
        raise ValueError("holdout_eval_seed_offset must differ from eval_seed_offset")
    cfg = _apply_overrides(load_experiment_config(args.config), args)
    _validate_experiment_config(cfg)
    output_dir = Path(args.output_dir or cfg.training.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _start_wandb(args, cfg, output_dir=output_dir)
    previous_signal_handlers = _install_shutdown_signal_handlers()
    try:
        reset_report = run_reset_sanity(cfg, device=args.device, num_envs=args.num_envs)
        _wandb_log(wandb_run, "pipeline/reset", reset_report, step=0)
        reset_gate = reset_report["max_abs_boundary_radii_error_m"] <= float(args.reset_error_tolerance_m) and reset_report["boundary_found_mean"] >= float(args.min_boundary_found)
        if not reset_gate:
            report = {
                "status": "failed_reset_sanity",
                "reset_sanity": reset_report,
                "gates": [{"name": "reset_sanity", "passed": False}],
            }
            _write_json(output_dir / "policy_validation.json", report)
            _wandb_log(wandb_run, "pipeline", {"passed": 0.0, "failed_reset_sanity": 1.0}, step=0)
            return 2 if not args.allow_failed_gates else 0

        trainer = Trainer(cfg, steps=args.steps, num_envs=args.num_envs, device=args.device, output_dir=output_dir, wandb_run=wandb_run, resume_checkpoint=args.resume_checkpoint)
        selection_seed_offset = int(args.eval_seed_offset)
        holdout_seed_offset = int(args.holdout_eval_seed_offset)
        baseline = trainer.evaluate_detailed(episodes=int(cfg.training.eval_episodes), max_steps=int(cfg.training.eval_max_steps), policy="no_control", seed_offset=selection_seed_offset)
        holdout_baseline = trainer.evaluate_detailed(episodes=int(cfg.training.eval_episodes), max_steps=int(cfg.training.eval_max_steps), policy="no_control", seed_offset=holdout_seed_offset)
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
        train_result = trainer.train()

        selected_checkpoint = _selected_checkpoint(output_dir)
        if selected_checkpoint is not None and selected_checkpoint.name == "best.pt":
            _load_actor_weights(trainer, selected_checkpoint)
        elif selected_checkpoint is None:
            trainer.restore_best_actor()
        actor_eval = trainer.evaluate_detailed(episodes=int(cfg.training.eval_episodes), max_steps=int(cfg.training.eval_max_steps), policy="actor", seed_offset=holdout_seed_offset)
        train_env_step = _train_env_step(train_result, cfg)
        _wandb_log(wandb_run, "pipeline/actor_eval_holdout", actor_eval, step=train_env_step)
        losses = summarize_training_losses(output_dir / "losses.csv")
        selected_export = _selected_export_dir(output_dir)
        rollout_report = validate_exported_controller(selected_export, cfg, steps=int(args.controller_rollout_steps)) if selected_export is not None else {"status": "missing_export"}

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
            min_action_rms=float(args.min_action_rms),
            max_action_rms=float(args.max_action_rms),
            min_policy_weight_extra=float(args.min_policy_weight_extra),
            min_sampled_q_spread=float(args.min_sampled_q_spread),
            require_controller_rollout=not bool(args.skip_controller_rollout_gate),
            controller_rollout=rollout_report,
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
    min_action_rms: float,
    max_action_rms: float,
    min_policy_weight_extra: float,
    min_sampled_q_spread: float,
    require_controller_rollout: bool,
    controller_rollout: Mapping[str, object],
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
    ip_improvement = baseline_ip - actor_ip if _finite(baseline_ip) and _finite(actor_ip) else float("nan")
    ip_improvement_frac = ip_improvement / max(abs(baseline_ip), 1.0e-12) if _finite(ip_improvement) and _finite(baseline_ip) else float("nan")
    if _finite(baseline_ip) and baseline_ip >= min_ip_improvement_a:
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

    baseline_ip_late = _metric(no_control, "ip_error_a_late", default=baseline_ip)
    actor_ip_late = _metric(actor_eval, "ip_error_a_late", default=actor_ip)
    ip_late_improvement = baseline_ip_late - actor_ip_late if _finite(baseline_ip_late) and _finite(actor_ip_late) else float("nan")
    ip_late_improvement_frac = ip_late_improvement / max(abs(baseline_ip_late), 1.0e-12) if _finite(ip_late_improvement) and _finite(baseline_ip_late) else float("nan")
    if _finite(baseline_ip_late) and baseline_ip_late >= min_ip_improvement_a:
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

    policy_weight_max = _metric(tail_losses, "tail100.policy_weight_max")
    uniform = 1.0 / max(int(action_samples), 1)
    policy_weight_threshold = uniform + float(min_policy_weight_extra)
    add("mpo_policy_weights_nonuniform", _finite(policy_weight_max) and policy_weight_max > policy_weight_threshold, value=policy_weight_max, threshold=f"> {policy_weight_threshold:g}")

    q_spread = _metric(tail_losses, "tail100.sampled_q_spread")
    add("mpo_sampled_q_spread", _finite(q_spread) and q_spread >= min_sampled_q_spread, value=q_spread, threshold=f">= {min_sampled_q_spread:g}")

    if require_controller_rollout:
        status = str(controller_rollout.get("status", "missing"))
        add("controller_rollout", status == "ok", value=status, threshold="ok")

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
        from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections

        rollout_config = replace(config, sim=replace(config.sim, compute_backend="cpu", gpu_device="cuda:0"))
        env = TokamakMagneticControlEnv(rollout_config, batch_size=1, device="cpu", seed=int(config.training.seed) + 910000)
        env.reset()
        if env.reference is None:
            raise RuntimeError("controller validation environment did not create references")
        model = env._cpu_models[0]
        controller = LearnedMagneticController(export_dir=export_dir)
        angles = np.asarray(env.angles, dtype=float)
        center = (model.R0, model.Z0)
        ref_ip = env.reference.ip[0].detach().cpu().numpy().astype(float)
        ref_radii_series = env.reference.radii[0, :, : int(config.sim.angles)].detach().cpu().numpy().astype(float)
        scenario = _ArrayReferenceScenario(
            ip=ref_ip,
            radii=ref_radii_series,
            dt=float(model.t_step),
        )

        action_rms: list[float] = []
        boundary_found: list[float] = []
        shape_error: list[float] = []
        ip_error: list[float] = []
        current_over: list[float] = []
        current_limits = env.current_limits.detach().cpu().numpy().astype(float)
        for _step in range(max(1, int(steps))):
            ref_index = min(int(env.step_index[0].item()), ref_ip.shape[0] - 1)
            ref_radii = np.nan_to_num(ref_radii_series[ref_index], nan=0.0, posinf=0.0, neginf=0.0)
            ip_target = float(ref_ip[ref_index])
            psi = model.compute_psi()
            try:
                poly, _level, _status = find_plasma_boundary_with_status(psi, model.grid, center, n_levels=80, limiter_shape=env.cfg.limiter_shape, boundary_mode=env.cfg.boundary_mode)
                found = 1.0
                radii = radii_from_polyline_ray_intersections(poly, center, angles)
                shape_error.append(float(np.nanmean(np.abs(np.nan_to_num(radii) - ref_radii))))
            except BoundaryNotFoundError:
                poly = None
                found = 0.0
                shape_error.append(float(config.reward.boundary_missing_error_m))
            ip_error.append(abs(float(model.state.Ip) - ip_target))
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
            pfc_derivs = np.asarray(action.pfc_derivs, dtype=float)
            sol_derivs = np.asarray(action.sol_derivs, dtype=float)
            if not np.all(np.isfinite(pfc_derivs)) or not np.all(np.isfinite(sol_derivs)):
                raise ValueError("controller emitted non-finite derivatives")
            deriv = np.concatenate([pfc_derivs, sol_derivs])
            action_rms.append(float(np.sqrt(np.mean(np.square(deriv)))) if deriv.size else 0.0)
            boundary_found.append(found)
            model.step(pfc_derivs, sol_derivs)
            env.step_index += 1
            currents = np.concatenate([np.asarray(model.state.pfc_currents, dtype=float), np.asarray(model.state.sol_currents, dtype=float)])
            current_over.append(float(np.nanmax(np.maximum(np.abs(currents) - current_limits, 0.0))))
        return {
            "status": "ok",
            "export_dir": str(export_dir),
            "steps": int(max(1, int(steps))),
            "reference_kind": {
                "boundary": str(config.reference.boundary.kind),
                "ip": str(config.reference.ip.kind),
            },
            "physical_derivative_rms": float(np.nanmean(np.asarray(action_rms, dtype=float))),
            "boundary_found_mean": float(np.nanmean(np.asarray(boundary_found, dtype=float))),
            "shape_error_mean_m": float(np.nanmean(np.asarray(shape_error, dtype=float))),
            "shape_error_late_m": float(np.nanmean(np.asarray(shape_error[-max(1, len(shape_error) // 5) :], dtype=float))),
            "ip_error_a": float(np.nanmean(np.asarray(ip_error, dtype=float))),
            "ip_error_late_a": float(np.nanmean(np.asarray(ip_error[-max(1, len(ip_error) // 5) :], dtype=float))),
            "current_over_limit_a_max": float(np.nanmax(np.asarray(current_over, dtype=float))) if current_over else 0.0,
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
    if any(v is not None for v in (args.batch_size, args.unroll_length, args.replay_capacity_episodes, args.rollout_chunk_length, args.updates_per_rollout_chunk, args.action_samples, args.actor_update_chunk_size)):
        cfg = replace(
            cfg,
            learner=replace(
                cfg.learner,
                batch_size=args.batch_size if args.batch_size is not None else cfg.learner.batch_size,
                unroll_length=args.unroll_length if args.unroll_length is not None else cfg.learner.unroll_length,
                replay_capacity_episodes=args.replay_capacity_episodes if args.replay_capacity_episodes is not None else cfg.learner.replay_capacity_episodes,
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
    if any(v is not None for v in (args.steps, args.num_envs, args.device, args.output_dir, args.save_checkpoints, args.checkpoint_interval_steps, args.eval_interval_steps, args.eval_episodes, args.eval_max_steps, args.actor_workers, args.actor_devices)):
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
    ap.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--reset-error-tolerance-m", type=float, default=1.0e-6)
    ap.add_argument("--min-boundary-found", type=float, default=0.999)
    ap.add_argument("--max-current-over-limit-a", type=float, default=0.0)
    ap.add_argument("--max-shape-error-m", type=float, default=0.03)
    ap.add_argument("--min-ip-improvement-frac", type=float, default=0.25)
    ap.add_argument("--min-ip-improvement-a", type=float, default=20000.0)
    ap.add_argument("--min-action-rms", type=float, default=0.005)
    ap.add_argument("--max-action-rms", type=float, default=0.5)
    ap.add_argument("--min-policy-weight-extra", type=float, default=1.0e-4)
    ap.add_argument("--min-sampled-q-spread", type=float, default=1.0e-8)
    ap.add_argument("--controller-rollout-steps", type=int, default=8)
    ap.add_argument("--skip-controller-rollout-gate", action="store_true")
    ap.add_argument("--allow-failed-gates", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="tokamak-rl-v2-policy")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
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
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is not visible: {value}; visible device count is {torch.cuda.device_count()}")
    return dev


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
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or cfg.name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        config={
            "experiment": cfg.name,
            "policy_pipeline": "hold_reset_boundary",
            "output_dir": str(output_dir),
            "eval_seed_offset": int(args.eval_seed_offset),
            "holdout_eval_seed_offset": int(args.holdout_eval_seed_offset),
        },
    )


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
    if payload:
        wandb_run.log({"global_step": int(step), **payload}, step=int(step))


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
