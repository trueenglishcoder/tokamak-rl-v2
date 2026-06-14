from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.config.loader import _validate_experiment_config
from tokamak_rl_v2.training.hinf_bootstrap import HinfBootstrapConfig, run_hinf_bootstrap
from tokamak_rl_v2.training.policy_pipeline import (
    _apply_overrides,
    _gate_metrics,
    _selected_export_dir,
    _start_wandb,
    _train_env_step,
    _wandb_log,
    _write_json,
    evaluate_policy_gates,
    run_reset_sanity,
    summarize_training_losses,
    validate_exported_controller,
)
from tokamak_rl_v2.training.trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.eval_seed_offset) == int(args.holdout_eval_seed_offset):
        raise ValueError("holdout_eval_seed_offset must differ from eval_seed_offset")

    cfg = _apply_overrides(load_experiment_config(args.config), args)
    if args.actor_initial_std is not None:
        cfg = replace(cfg, network=replace(cfg.network, actor_initial_std=float(args.actor_initial_std)))
    _validate_experiment_config(cfg)

    output_dir = Path(args.output_dir or cfg.training.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _start_wandb(args, cfg, output_dir=output_dir)
    try:
        reset_report = run_reset_sanity(cfg, device=args.device, num_envs=args.num_envs)
        _wandb_log(wandb_run, "pipeline/reset", reset_report, step=0)
        reset_gate = reset_report["max_abs_boundary_radii_error_m"] <= float(args.reset_error_tolerance_m) and reset_report["boundary_found_mean"] >= float(args.min_boundary_found)
        if not reset_gate:
            report = {"status": "failed_reset_sanity", "reset_sanity": reset_report, "gates": [{"name": "reset_sanity", "passed": False}]}
            _write_json(output_dir / "policy_validation.json", report)
            return 2 if not args.allow_failed_gates else 0

        trainer = Trainer(cfg, steps=args.steps, num_envs=args.num_envs, device=args.device, output_dir=output_dir, wandb_run=wandb_run, resume_checkpoint=args.resume_checkpoint)
        selection_seed_offset = int(args.eval_seed_offset)
        holdout_seed_offset = int(args.holdout_eval_seed_offset)
        baseline = trainer.evaluate_detailed(episodes=int(cfg.training.eval_episodes), max_steps=int(cfg.training.eval_max_steps), policy="no_control", seed_offset=selection_seed_offset)
        holdout_baseline = trainer.evaluate_detailed(episodes=int(cfg.training.eval_episodes), max_steps=int(cfg.training.eval_max_steps), policy="no_control", seed_offset=holdout_seed_offset)
        _wandb_log(wandb_run, "pipeline/no_control_selection", baseline, step=0)
        _wandb_log(wandb_run, "pipeline/no_control_holdout", holdout_baseline, step=0)

        bootstrap_cfg = HinfBootstrapConfig(
            steps=int(args.bootstrap_steps),
            lr=float(args.bootstrap_lr),
            q_error=float(args.hinf_q_error),
            q_ip=float(args.hinf_q_ip),
            gamma=float(args.hinf_gamma),
            u_clip=float(args.hinf_u_clip),
            j_curr=float(args.hinf_j_curr),
            target_std=float(args.bootstrap_target_std),
            std_loss_weight=float(args.bootstrap_std_loss_weight),
            log_interval=int(args.bootstrap_log_interval),
            reset_interval_steps=int(args.bootstrap_reset_interval_steps),
        )
        bootstrap_report = run_hinf_bootstrap(trainer, config=cfg, bootstrap=bootstrap_cfg, output_dir=output_dir, wandb_run=wandb_run)
        _write_json(output_dir / "hinf_bootstrap_summary.json", bootstrap_report)

        bootstrap_eval = trainer.evaluate_detailed(episodes=int(cfg.training.eval_episodes), max_steps=int(cfg.training.eval_max_steps), policy="actor", seed_offset=holdout_seed_offset)
        bootstrap_score = trainer._selection_score(bootstrap_eval)
        bootstrap_eval["selection_score"] = bootstrap_score
        trainer.best_eval = bootstrap_score
        trainer.best_eval_details = dict(bootstrap_eval)
        trainer._remember_best_actor()
        trainer._export("exports/bootstrap_actor", step=0, updates=0, eval_score=bootstrap_score)
        trainer._export("exports/best_actor", step=0, updates=0, eval_score=bootstrap_score)
        _wandb_log(wandb_run, "pipeline/bootstrap_actor_eval_holdout", bootstrap_eval, step=int(bootstrap_report.get("env_steps", 0)))

        train_result = trainer.train()

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
            "evaluation_seed_offsets": {"selection": selection_seed_offset, "holdout": holdout_seed_offset},
            "export_dir": None if selected_export is None else str(selected_export),
            "reset_sanity": reset_report,
            "no_control_selection": baseline,
            "no_control": holdout_baseline,
            "hinf_bootstrap": bootstrap_report,
            "bootstrap_actor_eval": bootstrap_eval,
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
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a T15 policy with H∞ behavior-cloning bootstrap before MPO.")
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
    ap.add_argument("--actor-initial-std", type=float, default=None)
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
    ap.add_argument("--bootstrap-steps", type=int, default=5000)
    ap.add_argument("--bootstrap-lr", type=float, default=1.0e-4)
    ap.add_argument("--bootstrap-target-std", type=float, default=0.08)
    ap.add_argument("--bootstrap-std-loss-weight", type=float, default=0.05)
    ap.add_argument("--bootstrap-log-interval", type=int, default=100)
    ap.add_argument("--bootstrap-reset-interval-steps", type=int, default=250)
    ap.add_argument("--hinf-q-error", type=float, default=3.0e6)
    ap.add_argument("--hinf-q-ip", type=float, default=1.0e7)
    ap.add_argument("--hinf-gamma", type=float, default=7905.694150420949)
    ap.add_argument("--hinf-u-clip", type=float, default=2.0e6)
    ap.add_argument("--hinf-j-curr", type=float, default=0.0)
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
    ap.add_argument("--wandb-project", default="tokamak-rl-v2-hinf-bootstrap")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return ap


if __name__ == "__main__":
    raise SystemExit(main())
