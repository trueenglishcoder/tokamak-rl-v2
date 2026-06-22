from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tokamak_control.control.learned_magnetic_controller import LearnedMagneticController
from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections

from tokamak_rl_v2.config.loader import load_experiment_config
from tokamak_rl_v2.env.batch_env import TokamakMagneticControlEnv
from tokamak_rl_v2.export.cli import _validate_export_checkpoint
from tokamak_rl_v2.export.policy import export_deterministic_actor
from tokamak_rl_v2.networks import FeedForwardGaussianActor


class ArrayReferenceScenario:
    def __init__(self, *, ip: np.ndarray, radii: np.ndarray, dt: float) -> None:
        self.ip = np.asarray(ip, dtype=float).reshape(-1)
        self.radii = np.asarray(radii, dtype=float)
        self.dt = max(float(dt), 1.0e-12)
        if self.radii.ndim != 2 or self.radii.shape[0] != self.ip.shape[0]:
            raise ValueError("reference radii must have shape [steps, angles] aligned with Ip")

    def _index(self, t: float) -> int:
        idx = int(round(float(t) / self.dt))
        return int(np.clip(idx, 0, self.ip.shape[0] - 1))

    def Ip_ref(self, t: float) -> float:
        return float(self.ip[self._index(t)])

    def ref_radii(self, angles: np.ndarray, t: float) -> np.ndarray:
        ref = self.radii[self._index(t)]
        if ref.shape != np.asarray(angles).reshape(-1).shape:
            raise ValueError("reference angle count does not match controller request")
        return np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    export_dir = _ensure_export_dir(args=args, out_dir=out_dir)
    cfg = load_experiment_config(args.config)
    cfg = replace(
        cfg,
        sim=replace(cfg.sim, compute_backend="cpu", gpu_device="cuda:0", csv_initial_state_split=args.split),
        training=replace(cfg.training, eval_episodes=int(args.episodes)),
    )
    steps = int(args.steps) if int(args.steps) > 0 else int(cfg.sim.max_episode_steps)

    controller = LearnedMagneticController(export_dir=export_dir)
    episode_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    boundary_snapshots: list[dict[str, Any]] = []

    for episode in range(int(args.episodes)):
        env = TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=int(args.seed) + episode)
        controller.reset()
        env.reset()
        if env.reference is None:
            raise RuntimeError("environment did not create a reference after reset")
        model = env._cpu_models[0]
        angles = np.asarray(env.angles, dtype=float)
        center = (float(model.R0), float(model.Z0))
        current_limits = env.current_limits.detach().cpu().numpy().astype(float)
        derivative_limits = controller.derivative_scale.astype(float)
        ref_ip = env.reference.ip[0].detach().cpu().numpy().astype(float)
        ref_radii_series = env.reference.radii[0, :, : int(cfg.sim.angles)].detach().cpu().numpy().astype(float)
        scenario = ArrayReferenceScenario(ip=ref_ip, radii=ref_radii_series, dt=float(model.t_step))
        reset_meta = env.reset_metadata[0] if env.reset_metadata else {}
        terminated_step: int | None = None

        for step in range(steps):
            ref_index = min(step, ref_ip.shape[0] - 1)
            psi = model.compute_psi()
            boundary_poly, boundary_found, measured_radii = _boundary(model=model, psi=psi, center=center, angles=angles, env=env)
            ref_radii = np.nan_to_num(ref_radii_series[ref_index], nan=0.0, posinf=0.0, neginf=0.0)
            action = controller.compute_control(
                model=model,
                psi=psi,
                boundary_poly=boundary_poly,
                center=center,
                measure_angles=angles,
                ref_radii=ref_radii,
                Ip_ref=float(ref_ip[ref_index]),
                scenario=scenario,
                max_episode_steps=int(cfg.sim.max_episode_steps),
            )
            pfc_now = np.asarray(model.state.pfc_currents, dtype=float).reshape(-1)
            sol_now = np.asarray(model.state.sol_currents, dtype=float).reshape(-1)
            pfc_next = np.asarray(action.pfc_currents_next, dtype=float).reshape(-1)
            sol_next = np.asarray(action.sol_currents_next, dtype=float).reshape(-1)
            physical_derivs = np.concatenate([pfc_next - pfc_now, sol_next - sol_now]) / float(model.t_step)
            normalized_action = physical_derivs / np.where(np.abs(derivative_limits) > 0.0, derivative_limits, 1.0)
            model.step_currents(pfc_currents_next=pfc_next, sol_currents_next=sol_next)

            next_index = min(step + 1, ref_ip.shape[0] - 1)
            post_psi = model.compute_psi()
            post_poly, post_found, post_radii = _boundary(model=model, psi=post_psi, center=center, angles=angles, env=env)
            post_ref_radii = np.nan_to_num(ref_radii_series[next_index], nan=0.0, posinf=0.0, neginf=0.0)
            shape_abs = np.abs(np.nan_to_num(post_radii, nan=0.0, posinf=0.0, neginf=0.0) - post_ref_radii)
            currents = np.concatenate([np.asarray(model.state.pfc_currents, dtype=float), np.asarray(model.state.sol_currents, dtype=float)])
            usage_by_coil = np.abs(currents) / np.where(current_limits > 0.0, current_limits, 1.0)
            current_over = np.maximum(np.abs(currents) - current_limits, 0.0)
            derivative_usage = np.abs(physical_derivs) / np.where(np.abs(derivative_limits) > 0.0, np.abs(derivative_limits), 1.0)

            row = {
                "episode": episode,
                "step": step + 1,
                "time_s": float(model.state.t),
                "shot_id": reset_meta.get("shot_id", ""),
                "source_time_s": reset_meta.get("source_time_s", ""),
                "ip_a": float(model.state.Ip),
                "ip_ref_a": float(ref_ip[next_index]),
                "ip_error_a": abs(float(model.state.Ip) - float(ref_ip[next_index])),
                "boundary_found": float(post_found),
                "shape_error_mean_m": float(np.nanmean(shape_abs)) if post_found else float(cfg.reward.boundary_missing_error_m),
                "shape_error_max_m": float(np.nanmax(shape_abs)) if post_found else float(cfg.reward.boundary_missing_error_m),
                "current_usage_fraction": float(np.nanmax(usage_by_coil)),
                "current_margin_fraction": float(np.nanmin(1.0 - usage_by_coil)),
                "current_over_limit_a": float(np.nanmax(current_over)),
                "derivative_usage": float(np.nanmax(derivative_usage)),
                "action_rms": float(np.sqrt(np.mean(np.square(normalized_action)))),
                "max_abs_action": float(np.nanmax(np.abs(normalized_action))),
            }
            all_rows.append(row)
            if episode == 0 and step in {0, max(0, steps // 2), max(0, steps - 1)}:
                boundary_snapshots.append(
                    {
                        "step": step + 1,
                        "measured": post_radii.tolist(),
                        "reference": post_ref_radii.tolist(),
                    }
                )
            if not post_found and terminated_step is None:
                terminated_step = step + 1
                if args.stop_on_failure:
                    break

        ep_rows = [r for r in all_rows if int(r["episode"]) == episode]
        completion = (terminated_step if terminated_step is not None else len(ep_rows)) / max(float(steps), 1.0)
        episode_summaries.append(_summarize_episode(episode=episode, rows=ep_rows, completion=completion, reset_meta=reset_meta))

    _write_csv(out_dir / "controller_rollout.csv", all_rows)
    summary = _summarize_all(
        export_dir=export_dir,
        config_path=Path(args.config),
        checkpoint=Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None,
        episodes=episode_summaries,
        rows=all_rows,
        steps=steps,
    )
    (out_dir / "controller_rollout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(out_dir / "controller_rollout_report.md", summary=summary)
    _plot(out_dir=out_dir, rows=all_rows, boundary_snapshots=boundary_snapshots)
    print(out_dir)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run an exported learned controller on its production objective and save plots.")
    ap.add_argument("--config", required=True, help="Experiment config or generated config snapshot used by the policy.")
    ap.add_argument("--checkpoint", help="Training checkpoint to export before running.")
    ap.add_argument("--export-dir", help="Existing exported actor bundle, or destination when --checkpoint is supplied.")
    ap.add_argument("--output-dir", required=True, help="Directory for CSV, plots, and report.")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="0 means full sim.max_episode_steps.")
    ap.add_argument("--seed", type=int, default=910000)
    ap.add_argument("--split", default="holdout", choices=("train", "holdout", "all"))
    ap.add_argument("--stop-on-failure", action="store_true", help="Stop each episode when boundary is lost.")
    return ap.parse_args(argv)


def _ensure_export_dir(*, args: argparse.Namespace, out_dir: Path) -> Path:
    if args.checkpoint is None and args.export_dir is None:
        raise ValueError("provide either --checkpoint or --export-dir")
    if args.checkpoint is None:
        export_dir = Path(args.export_dir).expanduser().resolve()
        if not export_dir.is_dir():
            raise FileNotFoundError(f"export directory does not exist: {export_dir}")
        return export_dir
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve() if args.export_dir else out_dir / "exported_actor"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _validate_export_checkpoint(ckpt, checkpoint=checkpoint)
    schema = ckpt["schema"]
    actor = FeedForwardGaussianActor(int(schema["obs_dim"]), int(schema["action_dim"]), int(ckpt["network"]["hidden_dim"]))
    actor.load_state_dict(ckpt["actor_state_dict"])
    export_deterministic_actor(
        actor=actor,
        export_dir=export_dir,
        schema=schema,
        normalization=ckpt["normalization"],
        metadata=ckpt["metadata"],
    )
    return export_dir


def _boundary(*, model, psi: np.ndarray, center: tuple[float, float], angles: np.ndarray, env: TokamakMagneticControlEnv):
    try:
        poly, _level, _status = find_plasma_boundary_with_status(
            psi,
            model.grid,
            center,
            n_levels=80,
            limiter_shape=env.cfg.limiter_shape,
            boundary_mode=env.cfg.boundary_mode,
        )
        radii = radii_from_polyline_ray_intersections(poly, center, angles)
        return poly, True, np.nan_to_num(radii, nan=0.0, posinf=0.0, neginf=0.0)
    except BoundaryNotFoundError:
        return None, False, np.zeros((angles.shape[0],), dtype=float)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(r[key]) for r in rows if r.get(key, "") != ""], dtype=float)
    return float(np.nanmean(values)) if values.size else float("nan")


def _max(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(r[key]) for r in rows if r.get(key, "") != ""], dtype=float)
    return float(np.nanmax(values)) if values.size else float("nan")


def _min(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(r[key]) for r in rows if r.get(key, "") != ""], dtype=float)
    return float(np.nanmin(values)) if values.size else float("nan")


def _summarize_episode(*, episode: int, rows: list[dict[str, Any]], completion: float, reset_meta: dict[str, Any]) -> dict[str, Any]:
    late_start = max(0, int(0.8 * len(rows)))
    late = rows[late_start:] if rows else []
    return {
        "episode": int(episode),
        "shot_id": reset_meta.get("shot_id", ""),
        "source_time_s": reset_meta.get("source_time_s", ""),
        "steps_recorded": len(rows),
        "completion": float(completion),
        "boundary_found_mean": _mean(rows, "boundary_found"),
        "boundary_found_late_min": _min(late, "boundary_found") if late else float("nan"),
        "shape_error_mean_m": _mean(rows, "shape_error_mean_m"),
        "shape_error_mean_late_m": _mean(late, "shape_error_mean_m") if late else float("nan"),
        "shape_error_max_m": _max(rows, "shape_error_max_m"),
        "ip_error_a": _mean(rows, "ip_error_a"),
        "ip_error_late_a": _mean(late, "ip_error_a") if late else float("nan"),
        "current_over_limit_a_max": _max(rows, "current_over_limit_a"),
        "current_usage_fraction_max": _max(rows, "current_usage_fraction"),
        "action_rms": _mean(rows, "action_rms"),
    }


def _summarize_all(*, export_dir: Path, config_path: Path, checkpoint: Path | None, episodes: list[dict[str, Any]], rows: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    completions = np.asarray([float(ep["completion"]) for ep in episodes], dtype=float)
    return {
        "status": "ok",
        "config": str(config_path),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "export_dir": str(export_dir),
        "episodes": len(episodes),
        "steps": int(steps),
        "mean_episode_completion": float(np.nanmean(completions)) if completions.size else float("nan"),
        "min_episode_completion": float(np.nanmin(completions)) if completions.size else float("nan"),
        "boundary_found_mean": _mean(rows, "boundary_found"),
        "boundary_found_late_min": min(float(ep["boundary_found_late_min"]) for ep in episodes if np.isfinite(float(ep["boundary_found_late_min"]))) if episodes else float("nan"),
        "shape_error_mean_m": _mean(rows, "shape_error_mean_m"),
        "shape_error_max_m": _max(rows, "shape_error_max_m"),
        "ip_error_a": _mean(rows, "ip_error_a"),
        "current_over_limit_a_max": _max(rows, "current_over_limit_a"),
        "current_usage_fraction_max": _max(rows, "current_usage_fraction"),
        "action_rms": _mean(rows, "action_rms"),
        "episode_summaries": episodes,
    }


def _write_report(path: Path, *, summary: dict[str, Any]) -> None:
    text = [
        "# Learned Controller Objective Rollout",
        "",
        f"- status: `{summary['status']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- export_dir: `{summary['export_dir']}`",
        f"- episodes: `{summary['episodes']}`",
        f"- steps: `{summary['steps']}`",
        f"- mean_episode_completion: `{summary['mean_episode_completion']:.6g}`",
        f"- min_episode_completion: `{summary['min_episode_completion']:.6g}`",
        f"- boundary_found_mean: `{summary['boundary_found_mean']:.6g}`",
        f"- boundary_found_late_min: `{summary['boundary_found_late_min']:.6g}`",
        f"- shape_error_mean_m: `{summary['shape_error_mean_m']:.6g}`",
        f"- shape_error_max_m: `{summary['shape_error_max_m']:.6g}`",
        f"- ip_error_a: `{summary['ip_error_a']:.6g}`",
        f"- current_over_limit_a_max: `{summary['current_over_limit_a_max']:.6g}`",
        f"- current_usage_fraction_max: `{summary['current_usage_fraction_max']:.6g}`",
        f"- action_rms: `{summary['action_rms']:.6g}`",
        "",
        "## Files",
        "",
        "- `controller_rollout.csv`",
        "- `controller_rollout_summary.json`",
        "- `objective_overview.png`",
        "- `boundary_radii_episode0.png`",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def _plot(*, out_dir: Path, rows: list[dict[str, Any]], boundary_snapshots: list[dict[str, Any]]) -> None:
    if not rows:
        return
    first_episode = [r for r in rows if int(r["episode"]) == 0]
    t = np.asarray([float(r["time_s"]) for r in first_episode], dtype=float)
    ip = np.asarray([float(r["ip_a"]) for r in first_episode], dtype=float)
    ip_ref = np.asarray([float(r["ip_ref_a"]) for r in first_episode], dtype=float)
    shape_mean = np.asarray([float(r["shape_error_mean_m"]) for r in first_episode], dtype=float)
    shape_max = np.asarray([float(r["shape_error_max_m"]) for r in first_episode], dtype=float)
    usage = np.asarray([float(r["current_usage_fraction"]) for r in first_episode], dtype=float)
    over = np.asarray([float(r["current_over_limit_a"]) for r in first_episode], dtype=float)
    action = np.asarray([float(r["action_rms"]) for r in first_episode], dtype=float)
    boundary = np.asarray([float(r["boundary_found"]) for r in first_episode], dtype=float)

    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    axes[0].plot(t, ip_ref, label="target Ip", linewidth=2.0)
    axes[0].plot(t, ip, label="controller Ip", linewidth=1.5)
    axes[0].set_ylabel("Ip [A]")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, np.abs(ip - ip_ref), label="|Ip - target|")
    axes[1].set_ylabel("Ip error [A]")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t, shape_mean, label="mean")
    axes[2].plot(t, shape_max, label="max")
    axes[2].axhline(0.03, color="tab:green", linestyle="--", linewidth=1.0, label="0.03 m")
    axes[2].set_ylabel("boundary error [m]")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(t, usage, label="current usage")
    axes[3].axhline(1.0, color="tab:red", linestyle="--", linewidth=1.0, label="limit")
    axes[3].set_ylabel("current usage")
    ax_over = axes[3].twinx()
    ax_over.plot(t, over, color="tab:orange", alpha=0.5, label="over limit A")
    ax_over.set_ylabel("over limit [A]")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, alpha=0.25)

    axes[4].plot(t, action, label="normalized action RMS")
    axes[4].plot(t, boundary, label="boundary found", alpha=0.7)
    axes[4].set_ylabel("action / found")
    axes[4].set_xlabel("time [s]")
    axes[4].legend(loc="best")
    axes[4].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "objective_overview.png", dpi=160)
    plt.close(fig)

    if boundary_snapshots:
        fig2, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(boundary_snapshots[0]["reference"]))
        for snap in boundary_snapshots:
            ax.plot(x, snap["reference"], linestyle="--", alpha=0.8, label=f"ref step {snap['step']}")
            ax.plot(x, snap["measured"], alpha=0.8, label=f"measured step {snap['step']}")
        ax.set_xlabel("boundary angle index")
        ax.set_ylabel("radius [m]")
        ax.set_title("Episode 0 Boundary Radii")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig2.tight_layout()
        fig2.savefig(out_dir / "boundary_radii_episode0.png", dpi=160)
        plt.close(fig2)


if __name__ == "__main__":
    raise SystemExit(main())
