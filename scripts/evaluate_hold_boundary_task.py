#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.config.schema import BoundaryReferenceConfig, IpReferenceConfig
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.training.trainer import Trainer

METRIC_KEYS = (
    "shape_error_mean_m",
    "shape_error_max_m",
    "ip_error_a",
    "current_usage_fraction",
    "current_over_limit_a",
    "action_rms",
    "action_saturation_fraction",
    "boundary_found",
    "terminated_boundary",
    "terminated_current",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate exported learned policy on synthetic held-boundary holdout tasks.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--split", default="holdout", choices=("holdout", "train", "all"))
    ap.add_argument("--expected-shot", default="3864")
    ap.add_argument("--seed", type=int, default=386400)
    ap.add_argument("--base-index", type=int, default=0)
    ap.add_argument(
        "--ip-profile-kind",
        default="hold_boundary_eval_profile",
        choices=("hold_boundary_eval_profile", "hold_boundary_eval_cut_profile"),
    )
    ap.add_argument("--parent-steps", type=int, default=900)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    if bool(args.export_dir) == bool(args.checkpoint):
        raise ValueError("provide exactly one of --export-dir or --checkpoint")
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive")
    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is false: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _forced_hold_boundary_config(
        load_experiment_config(args.config),
        steps=int(args.steps),
        split=str(args.split),
        device=str(device),
        ip_profile_kind=str(args.ip_profile_kind),
        parent_steps=int(args.parent_steps),
    )
    _write_config_snapshot(cfg, out_dir / "hold_boundary_eval_config_snapshot.json")
    indices = _select_heldout_indices(
        cfg,
        episodes=int(args.episodes),
        base_index=int(args.base_index),
        seed=int(args.seed),
        device=device,
        expected_shot=str(args.expected_shot),
    )

    trainer = Trainer(cfg, steps=max(1, int(args.steps)), num_envs=1, device=str(device), output_dir=out_dir / "_trainer_tmp", export_policy=False)
    if args.checkpoint:
        trainer._load_warm_start_checkpoint(args.checkpoint)
    else:
        _load_export_actor(trainer, Path(args.export_dir))
    trainer.actor.eval()
    trainer.critic.eval()

    results: dict[str, Any] = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": None if args.checkpoint is None else str(Path(args.checkpoint).resolve()),
        "export_dir": None if args.export_dir is None else str(Path(args.export_dir).resolve()),
        "output_dir": str(out_dir.resolve()),
        "split": str(args.split),
        "expected_shot": str(args.expected_shot),
        "episodes": int(args.episodes),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "base_index": int(args.base_index),
        "indices": indices.astype(int).tolist(),
        "policies": {},
    }

    for policy in ("policy", "no_control"):
        print(f"Starting hold_boundary_eval policy={policy} episodes={args.episodes} steps={args.steps} device={device}", flush=True)
        summary, windows, step_rows = _rollout_policy(
            cfg=cfg,
            trainer=trainer,
            policy=policy,
            indices=indices,
            steps=int(args.steps),
            seed=int(args.seed),
            device=device,
            show_progress=not bool(args.no_progress),
            expected_shot=str(args.expected_shot),
        )
        results["policies"][policy] = summary
        _write_csv(out_dir / f"hold_boundary_eval_{policy}_windows.csv", windows)
        _write_csv(out_dir / f"hold_boundary_eval_{policy}_step_summary.csv", step_rows)
        print(f"Finished hold_boundary_eval policy={policy}: {json.dumps(summary, sort_keys=True)}", flush=True)

    (out_dir / "hold_boundary_eval_summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(results, out_dir / "hold_boundary_eval_report.md")
    _plot_overview(results, out_dir / "hold_boundary_eval_overview.png")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _forced_hold_boundary_config(cfg, *, steps: int, split: str, device: str, ip_profile_kind: str = "hold_boundary_eval_profile", parent_steps: int = 900):
    limits_path = cfg.reference.ip.limits_path or Path("data/processed/t15_reference_limits.json").resolve()
    if ip_profile_kind not in {"hold_boundary_eval_profile", "hold_boundary_eval_cut_profile"}:
        raise ValueError(f"unsupported hold_boundary_eval Ip profile: {ip_profile_kind}")
    if ip_profile_kind == "hold_boundary_eval_cut_profile":
        ip_kwargs = {
            "kind": "hold_boundary_eval_cut_profile",
            "parent_steps": int(parent_steps),
            "segment_min_steps": 300,
            "segment_max_steps": int(parent_steps),
            "segment_count_min": 1,
            "segment_count_max": 3,
            "hold_probability": 0.45,
            "hold_min_steps": 300,
            "hold_max_steps": int(parent_steps),
        }
    else:
        ip_kwargs = {
            "kind": "hold_boundary_eval_profile",
            "parent_steps": 0,
            "segment_min_steps": 50,
            "segment_max_steps": 170,
            "segment_count_min": 1,
            "segment_count_max": 6,
            "hold_probability": 0.45,
            "hold_min_steps": 35,
            "hold_max_steps": 220,
        }
    ip = IpReferenceConfig(
        kind=ip_kwargs["kind"],
        limits_path=limits_path,
        start_mode="reset_ip",
        parent_steps=int(ip_kwargs["parent_steps"]),
        segment_min_steps=int(ip_kwargs["segment_min_steps"]),
        segment_max_steps=int(ip_kwargs["segment_max_steps"]),
        segment_count_min=int(ip_kwargs["segment_count_min"]),
        segment_count_max=int(ip_kwargs["segment_count_max"]),
        hold_probability=float(ip_kwargs["hold_probability"]),
        ramp_rate_reference="robust_mean",
        ramp_up_rate_min_fraction=0.05,
        ramp_up_rate_fraction=0.20,
        ramp_down_rate_min_fraction=0.05,
        ramp_down_rate_fraction=0.20,
        hold_min_steps=int(ip_kwargs["hold_min_steps"]),
        hold_max_steps=int(ip_kwargs["hold_max_steps"]),
        final_hold_min_steps=0,
        smooth_ramps=False,
        max_delta_fraction=0.35,
    )
    return replace(
        cfg,
        sim=replace(
            cfg.sim,
            compute_backend="gpu" if str(device).startswith("cuda") else cfg.sim.compute_backend,
            gpu_device=str(device),
            max_episode_steps=int(steps),
            csv_initial_state_split=str(split),
        ),
        reference=replace(
            cfg.reference,
            duration_s=float(steps) * 0.001,
            t_step=0.001,
            ip=ip,
            boundary=BoundaryReferenceConfig(kind="hold_reset_boundary"),
        ),
        training=replace(
            cfg.training,
            production_mode=False,
            eval_episodes=max(1, int(cfg.training.eval_episodes)),
            eval_max_steps=int(steps),
            distributed_mode="single",
            num_envs=max(1, int(cfg.training.num_envs)),
            device=str(device),
        ),
    )


def _select_heldout_indices(
    cfg,
    *,
    episodes: int,
    base_index: int,
    seed: int,
    device: torch.device,
    expected_shot: str,
) -> np.ndarray:
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device=device, seed=int(seed))
    library = getattr(env, "_csv_initial_states", None)
    if library is None:
        raise ValueError("hold_boundary_eval requires sim.reset_source=csv_initial_states")
    shots = {str(int(v)) for v in np.asarray(library.shot_id).reshape(-1).tolist()}
    if shots != {str(int(expected_shot))}:
        raise ValueError(f"hold_boundary_eval split must contain only shot {expected_shot}; got {sorted(shots)}")
    count = len(library)
    if count <= 0:
        raise ValueError("hold_boundary_eval selected split is empty")
    return (int(base_index) + np.arange(int(episodes), dtype=np.int64)) % int(count)


@torch.no_grad()
def _rollout_policy(
    *,
    cfg,
    trainer: Trainer,
    policy: str,
    indices: np.ndarray,
    steps: int,
    seed: int,
    device: torch.device,
    show_progress: bool,
    expected_shot: str,
) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
    episodes = int(indices.size)
    env = TokamakMagneticControlEnv(cfg, batch_size=episodes, device=device, seed=int(seed))
    obs = env.reset_to_csv_indices(indices)
    metadata = list(env.reset_metadata)
    shots = {str(int(row.get("shot_id", -1))) for row in metadata}
    if shots != {str(int(expected_shot))}:
        raise ValueError(f"hold_boundary_eval reset produced shots {sorted(shots)}, expected only {expected_shot}")

    per_episode: dict[str, list[list[float]]] = {name: [[] for _ in range(episodes)] for name in METRIC_KEYS}
    done_step = np.full((episodes,), int(steps), dtype=int)
    terminated_boundary = np.zeros((episodes,), dtype=bool)
    terminated_current = np.zeros((episodes,), dtype=bool)
    active = torch.ones((episodes,), dtype=torch.bool, device=device)
    step_rows: list[dict[str, object]] = []
    iterator = tqdm(
        range(int(steps)),
        desc=f"{policy} hold_boundary_eval",
        unit="step",
        disable=not bool(show_progress),
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for step in iterator:
        if policy == "policy":
            action = trainer.actor.deterministic(obs)
        elif policy == "no_control":
            action = torch.zeros((episodes, env.action_dim), dtype=torch.float32, device=device)
        else:
            raise ValueError(f"unknown policy: {policy}")
        out = env.step(action)
        comps = out.info.get("reward_components", {}) if isinstance(out.info, dict) else {}
        active_cpu = active.detach().cpu().numpy().astype(bool)
        for name in METRIC_KEYS:
            value = comps.get(name)
            if value is None:
                continue
            arr = np.asarray(value.detach().cpu().numpy(), dtype=float).reshape(-1)
            for index in np.nonzero(active_cpu & np.isfinite(arr))[0].tolist():
                per_episode[name][int(index)].append(float(arr[int(index)]))
        term_b = np.asarray(comps.get("terminated_boundary", torch.zeros_like(active)).detach().cpu().numpy(), dtype=float).reshape(-1) > 0.5
        term_c = np.asarray(comps.get("terminated_current", torch.zeros_like(active)).detach().cpu().numpy(), dtype=float).reshape(-1) > 0.5
        just_done = (out.terminated | out.truncated) & active
        just_done_cpu = just_done.detach().cpu().numpy().astype(bool)
        if np.any(just_done_cpu):
            done_step[just_done_cpu] = int(step) + 1
            terminated_boundary[just_done_cpu] |= term_b[just_done_cpu]
            terminated_current[just_done_cpu] |= term_c[just_done_cpu]
        step_rows.append(_step_summary_row(policy=policy, step=int(step), active=active_cpu, comps=comps))
        active = active & ~just_done
        obs = out.obs
        if not bool(torch.any(active).item()):
            break

    windows = _window_rows(
        policy=policy,
        metadata=metadata,
        indices=indices,
        per_episode=per_episode,
        done_step=done_step,
        steps=int(steps),
        terminated_boundary=terminated_boundary,
        terminated_current=terminated_current,
    )
    summary = _summary_from_windows(windows)
    return summary, windows, step_rows


def _step_summary_row(*, policy: str, step: int, active: np.ndarray, comps: dict[str, torch.Tensor]) -> dict[str, object]:
    row: dict[str, object] = {"policy": policy, "step": int(step), "active_episodes": int(np.count_nonzero(active))}
    for key in METRIC_KEYS:
        value = comps.get(key)
        if value is None:
            continue
        arr = np.asarray(value.detach().cpu().numpy(), dtype=float).reshape(-1)
        mask = active & np.isfinite(arr)
        row[key] = float(np.nanmean(arr[mask])) if np.any(mask) else float("nan")
        row[f"{key}_max"] = float(np.nanmax(arr[mask])) if np.any(mask) else float("nan")
    return row


def _window_rows(
    *,
    policy: str,
    metadata: list[dict[str, object]],
    indices: np.ndarray,
    per_episode: dict[str, list[list[float]]],
    done_step: np.ndarray,
    steps: int,
    terminated_boundary: np.ndarray,
    terminated_current: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    late_start = int(np.floor(0.8 * float(steps)))
    for i, row in enumerate(metadata):
        out: dict[str, object] = {
            "policy": policy,
            "episode": int(i),
            "library_index": int(indices[i]),
            "shot_id": row.get("shot_id", ""),
            "source_index": row.get("source_index", -1),
            "source_time_s": row.get("source_time_s", float("nan")),
            "difficulty_bin": row.get("difficulty_bin", ""),
            "episode_steps": int(done_step[i]),
            "episode_completion": float(done_step[i] / max(int(steps), 1)),
            "full_episode_success": float(done_step[i] >= int(steps) and not terminated_boundary[i] and not terminated_current[i]),
            "terminated_boundary": float(terminated_boundary[i]),
            "terminated_current": float(terminated_current[i]),
        }
        for name, values_by_episode in per_episode.items():
            values = np.asarray(values_by_episode[i], dtype=float)
            out[name] = float(np.nanmean(values)) if values.size else float("nan")
            out[f"{name}_max"] = float(np.nanmax(values)) if values.size else float("nan")
            if values.size:
                late = values[np.arange(values.size) >= late_start]
                out[f"{name}_late"] = float(np.nanmean(late)) if late.size else float("nan")
                out[f"{name}_late_max"] = float(np.nanmax(late)) if late.size else float("nan")
            else:
                out[f"{name}_late"] = float("nan")
                out[f"{name}_late_max"] = float("nan")
        rows.append(out)
    return rows


def _summary_from_windows(windows: list[dict[str, object]]) -> dict[str, float]:
    out: dict[str, float] = {"episodes": float(len(windows))}
    if not windows:
        return out
    for key in (
        "episode_completion",
        "full_episode_success",
        "shape_error_mean_m_late",
        "shape_error_max_m_late",
        "ip_error_a_late",
        "current_usage_fraction_late",
        "action_rms_late",
        "action_saturation_fraction_late",
        "terminated_boundary",
        "terminated_current",
    ):
        arr = _col(windows, key)
        out[_summary_key(key)] = float(np.nanmean(arr)) if arr.size else float("nan")
    for key in ("episode_completion", "boundary_found_late"):
        arr = _col(windows, key)
        out[f"{_summary_key(key)}_min"] = float(np.nanmin(arr)) if arr.size else float("nan")
    current_over = _col(windows, "current_over_limit_a_late")
    usage = _col(windows, "current_usage_fraction_late")
    out["current_over_limit_a_late_max"] = float(np.nanmax(current_over)) if current_over.size else float("nan")
    out["current_over_limit_fraction_late"] = float(np.nanmean(current_over > 0.0)) if current_over.size else float("nan")
    out["current_over_limit_5ka_fraction_late"] = float(np.nanmean(current_over > 5000.0)) if current_over.size else float("nan")
    out["current_over_limit_1pct_fraction_late"] = float(np.nanmean(usage > 1.01)) if usage.size else float("nan")
    return out


def _summary_key(key: str) -> str:
    return {
        "episode_completion": "mean_episode_completion",
        "full_episode_success": "full_episode_success",
        "terminated_boundary": "terminated_boundary",
        "terminated_current": "terminated_current",
    }.get(key, key)


def _col(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_config_snapshot(cfg, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "name": cfg.name,
                "sim": {
                    "config_path": str(cfg.sim.config_path),
                    "csv_initial_state_library": str(cfg.sim.csv_initial_state_library),
                    "csv_initial_state_split": cfg.sim.csv_initial_state_split,
                    "compute_backend": cfg.sim.compute_backend,
                    "gpu_device": cfg.sim.gpu_device,
                    "max_episode_steps": cfg.sim.max_episode_steps,
                    "action_contract": cfg.sim.action_contract,
                },
                "reference": {
                    "duration_s": cfg.reference.duration_s,
                    "t_step": cfg.reference.t_step,
                    "ip": {
                        "kind": cfg.reference.ip.kind,
                        "limits_path": str(cfg.reference.ip.limits_path),
                        "parent_steps": cfg.reference.ip.parent_steps,
                        "segment_min_steps": cfg.reference.ip.segment_min_steps,
                        "segment_max_steps": cfg.reference.ip.segment_max_steps,
                        "segment_count_min": cfg.reference.ip.segment_count_min,
                        "segment_count_max": cfg.reference.ip.segment_count_max,
                        "ramp_rate_reference": cfg.reference.ip.ramp_rate_reference,
                        "ramp_up_rate_min_fraction": cfg.reference.ip.ramp_up_rate_min_fraction,
                        "ramp_up_rate_fraction": cfg.reference.ip.ramp_up_rate_fraction,
                        "ramp_down_rate_min_fraction": cfg.reference.ip.ramp_down_rate_min_fraction,
                        "ramp_down_rate_fraction": cfg.reference.ip.ramp_down_rate_fraction,
                    },
                    "boundary": {"kind": cfg.reference.boundary.kind},
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_report(results: dict[str, Any], path: Path) -> None:
    policies = results.get("policies", {})
    metric_order = [
        "mean_episode_completion",
        "full_episode_success",
        "shape_error_mean_m_late",
        "shape_error_max_m_late",
        "ip_error_a_late",
        "current_usage_fraction_late",
        "current_over_limit_5ka_fraction_late",
        "action_rms_late",
        "action_saturation_fraction_late",
        "terminated_boundary",
        "terminated_current",
    ]
    lines = ["# hold_boundary_eval", "", f"Episodes: {results.get('episodes')} x {results.get('steps')} steps", ""]
    lines.append("| policy | " + " | ".join(metric_order) + " |")
    lines.append("|---|" + "|".join("---" for _ in metric_order) + "|")
    for name in ("policy", "no_control"):
        metrics = policies.get(name, {})
        lines.append("| " + name + " | " + " | ".join(_fmt(metrics.get(metric)) for metric in metric_order) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_overview(results: dict[str, Any], path: Path) -> None:
    policies = results.get("policies", {})
    names = [name for name in ("policy", "no_control") if name in policies]
    metrics = ["shape_error_mean_m_late", "ip_error_a_late", "current_usage_fraction_late", "action_rms_late"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    for ax, metric in zip(axes.reshape(-1), metrics, strict=True):
        values = [float(policies[name].get(metric, np.nan)) for name in names]
        ax.bar(names, values)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fmt(value: object) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(v):
        return "nan"
    if abs(v) >= 1000.0 or (abs(v) > 0.0 and abs(v) < 1.0e-3):
        return f"{v:.4g}"
    return f"{v:.5f}"


def _load_export_actor(trainer: Trainer, export_dir: Path) -> None:
    actor_path = export_dir / "actor.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"export actor does not exist: {actor_path}")
    data = torch.load(actor_path, map_location=trainer.device, weights_only=False)
    state = data.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"export actor is missing actor_state_dict: {actor_path}")
    trainer.actor.load_state_dict(state)


if __name__ == "__main__":
    raise SystemExit(main())
