#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.training.trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare zero, learned, and real-Jdot oracle policies on replay windows.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--split", default="holdout", choices=("train", "holdout", "all"))
    ap.add_argument("--seed", type=int, default=386400)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    if int(args.episodes) <= 0 or int(args.steps) <= 0:
        raise ValueError("--episodes and --steps must be positive")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    cfg = load_experiment_config(args.config)
    cfg = replace(
        cfg,
        sim=replace(cfg.sim, csv_initial_state_split=str(args.split)),
        training=replace(cfg.training, production_mode=False, eval_episodes=int(args.episodes), eval_max_steps=int(args.steps)),
    )
    policies = ["zero", "oracle"]
    trainer: Trainer | None = None
    if args.checkpoint:
        trainer = Trainer(cfg, steps=max(1, int(args.steps)), num_envs=int(args.episodes), device=str(device), output_dir=out_dir / "_trainer_tmp", export_policy=False)
        trainer._load_warm_start_checkpoint(args.checkpoint)
        trainer.actor.eval()
        trainer.critic.eval()
        policies.insert(1, "policy")

    results: dict[str, object] = {"config": str(Path(args.config).resolve()), "checkpoint": args.checkpoint, "split": str(args.split)}
    policy_metrics: dict[str, dict[str, float]] = {}
    category_rows: list[dict[str, object]] = []
    q_summary: dict[str, float] = {}
    for policy in policies:
        metrics, per_category, q_values = _evaluate_policy(
            cfg=cfg,
            trainer=trainer,
            policy=policy,
            episodes=int(args.episodes),
            steps=int(args.steps),
            seed=int(args.seed),
            device=device,
        )
        policy_metrics[policy] = metrics
        for category, row in per_category.items():
            category_rows.append({"policy": policy, "category": category, **row})
        if q_values:
            q_summary.update({f"{policy}/{k}": v for k, v in q_values.items()})

    results["policies"] = policy_metrics
    results["categories"] = category_rows
    results["q_diagnostics"] = q_summary
    if {"zero", "oracle", "policy"}.issubset(policy_metrics):
        results["regret"] = {
            "ip": _regret(policy_metrics["policy"].get("ip_error_a"), policy_metrics["zero"].get("ip_error_a"), policy_metrics["oracle"].get("ip_error_a")),
            "shape": _regret(policy_metrics["policy"].get("shape_error_mean_m"), policy_metrics["zero"].get("shape_error_mean_m"), policy_metrics["oracle"].get("shape_error_mean_m")),
        }

    (out_dir / "oracle_baseline_eval.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    with (out_dir / "oracle_baseline_eval_by_category.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["policy", "category", "episodes", "shape_error_mean_m", "shape_error_p90_m", "ip_error_a", "ip_error_p90_a", "current_usage_fraction", "action_rms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in category_rows:
            writer.writerow(row)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


@torch.no_grad()
def _evaluate_policy(
    *,
    cfg,
    trainer: Trainer | None,
    policy: str,
    episodes: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float]]:
    env = TokamakMagneticControlEnv(cfg, batch_size=int(episodes), device=device, seed=int(seed))
    obs = env.reset()
    metadata = list(env.reset_metadata)
    categories = [str(row.get("difficulty_bin") or "unknown") for row in metadata]
    oracle_actions = None
    if policy in {"oracle", "policy"}:
        library = getattr(env, "_boundary_replay_library", None)
        if library is None or not hasattr(library, "real_action_for_segment"):
            raise ValueError("oracle policy requires replay-window oracle targets with real_jdot_action")
        oracle_actions = np.stack(
            [
                library.real_action_for_segment(
                    str(row["shot_id"]),
                    source_index=int(row["source_index"]),
                    source_time_s=float(row.get("source_time_s", float("nan"))),
                    steps=int(steps),
                )
                for row in metadata
            ],
            axis=0,
        )

    totals: dict[str, list[float]] = {
        "shape_error_mean_m": [],
        "ip_error_a": [],
        "current_usage_fraction": [],
        "action_rms": [],
    }
    per_episode: dict[str, list[list[float]]] = {name: [[] for _ in range(int(episodes))] for name in totals}
    q_values: dict[str, float] = {}
    active = torch.ones((env.batch_size,), dtype=torch.bool, device=device)
    for step in range(int(steps)):
        if policy == "zero":
            action = torch.zeros((env.batch_size, env.action_dim), dtype=torch.float32, device=device)
        elif policy == "oracle":
            assert oracle_actions is not None
            action = torch.as_tensor(oracle_actions[:, step, :], dtype=torch.float32, device=device)
        elif policy == "policy":
            if trainer is None:
                raise ValueError("policy evaluation requires --checkpoint")
            action = trainer.actor.deterministic(obs)
            if step == 0:
                q_values = _q_diagnostics(trainer=trainer, env=env, obs=obs, oracle_actions=oracle_actions)
        else:
            raise ValueError(f"unsupported policy: {policy}")
        out = env.step(action)
        comps = out.info.get("reward_components", {}) if isinstance(out.info, dict) else {}
        done = out.terminated | out.truncated
        active_cpu = active.detach().cpu().numpy().astype(bool)
        for name in totals:
            value = comps.get(name)
            if value is None:
                continue
            arr = np.asarray(value.detach().cpu().numpy(), dtype=float).reshape(-1)
            finite = np.isfinite(arr) & active_cpu
            totals[name].extend(arr[finite].astype(float).tolist())
            for index in np.nonzero(finite)[0]:
                per_episode[name][int(index)].append(float(arr[int(index)]))
        active = active & ~done
        obs = out.obs

    metrics: dict[str, float] = {}
    for name, values in totals.items():
        arr = np.asarray(values, dtype=float)
        metrics[name] = float(np.nanmean(arr)) if arr.size else float("nan")
    per_category: dict[str, dict[str, float]] = {}
    for category in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == category]
        per_category[category] = _category_metrics(idx, per_episode)
    per_category["all"] = _category_metrics(list(range(int(episodes))), per_episode)
    return metrics, per_category, q_values


@torch.no_grad()
def _q_diagnostics(*, trainer: Trainer, env: TokamakMagneticControlEnv, obs: torch.Tensor, oracle_actions) -> dict[str, float]:
    critic_obs = env.critic_obs()
    zero = torch.zeros((env.batch_size, env.action_dim), dtype=torch.float32, device=obs.device)
    policy = trainer.actor.deterministic(obs)
    out = {
        "zero": float(torch.mean(trainer.critic(critic_obs, zero)[0]).detach().cpu().item()),
        "policy": float(torch.mean(trainer.critic(critic_obs, policy)[0]).detach().cpu().item()),
    }
    if oracle_actions is not None:
        oracle = torch.as_tensor(oracle_actions[:, 0, :], dtype=torch.float32, device=obs.device)
        out["real_jdot"] = float(torch.mean(trainer.critic(critic_obs, oracle)[0]).detach().cpu().item())
    return out


def _category_metrics(indices: list[int], per_episode: dict[str, list[list[float]]]) -> dict[str, float]:
    out: dict[str, float] = {"episodes": float(len(indices))}
    for src, mean_key, p90_key in (
        ("shape_error_mean_m", "shape_error_mean_m", "shape_error_p90_m"),
        ("ip_error_a", "ip_error_a", "ip_error_p90_a"),
    ):
        values = [np.nanmean(per_episode[src][i]) for i in indices if per_episode[src][i]]
        arr = np.asarray(values, dtype=float)
        out[mean_key] = float(np.nanmean(arr)) if arr.size else float("nan")
        out[p90_key] = float(np.nanpercentile(arr, 90)) if arr.size else float("nan")
    for src in ("current_usage_fraction", "action_rms"):
        values = [v for i in indices for v in per_episode[src][i]]
        arr = np.asarray(values, dtype=float)
        out[src] = float(np.nanmean(arr)) if arr.size else float("nan")
    return out


def _regret(policy_value: object, zero_value: object, oracle_value: object) -> float:
    p = _finite_or_nan(policy_value)
    z = _finite_or_nan(zero_value)
    o = _finite_or_nan(oracle_value)
    denom = z - o
    if not np.isfinite(p) or not np.isfinite(z) or not np.isfinite(o) or abs(denom) < 1.0e-12:
        return float("nan")
    return float((p - o) / denom)


def _finite_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
