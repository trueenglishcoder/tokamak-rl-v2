#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.training.trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare zero, learned, and real-Jdot oracle policies on replay windows.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--episodes", type=int, default=0, help="Number of split windows to evaluate; <=0 means all windows exactly once.")
    ap.add_argument("--all-windows", action="store_true", help="Evaluate every row in the requested split exactly once.")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--split", default="holdout", choices=("train", "holdout", "all"))
    ap.add_argument("--seed", type=int, default=386400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-progress", action="store_true", help="Disable per-policy rollout progress bars.")
    args = ap.parse_args(argv)

    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    cfg = load_experiment_config(args.config)
    cfg = replace(
        cfg,
        sim=replace(cfg.sim, csv_initial_state_split=str(args.split)),
        training=replace(cfg.training, production_mode=False, eval_episodes=int(args.episodes), eval_max_steps=int(args.steps)),
    )
    requested_episodes = int(args.episodes)
    if bool(args.all_windows) or requested_episodes <= 0:
        print(f"Counting replay windows for split={args.split}...", flush=True)
        requested_episodes = _split_window_count(cfg=cfg, device=device, seed=int(args.seed))
    if requested_episodes <= 0:
        raise ValueError("requested split has no windows")
    print(
        f"Evaluating {requested_episodes} windows for {args.steps} steps on {device}.",
        flush=True,
    )

    policies = ["zero", "oracle"]
    trainer: Trainer | None = None
    checkpoint = args.checkpoint
    export_dir = Path(args.export_dir) if args.export_dir else None
    if export_dir is not None and checkpoint is None:
        checkpoint = _find_checkpoint_for_export(export_dir)
    critic_available = checkpoint is not None
    if checkpoint is not None or export_dir is not None:
        trainer = Trainer(cfg, steps=max(1, int(args.steps)), num_envs=1, device=str(device), output_dir=out_dir / "_trainer_tmp", export_policy=False)
        if checkpoint is not None:
            trainer._load_warm_start_checkpoint(checkpoint)
        if export_dir is not None:
            _load_export_actor(trainer, export_dir)
        trainer.actor.eval()
        trainer.critic.eval()
        policies.insert(1, "policy")

    results: dict[str, object] = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": None if checkpoint is None else str(Path(checkpoint).resolve()),
        "export_dir": None if export_dir is None else str(export_dir.resolve()),
        "split": str(args.split),
        "episodes": int(requested_episodes),
        "steps": int(args.steps),
    }
    policy_metrics: dict[str, dict[str, float]] = {}
    category_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    q_summary: dict[str, float] = {}
    for policy in policies:
        print(f"Starting policy={policy}...", flush=True)
        metrics, per_category, q_values, per_window = _evaluate_policy(
            cfg=cfg,
            trainer=trainer,
            policy=policy,
            episodes=int(requested_episodes),
            steps=int(args.steps),
            seed=int(args.seed),
            device=device,
            critic_available=critic_available,
            show_progress=not bool(args.no_progress),
        )
        print(f"Finished policy={policy}.", flush=True)
        policy_metrics[policy] = metrics
        for category, row in per_category.items():
            category_rows.append({"policy": policy, "category": category, **row})
        for row in per_window:
            window_rows.append({"policy": policy, **row})
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
        fieldnames = ["policy", "category", "episodes", "shape_error_mean_m", "shape_error_p90_m", "ip_error_a", "ip_error_p90_a", "current_usage_fraction", "action_rms", "oracle_action_rms", "policy_oracle_action_rms_ratio", "policy_oracle_action_mae_mean", "policy_oracle_cosine"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in category_rows:
            writer.writerow(row)
    if window_rows:
        keys = sorted({str(key) for row in window_rows for key in row})
        with (out_dir / "oracle_baseline_eval_windows.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in window_rows:
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
    critic_available: bool = True,
    show_progress: bool = True,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float], list[dict[str, object]]]:
    print(
        f"  building env for policy={policy}, episodes={episodes}, steps={steps}, device={device}",
        flush=True,
    )
    env = TokamakMagneticControlEnv(cfg, batch_size=int(episodes), device=device, seed=int(seed))
    print(f"  resetting {episodes} replay windows for policy={policy}", flush=True)
    obs = env.reset_to_csv_indices(np.arange(int(episodes), dtype=np.int64))
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
        "shape_error_max_m": [],
        "ip_error_a": [],
        "current_usage_fraction": [],
        "action_rms": [],
        "boundary_found": [],
    }
    per_episode: dict[str, list[list[float]]] = {name: [[] for _ in range(int(episodes))] for name in totals}
    action_mae_sum = np.zeros((env.action_dim,), dtype=float)
    action_mae_count = 0
    action_rms_policy: list[float] = []
    action_rms_oracle: list[float] = []
    action_cosine: list[float] = []
    per_episode_action: dict[str, list[list[float]]] = {
        "policy_action_rms": [[] for _ in range(int(episodes))],
        "oracle_action_rms": [[] for _ in range(int(episodes))],
        "policy_oracle_action_mae_mean": [[] for _ in range(int(episodes))],
        "policy_oracle_cosine": [[] for _ in range(int(episodes))],
    }
    q_values: dict[str, float] = {}
    q_acc: dict[str, list[float]] = {"zero": [], "policy": [], "real_jdot": []}
    q_oracle_rank = 0
    q_rank_count = 0
    critic_state = trainer.critic.zero_state(env.batch_size, device) if policy == "policy" and trainer is not None and critic_available else None
    active = torch.ones((env.batch_size,), dtype=torch.bool, device=device)
    step_iter = tqdm(
        range(int(steps)),
        desc=f"{policy} rollout",
        unit="step",
        disable=not bool(show_progress),
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for step in step_iter:
        if policy == "zero":
            action = torch.zeros((env.batch_size, env.action_dim), dtype=torch.float32, device=device)
        elif policy == "oracle":
            assert oracle_actions is not None
            action = torch.as_tensor(oracle_actions[:, step, :], dtype=torch.float32, device=device)
        elif policy == "policy":
            if trainer is None:
                raise ValueError("policy evaluation requires --checkpoint or --export-dir")
            action = trainer.actor.deterministic(obs)
        else:
            raise ValueError(f"unsupported policy: {policy}")
        if policy == "policy" and trainer is not None and oracle_actions is not None and critic_state is not None:
            q_step, critic_state = _q_diagnostics(
                trainer=trainer,
                env=env,
                obs=obs,
                policy_action=action,
                oracle_action=torch.as_tensor(oracle_actions[:, step, :], dtype=torch.float32, device=device),
                critic_state=critic_state,
                active=active,
            )
            for name, values in q_step.items():
                q_acc[name].extend(values)
            if q_step["real_jdot"]:
                q_zero = np.asarray(q_step["zero"], dtype=float)
                q_policy = np.asarray(q_step["policy"], dtype=float)
                q_oracle = np.asarray(q_step["real_jdot"], dtype=float)
                q_oracle_rank += int(np.count_nonzero(q_oracle > np.maximum(q_policy, q_zero)))
                q_rank_count += int(q_oracle.size)
        out = env.step(action)
        comps = out.info.get("reward_components", {}) if isinstance(out.info, dict) else {}
        done = out.terminated | out.truncated
        active_cpu = active.detach().cpu().numpy().astype(bool)
        if oracle_actions is not None and policy == "policy":
            policy_action = action.detach().cpu().numpy().astype(float)
            oracle_action = np.asarray(oracle_actions[:, step, :], dtype=float)
            if np.any(active_cpu):
                delta = policy_action[active_cpu] - oracle_action[active_cpu]
                action_mae_sum += np.sum(np.abs(delta), axis=0)
                action_mae_count += int(delta.shape[0])
                p_norm = np.linalg.norm(policy_action[active_cpu], axis=1)
                o_norm = np.linalg.norm(oracle_action[active_cpu], axis=1)
                p_rms = np.sqrt(np.mean(policy_action[active_cpu] ** 2, axis=1)).astype(float)
                o_rms = np.sqrt(np.mean(oracle_action[active_cpu] ** 2, axis=1)).astype(float)
                mae_mean = np.mean(np.abs(delta), axis=1).astype(float)
                denom = np.maximum(p_norm * o_norm, 1.0e-12)
                cosine = (np.sum(policy_action[active_cpu] * oracle_action[active_cpu], axis=1) / denom).astype(float)
                action_rms_policy.extend(p_rms.tolist())
                action_rms_oracle.extend(o_rms.tolist())
                action_cosine.extend(cosine.tolist())
                active_indices = np.nonzero(active_cpu)[0]
                for local_i, episode_i in enumerate(active_indices.tolist()):
                    per_episode_action["policy_action_rms"][int(episode_i)].append(float(p_rms[local_i]))
                    per_episode_action["oracle_action_rms"][int(episode_i)].append(float(o_rms[local_i]))
                    per_episode_action["policy_oracle_action_mae_mean"][int(episode_i)].append(float(mae_mean[local_i]))
                    per_episode_action["policy_oracle_cosine"][int(episode_i)].append(float(cosine[local_i]))
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
        metrics[f"{name}_p90"] = float(np.nanpercentile(arr, 90)) if arr.size else float("nan")
    if action_mae_count > 0:
        mae = action_mae_sum / float(action_mae_count)
        metrics["policy_oracle_action_mae_mean"] = float(np.mean(mae))
        for coil, value in enumerate(mae.tolist()):
            metrics[f"policy_oracle_action_mae_coil_{coil}"] = float(value)
        policy_rms = float(np.nanmean(np.asarray(action_rms_policy, dtype=float))) if action_rms_policy else float("nan")
        oracle_rms = float(np.nanmean(np.asarray(action_rms_oracle, dtype=float))) if action_rms_oracle else float("nan")
        metrics["policy_action_rms"] = policy_rms
        metrics["oracle_action_rms"] = oracle_rms
        metrics["policy_oracle_action_rms_ratio"] = float(policy_rms / oracle_rms) if np.isfinite(policy_rms) and np.isfinite(oracle_rms) and oracle_rms > 0.0 else float("nan")
        metrics["policy_oracle_cosine"] = float(np.nanmean(np.asarray(action_cosine, dtype=float))) if action_cosine else float("nan")
    for name, values in q_acc.items():
        arr = np.asarray(values, dtype=float)
        if arr.size:
            q_values[f"q_{name}_mean"] = float(np.nanmean(arr))
            q_values[f"q_{name}_p90"] = float(np.nanpercentile(arr, 90))
    if q_rank_count > 0:
        q_values["oracle_rank_fraction"] = float(q_oracle_rank / q_rank_count)
    per_category: dict[str, dict[str, float]] = {}
    for category in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == category]
        per_category[category] = _category_metrics(idx, per_episode, per_episode_action)
    per_category["all"] = _category_metrics(list(range(int(episodes))), per_episode, per_episode_action)
    per_window = _window_rows(metadata, categories, per_episode, per_episode_action)
    return metrics, per_category, q_values, per_window


@torch.no_grad()
def _q_diagnostics(
    *,
    trainer: Trainer,
    env: TokamakMagneticControlEnv,
    obs: torch.Tensor,
    policy_action: torch.Tensor,
    oracle_action: torch.Tensor,
    critic_state,
    active: torch.Tensor,
) -> tuple[dict[str, list[float]], object]:
    critic_obs = env.critic_obs()
    zero = torch.zeros((env.batch_size, env.action_dim), dtype=torch.float32, device=obs.device)
    q_zero, _ = trainer.critic(critic_obs, zero, state=critic_state)
    q_policy, next_state = trainer.critic(critic_obs, policy_action, state=critic_state)
    q_oracle, _ = trainer.critic(critic_obs, oracle_action, state=critic_state)
    mask = active.detach().cpu().numpy().astype(bool)
    out = {
        "zero": q_zero.detach().cpu().numpy().reshape(-1)[mask].astype(float).tolist(),
        "policy": q_policy.detach().cpu().numpy().reshape(-1)[mask].astype(float).tolist(),
        "real_jdot": q_oracle.detach().cpu().numpy().reshape(-1)[mask].astype(float).tolist(),
    }
    return out, next_state


def _category_metrics(indices: list[int], per_episode: dict[str, list[list[float]]], per_episode_action: dict[str, list[list[float]]] | None = None) -> dict[str, float]:
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
    if per_episode_action:
        policy_rms = _episode_means(indices, per_episode_action.get("policy_action_rms", []))
        oracle_rms = _episode_means(indices, per_episode_action.get("oracle_action_rms", []))
        mae = _episode_means(indices, per_episode_action.get("policy_oracle_action_mae_mean", []))
        cosine = _episode_means(indices, per_episode_action.get("policy_oracle_cosine", []))
        out["oracle_action_rms"] = float(np.nanmean(oracle_rms)) if oracle_rms.size else float("nan")
        policy_action_rms = float(np.nanmean(policy_rms)) if policy_rms.size else float("nan")
        oracle_action_rms = float(np.nanmean(oracle_rms)) if oracle_rms.size else float("nan")
        if np.isfinite(policy_action_rms) and np.isfinite(oracle_action_rms) and oracle_action_rms > 0.0:
            out["policy_oracle_action_rms_ratio"] = float(policy_action_rms / oracle_action_rms)
        else:
            out["policy_oracle_action_rms_ratio"] = float("nan")
        out["policy_oracle_action_mae_mean"] = float(np.nanmean(mae)) if mae.size else float("nan")
        out["policy_oracle_cosine"] = float(np.nanmean(cosine)) if cosine.size else float("nan")
    return out


def _window_rows(metadata: list[dict[str, object]], categories: list[str], per_episode: dict[str, list[list[float]]], per_episode_action: dict[str, list[list[float]]] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, row in enumerate(metadata):
        out: dict[str, object] = {
            "window": int(i),
            "shot_id": row.get("shot_id", ""),
            "source_index": row.get("source_index", -1),
            "source_time_s": row.get("source_time_s", float("nan")),
            "category": categories[i] if i < len(categories) else "unknown",
        }
        for name, values_by_episode in per_episode.items():
            values = np.asarray(values_by_episode[i], dtype=float)
            out[name] = float(np.nanmean(values)) if values.size else float("nan")
            out[f"{name}_p90"] = float(np.nanpercentile(values, 90)) if values.size else float("nan")
        if per_episode_action:
            for name, values_by_episode in per_episode_action.items():
                if i >= len(values_by_episode):
                    continue
                values = np.asarray(values_by_episode[i], dtype=float)
                out[name] = float(np.nanmean(values)) if values.size else float("nan")
            policy_rms = float(out.get("policy_action_rms", float("nan")))
            oracle_rms = float(out.get("oracle_action_rms", float("nan")))
            if np.isfinite(policy_rms) and np.isfinite(oracle_rms) and oracle_rms > 0.0:
                out["policy_oracle_action_rms_ratio"] = float(policy_rms / oracle_rms)
        rows.append(out)
    return rows


def _episode_means(indices: list[int], values_by_episode: list[list[float]]) -> np.ndarray:
    values = []
    for i in indices:
        if i >= len(values_by_episode):
            continue
        vals = values_by_episode[i]
        if vals:
            values.append(float(np.nanmean(np.asarray(vals, dtype=float))))
    return np.asarray(values, dtype=float)


def _split_window_count(*, cfg, device: torch.device, seed: int) -> int:
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device=device, seed=int(seed))
    library = getattr(env, "_csv_initial_states", None)
    if library is None:
        raise ValueError("all-window oracle evaluation requires csv initial states")
    return len(library)


def _find_checkpoint_for_export(export_dir: Path) -> str | None:
    meta_path = export_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    root = export_dir.parent.parent
    candidates: list[Path] = []
    env_step = metadata.get("env_step")
    if env_step is not None:
        try:
            candidates.append(root / "checkpoints" / f"eval_step_{int(env_step):012d}.pt")
        except (TypeError, ValueError):
            pass
    candidates.extend([root / "checkpoints" / "best.pt", root / "checkpoints" / "latest.pt"])
    for path in candidates:
        if path.exists() or path.is_symlink():
            return str(path)
    return None


def _load_export_actor(trainer: Trainer, export_dir: Path) -> None:
    actor_path = export_dir / "actor.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"export actor does not exist: {actor_path}")
    data = torch.load(actor_path, map_location=trainer.device, weights_only=False)
    state = data.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"export actor is missing actor_state_dict: {actor_path}")
    trainer.actor.load_state_dict(state)


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
