from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.config.schema import ExperimentConfig, RewardConfig
from tokamak_rl_v2.training.trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base = _apply_overrides(load_experiment_config(args.config), args)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = list(_candidate_rewards(base.reward, args))
    if args.search_seed is not None:
        rng = np.random.default_rng(int(args.search_seed))
        order = rng.permutation(len(candidates))
        candidates = [candidates[int(i)] for i in order]
    if args.max_candidates is not None:
        candidates = candidates[: int(args.max_candidates)]
    manifest = {
        "config": str(Path(args.config).resolve()),
        "candidate_count": len(candidates),
        "base_reward": asdict(base.reward),
        "reward_fields": ["shape_good_m", "shape_bad_m", "ip_good_a", "ip_bad_a", "current_good_a", "current_bad_a", "terminal_reward", "reward_scale"],
    }
    (out / "search_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows: list[dict[str, object]] = []
    for idx, reward in enumerate(candidates):
        cand_dir = out / f"candidate_{idx:04d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        cfg = replace(base, reward=reward, training=replace(base.training, output_dir=cand_dir))
        (cand_dir / "reward_config.json").write_text(json.dumps(asdict(reward), indent=2), encoding="utf-8")
        row: dict[str, object] = {"candidate": idx, "status": "pending", "candidate_dir": str(cand_dir), **{f"reward.{k}": v for k, v in asdict(reward).items()}}
        print(f"[{idx + 1}/{len(candidates)}] reward={_reward_label(reward)}", flush=True)
        if args.dry_run:
            row.update({"status": "dry_run", "score": ""})
            rows.append(row)
            _write_results(out / "results.csv", rows)
            continue
        wandb_run = None
        try:
            if args.wandb and args.wandb_mode != "disabled":
                import wandb
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=f"{args.wandb_name or out.name}_candidate_{idx:04d}",
                    group=args.wandb_group or out.name,
                    mode=args.wandb_mode,
                    config={"experiment": cfg.name, "candidate": idx, "reward": asdict(reward)},
                    reinit=True,
                )
            result = Trainer(cfg, steps=args.steps, num_envs=args.num_envs, device=args.device, output_dir=cand_dir, wandb_run=wandb_run).train()
            summary = _summarize_reward_components(cand_dir / "reward_components.csv")
            score = float(result.get("best_eval", -float("inf")))
            row.update({"status": "ok", "score": score, **{f"metric.{k}": v for k, v in result.items()}, **summary})
        except Exception as exc:
            row.update({"status": "error", "score": "", "error": repr(exc)})
            (cand_dir / "error.txt").write_text(repr(exc), encoding="utf-8")
            if not args.continue_on_error:
                rows.append(row)
                _write_results(out / "results.csv", rows)
                raise
        finally:
            if wandb_run is not None:
                wandb_run.finish()
        rows.append(row)
        _write_results(out / "results.csv", rows)
    _write_results(out / "results.csv", rows)
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Search T15 static-boundary reward transform settings.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default="outputs/reward_search_t15_static_boundary")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    ap.add_argument("--sim-compute-backend", choices=("cpu", "gpu"), default=None)
    ap.add_argument("--sim-gpu-device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--unroll-length", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--critic-hidden-dim", type=int, default=None)
    ap.add_argument("--critic-mlp-hidden-dim", type=int, default=None)
    ap.add_argument("--rollout-chunk-length", type=int, default=None)
    ap.add_argument("--updates-per-rollout-chunk", type=int, default=None)
    ap.add_argument("--action-samples", type=int, default=None)
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
    if any(v is not None for v in (args.batch_size, args.unroll_length, args.rollout_chunk_length, args.updates_per_rollout_chunk, args.action_samples)):
        cfg = replace(cfg, learner=replace(cfg.learner, batch_size=args.batch_size or cfg.learner.batch_size, unroll_length=args.unroll_length or cfg.learner.unroll_length, rollout_chunk_length=args.rollout_chunk_length or cfg.learner.rollout_chunk_length, updates_per_rollout_chunk=args.updates_per_rollout_chunk or cfg.learner.updates_per_rollout_chunk, action_samples=args.action_samples or cfg.learner.action_samples))
    if any(v is not None for v in (args.hidden_dim, args.critic_hidden_dim, args.critic_mlp_hidden_dim)):
        cfg = replace(cfg, network=replace(cfg.network, hidden_dim=args.hidden_dim or cfg.network.hidden_dim, critic_hidden_dim=args.critic_hidden_dim or cfg.network.critic_hidden_dim, critic_mlp_hidden_dim=args.critic_mlp_hidden_dim or cfg.network.critic_mlp_hidden_dim))
    if any(v is not None for v in (args.checkpoint_interval_steps, args.eval_interval_steps, args.eval_episodes, args.eval_max_steps, args.actor_workers)):
        cfg = replace(cfg, training=replace(cfg.training, checkpoint_interval_steps=args.checkpoint_interval_steps or cfg.training.checkpoint_interval_steps, eval_interval_steps=args.eval_interval_steps or cfg.training.eval_interval_steps, eval_episodes=args.eval_episodes or cfg.training.eval_episodes, eval_max_steps=args.eval_max_steps or cfg.training.eval_max_steps, actor_workers=args.actor_workers or cfg.training.actor_workers))
    return cfg


def _candidate_rewards(base: RewardConfig, args: argparse.Namespace) -> Iterable[RewardConfig]:
    names = [
        ("shape_good_m", args.shape_good_values),
        ("shape_bad_m", args.shape_bad_values),
        ("ip_good_a", args.ip_good_values),
        ("ip_bad_a", args.ip_bad_values),
        ("current_good_a", args.current_good_values),
        ("current_bad_a", args.current_bad_values),
        ("terminal_reward", args.terminal_reward_values),
        ("reward_scale", args.reward_scale_values),
    ]
    values = [(name, _float_values(raw, getattr(base, name))) for name, raw in names]
    for combo in itertools.product(*(vals for _name, vals in values)):
        data = asdict(base)
        for (name, _vals), value in zip(values, combo):
            data[name] = float(value)
        _validate_reward_candidate(data)
        yield RewardConfig(**data)


def _float_values(raw: str | None, default: float) -> list[float]:
    if raw is None or str(raw).strip() == "":
        return [float(default)]
    out = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    if not out:
        raise ValueError("value list must not be empty")
    return out


def _validate_reward_candidate(data: dict[str, float]) -> None:
    if data["shape_bad_m"] <= data["shape_good_m"]:
        raise ValueError("shape_bad_m must be greater than shape_good_m")
    if data["ip_bad_a"] <= data["ip_good_a"]:
        raise ValueError("ip_bad_a must be greater than ip_good_a")
    if data["current_bad_a"] <= data["current_good_a"]:
        raise ValueError("current_bad_a must be greater than current_good_a")
    if data["reward_scale"] <= 0.0:
        raise ValueError("reward_scale must be positive")


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


def _reward_label(reward: RewardConfig) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in asdict(reward).items())


if __name__ == "__main__":
    raise SystemExit(main())
