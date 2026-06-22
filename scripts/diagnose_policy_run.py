from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_output = Path(args.run_output).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run_output / "checkpoints" / "best.pt"
    config_path = Path(args.config).resolve() if args.config else run_output / "config_snapshot.json"
    out = Path(args.out).resolve() if args.out else run_output / "diagnostics" / "policy_run_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "run_output": str(run_output),
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "validation": _load_json(run_output / "policy_validation.json"),
        "metrics": _load_json(run_output / "metrics.json"),
        "loss_trends": _csv_trends(run_output / "losses.csv"),
        "reward_trends": _csv_trends(run_output / "reward_components.csv"),
        "eval_trends": _csv_trends(run_output / "eval_history.csv"),
    }
    if checkpoint.exists() and config_path.exists():
        report["checkpoint_diagnostics"] = _checkpoint_diagnostics(
            config_path=config_path,
            checkpoint=checkpoint,
            device=args.device,
            batches=int(args.batches),
            batch_size=int(args.batch_size),
            sequence_length=int(args.sequence_length),
        )
    else:
        report["checkpoint_diagnostics"] = {"status": "missing_checkpoint_or_config"}
    out.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print(out)
    _print_report(report)
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Diagnose a trained policy run: eval drift, reward trends, and critic/replay quality.")
    ap.add_argument("--run-output", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--sequence-length", type=int, default=64)
    return ap


def _checkpoint_diagnostics(*, config_path: Path, checkpoint: Path, device: str, batches: int, batch_size: int, sequence_length: int) -> dict[str, Any]:
    import torch

    from tokamak_rl_v2.config import load_experiment_config
    from tokamak_rl_v2.training.trainer import Trainer

    cfg = load_experiment_config(config_path)
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    step = int((data.get("training_state") or {}).get("step", (data.get("metadata") or {}).get("step", 0)))
    trainer = Trainer(cfg, steps=max(step + 1, int(cfg.training.steps) + 1), num_envs=int(cfg.training.num_envs), device=device, output_dir=Path("/tmp/tokamak_rl_v2_diagnostics"), resume_checkpoint=None)
    trainer._load_checkpoint(checkpoint, restore_env=False)
    replay = trainer.replay
    if not replay.ready(sequence_length, batch_size):
        return {"status": "replay_not_ready", "replay_size": replay.size}

    rows: list[dict[str, float]] = []
    for _ in range(max(1, int(batches))):
        seq = replay.sample(batch_size=batch_size, sequence_length=sequence_length)
        rows.append(_critic_batch_metrics(trainer, seq))
    return {"status": "ok", "replay_size": replay.size, "step": step, **_mean_dicts(rows)}


def _critic_batch_metrics(trainer: Trainer, seq) -> dict[str, float]:
    import torch

    mask = seq.mask.to(dtype=torch.float32)
    with torch.no_grad():
        q, _ = trainer.critic(seq.obs, seq.action, mask=mask)
        B, T, O = seq.next_obs.shape
        next_action = trainer.target_actor.deterministic(seq.next_obs.reshape(B * T, O)).reshape(B, T, -1)
        q_next, _ = trainer.target_critic(seq.next_obs, next_action, mask=mask)
        target = seq.reward + seq.discount * (~seq.done).to(torch.float32) * q_next
        returns = _discounted_returns(seq.reward, seq.discount, seq.done, mask)
        actor_action = trainer.actor.deterministic(seq.obs.reshape(B * T, O)).reshape(B, T, -1)
        q_actor, _ = trainer.critic(seq.obs, actor_action, mask=mask)
        denom = torch.clamp(mask.sum(), min=1.0)
        td_error = (q - target) * mask
        actor_adv = (q_actor - q) * mask
    q_np = _masked_np(q, mask)
    target_np = _masked_np(target, mask)
    returns_np = _masked_np(returns, mask)
    actor_adv_np = _masked_np(actor_adv, mask)
    return {
        "critic_td_mse": float(torch.sum(td_error.pow(2)) / denom),
        "critic_td_mae": float(torch.sum(torch.abs(td_error)) / denom),
        "q_mean": _mean(q_np),
        "q_std": _std(q_np),
        "target_mean": _mean(target_np),
        "target_std": _std(target_np),
        "q_target_corr": _corr(q_np, target_np),
        "q_return_corr": _corr(q_np, returns_np),
        "actor_minus_replay_q_mean": _mean(actor_adv_np),
        "actor_minus_replay_q_std": _std(actor_adv_np),
    }


def _discounted_returns(reward, discount, done, mask):
    import torch

    out = torch.zeros_like(reward)
    running = torch.zeros((reward.shape[0],), dtype=reward.dtype, device=reward.device)
    for t in range(int(reward.shape[1]) - 1, -1, -1):
        running = reward[:, t] + discount[:, t] * (~done[:, t]).to(reward.dtype) * running
        out[:, t] = running * mask[:, t]
    return out


def _csv_trends(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing"}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"status": "empty"}
    keys = [key for key in rows[0] if key not in {"step", "env_step"}]
    out: dict[str, Any] = {"status": "ok", "rows": len(rows)}
    thirds = _thirds(rows)
    for label, subset in thirds.items():
        for key in keys:
            values = [_float(row.get(key)) for row in subset]
            finite = [value for value in values if math.isfinite(value)]
            if finite:
                out[f"{label}.{key}.mean"] = float(np.mean(finite))
    return out


def _thirds(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    n = len(rows)
    a = max(1, n // 3)
    return {"first": rows[:a], "middle": rows[a : 2 * a] or rows, "last": rows[-a:]}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "error": repr(exc)}
    return data if isinstance(data, dict) else {"value": data}


def _masked_np(value: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    arr = value.detach().cpu().numpy().reshape(-1)
    m = mask.detach().cpu().numpy().reshape(-1).astype(bool)
    return arr[m]


def _mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if float(np.std(a)) <= 1.0e-12 or float(np.std(b)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _mean(a: np.ndarray) -> float:
    return float(np.mean(a)) if a.size else float("nan")


def _std(a: np.ndarray) -> float:
    return float(np.std(a)) if a.size else float("nan")


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print_report(report: dict[str, Any]) -> None:
    validation = report.get("validation", {})
    actor = validation.get("actor_eval", {}) if isinstance(validation, dict) else {}
    print("actor_eval", {k: actor.get(k) for k in ("shape_error_mean_m", "ip_error_a", "current_over_limit_a_max", "boundary_found", "action_rms")})
    print("checkpoint_diagnostics", report.get("checkpoint_diagnostics", {}))


if __name__ == "__main__":
    raise SystemExit(main())
