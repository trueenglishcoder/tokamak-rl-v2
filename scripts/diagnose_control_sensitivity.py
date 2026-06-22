from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import torch

    from tokamak_rl_v2.config import load_experiment_config
    from tokamak_rl_v2.env import TokamakMagneticControlEnv

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_experiment_config(args.config)
    device = torch.device(args.device)
    env = TokamakMagneticControlEnv(cfg, batch_size=int(args.episodes), device=device, seed=int(args.seed))
    obs0 = env.reset()
    base_state = env.state_dict()
    schema = env.export_schema()
    actor = _load_actor(args.checkpoint, schema=schema, device=device) if args.checkpoint else None

    candidates = _action_candidates(env.action_dim, amplitudes=_parse_floats(args.amplitudes), random_count=int(args.random_candidates), seed=int(args.seed) + 17)
    policies: list[tuple[str, Callable[[torch.Tensor, int], torch.Tensor]]] = [
        ("zero", lambda obs, _step: torch.zeros((obs.shape[0], env.action_dim), dtype=torch.float32, device=device)),
    ]
    if actor is not None:
        policies.append(("actor", lambda obs, _step: actor.deterministic(obs)))
    for name, action_np in candidates:
        action = torch.as_tensor(action_np, dtype=torch.float32, device=device)
        policies.append((name, lambda obs, _step, a=action: a.reshape(1, -1).repeat(obs.shape[0], 1)))

    rows: list[dict[str, Any]] = []
    for name, policy in policies:
        env.load_state_dict(base_state)
        rows.append(_rollout(env, policy, name=name, horizon=int(args.horizon)))
        print(_format_row(rows[-1]), flush=True)

    ranked = sorted(rows, key=_sensitivity_score, reverse=True)
    _write_csv(out_dir / "control_sensitivity.csv", ranked)
    (out_dir / "control_sensitivity.json").write_text(json.dumps(_jsonable(ranked), indent=2), encoding="utf-8")
    (out_dir / "initial_observation_shape.txt").write_text(str(tuple(obs0.shape)), encoding="utf-8")
    print(out_dir / "control_sensitivity.csv")
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Probe plant/action sensitivity from identical reset states.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--amplitudes", default="0.1,0.25,0.5")
    ap.add_argument("--random-candidates", type=int, default=48)
    return ap


def _load_actor(checkpoint: str, *, schema: dict[str, Any], device):
    import torch

    from tokamak_rl_v2.networks import FeedForwardGaussianActor

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    network = ckpt.get("network", {}) if isinstance(ckpt, dict) else {}
    actor = FeedForwardGaussianActor(
        int(schema["obs_dim"]),
        int(schema["action_dim"]),
        int(network.get("hidden_dim", 256)),
        min_std=float(network.get("actor_min_std", 1.0e-4)),
        initial_std=float(network.get("actor_initial_std", 0.2)),
    ).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor


def _action_candidates(action_dim: int, *, amplitudes: list[float], random_count: int, seed: int) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for amp in amplitudes:
        for index in range(int(action_dim)):
            for sign in (-1.0, 1.0):
                action = np.zeros((int(action_dim),), dtype=np.float32)
                action[index] = float(sign * amp)
                out.append((f"coil{index}_{'p' if sign > 0 else 'm'}_{amp:g}", action))
    rng = np.random.default_rng(seed)
    for idx in range(max(0, int(random_count))):
        amp = float(rng.choice(np.asarray(amplitudes, dtype=float)))
        raw = rng.normal(size=(int(action_dim),))
        raw /= max(float(np.max(np.abs(raw))), 1.0e-12)
        out.append((f"random{idx:03d}_{amp:g}", (amp * raw).astype(np.float32)))
    return out


def _rollout(env, policy: Callable[[Any, int], Any], *, name: str, horizon: int) -> dict[str, Any]:
    import torch

    obs = env._obs_gpu() if env.config.sim.compute_backend == "gpu" else env._obs_cpu()
    values: dict[str, list[float]] = {}
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for step in range(max(1, int(horizon))):
        with torch.no_grad():
            action = policy(obs, step)
        out = env.step(action)
        comps = out.info.get("reward_components", {}) if isinstance(out.info, dict) else {}
        for key, value in comps.items():
            arr = np.asarray(value, dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                mean = float(np.mean(finite))
                values.setdefault(key, []).append(mean)
                if key not in first:
                    first[key] = mean
                last[key] = mean
        done = out.terminated | out.truncated
        obs = env.reset_indices(done) if bool(torch.any(done).item()) else out.obs
    row: dict[str, Any] = {"policy": name}
    for key, series in values.items():
        arr = np.asarray(series, dtype=float)
        row[key] = float(np.mean(arr))
        row[f"{key}_final"] = float(last.get(key, float("nan")))
        row[f"{key}_initial"] = float(first.get(key, float("nan")))
    row["score"] = _sensitivity_score(row)
    return row


def _sensitivity_score(row: dict[str, Any]) -> float:
    shape = _float(row.get("shape_error_mean_m"))
    ip = _float(row.get("ip_error_a"))
    current = _float(row.get("current_over_limit_a"))
    boundary = _float(row.get("boundary_found", 1.0))
    action = _float(row.get("action_rms", 0.0))
    score = 0.0
    if math.isfinite(shape):
        score -= 100.0 * shape / 0.03
    if math.isfinite(ip):
        score -= 40.0 * ip / 20_000.0
    if math.isfinite(current) and current > 0.0:
        score -= 100_000.0 + current
    if math.isfinite(boundary):
        score -= 100_000.0 * max(0.0, 0.999 - boundary)
    if math.isfinite(action) and action >= 0.5:
        score -= 1000.0
    return float(score)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in str(raw).split(",") if part.strip()]


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _format_row(row: dict[str, Any]) -> str:
    return "policy={policy} score={score:.3f} shape={shape} ip={ip} current={current} boundary={boundary}".format(
        policy=row.get("policy"),
        score=_float(row.get("score")),
        shape=row.get("shape_error_mean_m"),
        ip=row.get("ip_error_a"),
        current=row.get("current_over_limit_a"),
        boundary=row.get("boundary_found"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
