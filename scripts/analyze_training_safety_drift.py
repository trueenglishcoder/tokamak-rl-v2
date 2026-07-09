#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        raw = row.get(key, "")
        if raw == "":
            return default
        value = float(raw)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def _step(row: dict[str, str]) -> int:
    for key in ("step", "env_step", "global_step"):
        if row.get(key, "") != "":
            try:
                return int(float(row[key]))
            except Exception:
                return 0
    return 0


def _nearest(rows: list[dict[str, str]], target_step: int) -> dict[str, str] | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(_step(r) - int(target_step)))


def _corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize long-run safety drift from tokamak-rl-v2 CSV outputs.")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    health = _read_csv(run_dir / "replay_health.csv")
    rewards = _read_csv(run_dir / "reward_components.csv")
    losses = _read_csv(run_dir / "losses.csv")
    evals = _read_csv(run_dir / "eval_history.csv")

    lines: list[str] = []
    lines.append(f"run_dir={run_dir}")
    lines.append(f"row_counts replay_health={len(health)} rewards={len(rewards)} losses={len(losses)} eval={len(evals)}")
    lines.append("")
    lines.append("episode_length_crossings")
    for threshold in (99.5, 99.0, 98.0, 95.0, 90.0):
        hits = [r for r in health if _num(r, "replay_mean_episode_length") < threshold]
        if hits:
            r = hits[0]
            lines.append(
                f"  mean_len<{threshold}: step={_step(r)} mean={r.get('replay_mean_episode_length','')} min={r.get('replay_min_episode_length','')} max={r.get('replay_max_episode_length','')}"
            )
        else:
            lines.append(f"  mean_len<{threshold}: never")

    lines.append("")
    lines.append("key_points")
    for target in (20_000_000, 40_000_000, 60_000_000, 80_000_000, 100_000_000):
        h = _nearest(health, target)
        rw = _nearest(rewards, target)
        ev = _nearest(evals, target)
        lines.append(f"  target_step={target}")
        if h:
            lines.append(f"    health step={_step(h)} mean_len={h.get('replay_mean_episode_length','')} min_len={h.get('replay_min_episode_length','')}")
        if rw:
            keys = [
                "physical_cost",
                "current_over_limit_a",
                "current_margin_loss",
                "derivative_margin_loss",
                "current_usage_fraction",
                "derivative_usage_mean_fraction",
                "mean_jdot_bias_fraction",
                "terminated_current",
                "boundary_found",
                "max_current_fraction",
                "max_derivative_fraction",
            ]
            parts = " ".join(f"{k}={rw.get(k, '')}" for k in keys if k in rw)
            lines.append(f"    reward step={_step(rw)} {parts}")
        if ev:
            keys = [
                "mean_episode_completion",
                "current_over_limit_fraction_late",
                "current_margin_loss_late_max",
                "derivative_margin_loss_late_max",
                "max_current_fraction_late_max",
                "max_derivative_fraction_late_max",
                "selection_score",
            ]
            parts = " ".join(f"{k}={ev.get(k, '')}" for k in keys if k in ev)
            lines.append(f"    eval step={_step(ev)} {parts}")

    lines.append("")
    lines.append("correlations_with_replay_mean_episode_length")
    y = [_num(r, "replay_mean_episode_length") for r in health]
    scored: list[tuple[float, float, str, str]] = []
    for label, rows in (("reward", rewards), ("loss", losses), ("eval", evals)):
        if not rows:
            continue
        nearest = [_nearest(rows, _step(h)) for h in health]
        keys: set[str] = set()
        for row in rows[:1]:
            for key, value in row.items():
                if key in {"step", "env_step", "global_step"}:
                    continue
                try:
                    float(value)
                except Exception:
                    continue
                keys.add(key)
        for key in keys:
            xs = [_num(row, key) if row is not None else float("nan") for row in nearest]
            c = _corr(xs, y)
            if math.isfinite(c):
                scored.append((abs(c), c, label, key))
    for _abs_c, c, label, key in sorted(scored, reverse=True)[:40]:
        lines.append(f"  {label}.{key}: {c:.4f}")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
