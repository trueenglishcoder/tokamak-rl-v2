from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


SHOT_RE = re.compile(r"t15md_(\d+)_ip\.csv$")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build aggregate T15 Ip bounds/rates for T15 reference diagnostics.")
    ap.add_argument("--data-root", default="../tokamak-sim/data/t15_data_new")
    ap.add_argument("--ip-glob", default="ip/t15md_*_ip.csv")
    ap.add_argument("--delimiter", default=";")
    ap.add_argument("--smooth-window", type=int, default=11)
    ap.add_argument("--out", default="data/processed/t15_reference_limits.json")
    args = ap.parse_args(argv)

    root = Path(args.data_root).resolve()
    ip_values: list[np.ndarray] = []
    step_rates: list[np.ndarray] = []
    durations: dict[str, float] = {}
    for path in sorted(root.glob(str(args.ip_glob))):
        match = SHOT_RE.search(path.name)
        if not match:
            continue
        shot_id = match.group(1)
        arr = _load_two_column(path, delimiter=str(args.delimiter))
        t = arr[:, 0]
        ip = _smooth(arr[:, 1], window=int(args.smooth_window))
        _validate_time(t, path)
        finite_ip = np.isfinite(ip)
        ip_values.append(ip[finite_ip])
        step_dt = np.diff(t)
        step_dipdt = np.diff(ip) / step_dt
        step_valid = np.isfinite(step_dipdt) & (step_dt >= 5.0e-4)
        step_rates.append(step_dipdt[step_valid])
        durations[shot_id] = float(t[-1] - t[0])
    if not ip_values:
        raise ValueError(f"no Ip CSV files matched under {root}")
    ip_all = np.concatenate(ip_values)
    step_rate_all = np.concatenate(step_rates)
    positive = step_rate_all[step_rate_all > 0.0]
    negative = -step_rate_all[step_rate_all < 0.0]
    if ip_all.size < 1000:
        raise ValueError(f"reference-limit build needs at least 1000 samples, got {ip_all.size}")
    if positive.size == 0 or negative.size == 0:
        raise ValueError("reference-limit build needs both positive and negative Ip rates")
    positive_p95 = float(np.nanpercentile(positive, 95.0))
    positive_p99 = float(np.nanpercentile(positive, 99.0))
    negative_p95 = float(np.nanpercentile(negative, 95.0))
    negative_p99 = float(np.nanpercentile(negative, 99.0))
    positive_ramp = positive[positive >= 0.1 * positive_p95]
    negative_ramp = negative[negative >= 0.1 * negative_p95]
    if positive_ramp.size == 0 or negative_ramp.size == 0:
        raise ValueError("reference-limit build could not identify robust Ip ramp portions")
    out = {
        "source_layout": "split_t15_data_new",
        "source_root": str(root),
        "shot_ids": sorted(durations),
        "shot_count": len(durations),
        "sample_count": int(ip_all.size),
        "ip_min_a": float(np.nanmin(ip_all)),
        "ip_max_a": float(np.nanmax(ip_all)),
        "ip_p01": float(np.nanpercentile(ip_all, 1.0)),
        "ip_p99": float(np.nanpercentile(ip_all, 99.0)),
        "ip_p01_a": float(np.nanpercentile(ip_all, 1.0)),
        "ip_p99_a": float(np.nanpercentile(ip_all, 99.0)),
        "positive_dipdt_p95_a_per_s": positive_p95,
        "positive_dipdt_p99_a_per_s": positive_p99,
        "negative_dipdt_abs_p95_a_per_s": negative_p95,
        "negative_dipdt_abs_p99_a_per_s": negative_p99,
        "positive_dip_dt_p95_a_per_s": positive_p95,
        "positive_dip_dt_p99_a_per_s": positive_p99,
        "negative_dip_dt_abs_p95_a_per_s": negative_p95,
        "negative_dip_dt_abs_p99_a_per_s": negative_p99,
        "ramp_mean_threshold_fraction_of_p95": 0.1,
        "positive_ramp_mean_a_per_s": float(np.nanmean(positive_ramp)),
        "negative_ramp_abs_mean_a_per_s": float(np.nanmean(negative_ramp)),
        "positive_dipdt_ramp_mean_a_per_s": float(np.nanmean(positive_ramp)),
        "negative_dipdt_abs_ramp_mean_a_per_s": float(np.nanmean(negative_ramp)),
        "duration_s_by_shot": durations,
        "duration_s_min": float(min(durations.values())),
        "duration_s_max": float(max(durations.values())),
    }
    if not (out["ip_p99_a"] > out["ip_p01_a"] > 0.0):
        raise ValueError("computed Ip p01/p99 are invalid")
    if out["positive_dipdt_p95_a_per_s"] <= 0.0 or out["negative_dipdt_abs_p95_a_per_s"] <= 0.0:
        raise ValueError("computed Ip rate limits are invalid")
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(target)
    return 0


def _load_two_column(path: Path, *, delimiter: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=delimiter, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two numeric columns")
    arr = arr[:, :2]
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path} contains non-finite values")
    return arr


def _validate_time(t: np.ndarray, path: Path) -> None:
    if t.ndim != 1 or t.size < 3 or np.any(np.diff(t) <= 0.0):
        raise ValueError(f"{path} time column must be strictly increasing")


def _smooth(x: np.ndarray, *, window: int) -> np.ndarray:
    w = max(1, int(window))
    if w <= 1:
        return np.asarray(x, dtype=float)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.pad(np.asarray(x, dtype=float), (pad, pad), mode="edge")
    kernel = np.ones((w,), dtype=float) / float(w)
    return np.convolve(padded, kernel, mode="valid")


if __name__ == "__main__":
    raise SystemExit(main())
