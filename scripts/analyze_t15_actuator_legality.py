#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PFC_CURRENT_LIMITS = [198365.72, 345047.416, 164882.78, 204574.776, 606866.803, 2138757.022]
SOL_CURRENT_LIMITS = [4760890.355, 13912524.5, 4771978.708]
PFC_DERIV_LIMIT = 5070038.4
SOL_DERIV_LIMIT = 20950244.0


def _read_coil_csv(path: Path) -> tuple[list[float], list[list[float]], list[list[float]]]:
    times: list[float] = []
    sol: list[list[float]] = []
    pfc: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) < 10:
                continue
            try:
                values = [float(value) for value in row[:10]]
            except ValueError:
                continue
            if not all(math.isfinite(value) for value in values):
                continue
            times.append(values[0])
            sol.append(values[1:4])
            pfc.append(values[4:10])
    return times, sol, pfc


def _max_abs(values: list[list[float]], limits: list[float]) -> tuple[list[float], list[float]]:
    max_values = [0.0 for _ in limits]
    for row in values:
        for idx, value in enumerate(row):
            max_values[idx] = max(max_values[idx], abs(float(value)))
    usage = [max_values[idx] / limits[idx] for idx in range(len(limits))]
    return max_values, usage


def _max_abs_derivative(times: list[float], values: list[list[float]], *, min_dt_s: float, limit: float) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    max_values = [0.0 for _ in values[0]]
    for prev_time, time, prev_row, row in zip(times[:-1], times[1:], values[:-1], values[1:]):
        dt = float(time) - float(prev_time)
        if not math.isfinite(dt) or dt < min_dt_s:
            continue
        for idx, (prev_value, value) in enumerate(zip(prev_row, row)):
            deriv = abs((float(value) - float(prev_value)) / dt)
            if math.isfinite(deriv):
                max_values[idx] = max(max_values[idx], deriv)
    usage = [value / limit for value in max_values]
    return max_values, usage


def analyze(root: Path, *, min_dt_s: float) -> dict[str, object]:
    coil_dir = root / "coils"
    paths = sorted(coil_dir.glob("t15md_*_coils.csv"))
    if not paths:
        raise FileNotFoundError(f"no coil CSV files found under {coil_dir}")

    pfc_current_max = [0.0 for _ in PFC_CURRENT_LIMITS]
    sol_current_max = [0.0 for _ in SOL_CURRENT_LIMITS]
    pfc_deriv_max = [0.0 for _ in PFC_CURRENT_LIMITS]
    sol_deriv_max = [0.0 for _ in SOL_CURRENT_LIMITS]
    shot_rows: list[dict[str, object]] = []

    for path in paths:
        times, sol, pfc = _read_coil_csv(path)
        shot_id = path.stem.replace("t15md_", "").replace("_coils", "")
        pfc_abs, pfc_usage = _max_abs(pfc, PFC_CURRENT_LIMITS)
        sol_abs, sol_usage = _max_abs(sol, SOL_CURRENT_LIMITS)
        pfc_deriv, pfc_deriv_usage = _max_abs_derivative(times, pfc, min_dt_s=min_dt_s, limit=PFC_DERIV_LIMIT)
        sol_deriv, sol_deriv_usage = _max_abs_derivative(times, sol, min_dt_s=min_dt_s, limit=SOL_DERIV_LIMIT)

        pfc_current_max = [max(a, b) for a, b in zip(pfc_current_max, pfc_abs)]
        sol_current_max = [max(a, b) for a, b in zip(sol_current_max, sol_abs)]
        pfc_deriv_max = [max(a, b) for a, b in zip(pfc_deriv_max, pfc_deriv)]
        sol_deriv_max = [max(a, b) for a, b in zip(sol_deriv_max, sol_deriv)]
        shot_rows.append(
            {
                "shot_id": shot_id,
                "rows": len(times),
                "max_pfc_current_usage": max(pfc_usage) if pfc_usage else 0.0,
                "max_sol_current_usage": max(sol_usage) if sol_usage else 0.0,
                "max_pfc_derivative_usage": max(pfc_deriv_usage) if pfc_deriv_usage else 0.0,
                "max_sol_derivative_usage": max(sol_deriv_usage) if sol_deriv_usage else 0.0,
            }
        )

    pfc_current_usage = [value / limit for value, limit in zip(pfc_current_max, PFC_CURRENT_LIMITS)]
    sol_current_usage = [value / limit for value, limit in zip(sol_current_max, SOL_CURRENT_LIMITS)]
    pfc_deriv_usage = [value / PFC_DERIV_LIMIT for value in pfc_deriv_max]
    sol_deriv_usage = [value / SOL_DERIV_LIMIT for value in sol_deriv_max]
    return {
        "root": str(root),
        "min_derivative_dt_s": min_dt_s,
        "shots": shot_rows,
        "limits": {
            "pfc_current": PFC_CURRENT_LIMITS,
            "sol_current": SOL_CURRENT_LIMITS,
            "pfc_derivative": PFC_DERIV_LIMIT,
            "sol_derivative": SOL_DERIV_LIMIT,
        },
        "observed_max": {
            "pfc_current": pfc_current_max,
            "sol_current": sol_current_max,
            "pfc_derivative": pfc_deriv_max,
            "sol_derivative": sol_deriv_max,
        },
        "max_usage": {
            "pfc_current": pfc_current_usage,
            "sol_current": sol_current_usage,
            "pfc_derivative": pfc_deriv_usage,
            "sol_derivative": sol_deriv_usage,
            "overall": max(pfc_current_usage + sol_current_usage + pfc_deriv_usage + sol_deriv_usage),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check real T15 actuation against production legality limits.")
    parser.add_argument("--root", type=Path, default=Path("../tokamak-sim/data/t15_data_new"))
    parser.add_argument("--min-derivative-dt-s", type=float, default=0.0005)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    result = analyze(args.root, min_dt_s=float(args.min_derivative_dt_s))
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if float(result["max_usage"]["overall"]) <= 1.0 + 1.0e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
