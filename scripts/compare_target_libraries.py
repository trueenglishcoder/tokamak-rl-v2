#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TARGET_FILENAMES = (
    "t15_replay_window_oracle_targets.npz",
    "t15_feasible_generated_trim50_idealized_0p1s_targets.npz",
)


@dataclass(frozen=True, slots=True)
class TargetLibrary:
    name: str
    target_path: Path
    initial_path: Path | None
    schema: str
    ip: np.ndarray
    radii: np.ndarray
    shot_id: np.ndarray
    split: np.ndarray
    source_index: np.ndarray
    time_s: np.ndarray
    params: np.ndarray | None
    currents: np.ndarray | None
    mode: np.ndarray | None
    current_limits: np.ndarray | None
    jdot: np.ndarray | None
    normalized_action: np.ndarray | None
    derivative_limits: np.ndarray | None
    current_order: str | None
    derivative_order: str | None

    @property
    def row_count(self) -> int:
        return int(self.ip.shape[0])

    @property
    def steps(self) -> int:
        return int(self.ip.shape[1] - 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare replay-window target libraries. Intended to diagnose why "
            "idealized/generated targets train differently from the old real replay-window pipeline."
        )
    )
    parser.add_argument("--reference-name", default="old_real")
    parser.add_argument("--reference-target", type=Path, required=True)
    parser.add_argument("--reference-initial", type=Path)
    parser.add_argument("--reference-config", type=Path)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--candidate-target", type=Path, required=True)
    parser.add_argument("--candidate-initial", type=Path)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--reference-mode", help="Optional mode filter, e.g. real or perturbed.")
    parser.add_argument("--candidate-mode", help="Optional mode filter, e.g. real or perturbed.")
    parser.add_argument("--reference-split", help="Optional split filter, e.g. train or holdout.")
    parser.add_argument("--candidate-split", help="Optional split filter, e.g. train or holdout.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nearest-chunk", type=int, default=256)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = load_library(
        name=str(args.reference_name),
        target_path=args.reference_target,
        initial_path=args.reference_initial,
        config_path=args.reference_config,
        dt=float(args.dt),
    )
    reference = filter_library(reference, mode=args.reference_mode, split=args.reference_split)
    candidate = load_library(
        name=str(args.candidate_name),
        target_path=args.candidate_target,
        initial_path=args.candidate_initial,
        config_path=args.candidate_config,
        dt=float(args.dt),
    )
    candidate = filter_library(candidate, mode=args.candidate_mode, split=args.candidate_split)
    if reference.steps != candidate.steps:
        raise ValueError(f"step count mismatch: {reference.name}={reference.steps}, {candidate.name}={candidate.steps}")
    if reference.radii.shape[2] != candidate.radii.shape[2]:
        raise ValueError(
            f"angle count mismatch: {reference.name}={reference.radii.shape[2]}, "
            f"{candidate.name}={candidate.radii.shape[2]}"
        )

    ref_metrics = compute_metrics(reference, dt=float(args.dt))
    cand_metrics = compute_metrics(candidate, dt=float(args.dt))
    nearest = nearest_window_distances(reference, candidate, chunk_size=int(args.nearest_chunk))
    cand_metrics["nearest_reference_window_distance"] = nearest

    summary_rows = []
    for lib, metrics in ((reference, ref_metrics), (candidate, cand_metrics)):
        for metric, values in sorted(metrics.items()):
            summary_rows.append(metric_summary_row(lib.name, metric, values))

    summary_csv = out_dir / "target_library_metric_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "metric", "count", "mean", "std", "p50", "p90", "p99", "max"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "reference": library_metadata(reference),
        "candidate": library_metadata(candidate),
        "metric_summary": summary_rows,
        "ratios": ratio_summary(reference.name, candidate.name, ref_metrics, cand_metrics),
    }
    summary_json = out_dir / "target_library_comparison_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    report = out_dir / "target_library_comparison_report.md"
    report.write_text(
        render_report(reference, candidate, summary_rows, payload["ratios"]),
        encoding="utf-8",
    )

    if bool(args.plots):
        write_plots(reference, candidate, ref_metrics, cand_metrics, out_dir=out_dir)

    print(report)
    return 0


def load_library(
    *,
    name: str,
    target_path: Path,
    initial_path: Path | None,
    config_path: Path | None,
    dt: float,
) -> TargetLibrary:
    target = resolve_target_path(target_path)
    with np.load(target, allow_pickle=False) as data:
        keys = set(data.files)
        schema = _schema_to_str(data["schema"]) if "schema" in keys else "unknown"
        if "ip_target" in keys:
            ip = np.asarray(data["ip_target"], dtype=np.float64)
            radii = np.asarray(data["boundary_radii"], dtype=np.float64)
            params = None
            shot_id = _strings(data["shot_id"]) if "shot_id" in keys else np.full(ip.shape[0], "")
            split = _strings(data["split"]) if "split" in keys else np.full(ip.shape[0], "")
            source_index = np.asarray(data["source_index"], dtype=np.int64) if "source_index" in keys else np.arange(ip.shape[0])
            time_s = np.asarray(data["time_s"], dtype=np.float64) if "time_s" in keys else np.full(ip.shape[0], np.nan)
            mode = _strings(data["mode"]) if "mode" in keys else None
            action = np.asarray(data["real_jdot_action"], dtype=np.float64) if "real_jdot_action" in keys else None
            derivative_limits = np.asarray(data["derivative_limits"], dtype=np.float64) if "derivative_limits" in keys else None
            current_limits = np.asarray(data["current_limits"], dtype=np.float64) if "current_limits" in keys else None
            currents = reconstruct_currents_from_initial_and_actions(
                target_data=data,
                initial_path=initial_path,
                normalized_action=action,
                derivative_limits=derivative_limits,
                dt=float(dt),
            )
            jdot = None
            if action is not None and derivative_limits is not None:
                jdot = action * derivative_limits.reshape(1, 1, -1)
            current_order = "pfc_sol" if currents is not None else None
            derivative_order = "pfc_sol" if derivative_limits is not None else None
        elif "ip_ref" in keys:
            ip = np.asarray(data["ip_ref"], dtype=np.float64)
            radii = np.asarray(data["radii_ref"], dtype=np.float64)
            params = np.asarray(data["params_ref"], dtype=np.float64) if "params_ref" in keys else None
            shot_id = _strings(data["shot_id"]) if "shot_id" in keys else np.full(ip.shape[0], "")
            split = _strings(data["split"]) if "split" in keys else np.full(ip.shape[0], "")
            source_index = np.asarray(data["source_index"], dtype=np.int64) if "source_index" in keys else np.arange(ip.shape[0])
            time_s = np.asarray(data["time_s"], dtype=np.float64) if "time_s" in keys else np.full(ip.shape[0], np.nan)
            mode = _strings(data["mode"]) if "mode" in keys else None
            currents = np.asarray(data["coil_witness"], dtype=np.float64) if "coil_witness" in keys else None
            current_limits, derivative_limits = limits_from_config(config_path, current_order="sol_pfc")
            jdot = np.diff(currents, axis=1) / float(dt) if currents is not None else None
            action = None
            if jdot is not None and derivative_limits is not None:
                action = jdot / derivative_limits.reshape(1, 1, -1)
            current_order = "sol_pfc" if currents is not None else None
            derivative_order = "sol_pfc" if derivative_limits is not None else None
        else:
            raise ValueError(f"{target} is not a recognized target library; keys={sorted(keys)}")

    validate_shapes(name=name, ip=ip, radii=radii, params=params, currents=currents, jdot=jdot, action=action)
    return TargetLibrary(
        name=str(name),
        target_path=target,
        initial_path=None if initial_path is None else initial_path.expanduser().resolve(),
        schema=schema,
        ip=ip,
        radii=radii,
        shot_id=shot_id,
        split=split,
        source_index=source_index,
        time_s=time_s,
        params=params,
        currents=currents,
        mode=mode,
        current_limits=current_limits,
        jdot=jdot,
        normalized_action=action,
        derivative_limits=derivative_limits,
        current_order=current_order,
        derivative_order=derivative_order,
    )


def filter_library(lib: TargetLibrary, *, mode: str | None, split: str | None) -> TargetLibrary:
    mask = np.ones((lib.row_count,), dtype=bool)
    suffix: list[str] = []
    if split is not None:
        split_s = str(split)
        mask &= lib.split.astype(str) == split_s
        suffix.append(f"split={split_s}")
    if mode is not None:
        if lib.mode is None:
            raise ValueError(f"{lib.name} has no mode array; cannot filter mode={mode}")
        mode_s = str(mode)
        mask &= lib.mode.astype(str) == mode_s
        suffix.append(f"mode={mode_s}")
    if np.all(mask):
        return lib
    if not np.any(mask):
        raise ValueError(f"{lib.name} filter matched zero rows: {', '.join(suffix)}")
    return TargetLibrary(
        name=lib.name if not suffix else f"{lib.name} ({', '.join(suffix)})",
        target_path=lib.target_path,
        initial_path=lib.initial_path,
        schema=lib.schema,
        ip=lib.ip[mask],
        radii=lib.radii[mask],
        shot_id=lib.shot_id[mask],
        split=lib.split[mask],
        source_index=lib.source_index[mask],
        time_s=lib.time_s[mask],
        params=None if lib.params is None else lib.params[mask],
        currents=None if lib.currents is None else lib.currents[mask],
        mode=None if lib.mode is None else lib.mode[mask],
        current_limits=lib.current_limits,
        jdot=None if lib.jdot is None else lib.jdot[mask],
        normalized_action=None if lib.normalized_action is None else lib.normalized_action[mask],
        derivative_limits=lib.derivative_limits,
        current_order=lib.current_order,
        derivative_order=lib.derivative_order,
    )


def resolve_target_path(path: Path) -> Path:
    p = path.expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        for name in TARGET_FILENAMES:
            candidate = p / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"target library not found: {path}")


def _schema_to_str(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return ",".join(str(v) for v in arr.reshape(-1).tolist())


def _strings(value: np.ndarray) -> np.ndarray:
    return np.asarray(value).astype(str)


def reconstruct_currents_from_initial_and_actions(
    *,
    target_data: Any,
    initial_path: Path | None,
    normalized_action: np.ndarray | None,
    derivative_limits: np.ndarray | None,
    dt: float,
) -> np.ndarray | None:
    if normalized_action is None or derivative_limits is None:
        return None
    if "pfc0" in target_data.files and "sol0" in target_data.files:
        pfc0 = np.asarray(target_data["pfc0"], dtype=np.float64)
        sol0 = np.asarray(target_data["sol0"], dtype=np.float64)
    elif initial_path is not None and initial_path.exists():
        with np.load(initial_path, allow_pickle=False) as init:
            pfc0 = np.asarray(init["pfc0"], dtype=np.float64)
            sol0 = np.asarray(init["sol0"], dtype=np.float64)
    else:
        return None
    initial = np.concatenate([pfc0, sol0], axis=1)
    jdot = normalized_action * derivative_limits.reshape(1, 1, -1)
    currents = np.empty((initial.shape[0], jdot.shape[1] + 1, initial.shape[1]), dtype=np.float64)
    currents[:, 0, :] = initial
    currents[:, 1:, :] = initial[:, None, :] + np.cumsum(jdot * float(dt), axis=1)
    return currents


def limits_from_config(config_path: Path | None, *, current_order: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if config_path is None:
        return None, None
    cfg_path = config_path.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config does not exist: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    sim = raw.get("sim", {})
    current_limits = None
    limits = sim.get("current_safety_limits")
    if isinstance(limits, dict):
        pfc = np.asarray(_ordered_values(limits.get("pfc_currents")), dtype=np.float64)
        sol = np.asarray(_ordered_values(limits.get("sol_currents")), dtype=np.float64)
        current_limits = np.concatenate([sol, pfc]) if current_order == "sol_pfc" else np.concatenate([pfc, sol])

    derivative_limits = None
    machine = sim.get("config_path")
    if isinstance(machine, str) and machine:
        machine_path = Path(machine)
        if not machine_path.is_absolute():
            machine_path = (cfg_path.parent / machine_path).resolve()
        if machine_path.exists():
            text = machine_path.read_text(encoding="utf-8")
            pfc_deriv = _toml_float(text, "pfc_deriv_limit")
            sol_deriv = _toml_float(text, "sol_deriv_limit")
            if pfc_deriv is not None and sol_deriv is not None:
                pfc = np.full((6,), float(pfc_deriv), dtype=np.float64)
                sol = np.full((3,), float(sol_deriv), dtype=np.float64)
                derivative_limits = np.concatenate([sol, pfc]) if current_order == "sol_pfc" else np.concatenate([pfc, sol])
    return current_limits, derivative_limits


def _ordered_values(raw: object) -> list[float]:
    if isinstance(raw, dict):
        return [float(raw[k]) for k in sorted(raw)]
    if isinstance(raw, list):
        return [float(v) for v in raw]
    raise ValueError("current_safety_limits entries must be lists or dicts")


def _toml_float(text: str, key: str) -> float | None:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([-+0-9.eE]+)", text, flags=re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def validate_shapes(
    *,
    name: str,
    ip: np.ndarray,
    radii: np.ndarray,
    params: np.ndarray | None,
    currents: np.ndarray | None,
    jdot: np.ndarray | None,
    action: np.ndarray | None,
) -> None:
    if ip.ndim != 2 or ip.shape[0] == 0 or ip.shape[1] < 2:
        raise ValueError(f"{name}: ip must have shape [N,T+1], got {ip.shape}")
    if radii.ndim != 3 or radii.shape[0] != ip.shape[0] or radii.shape[1] != ip.shape[1]:
        raise ValueError(f"{name}: radii shape {radii.shape} incompatible with ip {ip.shape}")
    if params is not None and (params.ndim != 3 or params.shape[0] != ip.shape[0] or params.shape[1] != ip.shape[1]):
        raise ValueError(f"{name}: params shape {params.shape} incompatible with ip {ip.shape}")
    if currents is not None and (currents.ndim != 3 or currents.shape[0] != ip.shape[0] or currents.shape[1] != ip.shape[1]):
        raise ValueError(f"{name}: currents shape {currents.shape} incompatible with ip {ip.shape}")
    if jdot is not None and (jdot.ndim != 3 or jdot.shape[0] != ip.shape[0] or jdot.shape[1] != ip.shape[1] - 1):
        raise ValueError(f"{name}: jdot shape {jdot.shape} incompatible with ip {ip.shape}")
    if action is not None and (action.ndim != 3 or action.shape[0] != ip.shape[0] or action.shape[1] != ip.shape[1] - 1):
        raise ValueError(f"{name}: action shape {action.shape} incompatible with ip {ip.shape}")
    for label, arr in (("ip", ip), ("radii", radii), ("params", params), ("currents", currents), ("jdot", jdot), ("action", action)):
        if arr is not None and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: {label} contains non-finite values")


def compute_metrics(lib: TargetLibrary, *, dt: float) -> dict[str, np.ndarray]:
    metrics: dict[str, np.ndarray] = {}
    ip = lib.ip
    radii = lib.radii
    d_ip = ip[:, -1] - ip[:, 0]
    d_radii = radii[:, -1, :] - radii[:, 0, :]
    step_d_ip = np.diff(ip, axis=1)
    step_d_radii = np.diff(radii, axis=1)
    step_mean_radii = np.mean(np.abs(step_d_radii), axis=2)
    metrics["endpoint_abs_dip_a"] = np.abs(d_ip)
    metrics["endpoint_signed_dip_a"] = d_ip
    metrics["endpoint_mean_abs_dradii_m"] = np.mean(np.abs(d_radii), axis=1)
    metrics["endpoint_max_abs_dradii_m"] = np.max(np.abs(d_radii), axis=1)
    metrics["ip_step_abs_max_a"] = np.max(np.abs(step_d_ip), axis=1)
    metrics["ip_rate_abs_max_aps"] = metrics["ip_step_abs_max_a"] / float(dt)
    metrics["boundary_step_mean_abs_m_mean"] = np.mean(step_mean_radii, axis=1)
    metrics["boundary_step_mean_abs_m_max"] = np.max(step_mean_radii, axis=1)
    metrics["boundary_step_any_angle_abs_m_max"] = np.max(np.abs(step_d_radii), axis=(1, 2))
    metrics["boundary_mean_radius_range_m"] = np.max(np.mean(radii, axis=2), axis=1) - np.min(np.mean(radii, axis=2), axis=1)

    if lib.params is not None:
        params = lib.params
        names = ("R0", "Z0", "A0", "kappa", "delta")
        for i, name in enumerate(names):
            metrics[f"endpoint_abs_d{name}"] = np.abs(params[:, -1, i] - params[:, 0, i])
            metrics[f"{name}_step_abs_max"] = np.max(np.abs(np.diff(params[:, :, i], axis=1)), axis=1)

    if lib.currents is not None:
        metrics["current_abs_max_a"] = np.max(np.abs(lib.currents), axis=(1, 2))
        if lib.current_limits is not None:
            usage = np.abs(lib.currents) / lib.current_limits.reshape(1, 1, -1)
            metrics["current_usage_fraction_max"] = np.max(usage, axis=(1, 2))
            metrics["current_over_limit_a_max"] = np.max(
                np.maximum(np.abs(lib.currents) - lib.current_limits.reshape(1, 1, -1), 0.0),
                axis=(1, 2),
            )

    if lib.jdot is not None:
        metrics["jdot_abs_max_aps"] = np.max(np.abs(lib.jdot), axis=(1, 2))
        metrics["jdot_rms_aps"] = np.sqrt(np.mean(lib.jdot**2, axis=(1, 2)))
        djdot = np.diff(lib.jdot, axis=1)
        if djdot.size:
            metrics["jdot_step_jump_abs_max_aps"] = np.max(np.abs(djdot), axis=(1, 2))
            metrics["jdot_step_jump_rms_aps"] = np.sqrt(np.mean(djdot**2, axis=(1, 2)))

    if lib.normalized_action is not None:
        action = lib.normalized_action
        metrics["action_abs_max"] = np.max(np.abs(action), axis=(1, 2))
        metrics["action_rms"] = np.sqrt(np.mean(action**2, axis=(1, 2)))
        da = np.diff(action, axis=1)
        if da.size:
            metrics["action_step_jump_abs_max"] = np.max(np.abs(da), axis=(1, 2))
            metrics["action_step_jump_rms"] = np.sqrt(np.mean(da**2, axis=(1, 2)))

    return metrics


def nearest_window_distances(reference: TargetLibrary, candidate: TargetLibrary, *, chunk_size: int) -> np.ndarray:
    ref = endpoint_feature_matrix(reference)
    cand = endpoint_feature_matrix(candidate)
    scale = np.nanpercentile(np.abs(ref), 99, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    ref_n = ref / scale[None, :]
    cand_n = cand / scale[None, :]
    out = np.empty((cand_n.shape[0],), dtype=np.float64)
    chunk = max(1, int(chunk_size))
    for start in range(0, cand_n.shape[0], chunk):
        block = cand_n[start : start + chunk]
        dist2 = np.sum((block[:, None, :] - ref_n[None, :, :]) ** 2, axis=2)
        out[start : start + block.shape[0]] = np.sqrt(np.min(dist2, axis=1))
    return out


def endpoint_feature_matrix(lib: TargetLibrary) -> np.ndarray:
    dip = (lib.ip[:, -1] - lib.ip[:, 0]).reshape(-1, 1)
    dr = lib.radii[:, -1, :] - lib.radii[:, 0, :]
    return np.concatenate([dip, dr], axis=1).astype(np.float64)


def metric_summary_row(dataset: str, metric: str, values: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "dataset": dataset,
            "metric": metric,
            "count": 0,
            "mean": "",
            "std": "",
            "p50": "",
            "p90": "",
            "p99": "",
            "max": "",
        }
    return {
        "dataset": dataset,
        "metric": metric,
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def ratio_summary(
    reference_name: str,
    candidate_name: str,
    ref_metrics: dict[str, np.ndarray],
    cand_metrics: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in sorted(set(ref_metrics) & set(cand_metrics)):
        ref = np.asarray(ref_metrics[metric], dtype=np.float64).reshape(-1)
        cand = np.asarray(cand_metrics[metric], dtype=np.float64).reshape(-1)
        ref = ref[np.isfinite(ref)]
        cand = cand[np.isfinite(cand)]
        if ref.size == 0 or cand.size == 0:
            continue
        ref_p90 = float(np.percentile(ref, 90))
        cand_p90 = float(np.percentile(cand, 90))
        ref_mean = float(np.mean(ref))
        cand_mean = float(np.mean(cand))
        rows.append(
            {
                "metric": metric,
                "reference": reference_name,
                "candidate": candidate_name,
                "mean_ratio": _safe_ratio(cand_mean, ref_mean),
                "p90_ratio": _safe_ratio(cand_p90, ref_p90),
                "reference_mean": ref_mean,
                "candidate_mean": cand_mean,
                "reference_p90": ref_p90,
                "candidate_p90": cand_p90,
            }
        )
    return rows


def _safe_ratio(numer: float, denom: float) -> float | None:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) < 1.0e-12:
        return None
    return float(numer / denom)


def library_metadata(lib: TargetLibrary) -> dict[str, object]:
    return {
        "name": lib.name,
        "target_path": str(lib.target_path),
        "initial_path": None if lib.initial_path is None else str(lib.initial_path),
        "schema": lib.schema,
        "rows": lib.row_count,
        "steps": lib.steps,
        "angles": int(lib.radii.shape[2]),
        "splits": _counts(lib.split),
        "shots": _counts(lib.shot_id),
        "modes": None if lib.mode is None else _counts(lib.mode),
        "has_params": lib.params is not None,
        "has_currents": lib.currents is not None,
        "has_jdot": lib.jdot is not None,
        "has_normalized_action": lib.normalized_action is not None,
        "current_order": lib.current_order,
        "derivative_order": lib.derivative_order,
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values.astype(str).tolist():
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def render_report(
    reference: TargetLibrary,
    candidate: TargetLibrary,
    summary_rows: list[dict[str, object]],
    ratios: list[dict[str, object]],
) -> str:
    interesting = [
        "endpoint_abs_dip_a",
        "endpoint_mean_abs_dradii_m",
        "ip_rate_abs_max_aps",
        "boundary_step_mean_abs_m_max",
        "boundary_step_any_angle_abs_m_max",
        "current_usage_fraction_max",
        "action_rms",
        "action_step_jump_abs_max",
        "jdot_rms_aps",
        "jdot_step_jump_abs_max_aps",
        "nearest_reference_window_distance",
    ]
    rows_by_key = {(r["dataset"], r["metric"]): r for r in summary_rows}

    lines = [
        "# Target Library Comparison",
        "",
        "## Libraries",
        "",
        f"- Reference: `{reference.name}` rows={reference.row_count}, steps={reference.steps}, schema=`{reference.schema}`",
        f"- Candidate: `{candidate.name}` rows={candidate.row_count}, steps={candidate.steps}, schema=`{candidate.schema}`",
        "",
        "## High-Signal Metrics",
        "",
        "| metric | reference mean | reference p90 | candidate mean | candidate p90 | p90 ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ratio_by_metric = {str(r["metric"]): r for r in ratios}
    for metric in interesting:
        ref = rows_by_key.get((reference.name, metric))
        cand = rows_by_key.get((candidate.name, metric))
        if ref is None and cand is None:
            continue
        ratio = ratio_by_metric.get(metric, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{metric}`",
                    fmt(ref.get("mean") if ref else ""),
                    fmt(ref.get("p90") if ref else ""),
                    fmt(cand.get("mean") if cand else ""),
                    fmt(cand.get("p90") if cand else ""),
                    fmt(ratio.get("p90_ratio", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Largest Distribution Changes",
            "",
            "| metric | candidate/reference p90 | reference p90 | candidate p90 |",
            "|---|---:|---:|---:|",
        ]
    )
    ranked = [
        r
        for r in ratios
        if isinstance(r.get("p90_ratio"), float)
        and np.isfinite(float(r["p90_ratio"]))
        and str(r["metric"]) != "nearest_reference_window_distance"
    ]
    ranked.sort(key=lambda r: abs(math.log(max(float(r["p90_ratio"]), 1.0e-12))), reverse=True)
    for row in ranked[:20]:
        lines.append(
            f"| `{row['metric']}` | {fmt(row['p90_ratio'])} | {fmt(row['reference_p90'])} | {fmt(row['candidate_p90'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `nearest_reference_window_distance` is computed from endpoint `Ip` and all 32 endpoint boundary-radii deltas, normalized by the reference p99 scales.",
            "- For oracle replay-window libraries, currents are reconstructed from initial currents plus stored normalized `real_jdot_action`.",
            "- For generated libraries with `coil_witness`, current and Jdot metrics describe the witness/open-loop trajectory used to generate the target, not necessarily what the learned policy did.",
            "- This is a data-distribution audit. It does not run LQR or closed-loop policy simulations.",
            "",
        ]
    )
    return "\n".join(lines)


def fmt(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(x):
        return ""
    ax = abs(x)
    if ax == 0.0:
        return "0"
    if ax >= 1.0e5 or ax < 1.0e-3:
        return f"{x:.4g}"
    return f"{x:.4f}"


def write_plots(
    reference: TargetLibrary,
    candidate: TargetLibrary,
    ref_metrics: dict[str, np.ndarray],
    cand_metrics: dict[str, np.ndarray],
    *,
    out_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_metrics = [
        "endpoint_abs_dip_a",
        "endpoint_mean_abs_dradii_m",
        "boundary_step_mean_abs_m_max",
        "boundary_step_any_angle_abs_m_max",
        "current_usage_fraction_max",
        "action_rms",
        "action_step_jump_abs_max",
        "jdot_rms_aps",
        "jdot_step_jump_abs_max_aps",
        "nearest_reference_window_distance",
    ]
    available = [m for m in plot_metrics if m in ref_metrics or m in cand_metrics]
    if available:
        cols = 2
        rows = int(math.ceil(len(available) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows))
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, metric in zip(axes_arr, available, strict=False):
            if metric in ref_metrics:
                ax.hist(_finite(ref_metrics[metric]), bins=60, alpha=0.45, label=reference.name, density=True)
            if metric in cand_metrics:
                ax.hist(_finite(cand_metrics[metric]), bins=60, alpha=0.45, label=candidate.name, density=True)
            ax.set_title(metric)
            ax.grid(True, alpha=0.25)
            ax.legend()
        for ax in axes_arr[len(available) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "target_library_histograms.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for lib, color in ((reference, "tab:blue"), (candidate, "tab:orange")):
        x = lib.ip[:, -1] - lib.ip[:, 0]
        y = np.mean(lib.radii[:, -1, :] - lib.radii[:, 0, :], axis=1)
        ax.scatter(x / 1000.0, y * 100.0, s=8, alpha=0.35, label=lib.name, color=color)
    ax.set_xlabel("endpoint dIp [kA]")
    ax.set_ylabel("endpoint mean d-radius [cm]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "endpoint_delta_scatter.png", dpi=150)
    plt.close(fig)


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


if __name__ == "__main__":
    raise SystemExit(main())
