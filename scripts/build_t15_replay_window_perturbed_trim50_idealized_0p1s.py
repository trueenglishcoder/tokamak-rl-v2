#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_t15_simple_manifold_generated_trim50_idealized_0p1s as simple


DEFAULT_OUT_DIR = Path("data/processed/t15_replay_window_perturbed_trim50_idealized_0p1s")
DEFAULT_INITIAL_STATES_OUT = Path("data/processed/t15_replay_window_perturbed_trim50_idealized_0p1s_initial_states.npz")
DEFAULT_TARGETS_OUT = DEFAULT_OUT_DIR / "t15_feasible_generated_trim50_idealized_0p1s_targets.npz"


@dataclass(frozen=True, slots=True)
class PerturbedCandidate:
    window: simple.ReplayWindow
    mode: str
    variant_index: int
    ip_ref: np.ndarray
    params_ref: np.ndarray
    radii_ref: np.ndarray
    coil_witness: np.ndarray
    max_fractional_perturbation: float
    max_step_fractional_perturbation: float


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build 0.1 s generated targets by taking real replay windows and "
            "adding smooth bounded perturbations. The first target sample is "
            "always the exact real reset state."
        )
    )
    parser.add_argument("--boundary-param-dir", type=Path, default=simple.DEFAULT_BOUNDARY_PARAM_DIR)
    parser.add_argument("--data-root", type=Path, default=simple.DEFAULT_DATA_ROOT)
    parser.add_argument("--machine-config", type=Path, default=simple.DEFAULT_MACHINE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial-states-out", type=Path, default=DEFAULT_INITIAL_STATES_OUT)
    parser.add_argument("--targets-out", type=Path, default=DEFAULT_TARGETS_OUT)
    parser.add_argument("--train-shots", nargs="+", default=list(simple.DEFAULT_TRAIN_SHOTS))
    parser.add_argument("--holdout-shots", nargs="+", default=list(simple.DEFAULT_HOLDOUT_SHOTS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--variants-per-window", type=int, default=2)
    parser.add_argument("--max-fraction", type=float, default=0.05)
    parser.add_argument("--knot-count", type=int, default=6)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--max-windows-per-shot", type=int, default=0, help="0 keeps every valid real replay window.")
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--perturb-center", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    steps = int(args.steps)
    if steps != 100:
        raise SystemExit("this builder intentionally targets 0.1 s / 100-step windows")
    if int(args.variants_per_window) < 0:
        raise SystemExit("variants_per_window must be non-negative")
    if not bool(args.include_original) and int(args.variants_per_window) == 0:
        raise SystemExit("nothing to write: use --include-original or variants_per_window > 0")
    max_fraction = float(args.max_fraction)
    if not np.isfinite(max_fraction) or max_fraction <= 0.0 or max_fraction > 0.25:
        raise SystemExit("max_fraction must be finite and in (0, 0.25]")
    if int(args.knot_count) < 3:
        raise SystemExit("knot_count must be at least 3")

    train_shots = tuple(str(int(v)) for v in args.train_shots)
    holdout_shots = tuple(str(int(v)) for v in args.holdout_shots)
    overlap = sorted(set(train_shots) & set(holdout_shots), key=int)
    if overlap:
        raise SystemExit("train and holdout shots overlap: " + ", ".join(overlap))

    rng = np.random.default_rng(int(args.seed))
    windows = simple._load_replay_windows(
        args.boundary_param_dir.resolve(),
        args.data_root.resolve(),
        train_shots=train_shots,
        holdout_shots=holdout_shots,
        steps=steps,
    )
    windows = _limit_windows_per_shot(windows, int(args.max_windows_per_shot))
    if not windows:
        raise SystemExit("no real replay windows found")

    theta = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    candidates: list[PerturbedCandidate] = []
    for window in windows:
        if bool(args.include_original):
            candidates.append(_candidate_from_window(window, theta=theta))
        for variant in range(int(args.variants_per_window)):
            candidates.append(
                _perturbed_candidate_from_window(
                    window,
                    theta=theta,
                    rng=rng,
                    variant_index=variant,
                    max_fraction=max_fraction,
                    knot_count=int(args.knot_count),
                    smooth_window=int(args.smooth_window),
                    perturb_center=bool(args.perturb_center),
                )
            )

    _write_libraries(candidates, args.initial_states_out, args.targets_out, train_shots=train_shots, holdout_shots=holdout_shots)
    summary = _summary(candidates, windows=windows, args=args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "perturbed_replay_window_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if bool(args.plots):
        _write_plots(candidates, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _limit_windows_per_shot(windows: list[simple.ReplayWindow], max_per_shot: int) -> list[simple.ReplayWindow]:
    limit = int(max_per_shot)
    if limit <= 0:
        return windows
    grouped: dict[str, list[simple.ReplayWindow]] = {}
    for window in windows:
        grouped.setdefault(window.shot, []).append(window)
    limited: list[simple.ReplayWindow] = []
    for shot in sorted(grouped, key=int):
        rows = grouped[shot]
        if len(rows) <= limit:
            limited.extend(rows)
            continue
        indices = np.linspace(0, len(rows) - 1, limit, dtype=int)
        limited.extend(rows[int(i)] for i in indices)
    return limited


def _candidate_from_window(window: simple.ReplayWindow, *, theta: np.ndarray) -> PerturbedCandidate:
    params = np.asarray(window.params, dtype=float).copy()
    ip = np.asarray(window.x[:, 0], dtype=float).copy()
    return PerturbedCandidate(
        window=window,
        mode="real",
        variant_index=-1,
        ip_ref=ip.astype(np.float32),
        params_ref=params.astype(np.float32),
        radii_ref=simple._radii_from_params(params, theta).astype(np.float32),
        coil_witness=np.asarray(window.coils, dtype=np.float32),
        max_fractional_perturbation=0.0,
        max_step_fractional_perturbation=0.0,
    )


def _perturbed_candidate_from_window(
    window: simple.ReplayWindow,
    *,
    theta: np.ndarray,
    rng: np.random.Generator,
    variant_index: int,
    max_fraction: float,
    knot_count: int,
    smooth_window: int,
    perturb_center: bool,
) -> PerturbedCandidate:
    series = np.column_stack([window.x[:, 0], window.params]).astype(float)
    perturb_cols = [0, 3, 4, 5]  # Ip, A0, kappa, delta. Keep R0/Z0 fixed by default.
    if bool(perturb_center):
        perturb_cols = [0, 1, 2, 3, 4, 5]
    frac = np.zeros_like(series, dtype=float)
    frac[:, perturb_cols] = _smooth_fractional_noise(
        rng,
        steps=series.shape[0] - 1,
        dims=len(perturb_cols),
        max_fraction=float(max_fraction),
        knot_count=int(knot_count),
        smooth_window=int(smooth_window),
    )

    perturbed = series * (1.0 + frac)
    # The environment reset must match the first target sample exactly.
    perturbed[0] = series[0]
    # Keep parameterization physically meaningful after the small perturbation.
    perturbed[:, 0] = np.maximum(perturbed[:, 0], 1.0)  # Ip
    perturbed[:, 3] = np.maximum(perturbed[:, 3], 1.0e-3)  # A0
    perturbed[:, 4] = np.maximum(perturbed[:, 4], 0.5)  # kappa
    perturbed[:, 5] = np.maximum(perturbed[:, 5], 0.0)  # delta
    perturbed[0] = series[0]

    actual = _fractional_difference(perturbed[:, perturb_cols], series[:, perturb_cols])
    max_actual = float(np.max(np.abs(actual))) if actual.size else 0.0
    if max_actual > float(max_fraction) + 1.0e-6:
        raise RuntimeError(f"perturbation exceeded requested bound: {max_actual} > {max_fraction}")
    max_step = float(np.max(np.abs(np.diff(actual, axis=0)))) if actual.shape[0] > 1 else 0.0

    params = perturbed[:, 1:6]
    return PerturbedCandidate(
        window=window,
        mode="perturbed",
        variant_index=int(variant_index),
        ip_ref=perturbed[:, 0].astype(np.float32),
        params_ref=params.astype(np.float32),
        radii_ref=simple._radii_from_params(params, theta).astype(np.float32),
        coil_witness=np.asarray(window.coils, dtype=np.float32),
        max_fractional_perturbation=max_actual,
        max_step_fractional_perturbation=max_step,
    )


def _smooth_fractional_noise(
    rng: np.random.Generator,
    *,
    steps: int,
    dims: int,
    max_fraction: float,
    knot_count: int,
    smooth_window: int,
) -> np.ndarray:
    steps = int(steps)
    dims = int(dims)
    if steps <= 0 or dims <= 0:
        raise ValueError("steps and dims must be positive")
    knot_count = max(3, int(knot_count))
    knot_t = np.linspace(0.0, float(steps), knot_count)
    knot_y = rng.uniform(-float(max_fraction), float(max_fraction), size=(knot_count, dims))
    knot_y[0, :] = 0.0
    t = np.arange(steps + 1, dtype=float)
    out = np.empty((steps + 1, dims), dtype=float)
    for col in range(dims):
        out[:, col] = np.interp(t, knot_t, knot_y[:, col])
    out = _hann_smooth(out, int(smooth_window))
    out[0, :] = 0.0
    scale = np.max(np.abs(out), axis=0)
    over = scale > float(max_fraction)
    if np.any(over):
        out[:, over] *= float(max_fraction) / scale[over][None, :]
    out[0, :] = 0.0
    return out


def _hann_smooth(values: np.ndarray, window: int) -> np.ndarray:
    width = int(window)
    if width <= 2:
        return np.asarray(values, dtype=float).copy()
    if width % 2 == 0:
        width += 1
    kernel = np.hanning(width)
    if float(np.sum(kernel)) <= 0.0:
        return np.asarray(values, dtype=float).copy()
    kernel = kernel / float(np.sum(kernel))
    pad = width // 2
    arr = np.asarray(values, dtype=float)
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(arr, dtype=float)
    for col in range(arr.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return out


def _fractional_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(b), 1.0e-12)
    return (a - b) / denom


def _write_libraries(
    candidates: list[PerturbedCandidate],
    initial_states_out: Path,
    targets_out: Path,
    *,
    train_shots: tuple[str, ...],
    holdout_shots: tuple[str, ...],
) -> None:
    initial_states_out.parent.mkdir(parents=True, exist_ok=True)
    targets_out.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(candidates)
    source_index = np.arange(row_count, dtype=np.int64)
    shot_id = np.asarray([c.window.shot for c in candidates], dtype="<U8")
    split = np.asarray([c.window.split for c in candidates], dtype="<U8")
    difficulty_bin = np.full((row_count,), "core", dtype="<U16")
    mode = np.asarray([c.mode for c in candidates], dtype="<U16")
    params0 = np.asarray([c.params_ref[0] for c in candidates], dtype=np.float32)
    np.savez_compressed(
        initial_states_out,
        schema=np.asarray("t15_replay_window_perturbed_trim50_idealized_initial_states_v1"),
        shot_id=shot_id,
        source_index=source_index,
        time_s=np.asarray([c.window.time_s for c in candidates], dtype=np.float64),
        ip0=np.asarray([c.ip_ref[0] for c in candidates], dtype=np.float32),
        pfc0=np.asarray([c.coil_witness[0, 3:] for c in candidates], dtype=np.float32),
        sol0=np.asarray([c.coil_witness[0, :3] for c in candidates], dtype=np.float32),
        params0=params0,
        split=split,
        difficulty_bin=difficulty_bin,
        mode=mode,
    )
    np.savez_compressed(
        targets_out,
        schema=np.asarray("t15_replay_window_perturbed_trim50_idealized_targets_v1"),
        ip_ref=np.asarray([c.ip_ref for c in candidates], dtype=np.float32),
        params_ref=np.asarray([c.params_ref for c in candidates], dtype=np.float32),
        radii_ref=np.asarray([c.radii_ref for c in candidates], dtype=np.float32),
        coil_witness=np.asarray([c.coil_witness for c in candidates], dtype=np.float32),
        zone=difficulty_bin,
        mode=mode,
        shot_id=shot_id,
        source_index=source_index,
        replay_source_index=np.asarray([c.window.source_index for c in candidates], dtype=np.int64),
        replay_start_row=np.asarray([c.window.start_row for c in candidates], dtype=np.int64),
        variant_index=np.asarray([c.variant_index for c in candidates], dtype=np.int64),
        time_s=np.asarray([c.window.time_s for c in candidates], dtype=np.float64),
        split=split,
        max_fractional_perturbation=np.asarray([c.max_fractional_perturbation for c in candidates], dtype=np.float32),
        max_step_fractional_perturbation=np.asarray([c.max_step_fractional_perturbation for c in candidates], dtype=np.float32),
        train_shots=np.asarray(train_shots, dtype="<U8"),
        holdout_shots=np.asarray(holdout_shots, dtype="<U8"),
    )


def _summary(candidates: list[PerturbedCandidate], *, windows: list[simple.ReplayWindow], args: argparse.Namespace) -> dict[str, object]:
    perturb = [c.max_fractional_perturbation for c in candidates]
    step_perturb = [c.max_step_fractional_perturbation for c in candidates]
    return {
        "schema": "t15_replay_window_perturbed_trim50_idealized_summary_v1",
        "boundary_param_dir": str(args.boundary_param_dir),
        "data_root": str(args.data_root),
        "machine_config": str(args.machine_config),
        "initial_states": str(args.initial_states_out),
        "targets": str(args.targets_out),
        "source_replay_windows": len(windows),
        "accepted_targets": len(candidates),
        "accepted_by_mode": dict(sorted(Counter(c.mode for c in candidates).items())),
        "accepted_by_split": dict(sorted(Counter(c.window.split for c in candidates).items())),
        "accepted_by_shot": dict(sorted(Counter(c.window.shot for c in candidates).items(), key=lambda kv: int(kv[0]))),
        "include_original": bool(args.include_original),
        "variants_per_window": int(args.variants_per_window),
        "max_fraction_requested": float(args.max_fraction),
        "max_fraction_observed": float(np.max(perturb)) if perturb else 0.0,
        "max_step_fraction_observed": float(np.max(step_perturb)) if step_perturb else 0.0,
        "perturb_center": bool(args.perturb_center),
        "note": (
            "Targets are real replay windows plus smooth low-frequency fractional perturbations. "
            "The first target sample is exact, so reset state and reference state agree."
        ),
    }


def _write_plots(candidates: list[PerturbedCandidate], *, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    perturbed = [c for c in candidates if c.mode == "perturbed"]
    if not perturbed:
        perturbed = candidates
    subset = [perturbed[int(i)] for i in rng.choice(np.arange(len(perturbed)), size=min(80, len(perturbed)), replace=False)]
    t = np.arange(candidates[0].ip_ref.shape[0])
    labels = ("Ip [A]", "A0 [m]", "kappa", "delta")
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for c in subset:
        window = c.window
        target = np.column_stack([c.ip_ref, c.params_ref[:, 2], c.params_ref[:, 3], c.params_ref[:, 4]])
        base = np.column_stack([window.x[:, 0], window.params[:, 2], window.params[:, 3], window.params[:, 4]])
        for col, axis in enumerate(axes):
            axis.plot(t, base[:, col], color="0.75", alpha=0.25, linewidth=0.8)
            axis.plot(t, target[:, col], alpha=0.25, linewidth=1.0)
    for axis, label in zip(axes, labels, strict=True):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_perturbed_replay_windows.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    values = [c.max_fractional_perturbation for c in candidates if c.mode == "perturbed"]
    ax.hist(values, bins=40)
    ax.set_xlabel("max fractional perturbation")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "perturbation_fraction_histogram.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
