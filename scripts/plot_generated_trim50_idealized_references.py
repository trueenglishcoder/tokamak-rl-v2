#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env.references import (
    GENERATED_BOUNDARY_COMBOS,
    GENERATED_IP_MODES,
    boundary_points_from_parameters,
    generate_reference_batch,
    load_generated_envelope,
    sample_generated_boundary_parameters,
    sample_generated_segment_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/t15_generated_trim50_idealized_matched_0p1s_tcvjdot_balanced_mpo.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/generated_trim50_idealized_reference_examples"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    reset = _load_reset_rows(cfg.sim.csv_initial_state_library, count=max(int(args.samples), len(GENERATED_IP_MODES), len(GENERATED_BOUNDARY_COMBOS)), rng=rng)
    envelope = load_generated_envelope(cfg.reference.ip.limits_path)

    _plot_ip_modes(cfg, reset, args.out_dir)
    _plot_boundary_modes(cfg, reset, args.out_dir)
    _plot_boundary_shapes(cfg, reset, args.out_dir)
    _plot_coverage(cfg, reset, envelope, args.out_dir, samples=int(args.samples), seed=int(args.seed))
    print(args.out_dir)
    return 0


def _load_reset_rows(path: Path, *, count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        split = data["split"].astype(str)
        train_idx = np.nonzero(split == "train")[0]
        if train_idx.size == 0:
            raise ValueError(f"{path} contains no train rows")
        idx = rng.choice(train_idx, size=int(count), replace=train_idx.size < int(count))
        return {
            "ip0": np.asarray(data["ip0"], dtype=float)[idx],
            "params0": np.asarray(data["params0"], dtype=float)[idx],
        }


def _plot_ip_modes(cfg, reset: dict[str, np.ndarray], out_dir: Path) -> None:
    t = np.arange(int(cfg.sim.max_episode_steps) + 1) * float(cfg.reference.t_step)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    for i, mode in enumerate(GENERATED_IP_MODES):
        rng = np.random.default_rng(1000 + i)
        sample = sample_generated_segment_profile(
            cfg.reference.ip,
            float(reset["ip0"][i]),
            int(cfg.sim.max_episode_steps),
            rng,
            dt=float(cfg.reference.t_step),
            forced_mode=mode,
        )
        ax.plot(t, sample.ip, label=mode)
    ax.set_title("Generated Ip reference modes")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Ip [A]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "generated_ip_modes.png")
    plt.close(fig)


def _plot_boundary_modes(cfg, reset: dict[str, np.ndarray], out_dir: Path) -> None:
    t = np.arange(int(cfg.sim.max_episode_steps) + 1) * float(cfg.reference.t_step)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=140, sharex=True)
    names = ("A0", "kappa - 1", "delta")
    for i, combo in enumerate(GENERATED_BOUNDARY_COMBOS):
        rng = np.random.default_rng(2000 + i)
        sample = sample_generated_boundary_parameters(
            cfg.reference.boundary,
            reset["params0"][i],
            int(cfg.sim.max_episode_steps),
            rng,
            dt=float(cfg.reference.t_step),
            forced_combo=tuple(combo),
        )
        values = np.column_stack([sample.parameters[:, 2], sample.parameters[:, 3] - 1.0, sample.parameters[:, 4]])
        label = _boundary_sample_label(combo, sample.modes)
        for ax, name, column in zip(axes, names, values.T, strict=True):
            ax.plot(t, column, label=label)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    axes[0].set_title("Generated boundary parameter modes")
    axes[0].legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "generated_boundary_parameter_modes.png")
    plt.close(fig)


def _plot_boundary_shapes(cfg, reset: dict[str, np.ndarray], out_dir: Path) -> None:
    theta = torch.linspace(-torch.pi, torch.pi, int(cfg.reference.theta_count) + 1, dtype=torch.float64)[:-1]
    steps = int(cfg.sim.max_episode_steps)
    sample_steps = [0, steps // 2, steps]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), dpi=140, sharex=True, sharey=True)
    for i, combo in enumerate(GENERATED_BOUNDARY_COMBOS):
        ax = axes.flat[i]
        rng = np.random.default_rng(3000 + i)
        sample = sample_generated_boundary_parameters(
            cfg.reference.boundary,
            reset["params0"][i],
            int(cfg.sim.max_episode_steps),
            rng,
            dt=float(cfg.reference.t_step),
            forced_combo=tuple(combo),
        )
        params = torch.as_tensor(sample.parameters[sample_steps], dtype=torch.float64)
        points = boundary_points_from_parameters(params, theta).detach().cpu().numpy()
        label = _boundary_sample_label(combo, sample.modes)
        for k, step in enumerate(sample_steps):
            ax.plot(points[k, :, 0], points[k, :, 1], label=f"step {step}")
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Generated 32-angle boundary examples")
    fig.tight_layout()
    fig.savefig(out_dir / "generated_boundary_shapes.png")
    plt.close(fig)


def _plot_coverage(cfg, reset: dict[str, np.ndarray], envelope, out_dir: Path, *, samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed + 999)
    idx = rng.choice(np.arange(reset["ip0"].shape[0]), size=int(samples), replace=reset["ip0"].shape[0] < int(samples))
    refs = generate_reference_batch(
        config=cfg.reference,
        initial_ip=reset["ip0"][idx],
        initial_parameters=reset["params0"][idx],
        steps=int(cfg.sim.max_episode_steps),
        device="cpu",
        seed=seed + 111,
    )
    ip = refs.ip.detach().cpu().numpy().reshape(-1)
    params = refs.parameters.detach().cpu().numpy()
    values = {
        "Ip [A]": (ip, envelope.ip_min_a, envelope.ip_max_a),
        "A0 [m]": (params[:, :, 2].reshape(-1), envelope.A0_min_m, envelope.A0_max_m),
        "kappa - 1": ((params[:, :, 3] - 1.0).reshape(-1), envelope.elongation_excess_min, envelope.elongation_excess_max),
        "delta": (params[:, :, 4].reshape(-1), envelope.delta_min, envelope.delta_max),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=140)
    for ax, (name, (arr, lo, hi)) in zip(axes.flat, values.items(), strict=True):
        ax.hist(arr, bins=60, color="#4477aa", alpha=0.8)
        ax.axvline(lo, color="#228833", linestyle="--", linewidth=1.5)
        ax.axvline(hi, color="#228833", linestyle="--", linewidth=1.5)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Generated target coverage against replay envelope")
    fig.tight_layout()
    fig.savefig(out_dir / "generated_coverage_histograms.png")
    plt.close(fig)


def _boundary_sample_label(combo: tuple[str, ...], modes: dict[str, str]) -> str:
    if not combo:
        return "hold"
    return ", ".join(f"{key}:{modes.get(key, '?')}" for key in combo)


if __name__ == "__main__":
    raise SystemExit(main())
