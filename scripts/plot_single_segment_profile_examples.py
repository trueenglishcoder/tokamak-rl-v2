#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env.references import generate_reference_batch


def _classify(values: np.ndarray) -> str:
    delta = float(values[-1] - values[0])
    if abs(delta) <= 1.0e-6:
        return "hold"
    return "ramp_up" if delta > 0.0 else "ramp_down"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot examples from the 0.1 s single-segment Ip generator.")
    parser.add_argument(
        "--config",
        default="configs/experiments/t15_csv_initial_single_segment_0p1s_static_boundary_mpo.yaml",
    )
    parser.add_argument("--output-dir", default="analysis_outputs/single_segment_profile_0p1s_examples")
    parser.add_argument("--samples", type=int, default=96)
    args = parser.parse_args()

    cfg = load_experiment_config(Path(args.config))
    theta_count = int(cfg.reference.theta_count)
    sample_count = max(3, int(args.samples))
    points0 = np.zeros((sample_count, theta_count, 2), dtype=float)
    radii0 = np.ones((sample_count, theta_count), dtype=float)
    initial_ip = np.linspace(180000.0, 320000.0, sample_count, dtype=float)
    ref = generate_reference_batch(
        config=cfg.reference,
        initial_ip=initial_ip,
        initial_parameters=np.zeros((sample_count, 5), dtype=float),
        steps=int(cfg.sim.max_episode_steps),
        device="cpu",
        seed=int(cfg.reference.seed),
        initial_boundary_points=points0,
        initial_boundary_radii=radii0,
    )
    ip = ref.ip.detach().cpu().numpy()
    radii = ref.radii.detach().cpu().numpy()
    examples: dict[str, np.ndarray] = {}
    for row in ip:
        kind = _classify(row)
        examples.setdefault(kind, row)
    missing = sorted({"hold", "ramp_up", "ramp_down"} - set(examples))
    if missing:
        raise SystemExit(f"did not sample required modes: {', '.join(missing)}")

    if not np.allclose(radii, radii[:, :1, :]):
        raise SystemExit("hold_reset_boundary radii changed across the sampled episode")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(cfg.sim.max_episode_steps) + 1, dtype=float) * float(cfg.reference.t_step)
    fig, ax = plt.subplots(figsize=(9, 5))
    for kind in ("hold", "ramp_up", "ramp_down"):
        ax.plot(t, examples[kind] / 1000.0, label=kind)
    ax.set_title("0.1 s single-segment Ip reference examples")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Ip (kA)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "single_segment_ip_examples.png", dpi=160)
    plt.close(fig)
    print(out_dir / "single_segment_ip_examples.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
