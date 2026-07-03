#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET_DIR = Path("data/processed/t15_actuator_generated_trim50_plain_gpu1e6_0p1s")
DEFAULT_INITIAL_STATES = Path("data/processed/t15_actuator_generated_trim50_plain_gpu1e6_0p1s_initial_states.npz")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot reconstructed coil-current trajectories used by the actuator-generated T15 dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--initial-states", type=Path, default=DEFAULT_INITIAL_STATES)
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument("--accepted-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rows", type=int, nargs="*", default=None, help="Specific generated rows to plot.")
    parser.add_argument("--per-mode", type=int, default=2, help="Rows per mode when --rows is omitted.")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir
    targets_path = args.targets or dataset_dir / "t15_replay_window_oracle_targets.npz"
    accepted_path = args.accepted_csv or dataset_dir / "actuator_generated_accepted.csv"
    out_dir = args.out_dir or dataset_dir / "coil_trajectory_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (targets_path, args.initial_states, accepted_path):
        if not path.exists():
            raise SystemExit(f"missing required artifact: {path}")

    accepted = _read_accepted(accepted_path)
    rows = _select_rows(accepted, rows=args.rows, per_mode=int(args.per_mode), seed=int(args.seed))
    if not rows:
        raise SystemExit("no rows selected")

    with np.load(targets_path, allow_pickle=False) as targets, np.load(args.initial_states, allow_pickle=False) as initial:
        _validate(targets=targets, initial=initial)
        pfc0 = np.asarray(initial["pfc0"], dtype=float)
        sol0 = np.asarray(initial["sol0"], dtype=float)
        action = np.asarray(targets["real_jdot_action"], dtype=float)
        derivative_limits = np.asarray(targets["derivative_limits"], dtype=float)
        mode = np.asarray(targets["mode"]).astype(str)
        shot_id = np.asarray(targets["shot_id"]).astype(str)
        difficulty = np.asarray(targets["difficulty_bin"]).astype(str)
        scale = np.asarray(targets["scale"], dtype=float)

        combined_path = out_dir / "selected_coil_trajectories.png"
        _plot_combined(
            rows=rows,
            pfc0=pfc0,
            sol0=sol0,
            action=action,
            derivative_limits=derivative_limits,
            mode=mode,
            shot_id=shot_id,
            difficulty=difficulty,
            scale=scale,
            dt=float(args.dt),
            out_path=combined_path,
        )

        for row in rows:
            _plot_single(
                row=row,
                pfc0=pfc0,
                sol0=sol0,
                action=action,
                derivative_limits=derivative_limits,
                mode=mode,
                shot_id=shot_id,
                difficulty=difficulty,
                scale=scale,
                dt=float(args.dt),
                out_path=out_dir / f"coil_trajectory_row_{row:05d}.png",
            )

    print(combined_path)
    print(f"wrote {len(rows)} single-row plots to {out_dir}")
    print("selected rows:", " ".join(str(row) for row in rows))
    return 0


def _read_accepted(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _select_rows(accepted: list[dict[str, str]], *, rows: list[int] | None, per_mode: int, seed: int) -> list[int]:
    if rows:
        return [int(row) for row in rows]
    by_mode: dict[str, list[int]] = {}
    for row in accepted:
        by_mode.setdefault(row["mode"], []).append(int(row["row"]))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for mode in sorted(by_mode):
        candidates = np.asarray(by_mode[mode], dtype=int)
        count = min(int(per_mode), int(candidates.shape[0]))
        if count <= 0:
            continue
        choice = rng.choice(candidates, size=count, replace=False)
        selected.extend(int(x) for x in np.sort(choice))
    return selected


def _validate(*, targets: np.lib.npyio.NpzFile, initial: np.lib.npyio.NpzFile) -> None:
    target_rows = int(targets["real_jdot_action"].shape[0])
    initial_rows = int(initial["pfc0"].shape[0])
    if target_rows != initial_rows:
        raise SystemExit(f"target/reset row mismatch: {target_rows} != {initial_rows}")
    if targets["real_jdot_action"].shape[1:] != (100, 9):
        raise SystemExit(f"expected action shape [N,100,9], got {targets['real_jdot_action'].shape}")


def _currents_for_row(
    *,
    row: int,
    pfc0: np.ndarray,
    sol0: np.ndarray,
    action: np.ndarray,
    derivative_limits: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    current0 = np.concatenate([pfc0[row], sol0[row]], axis=0)
    jdot = action[row] * derivative_limits[None, :]
    delta = np.concatenate([np.zeros((1, 9), dtype=float), np.cumsum(jdot * dt, axis=0)], axis=0)
    currents = current0[None, :] + delta
    return currents, jdot


def _plot_single(
    *,
    row: int,
    pfc0: np.ndarray,
    sol0: np.ndarray,
    action: np.ndarray,
    derivative_limits: np.ndarray,
    mode: np.ndarray,
    shot_id: np.ndarray,
    difficulty: np.ndarray,
    scale: np.ndarray,
    dt: float,
    out_path: Path,
) -> None:
    currents, jdot = _currents_for_row(
        row=row,
        pfc0=pfc0,
        sol0=sol0,
        action=action,
        derivative_limits=derivative_limits,
        dt=dt,
    )
    t = np.arange(currents.shape[0], dtype=float) * dt
    tj = np.arange(jdot.shape[0], dtype=float) * dt
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False, constrained_layout=True)
    fig.suptitle(
        f"row {row} | shot {shot_id[row]} | {mode[row]} | {difficulty[row]} | scale={scale[row]:.3g}",
        fontsize=13,
    )
    _plot_currents(axes[0], t, currents[:, :6], "PFC currents", "I_pfc [A]", [f"PFC{i}" for i in range(6)])
    _plot_currents(axes[1], t, currents[:, 6:], "SOL currents", "I_sol [A]", [f"SOL{i}" for i in range(3)])
    _plot_currents(axes[2], tj, jdot[:, :6], "PFC Jdot commands", "dI_pfc/dt [A/s]", [f"PFC{i}" for i in range(6)])
    _plot_currents(axes[3], tj, action[row], "Normalized Jdot commands", "action [-1,1]", [f"C{i}" for i in range(9)])
    axes[3].set_xlabel("time [s]")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_combined(
    *,
    rows: list[int],
    pfc0: np.ndarray,
    sol0: np.ndarray,
    action: np.ndarray,
    derivative_limits: np.ndarray,
    mode: np.ndarray,
    shot_id: np.ndarray,
    difficulty: np.ndarray,
    scale: np.ndarray,
    dt: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(16, max(3.0 * len(rows), 6.0)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.asarray([axes])
    for ax_row, row in zip(axes, rows, strict=True):
        currents, jdot = _currents_for_row(
            row=row,
            pfc0=pfc0,
            sol0=sol0,
            action=action,
            derivative_limits=derivative_limits,
            dt=dt,
        )
        t = np.arange(currents.shape[0], dtype=float) * dt
        tj = np.arange(jdot.shape[0], dtype=float) * dt
        title = f"row {row} | shot {shot_id[row]} | {mode[row]} | {difficulty[row]} | scale={scale[row]:.3g}"
        _plot_currents(ax_row[0], t, currents[:, :6], title, "PFC [A]", [f"PFC{i}" for i in range(6)], legend=False)
        _plot_currents(ax_row[1], t, currents[:, 6:], "SOL currents", "SOL [A]", [f"SOL{i}" for i in range(3)], legend=False)
        ax_row[0].set_xlabel("time [s]")
        ax_row[1].set_xlabel("time [s]")
        ax_row[1].text(
            0.98,
            0.06,
            f"action rms={np.sqrt(np.mean(action[row] * action[row])):.3f}",
            transform=ax_row[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
        _ = tj
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_currents(
    ax: plt.Axes,
    t: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    labels: list[str],
    *,
    legend: bool = True,
) -> None:
    for i, label in enumerate(labels):
        ax.plot(t, values[:, i], linewidth=1.4, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(loc="best", fontsize=8, ncol=min(len(labels), 3))


if __name__ == "__main__":
    raise SystemExit(main())
