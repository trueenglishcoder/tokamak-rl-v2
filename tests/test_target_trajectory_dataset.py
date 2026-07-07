from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokamak_rl_v2.config.machine_envelope import load_machine_envelope
from tokamak_rl_v2.data.target_trajectories import (
    TARGET_FILE,
    WINDOW_STEPS,
    build_target_dataset,
    limiter_radii_at_angles,
)
from tokamak_rl_v2.env.references import T15ReplayBoundaryLibrary


def _write_machine(path: Path) -> Path:
    path.write_text(
        """
machine_envelope:
  name: test_proxy
  verified_geometry:
    limiter_surface: true
    coil_positions: true
  observed_data:
    source: synthetic_test
  proxy_training:
    ip_range_a: [100000.0, 500000.0]
    ip_rate_a_per_s: 500000.0
    boundary_margin_m: 0.05
    boundary_rate_m_per_s: 1.0
    coil_current_range_a:
      - [-1.0, 1.0]
      - [-1.0, 1.0]
    coil_jdot_range_a_per_s:
      - [-1.0, 1.0]
      - [-1.0, 1.0]
  soft_penalties:
    ip_soft_range_a: [100000.0, 500000.0]
    coil_current_soft_fraction: 1.0
    coil_jdot_soft_fraction: 1.0
  termination:
    limiter_invalid_reference: true
    boundary_loss_in_sim: true
    invalid_state: true
    extreme_proxy_actuator_violation: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_seed_libraries(tmp_path: Path) -> tuple[Path, Path]:
    target_path = tmp_path / "seed_targets.npz"
    initial_path = tmp_path / "seed_initial.npz"
    rows = 4
    points = 101
    angles = 32
    ip = np.empty((rows, points), dtype=np.float32)
    radii = np.empty((rows, points, angles), dtype=np.float32)
    theta = np.linspace(-np.pi, np.pi, angles, endpoint=False)
    for row in range(rows):
        ip[row] = 180000.0 + row * 5000.0
        radii[row] = 0.35 + 0.02 * np.cos(theta)[None, :]
    split = np.asarray(["train", "train", "holdout", "holdout"])
    shot_id = np.asarray([3856, 3857, 3863, 3864], dtype=np.int64)
    source_index = np.arange(rows, dtype=np.int64)
    time_s = source_index.astype(float) * 0.001
    np.savez_compressed(
        target_path,
        shot_id=shot_id,
        source_index=source_index,
        time_s=time_s,
        split=split,
        difficulty_bin=np.asarray(["seed"] * rows),
        ip_target=ip,
        boundary_radii=radii,
    )
    np.savez_compressed(
        initial_path,
        shot_id=shot_id,
        source_index=source_index,
        time_s=time_s,
        split=split,
        difficulty_bin=np.asarray(["seed"] * rows),
        ip0=ip[:, 0],
        pfc0=np.zeros((rows, 6), dtype=np.float32),
        sol0=np.zeros((rows, 3), dtype=np.float32),
        params0=np.zeros((rows, 5), dtype=np.float32),
    )
    return target_path, initial_path


def test_machine_envelope_loads_proxy_not_verified_limits(tmp_path: Path) -> None:
    envelope = load_machine_envelope(_write_machine(tmp_path / "machine.yaml"))

    assert envelope.verified_geometry["limiter_surface"] is True
    assert envelope.proxy_training.ip_range_a == (100000.0, 500000.0)


def test_limiter_radii_at_angles_for_square() -> None:
    limiter = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float)
    theta = np.asarray([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])

    radii = limiter_radii_at_angles(limiter, center=np.asarray([0.0, 0.0]), theta=theta)

    assert np.allclose(radii, [1.0, 1.0, 1.0, 1.0])


def test_target_builder_writes_overlapping_windows_and_no_oracle_action(tmp_path: Path) -> None:
    target_seed, initial_seed = _write_seed_libraries(tmp_path)
    machine = _write_machine(tmp_path / "machine.yaml")
    limiter = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float)
    out_dir = tmp_path / "out"

    summary = build_target_dataset(
        target_seed_path=target_seed,
        initial_library_path=initial_seed,
        machine_envelope_path=machine,
        out_dir=out_dir,
        limiter_shape=limiter,
        boundary_center=(0.0, 0.0),
        theta_count=32,
        train_parents=1,
        holdout_parents=1,
        min_steps=105,
        max_steps=105,
        window_steps=WINDOW_STEPS,
        window_stride_steps=1,
        dt=0.001,
        seed=5,
    )

    assert summary.windows == 12
    with np.load(out_dir / TARGET_FILE, allow_pickle=False) as target:
        assert "real_jdot_action" not in target.files
        assert target["ip_target"].shape[1] == 101
        assert target["boundary_radii"].shape[1:] == (101, 32)
        assert np.array_equal(np.unique(target["source_index"]), np.arange(6))
    with np.load(out_dir / "t15_proxy_target_v1_initial_states.npz", allow_pickle=False) as initial:
        assert initial["ip0"].shape[0] == summary.windows
        assert initial["pfc0"].shape[1] == 6
        assert initial["sol0"].shape[1] == 3


def test_target_only_library_replays_without_real_action(tmp_path: Path) -> None:
    target_seed, initial_seed = _write_seed_libraries(tmp_path)
    machine = _write_machine(tmp_path / "machine.yaml")
    limiter = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float)
    out_dir = tmp_path / "out"
    build_target_dataset(
        target_seed_path=target_seed,
        initial_library_path=initial_seed,
        machine_envelope_path=machine,
        out_dir=out_dir,
        limiter_shape=limiter,
        boundary_center=(0.0, 0.0),
        theta_count=32,
        train_parents=1,
        holdout_parents=1,
        min_steps=105,
        max_steps=105,
        window_steps=100,
        window_stride_steps=1,
        dt=0.001,
        seed=7,
    )
    with np.load(out_dir / TARGET_FILE, allow_pickle=False) as target:
        shot = str(int(target["shot_id"][0]))
        source = int(target["source_index"][0])
        expected_ip = np.asarray(target["ip_target"][0], dtype=float)
        expected_radii = np.asarray(target["boundary_radii"][0], dtype=float)

    library = T15ReplayBoundaryLibrary(out_dir, theta_count=32)

    assert np.allclose(library.ip_for_segment(shot, steps=100, source_index=source), expected_ip)
    assert np.allclose(library.radii_for_segment(shot, steps=100, reset_radii=np.ones(32), source_index=source), expected_radii)
    with pytest.raises(ValueError, match="target-only"):
        library.real_action_for_segment(shot, steps=100, source_index=source)
