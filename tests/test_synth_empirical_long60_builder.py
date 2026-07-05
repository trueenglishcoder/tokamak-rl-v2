from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_t15_synth_empirical_long60 import (
    WINDOW_STEPS,
    _generate_parent,
    _infer_currents,
    _load_empirical_space,
    _project_features,
    _windows_from_parent,
    main,
)


def _write_fake_real_library(tmp_path: Path) -> tuple[Path, Path]:
    target_path = tmp_path / "targets" / "t15_replay_window_oracle_targets.npz"
    initial_path = tmp_path / "initial_states.npz"
    target_path.parent.mkdir(parents=True)

    rows = 18
    points = 101
    steps = 100
    angles = 32
    coils = 9
    theta = np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False)
    split = np.asarray(["train"] * 12 + ["holdout"] * 6)
    shot_id = np.asarray([3856] * 6 + [3857] * 6 + [3863] * 6, dtype=np.int64)
    source_index = np.arange(rows, dtype=np.int64)
    time_s = source_index.astype(float) * 0.001

    ip = np.empty((rows, points), dtype=np.float32)
    radii = np.empty((rows, points, angles), dtype=np.float32)
    action = np.empty((rows, steps, coils), dtype=np.float32)
    pfc0 = np.empty((rows, 6), dtype=np.float32)
    sol0 = np.empty((rows, 3), dtype=np.float32)
    for r in range(rows):
        t = np.linspace(0.0, 1.0, points)
        slope = (-1.0 if r % 3 == 0 else 1.0) * (15000.0 + 500.0 * r)
        ip[r] = 180000.0 + 6000.0 * r + slope * t + 1200.0 * np.sin(2.0 * np.pi * t + 0.2 * r)
        mean = 0.58 + 0.004 * r + 0.012 * np.sin(2.0 * np.pi * t + 0.13 * r)
        amp = 0.055 + 0.003 * np.cos(2.0 * np.pi * t + 0.1 * r)
        tri = 0.014 * np.sin(np.pi * t + 0.3 * r)
        radii[r] = mean[:, None] + amp[:, None] * np.cos(2.0 * theta)[None, :] + tri[:, None] * np.sin(theta)[None, :]
        base = np.asarray([-60000, 80000, 50000, 60000, 120000, -250000, 800000, 2500000, 900000], dtype=float)
        pfc0[r] = base[:6] + 2000.0 * r
        sol0[r] = base[6:] + 5000.0 * r
        action[r] = 0.02 * np.sin(2.0 * np.pi * t[:-1, None] * (1.0 + np.arange(coils)[None, :] / 10.0) + 0.1 * r)

    current_limits = np.asarray([5e5, 5e5, 5e5, 5e5, 5e5, 1.0e6, 4.0e6, 1.0e7, 4.0e6], dtype=np.float32)
    derivative_limits = np.asarray([2.0e6] * 6 + [1.0e7] * 3, dtype=np.float32)
    np.savez_compressed(
        target_path,
        schema=np.asarray(["t15_replay_window_oracle_targets_v1"]),
        shot_id=shot_id,
        split=split,
        source_index=source_index,
        time_s=time_s,
        difficulty_bin=np.asarray(["flat"] * rows),
        ip0=ip[:, 0],
        pfc0=pfc0,
        sol0=sol0,
        ip_target=ip,
        boundary_radii=radii,
        real_jdot_action=action,
        oracle_ip_mean_error_a=np.zeros(rows, dtype=np.float32),
        oracle_ip_max_error_a=np.zeros(rows, dtype=np.float32),
        current_limits=current_limits,
        derivative_limits=derivative_limits,
    )
    np.savez_compressed(
        initial_path,
        shot_id=shot_id,
        source_index=source_index,
        time_s=time_s,
        ip0=ip[:, 0],
        pfc0=pfc0,
        sol0=sol0,
        split=split,
        difficulty_bin=np.asarray(["flat"] * rows),
    )
    return target_path, initial_path


def _space(tmp_path: Path):
    target_path, initial_path = _write_fake_real_library(tmp_path)
    args = SimpleNamespace(
        dt=0.001,
        pca_components=5,
        wiggle_room=1.15,
        radii_margin_m=0.025,
        current_envelope_margin=0.08,
        max_cloud_rows=4000,
    )
    rng = np.random.default_rng(5)
    return _load_empirical_space(target_path=target_path, initial_path=initial_path, split="train", args=args, rng=rng)


def test_empirical_parent_starts_from_real_reset_and_has_no_mode(tmp_path: Path) -> None:
    space = _space(tmp_path)
    rng = np.random.default_rng(12)
    parent = _generate_parent(
        space=space,
        parent_id=940000,
        split="train",
        min_steps=1000,
        max_steps=1000,
        dt=0.001,
        knn_k=4,
        rng=rng,
    )

    assert parent["steps"] == 1000
    assert "mode" not in parent
    assert parent["ip"].shape == (1001,)
    assert parent["radii"].shape == (1001, 32)
    assert parent["currents"].shape == (1001, 9)
    assert np.any(np.all(np.isclose(space.reset_features, parent["features"][0]), axis=1))
    assert np.any(np.all(np.isclose(space.reset_currents, parent["currents"][0]), axis=1))


def test_empirical_parent_respects_current_and_jdot_limits(tmp_path: Path) -> None:
    space = _space(tmp_path)
    parent = _generate_parent(
        space=space,
        parent_id=940001,
        split="train",
        min_steps=1000,
        max_steps=1000,
        dt=0.001,
        knn_k=4,
        rng=np.random.default_rng(13),
    )

    currents = parent["currents"]
    action = parent["real_jdot_action"]
    assert np.max(np.abs(currents) / space.real.current_limits.reshape(1, -1)) <= 1.0 + 1e-6
    assert np.max(np.abs(action)) <= 1.0 + 1e-6


def test_feature_projection_scales_instead_of_clipping() -> None:
    space = SimpleNamespace(
        feature_low=np.asarray([-2.0], dtype=np.float64),
        feature_high=np.asarray([1.5], dtype=np.float64),
    )
    raw = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    projected, scale = _project_features(features=raw, start_feature=np.asarray([0.0]), space=space)

    assert np.isclose(scale, 0.5)
    assert np.allclose(projected[:, 0], [0.0, 0.5, 1.0, 1.5])
    assert np.allclose(np.diff(projected[:, 0]), [0.5, 0.5, 0.5])


def test_current_inference_smooths_neighbor_identity_jumps() -> None:
    features = np.zeros((301, 2), dtype=np.float64)
    velocities = np.zeros((300, 2), dtype=np.float64)
    current_a = np.zeros((1, 9), dtype=np.float64)
    current_b = np.full((1, 9), 900000.0, dtype=np.float64)
    space = SimpleNamespace(
        real=SimpleNamespace(
            feature_center=np.zeros(2, dtype=np.float64),
            feature_scale=np.ones(2, dtype=np.float64),
        ),
        velocity_center=np.zeros(2, dtype=np.float64),
        velocity_scale=np.ones(2, dtype=np.float64),
        key_center=np.zeros(4, dtype=np.float64),
        key_scale=np.ones(4, dtype=np.float64),
        knn_keys=np.asarray([[0.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        knn_currents=np.concatenate([current_a, current_b], axis=0),
    )
    features[:150, 0] = 0.0
    features[150:, 0] = 100.0

    currents = _infer_currents(
        space=space,
        features=features,
        velocities=velocities,
        start_current=np.zeros(9, dtype=np.float64),
        knn_k=1,
    )
    jdot = np.diff(currents[:, 0])

    assert np.max(np.abs(jdot)) < 40000.0
    assert currents[0, 0] == 0.0
    assert currents[-1, 0] > 850000.0


def test_empirical_cutting_uses_overlapping_100_step_windows(tmp_path: Path) -> None:
    space = _space(tmp_path)
    parent = _generate_parent(
        space=space,
        parent_id=940002,
        split="train",
        min_steps=105,
        max_steps=105,
        dt=0.001,
        knn_k=4,
        rng=np.random.default_rng(14),
    )
    rows = _windows_from_parent(parent=parent, current_limits=space.real.current_limits)

    assert len(rows) == 6
    assert rows[0]["source_index"] == 0
    assert rows[-1]["source_index"] == 105 - WINDOW_STEPS
    assert rows[0]["ip_target"].shape == (101,)
    assert rows[0]["boundary_radii"].shape == (101, 32)
    assert rows[0]["real_jdot_action"].shape == (100, 9)


def test_empirical_tiny_build_writes_expected_files(tmp_path: Path) -> None:
    target_path, initial_path = _write_fake_real_library(tmp_path)
    out_dir = tmp_path / "out"

    main(
        [
            "--oracle-target",
            str(target_path),
            "--initial-library",
            str(initial_path),
            "--out-dir",
            str(out_dir),
            "--train-parents",
            "2",
            "--holdout-parents",
            "1",
            "--preview-examples",
            "0",
            "--min-steps",
            "1000",
            "--max-steps",
            "1000",
            "--max-cloud-rows",
            "4000",
            "--knn-k",
            "4",
        ]
    )

    assert (out_dir / "t15_replay_window_oracle_targets.npz").is_file()
    assert (out_dir / "t15_synth_empirical_long60_initial_states.npz").is_file()
    assert (out_dir / "t15_synth_empirical_long60_report.md").is_file()
    with np.load(out_dir / "t15_replay_window_oracle_targets.npz", allow_pickle=False) as target:
        assert target["ip_target"].shape[1] == 101
        assert target["boundary_radii"].shape[1:] == (101, 32)
        assert target["real_jdot_action"].shape[1:] == (100, 9)
