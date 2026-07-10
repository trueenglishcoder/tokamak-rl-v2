from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_t15_realstyle_simple_long60 import (
    ALLOWED_FAMILIES,
    WINDOW_STEPS,
    _generate_parent,
    _load_build_space,
    _project_features,
    _windows_from_parent,
    main,
)
from tokamak_rl_v2.config import load_experiment_config


def _write_fake_real_library(tmp_path: Path) -> tuple[Path, Path]:
    target_path = tmp_path / "targets" / "t15_replay_window_oracle_targets.npz"
    initial_path = tmp_path / "initial_states.npz"
    target_path.parent.mkdir(parents=True)

    rows = 24
    points = WINDOW_STEPS + 1
    steps = WINDOW_STEPS
    angles = 32
    coils = 9
    theta = np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False)
    split = np.asarray(["train"] * 16 + ["holdout"] * 8)
    shot_id = np.asarray([3856] * 8 + [3857] * 8 + [3863] * 8, dtype=np.int64)
    source_index = np.asarray(list(range(12)) + [510, 511, 512, 513] + list(range(8)), dtype=np.int64)
    time_s = source_index.astype(np.float64) * 0.001

    ip = np.empty((rows, points), dtype=np.float32)
    radii = np.empty((rows, points, angles), dtype=np.float32)
    action = np.empty((rows, steps, coils), dtype=np.float32)
    pfc0 = np.empty((rows, 6), dtype=np.float32)
    sol0 = np.empty((rows, 3), dtype=np.float32)
    for r in range(rows):
        t = np.linspace(0.0, 1.0, points)
        if r % 4 == 0:
            shape = np.r_[np.linspace(0.0, 1.0, 42), np.ones(28), np.linspace(1.0, 0.2, 31)]
        elif r % 4 == 1:
            shape = np.r_[np.linspace(0.0, 1.0, 58), np.ones(43)]
        elif r % 4 == 2:
            shape = np.r_[np.ones(45), np.linspace(1.0, 0.1, 56)]
        else:
            shape = np.linspace(0.0, 1.0, points)
        ip[r] = 170000.0 + 6500.0 * r + (18000.0 + 600.0 * r) * shape
        mean = 0.55 + 0.004 * r + 0.018 * np.sin(np.pi * t + 0.15 * r)
        amp = 0.060 + 0.005 * np.cos(np.pi * t + 0.1 * r)
        tri = 0.018 * np.sin(0.5 * np.pi * t + 0.2 * r)
        radii[r] = mean[:, None] + amp[:, None] * np.cos(2.0 * theta)[None, :] + tri[:, None] * np.sin(theta)[None, :]
        base = np.asarray([-60000, 80000, 50000, 60000, 120000, -250000, 800000, 2500000, 900000], dtype=float)
        pfc0[r] = base[:6] + 1800.0 * r
        sol0[r] = base[6:] + 4500.0 * r
        phase = 0.1 * r + np.arange(coils)[None, :] / 9.0
        action[r] = 0.015 * np.sin(2.0 * np.pi * t[:-1, None] + phase)

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


def _args(**kwargs):
    defaults = dict(
        pca_components=5,
        wiggle_room=1.15,
        radii_margin_m=0.025,
        current_envelope_margin=0.08,
        max_cloud_rows=4000,
        reset_max_source_index=500,
        min_steps=120,
        max_steps=120,
        knn_k=4,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _space(tmp_path: Path, split: str = "train"):
    target_path, initial_path = _write_fake_real_library(tmp_path)
    return _load_build_space(
        target_path=target_path,
        initial_path=initial_path,
        split=split,
        args=_args(),
        rng=np.random.default_rng(5),
    )


def test_parent_starts_from_first_500_real_reset_and_uses_allowed_family(tmp_path: Path) -> None:
    space = _space(tmp_path)
    parent = _generate_parent(
        space=space,
        parent_id=960000,
        split="train",
        min_steps=120,
        max_steps=120,
        knn_k=4,
        rng=np.random.default_rng(12),
    )

    assert 0 <= parent["reset_source_index"] < 500
    assert parent["ip_family"] in ALLOWED_FAMILIES
    assert parent["steps"] == 120
    assert parent["ip"].shape == (121,)
    assert parent["radii"].shape == (121, 32)
    assert parent["currents"].shape == (121, 9)
    assert np.any(np.all(np.isclose(space.reset_pool.features, parent["features"][0]), axis=1))
    assert np.any(np.all(np.isclose(space.reset_pool.currents, parent["currents"][0]), axis=1))


def test_parent_respects_safe_state_current_and_jdot_limits(tmp_path: Path) -> None:
    space = _space(tmp_path)
    parent = _generate_parent(
        space=space,
        parent_id=960001,
        split="train",
        min_steps=130,
        max_steps=130,
        knn_k=4,
        rng=np.random.default_rng(13),
    )

    assert np.all(parent["features"] >= space.real.feature_low.reshape(1, -1) - 1e-6)
    assert np.all(parent["features"] <= space.real.feature_high.reshape(1, -1) + 1e-6)
    assert np.all(parent["radii"] >= space.real.radii_low.reshape(1, -1) - 1e-6)
    assert np.all(parent["radii"] <= space.real.radii_high.reshape(1, -1) + 1e-6)
    assert np.max(np.abs(parent["currents"]) / space.real.current_limits.reshape(1, -1)) <= 1.0 + 1e-6
    assert np.max(np.abs(parent["real_jdot_action"])) <= 1.0 + 1e-6


def test_feature_projection_scales_motion_instead_of_rejecting() -> None:
    space = SimpleNamespace(
        real=SimpleNamespace(
            feature_low=np.asarray([-2.0], dtype=np.float64),
            feature_high=np.asarray([1.5], dtype=np.float64),
        )
    )
    raw = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    projected, scale = _project_features(features=raw, start_feature=np.asarray([0.0]), space=space)

    assert np.isclose(scale, 0.5)
    assert np.allclose(projected[:, 0], [0.0, 0.5, 1.0, 1.5])


def test_cutting_writes_all_overlapping_100_step_windows(tmp_path: Path) -> None:
    space = _space(tmp_path)
    parent = _generate_parent(
        space=space,
        parent_id=960002,
        split="train",
        min_steps=105,
        max_steps=105,
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


def test_tiny_build_writes_realstyle_schema(tmp_path: Path) -> None:
    target_path, initial_path = _write_fake_real_library(tmp_path)
    out_dir = tmp_path / "out"
    initial_out = tmp_path / "initial_out.npz"

    main(
        [
            "--oracle-target",
            str(target_path),
            "--initial-library",
            str(initial_path),
            "--out-dir",
            str(out_dir),
            "--initial-library-out",
            str(initial_out),
            "--train-parents",
            "2",
            "--holdout-parents",
            "1",
            "--preview-examples",
            "0",
            "--min-steps",
            "120",
            "--max-steps",
            "120",
            "--max-cloud-rows",
            "4000",
            "--knn-k",
            "4",
        ]
    )

    assert (out_dir / "t15_replay_window_oracle_targets.npz").is_file()
    assert initial_out.is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "report.md").is_file()
    with np.load(out_dir / "t15_replay_window_oracle_targets.npz", allow_pickle=False) as target:
        assert target["ip_target"].shape[1] == 101
        assert target["boundary_radii"].shape[1:] == (101, 32)
        assert target["real_jdot_action"].shape[1:] == (100, 9)


def test_realstyle_config_keeps_successful_real_training_contract() -> None:
    cfg = load_experiment_config(
        Path(__file__).resolve().parents[1]
        / "configs/experiments/t15_realstyle_simple_long60_0p1s_tcvjdot_mpo_balanced.yaml"
    )

    assert cfg.sim.max_episode_steps == 100
    assert cfg.sim.action_contract == "jdot_command"
    assert cfg.reference.ip.kind == "replay_window"
    assert cfg.reference.boundary.kind == "t15_replay_segment_conditioned"
    assert cfg.observation.actor_kind == "controller_state_v6"
    assert cfg.observation.critic_kind == "compact_training_state_v2"
    assert cfg.reward.kind == "tcv_derivative"
    assert cfg.reward.ip_scale_a == 25000.0
    assert cfg.learner.rollout_chunk_length == 100
    assert cfg.learner.updates_per_rollout_chunk == 64
