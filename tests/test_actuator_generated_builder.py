from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_t15_actuator_generated_trim50_plain_gpu1e6_0p1s.py"
    spec = importlib.util.spec_from_file_location("build_t15_actuator_generated_trim50_plain_gpu1e6_0p1s", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_window(builder, *, shot: str = "3856", split: str = "train", start: int = 17, offset: float = 0.0):
    steps = 100
    t = np.arange(steps + 1, dtype=float)
    currents = np.zeros((steps + 1, 9), dtype=float)
    currents[:, 0] = offset + 50_000.0 + 100.0 * t
    currents[:, 1] = offset - 20_000.0 - 50.0 * t
    currents[:, 6] = offset + 500_000.0 + 400.0 * t
    return builder.RealWindow(
        shot_id=shot,
        split=split,
        start=start,
        time_s=0.05,
        ip=200_000.0 + 50.0 * t,
        currents=currents,
    )


def test_sample_coil_candidates_are_reset_anchored_and_within_action_limits() -> None:
    builder = _load_builder_module()
    windows = [
        _fake_window(builder, shot="3856", split="train", start=17, offset=0.0),
        _fake_window(builder, shot="3857", split="train", start=23, offset=10_000.0),
    ]
    limits = builder.Limits(
        pfc_current=1.0e7,
        sol_current=1.0e7,
        pfc_deriv=2.0e6,
        sol_deriv=2.0e6,
    )

    out = builder._sample_coil_candidates(
        windows,
        count=8,
        rng=np.random.default_rng(123),
        limits=limits,
        steps=100,
    )

    assert len(out) == 8
    assert {c.mode for c in out} >= {"exact_real", "scaled_same", "scaled_same_channel_jitter", "borrowed_motion"}
    for candidate in out:
        assert candidate.currents.shape == (101, 9)
        assert candidate.action.shape == (100, 9)
        assert candidate.currents[0] == pytest.approx(candidate.reset.currents[0])
        assert float(np.max(np.abs(candidate.action))) <= 1.0001


def test_write_libraries_uses_unique_generated_source_indices(tmp_path: Path) -> None:
    builder = _load_builder_module()
    limits = builder.Limits(
        pfc_current=1.0e7,
        sol_current=1.0e7,
        pfc_deriv=2.0e6,
        sol_deriv=2.0e6,
    )
    base_row = {
        "shot_id": "3856",
        "split": "train",
        "source_index": 17,
        "time_s": 0.05,
        "ip0": 200_000.0,
        "pfc0": np.zeros((6,), dtype=np.float32),
        "sol0": np.zeros((3,), dtype=np.float32),
        "ip_target": np.linspace(200_000.0, 210_000.0, 101, dtype=np.float32),
        "boundary_radii": np.ones((101, 32), dtype=np.float32) * 0.55,
        "real_jdot_action": np.zeros((100, 9), dtype=np.float32),
        "difficulty_bin": "medium_up",
        "mode": "exact_real",
        "motion_shot_id": "3856",
        "motion_source_index": 17,
        "scale": 1.0,
        "state_feature_distance": 0.0,
        "oracle_ip_mean_error_a": 0.0,
        "oracle_ip_max_error_a": 0.0,
    }
    rows = [dict(base_row), dict(base_row)]
    rows[1]["ip0"] = 201_000.0

    initial = tmp_path / "initial.npz"
    targets = tmp_path / "targets" / "t15_replay_window_oracle_targets.npz"
    diagnostics = tmp_path / "targets" / "t15_actuator_generated_targets.npz"
    builder._write_libraries(
        rows,
        initial_states_out=initial,
        targets_out=targets,
        diagnostic_targets_out=diagnostics,
        limits=limits,
        train_shots=("3856",),
        holdout_shots=("3864",),
    )

    with np.load(initial, allow_pickle=False) as data:
        assert data["source_index"].tolist() == [0, 1]
        assert data["reset_source_index"].tolist() == [17, 17]

    with np.load(targets, allow_pickle=False) as data:
        assert data["schema"].item() == "t15_replay_window_oracle_targets_v1"
        assert data["source_index"].tolist() == [0, 1]
        assert data["reset_source_index"].tolist() == [17, 17]
        assert data["ip_target"].shape == (2, 101)
        assert data["boundary_radii"].shape == (2, 101, 32)
        assert data["real_jdot_action"].shape == (2, 100, 9)

    with np.load(diagnostics, allow_pickle=False) as data:
        assert data["schema"].item() == "t15_actuator_generated_trim50_plain_gpu1e6_0p1s_targets_v1"


def test_rollout_filter_rejects_outside_observed_state_envelope() -> None:
    builder = _load_builder_module()
    envelope = builder.ObservedEnvelope(
        ip_min=100_000.0,
        ip_max=300_000.0,
        radii_min=np.zeros((32,), dtype=float) + 0.4,
        radii_max=np.zeros((32,), dtype=float) + 0.8,
        current_min=np.zeros((9,), dtype=float) - 1.0e6,
        current_max=np.zeros((9,), dtype=float) + 1.0e6,
        feature_values=np.zeros((1, 4), dtype=float),
    )

    ok, reason, _ = builder._rollout_ok(
        ip=np.linspace(150_000.0, 200_000.0, 101),
        radii=np.ones((101, 32), dtype=float) * 0.55,
        found=np.ones((101,), dtype=bool),
        currents=np.zeros((101, 9), dtype=float),
        envelope=envelope,
        state_feature_distance_limit=0.0,
    )
    assert ok
    assert reason == "ok"

    ok, reason, _ = builder._rollout_ok(
        ip=np.linspace(150_000.0, 350_000.0, 101),
        radii=np.ones((101, 32), dtype=float) * 0.55,
        found=np.ones((101,), dtype=bool),
        currents=np.zeros((101, 9), dtype=float),
        envelope=envelope,
        state_feature_distance_limit=0.0,
    )
    assert not ok
    assert reason == "ip_outside_observed_envelope"
