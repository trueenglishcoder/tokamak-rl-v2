from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.build_t15_synthetic_long30_trim50_plain_gpu1e6_oracle_windows import (
    WINDOW_STEPS,
    _windows_from_parent,
)
from scripts.build_t15_synthetic_long_preview import _check_boundary_motion, _parent_mode_for_index, _sample_ip_profile


def test_windows_from_parent_uses_overlapping_100_step_windows() -> None:
    steps = 105
    current_limits = np.full((9,), 1000.0, dtype=float)
    derivative_limits = np.full((9,), 1.0e6, dtype=float)
    parent = {
        "steps": steps,
        "ip": np.linspace(100000.0, 101050.0, steps + 1),
        "radii": np.full((steps + 1, 32), 0.6, dtype=float),
        "currents": np.zeros((steps + 1, 9), dtype=float),
    }
    parent["currents"][:, 0] = np.arange(steps + 1, dtype=float)
    real = SimpleNamespace(current_limits=current_limits, derivative_limits=derivative_limits)

    rows = _windows_from_parent(parent_id=900000, split="train", parent=parent, real=real)

    assert len(rows) == 6
    assert rows[0]["source_index"] == 0
    assert rows[-1]["source_index"] == steps - WINDOW_STEPS
    assert rows[0]["ip_target"].shape == (WINDOW_STEPS + 1,)
    assert rows[0]["boundary_radii"].shape == (WINDOW_STEPS + 1, 32)
    assert rows[0]["real_jdot_action"].shape == (WINDOW_STEPS, 9)
    assert rows[0]["pfc0"].shape == (6,)
    assert rows[0]["sol0"].shape == (3,)
    assert rows[-1]["ip_target"][0] == parent["ip"][steps - WINDOW_STEPS]


def test_synthetic_long_ip_profile_is_piecewise_ramp_or_hold() -> None:
    real = SimpleNamespace(
        feature_low=np.asarray([150000.0]),
        feature_high=np.asarray([420000.0]),
        ip_rate_abs_p99=900000.0,
    )
    rng = np.random.default_rng(123)

    for _ in range(32):
        profile, edges = _sample_ip_profile(real=real, start_ip=260000.0, steps=1200, rng=rng)
        diff = np.diff(profile)
        active = np.abs(diff) > 1.0
        signs = np.sign(diff[active])
        sign_changes = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0

        assert profile.shape == (1201,)
        assert edges[0] == 0
        assert edges[-1] == 1200
        assert np.all(profile >= real.feature_low[0])
        assert np.all(profile <= real.feature_high[0])
        assert sign_changes <= 1


def test_synthetic_long_ip_profile_rounds_segment_bends() -> None:
    real = SimpleNamespace(
        feature_low=np.asarray([150000.0]),
        feature_high=np.asarray([420000.0]),
        ip_rate_abs_p99=900000.0,
    )
    rng = np.random.default_rng(11)

    profile, _ = _sample_ip_profile(real=real, start_ip=260000.0, steps=1200, mode="ramp_hold", rng=rng)
    slope = np.diff(profile)
    slope_jump = np.abs(np.diff(slope))

    assert float(np.max(slope_jump)) < 0.25 * float(np.max(np.abs(slope)))


def test_synthetic_long_preview_cycles_named_parent_modes() -> None:
    assert [_parent_mode_for_index(i) for i in range(6)] == [
        "ramp_hold",
        "hold_ramp",
        "ramp_hold_reverse",
        "ramp_rate_change",
        "hold_ramp_hold",
        "ramp",
    ]


def test_synthetic_long_boundary_motion_rejects_flat_parent() -> None:
    real = SimpleNamespace(
        radii_mean_range_p70=0.02,
        radii_angle_range_p70=0.025,
        radii_mean_safe_span=0.25,
        radii_angle_safe_span=0.20,
    )
    flat = np.full((1201, 32), 0.6, dtype=float)
    moving = flat.copy()
    moving += np.linspace(0.0, 0.10, moving.shape[0]).reshape(-1, 1)

    assert _check_boundary_motion(real, flat) == "boundary_motion_too_small"
    assert _check_boundary_motion(real, moving) is None
