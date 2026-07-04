from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.build_t15_synthetic_long30_trim50_plain_gpu1e6_oracle_windows import (
    WINDOW_STEPS,
    _windows_from_parent,
)


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
