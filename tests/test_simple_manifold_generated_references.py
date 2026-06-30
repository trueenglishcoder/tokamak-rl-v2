from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_t15_simple_manifold_generated_trim50_idealized_0p1s.py"
    spec = importlib.util.spec_from_file_location("build_t15_simple_manifold_generated_trim50_idealized_0p1s", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_piecewise_linear_blends_only_internal_joins_and_never_episode_edges() -> None:
    builder = _load_builder_module()
    out = builder._piecewise_linear(
        [0, 40, 100],
        [np.asarray([0.0]), np.asarray([0.0]), np.asarray([60.0])],
        steps=100,
        join_blend_steps=4,
    ).reshape(-1)

    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(60.0)
    assert out[-1] - out[-2] == pytest.approx(1.0)
    assert np.all(np.diff(out) >= -1.0e-12)


def test_piecewise_linear_ramp_then_ramp_keeps_final_slope() -> None:
    builder = _load_builder_module()
    out = builder._piecewise_linear(
        [0, 50, 100],
        [np.asarray([0.0]), np.asarray([20.0]), np.asarray([70.0])],
        steps=100,
        join_blend_steps=6,
    ).reshape(-1)

    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(70.0)
    assert out[-1] - out[-2] == pytest.approx(1.0)
    assert out[1] - out[0] == pytest.approx(0.4)


def test_piecewise_linear_ramp_then_hold_does_not_overshoot_hold_value() -> None:
    builder = _load_builder_module()
    out = builder._piecewise_linear(
        [0, 70, 100],
        [np.asarray([410.0]), np.asarray([398.5]), np.asarray([398.5])],
        steps=100,
        join_blend_steps=6,
    ).reshape(-1)

    assert out[0] == pytest.approx(410.0)
    assert out[-1] == pytest.approx(398.5)
    assert np.min(out) == pytest.approx(398.5)
    assert out[-1] - out[-2] == pytest.approx(0.0)
