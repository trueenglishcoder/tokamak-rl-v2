from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.training.policy_pipeline import _preflight_artifact_failure


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


def test_simple_manifold_reset_library_mode_metadata_passes_artifact_preflight(tmp_path: Path) -> None:
    reset_path = tmp_path / "simple_initial_states.npz"
    np.savez_compressed(
        reset_path,
        schema=np.asarray("t15_simple_manifold_generated_trim50_idealized_matched_initial_states_v1"),
        shot_id=np.asarray(["3856", "3864"]),
        source_index=np.asarray([0, 1], dtype=np.int64),
        time_s=np.asarray([0.05, 0.06], dtype=float),
        ip0=np.asarray([200000.0, 210000.0], dtype=float),
        pfc0=np.zeros((2, 6), dtype=float),
        sol0=np.zeros((2, 3), dtype=float),
        params0=np.asarray([[1.5, 0.0, 0.5, 1.2, 0.1], [1.5, 0.0, 0.5, 1.2, 0.1]], dtype=float),
        split=np.asarray(["train", "holdout"]),
        difficulty_bin=np.asarray(["core", "core"]),
        mode=np.asarray(["ramp", "hold_then_ramp"]),
    )

    cfg = load_experiment_config("configs/experiments/t15_simple_manifold_generated_trim50_idealized_matched_0p1s_tcvjdot_balanced_mpo.yaml")
    cfg = replace(cfg, sim=replace(cfg.sim, csv_initial_state_library=str(reset_path)))

    assert _preflight_artifact_failure(cfg) is None
