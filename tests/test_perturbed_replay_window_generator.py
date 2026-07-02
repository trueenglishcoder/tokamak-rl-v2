from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_t15_replay_window_perturbed_trim50_idealized_0p1s.py"
    spec = importlib.util.spec_from_file_location("build_t15_replay_window_perturbed_trim50_idealized_0p1s", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_window(builder):
    steps = 100
    t = np.linspace(0.0, 1.0, steps + 1)
    ip = 220000.0 + 50000.0 * t
    a0 = 0.50 + 0.04 * t
    kappa = 1.20 + 0.05 * t
    delta = 0.10 + 0.03 * t
    x = np.column_stack([ip, a0, kappa - 1.0, delta])
    params = np.column_stack(
        [
            np.full_like(t, 1.50),
            np.full_like(t, 0.02),
            a0,
            kappa,
            delta,
        ]
    )
    coils = np.zeros((steps + 1, 9), dtype=float)
    return builder.simple.ReplayWindow(
        shot="3856",
        start_row=12,
        source_index=12,
        time_s=0.012,
        split="train",
        x=x,
        params=params,
        coils=coils,
    )


def test_smooth_fractional_noise_is_bounded_and_anchored() -> None:
    builder = _load_builder_module()
    rng = np.random.default_rng(7)
    frac = builder._smooth_fractional_noise(
        rng,
        steps=100,
        dims=4,
        max_fraction=0.05,
        knot_count=6,
        smooth_window=9,
    )

    assert frac.shape == (101, 4)
    assert float(np.max(np.abs(frac))) <= 0.050000000001
    assert frac[0] == pytest.approx(np.zeros(4))
    # The default knots are low-frequency; a single step cannot carry the full 5% jump.
    assert float(np.max(np.abs(np.diff(frac, axis=0)))) < 0.02


def test_perturbed_window_starts_at_real_reset_and_stays_within_five_percent() -> None:
    builder = _load_builder_module()
    window = _fake_window(builder)
    theta = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    candidate = builder._perturbed_candidate_from_window(
        window,
        theta=theta,
        rng=np.random.default_rng(123),
        variant_index=0,
        max_fraction=0.05,
        knot_count=6,
        smooth_window=9,
        perturb_center=False,
    )

    assert candidate.ip_ref.shape == (101,)
    assert candidate.params_ref.shape == (101, 5)
    assert candidate.radii_ref.shape == (101, 32)
    assert float(candidate.ip_ref[0]) == pytest.approx(float(window.x[0, 0]))
    assert candidate.params_ref[0] == pytest.approx(window.params[0])
    assert candidate.params_ref[:, :2] == pytest.approx(window.params[:, :2])

    base = np.column_stack([window.x[:, 0], window.params[:, 2], window.params[:, 3], window.params[:, 4]])
    target = np.column_stack([candidate.ip_ref, candidate.params_ref[:, 2], candidate.params_ref[:, 3], candidate.params_ref[:, 4]])
    frac = np.abs((target - base) / np.maximum(np.abs(base), 1.0e-12))
    assert float(np.max(frac)) <= 0.050001
    assert candidate.max_fractional_perturbation <= 0.050001
    assert candidate.max_step_fractional_perturbation < 0.02


def test_written_reset_library_uses_training_preflight_schema(tmp_path: Path) -> None:
    builder = _load_builder_module()
    window = _fake_window(builder)
    theta = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    candidates = [
        builder._candidate_from_window(window, theta=theta),
        builder._perturbed_candidate_from_window(
            window,
            theta=theta,
            rng=np.random.default_rng(123),
            variant_index=0,
            max_fraction=0.05,
            knot_count=6,
            smooth_window=9,
            perturb_center=False,
        ),
    ]
    initial_states = tmp_path / "initial_states.npz"
    targets = tmp_path / "targets" / "t15_feasible_generated_trim50_idealized_0p1s_targets.npz"

    builder._write_libraries(
        candidates,
        initial_states,
        targets,
        train_shots=("3856",),
        holdout_shots=("3864",),
    )

    with np.load(initial_states, allow_pickle=False) as data:
        allowed = {
            "schema",
            "shot_id",
            "source_index",
            "time_s",
            "ip0",
            "pfc0",
            "sol0",
            "params0",
            "split",
            "difficulty_bin",
            "mode",
        }
        assert set(data.files) <= allowed
        assert "replay_source_index" not in data.files
        assert "replay_start_row" not in data.files
        assert "variant_index" not in data.files
        assert data["source_index"].tolist() == [0, 1]

    with np.load(targets, allow_pickle=False) as data:
        assert data["source_index"].tolist() == [0, 1]
        assert data["replay_source_index"].tolist() == [12, 12]
        assert data["replay_start_row"].tolist() == [12, 12]
        assert data["variant_index"].tolist() == [-1, 0]
