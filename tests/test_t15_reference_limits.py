from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from tokamak_rl_v2.config.schema import BoundaryReferenceConfig, IpReferenceConfig, ReferenceConfig
from tokamak_rl_v2.env.references import generate_reference_batch
from tokamak_rl_v2.env.t15_reference_limits import load_reference_limits


def _limits(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "ip_min_a": 80000.0,
                "ip_max_a": 420000.0,
                "ip_p01_a": 100000.0,
                "ip_p99_a": 400000.0,
                "positive_dipdt_p95_a_per_s": 2.0e6,
                "positive_dipdt_p99_a_per_s": 2.5e6,
                "negative_dipdt_abs_p95_a_per_s": 2.0e6,
                "negative_dipdt_abs_p99_a_per_s": 2.5e6,
                "positive_ramp_mean_a_per_s": 1.0e6,
                "negative_ramp_abs_mean_a_per_s": 1.0e6,
                "sample_count": 1000,
                "shot_count": 2,
                "shot_ids": ["3856", "3857"],
            }
        ),
        encoding="utf-8",
    )
    load_reference_limits.cache_clear()
    return path


def test_reference_limits_loader_validates_bounds(tmp_path: Path) -> None:
    limits = load_reference_limits(_limits(tmp_path / "limits.json"))
    assert limits.ip_p01_a == 100000.0
    assert limits.ip_p99_a == 400000.0
    assert limits.ip_min_a == 80000.0
    assert limits.ip_max_a == 420000.0
    assert limits.ip_width_a == 300000.0
    assert limits.positive_ramp_mean_a_per_s == 1000000.0
    assert limits.negative_ramp_abs_mean_a_per_s == 1000000.0


def test_segmented_profile_starts_at_reset_ip_and_stays_positive(tmp_path: Path) -> None:
    limits_path = _limits(tmp_path / "limits.json")
    cfg = ReferenceConfig(
        duration_s=0.5,
        t_step=0.001,
        theta_count=4,
        seed=1,
        ip=IpReferenceConfig(
            kind="segmented_profile",
            limits_path=limits_path,
            ramp_up_rate_fraction=0.25,
            ramp_down_rate_fraction=0.25,
            hold_min_steps=20,
            hold_max_steps=120,
            final_hold_min_steps=20,
            plateau_min_fraction=0.10,
            plateau_max_fraction=0.20,
            end_min_fraction=0.10,
            end_max_fraction=0.20,
            max_delta_fraction=0.2,
            smooth_ramps=False,
        ),
        boundary=BoundaryReferenceConfig(kind="hold_reset_boundary"),
    )
    points0 = np.zeros((2, 4, 2), dtype=float)
    radii0 = np.ones((2, 4), dtype=float)
    initial_ip = np.asarray([120000.0, 180000.0], dtype=float)
    ref = generate_reference_batch(
        config=cfg,
        initial_ip=initial_ip,
        initial_parameters=np.zeros((2, 5), dtype=float),
        steps=500,
        device="cpu",
        seed=123,
        initial_boundary_points=points0,
        initial_boundary_radii=radii0,
    )
    arr = ref.ip.detach().cpu().numpy()
    assert arr.shape == (2, 501)
    assert np.allclose(arr[:, 0], initial_ip)
    assert np.all(arr > 0.0)
    assert np.all(np.isfinite(arr))
    assert float(np.min(arr)) >= 100000.0
    assert float(np.max(arr)) <= 400000.0
    rates = np.diff(arr, axis=1) / 0.001
    assert float(np.nanmax(rates)) <= 0.25 * 2.0e6 * (1.0 + 1.0e-6)
    assert float(np.nanmax(-rates)) <= 0.25 * 2.0e6 * (1.0 + 1.0e-6)
    diff = np.diff(arr, axis=1)
    assert np.any(np.abs(diff) > 0.0)
    assert np.any(np.abs(diff) <= 1.0e-9)


def test_single_segment_profile_samples_hold_up_and_down_with_static_boundary(tmp_path: Path) -> None:
    limits_path = _limits(tmp_path / "limits.json")
    cfg = ReferenceConfig(
        duration_s=0.1,
        t_step=0.001,
        theta_count=4,
        seed=1,
        ip=IpReferenceConfig(
            kind="single_segment_profile",
            limits_path=limits_path,
            ramp_rate_reference="robust_mean",
            ramp_up_rate_min_fraction=0.3,
            ramp_up_rate_fraction=0.55,
            ramp_down_rate_min_fraction=0.3,
            ramp_down_rate_fraction=0.55,
            max_delta_fraction=0.6,
            smooth_ramps=False,
        ),
        boundary=BoundaryReferenceConfig(kind="hold_reset_boundary"),
    )
    sample_count = 96
    points0 = np.zeros((sample_count, 4, 2), dtype=float)
    radii0 = np.ones((sample_count, 4), dtype=float)
    initial_ip = np.linspace(180000.0, 320000.0, sample_count, dtype=float)
    ref = generate_reference_batch(
        config=cfg,
        initial_ip=initial_ip,
        initial_parameters=np.zeros((sample_count, 5), dtype=float),
        steps=100,
        device="cpu",
        seed=123,
        initial_boundary_points=points0,
        initial_boundary_radii=radii0,
    )
    arr = ref.ip.detach().cpu().numpy()
    assert arr.shape == (sample_count, 101)
    assert np.allclose(arr[:, 0], initial_ip)
    assert float(np.min(arr)) >= 100000.0
    assert float(np.max(arr)) <= 400000.0
    delta = arr[:, -1] - arr[:, 0]
    assert np.any(np.isclose(delta, 0.0, atol=1.0e-6))
    assert np.any(delta > 1.0e-6)
    assert np.any(delta < -1.0e-6)
    for row, total_delta in zip(arr, delta, strict=True):
        diff = np.diff(row)
        if abs(float(total_delta)) <= 1.0e-6:
            assert np.allclose(row, row[0])
        elif total_delta > 0.0:
            assert np.all(diff >= -1.0e-7)
            assert diff[0] > 0.0
            assert np.allclose(diff, diff[0], rtol=1.0e-6, atol=1.0e-6)
        else:
            assert np.all(diff <= 1.0e-7)
            assert diff[0] < 0.0
            assert np.allclose(diff, diff[0], rtol=1.0e-6, atol=1.0e-6)
    rates = np.diff(arr, axis=1) / 0.001
    assert float(np.nanmax(rates)) <= 0.55 * 1.0e6 * (1.0 + 1.0e-6)
    assert float(np.nanmax(-rates)) <= 0.55 * 1.0e6 * (1.0 + 1.0e-6)
    radii = ref.radii.detach().cpu().numpy()
    assert radii.shape == (sample_count, 101, 4)
    assert np.allclose(radii, radii[:, :1, :])


def test_smoothed_segmented_profile_obeys_signed_rate_limits(tmp_path: Path) -> None:
    limits_path = _limits(tmp_path / "limits.json")
    cfg = ReferenceConfig(
        duration_s=0.5,
        t_step=0.001,
        theta_count=4,
        seed=1,
        ip=IpReferenceConfig(
            kind="segmented_profile",
            limits_path=limits_path,
            segment_min_steps=40,
            segment_max_steps=100,
            segment_count_min=3,
            segment_count_max=6,
            hold_min_steps=20,
            hold_max_steps=100,
            ramp_up_rate_fraction=0.25,
            ramp_down_rate_fraction=0.25,
            max_delta_fraction=0.25,
            smooth_ramps=True,
        ),
        boundary=BoundaryReferenceConfig(kind="hold_reset_boundary"),
    )
    points0 = np.zeros((1, 4, 2), dtype=float)
    radii0 = np.ones((1, 4), dtype=float)
    for seed in range(1, 24):
        ref = generate_reference_batch(
            config=cfg,
            initial_ip=np.asarray([250000.0], dtype=float),
            initial_parameters=np.zeros((1, 5), dtype=float),
            steps=500,
            device="cpu",
            seed=seed,
            initial_boundary_points=points0,
            initial_boundary_radii=radii0,
        )
        rates = np.diff(ref.ip.detach().cpu().numpy()[0]) / 0.001
        positive = rates[rates > 0.0]
        negative = -rates[rates < 0.0]
        if positive.size:
            assert float(np.nanmax(positive)) <= 0.25 * 2.0e6 * (1.0 + 1.0e-6)
        if negative.size:
            assert float(np.nanmax(negative)) <= 0.25 * 2.0e6 * (1.0 + 1.0e-6)


def test_segmented_profile_can_start_with_either_ramp_direction_under_fixed_seeds(tmp_path: Path) -> None:
    limits_path = _limits(tmp_path / "limits.json")
    cfg = ReferenceConfig(
        duration_s=0.5,
        t_step=0.001,
        theta_count=4,
        seed=1,
        ip=IpReferenceConfig(
            kind="segmented_profile",
            limits_path=limits_path,
            segment_min_steps=30,
            segment_max_steps=80,
            segment_count_min=3,
            segment_count_max=8,
            hold_min_steps=20,
            hold_max_steps=80,
            final_hold_min_steps=20,
            plateau_min_fraction=0.10,
            plateau_max_fraction=0.25,
            end_min_fraction=0.10,
            end_max_fraction=0.25,
            max_delta_fraction=0.25,
            smooth_ramps=False,
        ),
        boundary=BoundaryReferenceConfig(kind="hold_reset_boundary"),
    )
    points0 = np.zeros((1, 4, 2), dtype=float)
    radii0 = np.ones((1, 4), dtype=float)
    first_ramp_directions: set[int] = set()
    for seed in range(1, 33):
        ref = generate_reference_batch(
            config=cfg,
            initial_ip=np.asarray([250000.0], dtype=float),
            initial_parameters=np.zeros((1, 5), dtype=float),
            steps=500,
            device="cpu",
            seed=seed,
            initial_boundary_points=points0,
            initial_boundary_radii=radii0,
        )
        diff = np.diff(ref.ip.detach().cpu().numpy()[0])
        nz = diff[np.abs(diff) > 1.0e-9]
        assert nz.size > 0
        first_ramp_directions.add(1 if float(nz[0]) > 0.0 else -1)
    assert first_ramp_directions == {-1, 1}


def test_segmented_profile_never_places_ramps_back_to_back(tmp_path: Path) -> None:
    limits_path = _limits(tmp_path / "limits.json")
    cfg = ReferenceConfig(
        duration_s=0.5,
        t_step=0.001,
        theta_count=4,
        seed=1,
        ip=IpReferenceConfig(
            kind="segmented_profile",
            limits_path=limits_path,
            segment_min_steps=30,
            segment_max_steps=80,
            segment_count_min=3,
            segment_count_max=8,
            hold_probability=0.0,
            hold_min_steps=20,
            hold_max_steps=80,
            final_hold_min_steps=20,
            max_delta_fraction=0.25,
            smooth_ramps=False,
        ),
        boundary=BoundaryReferenceConfig(kind="hold_reset_boundary"),
    )
    points0 = np.zeros((1, 4, 2), dtype=float)
    radii0 = np.ones((1, 4), dtype=float)
    for seed in range(1, 65):
        ref = generate_reference_batch(
            config=cfg,
            initial_ip=np.asarray([250000.0], dtype=float),
            initial_parameters=np.zeros((1, 5), dtype=float),
            steps=500,
            device="cpu",
            seed=seed,
            initial_boundary_points=points0,
            initial_boundary_radii=radii0,
        )
        diff = np.diff(ref.ip.detach().cpu().numpy()[0])
        signs = np.sign(diff).astype(int)
        signs[np.abs(diff) <= 1.0e-9] = 0
        runs: list[int] = []
        for sign in signs.tolist():
            if not runs or int(sign) != runs[-1]:
                runs.append(int(sign))
        assert 0 in runs
        assert any(sign != 0 for sign in runs)
        for left, right in zip(runs, runs[1:], strict=False):
            assert not (left != 0 and right != 0)


def test_segmented_profile_rejects_impossible_nonzero_ramp_requests(tmp_path: Path) -> None:
    limits_path = _limits(tmp_path / "limits.json")
    cfg = ReferenceConfig(
        duration_s=0.5,
        t_step=0.001,
        theta_count=4,
        seed=1,
        ip=IpReferenceConfig(
            kind="segmented_profile",
            limits_path=limits_path,
            segment_min_steps=30,
            segment_max_steps=60,
            segment_count_min=3,
            segment_count_max=16,
            hold_min_steps=20,
            hold_max_steps=60,
            final_hold_min_steps=20,
            plateau_min_fraction=0.10,
            plateau_max_fraction=0.20,
            end_min_fraction=0.10,
            end_max_fraction=0.20,
            ramp_up_rate_fraction=1.0e-12,
            ramp_down_rate_fraction=1.0e-12,
            max_delta_fraction=0.20,
            smooth_ramps=False,
        ),
        boundary=BoundaryReferenceConfig(kind="hold_reset_boundary"),
    )
    points0 = np.zeros((1, 4, 2), dtype=float)
    radii0 = np.ones((1, 4), dtype=float)
    import pytest

    with pytest.raises(ValueError, match="failed to sample a segmented_profile"):
        generate_reference_batch(
            config=cfg,
            initial_ip=np.asarray([250000.0], dtype=float),
            initial_parameters=np.zeros((1, 5), dtype=float),
            steps=500,
            device="cpu",
            seed=7,
            initial_boundary_points=points0,
            initial_boundary_radii=radii0,
        )


def test_segmented_profile_rejects_reset_ip_outside_limits(tmp_path: Path) -> None:
    limits_path = _limits(tmp_path / "limits.json")
    cfg = ReferenceConfig(
        duration_s=0.5,
        t_step=0.001,
        theta_count=4,
        seed=1,
        ip=IpReferenceConfig(kind="segmented_profile", limits_path=limits_path),
        boundary=BoundaryReferenceConfig(kind="hold_reset_boundary"),
    )
    points0 = np.zeros((1, 4, 2), dtype=float)
    radii0 = np.ones((1, 4), dtype=float)
    import pytest

    with pytest.raises(ValueError, match="outside production bounds"):
        generate_reference_batch(
            config=cfg,
            initial_ip=np.asarray([50000.0], dtype=float),
            initial_parameters=np.zeros((1, 5), dtype=float),
            steps=500,
            device="cpu",
            seed=123,
            initial_boundary_points=points0,
            initial_boundary_radii=radii0,
        )
