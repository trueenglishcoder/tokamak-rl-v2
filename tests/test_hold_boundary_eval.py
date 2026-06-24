from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "tokamak-sim"))

from tokamak_rl_v2.config.schema import BoundaryReferenceConfig, IpReferenceConfig, ReferenceConfig
from tokamak_rl_v2.env.references import generate_reference_batch, sample_hold_boundary_eval_cut_profile
from tokamak_rl_v2.env.t15_reference_limits import load_reference_limits

_SUMMARY_SPEC = importlib.util.spec_from_file_location("summarize_hold_boundary_eval", ROOT / "scripts/summarize_hold_boundary_eval.py")
assert _SUMMARY_SPEC is not None and _SUMMARY_SPEC.loader is not None
_SUMMARY_MODULE = importlib.util.module_from_spec(_SUMMARY_SPEC)
_SUMMARY_SPEC.loader.exec_module(_SUMMARY_MODULE)
summarize_hold_boundary_eval_main = _SUMMARY_MODULE.main


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
                "shot_ids": ["3856", "3864"],
            }
        ),
        encoding="utf-8",
    )
    load_reference_limits.cache_clear()
    return path


def _hold_eval_config(limits_path: Path) -> ReferenceConfig:
    return ReferenceConfig(
        duration_s=0.5,
        t_step=0.001,
        theta_count=4,
        seed=1,
        ip=IpReferenceConfig(
            kind="hold_boundary_eval_profile",
            limits_path=limits_path,
            start_mode="reset_ip",
            segment_min_steps=30,
            segment_max_steps=120,
            segment_count_min=1,
            segment_count_max=6,
            hold_probability=0.35,
            ramp_rate_reference="robust_mean",
            ramp_up_rate_min_fraction=0.05,
            ramp_up_rate_fraction=0.20,
            ramp_down_rate_min_fraction=0.05,
            ramp_down_rate_fraction=0.20,
            hold_min_steps=20,
            hold_max_steps=140,
            final_hold_min_steps=0,
            smooth_ramps=False,
            max_delta_fraction=0.35,
        ),
        boundary=BoundaryReferenceConfig(kind="hold_reset_boundary"),
    )


def _hold_eval_cut_config(limits_path: Path) -> ReferenceConfig:
    return ReferenceConfig(
        duration_s=0.5,
        t_step=0.001,
        theta_count=4,
        seed=1,
        ip=IpReferenceConfig(
            kind="hold_boundary_eval_cut_profile",
            limits_path=limits_path,
            start_mode="reset_ip",
            parent_steps=900,
            segment_min_steps=300,
            segment_max_steps=900,
            segment_count_min=1,
            segment_count_max=3,
            hold_probability=0.35,
            ramp_rate_reference="robust_mean",
            ramp_up_rate_min_fraction=0.05,
            ramp_up_rate_fraction=0.20,
            ramp_down_rate_min_fraction=0.05,
            ramp_down_rate_fraction=0.20,
            hold_min_steps=300,
            hold_max_steps=900,
            final_hold_min_steps=0,
            smooth_ramps=False,
            max_delta_fraction=0.35,
        ),
        boundary=BoundaryReferenceConfig(kind="hold_reset_boundary"),
    )


def test_hold_boundary_eval_profile_shapes_bounds_and_static_boundary(tmp_path: Path) -> None:
    cfg = _hold_eval_config(_limits(tmp_path / "limits.json"))
    batch_size = 64
    initial_ip = np.linspace(180000.0, 320000.0, batch_size, dtype=float)
    initial_radii = np.linspace(0.50, 0.65, batch_size * 4, dtype=float).reshape(batch_size, 4)
    ref = generate_reference_batch(
        config=cfg,
        initial_ip=initial_ip,
        initial_parameters=np.zeros((batch_size, 5), dtype=float),
        steps=500,
        device="cpu",
        seed=123,
        initial_boundary_points=np.zeros((batch_size, 4, 2), dtype=float),
        initial_boundary_radii=initial_radii,
    )

    ip = ref.ip.detach().cpu().numpy()
    radii = ref.radii.detach().cpu().numpy()
    assert ip.shape == (batch_size, 501)
    assert radii.shape == (batch_size, 501, 4)
    assert np.allclose(ip[:, 0], initial_ip)
    assert float(np.min(ip)) >= 100000.0
    assert float(np.max(ip)) <= 400000.0
    assert np.allclose(radii, initial_radii[:, None, :])

    rates = np.diff(ip, axis=1) / 0.001
    assert float(np.nanmax(rates)) <= 0.20e6 * (1.0 + 1.0e-6)
    assert float(np.nanmax(-rates)) <= 0.20e6 * (1.0 + 1.0e-6)
    assert np.any(np.isclose(ip[:, -1] - ip[:, 0], 0.0, atol=1.0e-6))
    assert np.any(ip[:, -1] > ip[:, 0])
    assert np.any(ip[:, -1] < ip[:, 0])


def test_hold_boundary_eval_profile_has_no_adjacent_ramps(tmp_path: Path) -> None:
    cfg = _hold_eval_config(_limits(tmp_path / "limits.json"))
    ref = generate_reference_batch(
        config=cfg,
        initial_ip=np.full((96,), 250000.0, dtype=float),
        initial_parameters=np.zeros((96, 5), dtype=float),
        steps=500,
        device="cpu",
        seed=456,
        initial_boundary_points=np.zeros((96, 4, 2), dtype=float),
        initial_boundary_radii=np.ones((96, 4), dtype=float),
    )

    for row in ref.ip.detach().cpu().numpy():
        diff = np.diff(row)
        signs = np.sign(diff).astype(int)
        signs[np.abs(diff) <= 1.0e-9] = 0
        runs: list[int] = []
        for sign in signs.tolist():
            if not runs or int(sign) != runs[-1]:
                runs.append(int(sign))
        for left, right in zip(runs, runs[1:], strict=False):
            assert not (left != 0 and right != 0), runs


def test_hold_boundary_eval_cut_profile_uses_long_parent_and_static_boundary(tmp_path: Path) -> None:
    cfg = _hold_eval_cut_config(_limits(tmp_path / "limits.json"))
    batch_size = 32
    initial_ip = np.linspace(180000.0, 320000.0, batch_size, dtype=float)
    initial_radii = np.linspace(0.50, 0.65, batch_size * 4, dtype=float).reshape(batch_size, 4)
    ref = generate_reference_batch(
        config=cfg,
        initial_ip=initial_ip,
        initial_parameters=np.zeros((batch_size, 5), dtype=float),
        steps=500,
        device="cpu",
        seed=789,
        initial_boundary_points=np.zeros((batch_size, 4, 2), dtype=float),
        initial_boundary_radii=initial_radii,
    )

    ip = ref.ip.detach().cpu().numpy()
    radii = ref.radii.detach().cpu().numpy()
    assert ip.shape == (batch_size, 501)
    assert radii.shape == (batch_size, 501, 4)
    assert np.allclose(ip[:, 0], initial_ip)
    assert np.allclose(radii, initial_radii[:, None, :])
    assert float(np.min(ip)) >= 100000.0
    assert float(np.max(ip)) <= 400000.0

    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)
    sample1 = sample_hold_boundary_eval_cut_profile(cfg.ip, 250000.0, 500, rng1, dt=0.001)
    sample2 = sample_hold_boundary_eval_cut_profile(cfg.ip, 250000.0, 500, rng2, dt=0.001)
    assert sample1.parent_ip.shape == (901,)
    assert sample1.ip.shape == (501,)
    assert 0 <= sample1.cut_start_step <= 400
    assert np.allclose(sample1.ip, sample2.ip)
    assert sample1.cut_start_step == sample2.cut_start_step
    assert sample1.ip[0] == pytest.approx(250000.0)
    assert np.allclose(sample1.ip, sample1.parent_ip[sample1.cut_start_step : sample1.cut_start_step + 501])
    signs = np.sign(np.diff(sample1.parent_ip)).astype(int)
    signs[np.abs(np.diff(sample1.parent_ip)) <= 1.0e-9] = 0
    run_lengths: list[int] = []
    current_sign: int | None = None
    current_len = 0
    for sign in signs.tolist():
        if current_sign is None or int(sign) == current_sign:
            current_sign = int(sign)
            current_len += 1
        else:
            run_lengths.append(current_len)
            current_sign = int(sign)
            current_len = 1
    if current_len:
        run_lengths.append(current_len)
    assert run_lengths
    assert min(run_lengths) >= 300


def test_summarize_hold_boundary_eval_aggregates_fake_shards(tmp_path: Path) -> None:
    root = tmp_path / "hold_boundary_eval_123"
    for shard_id in range(2):
        shard = root / f"shard_{shard_id}"
        shard.mkdir(parents=True)
        summary = {
            "episodes": 2,
            "steps": 500,
            "policies": {
                "policy": {"mean_episode_completion": 1.0},
                "no_control": {"mean_episode_completion": 0.5},
            },
        }
        (shard / "hold_boundary_eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        for policy in ("policy", "no_control"):
            rows = [
                {
                    "policy": policy,
                    "episode": i,
                    "episode_completion": 1.0 if policy == "policy" else 0.5,
                    "full_episode_success": 1.0 if policy == "policy" else 0.0,
                    "shape_error_mean_m_late": 0.01 + shard_id * 0.001,
                    "shape_error_max_m_late": 0.02 + shard_id * 0.001,
                    "ip_error_a_late": 1000.0 + shard_id,
                    "current_usage_fraction_late": 0.7,
                    "current_over_limit_a_late": 0.0,
                    "action_rms_late": 0.05,
                    "action_saturation_fraction_late": 0.0,
                    "terminated_boundary": 0.0,
                    "terminated_current": 0.0,
                    "boundary_found_late": 1.0,
                }
                for i in range(2)
            ]
            with (shard / f"hold_boundary_eval_{policy}_windows.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    out_dir = tmp_path / "summary"
    summarize_hold_boundary_eval_main([str(root), "--out-dir", str(out_dir)])

    assert (out_dir / "hold_boundary_eval_summary.json").exists()
    assert (out_dir / "hold_boundary_eval_summary.csv").exists()
    assert (out_dir / "hold_boundary_eval_report.md").exists()
    aggregate = json.loads((out_dir / "hold_boundary_eval_summary.json").read_text(encoding="utf-8"))
    assert aggregate["shard_count"] == 2
    assert aggregate["policies"]["policy"]["episodes"] == 4.0
    assert aggregate["policies"]["policy"]["mean_episode_completion"] == 1.0
