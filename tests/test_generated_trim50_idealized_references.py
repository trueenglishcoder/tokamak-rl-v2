from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_t15_generated_trim50_idealized_references import main as build_generated_main
from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env.batch_env import TokamakMagneticControlEnv
from tokamak_rl_v2.env.references import (
    GENERATED_BOUNDARY_COMBOS,
    GENERATED_BOUNDARY_KEYS,
    GENERATED_RAMP_MODES,
    GENERATED_IP_MODES,
    boundary_points_from_parameters,
    generate_reference_batch,
    load_generated_envelope,
    sample_generated_boundary_parameters,
    sample_generated_segment_profile,
)
from tokamak_rl_v2.env.t15_csv_initial_states import CsvInitialStateLibrary


def test_generated_preprocessing_writes_exact_current_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    envelope_out = tmp_path / "envelope.json"
    states_out = tmp_path / "states.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            "--envelope-out",
            str(envelope_out),
            "--initial-states-out",
            str(states_out),
        ],
    )
    assert build_generated_main() == 0

    envelope = json.loads(envelope_out.read_text(encoding="utf-8"))
    assert envelope["holdout_shots"] == ["3863"]
    assert envelope["train_shots"] == ["3854", "3855", "3856", "3859", "3862"]
    assert envelope["ip"]["min_a"] == pytest.approx(112228.39132076583)
    assert envelope["ip"]["max_a"] == pytest.approx(511681.54135367385)
    assert envelope["ip"]["abs_rate_max_aps"] == pytest.approx(969677.3646181119)
    assert envelope["boundary"]["A0"]["min"] == pytest.approx(0.43414684038215856)
    assert envelope["boundary"]["A0"]["max"] == pytest.approx(0.6585471689773374)
    assert envelope["boundary"]["A0"]["abs_rate_max"] == pytest.approx(9.352908467497736)
    assert envelope["boundary"]["elongation_excess"]["min"] == pytest.approx(0.0)
    assert envelope["boundary"]["elongation_excess"]["max"] == pytest.approx(0.4732542848639277)
    assert envelope["boundary"]["elongation_excess"]["abs_rate_max"] == pytest.approx(8.682840529589246)
    assert envelope["boundary"]["delta"]["min"] == pytest.approx(0.0)
    assert envelope["boundary"]["delta"]["max"] == pytest.approx(0.28205171039598476)
    assert envelope["boundary"]["delta"]["abs_rate_max"] == pytest.approx(7.561204903826113)

    with np.load(states_out, allow_pickle=False) as data:
        assert {"ip0", "pfc0", "sol0", "params0", "shot_id", "source_index", "time_s", "split"} <= set(data.files)
        assert data["pfc0"].shape[1] == 6
        assert data["sol0"].shape[1] == 3
        assert data["params0"].shape == (data["ip0"].shape[0], 5)
        assert set(data["shot_id"][data["split"].astype(str) == "holdout"].astype(str).tolist()) == {"3863"}


def test_generated_ip_modes_are_bounded_and_rate_limited() -> None:
    cfg = load_experiment_config("configs/experiments/t15_generated_trim50_idealized_0p1s_tcvjdot_balanced_mpo.yaml")
    envelope = load_generated_envelope(cfg.reference.ip.limits_path)
    start = 250000.0
    seen = set()
    for i, mode in enumerate(GENERATED_IP_MODES):
        sample = sample_generated_segment_profile(
            cfg.reference.ip,
            start,
            int(cfg.sim.max_episode_steps),
            np.random.default_rng(i),
            dt=float(cfg.reference.t_step),
            forced_mode=mode,
        )
        seen.add(sample.mode)
        assert sample.ip.shape == (101,)
        assert sample.ip[0] == pytest.approx(start)
        assert np.min(sample.ip) >= envelope.ip_min_a - 1e-6
        assert np.max(sample.ip) <= envelope.ip_max_a + 1e-6
        assert np.max(np.abs(np.diff(sample.ip) / cfg.reference.t_step)) <= envelope.ip_abs_rate_max_aps * (1.0 + 1e-6)
        if "then" in mode:
            assert min(sample.segment_lengths) >= cfg.reference.ip.segment_min_steps
        last_delta = float(sample.ip[-1] - sample.ip[-2])
        if mode in {"ramp_up", "hold_then_up"}:
            assert last_delta > 0.0
        elif mode in {"ramp_down", "hold_then_down"}:
            assert last_delta < 0.0
        elif mode in {"up_then_hold", "down_then_hold", "hold"}:
            assert last_delta == pytest.approx(0.0)
    assert seen == set(GENERATED_IP_MODES)


def test_generated_boundary_modes_keep_center_and_obey_bounds() -> None:
    cfg = load_experiment_config("configs/experiments/t15_generated_trim50_idealized_0p1s_tcvjdot_balanced_mpo.yaml")
    envelope = load_generated_envelope(cfg.reference.boundary.envelope_path)
    start = np.asarray([1.45, -0.01, 0.55, 1.15, 0.10], dtype=float)
    seen = set()
    for i, combo in enumerate(GENERATED_BOUNDARY_COMBOS):
        sample = sample_generated_boundary_parameters(
            cfg.reference.boundary,
            start,
            int(cfg.sim.max_episode_steps),
            np.random.default_rng(100 + i),
            dt=float(cfg.reference.t_step),
            forced_combo=tuple(combo),
            forced_modes={key: "hold_then_up" for key in combo},
        )
        seen.add(tuple(combo))
        params = sample.parameters
        assert params.shape == (101, 5)
        assert np.allclose(params[:, 0], start[0])
        assert np.allclose(params[:, 1], start[1])
        assert np.min(params[:, 2]) >= envelope.A0_min_m - 1e-9
        assert np.max(params[:, 2]) <= envelope.A0_max_m + 1e-9
        assert np.min(params[:, 3] - 1.0) >= envelope.elongation_excess_min - 1e-9
        assert np.max(params[:, 3] - 1.0) <= envelope.elongation_excess_max + 1e-9
        assert np.min(params[:, 4]) >= envelope.delta_min - 1e-9
        assert np.max(params[:, 4]) <= envelope.delta_max + 1e-9
        for key in combo:
            assert sample.modes[key] == "hold_then_up"
            col = {"A0": 2, "elongation_excess": 3, "delta": 4}[key]
            assert float(params[-1, col] - params[-2, col]) > 0.0
    assert seen == set(GENERATED_BOUNDARY_COMBOS)


def test_generated_boundary_parameter_modes_match_ip_mode_shapes() -> None:
    cfg = load_experiment_config("configs/experiments/t15_generated_trim50_idealized_0p1s_tcvjdot_balanced_mpo.yaml")
    start = np.asarray([1.45, -0.01, 0.55, 1.15, 0.10], dtype=float)
    for i, key in enumerate(GENERATED_BOUNDARY_KEYS):
        col = {"A0": 2, "elongation_excess": 3, "delta": 4}[key]
        for j, mode in enumerate(GENERATED_RAMP_MODES):
            sample = sample_generated_boundary_parameters(
                cfg.reference.boundary,
                start,
                int(cfg.sim.max_episode_steps),
                np.random.default_rng(1000 + 10 * i + j),
                dt=float(cfg.reference.t_step),
                forced_combo=(key,),
                forced_modes={key: mode},
            )
            assert sample.modes[key] == mode
            last_delta = float(sample.parameters[-1, col] - sample.parameters[-2, col])
            if mode in {"ramp_up", "hold_then_up"}:
                assert last_delta > 0.0
            elif mode in {"ramp_down", "hold_then_down"}:
                assert last_delta < 0.0
            elif mode in {"up_then_hold", "down_then_hold"}:
                assert last_delta == pytest.approx(0.0)


def test_generated_reference_batch_has_expected_shapes() -> None:
    cfg = load_experiment_config("configs/experiments/t15_generated_trim50_idealized_0p1s_tcvjdot_balanced_mpo.yaml")
    ip0 = np.asarray([180000.0, 260000.0], dtype=float)
    params0 = np.asarray([[1.44, 0.0, 0.55, 1.12, 0.08], [1.46, -0.01, 0.60, 1.18, 0.12]], dtype=float)
    ref = generate_reference_batch(
        config=cfg.reference,
        initial_ip=ip0,
        initial_parameters=params0,
        steps=100,
        device="cpu",
        seed=7,
    )
    assert ref.ip.shape == (2, 101)
    assert ref.parameters.shape == (2, 101, 5)
    assert ref.radii.shape == (2, 101, 32)
    assert ref.points.shape == (2, 101, 32, 2)
    assert torch.allclose(ref.parameters[:, :, 0], ref.parameters[:, 0:1, 0])
    assert torch.allclose(ref.parameters[:, :, 1], ref.parameters[:, 0:1, 1])
    points = boundary_points_from_parameters(ref.parameters, ref.theta)
    assert torch.allclose(points, ref.points)


def test_generated_0p5s_config_loads_and_samples_500_step_references() -> None:
    cfg = load_experiment_config("configs/experiments/t15_generated_trim50_idealized_0p5s_tcvjdot_balanced_mpo.yaml")
    assert cfg.sim.max_episode_steps == 500
    assert cfg.reference.duration_s == pytest.approx(0.5)
    assert cfg.reference.t_step == pytest.approx(0.001)
    assert cfg.reference.ip.kind == "generated_segment_profile"
    assert cfg.reference.boundary.kind == "generated_parameter_profile"
    assert cfg.reference.ip.segment_min_steps == 150
    assert cfg.reference.boundary.segment_min_steps == 150
    assert cfg.observation.target_preview_steps == 10
    assert cfg.observation.target_preview_stride == 50
    assert cfg.learner.unroll_length == 100
    assert cfg.learner.rollout_chunk_length == 100

    ip0 = np.asarray([180000.0, 260000.0], dtype=float)
    params0 = np.asarray([[1.44, 0.0, 0.55, 1.12, 0.08], [1.46, -0.01, 0.60, 1.18, 0.12]], dtype=float)
    ref = generate_reference_batch(
        config=cfg.reference,
        initial_ip=ip0,
        initial_parameters=params0,
        steps=int(cfg.sim.max_episode_steps),
        device="cpu",
        seed=17,
    )
    assert ref.ip.shape == (2, 501)
    assert ref.parameters.shape == (2, 501, 5)
    assert ref.radii.shape == (2, 501, 32)
    assert ref.points.shape == (2, 501, 32, 2)
    assert torch.allclose(ref.parameters[:, :, 0], ref.parameters[:, 0:1, 0])
    assert torch.allclose(ref.parameters[:, :, 1], ref.parameters[:, 0:1, 1])


def test_generated_config_loads_and_env_uses_params0() -> None:
    cfg = load_experiment_config("configs/experiments/t15_generated_trim50_idealized_0p1s_tcvjdot_balanced_mpo.yaml")
    assert cfg.reference.ip.kind == "generated_segment_profile"
    assert cfg.reference.boundary.kind == "generated_parameter_profile"
    lib = CsvInitialStateLibrary(cfg.sim.csv_initial_state_library, n_pfc=6, n_sol=3, split="train")
    sample = lib.take([0])
    assert sample.params0 is not None
    env_cfg = replace(cfg, sim=replace(cfg.sim, compute_backend="cpu"))
    env = TokamakMagneticControlEnv(env_cfg, batch_size=1, device="cpu", seed=123)
    payload = env._reset_payload_from_csv_sample(sample)
    assert np.allclose(payload.params0, sample.params0)
