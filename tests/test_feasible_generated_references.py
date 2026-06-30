from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_t15_feasible_generated_trim50_idealized_0p1s import main as build_feasible_main
from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env.references import FeasibleGeneratedTargetLibrary, generate_reference_batch
from tokamak_rl_v2.env.t15_csv_initial_states import CsvInitialStateLibrary


def test_feasible_builder_writes_row_aligned_reset_and_target_libraries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "targets"
    initial_states = tmp_path / "initial_states.npz"
    targets = out_dir / "t15_feasible_generated_trim50_idealized_0p1s_targets.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            "--target-count",
            "120",
            "--out-dir",
            str(out_dir),
            "--initial-states-out",
            str(initial_states),
            "--targets-out",
            str(targets),
            "--no-plots",
        ],
    )
    assert build_feasible_main() == 0

    with np.load(initial_states, allow_pickle=False) as reset, np.load(targets, allow_pickle=False) as target:
        assert reset["ip0"].shape[0] == target["ip_ref"].shape[0]
        assert target["ip_ref"].shape[1:] == (101,)
        assert target["params_ref"].shape[1:] == (101, 5)
        assert target["radii_ref"].shape[1:] == (101, 32)
        assert set(reset["difficulty_bin"].astype(str).tolist()) == {"core", "moderate", "ambitious"}
        assert set(reset["shot_id"][reset["split"].astype(str) == "holdout"].astype(str).tolist()) == {"3863"}
        assert np.max(np.abs(reset["ip0"] - target["ip_ref"][:, 0])) == pytest.approx(0.0)
        assert np.max(np.abs(reset["params0"] - target["params_ref"][:, 0, :])) == pytest.approx(0.0)
        assert np.array_equal(reset["source_index"], np.arange(reset["ip0"].shape[0]))


def test_feasible_reference_batch_uses_matching_target_rows() -> None:
    cfg = load_experiment_config("configs/experiments/t15_feasible_generated_trim50_idealized_0p1s_tcvjdot_balanced_mpo.yaml")
    reset_lib = CsvInitialStateLibrary(cfg.sim.csv_initial_state_library, n_pfc=6, n_sol=3, split="train")
    sample = reset_lib.take([0, 1, 2])
    target_lib = FeasibleGeneratedTargetLibrary(cfg.reference.ip.feasible_reference_dir, theta_count=32)

    ref = generate_reference_batch(
        config=cfg.reference,
        initial_ip=sample.ip0,
        initial_parameters=sample.params0,
        steps=100,
        device="cpu",
        seed=123,
        target_indices=np.asarray(sample.source_indices, dtype=np.int64),
        feasible_target_library=target_lib,
    )
    ip_np, params_np, radii_np = target_lib.rows(sample.source_indices, steps=100)
    assert ref.ip.shape == (3, 101)
    assert ref.parameters.shape == (3, 101, 5)
    assert ref.radii.shape == (3, 101, 32)
    assert torch.allclose(ref.ip, torch.as_tensor(ip_np, dtype=torch.float64))
    assert torch.allclose(ref.parameters, torch.as_tensor(params_np, dtype=torch.float64))
    assert torch.allclose(ref.radii, torch.as_tensor(radii_np, dtype=torch.float64), rtol=1.0e-4, atol=1.0e-5)


def test_feasible_curriculum_sampling_can_select_only_core_rows() -> None:
    cfg = load_experiment_config("configs/experiments/t15_feasible_generated_trim50_idealized_0p1s_tcvjdot_balanced_mpo.yaml")
    reset_lib = CsvInitialStateLibrary(cfg.sim.csv_initial_state_library, n_pfc=6, n_sol=3, split="train")
    sample = reset_lib.sample(np.random.default_rng(7), 32, difficulty_weights={"core": 1.0, "moderate": 0.0, "ambitious": 0.0})
    assert set(sample.difficulty_bins) == {"core"}
