from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokamak_rl_v2.env.t15_csv_initial_states import CsvInitialStateLibrary, validate_split_nonoverlap
from scripts.build_t15_csv_initial_state_library import _assign_splits


def test_csv_initial_state_library_samples_coherent_rows(tmp_path: Path) -> None:
    path = tmp_path / "states.npz"
    np.savez(
        path,
        shot_id=np.asarray(["3856", "3857"]),
        source_index=np.asarray([10, 20], dtype=np.int64),
        time_s=np.asarray([0.1, 0.2], dtype=float),
        ip0=np.asarray([120000.0, 130000.0], dtype=float),
        pfc0=np.asarray([[1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16]], dtype=float),
        sol0=np.asarray([[7, 8, 9], [17, 18, 19]], dtype=float),
        split=np.asarray(["train", "holdout"]),
    )
    lib = CsvInitialStateLibrary(path, n_pfc=6, n_sol=3)
    rng = np.random.default_rng(1)
    sample = lib.sample(rng, 8)
    assert sample.ip0.shape == (8,)
    assert sample.pfc0.shape == (8, 6)
    assert sample.sol0.shape == (8, 3)
    for row in range(8):
        assert sample.shot_ids[row] == "3856"
        assert sample.source_indices[row] == 10
        assert np.allclose(sample.pfc0[row], [1, 2, 3, 4, 5, 6])
        assert np.allclose(sample.sol0[row], [7, 8, 9])

    holdout = CsvInitialStateLibrary(path, n_pfc=6, n_sol=3, split="holdout")
    assert len(holdout) == 1
    assert holdout.take([0]).shot_ids == ("3857",)

    all_rows = CsvInitialStateLibrary(path, n_pfc=6, n_sol=3, split="all")
    assert len(all_rows) == 2


def test_csv_initial_state_library_rejects_wrong_shapes(tmp_path: Path) -> None:
    path = tmp_path / "states.npz"
    np.savez(
        path,
        shot_id=np.asarray(["3856"]),
        source_index=np.asarray([10], dtype=np.int64),
        time_s=np.asarray([0.1], dtype=float),
        ip0=np.asarray([120000.0], dtype=float),
        pfc0=np.asarray([[1, 2]], dtype=float),
        sol0=np.asarray([[3]], dtype=float),
        split=np.asarray(["train"]),
    )
    with pytest.raises(ValueError, match="PFC shape"):
        CsvInitialStateLibrary(path, n_pfc=6, n_sol=3)


def test_csv_initial_state_library_requires_split_for_production(tmp_path: Path) -> None:
    path = tmp_path / "states.npz"
    np.savez(
        path,
        shot_id=np.asarray(["3856"]),
        source_index=np.asarray([10], dtype=np.int64),
        time_s=np.asarray([0.1], dtype=float),
        ip0=np.asarray([120000.0], dtype=float),
        pfc0=np.zeros((1, 6), dtype=float),
        sol0=np.zeros((1, 3), dtype=float),
    )
    with pytest.raises(ValueError, match="missing split"):
        CsvInitialStateLibrary(path, n_pfc=6, n_sol=3)
    assert len(CsvInitialStateLibrary(path, n_pfc=6, n_sol=3, split="all")) == 1


def test_validate_split_nonoverlap_rejects_same_shot_rows_within_episode_gap() -> None:
    with pytest.raises(ValueError, match="overlap within one episode"):
        validate_split_nonoverlap(
            np.asarray(["3856", "3856"], dtype=str),
            np.asarray([0.00, 0.50], dtype=float),
            np.asarray(["train", "holdout"], dtype=str),
            min_gap_s=0.5,
        )


def test_builder_assign_splits_uses_strict_episode_gap() -> None:
    accepted = [
        {
            "shot_id": "3856",
            "time_s": float(idx) * 0.01,
            "source_index": idx,
            "ip0": 120000.0,
            "pfc0": np.zeros((6,), dtype=float),
            "sol0": np.zeros((3,), dtype=float),
        }
        for idx in range(1000)
    ]
    splits = _assign_splits(accepted, gap_s=0.5)
    shot_id = np.asarray([str(row["shot_id"]) for row in accepted], dtype=str)
    time_s = np.asarray([float(row["time_s"]) for row in accepted], dtype=float)
    validate_split_nonoverlap(shot_id, time_s, splits, min_gap_s=0.5)
