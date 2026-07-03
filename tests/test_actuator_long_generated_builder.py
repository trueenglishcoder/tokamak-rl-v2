from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_t15_actuator_long_generated_trim50_plain_gpu1e6_0p1s.py"
    spec = importlib.util.spec_from_file_location("build_t15_actuator_long_generated_trim50_plain_gpu1e6_0p1s", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Reset:
    shot_id = "3856"
    split = "train"
    start = 42
    ip = np.asarray([200000.0], dtype=float)
    currents = np.zeros((1, 9), dtype=float)


class _Envelope:
    ip_min = 0.0
    ip_max = 1.0e6
    radii_min = np.zeros((32,), dtype=float)
    radii_max = np.ones((32,), dtype=float)
    current_min = -np.ones((9,), dtype=float) * 1.0e9
    current_max = np.ones((9,), dtype=float) * 1.0e9
    feature_values = np.zeros((1, 4), dtype=float)


def _fake_parent(builder, steps: int = 105):
    candidate = builder.ParentCandidate(
        parent_id=3,
        split="train",
        reset=_Reset(),
        mode="synthetic_ladder_jdot",
        scale=1.0,
        style_source="train",
        currents=np.zeros((steps + 1, 9), dtype=float),
        action=np.zeros((steps, 9), dtype=float),
    )
    ip = np.linspace(200000.0, 230000.0, steps + 1, dtype=np.float32)
    radii = np.ones((steps + 1, 32), dtype=np.float32) * 0.55
    found = np.ones((steps + 1,), dtype=bool)
    return builder.ParentRollout(candidate=candidate, ip=ip, radii=radii, found=found, state_feature_distance=0.0)


def test_cut_parent_windows_uses_dense_overlapping_starts() -> None:
    builder = _load_builder_module()
    rows, rejected = builder._cut_parent_windows(
        [_fake_parent(builder, steps=105)],
        window_steps=100,
        stride=1,
        envelope=_Envelope(),
        state_feature_distance_limit=0.0,
    )

    assert not rejected
    assert len(rows) == 6
    assert [int(r["source_index"]) for r in rows] == [0, 1, 2, 3, 4, 5]
    assert [str(r["shot_id"]) for r in rows] == ["gen0003"] * 6
    assert all(np.asarray(r["ip_target"]).shape == (101,) for r in rows)
    assert all(np.asarray(r["boundary_radii"]).shape == (101, 32) for r in rows)
    assert all(np.asarray(r["real_jdot_action"]).shape == (100, 9) for r in rows)


def test_balanced_subsample_preserves_holdout_when_present() -> None:
    builder = _load_builder_module()
    rows = [{"split": "train", "shot_id": "gen0000", "source_index": i} for i in range(20)]
    rows.extend({"split": "holdout", "shot_id": "gen0001", "source_index": i} for i in range(10))
    selected = builder._balanced_subsample_rows(rows, max_windows=12, rng=np.random.default_rng(1))

    assert len(selected) == 12
    splits = [r["split"] for r in selected]
    assert "train" in splits
    assert "holdout" in splits
