from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SIM_ROOT = ROOT.parent / "tokamak-sim"
if SIM_ROOT.exists() and str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from tokamak_rl_v2.env.references import T15ReplayBoundaryLibrary


def _load_builder_module():
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / "build_t15_long_target_generated_trim50_idealized_matched_0p1s.py"
    spec = importlib.util.spec_from_file_location("build_t15_long_target_generated_trim50_idealized_matched_0p1s", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source(builder, shot: str, split: str, offset: float) -> object:
    n = 201
    step = np.arange(n, dtype=float)
    frac = step / float(n - 1)
    time_s = 0.05 + 0.001 * step
    ip = 145000.0 + offset + 120000.0 * frac
    a0 = 0.46 + 0.05 * frac
    e = 0.14 + 0.04 * frac
    delta = 0.08 + 0.03 * frac
    x = np.column_stack([ip, a0, e, delta])
    params = np.column_stack([np.full(n, 1.5), np.zeros(n), a0, 1.0 + e, delta])
    coils = np.column_stack(
        [
            1.0e5 + 2.0e4 * frac,
            2.0e5 + 2.5e4 * frac,
            1.5e5 + 1.0e4 * frac,
            -5.0e4 + 2.0e4 * frac,
            4.0e4 + 1.0e4 * frac,
            3.0e4 + 1.5e4 * frac,
            2.0e4 + 2.0e4 * frac,
            1.0e5 + 3.0e4 * frac,
            -8.0e4 + 1.0e4 * frac,
        ]
    )
    return builder.SourceShot(
        shot=shot,
        split=split,
        time_s=time_s,
        source_index=np.arange(n, dtype=np.int64),
        x=x,
        params=params,
        coils=coils,
    )


def test_long_target_builder_writes_dense_oracle_windows_without_duplicate_keys(tmp_path: Path) -> None:
    builder = _load_builder_module()
    sources = [_source(builder, "3856", "train", 0.0), _source(builder, "3864", "holdout", 5000.0)]
    limits = builder.simple.Limits(pfc_current=1.0e7, sol_current=1.0e7, pfc_deriv=1.0e8, sol_deriv=1.0e8)
    theta = np.linspace(-np.pi, np.pi, 32, endpoint=False, dtype=float)
    envelope = builder._build_envelope(sources=sources, theta=theta, limits=limits)
    points = builder._safe_reset_points_from_sources(sources, limits=limits, current_usage_cap=1.0)
    state_space = builder._state_space_from_points(points, limits=limits)
    move_samples = builder._move_samples_from_sources(sources, steps=100, limits=limits)
    move_space = builder._move_space_from_samples(move_samples, limits=limits)
    rejections = builder.Counter()
    parents = []
    rng = np.random.default_rng(7)
    for split, first_id in (("train", 0), ("holdout", 1)):
        parents.extend(
            builder._generate_parents(
                [p for p in points if p.split == split],
                count=1,
                first_parent_id=first_id,
                rng=rng,
                parent_min_steps=120,
                parent_max_steps=120,
                segment_min_steps=50,
                segment_max_steps=70,
                join_blend_steps=4,
                endpoint_distance_min=0.001,
                endpoint_distance_max=10.0,
                endpoint_mix_min=0.2,
                endpoint_mix_max=0.4,
                hold_probability=0.0,
                theta=theta,
                limits=limits,
                envelope=envelope,
                state_space=state_space,
                move_samples=move_samples,
                state_distance_limit=10.0,
                current_usage_cap=1.0,
                max_attempts_per_parent=10,
                rejections=rejections,
            )
        )

    windows = builder._cut_parents(
        parents,
        window_steps=100,
        stride=1,
        move_space=move_space,
        limits=limits,
        move_distance_limit=10.0,
        rejections=builder.Counter(),
    )
    assert len(windows) == 42
    assert {w.parent.source_shot for w in windows} == {"3856", "3864"}
    assert any(w.difficulty_bin != "flat" for w in windows)
    assert any(w.move_distance > 0.0 for w in windows)

    initial = tmp_path / "initial.npz"
    targets = tmp_path / "targets.npz"
    oracle = tmp_path / "oracle" / "t15_replay_window_oracle_targets.npz"
    parents_path = tmp_path / "parents.npz"
    builder._write_libraries(
        windows,
        parents=parents,
        initial_states_out=initial,
        targets_out=targets,
        oracle_targets_out=oracle,
        parents_out=parents_path,
        limits=limits,
        train_shots=("3856",),
        holdout_shots=("3864",),
    )

    with np.load(initial, allow_pickle=False) as data:
        assert data["ip0"].shape == (42,)
        assert data["pfc0"].shape == (42, 6)
        assert data["sol0"].shape == (42, 3)
        assert data["params0"].shape == (42, 5)
        assert "source_shot" in data.files
        assert set(data["shot_id"].astype(str).tolist()) == {"900000", "900001"}

    with np.load(oracle, allow_pickle=False) as data:
        assert data["ip_target"].shape == (42, 101)
        assert data["boundary_radii"].shape == (42, 101, 32)
        assert data["real_jdot_action"].shape == (42, 100, 9)
        keys = list(zip(data["shot_id"].astype(str).tolist(), data["source_index"].astype(int).tolist(), strict=True))
        assert len(keys) == len(set(keys))
        assert "source_shot" in data.files

    library = T15ReplayBoundaryLibrary(oracle.parent, theta_count=32)
    with np.load(initial, allow_pickle=False) as data:
        row = 0
        radii = library.radii_for_segment(
            data["shot_id"][row],
            steps=100,
            reset_radii=np.ones((32,), dtype=float),
            source_index=int(data["source_index"][row]),
            source_time_s=float(data["time_s"][row]),
        )
        ip = library.ip_for_segment(
            data["shot_id"][row],
            steps=100,
            source_index=int(data["source_index"][row]),
            source_time_s=float(data["time_s"][row]),
        )
    assert radii.shape == (101, 32)
    assert ip.shape == (101,)
