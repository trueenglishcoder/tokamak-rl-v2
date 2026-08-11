from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tokamak_rl_v2.env.references import T15ReplayBoundaryLibrary
from tokamak_rl_v2.training.replay import FIFOSequenceReplay


ROOT = Path(__file__).resolve().parents[1]


def _candidate(*, source_index: int = 0, target_offset_a: float = 0.0) -> SimpleNamespace:
    """Создать минимального oracle candidate с 3 endpoint и 2 actions."""
    ip = np.asarray([100000.0, 100100.0, 100200.0], dtype=float) + float(target_offset_a)
    currents = np.zeros((3, 9), dtype=float)
    currents[:, 0] = np.asarray([10.0, 11.0, 12.0])
    return SimpleNamespace(
        shot_id="3856",
        split="train",
        source_index=int(source_index),
        time_s=float(source_index) * 0.001,
        ip_target=ip,
        currents=currents,
        normalized_action=np.zeros((2, 9), dtype=float),
        difficulty_bin="flat",
    )


def _write_reference_v2(
    root: Path,
    *,
    found: np.ndarray | None = None,
    projection_valid: np.ndarray | None = None,
    radii: np.ndarray | None = None,
) -> Path:
    """Записать endpoint-aligned replay reference v2 для contract tests."""
    root.mkdir(parents=True, exist_ok=True)
    endpoint_count = 3
    radii_arr = np.ones((endpoint_count, 32), dtype=float) if radii is None else np.asarray(radii, dtype=float)
    found_arr = np.ones((endpoint_count,), dtype=bool) if found is None else np.asarray(found, dtype=bool)
    projection_arr = (
        np.ones((endpoint_count,), dtype=bool)
        if projection_valid is None
        else np.asarray(projection_valid, dtype=bool)
    )
    pfc = np.zeros((endpoint_count, 6), dtype=float)
    pfc[:, 0] = np.asarray([10.0, 11.0, 12.0])
    sol = np.zeros((endpoint_count, 3), dtype=float)
    path = root / "lqr_boundary_reference_3856.npz"
    np.savez_compressed(
        path,
        schema=np.asarray(["t15md_replay_reference_v2"]),
        source_index=np.arange(endpoint_count, dtype=np.int64),
        t=np.arange(endpoint_count, dtype=float) * 0.001,
        Ip=np.asarray([99000.0, 99100.0, 99200.0], dtype=float),
        radii_true=radii_arr,
        boundary_found=found_arr,
        boundary_fixed_angle_valid=projection_arr,
        pfc_currents=pfc,
        sol_currents=sol,
        hidden_state_a=np.asarray([1.25, 1.5, 1.75], dtype=float),
        passive_currents_a=np.arange(endpoint_count * 303, dtype=float).reshape(endpoint_count, 303),
    )
    return path


def test_oracle_boundary_library_returns_exact_window_without_reanchoring(tmp_path: Path) -> None:
    root = tmp_path / "oracle"
    root.mkdir()
    radii = np.ones((1, 101, 32), dtype=np.float32)
    radii[0, :, :] += np.linspace(0.0, 0.1, 101, dtype=np.float32)[:, None]
    ip = np.linspace(120000.0, 140000.0, 101, dtype=np.float32).reshape(1, 101)
    real_action = np.zeros((1, 100, 9), dtype=np.float32)
    real_action[0, :, 4] = 0.25
    np.savez_compressed(
        root / "t15_replay_window_oracle_targets.npz",
        shot_id=np.asarray(["3864"]),
        source_index=np.asarray([123], dtype=np.int64),
        time_s=np.asarray([0.123], dtype=float),
        split=np.asarray(["holdout"]),
        difficulty_bin=np.asarray(["medium_up"]),
        ip_target=ip,
        boundary_radii=radii,
        real_jdot_action=real_action,
    )
    library = T15ReplayBoundaryLibrary(root, theta_count=32)
    reset_radii = np.full((32,), 99.0, dtype=float)

    got_radii = library.radii_for_segment("3864", steps=100, reset_radii=reset_radii, source_index=123)
    got_ip = library.ip_for_segment("3864", steps=100, source_index=123)
    got_action = library.real_action_for_segment("3864", steps=100, source_index=123)

    assert np.allclose(got_radii, radii[0])
    assert not np.allclose(got_radii[0], reset_radii)
    assert np.allclose(got_ip, ip[0])
    assert np.allclose(got_action, real_action[0])


def test_full_episode_replay_sampling_starts_at_zero() -> None:
    replay = FIFOSequenceReplay(
        capacity_episodes=2,
        max_episode_steps=5,
        active_envs=1,
        obs_dim=1,
        critic_obs_dim=1,
        action_dim=1,
        device="cpu",
    )
    for t in range(5):
        value = torch.tensor([[float(t)]])
        replay.add_batch(
            obs=value,
            critic_obs=value,
            action=value,
            reward=torch.tensor([float(t)]),
            discount=torch.tensor([0.99]),
            next_obs=value + 1.0,
            next_critic_obs=value + 1.0,
            done=torch.tensor([t == 4]),
        )
    batch = replay.sample(batch_size=4, sequence_length=5, min_sequence_length=5)
    assert torch.allclose(batch.obs[:, 0, 0], torch.zeros((4,)))
    assert torch.allclose(batch.obs[:, :, 0], torch.arange(5, dtype=torch.float32).repeat(4, 1))


def test_replay_oracle_boundary_failure_is_fatal(tmp_path: Path) -> None:
    from scripts._oracle_from_replay import load_oracle_from_replay

    replay_root = tmp_path / "replay"
    _write_reference_v2(replay_root, found=np.asarray([True, False, True], dtype=bool))

    with pytest.raises(RuntimeError, match=r"boundary extractor failed.*source endpoints \[1\]"):
        load_oracle_from_replay([_candidate()], replay_root=replay_root, angles=32, n_pfc=6)


def test_replay_oracle_projection_failure_is_fatal(tmp_path: Path) -> None:
    from scripts._oracle_from_replay import load_oracle_from_replay

    replay_root = tmp_path / "replay"
    _write_reference_v2(replay_root, projection_valid=np.asarray([True, True, False], dtype=bool))

    with pytest.raises(RuntimeError, match=r"fixed-angle projection invalid.*source endpoints \[2\]"):
        load_oracle_from_replay([_candidate()], replay_root=replay_root, angles=32, n_pfc=6)


def test_replay_oracle_invalid_fixed_angle_radius_is_fatal(tmp_path: Path) -> None:
    from scripts._oracle_from_replay import load_oracle_from_replay

    replay_root = tmp_path / "replay"
    radii = np.ones((3, 32), dtype=float)
    radii[2, 7] = np.nan
    _write_reference_v2(replay_root, radii=radii)

    with pytest.raises(RuntimeError, match=r"radii_true contains non-finite values|invalid fixed-angle boundary radii"):
        load_oracle_from_replay([_candidate()], replay_root=replay_root, angles=32, n_pfc=6)


def test_replay_oracle_uses_exact_replay_reset_state_and_does_not_filter_ip_error(tmp_path: Path) -> None:
    from scripts._oracle_from_replay import load_oracle_from_replay

    replay_root = tmp_path / "replay"
    _write_reference_v2(replay_root)
    candidate = _candidate(target_offset_a=50000.0)

    accepted, rejected = load_oracle_from_replay(
        [candidate], replay_root=replay_root, angles=32, n_pfc=6
    )

    assert rejected == []
    assert len(accepted) == 1
    row = accepted[0]
    assert row["ip0"] == pytest.approx(99000.0)
    assert row["hidden_state_a"] == pytest.approx(1.25)
    assert np.asarray(row["passive_currents_a"]).shape == (303,)
    assert float(row["oracle_ip_mean_error_a"]) > 50000.0
    assert np.allclose(row["pfc0"], np.asarray([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_initial_library_writes_compact_and_passive_reset_state(tmp_path: Path) -> None:
    from scripts.build_t15_new_replay_window_oracle_targets import _write_initial_library

    row = {
        "shot_id": "3856",
        "source_index": 10,
        "time_s": 0.01,
        "ip0": 100000.0,
        "pfc0": np.zeros((6,), dtype=float),
        "sol0": np.zeros((3,), dtype=float),
        "hidden_state_a": 123.0,
        "passive_currents_a": np.arange(303, dtype=float),
        "split": "train",
        "difficulty_bin": "flat",
    }
    path = tmp_path / "initial_states.npz"
    _write_initial_library(path, [row])

    with np.load(path, allow_pickle=False) as data:
        assert np.asarray(data["schema"]).astype(str).tolist() == [
            "t15_replay_window_oracle_initial_states_v2"
        ]
        assert np.asarray(data["hidden_state_a"]).shape == (1,)
        assert np.asarray(data["passive_currents_a"]).shape == (1, 303)


def test_equilibrium_oracle_job_requires_fresh_endpoint_aligned_compact_replay() -> None:
    job = (
        ROOT / "jobs" / "build_t15_ip15_equilibrium_lcfs_replay_window_oracle_targets_1gpu.sbatch"
    ).read_text(encoding="utf-8")
    assert "t15md_replay_reference_v2" in job
    assert '"hidden_state_a"' in job
    assert '"passive_currents_a"' in job
    assert '"boundary_fixed_angle_valid"' in job
    assert 'ref_source_index, np.arange(endpoints' in job
    assert '--replay-root "$REPLAY_ROOT"' in job
    assert '--raw-data-root data/t15_data_new' in job
    assert 'server stage 1/3: compact GPU batch=32 readiness' in job
    assert 'scripts/validate_t15_compact_gpu_batch32.py' in job
    assert '--batch-size 32' in job
    assert '--json-out "$READINESS_JSON"' in job
    assert 'reset_indices_unaffected_lanes_exact' in job
    assert 'identical_lane_state_parity' in job
    assert 'fresh canonical oracle rejected' in job
    assert 'accepted_by_shot mismatch' in job
    assert 'server_stage_acceptance.json' in job
    assert job.index('scripts/validate_t15_compact_gpu_batch32.py') < job.index('--shots 3854 3855 3856 3857 3858 3859 3862 3863')
    assert '--train-shots 3854 3855 3856 3857 3858 3859 3862 3863' in job
    assert '--holdout-shots 3864' not in job
    assert 'expected_shots = {"3854", "3855", "3856", "3857", "3858", "3859", "3862", "3863"}' in job
