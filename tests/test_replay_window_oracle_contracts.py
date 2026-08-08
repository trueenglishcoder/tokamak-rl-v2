from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tokamak_rl_v2.env.references import T15ReplayBoundaryLibrary
from tokamak_rl_v2.training.replay import FIFOSequenceReplay


ROOT = Path(__file__).resolve().parents[1]


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
    replay_root.mkdir()
    radii = np.ones((3, 32), dtype=float)
    np.savez_compressed(
        replay_root / "lqr_boundary_reference_3856.npz",
        t=np.asarray([0.0, 0.001, 0.002], dtype=float),
        Ip=np.asarray([100000.0, 100100.0, 100200.0], dtype=float),
        radii_true=radii,
        boundary_found=np.asarray([True, False, True], dtype=bool),
    )
    candidate = SimpleNamespace(
        shot_id="3856",
        split="train",
        source_index=0,
        time_s=0.0,
        ip_target=np.asarray([100000.0, 100100.0, 100200.0], dtype=float),
        currents=np.zeros((3, 9), dtype=float),
        normalized_action=np.zeros((2, 9), dtype=float),
        difficulty_bin="flat",
    )

    with pytest.raises(RuntimeError, match=r"boundary extractor failed.*steps \[1\]"):
        load_oracle_from_replay(
            [candidate],
            replay_root=replay_root,
            angles=32,
            mean_ip_limit=10000.0,
            max_ip_limit=20000.0,
            n_pfc=6,
        )


def test_replay_oracle_invalid_fixed_angle_radius_is_fatal(tmp_path: Path) -> None:
    from scripts._oracle_from_replay import load_oracle_from_replay

    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    radii = np.ones((3, 32), dtype=float)
    radii[2, 7] = np.nan
    np.savez_compressed(
        replay_root / "lqr_boundary_reference_3856.npz",
        t=np.asarray([0.0, 0.001, 0.002], dtype=float),
        Ip=np.asarray([100000.0, 100100.0, 100200.0], dtype=float),
        radii_true=radii,
        boundary_found=np.asarray([True, True, True], dtype=bool),
    )
    candidate = SimpleNamespace(
        shot_id="3856",
        split="train",
        source_index=0,
        time_s=0.0,
        ip_target=np.asarray([100000.0, 100100.0, 100200.0], dtype=float),
        currents=np.zeros((3, 9), dtype=float),
        normalized_action=np.zeros((2, 9), dtype=float),
        difficulty_bin="flat",
    )

    with pytest.raises(RuntimeError, match=r"invalid fixed-angle boundary radii"):
        load_oracle_from_replay(
            [candidate],
            replay_root=replay_root,
            angles=32,
            mean_ip_limit=10000.0,
            max_ip_limit=20000.0,
            n_pfc=6,
        )


def test_equilibrium_oracle_job_uses_fresh_replay_as_single_boundary_source() -> None:
    job = (ROOT / "jobs" / "build_t15_ip15_equilibrium_lcfs_replay_window_oracle_targets_1gpu.sbatch").read_text(encoding="utf-8")
    assert "/workspace/tokamak-sim/runs/t15md_trim50_ip15_equilibrium_lcfs_top5_replay_gpu_full_" in job
    assert '"boundary_found"' in job
    assert 'np.flatnonzero(~ref_found)' in job
    assert '--replay-root "$REPLAY_ROOT"' in job
