from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tests/fixtures/t15_static_boundary_test.json"
NO_STEP_KIND = "controller_state_v7_no_step_norm"


def _env_pair() -> tuple[TokamakMagneticControlEnv, TokamakMagneticControlEnv]:
    base = load_experiment_config(CONFIG)
    base = replace(base, sim=replace(base.sim, compute_backend="cpu", max_episode_steps=4))
    v6 = replace(base, observation=replace(base.observation, actor_kind="controller_state_v6"))
    v7 = replace(base, observation=replace(base.observation, actor_kind=NO_STEP_KIND))
    return (
        TokamakMagneticControlEnv(v6, batch_size=2, device="cpu", seed=11),
        TokamakMagneticControlEnv(v7, batch_size=2, device="cpu", seed=11),
    )


def test_controller_state_v7_no_step_feature_order_matches_v6_minus_step_norm() -> None:
    env_v6, env_v7 = _env_pair()

    schema_v6 = env_v6.export_schema()
    schema_v7 = env_v7.export_schema()

    assert schema_v7["observation_kind"] == NO_STEP_KIND
    assert "step_norm" in schema_v6["feature_order"]
    assert "step_norm" not in schema_v7["feature_order"]
    assert schema_v7["feature_order"] == [name for name in schema_v6["feature_order"] if name != "step_norm"]
    assert schema_v7["obs_dim"] == int(schema_v6["obs_dim"]) - 1
    assert schema_v7["critic_obs_dim"] == int(schema_v6["critic_obs_dim"]) - 1


@pytest.mark.parametrize(
    "feature",
    [
        "ip_ref_rate",
        "boundary_ref_rate",
        "ip_measured_rate",
        "integral_ip_error",
        "integral_boundary_radii_error",
        "previous_action",
        "target_preview",
    ],
)
def test_controller_state_v7_keeps_rate_integral_previous_action_and_preview_features(feature: str) -> None:
    _env_v6, env_v7 = _env_pair()

    schema = env_v7.export_schema()

    assert feature in schema["feature_order"]
    assert feature in schema["feature_slices"]


def test_controller_state_v7_runtime_observation_has_no_episode_phase_signal() -> None:
    _env_v6, env_v7 = _env_pair()

    obs = env_v7.reset()
    result = env_v7.step(torch.zeros((2, env_v7.action_dim), dtype=torch.float32))
    schema = env_v7.export_schema()

    assert obs.shape == (2, env_v7.obs_dim)
    assert result.obs.shape == (2, env_v7.obs_dim)
    assert env_v7.critic_obs().shape == (2, env_v7.critic_obs_dim)
    assert "step_norm" not in schema["feature_slices"]
    assert torch.isfinite(obs).all()
    assert torch.isfinite(result.obs).all()
