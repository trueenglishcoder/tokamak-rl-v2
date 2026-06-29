from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.export.cli import _validate_export_checkpoint
from tokamak_rl_v2.training.cli import main as training_cli_main
from tokamak_rl_v2.training.policy_pipeline import evaluate_policy_gates


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_CONFIG = ROOT / "configs/experiments/t15_new_trim50_plain_gpu1e6_replay_window_0p1s_tcvjdot_mpo_balanced.yaml"


def _write_loadable_active_config(tmp_path: Path, mutate=None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = json.loads(ACTIVE_CONFIG.read_text(encoding="utf-8"))
    machine = tmp_path / "T15MD_new_data_trim50_plain_gpu_1e6_3856.toml"
    machine.write_text("# test placeholder\n", encoding="utf-8")
    replay_dir = tmp_path / "t15_new_trim50_plain_gpu1e6_replay_window_0p1s_oracle_targets"
    replay_dir.mkdir()
    raw["sim"]["config_path"] = str(machine)
    raw["sim"]["csv_initial_state_library"] = str(tmp_path / "oracle_initial_states.npz")
    raw["reference"]["boundary"]["replay_reference_dir"] = str(replay_dir)
    if mutate is not None:
        mutate(raw)
    path = tmp_path / "active_config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_active_trim50_config_uses_final_pipeline(tmp_path: Path) -> None:
    cfg = load_experiment_config(_write_loadable_active_config(tmp_path))

    assert cfg.training.production_mode is True
    assert cfg.training.steps == 100_000_000
    assert cfg.training.num_envs == 2048
    assert cfg.training.distributed_mode == "local_replay"
    assert cfg.sim.reset_source == "csv_initial_states"
    assert cfg.sim.csv_initial_state_split == "train"
    assert cfg.sim.action_contract == "jdot_command"
    assert cfg.sim.max_episode_steps == 100
    assert cfg.sim.delta_derivative_limits_aps is None
    assert cfg.reference.duration_s == pytest.approx(0.1)
    assert cfg.reference.t_step == pytest.approx(0.001)
    assert cfg.reference.ip.kind == "replay_window"
    assert cfg.reference.boundary.kind == "t15_replay_segment_conditioned"
    assert cfg.observation.actor_kind == "controller_state_v6"
    assert cfg.observation.critic_kind == "compact_training_state_v2"
    assert cfg.reward.kind == "tcv_derivative"
    assert cfg.reward.shape_mean_weight == pytest.approx(3.2)
    assert cfg.reward.shape_max_weight == pytest.approx(0.8)
    assert cfg.reward.ip_weight == pytest.approx(1.8)
    assert cfg.reward.ip_scale_a == pytest.approx(25_000.0)
    assert cfg.reward.smoothmax_alpha == pytest.approx(-5.0)


def test_active_config_rejects_old_boundary_and_action_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="t15_replay_segment_conditioned"):
        load_experiment_config(
            _write_loadable_active_config(
                tmp_path / "boundary",
                lambda raw: raw["reference"].__setitem__("boundary", {"kind": "hold_reset_boundary"}),
            )
        )

    with pytest.raises(ValueError, match="jdot_command"):
        load_experiment_config(
            _write_loadable_active_config(
                tmp_path / "action",
                lambda raw: raw["sim"].__setitem__("action_contract", "delta_jdot"),
            )
        )


def test_active_config_rejects_stale_observation_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="controller_state_v6"):
        load_experiment_config(
            _write_loadable_active_config(
                tmp_path,
                lambda raw: raw["observation"].__setitem__("actor_kind", "controller_state_v4"),
            )
        )


def test_loader_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config contains unsupported keys"):
        load_experiment_config(
            _write_loadable_active_config(
                tmp_path,
                lambda raw: raw.__setitem__("unknown_top_level", 1),
            )
        )


def test_plain_training_cli_rejects_production_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train_policy_pipeline.py"):
        training_cli_main(["--config", str(_write_loadable_active_config(tmp_path))])


def test_manual_export_rejects_compact_joint_schema_checkpoint(tmp_path: Path) -> None:
    checkpoint = {
        "checkpoint_version": 2,
        "schema": {"observation_kind": "compact_joint_state_v2", "obs_dim": 4, "action_dim": 2},
        "network": {"hidden_dim": 8},
        "actor_state_dict": {},
        "normalization": {},
        "metadata": {},
    }
    with pytest.raises(ValueError, match="compact_joint_state_v2"):
        _validate_export_checkpoint(checkpoint, checkpoint=tmp_path / "bad.pt")


def test_manual_export_accepts_controller_state_v6_checkpoint(tmp_path: Path) -> None:
    checkpoint = {
        "checkpoint_version": 2,
        "schema": {
            "observation_kind": "controller_state_v6",
            "action_contract": "absolute_jdot_command_v1",
            "obs_dim": 535,
            "action_dim": 9,
        },
        "network": {"hidden_dim": 8},
        "actor_state_dict": {},
        "normalization": {},
        "metadata": {},
    }

    _validate_export_checkpoint(checkpoint, checkpoint=tmp_path / "v6.pt")


def test_production_policy_gates_do_not_include_mpo_diagnostics() -> None:
    gates = evaluate_policy_gates(
        actor_eval={
            "boundary_found": 1.0,
            "boundary_found_late_min": 1.0,
            "current_over_limit_a": 0.0,
            "current_over_limit_a_max": 0.0,
            "current_over_limit_a_late_max": 0.0,
            "shape_error_mean_m": 0.02,
            "shape_error_mean_m_late": 0.02,
            "ip_error_a": 20_000.0,
            "ip_error_a_late": 20_000.0,
            "action_rms": 0.02,
            "mean_episode_completion": 1.0,
            "min_episode_completion": 1.0,
        },
        no_control={"ip_error_a": 50_000.0, "ip_error_a_late": 50_000.0},
        tail_losses={"tail100.policy_weight_max": 0.0, "tail100.sampled_q_spread": 0.0},
        action_samples=20,
        min_boundary_found=0.999,
        max_current_over_limit_a=0.0,
        max_shape_error_m=0.03,
        min_ip_improvement_frac=0.25,
        min_ip_improvement_a=20_000.0,
        max_ip_error_a=25_000.0,
        max_ip_error_late_a=25_000.0,
        min_action_rms=0.005,
        max_action_rms=0.5,
        min_mean_episode_completion=0.95,
        min_episode_completion=0.90,
        min_baseline_ip_error_a=0.0,
        min_baseline_ip_error_late_a=0.0,
        always_require_ip_improvement=False,
        min_policy_weight_extra=1.0e-4,
        min_sampled_q_spread=1.0e-8,
        include_mpo_gates=False,
        require_controller_rollout=False,
        controller_rollout={},
        max_controller_shape_error_m=0.03,
        max_controller_ip_error_a=25_000.0,
    )
    names = {check["name"] for check in gates["checks"]}
    assert "mpo_policy_weights_nonuniform" not in names
    assert "mpo_sampled_q_spread" not in names
