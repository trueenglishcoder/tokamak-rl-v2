from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.export.cli import _validate_export_checkpoint
from tokamak_rl_v2.training.policy_pipeline import evaluate_policy_gates
from tokamak_rl_v2.training.policy_pipeline import main as policy_pipeline_main
from tokamak_rl_v2.training.policy_pipeline import _preflight_artifact_failure
from tokamak_rl_v2.training.cli import main as training_cli_main
from tokamak_rl_v2.env import TokamakMagneticControlEnv


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / "configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml"
REPLAY_BOUNDARY_DIR = (ROOT.parent / "tokamak-sim/runs/t15md_limited_replay_dataset").resolve()


def _pin_replay_boundary_dir(raw: dict[str, object]) -> None:
    reference = raw.setdefault("reference", {})
    assert isinstance(reference, dict)
    boundary = reference.setdefault("boundary", {})
    assert isinstance(boundary, dict)
    boundary["replay_reference_dir"] = str(REPLAY_BOUNDARY_DIR)


def test_production_config_uses_single_clean_path() -> None:
    cfg = load_experiment_config(PRODUCTION_CONFIG)
    assert cfg.training.production_mode is True
    assert cfg.sim.reset_source == "csv_initial_states"
    assert cfg.reference.ip.kind == "segmented_profile"
    assert cfg.reference.boundary.kind == "t15_replay_segment_conditioned"
    assert cfg.reference.boundary.replay_reference_dir is not None
    assert cfg.reference.boundary.replay_reference_dir.exists()
    assert cfg.reward.kind == "tcv_derivative"
    assert cfg.sim.terminate_on_boundary_loss is True
    assert cfg.sim.terminate_on_current_limit is True
    assert cfg.sim.action_contract == "jdot_command"
    assert cfg.observation.actor_kind == "controller_state_v4"
    assert cfg.observation.critic_kind == "privileged_training_state_v1"


def test_current_canonical_experiment_configs_load_cleanly() -> None:
    paths = [
        ROOT / "configs/experiments/t15_csv_initial_segmented_profile_boundary_mpo.yaml",
        ROOT / "configs/experiments/t15_csv_initial_single_segment_0p1s_static_boundary_mpo.yaml",
    ]
    for path in paths:
        load_experiment_config(path)


def test_production_config_rejects_shot_fragments(tmp_path: Path) -> None:
    raw = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["sim"]["shot_fragments"] = {"shot_ids": ["3856"]}
    path = tmp_path / "bad_shot_fragments.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="shot_fragments"):
        load_experiment_config(path)


def test_production_config_rejects_curriculum(tmp_path: Path) -> None:
    raw = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["curriculum"] = {"enabled": True}
    path = tmp_path / "bad_curriculum.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="curriculum"):
        load_experiment_config(path)


def test_production_config_rejects_old_hold_reset_boundary(tmp_path: Path) -> None:
    raw = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["reference"]["boundary"] = {"kind": "hold_reset_boundary"}
    path = tmp_path / "bad_hold_reset_boundary.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="t15_replay_segment_conditioned"):
        load_experiment_config(path)


def test_production_pipeline_rejects_allow_failed_gates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow-failed-gates"):
        policy_pipeline_main(
            [
                "--config",
                str(PRODUCTION_CONFIG),
                "--output-dir",
                str(tmp_path / "run"),
                "--allow-failed-gates",
                "--wandb-mode",
                "disabled",
            ]
        )


def test_production_pipeline_rejects_skip_controller_rollout_gate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="skip-controller-rollout-gate"):
        policy_pipeline_main(
            [
                "--config",
                str(PRODUCTION_CONFIG),
                "--output-dir",
                str(tmp_path / "run"),
                "--skip-controller-rollout-gate",
                "--wandb-mode",
                "disabled",
            ]
        )


def test_production_pipeline_rejects_non_full_controller_rollout_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="controller-rollout-steps"):
        policy_pipeline_main(
            [
                "--config",
                str(PRODUCTION_CONFIG),
                "--output-dir",
                str(tmp_path / "run"),
                "--controller-rollout-steps",
                "100",
                "--wandb-mode",
                "disabled",
            ]
        )


def test_production_config_rejects_corner_smoothing_seconds(tmp_path: Path) -> None:
    raw = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["reference"]["ip"]["corner_smoothing_s"] = 0.02
    path = tmp_path / "bad_corner_smoothing.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="smooth_ramps"):
        load_experiment_config(path)


def test_plain_training_cli_rejects_production_mode() -> None:
    with pytest.raises(ValueError, match="train_policy_pipeline.py"):
        training_cli_main(["--config", str(PRODUCTION_CONFIG)])


def test_plain_training_cli_accepts_non_production_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["training"]["production_mode"] = False
    _pin_replay_boundary_dir(raw)
    path = tmp_path / "non_production.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    calls: list[dict[str, object]] = []

    class DummyTrainer:
        def __init__(self, *args, **kwargs) -> None:
            calls.append({"args": args, "kwargs": kwargs})

        def train(self) -> dict[str, object]:
            return {"status": "ok"}

    monkeypatch.setattr("tokamak_rl_v2.training.cli.Trainer", DummyTrainer)
    rc = training_cli_main(["--config", str(path)])
    assert rc == 0
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.__setitem__("unknown_top_level", 1), "config contains unsupported keys"),
        (lambda raw: raw["sim"].__setitem__("unknown_sim_key", 1), "sim contains unsupported keys"),
        (lambda raw: raw["reference"].__setitem__("unknown_reference_key", 1), "reference contains unsupported keys"),
        (lambda raw: raw["reference"]["ip"].__setitem__("unknown_ip_key", 1), "reference.ip contains unsupported keys"),
        (lambda raw: raw["reference"]["boundary"].__setitem__("unknown_boundary_key", 1), "reference.boundary contains unsupported keys"),
        (lambda raw: raw["observation"].__setitem__("unknown_observation_key", 1), "observation contains unsupported keys"),
        (lambda raw: raw["randomization"].__setitem__("unknown_randomization_key", 1), "randomization contains unsupported keys"),
        (lambda raw: raw["network"].__setitem__("unknown_network_key", 1), "network contains unsupported keys"),
        (lambda raw: raw["learner"].__setitem__("unknown_learner_key", 1), "learner contains unsupported keys"),
        (lambda raw: raw["training"].__setitem__("unknown_training_key", 1), "training contains unsupported keys"),
    ],
)
def test_loader_rejects_unknown_keys_in_strict_sections(tmp_path: Path, mutate, message: str) -> None:
    raw = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "bad_unknown_key.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)


def test_production_config_rejects_duration_mismatch(tmp_path: Path) -> None:
    raw = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["reference"]["duration_s"] = 0.499
    _pin_replay_boundary_dir(raw)
    path = tmp_path / "bad_duration.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="reference.duration_s == sim.max_episode_steps"):
        load_experiment_config(path)


def test_environment_rejects_reference_timestep_mismatch(tmp_path: Path) -> None:
    raw = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    raw["sim"]["config_path"] = str((ROOT.parent / "tokamak-sim/configs/T15MD_new_data.toml").resolve())
    raw["training"]["production_mode"] = False
    raw["sim"]["reset_source"] = "initial_ranges"
    raw["sim"]["csv_initial_state_library"] = None
    raw["sim"].pop("csv_initial_state_split", None)
    raw["sim"]["initial_ranges"] = json.loads(
        (ROOT / "configs/experiments/t15_static_boundary.yaml").read_text(encoding="utf-8")
    )["sim"]["initial_ranges"]
    raw["reference"]["ip"] = json.loads(
        (ROOT / "configs/experiments/t15_static_boundary.yaml").read_text(encoding="utf-8")
    )["reference"]["ip"]
    raw["reference"]["boundary"] = {"kind": "static_initial_parameters"}
    raw["reference"]["duration_s"] = 1.0
    raw["reference"]["t_step"] = 0.002
    raw["sim"]["compute_backend"] = "cpu"
    raw["sim"]["max_episode_steps"] = 4
    path = tmp_path / "bad_timestep.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    cfg = load_experiment_config(path)
    with pytest.raises(ValueError, match="reference.t_step must match"):
        TokamakMagneticControlEnv(cfg, batch_size=1, device="cpu", seed=1)


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


def test_manual_export_accepts_controller_state_v5_checkpoint(tmp_path: Path) -> None:
    checkpoint = {
        "checkpoint_version": 2,
        "schema": {
            "observation_kind": "controller_state_v5",
            "action_contract": "absolute_jdot_command_v1",
            "obs_dim": 502,
            "action_dim": 9,
        },
        "network": {"hidden_dim": 8},
        "actor_state_dict": {},
        "normalization": {},
        "metadata": {},
    }

    _validate_export_checkpoint(checkpoint, checkpoint=tmp_path / "v5.pt")


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
            "ip_error_a": 20000.0,
            "ip_error_a_late": 20000.0,
            "action_rms": 0.02,
            "mean_episode_completion": 1.0,
            "min_episode_completion": 1.0,
        },
        no_control={"ip_error_a": 50000.0, "ip_error_a_late": 50000.0},
        tail_losses={"tail100.policy_weight_max": 0.0, "tail100.sampled_q_spread": 0.0},
        action_samples=20,
        min_boundary_found=0.999,
        max_current_over_limit_a=0.0,
        max_shape_error_m=0.03,
        min_ip_improvement_frac=0.25,
        min_ip_improvement_a=20000.0,
        max_ip_error_a=25000.0,
        max_ip_error_late_a=25000.0,
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
        max_controller_ip_error_a=25000.0,
    )
    names = {check["name"] for check in gates["checks"]}
    assert "mpo_policy_weights_nonuniform" not in names
    assert "mpo_sampled_q_spread" not in names


def test_production_preflight_allows_same_shot_close_train_holdout_rows(tmp_path: Path) -> None:
    cfg = _temp_production_config_with_artifacts(tmp_path, times=[0.00, 0.20], splits=["train", "holdout"])
    assert _preflight_artifact_failure(cfg) is None


def test_production_preflight_accepts_non_overlapping_split(tmp_path: Path) -> None:
    cfg = _temp_production_config_with_artifacts(tmp_path, times=[0.00, 1.10], splits=["train", "holdout"])
    assert _preflight_artifact_failure(cfg) is None


def _temp_production_config_with_artifacts(tmp_path: Path, *, times: list[float], splits: list[str]):
    raw = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    npz_path = tmp_path / "t15_csv_initial_states.npz"
    json_path = tmp_path / "t15_csv_initial_states.json"
    limits_path = tmp_path / "t15_reference_limits.json"
    count = len(times)
    np.savez(
        npz_path,
        shot_id=np.asarray(["3856"] * count),
        source_index=np.asarray(list(range(count)), dtype=np.int64),
        time_s=np.asarray(times, dtype=float),
        ip0=np.asarray([120000.0 + 1000.0 * idx for idx in range(count)], dtype=float),
        pfc0=np.zeros((count, 6), dtype=float),
        sol0=np.zeros((count, 3), dtype=float),
        split=np.asarray(splits),
    )
    json_path.write_text(
        json.dumps(
            {
                "accepted_rows": 1000,
                "train_rows": 1000,
                "holdout_rows": 100,
                "accepted_by_shot": {"3856": 100},
                "split_by_shot": {"3856": {"train": 80, "holdout": 10}},
            }
        ),
        encoding="utf-8",
    )
    limits_path.write_text(
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
                "shot_count": 1,
                "shot_ids": ["3856"],
            }
        ),
        encoding="utf-8",
    )
    raw["sim"]["csv_initial_state_library"] = str(npz_path)
    raw["reference"]["ip"]["limits_path"] = str(limits_path)
    _pin_replay_boundary_dir(raw)
    cfg_path = tmp_path / "production.json"
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")
    return load_experiment_config(cfg_path)
