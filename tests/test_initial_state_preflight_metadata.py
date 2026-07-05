from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.training.policy_pipeline import _preflight_artifact_failure


ROOT = Path(__file__).resolve().parents[1]
NO_STEP_LONG60_CONFIG = ROOT / "configs/experiments/t15_synth_empirical_long60_no_step_0p1s_tcvjdot_mpo_balanced.yaml"


def _write_initial_library_with_parent_reset_metadata(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shot_id=np.asarray([940000, 950000], dtype=np.int64),
        source_index=np.asarray([0, 0], dtype=np.int64),
        time_s=np.asarray([0.0, 0.0], dtype=np.float64),
        ip0=np.asarray([350000.0, 360000.0], dtype=np.float32),
        pfc0=np.zeros((2, 6), dtype=np.float32),
        sol0=np.zeros((2, 3), dtype=np.float32),
        split=np.asarray(["train", "holdout"]),
        difficulty_bin=np.asarray(["flat", "medium_up"]),
        parent_reset_shot_id=np.asarray([3856, 3857], dtype=np.int64),
        parent_reset_source_index=np.asarray([525, 290], dtype=np.int64),
        parent_reset_time_s=np.asarray([0.574, 0.339], dtype=np.float64),
    )


def test_initial_state_preflight_accepts_synthetic_parent_reset_metadata(tmp_path: Path) -> None:
    raw = json.loads(NO_STEP_LONG60_CONFIG.read_text(encoding="utf-8"))
    machine = tmp_path / "T15MD_new_data.toml"
    initial_library = tmp_path / "t15_synth_empirical_long60_initial_states.npz"
    replay_reference_dir = tmp_path / "t15_synth_empirical_long60"
    machine.write_text("# test placeholder\n", encoding="utf-8")
    replay_reference_dir.mkdir()
    _write_initial_library_with_parent_reset_metadata(initial_library)

    raw["sim"]["config_path"] = str(machine)
    raw["sim"]["csv_initial_state_library"] = str(initial_library)
    raw["reference"]["boundary"]["replay_reference_dir"] = str(replay_reference_dir)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    cfg = load_experiment_config(config_path)

    assert _preflight_artifact_failure(cfg) is None
