from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.aggregate_reward_sweep import aggregate, score_eval_row, write_outputs
from scripts.build_reward_sweep_manifest import build_manifest, build_variants


ROOT = Path(__file__).resolve().parents[1]
SWEEP_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_48gpu_144x1m.sbatch"


def test_reward_sweep_manifest_has_144_unique_variants() -> None:
    variants = build_variants()
    assert len(variants) == 144
    assert [variant["index"] for variant in variants] == list(range(144))
    assert len({variant["name"] for variant in variants}) == 144
    assert len({variant["folder"] for variant in variants}) == 144
    assert variants[0]["folder"] == "v000_s0_i0_c0_a0"
    assert variants[-1]["folder"] == "v143_s3_i2_c3_a2"
    current_by_regime = {
        variant["current_regime"]: (
            variant["reward"]["current_weight"],
            variant["reward"]["current_soft_fraction"],
        )
        for variant in variants
        if variant["shape_regime"] == "s0" and variant["ip_regime"] == "i0" and variant["actuator_regime"] == "a0"
    }
    assert current_by_regime == {
        "c0": (4.0, 0.90),
        "c1": (6.0, 0.90),
        "c2": (6.0, 0.85),
        "c3": (8.0, 0.85),
    }


def test_reward_sweep_array_task_mapping_is_three_variants_each() -> None:
    variants = build_variants()
    for task_id in range(48):
        mapped = [task_id * 3 + local_index for local_index in range(3)]
        assert mapped == [variants[index]["index"] for index in mapped]
    assert 47 * 3 + 2 == 143


def test_reward_sweep_job_blocks_stale_name_leaks() -> None:
    text = SWEEP_JOB.read_text(encoding="utf-8")
    assert "unset RUN_NAME" in text
    assert "unset TRAIN_OUTPUT" in text
    assert "unset WANDB_PROJECT_NAME" in text
    assert "--no-save-checkpoints" in text
    assert "--reward-sweep-mode" in text
    assert "--wandb-name \"${VARIANT_FOLDER}\"" in text


def test_reward_sweep_score_penalizes_bad_current_and_completion() -> None:
    good = {
        "shape_error_mean_m_late": "0.03",
        "shape_error_max_m_late": "0.08",
        "ip_error_a_late": "25000",
        "current_over_limit_a_late": "0",
        "current_over_limit_fraction_late": "0",
        "mean_episode_completion": "1.0",
        "boundary_found_late_min": "1.0",
    }
    bad = dict(good)
    bad["current_over_limit_a_late"] = "20000"
    bad["mean_episode_completion"] = "0.5"
    assert score_eval_row(bad) > score_eval_row(good)


def test_reward_sweep_aggregator_handles_complete_and_missing_runs(tmp_path: Path) -> None:
    manifest = build_manifest()
    manifest["variants"] = manifest["variants"][:2]
    (tmp_path / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    run_dir = tmp_path / manifest["variants"][0]["folder"]
    run_dir.mkdir(parents=True)
    with (run_dir / "eval_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "shape_error_mean_m_late",
                "shape_error_max_m_late",
                "ip_error_a_late",
                "current_over_limit_a_late",
                "current_over_limit_fraction_late",
                "mean_episode_completion",
                "boundary_found_late_min",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": 100000,
                "shape_error_mean_m_late": 0.03,
                "shape_error_max_m_late": 0.08,
                "ip_error_a_late": 25000,
                "current_over_limit_a_late": 0,
                "current_over_limit_fraction_late": 0,
                "mean_episode_completion": 1.0,
                "boundary_found_late_min": 1.0,
            }
        )
    (run_dir / "policy_validation.json").write_text(json.dumps({"status": "sweep_completed"}), encoding="utf-8")

    result = aggregate(tmp_path)
    write_outputs(tmp_path, result)

    assert result["summaries"][0]["variant_index"] == 0
    assert result["failures"][0]["variant_index"] == 1
    assert (tmp_path / "reward_sweep_summary.csv").exists()
    assert (tmp_path / "reward_sweep_top20.md").exists()
    assert (tmp_path / "reward_sweep_best.json").exists()
    assert (tmp_path / "reward_sweep_failures.csv").exists()
