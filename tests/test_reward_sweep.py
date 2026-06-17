from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.aggregate_reward_sweep import aggregate, score_eval_row, write_outputs
from scripts.build_reward_sweep_manifest import build_manifest, build_variants
from scripts.summarize_reward_sweep_physical import summarize


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


def _write_physical_run(
    root: Path,
    variant: dict,
    *,
    completion: float = 1.0,
    boundary: float = 1.0,
    shape: float = 0.04,
    ip: float = 40000.0,
    current_max: float = 0.0,
    current_fraction: float = 0.0,
    actor_eval: bool = True,
) -> None:
    run_dir = root / variant["folder"]
    run_dir.mkdir(parents=True)
    (run_dir / "reward_variant.json").write_text(json.dumps({"variant": variant}), encoding="utf-8")
    validation = {"status": "sweep_completed"}
    if actor_eval:
        validation["actor_eval"] = {
            "mean_episode_completion": completion,
            "boundary_found_late_min": boundary,
            "terminated_boundary": 0.0 if boundary >= 0.999 else 0.1,
            "shape_error_mean_m_late": shape,
            "shape_error_max_m_late": shape * 2.0,
            "ip_error_a_late": ip,
            "current_over_limit_a_late_max": current_max,
            "current_over_limit_fraction_late": current_fraction,
            "current_usage_fraction_late_max": 0.8 if current_max == 0.0 else 1.2,
            "action_rms_late": 0.1,
            "delta_action_rms_late": 0.01,
        }
    (run_dir / "policy_validation.json").write_text(json.dumps(validation), encoding="utf-8")
    with (run_dir / "eval_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env_step",
                "mean_episode_completion",
                "boundary_found_late_min",
                "shape_error_mean_m_late",
                "ip_error_a_late",
                "current_over_limit_fraction_late",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "env_step": 100000,
                "mean_episode_completion": completion,
                "boundary_found_late_min": boundary,
                "shape_error_mean_m_late": shape,
                "ip_error_a_late": ip,
                "current_over_limit_fraction_late": current_fraction,
            }
        )


def test_physical_sweep_summary_uses_actor_eval_and_marks_missing_actor_eval(tmp_path: Path) -> None:
    manifest = build_manifest()
    manifest["variants"] = manifest["variants"][:2]
    (tmp_path / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_physical_run(tmp_path, manifest["variants"][0], ip=45000.0)
    _write_physical_run(tmp_path, manifest["variants"][1], actor_eval=False)

    out_dir = tmp_path / "analysis"
    summarize(tmp_path, out_dir, top=10, min_completion=0.95, min_boundary_late=0.999, max_terminated_boundary=0.001, max_current_over_limit_a_max=250000.0, max_current_over_limit_fraction_late=0.5)

    rows = list(csv.DictReader((out_dir / "physical_sweep_summary.csv").open()))
    assert len(rows) == 2
    assert rows[0]["folder"] == manifest["variants"][0]["folder"]
    assert rows[0]["selection_valid"] == "True"
    assert rows[0]["ip_error_a_late"] == "45000.0"
    assert rows[1]["selection_valid"] == "False"
    assert rows[1]["selection_reason"] == "missing_actor_eval"
    assert (out_dir / "physical_best_candidate.json").exists()
    assert (out_dir / "physical_selection_report.md").exists()


def test_physical_sweep_pareto_excludes_dominated_runs(tmp_path: Path) -> None:
    variants = build_variants()[:2]
    (tmp_path / "variants.json").write_text(json.dumps({"variants": variants}), encoding="utf-8")
    _write_physical_run(tmp_path, variants[0], shape=0.03, ip=30000.0, current_max=0.0, current_fraction=0.0)
    _write_physical_run(tmp_path, variants[1], shape=0.06, ip=60000.0, current_max=1000.0, current_fraction=0.1)

    out_dir = tmp_path / "analysis"
    summarize(tmp_path, out_dir, top=10, min_completion=0.95, min_boundary_late=0.999, max_terminated_boundary=0.001, max_current_over_limit_a_max=250000.0, max_current_over_limit_fraction_late=0.5)

    front = list(csv.DictReader((out_dir / "physical_pareto_front.csv").open()))
    assert [row["folder"] for row in front] == [variants[0]["folder"]]


def test_physical_sweep_regime_summary_groups_reward_axes(tmp_path: Path) -> None:
    manifest = build_manifest()
    manifest["variants"] = manifest["variants"][:4]
    (tmp_path / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    for variant in manifest["variants"]:
        _write_physical_run(tmp_path, variant)

    out_dir = tmp_path / "analysis"
    summarize(tmp_path, out_dir, top=2, min_completion=0.95, min_boundary_late=0.999, max_terminated_boundary=0.001, max_current_over_limit_a_max=250000.0, max_current_over_limit_fraction_late=0.5)

    regimes = list(csv.DictReader((out_dir / "physical_regime_summary.csv").open()))
    assert {row["regime_kind"] for row in regimes} == {"shape", "ip", "current", "actuator"}
    assert any(row["regime_kind"] == "current" and row["regime"] == "c0" for row in regimes)
