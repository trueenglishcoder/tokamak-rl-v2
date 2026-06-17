from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.aggregate_reward_sweep import aggregate, score_eval_row, write_outputs
from scripts.build_reward_sweep_manifest import build_manifest, build_variants
from scripts.build_reward_sweep_rerun_manifest import build as build_rerun_manifest
from scripts.summarize_two_pass_reward_sweep_physical import summarize_two_pass
from scripts.summarize_reward_sweep_physical import summarize
from scripts.submit_two_pass_reward_sweep import submit_chain


ROOT = Path(__file__).resolve().parents[1]
PASS1_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_96gpu_pass1_broad.sbatch"
PASS2_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_96gpu_pass2_focused.sbatch"
RERUN_JOB = ROOT / "jobs/sweep_t15_csv_segmented_profile_rewards_rerun_1gpu.sbatch"


def test_reward_sweep_broad_manifest_has_96_unique_variants() -> None:
    variants = build_variants("broad")
    assert len(variants) == 96
    assert [variant["index"] for variant in variants] == list(range(96))
    assert len({variant["name"] for variant in variants}) == 96
    assert len({variant["folder"] for variant in variants}) == 96
    assert variants[0]["folder"] == "b000_s0_i0_c0_d0"
    assert variants[-1]["folder"] == "b095_s3_i2_c3_d1"
    current_by_regime = {
        variant["current_regime"]: (
            variant["reward"]["current_weight"],
            variant["reward"]["current_soft_fraction"],
            variant["reward"]["current_bad_fraction"],
        )
        for variant in variants
        if variant["shape_regime"] == "s0" and variant["ip_regime"] == "i0" and variant["derivative_regime"] == "d0"
    }
    assert current_by_regime == {
        "c0": (0.25, 1.0, 1.4),
        "c1": (1.0, 1.0, 1.4),
        "c2": (3.0, 1.0, 1.4),
        "c3": (6.0, 1.0, 1.4),
    }
    derivative_by_regime = {
        variant["derivative_regime"]: (
            variant["reward"]["derivative_weight"],
            variant["reward"]["derivative_soft_fraction"],
            variant["reward"]["derivative_bad_fraction"],
            variant["reward"]["action_weight"],
            variant["reward"]["delta_action_weight"],
        )
        for variant in variants
        if variant["shape_regime"] == "s0" and variant["ip_regime"] == "i0" and variant["current_regime"] == "c0"
    }
    assert derivative_by_regime == {
        "d0": (0.05, 1.0, 1.4, 0.0, 0.0),
        "d1": (0.5, 1.0, 1.4, 0.0, 0.0),
    }


def test_reward_sweep_focused_manifest_has_192_unique_variants() -> None:
    center = {
        "shape_mean_weight": 8.0,
        "shape_max_weight": 2.5,
        "ip_weight": 4.0,
        "current_weight": 3.0,
        "derivative_weight": 0.5,
    }
    variants = build_variants("focused", center)
    assert len(variants) == 192
    assert [variant["index"] for variant in variants] == list(range(192))
    assert len({variant["folder"] for variant in variants}) == 192
    assert variants[0]["folder"] == "f000_sf0_if0_cf0_df0"
    assert variants[-1]["folder"] == "f191_sf3_if3_cf3_df2"
    assert variants[0]["reward"]["current_soft_fraction"] == 1.0
    assert variants[0]["reward"]["current_bad_fraction"] == 1.4
    assert variants[0]["reward"]["action_weight"] == 0.0
    assert variants[0]["reward"]["delta_action_weight"] == 0.0


def test_reward_sweep_array_task_mappings() -> None:
    broad = build_variants("broad")
    for task_id in range(96):
        assert task_id == broad[task_id]["index"]
    focused = build_variants(
        "focused",
        {
            "shape_mean_weight": 8.0,
            "shape_max_weight": 2.5,
            "ip_weight": 4.0,
            "current_weight": 3.0,
            "derivative_weight": 0.5,
        },
    )
    for task_id in range(96):
        mapped = [task_id * 2 + local_index for local_index in range(2)]
        assert mapped == [focused[index]["index"] for index in mapped]
    assert 95 * 2 + 1 == 191


def test_reward_sweep_job_blocks_stale_name_leaks() -> None:
    for path in (PASS1_JOB, PASS2_JOB, RERUN_JOB):
        text = path.read_text(encoding="utf-8")
        assert "unset RUN_NAME" in text
        assert "unset TRAIN_OUTPUT" in text
        assert "unset WANDB_PROJECT_NAME" in text
        assert "run_reward_sweep_candidate.py" in text
        assert "apply_t15_actuator_limits.py" in text
    runner = (ROOT / "scripts/run_reward_sweep_candidate.py").read_text(encoding="utf-8")
    assert "--no-save-checkpoints" in runner
    assert "--reward-sweep-mode" in runner
    assert "--no-export" in runner
    assert "--wandb-optional" in runner
    assert "shutil.rmtree(output_dir / \"exports\"" in runner
    assert "shutil.rmtree(output_dir / \"checkpoints\"" in runner


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


def test_physical_sweep_best_candidate_falls_back_to_imperfect_actor_eval(tmp_path: Path) -> None:
    manifest = build_manifest()
    manifest["variants"] = manifest["variants"][:2]
    (tmp_path / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_physical_run(tmp_path, manifest["variants"][0], completion=0.4, boundary=0.9, shape=0.08, ip=90000.0)
    _write_physical_run(tmp_path, manifest["variants"][1], actor_eval=False)

    out_dir = tmp_path / "analysis"
    summarize(tmp_path, out_dir, top=10, min_completion=0.95, min_boundary_late=0.999, max_terminated_boundary=0.001, max_current_over_limit_a_max=250000.0, max_current_over_limit_fraction_late=0.5)

    best = json.loads((out_dir / "physical_best_candidate.json").read_text(encoding="utf-8"))
    assert best["valid_candidates"] == 0
    assert best["best_candidate_passed_hard_filters"] is False
    assert best["best_candidate"]["folder"] == manifest["variants"][0]["folder"]
    assert "shape_mean_weight" in best["best_candidate"]


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
    assert {row["regime_kind"] for row in regimes} == {"shape", "ip", "current", "derivative"}
    assert any(row["regime_kind"] == "current" and row["regime"] == "c0" for row in regimes)


def test_reward_sweep_rerun_manifest_finds_missing_and_failed_variants(tmp_path: Path) -> None:
    pass1 = tmp_path / "pass1_broad"
    manifest = build_manifest("broad")
    manifest["variants"] = manifest["variants"][:3]
    pass1.mkdir()
    (pass1 / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_physical_run(pass1, manifest["variants"][0])
    failed_dir = pass1 / manifest["variants"][1]["folder"]
    failed_dir.mkdir()
    (failed_dir / "policy_validation.json").write_text(json.dumps({"status": "sweep_failed_training"}), encoding="utf-8")

    rerun = build_rerun_manifest(tmp_path)

    reasons = {item["folder"]: item["reason"] for item in rerun["variants"]}
    assert reasons[manifest["variants"][1]["folder"]] == "sweep_failed_training"
    assert reasons[manifest["variants"][2]["folder"]] == "missing_folder"
    assert rerun["missing_or_failed_count"] == 2


def test_two_pass_summary_writes_final_recommendation(tmp_path: Path) -> None:
    pass1 = tmp_path / "pass1_broad"
    pass2 = tmp_path / "pass2_focused"
    pass1.mkdir()
    pass2.mkdir()
    pass1_manifest = build_manifest("broad")
    pass1_manifest["variants"] = pass1_manifest["variants"][:2]
    center = pass1_manifest["variants"][0]["reward"]
    pass2_manifest = build_manifest("focused", center)
    pass2_manifest["variants"] = pass2_manifest["variants"][:2]
    (pass1 / "variants.json").write_text(json.dumps(pass1_manifest), encoding="utf-8")
    (pass2 / "variants.json").write_text(json.dumps(pass2_manifest), encoding="utf-8")
    _write_physical_run(pass1, pass1_manifest["variants"][0], shape=0.04, ip=40000.0)
    _write_physical_run(pass1, pass1_manifest["variants"][1], shape=0.05, ip=50000.0)
    _write_physical_run(pass2, pass2_manifest["variants"][0], shape=0.03, ip=25000.0)
    _write_physical_run(pass2, pass2_manifest["variants"][1], shape=0.06, ip=60000.0, current_max=1000.0)

    out_dir = tmp_path / "selection"
    result = summarize_two_pass(tmp_path, out_dir)

    assert result["recommended_candidate"]["sweep_pass"] == "pass2_focused"
    assert result["recommended_candidate"]["folder"] == pass2_manifest["variants"][0]["folder"]
    assert (out_dir / "pass1_physical_summary.csv").exists()
    assert (out_dir / "pass1_physical_best_candidate.json").exists()
    assert (out_dir / "pass2_physical_summary.csv").exists()
    assert (out_dir / "combined_physical_summary.csv").exists()
    assert (out_dir / "combined_pareto_front.csv").exists()
    assert (out_dir / "combined_regime_summary.csv").exists()
    assert (out_dir / "missing_or_failed_variants.json").exists()
    assert (out_dir / "final_reward_recommendation.json").exists()
    assert (out_dir / "final_reward_selection_report.md").exists()


def test_submit_two_pass_chain_uses_slurm_dependencies(tmp_path: Path, monkeypatch) -> None:
    submitted: list[list[str]] = []
    jobids = iter(["111\n", "112\n", "113\n", "114\n"])

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, check, text, stdout, stderr):
        submitted.append(list(args))
        return Result(next(jobids))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.submit_two_pass_reward_sweep.subprocess.run", fake_run)

    payload = submit_chain(
        pass1_job=Path("jobs/pass1.sbatch"),
        pass1_aggregate_job=Path("jobs/agg1.sbatch"),
        pass2_job=Path("jobs/pass2.sbatch"),
        final_aggregate_job=Path("jobs/final.sbatch"),
    )

    assert payload["pass1_jobid"] == "111"
    assert payload["pass1_aggregate_jobid"] == "112"
    assert payload["pass2_jobid"] == "113"
    assert payload["final_aggregate_jobid"] == "114"
    assert payload["root"] == "outputs/t15_reward_sweep288_legal_1m_111"
    assert submitted[1][2] == "--dependency=afterany:111"
    assert submitted[2][2] == "--dependency=afterok:112"
    assert submitted[3][2] == "--dependency=afterany:113"
    assert (tmp_path / payload["root"] / "selection" / "submission_chain.json").exists()
