from __future__ import annotations

from tokamak_rl_v2.training.trainer import Trainer


def _safe_metrics(
    *,
    physical_cost_late: float,
    physical_cost_late_max: float,
) -> dict[str, float]:
    return {
        "mean_episode_completion": 1.0,
        "full_episode_success": 1.0,
        "min_episode_completion": 1.0,
        "padded_boundary_found_late_min": 1.0,
        "terminated_boundary_late_max": 0.0,
        "terminated_current_late_max": 0.0,
        "padded_current_over_limit_a_late_max": 0.0,
        "padded_current_over_limit_5ka_fraction_late": 0.0,
        "physical_cost_late": physical_cost_late,
        "physical_cost_late_max": physical_cost_late_max,
    }


def test_selection_score_prefers_lower_configured_physical_cost() -> None:
    worse = _safe_metrics(
        physical_cost_late=1.0,
        physical_cost_late_max=1.5,
    )
    better = _safe_metrics(
        physical_cost_late=0.5,
        physical_cost_late_max=0.7,
    )

    assert (
        Trainer._selection_score(better)
        > Trainer._selection_score(worse)
    )


def test_selection_score_ignores_auxiliary_usage_metrics() -> None:
    baseline = _safe_metrics(
        physical_cost_late=0.5,
        physical_cost_late_max=0.7,
    )
    high_usage = {
        **baseline,
        "current_margin_loss_late_max": 1.0,
        "derivative_margin_loss_late_max": 1.0,
        "max_current_fraction_late_max": 0.99,
        "max_derivative_fraction_late_max": 0.99,
        "current_usage_fraction_late_max": 0.99,
        "derivative_usage_late_max": 0.99,
        "action_saturation_fraction_late": 0.5,
    }

    assert (
        Trainer._selection_score(high_usage)
        == Trainer._selection_score(baseline)
    )


def test_selection_score_penalizes_real_failures_before_cost() -> None:
    safe = _safe_metrics(
        physical_cost_late=5.0,
        physical_cost_late_max=5.0,
    )

    current_violation = {
        **_safe_metrics(
            physical_cost_late=0.1,
            physical_cost_late_max=0.1,
        ),
        "padded_current_over_limit_a_late_max": 1.0,
        "padded_current_over_limit_5ka_fraction_late": 0.001,
    }

    boundary_failure = {
        **_safe_metrics(
            physical_cost_late=0.1,
            physical_cost_late_max=0.1,
        ),
        "padded_boundary_found_late_min": 0.99,
        "terminated_boundary_late_max": 0.01,
    }

    incomplete = {
        **_safe_metrics(
            physical_cost_late=0.1,
            physical_cost_late_max=0.1,
        ),
        "mean_episode_completion": 0.99,
        "full_episode_success": 0.99,
        "min_episode_completion": 0.99,
    }

    safe_score = Trainer._selection_score(safe)

    assert safe_score > Trainer._selection_score(current_violation)
    assert safe_score > Trainer._selection_score(boundary_failure)
    assert safe_score > Trainer._selection_score(incomplete)
