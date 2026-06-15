from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import math

try:
    import yaml
except Exception:  # pragma: no cover - optional local fallback
    yaml = None

from tokamak_rl_v2.config.schema import (
    BoundaryReferenceConfig,
    CurrentSafetyLimits,
    ExperimentConfig,
    InitialRanges,
    IpReferenceConfig,
    LearnerConfig,
    NetworkConfig,
    ObservationConfig,
    RandomizationConfig,
    Range,
    ReferenceConfig,
    RewardConfig,
    ShotFragmentConfig,
    SimConfig,
    TrainingConfig,
    TrapezoidValueRanges,
)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    text = source.read_text(encoding="utf-8")
    raw = json.loads(text) if yaml is None else yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ValueError(f"experiment config must be a mapping: {source}")
    base = source.parent
    sim = _sim(_mapping(raw.get("sim"), "sim"), base)
    cfg = ExperimentConfig(
        name=str(raw.get("name", source.stem)),
        sim=sim,
        reference=_reference(_mapping(raw.get("reference"), "reference")),
        observation=_observation(_mapping(raw.get("observation", {}), "observation")),
        reward=_reward(_mapping(raw.get("reward", {}), "reward")),
        randomization=_randomization(_mapping(raw.get("randomization", {}), "randomization")),
        network=_network(_mapping(raw.get("network", {}), "network")),
        learner=_learner(_mapping(raw.get("learner", {}), "learner")),
        training=_training(_mapping(raw.get("training", {}), "training"), base),
    )
    _validate_experiment_config(cfg)
    return cfg


def _mapping(raw: object, name: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return raw


def _range(raw: Mapping[str, Any], name: str) -> Range:
    out = Range(min=float(raw["min"]), max=float(raw["max"]))
    out.validate(name)
    return out


def _ranges(raw: Mapping[str, Any] | None) -> InitialRanges | None:
    if not raw:
        return None
    return InitialRanges(
        ip=_range(_mapping(raw.get("ip"), "initial_ranges.ip"), "initial_ranges.ip"),
        pfc_currents=tuple(_range(_mapping(v, f"pfc_currents.{k}"), f"pfc_currents.{k}") for k, v in sorted(_mapping(raw.get("pfc_currents"), "pfc_currents").items())),
        sol_currents=tuple(_range(_mapping(v, f"sol_currents.{k}"), f"sol_currents.{k}") for k, v in sorted(_mapping(raw.get("sol_currents"), "sol_currents").items())),
        boundary_parameters={str(k): _range(_mapping(v, f"boundary_parameters.{k}"), f"boundary_parameters.{k}") for k, v in _mapping(raw.get("boundary_parameters"), "boundary_parameters").items()},
    )


def _float_sequence(raw: object, name: str) -> tuple[float, ...]:
    if isinstance(raw, Mapping):
        values = [v for _k, v in sorted(raw.items())]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        raise ValueError(f"{name} must be a list or mapping")
    out = tuple(float(v) for v in values)
    for idx, value in enumerate(out):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name}[{idx}] must be finite and positive")
    return out


def _string_tuple(raw: object, name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        values = [str(part).strip() for part in raw]
    else:
        raise ValueError(f"{name} must be a comma-separated string or list")
    out = tuple(value for value in values if value)
    if raw is not None and not out:
        raise ValueError(f"{name} must not be empty")
    return out


def _current_safety_limits(raw: Mapping[str, Any] | None) -> CurrentSafetyLimits | None:
    if not raw:
        return None
    return CurrentSafetyLimits(
        pfc_currents=_float_sequence(raw.get("pfc_currents"), "current_safety_limits.pfc_currents"),
        sol_currents=_float_sequence(raw.get("sol_currents"), "current_safety_limits.sol_currents"),
    )


def _trapezoid_value_ranges(raw: Mapping[str, Any], name: str) -> TrapezoidValueRanges:
    return TrapezoidValueRanges(
        start=_range(_mapping(raw.get("start"), f"{name}.start"), f"{name}.start"),
        plateau=_range(_mapping(raw.get("plateau"), f"{name}.plateau"), f"{name}.plateau"),
        end=_range(_mapping(raw.get("end"), f"{name}.end"), f"{name}.end"),
    )


def _trapezoid_profile_tuple(raw: object, name: str) -> tuple[TrapezoidValueRanges, ...]:
    if isinstance(raw, Mapping):
        values = [v for _k, v in sorted(raw.items())]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        raise ValueError(f"{name} must be a list or mapping")
    out = tuple(_trapezoid_value_ranges(_mapping(value, f"{name}.{idx}"), f"{name}.{idx}") for idx, value in enumerate(values))
    if not out:
        raise ValueError(f"{name} must contain at least one profile")
    return out


def _shot_fragments(raw: Mapping[str, Any] | None, base: Path) -> ShotFragmentConfig | None:
    if not raw:
        return None
    kind = str(raw.get("kind", "idealized_t15_trapezoid"))
    return ShotFragmentConfig(
        kind=kind,
        ip_a=_trapezoid_value_ranges(_mapping(raw.get("ip_a"), "sim.shot_fragments.ip_a"), "sim.shot_fragments.ip_a"),
        pfc_currents=_trapezoid_profile_tuple(raw.get("pfc_currents"), "sim.shot_fragments.pfc_currents"),
        sol_currents=_trapezoid_profile_tuple(raw.get("sol_currents"), "sim.shot_fragments.sol_currents"),
        ramp_up_s=_range(_mapping(raw.get("ramp_up_s"), "sim.shot_fragments.ramp_up_s"), "sim.shot_fragments.ramp_up_s"),
        hold_s=_range(_mapping(raw.get("hold_s"), "sim.shot_fragments.hold_s"), "sim.shot_fragments.hold_s"),
        ramp_down_s=_range(_mapping(raw.get("ramp_down_s"), "sim.shot_fragments.ramp_down_s"), "sim.shot_fragments.ramp_down_s"),
        start_time_min_s=float(raw.get("start_time_min_s", 0.0)),
        start_time_max_s=None if raw.get("start_time_max_s") is None else float(raw.get("start_time_max_s")),
        trim_end_s=float(raw.get("trim_end_s", 0.02)),
        corner_smoothing_s=float(raw.get("corner_smoothing_s", 0.05)),
    )


def _sim(raw: Mapping[str, Any], base: Path) -> SimConfig:
    config_path = _resolve(base, raw["config_path"])
    initial_raw = raw.get("initial_currents_path")
    return SimConfig(
        config_path=config_path,
        initial_currents_path=None if initial_raw is None else _resolve(base, initial_raw),
        compute_backend=str(raw.get("compute_backend", "cpu")).lower(),
        gpu_device=str(raw.get("gpu_device", "cuda:0")),
        angles=int(raw.get("angles", 32)),
        max_episode_steps=int(raw.get("max_episode_steps", 1000)),
        initial_ranges=_ranges(_mapping(raw.get("initial_ranges", {}), "initial_ranges")),
        current_safety_limits=_current_safety_limits(_mapping(raw.get("current_safety_limits", {}), "current_safety_limits")),
        action_scale=float(raw.get("action_scale", 1.0)),
        shot_fragments=_shot_fragments(_mapping(raw.get("shot_fragments", {}), "shot_fragments"), base),
        terminate_on_boundary_loss=bool(raw.get("terminate_on_boundary_loss", True)),
        terminate_on_current_limit=bool(raw.get("terminate_on_current_limit", True)),
        current_termination_over_limit_a=float(raw.get("current_termination_over_limit_a", 0.0)),
        project_actions_to_current_limits=bool(raw.get("project_actions_to_current_limits", False)),
        current_projection_margin_fraction=float(raw.get("current_projection_margin_fraction", 0.0)),
        action_projection_termination_rms=float(raw.get("action_projection_termination_rms", 0.05)),
    )


def _reference(raw: Mapping[str, Any]) -> ReferenceConfig:
    ip_raw = _mapping(raw.get("ip"), "reference.ip")
    b_raw = _mapping(raw.get("boundary", {}), "reference.boundary")
    return ReferenceConfig(
        duration_s=float(raw.get("duration_s", 1.0)),
        t_step=float(raw.get("t_step", 0.001)),
        theta_count=int(raw.get("theta_count", 512)),
        seed=int(raw.get("seed", 1)),
        ip=IpReferenceConfig(
            min=float(ip_raw["min"]), max=float(ip_raw["max"]), rate_limit=float(ip_raw["rate_limit"]),
            segment_min_steps=int(ip_raw.get("segment_min_steps", 50)), segment_max_steps=int(ip_raw.get("segment_max_steps", 300)),
            segment_count_min=int(ip_raw.get("segment_count_min", 3)), segment_count_max=int(ip_raw.get("segment_count_max", 8)),
            hold_probability=float(ip_raw.get("hold_probability", 0.35)),
            kind=str(ip_raw.get("kind", "segmented")).lower(),
        ),
        boundary=BoundaryReferenceConfig(kind=str(b_raw.get("kind", "static_initial_parameters")), rate_limits={str(k): float(v) for k, v in _mapping(b_raw.get("rate_limits", {}), "rate_limits").items()}),
    )


def _observation(raw: Mapping[str, Any]) -> ObservationConfig:
    return ObservationConfig(
        target_preview_steps=int(raw.get("target_preview_steps", 8)),
        target_preview_stride=int(raw.get("target_preview_stride", 10)),
    )


def _reward(raw: Mapping[str, Any]) -> RewardConfig:
    defaults = RewardConfig()
    fields = set(RewardConfig.__dataclass_fields__)
    stale = sorted(set(raw) & _STALE_REWARD_KEYS)
    if stale:
        raise ValueError("reward contains stale keys from the removed reward: " + ", ".join(stale))
    unknown = sorted(set(raw) - fields)
    if unknown:
        raise ValueError("reward contains unsupported keys: " + ", ".join(unknown))
    return RewardConfig(**{field: type(getattr(defaults, field))(raw.get(field, getattr(defaults, field))) for field in RewardConfig.__dataclass_fields__})


def _randomization(raw: Mapping[str, Any]) -> RandomizationConfig:
    defaults = RandomizationConfig()
    return RandomizationConfig(**{field: type(getattr(defaults, field))(raw.get(field, getattr(defaults, field))) for field in RandomizationConfig.__dataclass_fields__})


def _network(raw: Mapping[str, Any]) -> NetworkConfig:
    defaults = NetworkConfig()
    return NetworkConfig(
        hidden_dim=int(raw.get("hidden_dim", defaults.hidden_dim)),
        critic_hidden_dim=int(raw.get("critic_hidden_dim", defaults.critic_hidden_dim)),
        critic_mlp_hidden_dim=int(raw.get("critic_mlp_hidden_dim", defaults.critic_mlp_hidden_dim)),
        actor_initial_std=float(raw.get("actor_initial_std", defaults.actor_initial_std)),
        actor_min_std=float(raw.get("actor_min_std", defaults.actor_min_std)),
    )


def _learner(raw: Mapping[str, Any]) -> LearnerConfig:
    defaults = LearnerConfig()
    values = {}
    for field in LearnerConfig.__dataclass_fields__:
        default = getattr(defaults, field)
        values[field] = type(default)(raw.get(field, default))
    return LearnerConfig(**values)


def _training(raw: Mapping[str, Any], base: Path) -> TrainingConfig:
    defaults = TrainingConfig()
    return TrainingConfig(
        steps=int(raw.get("steps", defaults.steps)), num_envs=int(raw.get("num_envs", defaults.num_envs)),
        device=str(raw.get("device", defaults.device)), seed=int(raw.get("seed", defaults.seed)),
        output_dir=_resolve(base, raw.get("output_dir", defaults.output_dir)),
        save_checkpoints=bool(raw.get("save_checkpoints", defaults.save_checkpoints)),
        checkpoint_interval_steps=int(raw.get("checkpoint_interval_steps", defaults.checkpoint_interval_steps)),
        eval_interval_steps=int(raw.get("eval_interval_steps", defaults.eval_interval_steps)),
        eval_episodes=int(raw.get("eval_episodes", defaults.eval_episodes)),
        eval_max_steps=int(raw.get("eval_max_steps", defaults.eval_max_steps)),
        actor_workers=int(raw.get("actor_workers", defaults.actor_workers)),
        actor_devices=_string_tuple(raw.get("actor_devices"), "training.actor_devices"),
    )


def _resolve(base: Path, value: object) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else (base / p).resolve()

def _validate_experiment_config(cfg: ExperimentConfig) -> None:
    if cfg.sim.compute_backend not in {"cpu", "gpu"}:
        raise ValueError("sim.compute_backend must be cpu or gpu")
    if int(cfg.sim.angles) <= 0:
        raise ValueError("sim.angles must be positive")
    if int(cfg.sim.max_episode_steps) <= 0:
        raise ValueError("sim.max_episode_steps must be positive")
    if cfg.sim.initial_ranges is not None:
        expected = {"R0", "Z0", "A0", "kappa", "delta"}
        missing = sorted(expected - set(cfg.sim.initial_ranges.boundary_parameters))
        if missing:
            raise ValueError("sim.initial_ranges.boundary_parameters missing: " + ", ".join(missing))
    if float(cfg.reference.duration_s) <= 0.0 or float(cfg.reference.t_step) <= 0.0:
        raise ValueError("reference duration_s and t_step must be positive")
    if int(cfg.reference.theta_count) <= 0:
        raise ValueError("reference.theta_count must be positive")
    ip = cfg.reference.ip
    if not (math.isfinite(ip.min) and math.isfinite(ip.max) and ip.max >= ip.min):
        raise ValueError("reference.ip min/max are invalid")
    if ip.kind not in {"segmented", "hold_reset", "shot_trapezoid_fragment"}:
        raise ValueError("reference.ip.kind is unsupported")
    if _range_crosses_or_touches_zero(float(ip.min), float(ip.max)):
        raise ValueError("reference.ip range must stay strictly on one side of zero")
    if cfg.sim.initial_ranges is not None:
        initial_ip = cfg.sim.initial_ranges.ip
        if _range_crosses_or_touches_zero(float(initial_ip.min), float(initial_ip.max)):
            raise ValueError("sim.initial_ranges.ip must stay strictly on one side of zero")
        if _sign(float(initial_ip.min)) != _sign(float(ip.min)):
            raise ValueError("sim.initial_ranges.ip must have the same sign as reference.ip")
    if ip.kind == "shot_trapezoid_fragment":
        if cfg.sim.shot_fragments is None:
            raise ValueError("reference.ip.kind=shot_trapezoid_fragment requires sim.shot_fragments")
    elif cfg.sim.shot_fragments is not None:
        raise ValueError("sim.shot_fragments requires reference.ip.kind=shot_trapezoid_fragment")
    if cfg.sim.shot_fragments is not None:
        shots = cfg.sim.shot_fragments
        if shots.kind != "idealized_t15_trapezoid":
            raise ValueError("sim.shot_fragments.kind is unsupported")
        if shots.ip_a is None:
            raise ValueError("sim.shot_fragments.ip_a is required")
        for name in ("start_time_min_s", "trim_end_s", "corner_smoothing_s"):
            value = getattr(shots, name)
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"sim.shot_fragments.{name} must be finite and non-negative")
        if shots.start_time_max_s is not None:
            if not math.isfinite(float(shots.start_time_max_s)) or float(shots.start_time_max_s) < float(shots.start_time_min_s):
                raise ValueError("sim.shot_fragments.start_time_max_s must be finite and >= start_time_min_s")
        if cfg.sim.initial_ranges is not None:
            if len(shots.pfc_currents) != len(cfg.sim.initial_ranges.pfc_currents):
                raise ValueError("sim.shot_fragments.pfc_currents length must match sim.initial_ranges.pfc_currents")
            if len(shots.sol_currents) != len(cfg.sim.initial_ranges.sol_currents):
                raise ValueError("sim.shot_fragments.sol_currents length must match sim.initial_ranges.sol_currents")
    if not math.isfinite(ip.rate_limit) or ip.rate_limit < 0.0:
        raise ValueError("reference.ip.rate_limit must be finite and non-negative")
    if ip.segment_min_steps <= 0 or ip.segment_max_steps < ip.segment_min_steps:
        raise ValueError("reference.ip segment step bounds are invalid")
    if ip.segment_count_min <= 0 or ip.segment_count_max < ip.segment_count_min:
        raise ValueError("reference.ip segment count bounds are invalid")
    if not 0.0 <= float(ip.hold_probability) <= 1.0:
        raise ValueError("reference.ip.hold_probability must be in [0, 1]")
    if cfg.reference.boundary.kind not in {"static_initial_parameters", "rate_limited_parameters", "hold_reset_boundary"}:
        raise ValueError("reference.boundary.kind is unsupported")
    if cfg.reference.boundary.kind == "hold_reset_boundary" and int(cfg.reference.theta_count) != int(cfg.sim.angles):
        raise ValueError("reference.theta_count must equal sim.angles for hold_reset_boundary")
    for key, value in cfg.reference.boundary.rate_limits.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"reference.boundary.rate_limits.{key} must be finite and non-negative")
    if cfg.observation.target_preview_steps < 0 or cfg.observation.target_preview_stride <= 0:
        raise ValueError("observation preview settings are invalid")
    if not math.isfinite(float(cfg.sim.action_scale)) or float(cfg.sim.action_scale) <= 0.0 or float(cfg.sim.action_scale) > 1.0:
        raise ValueError("sim.action_scale must be finite and in (0, 1]")
    if not math.isfinite(float(cfg.sim.current_termination_over_limit_a)) or float(cfg.sim.current_termination_over_limit_a) < 0.0:
        raise ValueError("sim.current_termination_over_limit_a must be finite and non-negative")
    if not math.isfinite(float(cfg.sim.current_projection_margin_fraction)) or not 0.0 <= float(cfg.sim.current_projection_margin_fraction) < 1.0:
        raise ValueError("sim.current_projection_margin_fraction must be finite and in [0, 1)")
    if not math.isfinite(float(cfg.sim.action_projection_termination_rms)) or not 0.0 < float(cfg.sim.action_projection_termination_rms) <= 1.0:
        raise ValueError("sim.action_projection_termination_rms must be finite and in (0, 1]")
    if cfg.sim.project_actions_to_current_limits and cfg.sim.current_safety_limits is None:
        raise ValueError("sim.project_actions_to_current_limits requires current_safety_limits")
    _validate_reward_config(cfg.reward, prefix="reward")
    if cfg.randomization.ip_measurement_noise_a < 0.0 or cfg.randomization.current_measurement_noise_a < 0.0:
        raise ValueError("randomization noise values must be non-negative")
    if cfg.randomization.action_offset_max < cfg.randomization.action_offset_min:
        raise ValueError("randomization.action_offset_max must be >= action_offset_min")
    if cfg.network.hidden_dim <= 0 or cfg.network.critic_hidden_dim <= 0 or cfg.network.critic_mlp_hidden_dim <= 0:
        raise ValueError("network dimensions must be positive")
    if not math.isfinite(float(cfg.network.actor_min_std)) or float(cfg.network.actor_min_std) <= 0.0:
        raise ValueError("network.actor_min_std must be finite and positive")
    if not math.isfinite(float(cfg.network.actor_initial_std)) or float(cfg.network.actor_initial_std) <= float(cfg.network.actor_min_std):
        raise ValueError("network.actor_initial_std must be finite and greater than actor_min_std")
    learner = cfg.learner
    for name in ("discount", "actor_lr", "critic_lr", "kl_lr", "temperature", "mpo_epsilon", "mean_kl_epsilon", "std_kl_epsilon", "target_update_tau"):
        value = float(getattr(learner, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"learner.{name} must be finite and positive")
    if learner.discount >= 1.0:
        raise ValueError("learner.discount must be < 1")
    for name in ("unroll_length", "batch_size", "replay_capacity_episodes", "action_samples", "actor_update_chunk_size", "rollout_chunk_length", "updates_per_rollout_chunk"):
        if int(getattr(learner, name)) <= 0:
            raise ValueError(f"learner.{name} must be positive")
    if int(learner.action_samples) <= 1:
        raise ValueError("learner.action_samples must be greater than 1")
    training = cfg.training
    for name in ("steps", "num_envs", "checkpoint_interval_steps", "eval_interval_steps", "eval_episodes", "eval_max_steps", "actor_workers"):
        if int(getattr(training, name)) <= 0:
            raise ValueError(f"training.{name} must be positive")


def _range_crosses_or_touches_zero(min_value: float, max_value: float) -> bool:
    return float(min_value) <= 0.0 <= float(max_value)


def _sign(value: float) -> int:
    return 1 if float(value) > 0.0 else -1


def _validate_reward_config(reward: RewardConfig, *, prefix: str) -> None:
    for name in ("shape_bad_m", "shape_max_bad_m", "ip_bad_a", "reward_scale"):
        value = float(getattr(reward, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{prefix}.{name} must be finite and positive")
    if not math.isfinite(float(reward.boundary_missing_error_m)) or float(reward.boundary_missing_error_m) < 0.0:
        raise ValueError(f"{prefix}.boundary_missing_error_m must be finite and non-negative")
    for name in ("shape_weight", "ip_weight"):
        value = float(getattr(reward, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{prefix}.{name} must be finite and positive")
    if not math.isfinite(float(reward.terminal_reward)):
        raise ValueError(f"{prefix}.terminal_reward must be finite")


_STALE_REWARD_KEYS = {
    "mode",
    "shape_good_m",
    "ip_good_a",
    "current_good_a",
    "current_bad_a",
    "derivative_good",
    "derivative_bad",
    "action_penalty_weight",
    "delta_action_penalty_weight",
    "current_weight",
    "derivative_weight",
    "action_saturation_weight",
    "delta_action_weight",
    "projection_weight",
    "current_margin_start_fraction",
    "derivative_penalty_start_fraction",
    "action_penalty_start_fraction",
    "delta_action_penalty_start",
    "delta_action_bad",
    "projection_bad",
    "late_error_weight",
    "late_error_power",
    "_".join(("tracking", "combiner")),
    "_".join(("shape", "aggregator")),
}
