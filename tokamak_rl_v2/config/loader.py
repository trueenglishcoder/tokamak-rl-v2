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
    SimConfig,
    TrainingConfig,
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
    return RewardConfig(**{field: type(getattr(defaults, field))(raw.get(field, getattr(defaults, field))) for field in RewardConfig.__dataclass_fields__})


def _randomization(raw: Mapping[str, Any]) -> RandomizationConfig:
    defaults = RandomizationConfig()
    return RandomizationConfig(**{field: type(getattr(defaults, field))(raw.get(field, getattr(defaults, field))) for field in RandomizationConfig.__dataclass_fields__})


def _network(raw: Mapping[str, Any]) -> NetworkConfig:
    return NetworkConfig(hidden_dim=int(raw.get("hidden_dim", 256)), critic_hidden_dim=int(raw.get("critic_hidden_dim", 256)), critic_mlp_hidden_dim=int(raw.get("critic_mlp_hidden_dim", 256)))


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
    _validate_reward_config(cfg.reward, prefix="reward")
    if cfg.randomization.ip_measurement_noise_a < 0.0 or cfg.randomization.current_measurement_noise_a < 0.0:
        raise ValueError("randomization noise values must be non-negative")
    if cfg.randomization.action_offset_max < cfg.randomization.action_offset_min:
        raise ValueError("randomization.action_offset_max must be >= action_offset_min")
    if cfg.network.hidden_dim <= 0 or cfg.network.critic_hidden_dim <= 0 or cfg.network.critic_mlp_hidden_dim <= 0:
        raise ValueError("network dimensions must be positive")
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


def _validate_reward_config(reward: RewardConfig, *, prefix: str) -> None:
    if reward.mode not in {"quality", "dense_physical"}:
        raise ValueError(f"{prefix}.mode is unsupported")
    if reward.shape_bad_m <= reward.shape_good_m:
        raise ValueError(f"{prefix}.shape_bad_m must be greater than shape_good_m")
    if reward.ip_bad_a <= reward.ip_good_a:
        raise ValueError(f"{prefix}.ip_bad_a must be greater than ip_good_a")
    if reward.current_bad_a <= reward.current_good_a:
        raise ValueError(f"{prefix}.current_bad_a must be greater than current_good_a")
    if reward.derivative_bad <= reward.derivative_good:
        raise ValueError(f"{prefix}.derivative_bad must be greater than derivative_good")
    for name in ("shape_weight", "ip_weight", "reward_scale"):
        if float(getattr(reward, name)) <= 0.0:
            raise ValueError(f"{prefix}.{name} must be positive")
    for name in ("action_penalty_weight", "delta_action_penalty_weight"):
        if float(getattr(reward, name)) < 0.0:
            raise ValueError(f"{prefix}.{name} must be non-negative")
    if reward.tracking_combiner not in {"smooth_min", "weighted_mean", "geometric_mean", "product"}:
        raise ValueError(f"{prefix}.tracking_combiner is unsupported")
    if reward.shape_aggregator not in {"smooth_worst", "mean", "geometric_mean"}:
        raise ValueError(f"{prefix}.shape_aggregator is unsupported")
