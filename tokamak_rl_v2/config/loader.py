from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

try:
    import yaml
except Exception:  # pragma: no cover - optional local fallback
    yaml = None

from tokamak_rl_v2.config.schema import (
    BoundaryReferenceConfig,
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
    return ExperimentConfig(
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
        diagnostic_flux_count=int(raw.get("diagnostic_flux_count", 38)),
        diagnostic_field_count=int(raw.get("diagnostic_field_count", 38)),
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
    )


def _resolve(base: Path, value: object) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else (base / p).resolve()
