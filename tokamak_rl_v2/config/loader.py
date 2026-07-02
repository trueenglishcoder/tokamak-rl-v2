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
    DeltaDerivativeLimits,
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
    if "curriculum" in raw:
        raise ValueError("curriculum is no longer supported")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, prefix="config")
    base = source.parent
    sim = _sim(_mapping(raw.get("sim"), "sim"), base)
    cfg = ExperimentConfig(
        name=str(raw.get("name", source.stem)),
        sim=sim,
        reference=_reference(_mapping(raw.get("reference"), "reference"), base),
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


def _reject_stale_keys(raw: Mapping[str, Any], stale_keys: set[str], *, prefix: str) -> None:
    stale = sorted(set(raw) & stale_keys)
    if stale:
        raise ValueError(f"{prefix} contains stale keys from the removed projection layer: " + ", ".join(stale))


def _reject_unknown_keys(raw: Mapping[str, Any], allowed_keys: set[str], *, prefix: str) -> None:
    unknown = sorted(set(raw) - set(allowed_keys))
    if unknown:
        raise ValueError(f"{prefix} contains unsupported keys: " + ", ".join(unknown))


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


def _delta_derivative_limits(raw: Mapping[str, Any] | None) -> DeltaDerivativeLimits | None:
    if not raw:
        return None
    allowed = {"pfc", "sol", "pfc_currents", "sol_currents", "pfc_delta_jdot", "sol_delta_jdot"}
    _reject_unknown_keys(raw, allowed, prefix="delta_derivative_limits_aps")
    pfc_raw = raw.get("pfc", raw.get("pfc_currents", raw.get("pfc_delta_jdot")))
    sol_raw = raw.get("sol", raw.get("sol_currents", raw.get("sol_delta_jdot")))
    return DeltaDerivativeLimits(
        pfc=_float_sequence(pfc_raw, "delta_derivative_limits_aps.pfc"),
        sol=_float_sequence(sol_raw, "delta_derivative_limits_aps.sol"),
    )


def _sim(raw: Mapping[str, Any], base: Path) -> SimConfig:
    _reject_stale_keys(raw, _STALE_SIM_KEYS, prefix="sim")
    if "initial_currents_path" in raw:
        raise ValueError("sim.initial_currents_path is no longer supported; resets must provide explicit ip0/pfc0/sol0")
    if "shot_fragments" in raw:
        raise ValueError("sim.shot_fragments is no longer supported")
    _reject_unknown_keys(raw, set(SimConfig.__dataclass_fields__), prefix="sim")
    defaults = SimConfig(
        config_path=Path("unused"),
    )
    config_path = _resolve(base, raw["config_path"])
    return SimConfig(
        config_path=config_path,
        compute_backend=str(raw.get("compute_backend", "cpu")).lower(),
        gpu_device=str(raw.get("gpu_device", "cuda:0")),
        angles=int(raw.get("angles", 32)),
        max_episode_steps=int(raw.get("max_episode_steps", 1000)),
        initial_ranges=_ranges(_mapping(raw.get("initial_ranges", {}), "initial_ranges")),
        reset_source=str(raw.get("reset_source", "initial_ranges")),
        csv_initial_state_library=None if raw.get("csv_initial_state_library") is None else _resolve(base, raw.get("csv_initial_state_library")),
        csv_initial_state_split=str(raw.get("csv_initial_state_split", defaults.csv_initial_state_split)),
        current_safety_limits=_current_safety_limits(_mapping(raw.get("current_safety_limits", {}), "current_safety_limits")),
        current_limit_scale=float(raw.get("current_limit_scale", defaults.current_limit_scale)),
        derivative_limit_scale=float(raw.get("derivative_limit_scale", defaults.derivative_limit_scale)),
        action_scale=float(raw.get("action_scale", 1.0)),
        action_contract=str(raw.get("action_contract", defaults.action_contract)),
        delta_derivative_scale_aps=float(raw.get("delta_derivative_scale_aps", defaults.delta_derivative_scale_aps)),
        delta_derivative_limits_aps=_delta_derivative_limits(_mapping(raw.get("delta_derivative_limits_aps", {}), "delta_derivative_limits_aps")),
        terminate_on_boundary_loss=bool(raw.get("terminate_on_boundary_loss", True)),
        terminate_on_current_limit=bool(raw.get("terminate_on_current_limit", True)),
        current_termination_over_limit_a=float(raw.get("current_termination_over_limit_a", defaults.current_termination_over_limit_a)),
        current_termination_grace_steps=int(raw.get("current_termination_grace_steps", defaults.current_termination_grace_steps)),
        current_hard_termination_fraction=float(raw.get("current_hard_termination_fraction", defaults.current_hard_termination_fraction)),
        current_saturation_fraction=float(raw.get("current_saturation_fraction", defaults.current_saturation_fraction)),
    )


def _reference(raw: Mapping[str, Any], base: Path) -> ReferenceConfig:
    _reject_unknown_keys(raw, set(ReferenceConfig.__dataclass_fields__), prefix="reference")
    ip_raw = _mapping(raw.get("ip"), "reference.ip")
    b_raw = _mapping(raw.get("boundary", {}), "reference.boundary")
    if "corner_smoothing_s" in ip_raw:
        raise ValueError("reference.ip.corner_smoothing_s is no longer supported; use reference.ip.smooth_ramps")
    _reject_unknown_keys(ip_raw, set(IpReferenceConfig.__dataclass_fields__), prefix="reference.ip")
    _reject_unknown_keys(b_raw, set(BoundaryReferenceConfig.__dataclass_fields__), prefix="reference.boundary")
    kind = str(ip_raw.get("kind", "segmented")).lower()
    if kind in {"segmented_profile", "single_segment_profile", "replay_window", "generated_segment_profile", "feasible_generated_window", "hold_boundary_eval_profile", "hold_boundary_eval_cut_profile"}:
        min_value = float(ip_raw.get("min", 1.0))
        max_value = float(ip_raw.get("max", min_value))
        rate_limit = float(ip_raw.get("rate_limit", 1.0))
    else:
        min_value = float(ip_raw["min"])
        max_value = float(ip_raw["max"])
        rate_limit = float(ip_raw["rate_limit"])
    replay_reference_raw = b_raw.get("replay_reference_dir")
    envelope_raw = b_raw.get("envelope_path")
    ip_feasible_raw = ip_raw.get("feasible_reference_dir")
    boundary_feasible_raw = b_raw.get("feasible_reference_dir", ip_feasible_raw)
    return ReferenceConfig(
        duration_s=float(raw.get("duration_s", 1.0)),
        t_step=float(raw.get("t_step", 0.001)),
        theta_count=int(raw.get("theta_count", 512)),
        seed=int(raw.get("seed", 1)),
        ip=IpReferenceConfig(
            min=min_value,
            max=max_value,
            rate_limit=rate_limit,
            segment_min_steps=int(ip_raw.get("segment_min_steps", 50)), segment_max_steps=int(ip_raw.get("segment_max_steps", 300)),
            segment_count_min=int(ip_raw.get("segment_count_min", 3)), segment_count_max=int(ip_raw.get("segment_count_max", 8)),
            hold_probability=float(ip_raw.get("hold_probability", 0.35)),
            kind=kind,
            limits_path=None if ip_raw.get("limits_path") is None else _resolve(base, ip_raw.get("limits_path")),
            feasible_reference_dir=None if ip_feasible_raw is None else _resolve(base, ip_feasible_raw),
            start_mode=str(ip_raw.get("start_mode", "reset_ip")),
            parent_steps=int(ip_raw.get("parent_steps", 0)),
            plateau_min_fraction=float(ip_raw.get("plateau_min_fraction", 0.25)),
            plateau_max_fraction=float(ip_raw.get("plateau_max_fraction", 1.0)),
            end_min_fraction=float(ip_raw.get("end_min_fraction", 0.25)),
            end_max_fraction=float(ip_raw.get("end_max_fraction", 1.0)),
            ramp_rate_reference=str(ip_raw.get("ramp_rate_reference", "p95")),
            ramp_up_rate_min_fraction=float(ip_raw.get("ramp_up_rate_min_fraction", 0.0)),
            ramp_up_rate_fraction=float(ip_raw.get("ramp_up_rate_fraction", 0.25)),
            ramp_down_rate_min_fraction=float(ip_raw.get("ramp_down_rate_min_fraction", 0.0)),
            ramp_down_rate_fraction=float(ip_raw.get("ramp_down_rate_fraction", 0.25)),
            hold_min_steps=int(ip_raw.get("hold_min_steps", 50)),
            hold_max_steps=int(ip_raw.get("hold_max_steps", 250)),
            final_hold_min_steps=int(ip_raw.get("final_hold_min_steps", 0)),
            smooth_ramps=bool(ip_raw.get("smooth_ramps", True)),
            max_delta_fraction=float(ip_raw.get("max_delta_fraction", 1.0)),
        ),
        boundary=BoundaryReferenceConfig(
            kind=str(b_raw.get("kind", "static_initial_parameters")),
            rate_limits={str(k): float(v) for k, v in _mapping(b_raw.get("rate_limits", {}), "rate_limits").items()},
            replay_reference_dir=None if replay_reference_raw is None else _resolve(base, replay_reference_raw),
            envelope_path=None if envelope_raw is None else _resolve(base, envelope_raw),
            feasible_reference_dir=None if boundary_feasible_raw is None else _resolve(base, boundary_feasible_raw),
            segment_min_steps=int(b_raw.get("segment_min_steps", 30)),
        ),
    )


def _observation(raw: Mapping[str, Any]) -> ObservationConfig:
    _reject_unknown_keys(raw, set(ObservationConfig.__dataclass_fields__), prefix="observation")
    return ObservationConfig(
        actor_kind=str(raw.get("actor_kind", "controller_state_v6")),
        critic_kind=str(raw.get("critic_kind", "compact_training_state_v2")),
        target_preview_steps=int(raw.get("target_preview_steps", 8)),
        target_preview_stride=int(raw.get("target_preview_stride", 10)),
        ip_rate_scale_aps=float(raw.get("ip_rate_scale_aps", 500000.0)),
        boundary_rate_scale_mps=float(raw.get("boundary_rate_scale_mps", 1.0)),
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
    _reject_unknown_keys(raw, set(RandomizationConfig.__dataclass_fields__), prefix="randomization")
    defaults = RandomizationConfig()
    return RandomizationConfig(**{field: type(getattr(defaults, field))(raw.get(field, getattr(defaults, field))) for field in RandomizationConfig.__dataclass_fields__})


def _network(raw: Mapping[str, Any]) -> NetworkConfig:
    _reject_unknown_keys(raw, set(NetworkConfig.__dataclass_fields__), prefix="network")
    defaults = NetworkConfig()
    return NetworkConfig(
        hidden_dim=int(raw.get("hidden_dim", defaults.hidden_dim)),
        critic_hidden_dim=int(raw.get("critic_hidden_dim", defaults.critic_hidden_dim)),
        critic_mlp_hidden_dim=int(raw.get("critic_mlp_hidden_dim", defaults.critic_mlp_hidden_dim)),
        actor_initial_std=float(raw.get("actor_initial_std", defaults.actor_initial_std)),
        actor_min_std=float(raw.get("actor_min_std", defaults.actor_min_std)),
    )


def _learner(raw: Mapping[str, Any]) -> LearnerConfig:
    _reject_unknown_keys(raw, set(LearnerConfig.__dataclass_fields__), prefix="learner")
    defaults = LearnerConfig()
    values = {}
    for field in LearnerConfig.__dataclass_fields__:
        default = getattr(defaults, field)
        values[field] = type(default)(raw.get(field, default))
    return LearnerConfig(**values)


def _training(raw: Mapping[str, Any], base: Path) -> TrainingConfig:
    _reject_unknown_keys(raw, set(TrainingConfig.__dataclass_fields__), prefix="training")
    defaults = TrainingConfig()
    return TrainingConfig(
        steps=int(raw.get("steps", defaults.steps)), num_envs=int(raw.get("num_envs", defaults.num_envs)),
        device=str(raw.get("device", defaults.device)), seed=int(raw.get("seed", defaults.seed)),
        output_dir=_resolve(base, raw.get("output_dir", defaults.output_dir)),
        save_checkpoints=bool(raw.get("save_checkpoints", defaults.save_checkpoints)),
        checkpoint_interval_steps=int(raw.get("checkpoint_interval_steps", defaults.checkpoint_interval_steps)),
        eval_checkpoint_top_k=int(raw.get("eval_checkpoint_top_k", defaults.eval_checkpoint_top_k)),
        milestone_checkpoint_interval_steps=int(raw.get("milestone_checkpoint_interval_steps", defaults.milestone_checkpoint_interval_steps)),
        keep_latest_checkpoint=bool(raw.get("keep_latest_checkpoint", defaults.keep_latest_checkpoint)),
        eval_interval_steps=int(raw.get("eval_interval_steps", defaults.eval_interval_steps)),
        eval_episodes=int(raw.get("eval_episodes", defaults.eval_episodes)),
        eval_max_steps=int(raw.get("eval_max_steps", defaults.eval_max_steps)),
        actor_workers=int(raw.get("actor_workers", defaults.actor_workers)),
        actor_devices=_string_tuple(raw.get("actor_devices"), "training.actor_devices"),
        distributed_mode=str(raw.get("distributed_mode", defaults.distributed_mode)),
        production_mode=bool(raw.get("production_mode", defaults.production_mode)),
        early_stop_patience_evals=int(raw.get("early_stop_patience_evals", defaults.early_stop_patience_evals)),
        early_stop_min_delta=float(raw.get("early_stop_min_delta", defaults.early_stop_min_delta)),
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
    if cfg.sim.reset_source not in {"initial_ranges", "csv_initial_states"}:
        raise ValueError("sim.reset_source must be initial_ranges or csv_initial_states")
    if cfg.sim.reset_source == "initial_ranges" and cfg.sim.initial_ranges is None:
        raise ValueError("sim.reset_source=initial_ranges requires sim.initial_ranges")
    if cfg.sim.reset_source == "csv_initial_states" and cfg.sim.csv_initial_state_library is None:
        raise ValueError("sim.reset_source=csv_initial_states requires sim.csv_initial_state_library")
    if cfg.sim.csv_initial_state_split not in {"train", "holdout", "all"}:
        raise ValueError("sim.csv_initial_state_split must be train, holdout, or all")
    csv_boundary_kinds = {"hold_reset_boundary", "t15_replay_segment_conditioned", "generated_parameter_profile", "feasible_generated_window"}
    if cfg.sim.reset_source == "csv_initial_states" and cfg.reference.boundary.kind not in csv_boundary_kinds:
        raise ValueError(
            "sim.reset_source=csv_initial_states requires reference.boundary.kind="
            "hold_reset_boundary, t15_replay_segment_conditioned, generated_parameter_profile, or feasible_generated_window"
        )
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
    profile_kinds = {"segmented_profile", "single_segment_profile", "generated_segment_profile", "hold_boundary_eval_profile", "hold_boundary_eval_cut_profile"}
    if ip.kind not in {"segmented", "hold_reset", "segmented_profile", "single_segment_profile", "replay_window", "generated_segment_profile", "feasible_generated_window", "hold_boundary_eval_profile", "hold_boundary_eval_cut_profile"}:
        raise ValueError("reference.ip.kind is unsupported")
    if ip.kind not in profile_kinds | {"replay_window", "feasible_generated_window"}:
        if not (math.isfinite(ip.min) and math.isfinite(ip.max) and ip.max >= ip.min):
            raise ValueError("reference.ip min/max are invalid")
        if _range_crosses_or_touches_zero(float(ip.min), float(ip.max)):
            raise ValueError("reference.ip range must stay strictly on one side of zero")
        if cfg.sim.initial_ranges is not None:
            initial_ip = cfg.sim.initial_ranges.ip
            if _range_crosses_or_touches_zero(float(initial_ip.min), float(initial_ip.max)):
                raise ValueError("sim.initial_ranges.ip must stay strictly on one side of zero")
            if _sign(float(initial_ip.min)) != _sign(float(ip.min)):
                raise ValueError("sim.initial_ranges.ip must have the same sign as reference.ip")
    if ip.kind in profile_kinds:
        if ip.limits_path is None:
            raise ValueError(f"reference.ip.kind={ip.kind} requires reference.ip.limits_path")
        if ip.start_mode != "reset_ip":
            raise ValueError(f"reference.ip.{ip.kind} only supports start_mode=reset_ip")
        if ip.kind == "segmented_profile":
            if ip.segment_min_steps <= 0 or ip.segment_max_steps < ip.segment_min_steps:
                raise ValueError("reference.ip segmented_profile step bounds are invalid")
            if ip.segment_count_min < 2 or ip.segment_count_max < ip.segment_count_min:
                raise ValueError("reference.ip segmented_profile count bounds are invalid")
        if ip.kind == "generated_segment_profile":
            if int(ip.segment_min_steps) <= 0:
                raise ValueError("reference.ip generated_segment_profile requires positive segment_min_steps")
            if int(cfg.sim.max_episode_steps) < 2 * int(ip.segment_min_steps):
                raise ValueError(
                    "reference.ip generated_segment_profile requires max_episode_steps >= "
                    "2 * segment_min_steps for mixed modes"
                )
            if not bool(ip.smooth_ramps):
                raise ValueError("reference.ip generated_segment_profile requires smooth_ramps=true")
        if ip.kind in {"hold_boundary_eval_profile", "hold_boundary_eval_cut_profile"}:
            if ip.segment_min_steps <= 0 or ip.segment_max_steps < ip.segment_min_steps:
                raise ValueError(f"reference.ip {ip.kind} step bounds are invalid")
            if ip.segment_count_min < 1 or ip.segment_count_max < ip.segment_count_min:
                raise ValueError(f"reference.ip {ip.kind} count bounds are invalid")
            if not 0.0 <= float(ip.hold_probability) <= 1.0:
                raise ValueError("reference.ip.hold_probability must be in [0, 1]")
        if ip.kind == "hold_boundary_eval_cut_profile":
            if int(ip.parent_steps) <= 0:
                raise ValueError("reference.ip.hold_boundary_eval_cut_profile requires parent_steps > 0")
            if int(ip.parent_steps) < int(cfg.sim.max_episode_steps):
                raise ValueError("reference.ip.parent_steps must be >= sim.max_episode_steps")
        if ip.ramp_rate_reference not in {"p95", "robust_mean"}:
            raise ValueError("reference.ip.ramp_rate_reference must be 'p95' or 'robust_mean'")
        for name in (
            "plateau_min_fraction",
            "plateau_max_fraction",
            "end_min_fraction",
            "end_max_fraction",
            "ramp_up_rate_min_fraction",
            "ramp_up_rate_fraction",
            "ramp_down_rate_min_fraction",
            "ramp_down_rate_fraction",
            "max_delta_fraction",
        ):
            value = float(getattr(ip, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"reference.ip.{name} must be finite and non-negative")
            if name in {"ramp_up_rate_fraction", "ramp_down_rate_fraction", "max_delta_fraction"} and value <= 0.0:
                raise ValueError(f"reference.ip.{name} must be positive")
        if float(ip.ramp_up_rate_min_fraction) > float(ip.ramp_up_rate_fraction):
            raise ValueError("reference.ip ramp-up rate fractions are invalid")
        if float(ip.ramp_down_rate_min_fraction) > float(ip.ramp_down_rate_fraction):
            raise ValueError("reference.ip ramp-down rate fractions are invalid")
        if float(ip.plateau_min_fraction) > float(ip.plateau_max_fraction):
            raise ValueError("reference.ip plateau fractions are invalid")
        if float(ip.end_min_fraction) > float(ip.end_max_fraction):
            raise ValueError("reference.ip end fractions are invalid")
        for name in ("hold_min_steps", "hold_max_steps", "final_hold_min_steps"):
            if int(getattr(ip, name)) < 0:
                raise ValueError(f"reference.ip.{name} must be non-negative")
        if int(ip.hold_max_steps) < int(ip.hold_min_steps):
            raise ValueError("reference.ip hold step bounds are invalid")
    if ip.kind == "feasible_generated_window":
        if ip.feasible_reference_dir is None:
            raise ValueError("reference.ip.kind=feasible_generated_window requires feasible_reference_dir")
        if not ip.feasible_reference_dir.exists():
            raise ValueError(f"reference.ip.feasible_reference_dir does not exist: {ip.feasible_reference_dir}")
        if cfg.sim.reset_source != "csv_initial_states":
            raise ValueError("reference.ip.kind=feasible_generated_window requires csv initial states")
    if ip.kind not in profile_kinds | {"replay_window", "feasible_generated_window"}:
        if not math.isfinite(ip.rate_limit) or ip.rate_limit < 0.0:
            raise ValueError("reference.ip.rate_limit must be finite and non-negative")
        if ip.segment_min_steps <= 0 or ip.segment_max_steps < ip.segment_min_steps:
            raise ValueError("reference.ip segment step bounds are invalid")
        if ip.segment_count_min <= 0 or ip.segment_count_max < ip.segment_count_min:
            raise ValueError("reference.ip segment count bounds are invalid")
        if not 0.0 <= float(ip.hold_probability) <= 1.0:
            raise ValueError("reference.ip.hold_probability must be in [0, 1]")
    if cfg.reference.boundary.kind not in {
        "static_initial_parameters",
        "rate_limited_parameters",
        "hold_reset_boundary",
        "t15_replay_segment_conditioned",
        "generated_parameter_profile",
        "feasible_generated_window",
    }:
        raise ValueError("reference.boundary.kind is unsupported")
    if cfg.reference.boundary.kind in {"hold_reset_boundary", "t15_replay_segment_conditioned", "generated_parameter_profile", "feasible_generated_window"} and int(cfg.reference.theta_count) != int(cfg.sim.angles):
        raise ValueError("reference.theta_count must equal sim.angles for sampled boundary references")
    if cfg.reference.boundary.kind == "feasible_generated_window":
        if cfg.reference.boundary.feasible_reference_dir is None:
            raise ValueError("reference.boundary.kind=feasible_generated_window requires feasible_reference_dir")
        if not cfg.reference.boundary.feasible_reference_dir.exists():
            raise ValueError(f"reference.boundary.feasible_reference_dir does not exist: {cfg.reference.boundary.feasible_reference_dir}")
        if cfg.reference.ip.kind != "feasible_generated_window":
            raise ValueError("reference.boundary.kind=feasible_generated_window requires reference.ip.kind=feasible_generated_window")
        if cfg.reference.ip.feasible_reference_dir != cfg.reference.boundary.feasible_reference_dir:
            raise ValueError("feasible_generated_window Ip and boundary must use the same feasible_reference_dir")
        if cfg.sim.reset_source != "csv_initial_states":
            raise ValueError("reference.boundary.kind=feasible_generated_window requires csv initial states with params0")
    if cfg.reference.boundary.kind == "generated_parameter_profile":
        if cfg.reference.boundary.envelope_path is None:
            raise ValueError("reference.boundary.kind=generated_parameter_profile requires envelope_path")
        if not cfg.reference.boundary.envelope_path.exists():
            raise ValueError(f"reference.boundary.envelope_path does not exist: {cfg.reference.boundary.envelope_path}")
        if cfg.sim.reset_source != "csv_initial_states":
            raise ValueError("reference.boundary.kind=generated_parameter_profile requires csv initial states with params0")
        if int(cfg.reference.boundary.segment_min_steps) <= 0:
            raise ValueError("reference.boundary generated_parameter_profile requires positive segment_min_steps")
        if int(cfg.sim.max_episode_steps) < 2 * int(cfg.reference.boundary.segment_min_steps):
            raise ValueError(
                "reference.boundary generated_parameter_profile requires max_episode_steps >= "
                "2 * segment_min_steps for mixed modes"
            )
    if cfg.reference.boundary.kind == "t15_replay_segment_conditioned":
        replay_dir = cfg.reference.boundary.replay_reference_dir
        if replay_dir is None:
            raise ValueError("reference.boundary.kind=t15_replay_segment_conditioned requires replay_reference_dir")
        if not replay_dir.exists():
            raise ValueError(f"reference.boundary.replay_reference_dir does not exist: {replay_dir}")
        if cfg.sim.reset_source != "csv_initial_states":
            raise ValueError("reference.boundary.kind=t15_replay_segment_conditioned requires csv initial states")
    if cfg.observation.actor_kind not in {"controller_state_v4", "controller_state_v5", "controller_state_v6"}:
        raise ValueError("observation.actor_kind must be controller_state_v4, controller_state_v5, or controller_state_v6")
    if cfg.observation.critic_kind not in {"privileged_training_state_v1", "compact_training_state_v2"}:
        raise ValueError("observation.critic_kind must be privileged_training_state_v1 or compact_training_state_v2")
    for name in ("ip_rate_scale_aps", "boundary_rate_scale_mps"):
        value = float(getattr(cfg.observation, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"observation.{name} must be finite and positive")
    for key, value in cfg.reference.boundary.rate_limits.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"reference.boundary.rate_limits.{key} must be finite and non-negative")
    if cfg.observation.target_preview_steps < 0 or cfg.observation.target_preview_stride <= 0:
        raise ValueError("observation preview settings are invalid")
    if not math.isfinite(float(cfg.sim.action_scale)) or float(cfg.sim.action_scale) <= 0.0 or float(cfg.sim.action_scale) > 1.0:
        raise ValueError("sim.action_scale must be finite and in (0, 1]")
    if cfg.sim.action_contract != "jdot_command":
        raise ValueError("sim.action_contract must be jdot_command")
    if cfg.sim.delta_derivative_limits_aps is not None:
        raise ValueError("learned-policy sim.action_contract=jdot_command must not set delta_derivative_limits_aps")
    for name in ("current_limit_scale", "derivative_limit_scale"):
        value = float(getattr(cfg.sim, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"sim.{name} must be finite and positive")
    if not math.isfinite(float(cfg.sim.current_termination_over_limit_a)) or float(cfg.sim.current_termination_over_limit_a) < 0.0:
        raise ValueError("sim.current_termination_over_limit_a must be finite and non-negative")
    if int(cfg.sim.current_termination_grace_steps) <= 0:
        raise ValueError("sim.current_termination_grace_steps must be positive")
    if not math.isfinite(float(cfg.sim.current_hard_termination_fraction)) or float(cfg.sim.current_hard_termination_fraction) <= 1.0:
        raise ValueError("sim.current_hard_termination_fraction must be finite and > 1")
    if not math.isfinite(float(cfg.sim.current_saturation_fraction)) or float(cfg.sim.current_saturation_fraction) < 1.0:
        raise ValueError("sim.current_saturation_fraction must be finite and >= 1")
    if cfg.reward.kind == "tcv_derivative":
        if cfg.sim.action_contract != "jdot_command":
            raise ValueError("reward.kind=tcv_derivative requires sim.action_contract=jdot_command")
        if cfg.training.production_mode and not cfg.sim.terminate_on_boundary_loss:
            raise ValueError("production reward.kind=tcv_derivative requires sim.terminate_on_boundary_loss=true")
        if cfg.training.production_mode and not cfg.sim.terminate_on_current_limit:
            raise ValueError("production reward.kind=tcv_derivative requires sim.terminate_on_current_limit=true")
        if not math.isclose(float(cfg.sim.current_saturation_fraction), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("reward.kind=tcv_derivative requires sim.current_saturation_fraction=1.0")
    _validate_reward_config(cfg.reward, prefix="reward")
    if cfg.randomization.ip_measurement_noise_a < 0.0 or cfg.randomization.current_measurement_noise_a < 0.0:
        raise ValueError("randomization noise values must be non-negative")
    if cfg.randomization.action_offset_max < cfg.randomization.action_offset_min:
        raise ValueError("randomization.action_offset_max must be >= action_offset_min")
    if cfg.sim.action_contract == "jdot_command" and (
        abs(float(cfg.randomization.action_offset_min)) > 1.0e-12
        or abs(float(cfg.randomization.action_offset_max)) > 1.0e-12
    ):
        raise ValueError("sim.action_contract=jdot_command requires zero randomization action offsets")
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
    for name in ("unroll_length", "batch_size", "replay_capacity_episodes", "action_samples", "min_replay_sequence_length", "actor_update_chunk_size", "rollout_chunk_length", "updates_per_rollout_chunk"):
        if int(getattr(learner, name)) <= 0:
            raise ValueError(f"learner.{name} must be positive")
    if int(learner.min_replay_sequence_length) > int(learner.unroll_length):
        raise ValueError("learner.min_replay_sequence_length must be <= learner.unroll_length")
    if int(learner.action_samples) <= 1:
        raise ValueError("learner.action_samples must be greater than 1")
    training = cfg.training
    for name in ("steps", "num_envs", "checkpoint_interval_steps", "eval_interval_steps", "eval_episodes", "eval_max_steps", "actor_workers"):
        if int(getattr(training, name)) <= 0:
            raise ValueError(f"training.{name} must be positive")
    for name in ("eval_checkpoint_top_k", "milestone_checkpoint_interval_steps", "early_stop_patience_evals"):
        if int(getattr(training, name)) < 0:
            raise ValueError(f"training.{name} must be non-negative")
    if not math.isfinite(float(training.early_stop_min_delta)) or float(training.early_stop_min_delta) < 0.0:
        raise ValueError("training.early_stop_min_delta must be finite and non-negative")
    if training.distributed_mode not in {"single", "local_replay"}:
        raise ValueError("training.distributed_mode must be single or local_replay")
    if training.distributed_mode == "local_replay" and int(training.actor_workers) != 1:
        raise ValueError("training.distributed_mode=local_replay does not use actor_workers; set actor_workers=1")
    if training.production_mode:
        if cfg.sim.reset_source != "csv_initial_states":
            raise ValueError("training.production_mode requires sim.reset_source=csv_initial_states")
        if cfg.sim.csv_initial_state_split != "train":
            raise ValueError("training.production_mode requires sim.csv_initial_state_split=train")
        config_name = cfg.sim.config_path.name
        is_t15_new_data = config_name == "T15MD_new_data.toml" or config_name.startswith("T15MD_new_data_")
        if not is_t15_new_data and config_name != "T15MD_4pfc.toml":
            raise ValueError("training.production_mode requires a current T15 tokamak-sim config")
        if cfg.sim.action_contract != "jdot_command" or cfg.sim.delta_derivative_limits_aps is not None:
            raise ValueError("training.production_mode requires jdot_command action contract without delta-Jdot limits")
        if not cfg.sim.terminate_on_boundary_loss or not cfg.sim.terminate_on_current_limit:
            raise ValueError("training.production_mode requires boundary and current terminations")
        if is_t15_new_data and cfg.reference.ip.kind not in {"segmented_profile", "single_segment_profile", "replay_window", "generated_segment_profile", "feasible_generated_window"}:
            raise ValueError("T15MD_new_data production requires reference.ip.kind=segmented_profile, single_segment_profile, replay_window, generated_segment_profile, or feasible_generated_window")
        if config_name == "T15MD_4pfc.toml" and cfg.reference.ip.kind != "replay_window":
            raise ValueError("T15MD_4pfc production requires reference.ip.kind=replay_window")
        if cfg.reference.ip.kind == "single_segment_profile":
            if cfg.reference.boundary.kind != "hold_reset_boundary":
                raise ValueError("training.production_mode single_segment_profile requires reference.boundary.kind=hold_reset_boundary")
        elif cfg.reference.ip.kind == "generated_segment_profile":
            if cfg.reference.boundary.kind != "generated_parameter_profile":
                raise ValueError("training.production_mode generated_segment_profile requires reference.boundary.kind=generated_parameter_profile")
        elif cfg.reference.ip.kind == "feasible_generated_window":
            if cfg.reference.boundary.kind != "feasible_generated_window":
                raise ValueError("training.production_mode feasible_generated_window requires reference.boundary.kind=feasible_generated_window")
        elif cfg.reference.boundary.kind != "t15_replay_segment_conditioned":
            raise ValueError("training.production_mode requires reference.boundary.kind=t15_replay_segment_conditioned")
        if cfg.reference.ip.kind == "replay_window" and cfg.observation.actor_kind != "controller_state_v6":
            raise ValueError("training.production_mode replay_window requires observation.actor_kind=controller_state_v6")
        if cfg.reference.ip.kind == "replay_window" and cfg.observation.critic_kind != "compact_training_state_v2":
            raise ValueError("training.production_mode replay_window requires observation.critic_kind=compact_training_state_v2")
        if cfg.reward.kind != "tcv_derivative":
            raise ValueError("training.production_mode requires reward.kind=tcv_derivative")
        expected_duration = float(cfg.sim.max_episode_steps) * float(cfg.reference.t_step)
        if not math.isclose(float(cfg.reference.duration_s), expected_duration, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("training.production_mode requires reference.duration_s == sim.max_episode_steps * reference.t_step")


def _range_crosses_or_touches_zero(min_value: float, max_value: float) -> bool:
    return float(min_value) <= 0.0 <= float(max_value)


def _sign(value: float) -> int:
    return 1 if float(value) > 0.0 else -1


def _validate_reward_config(reward: RewardConfig, *, prefix: str) -> None:
    if reward.kind not in {"physical_cost", "tcv_quality", "tcv_quality_legacy", "tcv_derivative"}:
        raise ValueError(f"{prefix}.kind must be physical_cost, tcv_quality_legacy, tcv_quality, or tcv_derivative")
    for name in ("shape_mean_scale_m", "shape_max_scale_m", "ip_scale_a", "reward_scale"):
        value = float(getattr(reward, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{prefix}.{name} must be finite and positive")
    if not math.isfinite(float(reward.smoothmax_alpha)) or float(reward.smoothmax_alpha) == 0.0:
        raise ValueError(f"{prefix}.smoothmax_alpha must be finite and non-zero")
    if reward.kind in {"tcv_quality", "tcv_quality_legacy"} and float(reward.smoothmax_alpha) <= 0.0:
        raise ValueError(f"{prefix}.smoothmax_alpha must be positive for legacy tcv_quality")
    if reward.kind == "tcv_derivative" and float(reward.smoothmax_alpha) >= 0.0:
        raise ValueError(f"{prefix}.smoothmax_alpha must be negative for tcv_derivative worst-component aggregation")
    if not math.isfinite(float(reward.boundary_missing_error_m)) or float(reward.boundary_missing_error_m) < 0.0:
        raise ValueError(f"{prefix}.boundary_missing_error_m must be finite and non-negative")
    for name in (
        "boundary_missing_weight",
        "shape_mean_weight",
        "shape_max_weight",
        "ip_weight",
        "current_weight",
        "derivative_weight",
        "current_drift_weight",
        "mean_jdot_bias_weight",
        "current_usage_weight",
        "derivative_usage_weight",
        "action_weight",
        "delta_action_weight",
        "jdot_switching_weight",
        "actuator_saturation_weight",
    ):
        value = float(getattr(reward, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{prefix}.{name} must be finite and non-negative")
    for name in ("jdot_switching_scale", "jdot_switching_cap"):
        value = float(getattr(reward, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{prefix}.{name} must be finite and positive")
    for name in ("current_soft_fraction", "derivative_soft_fraction"):
        value = float(getattr(reward, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{prefix}.{name} must be finite and in [0, 1]")
    for soft_name, bad_name in (
        ("current_soft_fraction", "current_bad_fraction"),
        ("derivative_soft_fraction", "derivative_bad_fraction"),
    ):
        soft = float(getattr(reward, soft_name))
        bad = float(getattr(reward, bad_name))
        if not math.isfinite(bad) or bad <= soft:
            raise ValueError(f"{prefix}.{bad_name} must be finite and greater than {prefix}.{soft_name}")
    for name in ("current_drift_bad_fraction", "mean_jdot_bias_bad_fraction"):
        value = float(getattr(reward, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{prefix}.{name} must be finite and positive")
    if not math.isfinite(float(reward.terminal_reward)):
        raise ValueError(f"{prefix}.terminal_reward must be finite")
    if not math.isfinite(float(reward.terminal_remaining_cost)) or float(reward.terminal_remaining_cost) < 0.0:
        raise ValueError(f"{prefix}.terminal_remaining_cost must be finite and non-negative")
    if reward.kind == "tcv_derivative":
        for name in ("current_usage_weight", "derivative_usage_weight", "action_weight", "delta_action_weight"):
            if abs(float(getattr(reward, name))) > 1.0e-12:
                raise ValueError(f"{prefix}.{name} must be 0 for reward.kind=tcv_derivative")


_STALE_REWARD_KEYS = {
    "mode",
    "shape_good_m",
    "shape_bad_m",
    "shape_max_bad_m",
    "ip_good_a",
    "ip_bad_a",
    "shape_weight",
    "current_good_fraction",
    "max_episode_reward",
    "current_good_a",
    "current_bad_a",
    "derivative_good",
    "derivative_bad",
    "action_penalty_weight",
    "delta_action_penalty_weight",
    "action_saturation_weight",
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


_STALE_SIM_KEYS = {
    "project_actions_to_current_limits",
    "current_projection_margin_fraction",
    "action_projection_termination_rms",
    "terminate_on_action_projection",
}


_TOP_LEVEL_KEYS = set(ExperimentConfig.__dataclass_fields__)
