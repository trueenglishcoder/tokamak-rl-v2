from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class Range:
    min: float
    max: float

    def validate(self, name: str) -> None:
        if self.max < self.min:
            raise ValueError(f"{name}.max must be >= {name}.min")


@dataclass(frozen=True, slots=True)
class InitialRanges:
    ip: Range
    pfc_currents: tuple[Range, ...]
    sol_currents: tuple[Range, ...]
    boundary_parameters: dict[str, Range]


@dataclass(frozen=True, slots=True)
class CurrentSafetyLimits:
    pfc_currents: tuple[float, ...]
    sol_currents: tuple[float, ...]

    def validate(self, *, n_pfc: int | None = None, n_sol: int | None = None) -> None:
        if n_pfc is not None and len(self.pfc_currents) != int(n_pfc):
            raise ValueError(f"current_safety_limits.pfc_currents must contain {int(n_pfc)} values")
        if n_sol is not None and len(self.sol_currents) != int(n_sol):
            raise ValueError(f"current_safety_limits.sol_currents must contain {int(n_sol)} values")
        for name, values in (("pfc_currents", self.pfc_currents), ("sol_currents", self.sol_currents)):
            for idx, value in enumerate(values):
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"current_safety_limits.{name}[{idx}] must be finite and positive")


@dataclass(frozen=True, slots=True)
class DeltaDerivativeLimits:
    pfc: tuple[float, ...]
    sol: tuple[float, ...]

    def validate(self, *, n_pfc: int | None = None, n_sol: int | None = None) -> None:
        if n_pfc is not None and len(self.pfc) != int(n_pfc):
            raise ValueError(f"delta_derivative_limits_aps.pfc must contain {int(n_pfc)} values")
        if n_sol is not None and len(self.sol) != int(n_sol):
            raise ValueError(f"delta_derivative_limits_aps.sol must contain {int(n_sol)} values")
        for name, values in (("pfc", self.pfc), ("sol", self.sol)):
            for idx, value in enumerate(values):
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"delta_derivative_limits_aps.{name}[{idx}] must be finite and positive")


@dataclass(frozen=True, slots=True)
class IpReferenceConfig:
    min: float = 0.0
    max: float = 1.0
    rate_limit: float = 0.0
    segment_min_steps: int = 50
    segment_max_steps: int = 300
    segment_count_min: int = 3
    segment_count_max: int = 8
    hold_probability: float = 0.35
    kind: Literal[
        "segmented",
        "hold_reset",
        "segmented_profile",
        "single_segment_profile",
        "replay_window",
        "hold_boundary_eval_profile",
        "hold_boundary_eval_cut_profile",
    ] = "segmented"
    limits_path: Path | None = None
    start_mode: Literal["reset_ip"] = "reset_ip"
    parent_steps: int = 0
    plateau_min_fraction: float = 0.25
    plateau_max_fraction: float = 1.0
    end_min_fraction: float = 0.25
    end_max_fraction: float = 1.0
    ramp_rate_reference: Literal["p95", "robust_mean"] = "p95"
    ramp_up_rate_min_fraction: float = 0.0
    ramp_up_rate_fraction: float = 0.25
    ramp_down_rate_min_fraction: float = 0.0
    ramp_down_rate_fraction: float = 0.25
    hold_min_steps: int = 50
    hold_max_steps: int = 250
    final_hold_min_steps: int = 0
    smooth_ramps: bool = True
    max_delta_fraction: float = 1.0


@dataclass(frozen=True, slots=True)
class BoundaryReferenceConfig:
    kind: Literal[
        "static_initial_parameters",
        "rate_limited_parameters",
        "hold_reset_boundary",
        "t15_replay_segment_conditioned",
    ] = "static_initial_parameters"
    rate_limits: dict[str, float] = field(default_factory=dict)
    replay_reference_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    duration_s: float
    t_step: float
    theta_count: int
    seed: int
    ip: IpReferenceConfig
    boundary: BoundaryReferenceConfig


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    actor_kind: Literal["controller_state_v4", "controller_state_v5", "controller_state_v6"] = "controller_state_v6"
    critic_kind: Literal["privileged_training_state_v1", "compact_training_state_v2"] = "compact_training_state_v2"
    target_preview_steps: int = 8
    target_preview_stride: int = 10
    ip_rate_scale_aps: float = 500000.0
    boundary_rate_scale_mps: float = 1.0


@dataclass(frozen=True, slots=True)
class RewardConfig:
    kind: Literal["physical_cost", "tcv_quality", "tcv_quality_legacy", "tcv_derivative"] = "physical_cost"
    shape_mean_scale_m: float = 0.03
    shape_max_scale_m: float = 0.08
    ip_scale_a: float = 25000.0
    boundary_missing_error_m: float = 0.10
    boundary_missing_weight: float = 0.0
    shape_mean_weight: float = 4.0
    shape_max_weight: float = 1.0
    ip_weight: float = 3.0
    current_weight: float = 2.0
    derivative_weight: float = 0.5
    current_drift_weight: float = 0.0
    current_drift_bad_fraction: float = 0.10
    mean_jdot_bias_weight: float = 0.0
    mean_jdot_bias_bad_fraction: float = 0.10
    current_usage_weight: float = 0.0
    derivative_usage_weight: float = 0.0
    action_weight: float = 0.02
    delta_action_weight: float = 0.05
    current_soft_fraction: float = 0.90
    current_bad_fraction: float = 1.40
    derivative_soft_fraction: float = 0.90
    derivative_bad_fraction: float = 1.40
    terminal_reward: float = -20.0
    terminal_remaining_cost: float = 0.0
    actuator_saturation_weight: float = 4.0
    reward_scale: float = 1.0
    smoothmax_alpha: float = 5.0


@dataclass(frozen=True, slots=True)
class RandomizationConfig:
    enabled: bool = False
    ip_measurement_noise_a: float = 0.0
    current_measurement_noise_a: float = 0.0
    action_offset_min: float = 0.0
    action_offset_max: float = 0.0


@dataclass(frozen=True, slots=True)
class SimConfig:
    config_path: Path
    initial_currents_path: Path | None
    compute_backend: Literal["cpu", "gpu"] = "cpu"
    gpu_device: str = "cuda:0"
    angles: int = 32
    max_episode_steps: int = 1000
    initial_ranges: InitialRanges | None = None
    reset_source: Literal["initial_ranges", "csv_initial_states"] = "initial_ranges"
    csv_initial_state_library: Path | None = None
    csv_initial_state_split: Literal["train", "holdout", "all"] = "train"
    current_safety_limits: CurrentSafetyLimits | None = None
    current_limit_scale: float = 1.0
    derivative_limit_scale: float = 1.0
    action_scale: float = 1.0
    action_contract: Literal["jdot_command"] = "jdot_command"
    delta_derivative_scale_aps: float = 500000.0
    delta_derivative_limits_aps: DeltaDerivativeLimits | None = None
    terminate_on_boundary_loss: bool = True
    terminate_on_current_limit: bool = True
    current_termination_over_limit_a: float = 5000.0
    current_termination_grace_steps: int = 8
    current_hard_termination_fraction: float = 1.05
    current_saturation_fraction: float = 1.15


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    hidden_dim: int = 256
    critic_hidden_dim: int = 256
    critic_mlp_hidden_dim: int = 256
    actor_initial_std: float = 0.2
    actor_min_std: float = 1.0e-4


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    discount: float = 0.99
    unroll_length: int = 64
    batch_size: int = 256
    replay_capacity_episodes: int = 2048
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    kl_lr: float = 3.0e-4
    action_samples: int = 20
    min_replay_sequence_length: int = 8
    actor_update_chunk_size: int = 2048
    temperature: float = 1.0
    mpo_epsilon: float = 0.1
    mean_kl_epsilon: float = 0.01
    std_kl_epsilon: float = 1.0e-4
    target_update_tau: float = 0.005
    rollout_chunk_length: int = 64
    updates_per_rollout_chunk: int = 16


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 10000
    num_envs: int = 16
    device: str = "auto"
    seed: int = 1
    output_dir: Path = Path("outputs/run")
    save_checkpoints: bool = False
    checkpoint_interval_steps: int = 10000
    eval_checkpoint_top_k: int = 0
    milestone_checkpoint_interval_steps: int = 0
    keep_latest_checkpoint: bool = True
    eval_interval_steps: int = 10000
    eval_episodes: int = 8
    eval_max_steps: int = 1000
    actor_workers: int = 1
    actor_devices: tuple[str, ...] = ()
    distributed_mode: Literal["single", "local_replay"] = "single"
    production_mode: bool = False
    early_stop_patience_evals: int = 0
    early_stop_min_delta: float = 0.0


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    sim: SimConfig
    reference: ReferenceConfig
    observation: ObservationConfig
    reward: RewardConfig
    randomization: RandomizationConfig
    network: NetworkConfig
    learner: LearnerConfig
    training: TrainingConfig
