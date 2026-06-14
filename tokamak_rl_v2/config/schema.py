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
class IpReferenceConfig:
    min: float
    max: float
    rate_limit: float
    segment_min_steps: int
    segment_max_steps: int
    segment_count_min: int
    segment_count_max: int
    hold_probability: float


@dataclass(frozen=True, slots=True)
class BoundaryReferenceConfig:
    kind: Literal["static_initial_parameters", "rate_limited_parameters", "hold_reset_boundary"] = "static_initial_parameters"
    rate_limits: dict[str, float] = field(default_factory=dict)


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
    target_preview_steps: int = 8
    target_preview_stride: int = 10


@dataclass(frozen=True, slots=True)
class RewardConfig:
    shape_bad_m: float = 0.03
    shape_max_bad_m: float = 0.08
    ip_bad_a: float = 20000.0
    boundary_missing_error_m: float = 0.10
    shape_weight: float = 2.0
    ip_weight: float = 2.0
    current_weight: float = 8.0
    derivative_weight: float = 0.1
    action_saturation_weight: float = 0.1
    delta_action_weight: float = 0.05
    current_margin_start_fraction: float = 0.75
    derivative_penalty_start_fraction: float = 0.85
    action_penalty_start_fraction: float = 0.85
    delta_action_penalty_start: float = 0.25
    delta_action_bad: float = 1.0
    terminal_reward: float = -20.0
    reward_scale: float = 1.0
    late_error_weight: float = 0.0
    late_error_power: float = 2.0


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
    current_safety_limits: CurrentSafetyLimits | None = None
    action_scale: float = 1.0
    terminate_on_boundary_loss: bool = True
    terminate_on_current_limit: bool = False
    current_termination_over_limit_a: float = 0.0
    project_actions_to_current_limits: bool = False
    current_projection_margin_fraction: float = 0.0


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
    checkpoint_interval_steps: int = 10000
    eval_interval_steps: int = 10000
    eval_episodes: int = 8
    eval_max_steps: int = 1000
    actor_workers: int = 1
    actor_devices: tuple[str, ...] = ()


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
