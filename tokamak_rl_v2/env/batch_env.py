from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tokamak_control.compute import ComputeSettings
from tokamak_control.core.batched_gpu_simulator import BatchedGpuTokamakSimulator
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections
from tokamak_control.io.config_io import load_config

from tokamak_rl_v2.config.schema import ExperimentConfig
from tokamak_rl_v2.env.references import ReferenceBatch, generate_reference_batch, sample_initial_conditions
from tokamak_rl_v2.env.t15_csv_initial_states import CsvInitialStateLibrary
from tokamak_rl_v2.rewards import build_reward


@dataclass(slots=True)
class BatchStep:
    obs: Tensor
    critic_obs: Tensor
    requested_action: Tensor
    applied_action: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    info: dict[str, object]


@dataclass(slots=True)
class ResetPayload:
    ip0: np.ndarray
    pfc0: np.ndarray
    sol0: np.ndarray
    params0: np.ndarray
    reference_seed: int
    shot_ids: tuple[str, ...] = ()
    source_indices: tuple[int, ...] = ()
    source_times_s: tuple[float, ...] = ()


class TokamakMagneticControlEnv:
    """Batched training environment using tokamak-sim as the plant."""

    def __init__(self, config: ExperimentConfig, *, batch_size: int, device: torch.device | str, seed: int) -> None:
        self.config = config
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(int(seed))
        self.cfg = load_config(config.sim.config_path, initial_currents_path=config.sim.initial_currents_path)
        if not math.isclose(float(config.reference.t_step), float(self.cfg.physics.t_step), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "reference.t_step must match tokamak-sim physics.t_step exactly for runtime consistency"
            )
        if config.sim.compute_backend == "gpu":
            self.cfg = replace(self.cfg, compute=ComputeSettings(backend="gpu", gpu_device=config.sim.gpu_device))
            self.cfg.compute.validate(require_available=True)
        self.cfg = _scale_simulator_limits(
            self.cfg,
            current_scale=float(config.sim.current_limit_scale),
            derivative_scale=float(config.sim.derivative_limit_scale),
        )
        if self.cfg.limiter_shape is None:
            raise ValueError("T15 training requires limiter geometry")
        self.angles = np.linspace(-np.pi, np.pi, int(config.sim.angles), endpoint=False, dtype=float)
        self.action_dim = self.cfg.pfc.n_coils + self.cfg.sol.n_coils
        self.actor_obs_dim = self._actor_obs_dim()
        self.obs_dim = self.actor_obs_dim
        self.critic_obs_dim = self._critic_obs_dim()
        self.reward_fn = build_reward(config.reward, control_rate_hz=1.0 / float(self.cfg.physics.t_step))
        self.current_limits = torch.as_tensor(_current_limit_vector(config, self.cfg), dtype=torch.float32, device=self.device)
        self.raw_derivative_limits = torch.as_tensor(np.concatenate([_limit_vec(self.cfg.physics.pfc_deriv_limit, self.cfg.pfc.n_coils), _limit_vec(self.cfg.physics.sol_deriv_limit, self.cfg.sol.n_coils)]), dtype=torch.float32, device=self.device)
        self.derivative_limits = self.raw_derivative_limits * float(config.sim.action_scale)
        self.previous_action = torch.zeros((self.batch_size, self.action_dim), dtype=torch.float32, device=self.device)
        self.action_offset = torch.zeros((self.batch_size, self.action_dim), dtype=torch.float32, device=self.device)
        self.current_over_limit_steps = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
        self.psi_reset_mean = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        self.psi_reset_std = torch.ones((self.batch_size,), dtype=torch.float32, device=self.device)
        self.reference: ReferenceBatch | None = None
        self._last_critic_obs = torch.zeros((self.batch_size, self.critic_obs_dim), dtype=torch.float32, device=self.device)
        self.step_index = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
        self.done = torch.ones((self.batch_size,), dtype=torch.bool, device=self.device)
        self._cpu_models: list[PlasmaModel] = []
        self._gpu_sim: BatchedGpuTokamakSimulator | None = None
        self._csv_initial_states = (
            CsvInitialStateLibrary(
                config.sim.csv_initial_state_library,
                n_pfc=self.cfg.pfc.n_coils,
                n_sol=self.cfg.sol.n_coils,
                split=config.sim.csv_initial_state_split,
            )
            if config.sim.reset_source == "csv_initial_states"
            else None
        )
        self.reset_metadata: list[dict[str, object]] = []

    @property
    def pfc(self):
        return self.cfg.pfc

    @property
    def sol(self):
        return self.cfg.sol

    def critic_obs(self) -> Tensor:
        return self._last_critic_obs

    def reset(self) -> Tensor:
        payload = self._sample_reset_payload(self.batch_size)
        ip0, pfc0, sol0, params0 = payload.ip0, payload.pfc0, payload.sol0, payload.params0
        self._record_reset_metadata(payload)
        self.previous_action.zero_()
        self.action_offset = self._sample_action_offset(self.batch_size)
        self.current_over_limit_steps.zero_()
        self.step_index.zero_()
        self.done.zero_()
        if self.config.sim.compute_backend == "gpu":
            self._gpu_sim = BatchedGpuTokamakSimulator(
                grid=self.cfg.grid,
                pfc=self.cfg.pfc,
                sol=self.cfg.sol,
                settings=self.cfg.physics,
                batch_size=self.batch_size,
                angles_rad=self.angles,
                limiter_shape=self.cfg.limiter_shape,
                gpu_device=self.config.sim.gpu_device,
            )
            result = self._gpu_sim.reset(ip=ip0, pfc_currents=pfc0, sol_currents=sol0)
            self._update_psi_reset_stats(result.state.psi)
            self.reference = self._reference_for_reset_payload(payload=payload, boundary_points=result.boundary.points, boundary_radii=result.boundary.radii)
            return self._obs_gpu(result=result)
        self._cpu_models = [self._new_cpu_model(ip=float(ip0[b]), pfc_currents=pfc0[b], sol_currents=sol0[b]) for b in range(self.batch_size)]
        self._update_psi_reset_stats_cpu()
        points0, radii0, _found0 = self._cpu_boundary_samples()
        self.reference = self._reference_for_reset_payload(payload=payload, boundary_points=points0, boundary_radii=radii0)
        return self._obs_cpu()

    def reset_indices(self, done_mask: Tensor | np.ndarray | list[bool]) -> Tensor:
        mask = torch.as_tensor(done_mask, dtype=torch.bool, device=self.device).reshape(self.batch_size)
        if not bool(torch.any(mask).item()):
            if self.config.sim.compute_backend == "gpu":
                return self._obs_gpu()
            return self._obs_cpu()
        indices_t = torch.nonzero(mask, as_tuple=False).reshape(-1)
        indices = [int(v) for v in indices_t.detach().cpu().tolist()]
        count = len(indices)
        payload = self._sample_reset_payload(count)
        ip0, pfc0, sol0 = payload.ip0, payload.pfc0, payload.sol0
        assert self.reference is not None
        self.previous_action[indices_t] = 0.0
        self.action_offset[indices_t] = self._sample_action_offset(count)
        self.current_over_limit_steps[indices_t] = 0
        self.step_index[indices_t] = 0
        self.done[indices_t] = False
        if self.config.sim.compute_backend == "gpu":
            assert self._gpu_sim is not None
            result = self._gpu_sim.reset_indices(indices, ip=ip0, pfc_currents=pfc0, sol_currents=sol0)
            self._update_psi_reset_stats(result.state.psi, indices=indices_t)
            idx_for_boundary = indices_t.to(device=result.boundary.points.device)
            reference = self._reference_for_reset_payload(payload=payload, boundary_points=result.boundary.points[idx_for_boundary], boundary_radii=result.boundary.radii[idx_for_boundary])
            self._write_reference_indices(indices_t, reference)
            self._record_reset_metadata(payload, indices=indices)
            return self._obs_gpu(result=result)
        for local, env_index in enumerate(indices):
            self._cpu_models[env_index] = self._new_cpu_model(ip=float(ip0[local]), pfc_currents=pfc0[local], sol_currents=sol0[local])
        self._update_psi_reset_stats_cpu(indices=indices)
        points0, radii0, _found0 = self._cpu_boundary_samples(indices=indices)
        reference = self._reference_for_reset_payload(payload=payload, boundary_points=points0, boundary_radii=radii0)
        self._write_reference_indices(indices_t, reference)
        self._record_reset_metadata(payload, indices=indices)
        return self._obs_cpu()

    def _sample_reset_payload(self, count: int) -> ResetPayload:
        if self._csv_initial_states is not None:
            reset = self._csv_initial_states.sample(self.rng, count=int(count))
            return ResetPayload(
                ip0=reset.ip0,
                pfc0=reset.pfc0,
                sol0=reset.sol0,
                params0=np.zeros((int(count), 5), dtype=float),
                reference_seed=int(self.rng.integers(0, 2**31 - 1)),
                shot_ids=reset.shot_ids,
                source_indices=reset.source_indices,
                source_times_s=reset.source_times_s,
            )
        if self.config.sim.initial_ranges is None:
            raise ValueError("training config must provide replay-bounded initial_ranges")
        ip0, pfc0, sol0, params0 = sample_initial_conditions(self.rng, self.config.sim.initial_ranges, int(count))
        return ResetPayload(ip0=ip0, pfc0=pfc0, sol0=sol0, params0=params0, reference_seed=int(self.rng.integers(0, 2**31 - 1)))

    def _reference_for_reset_payload(
        self,
        *,
        payload: ResetPayload,
        boundary_points,
        boundary_radii,
    ) -> ReferenceBatch:
        kwargs: dict[str, object] = {}
        if self.config.reference.boundary.kind == "hold_reset_boundary":
            kwargs["initial_boundary_points"] = boundary_points
            kwargs["initial_boundary_radii"] = boundary_radii
        return generate_reference_batch(
            config=self.config.reference,
            initial_ip=payload.ip0,
            initial_parameters=payload.params0,
            steps=int(self.config.sim.max_episode_steps),
            device=self.device,
            seed=int(payload.reference_seed),
            **kwargs,
        )

    def _record_reset_metadata(self, payload: ResetPayload, *, indices: list[int] | None = None) -> None:
        if not payload.shot_ids:
            self.reset_metadata = []
            return
        env_indices = list(range(len(payload.shot_ids))) if indices is None else list(indices)
        self.reset_metadata = [
            {
                "env_index": int(env_indices[i]),
                "shot_id": str(payload.shot_ids[i]),
                "initial_ip": float(payload.ip0[i]),
                "source_index": int(payload.source_indices[i]) if i < len(payload.source_indices) else -1,
                "source_time_s": float(payload.source_times_s[i]) if i < len(payload.source_times_s) else float("nan"),
            }
            for i in range(len(payload.shot_ids))
        ]

    def _write_reference_indices(self, indices_t: Tensor, reference: ReferenceBatch) -> None:
        assert self.reference is not None
        self.reference.ip[indices_t] = reference.ip
        self.reference.parameters[indices_t] = reference.parameters
        self.reference.points[indices_t] = reference.points
        self.reference.radii[indices_t] = reference.radii

    def _sample_action_offset(self, count: int) -> Tensor:
        if self.config.randomization.enabled and (self.config.randomization.action_offset_min != 0.0 or self.config.randomization.action_offset_max != 0.0):
            low = float(self.config.randomization.action_offset_min)
            high = float(self.config.randomization.action_offset_max)
            return torch.empty((int(count), self.action_dim), dtype=torch.float32, device=self.device).uniform_(low, high)
        return torch.zeros((int(count), self.action_dim), dtype=torch.float32, device=self.device)

    def _update_psi_reset_stats(self, psi: Tensor, *, indices: Tensor | None = None) -> None:
        raw = torch.as_tensor(psi, dtype=torch.float32, device=self.device)
        first_dim = int(raw.shape[0]) if raw.ndim >= 2 else self.batch_size
        flat = raw.reshape(first_dim, -1)
        mean = torch.mean(flat, dim=1)
        std = torch.std(flat, dim=1, unbiased=False).clamp_min(1.0e-6)
        if indices is None:
            if first_dim != self.batch_size:
                raise ValueError(f"reset psi batch size mismatch: expected {self.batch_size}, got {first_dim}")
            self.psi_reset_mean = mean.detach().clone()
            self.psi_reset_std = std.detach().clone()
            return
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        if first_dim == self.batch_size:
            self.psi_reset_mean[idx] = mean[idx].detach()
            self.psi_reset_std[idx] = std[idx].detach()
        elif first_dim == int(idx.numel()):
            self.psi_reset_mean[idx] = mean.detach()
            self.psi_reset_std[idx] = std.detach()
        else:
            raise ValueError(f"reset psi batch size mismatch: expected full batch or {int(idx.numel())}, got {first_dim}")

    def _update_psi_reset_stats_cpu(self, *, indices: list[int] | None = None) -> None:
        selected = list(range(len(self._cpu_models))) if indices is None else [int(i) for i in indices]
        if not selected:
            return
        flat = np.stack([np.asarray(self._cpu_models[i].state.psi, dtype=float).reshape(-1) for i in selected], axis=0)
        mean = torch.as_tensor(np.mean(flat, axis=1), dtype=torch.float32, device=self.device)
        std = torch.as_tensor(np.std(flat, axis=1), dtype=torch.float32, device=self.device).clamp_min(1.0e-6)
        idx = torch.as_tensor(selected, dtype=torch.long, device=self.device)
        self.psi_reset_mean[idx] = mean.detach()
        self.psi_reset_std[idx] = std.detach()

    def _normalize_psi_flat(self, psi_flat: Tensor) -> Tensor:
        psi = torch.as_tensor(psi_flat, dtype=torch.float32, device=self.device).reshape(self.batch_size, -1)
        return (psi - self.psi_reset_mean[:, None]) / (self.psi_reset_std[:, None] + 1.0e-6)

    def _new_cpu_model(self, *, ip: float, pfc_currents: np.ndarray, sol_currents: np.ndarray) -> PlasmaModel:
        pfc = self.cfg.pfc.__class__(name=self.cfg.pfc.name, coils=list(self.cfg.pfc.coils), currents=pfc_currents)
        sol = self.cfg.sol.__class__(name=self.cfg.sol.name, coils=list(self.cfg.sol.coils), currents=sol_currents)
        physics = replace(self.cfg.physics, Ip0=float(ip))
        return PlasmaModel.from_settings(grid=self.cfg.grid, pfc=pfc, sol=sol, settings=physics)

    def step(self, action: Tensor) -> BatchStep:
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(self.batch_size, self.action_dim)
        commanded = torch.clamp(action, -1.0, 1.0)
        requested_action = torch.clamp(commanded + self.action_offset, -1.0, 1.0)
        applied_action = self._apply_action_contract(requested_action)
        physical = applied_action * self.derivative_limits[None, :]
        previous_action = self.previous_action.detach().clone()
        if self.config.sim.compute_backend == "gpu":
            assert self._gpu_sim is not None
            result = self._gpu_sim.step(physical.to(dtype=torch.float64, device=self._gpu_sim.device))
            self.step_index += 1
            self.previous_action = applied_action.detach().clone()
            obs = self._obs_gpu(result=result)
            reward, terminated, info = self._reward_gpu(result, applied_action, previous_action, requested_action)
        else:
            self._step_cpu(physical)
            self.step_index += 1
            self.previous_action = applied_action.detach().clone()
            obs = self._obs_cpu()
            reward, terminated, info = self._reward_cpu(applied_action, previous_action, requested_action)
        truncated = self.step_index >= int(self.config.sim.max_episode_steps)
        self.done = terminated | truncated
        return BatchStep(
            obs=obs,
            critic_obs=self.critic_obs(),
            requested_action=requested_action.detach().clone(),
            applied_action=applied_action.detach().clone(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _pre_step_bank_currents(self) -> tuple[Tensor, Tensor]:
        if self.config.sim.compute_backend == "gpu":
            assert self._gpu_sim is not None
            return (
                self._gpu_sim.pfc_currents.to(dtype=torch.float32, device=self.device),
                self._gpu_sim.sol_currents.to(dtype=torch.float32, device=self.device),
            )
        pfc = []
        sol = []
        for model in self._cpu_models:
            pfc.append(np.asarray(model.state.pfc_currents, dtype=float).reshape(-1))
            sol.append(np.asarray(model.state.sol_currents, dtype=float).reshape(-1))
        return (
            torch.as_tensor(np.stack(pfc), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.stack(sol), dtype=torch.float32, device=self.device),
        )

    def _pre_step_bank_derivs(self) -> Tensor:
        if self.config.sim.compute_backend == "gpu":
            assert self._gpu_sim is not None
            return torch.cat([self._gpu_sim.pfc_derivs, self._gpu_sim.sol_derivs], dim=1).to(dtype=torch.float32, device=self.device)
        derivs = []
        for model in self._cpu_models:
            derivs.append(np.concatenate([model.state.pfc_current_derivs, model.state.sol_current_derivs]).astype(float, copy=False))
        return torch.as_tensor(np.stack(derivs), dtype=torch.float32, device=self.device)

    def _saturate_action_to_current_limits(self, requested_action: Tensor) -> Tensor:
        pfc_current, sol_current = self._pre_step_bank_currents()
        current = torch.cat([pfc_current, sol_current], dim=1).to(dtype=torch.float32, device=self.device)
        previous_deriv = self._pre_step_bank_derivs()
        requested_physical = requested_action * self.derivative_limits[None, :]

        dt = float(self.cfg.physics.t_step)
        alpha = self._actuator_alpha()
        beta = max(1.0 - float(alpha), 1.0e-12)
        envelope = self.current_limits * float(self.config.sim.current_saturation_fraction)
        finite_envelope = torch.isfinite(envelope) & (envelope > 0.0)
        base_next_current = current + float(dt) * float(alpha) * previous_deriv

        lower_current = -envelope[None, :]
        upper_current = envelope[None, :]
        denom = float(dt) * beta
        cmd_min = (lower_current - base_next_current) / denom
        cmd_max = (upper_current - base_next_current) / denom
        cmd_low = torch.minimum(cmd_min, cmd_max)
        cmd_high = torch.maximum(cmd_min, cmd_max)

        saturated_physical = torch.minimum(torch.maximum(requested_physical, cmd_low), cmd_high)
        saturated_physical = torch.where(finite_envelope[None, :], saturated_physical, requested_physical)
        derivative_limit = torch.where(
            torch.isfinite(self.derivative_limits) & (self.derivative_limits > 0.0),
            self.derivative_limits,
            torch.full_like(self.derivative_limits, float("inf")),
        )
        saturated_physical = torch.minimum(torch.maximum(saturated_physical, -derivative_limit[None, :]), derivative_limit[None, :])
        scale = torch.where(
            torch.isfinite(self.derivative_limits) & (self.derivative_limits > 0.0),
            self.derivative_limits,
            torch.ones_like(self.derivative_limits),
        )
        return torch.clamp(saturated_physical / scale[None, :], -1.0, 1.0)

    def _apply_action_contract(self, requested_action: Tensor) -> Tensor:
        if self.config.reward.kind == "tcv_derivative":
            return torch.clamp(requested_action, -1.0, 1.0)
        return self._saturate_action_to_current_limits(requested_action)

    def _current_termination(
        self,
        *,
        current_over_limit: Tensor,
        current_usage_fraction: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.reward.kind == "tcv_derivative":
            over = current_usage_fraction > 1.0
        else:
            over = current_over_limit > float(self.config.sim.current_termination_over_limit_a)
        self.current_over_limit_steps = torch.where(
            over,
            self.current_over_limit_steps + 1,
            torch.zeros_like(self.current_over_limit_steps),
        )
        if not self.config.sim.terminate_on_current_limit:
            zeros = torch.zeros_like(over, dtype=torch.bool)
            return zeros, zeros, zeros
        if self.config.reward.kind == "tcv_derivative":
            zeros = torch.zeros_like(over, dtype=torch.bool)
            return over, over, zeros
        hard = current_usage_fraction > float(self.config.sim.current_hard_termination_fraction)
        grace = over & (self.current_over_limit_steps >= int(self.config.sim.current_termination_grace_steps))
        return hard | grace, hard, grace

    def _actuator_alpha(self) -> float:
        tau = float(self.cfg.physics.actuator_tau)
        if tau <= 0.0:
            return 0.0
        return float(np.exp(-float(self.cfg.physics.t_step) / tau))

    def _constraint_metrics(self, *, current: Tensor, derivs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        current_scale = torch.where(torch.isfinite(self.current_limits) & (self.current_limits > 0.0), self.current_limits, torch.ones_like(self.current_limits))
        current_usage_by_coil = torch.abs(current) / current_scale[None, :]
        current_usage_fraction = torch.max(current_usage_by_coil, dim=1).values
        current_margin_fraction = torch.min(1.0 - current_usage_by_coil, dim=1).values
        raw_deriv_scale = torch.where(torch.isfinite(self.raw_derivative_limits) & (self.raw_derivative_limits > 0.0), self.raw_derivative_limits, torch.ones_like(self.raw_derivative_limits))
        derivative_usage = torch.max(torch.abs(derivs) / raw_deriv_scale[None, :], dim=1).values
        return current_usage_fraction, current_margin_fraction, derivative_usage

    def _actor_obs_dim(self) -> int:
        return self._actor_feature_slices()["target_preview"][1]

    def _critic_obs_dim(self) -> int:
        return self._critic_feature_slices()["derivative_usage"][1]

    def _actor_feature_slices(self) -> dict[str, tuple[int, int]]:
        n_angles = int(self.config.sim.angles)
        preview = int(self.config.observation.target_preview_steps)
        sizes = (
            ("step_norm", 1),
            ("ip", 1),
            ("ip_ref", 1),
            ("ip_error", 1),
            ("active_currents", self.action_dim),
            ("active_current_derivs", self.action_dim),
            ("measured_boundary_radii", n_angles),
            ("ref_radii", n_angles),
            ("boundary_radii_error", n_angles),
            ("boundary_found", 1),
            ("previous_action", self.action_dim),
            ("target_preview", preview * (2 + n_angles)),
        )
        out: dict[str, tuple[int, int]] = {}
        start = 0
        for name, size in sizes:
            end = start + int(size)
            out[name] = (start, end)
            start = end
        return out

    def _critic_feature_slices(self) -> dict[str, tuple[int, int]]:
        start = self.actor_obs_dim
        nz, nr = int(self.cfg.grid.z.size), int(self.cfg.grid.r.size)
        sizes = (
            ("actor_obs", self.actor_obs_dim),
            ("psi_flat_normalized", nz * nr),
            ("current_usage_fraction", 1),
            ("current_margin_fraction", 1),
            ("derivative_usage", 1),
        )
        out: dict[str, tuple[int, int]] = {}
        start = 0
        for name, size in sizes:
            end = start + int(size)
            out[name] = (start, end)
            start = end
        return out

    def _feature_order(self) -> list[str]:
        return list(self._actor_feature_slices().keys())

    def _critic_feature_order(self) -> list[str]:
        return list(self._critic_feature_slices().keys())

    def _reference_at(self) -> tuple[Tensor, Tensor, Tensor]:
        assert self.reference is not None
        idx = torch.clamp(self.step_index, 0, self.reference.ip.shape[1] - 1)
        b = torch.arange(self.batch_size, device=self.device)
        return self.reference.ip[b, idx].to(torch.float32), self.reference.points[b, idx].to(torch.float32), self.reference.radii[b, idx].to(torch.float32)

    def _preview(self) -> Tensor:
        assert self.reference is not None
        P = int(self.config.observation.target_preview_steps)
        if P == 0:
            return torch.zeros((self.batch_size, 0), dtype=torch.float32, device=self.device)
        stride = int(self.config.observation.target_preview_stride)
        offsets = torch.arange(1, P + 1, dtype=torch.long, device=self.device) * stride
        idx = torch.clamp(self.step_index[:, None] + offsets[None, :], 0, self.reference.ip.shape[1] - 1)
        b = torch.arange(self.batch_size, device=self.device)[:, None]
        time_norm = offsets.to(torch.float32)[None, :].repeat(self.batch_size, 1) / max(float(self.config.sim.max_episode_steps), 1.0)
        ip = self.reference.ip[b, idx].to(torch.float32) / 5.0e5
        radii = self.reference.radii[b, idx][:, :, : int(self.config.sim.angles)].to(torch.float32)
        return torch.cat([time_norm, ip, radii.reshape(self.batch_size, -1)], dim=-1)

    def _obs_gpu(self, result=None) -> Tensor:
        assert self._gpu_sim is not None
        if result is None:
            result = self._gpu_sim._result()
        ip_ref, _ref_points, ref_radii = self._reference_at()
        currents = torch.cat([result.state.pfc_currents, result.state.sol_currents], dim=1).to(torch.float32)
        derivs = torch.cat([result.state.pfc_current_derivs, result.state.sol_current_derivs], dim=1).to(torch.float32)
        ip = result.state.Ip.to(torch.float32)
        if self.config.randomization.enabled:
            if self.config.randomization.ip_measurement_noise_a > 0.0:
                ip = ip + torch.randn_like(ip) * float(self.config.randomization.ip_measurement_noise_a)
            if self.config.randomization.current_measurement_noise_a > 0.0:
                currents = currents + torch.randn_like(currents) * float(self.config.randomization.current_measurement_noise_a)
        measured_radii = torch.nan_to_num(result.boundary.radii[:, : int(self.config.sim.angles)].to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        target_radii = ref_radii[:, : int(self.config.sim.angles)].to(torch.float32)
        boundary_found = result.boundary.found.to(torch.float32).reshape(self.batch_size, 1)
        current_scale = torch.where(torch.isfinite(self.current_limits) & (self.current_limits > 0.0), self.current_limits, torch.ones_like(self.current_limits))
        deriv_scale = torch.where(torch.isfinite(self.derivative_limits) & (self.derivative_limits > 0.0), self.derivative_limits, torch.ones_like(self.derivative_limits))
        psi = self._normalize_psi_flat(result.state.psi)
        current_usage_fraction, current_margin_fraction, derivative_usage = self._constraint_metrics(current=currents, derivs=derivs)
        actor_obs = torch.cat([
            self.step_index.to(torch.float32).reshape(self.batch_size, 1) / max(float(self.config.sim.max_episode_steps), 1.0),
            (ip / 5.0e5).reshape(self.batch_size, 1),
            (ip_ref / 5.0e5).reshape(self.batch_size, 1),
            ((ip - ip_ref) / 5.0e5).reshape(self.batch_size, 1),
            currents / current_scale[None, :],
            derivs / deriv_scale[None, :],
            measured_radii,
            target_radii,
            target_radii - measured_radii,
            boundary_found,
            self.previous_action,
            self._preview(),
        ], dim=1)
        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_critic_obs = torch.nan_to_num(
            torch.cat(
                [
                    actor_obs,
                    psi,
                    current_usage_fraction.reshape(self.batch_size, 1),
                    current_margin_fraction.reshape(self.batch_size, 1),
                    derivative_usage.reshape(self.batch_size, 1),
                ],
                dim=1,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return actor_obs

    def _obs_cpu(self) -> Tensor:
        obs = []
        critic_obs = []
        ip_ref, _ref_points, ref_radii = self._reference_at()
        current_scale = np.asarray(self.current_limits.detach().cpu().numpy(), dtype=float)
        current_scale = np.where(np.isfinite(current_scale) & (current_scale > 0.0), current_scale, 1.0)
        deriv_scale = np.asarray(self.derivative_limits.detach().cpu().numpy(), dtype=float)
        deriv_scale = np.where(np.isfinite(deriv_scale) & (deriv_scale > 0.0), deriv_scale, 1.0)
        preview = self._preview().detach().cpu().numpy()
        for b, model in enumerate(self._cpu_models):
            currents = np.concatenate([model.state.pfc_currents, model.state.sol_currents]).astype(float, copy=False)
            derivs = np.concatenate([model.state.pfc_current_derivs, model.state.sol_current_derivs]).astype(float, copy=False)
            measured_ip = float(model.state.Ip)
            if self.config.randomization.enabled:
                if self.config.randomization.ip_measurement_noise_a > 0.0:
                    measured_ip += float(self.rng.normal(0.0, float(self.config.randomization.ip_measurement_noise_a)))
                if self.config.randomization.current_measurement_noise_a > 0.0:
                    currents = currents + self.rng.normal(0.0, float(self.config.randomization.current_measurement_noise_a), size=currents.shape)
            try:
                poly, _level, _status = find_plasma_boundary_with_status(model.state.psi, model.grid, (model.R0, model.Z0), n_levels=80, limiter_shape=self.cfg.limiter_shape, boundary_mode=self.cfg.boundary_mode)
                measured_radii = radii_from_polyline_ray_intersections(poly, (model.R0, model.Z0), self.angles)
                boundary_found = 1.0
            except BoundaryNotFoundError:
                measured_radii = np.zeros((int(self.config.sim.angles),), dtype=float)
                boundary_found = 0.0
            target_radii = ref_radii[b, : int(self.config.sim.angles)].detach().cpu().numpy().astype(float, copy=False)
            parts = [
                np.array([float(self.step_index[b].item()) / max(float(self.config.sim.max_episode_steps), 1.0), measured_ip / 5.0e5, float(ip_ref[b].item()) / 5.0e5, (measured_ip - float(ip_ref[b].item())) / 5.0e5], dtype=float),
                currents / current_scale,
                derivs / deriv_scale,
                np.nan_to_num(measured_radii, nan=0.0, posinf=0.0, neginf=0.0),
                target_radii,
                target_radii - np.nan_to_num(measured_radii, nan=0.0, posinf=0.0, neginf=0.0),
                np.array([boundary_found], dtype=float),
                self.previous_action[b].detach().cpu().numpy().astype(float, copy=False),
                preview[b],
            ]
            actor = np.concatenate(parts)
            current_usage_by_coil = np.abs(currents) / current_scale
            raw_deriv_scale = np.asarray(self.raw_derivative_limits.detach().cpu().numpy(), dtype=float)
            raw_deriv_scale = np.where(np.isfinite(raw_deriv_scale) & (raw_deriv_scale > 0.0), raw_deriv_scale, 1.0)
            critic = np.concatenate(
                [
                    actor,
                    ((np.asarray(model.state.psi, dtype=float).reshape(-1) - float(self.psi_reset_mean[b].item())) / (float(self.psi_reset_std[b].item()) + 1.0e-6)),
                    np.array(
                        [
                            float(np.max(current_usage_by_coil)),
                            float(np.min(1.0 - current_usage_by_coil)),
                            float(np.max(np.abs(derivs) / raw_deriv_scale)),
                        ],
                        dtype=float,
                    ),
                ]
            )
            obs.append(actor)
            critic_obs.append(critic)
        actor_t = torch.nan_to_num(torch.as_tensor(np.stack(obs, axis=0), dtype=torch.float32, device=self.device), nan=0.0, posinf=0.0, neginf=0.0)
        self._last_critic_obs = torch.nan_to_num(torch.as_tensor(np.stack(critic_obs, axis=0), dtype=torch.float32, device=self.device), nan=0.0, posinf=0.0, neginf=0.0)
        return actor_t

    def _cpu_boundary_samples(self, *, indices: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = list(range(len(self._cpu_models))) if indices is None else [int(i) for i in indices]
        points: list[np.ndarray] = []
        radii_values: list[np.ndarray] = []
        found: list[bool] = []
        angle_count = int(self.config.sim.angles)
        for env_index in selected:
            model = self._cpu_models[env_index]
            try:
                poly, _level, _status = find_plasma_boundary_with_status(model.state.psi, model.grid, (model.R0, model.Z0), n_levels=80, limiter_shape=self.cfg.limiter_shape, boundary_mode=self.cfg.boundary_mode)
                radii = radii_from_polyline_ray_intersections(poly, (model.R0, model.Z0), self.angles)
                pts = np.column_stack([model.R0 + radii * np.cos(self.angles), model.Z0 + radii * np.sin(self.angles)])
                points.append(np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0))
                radii_values.append(np.nan_to_num(radii, nan=0.0, posinf=0.0, neginf=0.0))
                found.append(True)
            except BoundaryNotFoundError:
                points.append(np.zeros((angle_count, 2), dtype=float))
                radii_values.append(np.zeros((angle_count,), dtype=float))
                found.append(False)
        return np.stack(points, axis=0), np.stack(radii_values, axis=0), np.asarray(found, dtype=bool)

    def _step_cpu(self, physical: Tensor) -> None:
        arr = physical.detach().cpu().numpy()
        for b, model in enumerate(self._cpu_models):
            model.step(pfc_current_derivs=arr[b, : self.cfg.pfc.n_coils], sol_current_derivs=arr[b, self.cfg.pfc.n_coils :])

    def _reward_gpu(self, result, action: Tensor, previous_action: Tensor, requested_action: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
        ip_ref, ref_points, _ref_radii = self._reference_at()
        current = torch.cat([result.state.pfc_currents, result.state.sol_currents], dim=1).to(torch.float32)
        deriv = torch.cat([result.state.pfc_current_derivs, result.state.sol_current_derivs], dim=1).to(torch.float32)
        current_scale = torch.where(torch.isfinite(self.current_limits) & (self.current_limits > 0.0), self.current_limits, torch.ones_like(self.current_limits))
        current_abs = torch.abs(current)
        current_usage_by_coil = current_abs / current_scale[None, :]
        current_over_limit = torch.max(torch.clamp(current_abs - self.current_limits[None, :], min=0.0), dim=1).values
        current_usage_fraction = torch.max(current_usage_by_coil, dim=1).values
        current_usage_mean_fraction = torch.mean(current_usage_by_coil, dim=1)
        current_usage_loss = torch.mean(current_usage_by_coil.pow(2), dim=1)
        current_margin_fraction = torch.min(1.0 - current_usage_by_coil, dim=1).values
        derivative_scale = torch.where(torch.isfinite(self.raw_derivative_limits) & (self.raw_derivative_limits > 0.0), self.raw_derivative_limits, torch.ones_like(self.raw_derivative_limits))
        derivative_usage_by_coil = torch.abs(deriv) / derivative_scale[None, :]
        derivative_usage = torch.max(derivative_usage_by_coil, dim=1).values
        derivative_usage_mean_fraction = torch.mean(derivative_usage_by_coil, dim=1)
        derivative_usage_loss = torch.mean(derivative_usage_by_coil.pow(2), dim=1)
        boundary_points = result.boundary.points[:, : int(self.config.sim.angles)].to(torch.float32)
        ref = ref_points[:, : int(self.config.sim.angles)].to(torch.float32)
        found = result.boundary.found.to(torch.bool)
        boundary_terminated = ~found if self.config.sim.terminate_on_boundary_loss else torch.zeros_like(found, dtype=torch.bool)
        current_terminated, current_hard_terminated, current_grace_terminated = self._current_termination(current_over_limit=current_over_limit, current_usage_fraction=current_usage_fraction)
        terminated = boundary_terminated | current_terminated
        episode_progress = self.step_index.to(torch.float32) / max(float(self.config.sim.max_episode_steps), 1.0)
        rb = self.reward_fn(ip=result.state.Ip.to(torch.float32), ip_ref=ip_ref, boundary_points=boundary_points, reference_points=ref, action=action, previous_action=previous_action, requested_action=requested_action, current_over_limit_a=current_over_limit, current_usage_fraction=current_usage_fraction, current_margin_fraction=current_margin_fraction, derivative_usage=derivative_usage, current_usage_loss=current_usage_loss, derivative_usage_loss=derivative_usage_loss, current_usage_mean_fraction=current_usage_mean_fraction, derivative_usage_mean_fraction=derivative_usage_mean_fraction, boundary_found=found, terminated=terminated, episode_progress=episode_progress)
        components = dict(rb.components)
        components["terminated_boundary"] = boundary_terminated.to(dtype=rb.reward.dtype)
        components["terminated_current"] = current_terminated.to(dtype=rb.reward.dtype)
        components["terminated_current_hard"] = current_hard_terminated.to(dtype=rb.reward.dtype)
        components["terminated_current_grace"] = current_grace_terminated.to(dtype=rb.reward.dtype)
        components["current_over_limit_steps"] = self.current_over_limit_steps.to(dtype=rb.reward.dtype)
        return rb.reward, terminated, {"reward_components": {k: v.detach() for k, v in components.items()}}

    def _reward_cpu(self, action: Tensor, previous_action: Tensor, requested_action: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
        ip_ref, ref_points, _ref_radii = self._reference_at()
        boundary_points = []
        found = []
        currents = []
        derivs = []
        ips = []
        for model in self._cpu_models:
            ips.append(model.state.Ip)
            currents.append(np.concatenate([model.state.pfc_currents, model.state.sol_currents]))
            derivs.append(np.concatenate([model.state.pfc_current_derivs, model.state.sol_current_derivs]))
            try:
                poly, _level, _status = find_plasma_boundary_with_status(model.state.psi, model.grid, (model.R0, model.Z0), n_levels=80, limiter_shape=self.cfg.limiter_shape, boundary_mode=self.cfg.boundary_mode)
                radii = radii_from_polyline_ray_intersections(poly, (model.R0, model.Z0), self.angles)
                pts = np.column_stack([model.R0 + radii * np.cos(self.angles), model.Z0 + radii * np.sin(self.angles)])
                boundary_points.append(pts)
                found.append(True)
            except BoundaryNotFoundError:
                boundary_points.append(np.full((int(self.config.sim.angles), 2), np.nan, dtype=float))
                found.append(False)
        current_t = torch.as_tensor(np.stack(currents), dtype=torch.float32, device=self.device)
        deriv_t = torch.as_tensor(np.stack(derivs), dtype=torch.float32, device=self.device)
        current_scale = torch.where(torch.isfinite(self.current_limits) & (self.current_limits > 0.0), self.current_limits, torch.ones_like(self.current_limits))
        current_abs = torch.abs(current_t)
        current_usage_by_coil = current_abs / current_scale[None, :]
        current_over_limit = torch.max(torch.clamp(current_abs - self.current_limits[None, :], min=0.0), dim=1).values
        current_usage_fraction = torch.max(current_usage_by_coil, dim=1).values
        current_usage_mean_fraction = torch.mean(current_usage_by_coil, dim=1)
        current_usage_loss = torch.mean(current_usage_by_coil.pow(2), dim=1)
        current_margin_fraction = torch.min(1.0 - current_usage_by_coil, dim=1).values
        derivative_scale = torch.where(torch.isfinite(self.raw_derivative_limits) & (self.raw_derivative_limits > 0.0), self.raw_derivative_limits, torch.ones_like(self.raw_derivative_limits))
        derivative_usage_by_coil = torch.abs(deriv_t) / derivative_scale[None, :]
        derivative_usage = torch.max(derivative_usage_by_coil, dim=1).values
        derivative_usage_mean_fraction = torch.mean(derivative_usage_by_coil, dim=1)
        derivative_usage_loss = torch.mean(derivative_usage_by_coil.pow(2), dim=1)
        found_t = torch.as_tensor(found, dtype=torch.bool, device=self.device)
        boundary_terminated = ~found_t if self.config.sim.terminate_on_boundary_loss else torch.zeros_like(found_t, dtype=torch.bool)
        current_terminated, current_hard_terminated, current_grace_terminated = self._current_termination(current_over_limit=current_over_limit, current_usage_fraction=current_usage_fraction)
        terminated = boundary_terminated | current_terminated
        episode_progress = self.step_index.to(torch.float32) / max(float(self.config.sim.max_episode_steps), 1.0)
        rb = self.reward_fn(ip=torch.as_tensor(ips, dtype=torch.float32, device=self.device), ip_ref=ip_ref, boundary_points=torch.nan_to_num(torch.as_tensor(np.stack(boundary_points), dtype=torch.float32, device=self.device)), reference_points=ref_points[:, : int(self.config.sim.angles)].to(torch.float32), action=action, previous_action=previous_action, requested_action=requested_action, current_over_limit_a=current_over_limit, current_usage_fraction=current_usage_fraction, current_margin_fraction=current_margin_fraction, derivative_usage=derivative_usage, current_usage_loss=current_usage_loss, derivative_usage_loss=derivative_usage_loss, current_usage_mean_fraction=current_usage_mean_fraction, derivative_usage_mean_fraction=derivative_usage_mean_fraction, boundary_found=found_t, terminated=terminated, episode_progress=episode_progress)
        components = dict(rb.components)
        components["terminated_boundary"] = boundary_terminated.to(dtype=rb.reward.dtype)
        components["terminated_current"] = current_terminated.to(dtype=rb.reward.dtype)
        components["terminated_current_hard"] = current_hard_terminated.to(dtype=rb.reward.dtype)
        components["terminated_current_grace"] = current_grace_terminated.to(dtype=rb.reward.dtype)
        components["current_over_limit_steps"] = self.current_over_limit_steps.to(dtype=rb.reward.dtype)
        return rb.reward, terminated, {"reward_components": {k: v.detach() for k, v in components.items()}}

    def export_schema(self) -> dict[str, object]:
        return {
            "observation_kind": self.config.observation.actor_kind,
            "critic_observation_kind": self.config.observation.critic_kind,
            "obs_dim": self.actor_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "action_dim": self.action_dim,
            "n_active_total": self.action_dim,
            "n_pfc": self.cfg.pfc.n_coils,
            "n_sol": self.cfg.sol.n_coils,
            "n_angles": int(self.config.sim.angles),
            "angles_rad": np.asarray(self.angles, dtype=float).tolist(),
            "grid_shape": [int(self.cfg.grid.z.size), int(self.cfg.grid.r.size)],
            "feature_order": self._feature_order(),
            "feature_slices": {name: [int(start), int(end)] for name, (start, end) in self._actor_feature_slices().items()},
            "critic_feature_order": self._critic_feature_order(),
            "critic_feature_slices": {name: [int(start), int(end)] for name, (start, end) in self._critic_feature_slices().items()},
            "target_preview_steps": int(self.config.observation.target_preview_steps),
            "target_preview_stride": int(self.config.observation.target_preview_stride),
            "action_scale": float(self.config.sim.action_scale),
            "action_contract": self._action_contract_name(),
        }

    def normalization(self) -> dict[str, object]:
        out = {
            "ip_scale": 5.0e5,
            "radius_scale": 1.0,
            "current_scale": self.current_limits.detach().cpu().numpy().astype(float).tolist(),
            "derivative_scale": self.derivative_limits.detach().cpu().numpy().astype(float).tolist(),
            "critic_psi_normalization": "per_reset_standardization",
            "t_step": float(self.cfg.physics.t_step),
            "actuator_tau": float(self.cfg.physics.actuator_tau),
            "action_contract": self._action_contract_name(),
        }
        if self.config.reward.kind != "tcv_derivative":
            out["current_saturation_fraction"] = float(self.config.sim.current_saturation_fraction)
        return out

    def _action_contract_name(self) -> str:
        if self.config.reward.kind == "tcv_derivative":
            return "tcv_derivative_v1"
        return "requested_with_current_aware_saturation_v2"


    def state_dict(self) -> dict[str, object]:
        if self.reference is None:
            raise RuntimeError("environment has not been reset")
        state: dict[str, object] = {
            "batch_size": self.batch_size,
            "compute_backend": self.config.sim.compute_backend,
            "rng_state": self.rng.bit_generator.state,
            "reference": {
                "ip": self.reference.ip.detach().cpu(),
                "parameters": self.reference.parameters.detach().cpu(),
                "points": self.reference.points.detach().cpu(),
                "radii": self.reference.radii.detach().cpu(),
                "theta": self.reference.theta.detach().cpu(),
            },
            "step_index": self.step_index.detach().cpu(),
            "done": self.done.detach().cpu(),
            "previous_action": self.previous_action.detach().cpu(),
            "action_offset": self.action_offset.detach().cpu(),
            "current_over_limit_steps": self.current_over_limit_steps.detach().cpu(),
            "psi_reset_mean": self.psi_reset_mean.detach().cpu(),
            "psi_reset_std": self.psi_reset_std.detach().cpu(),
        }
        if self.config.sim.compute_backend == "gpu":
            assert self._gpu_sim is not None
            state["gpu_sim"] = {
                "Ip": self._gpu_sim.Ip.detach().cpu(),
                "pfc_currents": self._gpu_sim.pfc_currents.detach().cpu(),
                "sol_currents": self._gpu_sim.sol_currents.detach().cpu(),
                "pfc_derivs": self._gpu_sim.pfc_derivs.detach().cpu(),
                "sol_derivs": self._gpu_sim.sol_derivs.detach().cpu(),
                "step_index": self._gpu_sim.step_index.detach().cpu(),
                "time_s": self._gpu_sim.time_s.detach().cpu(),
                "psi": self._gpu_sim.psi.detach().cpu(),
            }
        else:
            state["cpu_model_states"] = [model.snapshot_state() for model in self._cpu_models]
        return state

    def load_state_dict(self, state: dict[str, object]) -> Tensor:
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError(f"environment batch size mismatch: expected {self.batch_size}, got {state['batch_size']}")
        if str(state["compute_backend"]) != self.config.sim.compute_backend:
            raise ValueError(f"environment backend mismatch: expected {self.config.sim.compute_backend}, got {state['compute_backend']}")
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng_state"]
        ref = state["reference"]
        self.reference = ReferenceBatch(
            ip=torch.as_tensor(ref["ip"], dtype=torch.float64, device=self.device),
            parameters=torch.as_tensor(ref["parameters"], dtype=torch.float64, device=self.device),
            points=torch.as_tensor(ref["points"], dtype=torch.float64, device=self.device),
            radii=torch.as_tensor(ref["radii"], dtype=torch.float64, device=self.device),
            theta=torch.as_tensor(ref["theta"], dtype=torch.float64, device=self.device),
        )
        self.step_index = torch.as_tensor(state["step_index"], dtype=torch.long, device=self.device).clone()
        self.done = torch.as_tensor(state["done"], dtype=torch.bool, device=self.device).clone()
        self.previous_action = torch.as_tensor(state["previous_action"], dtype=torch.float32, device=self.device).clone()
        self.action_offset = torch.as_tensor(state["action_offset"], dtype=torch.float32, device=self.device).clone()
        self.current_over_limit_steps = torch.as_tensor(state.get("current_over_limit_steps", torch.zeros((self.batch_size,), dtype=torch.long)), dtype=torch.long, device=self.device).clone()
        self.psi_reset_mean = torch.as_tensor(state.get("psi_reset_mean", torch.zeros((self.batch_size,), dtype=torch.float32)), dtype=torch.float32, device=self.device).clone()
        self.psi_reset_std = torch.as_tensor(state.get("psi_reset_std", torch.ones((self.batch_size,), dtype=torch.float32)), dtype=torch.float32, device=self.device).clone().clamp_min(1.0e-6)
        if self.config.sim.compute_backend == "gpu":
            if self._gpu_sim is None:
                self._gpu_sim = BatchedGpuTokamakSimulator(
                    grid=self.cfg.grid,
                    pfc=self.cfg.pfc,
                    sol=self.cfg.sol,
                    settings=self.cfg.physics,
                    batch_size=self.batch_size,
                    angles_rad=self.angles,
                    limiter_shape=self.cfg.limiter_shape,
                    gpu_device=self.config.sim.gpu_device,
                )
            sim = state["gpu_sim"]
            self._gpu_sim.Ip = torch.as_tensor(sim["Ip"], dtype=self._gpu_sim.dtype, device=self._gpu_sim.device).clone()
            self._gpu_sim.pfc_currents = torch.as_tensor(sim["pfc_currents"], dtype=self._gpu_sim.dtype, device=self._gpu_sim.device).clone()
            self._gpu_sim.sol_currents = torch.as_tensor(sim["sol_currents"], dtype=self._gpu_sim.dtype, device=self._gpu_sim.device).clone()
            self._gpu_sim.pfc_derivs = torch.as_tensor(sim["pfc_derivs"], dtype=self._gpu_sim.dtype, device=self._gpu_sim.device).clone()
            self._gpu_sim.sol_derivs = torch.as_tensor(sim["sol_derivs"], dtype=self._gpu_sim.dtype, device=self._gpu_sim.device).clone()
            self._gpu_sim.step_index = torch.as_tensor(sim["step_index"], dtype=torch.long, device=self._gpu_sim.device).clone()
            self._gpu_sim.time_s = torch.as_tensor(sim["time_s"], dtype=self._gpu_sim.dtype, device=self._gpu_sim.device).clone()
            self._gpu_sim.psi = torch.as_tensor(sim["psi"], dtype=self._gpu_sim.dtype, device=self._gpu_sim.device).clone()
            return self._obs_gpu()
        cpu_states = list(state["cpu_model_states"])
        self._cpu_models = []
        for saved in cpu_states:
            self._cpu_models.append(self._new_cpu_model(ip=float(saved.Ip), pfc_currents=np.asarray(saved.pfc_currents), sol_currents=np.asarray(saved.sol_currents)))
            self._cpu_models[-1].restore_state(saved)
        return self._obs_cpu()


def _current_limit_vector(config: ExperimentConfig, loaded_cfg) -> np.ndarray:
    n_pfc = int(loaded_cfg.pfc.n_coils)
    n_sol = int(loaded_cfg.sol.n_coils)
    if config.sim.current_safety_limits is not None:
        config.sim.current_safety_limits.validate(n_pfc=n_pfc, n_sol=n_sol)
        out = np.concatenate([
            np.asarray(config.sim.current_safety_limits.pfc_currents, dtype=float),
            np.asarray(config.sim.current_safety_limits.sol_currents, dtype=float),
        ])
        return out * float(config.sim.current_limit_scale)
    pfc = _limit_vec(loaded_cfg.physics.pfc_current_limit, n_pfc)
    sol = _limit_vec(loaded_cfg.physics.sol_current_limit, n_sol)
    out = np.concatenate([pfc, sol]) * float(config.sim.current_limit_scale)
    if not np.all(np.isfinite(out)):
        raise ValueError("Training reward requires explicit finite current_safety_limits when simulator current limits are absent")
    return out


def _limit_vec(limit: float | None, n: int) -> np.ndarray:
    if limit is None or not np.isfinite(float(limit)) or float(limit) <= 0.0:
        return np.full((int(n),), np.inf, dtype=float)
    return np.full((int(n),), float(limit), dtype=float)


def _scale_simulator_limits(loaded_cfg, *, current_scale: float, derivative_scale: float):
    physics = loaded_cfg.physics
    scaled = replace(
        physics,
        pfc_current_limit=_scale_optional_limit(physics.pfc_current_limit, current_scale),
        sol_current_limit=_scale_optional_limit(physics.sol_current_limit, current_scale),
        pfc_deriv_limit=_scale_optional_limit(physics.pfc_deriv_limit, derivative_scale),
        sol_deriv_limit=_scale_optional_limit(physics.sol_deriv_limit, derivative_scale),
    )
    return replace(loaded_cfg, physics=scaled)


def _scale_optional_limit(value: float | None, scale: float) -> float | None:
    if value is None:
        return None
    if not np.isfinite(float(value)):
        return value
    return float(value) * float(scale)
