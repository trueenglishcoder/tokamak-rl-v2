from __future__ import annotations

from dataclasses import dataclass, replace
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
from tokamak_rl_v2.env.shot_fragments import ShotFragmentLibrary
from tokamak_rl_v2.rewards import T15PhysicalReward


@dataclass(slots=True)
class BatchStep:
    obs: Tensor
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
    ip_reference: np.ndarray | None = None
    shot_ids: tuple[str, ...] = ()
    shot_start_times_s: np.ndarray | None = None


class TokamakMagneticControlEnv:
    """Batched training environment using tokamak-sim as the plant."""

    def __init__(self, config: ExperimentConfig, *, batch_size: int, device: torch.device | str, seed: int) -> None:
        self.config = config
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(int(seed))
        self.cfg = load_config(config.sim.config_path, initial_currents_path=config.sim.initial_currents_path)
        if config.sim.compute_backend == "gpu":
            self.cfg = replace(self.cfg, compute=ComputeSettings(backend="gpu", gpu_device=config.sim.gpu_device))
            self.cfg.compute.validate(require_available=True)
        if self.cfg.limiter_shape is None:
            raise ValueError("T15 training requires limiter geometry")
        self.angles = np.linspace(-np.pi, np.pi, int(config.sim.angles), endpoint=False, dtype=float)
        self.action_dim = self.cfg.pfc.n_coils + self.cfg.sol.n_coils
        self.obs_dim = self._obs_dim()
        self.reward_fn = T15PhysicalReward(config.reward, control_rate_hz=1.0 / float(self.cfg.physics.t_step))
        self.current_limits = torch.as_tensor(_current_limit_vector(config, self.cfg), dtype=torch.float32, device=self.device)
        self.raw_derivative_limits = torch.as_tensor(np.concatenate([_limit_vec(self.cfg.physics.pfc_deriv_limit, self.cfg.pfc.n_coils), _limit_vec(self.cfg.physics.sol_deriv_limit, self.cfg.sol.n_coils)]), dtype=torch.float32, device=self.device)
        self.derivative_limits = self.raw_derivative_limits * float(config.sim.action_scale)
        self.previous_action = torch.zeros((self.batch_size, self.action_dim), dtype=torch.float32, device=self.device)
        self.action_offset = torch.zeros((self.batch_size, self.action_dim), dtype=torch.float32, device=self.device)
        self.reference: ReferenceBatch | None = None
        self.step_index = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
        self.done = torch.ones((self.batch_size,), dtype=torch.bool, device=self.device)
        self._cpu_models: list[PlasmaModel] = []
        self._gpu_sim: BatchedGpuTokamakSimulator | None = None
        self._shot_fragments = (
            ShotFragmentLibrary(
                config.sim.shot_fragments,
                n_pfc=self.cfg.pfc.n_coils,
                n_sol=self.cfg.sol.n_coils,
                dt=float(self.cfg.physics.t_step),
            )
            if config.sim.shot_fragments is not None
            else None
        )
        self.reset_metadata: list[dict[str, object]] = []

    @property
    def pfc(self):
        return self.cfg.pfc

    @property
    def sol(self):
        return self.cfg.sol

    def reset(self) -> Tensor:
        payload = self._sample_reset_payload(self.batch_size)
        ip0, pfc0, sol0, params0 = payload.ip0, payload.pfc0, payload.sol0, payload.params0
        self._record_reset_metadata(payload)
        self.previous_action.zero_()
        self.action_offset = self._sample_action_offset(self.batch_size)
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
            self.reference = self._reference_for_reset_payload(payload=payload, boundary_points=result.boundary.points, boundary_radii=result.boundary.radii)
            return self._obs_gpu(result=result)
        self._cpu_models = [self._new_cpu_model(ip=float(ip0[b]), pfc_currents=pfc0[b], sol_currents=sol0[b]) for b in range(self.batch_size)]
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
        self.step_index[indices_t] = 0
        self.done[indices_t] = False
        if self.config.sim.compute_backend == "gpu":
            assert self._gpu_sim is not None
            result = self._gpu_sim.reset_indices(indices, ip=ip0, pfc_currents=pfc0, sol_currents=sol0)
            idx_for_boundary = indices_t.to(device=result.boundary.points.device)
            reference = self._reference_for_reset_payload(payload=payload, boundary_points=result.boundary.points[idx_for_boundary], boundary_radii=result.boundary.radii[idx_for_boundary])
            self._write_reference_indices(indices_t, reference)
            self._record_reset_metadata(payload, indices=indices)
            return self._obs_gpu(result=result)
        for local, env_index in enumerate(indices):
            self._cpu_models[env_index] = self._new_cpu_model(ip=float(ip0[local]), pfc_currents=pfc0[local], sol_currents=sol0[local])
        points0, radii0, _found0 = self._cpu_boundary_samples(indices=indices)
        reference = self._reference_for_reset_payload(payload=payload, boundary_points=points0, boundary_radii=radii0)
        self._write_reference_indices(indices_t, reference)
        self._record_reset_metadata(payload, indices=indices)
        return self._obs_cpu()

    def _sample_reset_payload(self, count: int) -> ResetPayload:
        if self.config.sim.initial_ranges is None:
            raise ValueError("training config must provide replay-bounded initial_ranges")
        if self._shot_fragments is not None:
            shot = self._shot_fragments.sample(self.rng, count=int(count), steps=int(self.config.sim.max_episode_steps))
            _ip_unused, _pfc_unused, _sol_unused, params0 = sample_initial_conditions(self.rng, self.config.sim.initial_ranges, int(count))
            return ResetPayload(
                ip0=shot.ip0,
                pfc0=shot.pfc_currents,
                sol0=shot.sol_currents,
                params0=params0,
                reference_seed=int(self.rng.integers(0, 2**31 - 1)),
                ip_reference=shot.ip_reference,
                shot_ids=shot.shot_ids,
                shot_start_times_s=shot.start_times_s,
            )
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
        if self.config.reference.ip.kind == "shot_trapezoid_fragment":
            kwargs["ip_reference"] = payload.ip_reference
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
        starts = payload.shot_start_times_s if payload.shot_start_times_s is not None else np.full((len(payload.shot_ids),), np.nan)
        self.reset_metadata = [
            {
                "env_index": int(env_indices[i]),
                "shot_id": str(payload.shot_ids[i]),
                "shot_start_time_s": float(starts[i]),
                "initial_ip": float(payload.ip0[i]),
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

    def _new_cpu_model(self, *, ip: float, pfc_currents: np.ndarray, sol_currents: np.ndarray) -> PlasmaModel:
        pfc = self.cfg.pfc.__class__(name=self.cfg.pfc.name, coils=list(self.cfg.pfc.coils), currents=pfc_currents)
        sol = self.cfg.sol.__class__(name=self.cfg.sol.name, coils=list(self.cfg.sol.coils), currents=sol_currents)
        physics = replace(self.cfg.physics, Ip0=float(ip))
        return PlasmaModel.from_settings(grid=self.cfg.grid, pfc=pfc, sol=sol, settings=physics)

    def step(self, action: Tensor) -> BatchStep:
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(self.batch_size, self.action_dim)
        commanded = torch.clamp(action, -1.0, 1.0)
        pre_projection = torch.clamp(commanded + self.action_offset, -1.0, 1.0)
        clipped = self._project_action_to_current_safety(pre_projection)
        projection_delta = clipped - pre_projection
        physical = clipped * self.derivative_limits[None, :]
        if self.config.sim.compute_backend == "gpu":
            assert self._gpu_sim is not None
            result = self._gpu_sim.step(physical.to(dtype=torch.float64, device=self._gpu_sim.device))
            self.step_index += 1
            obs = self._obs_gpu(result=result)
            reward, terminated, info = self._reward_gpu(result, clipped, projection_delta)
        else:
            self._step_cpu(physical)
            self.step_index += 1
            obs = self._obs_cpu()
            reward, terminated, info = self._reward_cpu(clipped, projection_delta)
        self.previous_action = clipped.detach().clone()
        truncated = self.step_index >= int(self.config.sim.max_episode_steps)
        self.done = terminated | truncated
        return BatchStep(obs=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)

    def _project_action_to_current_safety(self, normalized_action: Tensor) -> Tensor:
        if not self.config.sim.project_actions_to_current_limits:
            return normalized_action
        current, previous_derivative = self._pre_step_current_and_derivative()
        dt = max(float(self.cfg.physics.t_step), 1.0e-12)
        alpha = self._actuator_alpha()
        beta = max(1.0 - alpha, 1.0e-12)
        margin = float(self.config.sim.current_projection_margin_fraction)
        usable_limits = torch.clamp(self.current_limits * (1.0 - margin), min=1.0e-12)
        lower_applied = (-usable_limits[None, :] - current) / dt
        upper_applied = (usable_limits[None, :] - current) / dt
        lower_command = (lower_applied - alpha * previous_derivative) / beta
        upper_command = (upper_applied - alpha * previous_derivative) / beta
        deriv_scale = torch.where(torch.isfinite(self.derivative_limits) & (self.derivative_limits > 0.0), self.derivative_limits, torch.ones_like(self.derivative_limits))
        lower_norm = torch.minimum(lower_command, upper_command) / deriv_scale[None, :]
        upper_norm = torch.maximum(lower_command, upper_command) / deriv_scale[None, :]
        lower_norm = torch.clamp(lower_norm, -1.0, 1.0)
        upper_norm = torch.clamp(upper_norm, -1.0, 1.0)
        return torch.minimum(torch.maximum(normalized_action, lower_norm), upper_norm)

    def _pre_step_current_and_derivative(self) -> tuple[Tensor, Tensor]:
        if self.config.sim.compute_backend == "gpu":
            assert self._gpu_sim is not None
            current = torch.cat([self._gpu_sim.pfc_currents, self._gpu_sim.sol_currents], dim=1).to(dtype=torch.float32, device=self.device)
            derivative = torch.cat([self._gpu_sim.pfc_derivs, self._gpu_sim.sol_derivs], dim=1).to(dtype=torch.float32, device=self.device)
            return current, derivative
        currents = []
        derivatives = []
        for model in self._cpu_models:
            currents.append(np.concatenate([model.state.pfc_currents, model.state.sol_currents]))
            derivatives.append(np.concatenate([model.state.pfc_current_derivs, model.state.sol_current_derivs]))
        return (
            torch.as_tensor(np.stack(currents), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.stack(derivatives), dtype=torch.float32, device=self.device),
        )

    def _actuator_alpha(self) -> float:
        tau = float(self.cfg.physics.actuator_tau)
        if tau <= 0.0:
            return 0.0
        return float(np.exp(-float(self.cfg.physics.t_step) / tau))

    def _obs_dim(self) -> int:
        return self._feature_slices()["target_preview"][1]

    def _feature_slices(self) -> dict[str, tuple[int, int]]:
        n_angles = int(self.config.sim.angles)
        nz, nr = int(self.cfg.grid.z.size), int(self.cfg.grid.r.size)
        preview = int(self.config.observation.target_preview_steps)
        sizes = (
            ("step_norm", 1),
            ("ip", 1),
            ("ip_ref", 1),
            ("ip_error", 1),
            ("active_currents", self.action_dim),
            ("active_current_derivs", self.action_dim),
            ("psi_flat", nz * nr),
            ("measured_boundary_radii", n_angles),
            ("ref_radii", n_angles),
            ("boundary_radii_error", n_angles),
            ("boundary_found", 1),
            ("target_preview", preview * (2 + n_angles)),
        )
        out: dict[str, tuple[int, int]] = {}
        start = 0
        for name, size in sizes:
            end = start + int(size)
            out[name] = (start, end)
            start = end
        return out

    def _feature_order(self) -> list[str]:
        return list(self._feature_slices().keys())

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
        psi = result.state.psi.to(torch.float32).reshape(self.batch_size, -1)
        obs = torch.cat([
            self.step_index.to(torch.float32).reshape(self.batch_size, 1) / max(float(self.config.sim.max_episode_steps), 1.0),
            (ip / 5.0e5).reshape(self.batch_size, 1),
            (ip_ref / 5.0e5).reshape(self.batch_size, 1),
            ((ip - ip_ref) / 5.0e5).reshape(self.batch_size, 1),
            currents / current_scale[None, :],
            derivs / deriv_scale[None, :],
            psi / 1.0,
            measured_radii,
            target_radii,
            target_radii - measured_radii,
            boundary_found,
            self._preview(),
        ], dim=1)
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _obs_cpu(self) -> Tensor:
        obs = []
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
                np.asarray(model.state.psi, dtype=float).reshape(-1) / 1.0,
                np.nan_to_num(measured_radii, nan=0.0, posinf=0.0, neginf=0.0),
                target_radii,
                target_radii - np.nan_to_num(measured_radii, nan=0.0, posinf=0.0, neginf=0.0),
                np.array([boundary_found], dtype=float),
                preview[b],
            ]
            obs.append(np.concatenate(parts))
        return torch.nan_to_num(torch.as_tensor(np.stack(obs, axis=0), dtype=torch.float32, device=self.device), nan=0.0, posinf=0.0, neginf=0.0)

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

    def _reward_gpu(self, result, action: Tensor, projection_delta: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
        ip_ref, ref_points, _ref_radii = self._reference_at()
        current = torch.cat([result.state.pfc_currents, result.state.sol_currents], dim=1).to(torch.float32)
        deriv = torch.cat([result.state.pfc_current_derivs, result.state.sol_current_derivs], dim=1).to(torch.float32)
        current_scale = torch.where(torch.isfinite(self.current_limits) & (self.current_limits > 0.0), self.current_limits, torch.ones_like(self.current_limits))
        current_abs = torch.abs(current)
        current_usage_by_coil = current_abs / current_scale[None, :]
        current_over_limit = torch.max(torch.clamp(current_abs - self.current_limits[None, :], min=0.0), dim=1).values
        current_usage_fraction = torch.max(current_usage_by_coil, dim=1).values
        current_margin_fraction = torch.min(1.0 - current_usage_by_coil, dim=1).values
        derivative_usage = torch.max(torch.abs(deriv) / torch.where(torch.isfinite(self.raw_derivative_limits) & (self.raw_derivative_limits > 0.0), self.raw_derivative_limits, torch.ones_like(self.raw_derivative_limits))[None, :], dim=1).values
        boundary_points = result.boundary.points[:, : int(self.config.sim.angles)].to(torch.float32)
        ref = ref_points[:, : int(self.config.sim.angles)].to(torch.float32)
        found = result.boundary.found.to(torch.bool)
        boundary_terminated = ~found if self.config.sim.terminate_on_boundary_loss else torch.zeros_like(found, dtype=torch.bool)
        current_terminated = torch.zeros_like(found, dtype=torch.bool)
        if self.config.sim.terminate_on_current_limit:
            current_terminated = current_over_limit > float(self.config.sim.current_termination_over_limit_a)
        terminated = boundary_terminated | current_terminated
        episode_progress = self.step_index.to(torch.float32) / max(float(self.config.sim.max_episode_steps), 1.0)
        rb = self.reward_fn(ip=result.state.Ip.to(torch.float32), ip_ref=ip_ref, boundary_points=boundary_points, reference_points=ref, action=action, previous_action=self.previous_action, current_over_limit_a=current_over_limit, current_usage_fraction=current_usage_fraction, current_margin_fraction=current_margin_fraction, derivative_usage=derivative_usage, boundary_found=found, terminated=terminated, episode_progress=episode_progress, action_projection_delta=projection_delta)
        return rb.reward, terminated, {"reward_components": {k: v.detach().cpu().numpy() for k, v in rb.components.items()}}

    def _reward_cpu(self, action: Tensor, projection_delta: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
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
        current_margin_fraction = torch.min(1.0 - current_usage_by_coil, dim=1).values
        derivative_usage = torch.max(torch.abs(deriv_t) / torch.where(torch.isfinite(self.raw_derivative_limits) & (self.raw_derivative_limits > 0.0), self.raw_derivative_limits, torch.ones_like(self.raw_derivative_limits))[None, :], dim=1).values
        found_t = torch.as_tensor(found, dtype=torch.bool, device=self.device)
        boundary_terminated = ~found_t if self.config.sim.terminate_on_boundary_loss else torch.zeros_like(found_t, dtype=torch.bool)
        current_terminated = torch.zeros_like(found_t, dtype=torch.bool)
        if self.config.sim.terminate_on_current_limit:
            current_terminated = current_over_limit > float(self.config.sim.current_termination_over_limit_a)
        terminated = boundary_terminated | current_terminated
        episode_progress = self.step_index.to(torch.float32) / max(float(self.config.sim.max_episode_steps), 1.0)
        rb = self.reward_fn(ip=torch.as_tensor(ips, dtype=torch.float32, device=self.device), ip_ref=ip_ref, boundary_points=torch.nan_to_num(torch.as_tensor(np.stack(boundary_points), dtype=torch.float32, device=self.device)), reference_points=ref_points[:, : int(self.config.sim.angles)].to(torch.float32), action=action, previous_action=self.previous_action, current_over_limit_a=current_over_limit, current_usage_fraction=current_usage_fraction, current_margin_fraction=current_margin_fraction, derivative_usage=derivative_usage, boundary_found=found_t, terminated=terminated, episode_progress=episode_progress, action_projection_delta=projection_delta)
        return rb.reward, terminated, {"reward_components": {k: v.detach().cpu().numpy() for k, v in rb.components.items()}}

    def export_schema(self) -> dict[str, object]:
        return {
            "observation_kind": "joint_state_v1",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "n_active_total": self.action_dim,
            "n_pfc": self.cfg.pfc.n_coils,
            "n_sol": self.cfg.sol.n_coils,
            "n_angles": int(self.config.sim.angles),
            "angles_rad": np.asarray(self.angles, dtype=float).tolist(),
            "grid_shape": [int(self.cfg.grid.z.size), int(self.cfg.grid.r.size)],
            "feature_order": self._feature_order(),
            "feature_slices": {name: [int(start), int(end)] for name, (start, end) in self._feature_slices().items()},
            "target_preview_steps": int(self.config.observation.target_preview_steps),
            "target_preview_stride": int(self.config.observation.target_preview_stride),
            "action_scale": float(self.config.sim.action_scale),
        }

    def normalization(self) -> dict[str, object]:
        return {
            "ip_scale": 5.0e5,
            "radius_scale": 1.0,
            "psi_scale": 1.0,
            "current_scale": self.current_limits.detach().cpu().numpy().astype(float).tolist(),
            "derivative_scale": self.derivative_limits.detach().cpu().numpy().astype(float).tolist(),
            "current_projection_enabled": bool(self.config.sim.project_actions_to_current_limits),
            "current_projection_margin_fraction": float(self.config.sim.current_projection_margin_fraction),
            "t_step": float(self.cfg.physics.t_step),
            "actuator_tau": float(self.cfg.physics.actuator_tau),
        }


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
        return np.concatenate([
            np.asarray(config.sim.current_safety_limits.pfc_currents, dtype=float),
            np.asarray(config.sim.current_safety_limits.sol_currents, dtype=float),
        ])
    pfc = _limit_vec(loaded_cfg.physics.pfc_current_limit, n_pfc)
    sol = _limit_vec(loaded_cfg.physics.sol_current_limit, n_sol)
    out = np.concatenate([pfc, sol])
    if not np.all(np.isfinite(out)):
        raise ValueError("Training reward requires explicit finite current_safety_limits when simulator current limits are absent")
    return out


def _limit_vec(limit: float | None, n: int) -> np.ndarray:
    if limit is None or not np.isfinite(float(limit)) or float(limit) <= 0.0:
        return np.full((int(n),), np.inf, dtype=float)
    return np.full((int(n),), float(limit), dtype=float)
