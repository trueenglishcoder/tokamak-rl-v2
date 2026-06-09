from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tokamak_control.compute import ComputeSettings
from tokamak_control.core.batched_gpu_simulator import BatchedGpuTokamakSimulator
from tokamak_control.core.plasma_model import PlasmaModel
from tokamak_control.diagnostics import default_t15_diagnostic_layout, magnetic_diagnostics_numpy
from tokamak_control.geometry.boundary import BoundaryNotFoundError, find_plasma_boundary_with_status
from tokamak_control.geometry.coordinates import radii_from_polyline_ray_intersections
from tokamak_control.geometry.parametric_boundary import BoundaryParameters, evaluate_parametric_boundary
from tokamak_control.io.config_io import load_config
from tokamak_control.metrics import current_limit_margin, derivative_limit_margin

from tokamak_rl_v2.config.schema import ExperimentConfig
from tokamak_rl_v2.env.references import ReferenceBatch, generate_reference_batch, sample_initial_conditions
from tokamak_rl_v2.rewards.t15_static import T15StaticBoundaryReward


@dataclass(slots=True)
class BatchStep:
    obs: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    info: dict[str, object]


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
        self.diagnostic_layout = default_t15_diagnostic_layout(
            grid=self.cfg.grid,
            limiter_shape=self.cfg.limiter_shape,
            center=(self.cfg.physics.R0, self.cfg.physics.Z0),
            flux_count=config.observation.diagnostic_flux_count,
            field_count=config.observation.diagnostic_field_count,
        )
        self.action_dim = self.cfg.pfc.n_coils + self.cfg.sol.n_coils
        self.obs_dim = self._obs_dim()
        self.reward_fn = T15StaticBoundaryReward(config.reward, control_rate_hz=1.0 / float(self.cfg.physics.t_step))
        self.current_limits = torch.as_tensor(np.concatenate([_limit_vec(self.cfg.physics.pfc_current_limit, self.cfg.pfc.n_coils), _limit_vec(self.cfg.physics.sol_current_limit, self.cfg.sol.n_coils)]), dtype=torch.float32, device=self.device)
        self.derivative_limits = torch.as_tensor(np.concatenate([_limit_vec(self.cfg.physics.pfc_deriv_limit, self.cfg.pfc.n_coils), _limit_vec(self.cfg.physics.sol_deriv_limit, self.cfg.sol.n_coils)]), dtype=torch.float32, device=self.device)
        self.previous_action = torch.zeros((self.batch_size, self.action_dim), dtype=torch.float32, device=self.device)
        self.action_offset = torch.zeros((self.batch_size, self.action_dim), dtype=torch.float32, device=self.device)
        self.reference: ReferenceBatch | None = None
        self.step_index = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
        self.done = torch.ones((self.batch_size,), dtype=torch.bool, device=self.device)
        self._cpu_models: list[PlasmaModel] = []
        self._previous_flux: list[np.ndarray | None] = [None] * self.batch_size
        self._gpu_sim: BatchedGpuTokamakSimulator | None = None

    @property
    def pfc(self):
        return self.cfg.pfc

    @property
    def sol(self):
        return self.cfg.sol

    def reset(self) -> Tensor:
        if self.config.sim.initial_ranges is None:
            raise ValueError("training config must provide replay-bounded initial_ranges")
        ip0, pfc0, sol0, params0 = sample_initial_conditions(self.rng, self.config.sim.initial_ranges, self.batch_size)
        steps = int(self.config.sim.max_episode_steps)
        self.reference = generate_reference_batch(
            config=self.config.reference,
            initial_ip=ip0,
            initial_parameters=params0,
            steps=steps,
            device=self.device,
            seed=int(self.rng.integers(0, 2**31 - 1)),
        )
        self.previous_action.zero_()
        if self.config.randomization.enabled and (self.config.randomization.action_offset_min != 0.0 or self.config.randomization.action_offset_max != 0.0):
            low = float(self.config.randomization.action_offset_min)
            high = float(self.config.randomization.action_offset_max)
            self.action_offset = torch.empty((self.batch_size, self.action_dim), dtype=torch.float32, device=self.device).uniform_(low, high)
        else:
            self.action_offset = torch.zeros((self.batch_size, self.action_dim), dtype=torch.float32, device=self.device)
        self.step_index.zero_()
        self.done.zero_()
        self._previous_flux = [None] * self.batch_size
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
                diagnostic_layout=self.diagnostic_layout,
            )
            self._gpu_sim.reset(ip=ip0, pfc_currents=pfc0, sol_currents=sol0)
            return self._obs_gpu()
        self._cpu_models = []
        for b in range(self.batch_size):
            cfg_b = self.cfg
            pfc = cfg_b.pfc.__class__(name=cfg_b.pfc.name, coils=list(cfg_b.pfc.coils), currents=pfc0[b])
            sol = cfg_b.sol.__class__(name=cfg_b.sol.name, coils=list(cfg_b.sol.coils), currents=sol0[b])
            physics = replace(cfg_b.physics, Ip0=float(ip0[b]))
            self._cpu_models.append(PlasmaModel.from_settings(grid=cfg_b.grid, pfc=pfc, sol=sol, settings=physics))
        return self._obs_cpu()

    def step(self, action: Tensor) -> BatchStep:
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(self.batch_size, self.action_dim)
        commanded = torch.clamp(action, -1.0, 1.0)
        clipped = torch.clamp(commanded + self.action_offset, -1.0, 1.0)
        physical = clipped * self.derivative_limits[None, :]
        if self.config.sim.compute_backend == "gpu":
            assert self._gpu_sim is not None
            result = self._gpu_sim.step(physical.to(dtype=torch.float64, device=self._gpu_sim.device))
            obs = self._obs_gpu(result=result)
            reward, terminated, info = self._reward_gpu(result, clipped)
        else:
            obs = self._step_cpu(physical)
            reward, terminated, info = self._reward_cpu(clipped)
        self.previous_action = commanded.detach().clone()
        self.step_index += 1
        truncated = self.step_index >= int(self.config.sim.max_episode_steps)
        self.done = terminated | truncated
        return BatchStep(obs=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)

    def _obs_dim(self) -> int:
        preview = int(self.config.observation.target_preview_steps)
        return (
            4
            + self.action_dim
            + self.diagnostic_layout.flux_count
            + self.diagnostic_layout.field_count
            + self.diagnostic_layout.flux_count
            + int(self.config.sim.angles)
            + preview * (2 + int(self.config.sim.angles))
            + self.action_dim
        )

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
        diag = result.diagnostics
        currents = torch.cat([result.state.pfc_currents, result.state.sol_currents], dim=1).to(torch.float32)
        ip = result.state.Ip.to(torch.float32)
        flux = diag["flux"].to(torch.float32)
        field = diag["field"].to(torch.float32)
        bdot = diag["bdot"].to(torch.float32)
        if self.config.randomization.enabled:
            if self.config.randomization.ip_measurement_noise_a > 0.0:
                ip = ip + torch.randn_like(ip) * float(self.config.randomization.ip_measurement_noise_a)
            if self.config.randomization.current_measurement_noise_a > 0.0:
                currents = currents + torch.randn_like(currents) * float(self.config.randomization.current_measurement_noise_a)
            if self.config.randomization.flux_noise > 0.0:
                flux = flux + torch.randn_like(flux) * float(self.config.randomization.flux_noise)
            if self.config.randomization.field_noise > 0.0:
                field = field + torch.randn_like(field) * float(self.config.randomization.field_noise)
        obs = torch.cat([
            torch.stack([self.step_index.to(torch.float32) / max(float(self.config.sim.max_episode_steps), 1.0), ip / 5.0e5, ip_ref / 5.0e5, (ip - ip_ref) / 5.0e5], dim=1),
            currents / 1.5e6,
            flux,
            field,
            bdot,
            ref_radii[:, : int(self.config.sim.angles)].to(torch.float32),
            self._preview(),
            self.previous_action,
        ], dim=1)
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _obs_cpu(self) -> Tensor:
        obs = []
        ip_ref, _ref_points, ref_radii = self._reference_at()
        for b, model in enumerate(self._cpu_models):
            diag = magnetic_diagnostics_numpy(psi=model.state.psi, grid=model.grid, layout=self.diagnostic_layout, previous_flux=self._previous_flux[b], dt=float(model.t_step))
            self._previous_flux[b] = diag["flux"].copy()
            currents = np.concatenate([model.state.pfc_currents, model.state.sol_currents])
            measured_ip = float(model.state.Ip)
            flux = np.asarray(diag["flux"], dtype=float).copy()
            field = np.asarray(diag["field"], dtype=float).copy()
            bdot = np.asarray(diag["bdot"], dtype=float).copy()
            if self.config.randomization.enabled:
                if self.config.randomization.ip_measurement_noise_a > 0.0:
                    measured_ip += float(self.rng.normal(0.0, float(self.config.randomization.ip_measurement_noise_a)))
                if self.config.randomization.current_measurement_noise_a > 0.0:
                    currents = currents + self.rng.normal(0.0, float(self.config.randomization.current_measurement_noise_a), size=currents.shape)
                if self.config.randomization.flux_noise > 0.0:
                    flux = flux + self.rng.normal(0.0, float(self.config.randomization.flux_noise), size=flux.shape)
                if self.config.randomization.field_noise > 0.0:
                    field = field + self.rng.normal(0.0, float(self.config.randomization.field_noise), size=field.shape)
            parts = [
                np.array([float(self.step_index[b].item()) / max(float(self.config.sim.max_episode_steps), 1.0), measured_ip / 5.0e5, float(ip_ref[b].item()) / 5.0e5, (measured_ip - float(ip_ref[b].item())) / 5.0e5], dtype=float),
                currents / 1.5e6,
                flux,
                field,
                bdot,
                ref_radii[b, : int(self.config.sim.angles)].detach().cpu().numpy(),
                self._preview()[b].detach().cpu().numpy(),
                self.previous_action[b].detach().cpu().numpy(),
            ]
            obs.append(np.concatenate(parts))
        return torch.nan_to_num(torch.as_tensor(np.stack(obs, axis=0), dtype=torch.float32, device=self.device), nan=0.0, posinf=0.0, neginf=0.0)

    def _step_cpu(self, physical: Tensor) -> Tensor:
        arr = physical.detach().cpu().numpy()
        for b, model in enumerate(self._cpu_models):
            model.step(pfc_current_derivs=arr[b, : self.cfg.pfc.n_coils], sol_current_derivs=arr[b, self.cfg.pfc.n_coils :])
        return self._obs_cpu()

    def _reward_gpu(self, result, action: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
        ip_ref, ref_points, _ref_radii = self._reference_at()
        current = torch.cat([result.state.pfc_currents, result.state.sol_currents], dim=1).to(torch.float32)
        deriv = torch.cat([result.state.pfc_current_derivs, result.state.sol_current_derivs], dim=1).to(torch.float32)
        cur_margin = torch.min(1.0 - torch.abs(current) / self.current_limits[None, :], dim=1).values
        der_margin = torch.min(1.0 - torch.abs(deriv) / self.derivative_limits[None, :], dim=1).values
        boundary_points = result.boundary.points[:, : int(self.config.sim.angles)].to(torch.float32)
        ref = ref_points[:, : int(self.config.sim.angles)].to(torch.float32)
        found = result.boundary.found.to(torch.bool)
        terminated = ~found
        rb = self.reward_fn(ip=result.state.Ip.to(torch.float32), ip_ref=ip_ref, boundary_points=boundary_points, reference_points=ref, action=action, previous_action=self.previous_action, current_margin=cur_margin, derivative_margin=der_margin, boundary_found=found, terminated=terminated)
        return rb.reward, terminated, {"reward_components": {k: v.detach().cpu().numpy() for k, v in rb.components.items()}}

    def _reward_cpu(self, action: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
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
        cur_margin = torch.min(1.0 - torch.abs(current_t) / self.current_limits[None, :], dim=1).values
        der_margin = torch.min(1.0 - torch.abs(deriv_t) / self.derivative_limits[None, :], dim=1).values
        found_t = torch.as_tensor(found, dtype=torch.bool, device=self.device)
        terminated = ~found_t
        rb = self.reward_fn(ip=torch.as_tensor(ips, dtype=torch.float32, device=self.device), ip_ref=ip_ref, boundary_points=torch.nan_to_num(torch.as_tensor(np.stack(boundary_points), dtype=torch.float32, device=self.device)), reference_points=ref_points[:, : int(self.config.sim.angles)].to(torch.float32), action=action, previous_action=self.previous_action, current_margin=cur_margin, derivative_margin=der_margin, boundary_found=found_t, terminated=terminated)
        return rb.reward, terminated, {"reward_components": {k: v.detach().cpu().numpy() for k, v in rb.components.items()}}

    def export_schema(self) -> dict[str, object]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "n_active_total": self.action_dim,
            "n_pfc": self.cfg.pfc.n_coils,
            "n_sol": self.cfg.sol.n_coils,
            "n_angles": int(self.config.sim.angles),
            "target_preview_steps": int(self.config.observation.target_preview_steps),
            "target_preview_stride": int(self.config.observation.target_preview_stride),
            "diagnostics": {
                "flux_points": np.asarray(self.diagnostic_layout.flux_points, dtype=float).tolist(),
                "field_points": np.asarray(self.diagnostic_layout.field_points, dtype=float).tolist(),
                "field_angles": np.asarray(self.diagnostic_layout.field_angles, dtype=float).tolist(),
            },
        }

    def normalization(self) -> dict[str, object]:
        return {
            "ip_scale": 5.0e5,
            "radius_scale": 1.0,
            "flux_scale": 1.0,
            "field_scale": 1.0,
            "bdot_scale": 1.0,
            "current_scale": [1.5e6] * self.action_dim,
            "derivative_scale": self.derivative_limits.detach().cpu().numpy().astype(float).tolist(),
        }


def _limit_vec(limit: float | None, n: int) -> np.ndarray:
    if limit is None or not np.isfinite(float(limit)) or float(limit) <= 0.0:
        return np.full((int(n),), 1.0, dtype=float)
    return np.full((int(n),), float(limit), dtype=float)
