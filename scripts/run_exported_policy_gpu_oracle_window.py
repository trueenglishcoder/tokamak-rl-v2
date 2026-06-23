#!/usr/bin/env python3
"""Compare exported learned-controller deployment against the RL oracle-window path."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tokamak_control.control.learned_magnetic_controller import LearnedMagneticController
from tokamak_control.core.batched_gpu_simulator import BatchedGpuTokamakSimulator
from tokamak_rl_v2.config import load_experiment_config
from tokamak_rl_v2.env import TokamakMagneticControlEnv
from tokamak_rl_v2.networks import FeedForwardGaussianActor


@dataclass(slots=True)
class _StateView:
    """Numpy view of one batched-GPU simulator lane."""

    Ip: float
    pfc_currents: np.ndarray
    sol_currents: np.ndarray
    pfc_current_derivs: np.ndarray
    sol_current_derivs: np.ndarray
    psi: np.ndarray
    t: float
    step: int


@dataclass(slots=True)
class _ModelView:
    """Minimal model object consumed by LearnedMagneticController."""

    pfc: object
    sol: object
    t_step: float
    state: _StateView


class _ArrayReferenceScenario:
    """Reference scenario backed by one exact oracle-window target array."""

    def __init__(self, *, ip: np.ndarray, radii: np.ndarray, dt: float) -> None:
        self.ip = np.asarray(ip, dtype=float).reshape(-1)
        self.radii = np.asarray(radii, dtype=float)
        self.dt = max(float(dt), 1.0e-12)
        if self.radii.ndim != 2 or self.radii.shape[0] != self.ip.shape[0]:
            raise ValueError("reference radii must have shape [steps, angles] aligned with Ip")

    def _index(self, t: float) -> int:
        idx = int(round(float(t) / self.dt))
        return int(np.clip(idx, 0, self.ip.shape[0] - 1))

    def Ip_ref(self, t: float) -> float:
        return float(self.ip[self._index(t)])

    def ref_radii(self, angles: np.ndarray, t: float) -> np.ndarray:
        ref = self.radii[self._index(t)]
        if ref.shape != np.asarray(angles).reshape(-1).shape:
            raise ValueError("reference angle count does not match controller request")
        return np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)


def main(argv: list[str] | None = None) -> int:
    """Run one exact oracle window through RL env and exported GPU deployment."""
    args = _parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this parity runner requires CUDA so deployment uses BatchedGpuTokamakSimulator")

    cfg = load_experiment_config(args.config)
    cfg = _replace_split_and_gpu(cfg, split=str(args.split), device=str(device))
    env = TokamakMagneticControlEnv(cfg, batch_size=1, device=device, seed=int(args.seed))
    obs = env.reset_to_csv_indices([int(args.window_index)])
    if env.reference is None or env._gpu_sim is None:
        raise RuntimeError("environment did not initialize GPU replay-window reference")

    controller = LearnedMagneticController(export_dir=Path(args.export_dir), episode_norm_steps=int(cfg.sim.max_episode_steps), rolling_episode_norm=False)
    actor = _load_export_actor(Path(args.export_dir), obs_dim=env.obs_dim, action_dim=env.action_dim, device=device)
    actor.eval()
    sim_b = _deployment_simulator_from_env(env, device=str(device))
    result_b = sim_b.reset(
        ip=env._gpu_sim.Ip.detach().cpu().numpy(),
        pfc_currents=env._gpu_sim.pfc_currents.detach().cpu().numpy(),
        sol_currents=env._gpu_sim.sol_currents.detach().cpu().numpy(),
    )
    _validate_contract(controller=controller, env=env, cfg=cfg)

    scenario = _ArrayReferenceScenario(
        ip=env.reference.ip[0].detach().cpu().numpy().astype(float),
        radii=env.reference.radii[0, :, : int(cfg.sim.angles)].detach().cpu().numpy().astype(float),
        dt=float(cfg.reference.t_step),
    )
    rows: list[dict[str, Any]] = []
    max_obs_diff = 0.0
    max_action_diff = 0.0
    max_ip_diff = 0.0
    max_radii_diff = 0.0
    first_mismatch: dict[str, Any] | None = None

    for step in range(int(args.steps)):
        step_index = int(env.step_index[0].detach().cpu().item())
        ref_radii = scenario.ref_radii(_angles_np(env), step_index * float(cfg.reference.t_step))
        ref_ip = scenario.Ip_ref(step_index * float(cfg.reference.t_step))
        model_b = _model_view(sim_b, lane=0)
        measured_radii_b = result_b.boundary.radii[0].detach().cpu().numpy().astype(float)
        boundary_found_b = bool(result_b.boundary.found[0].detach().cpu().item())
        controller_obs = controller._observation(
            model=model_b,
            psi=model_b.state.psi,
            boundary_poly=None,
            measured_ip=model_b.state.Ip,
            measured_active_currents=np.concatenate([model_b.state.pfc_currents, model_b.state.sol_currents]),
            measured_radii=measured_radii_b,
            boundary_found=boundary_found_b,
            center=(float(env.cfg.physics.R0), float(env.cfg.physics.Z0)),
            measure_angles=_angles_np(env),
            ref_radii=ref_radii,
            ip_ref=ref_ip,
            scenario=scenario,
            max_episode_steps=int(cfg.sim.max_episode_steps),
        )
        env_obs = obs.detach().cpu().numpy().reshape(-1)
        obs_diff, feature_name = _obs_difference(controller_obs, env_obs, controller.schema)
        max_obs_diff = max(max_obs_diff, obs_diff)

        with torch.no_grad():
            torch_action = actor.deterministic(obs).detach().cpu().numpy().reshape(-1)
        control = controller.compute_control(
            model=model_b,
            psi=model_b.state.psi,
            boundary_poly=None,
            measured_ip=model_b.state.Ip,
            measured_active_currents=np.concatenate([model_b.state.pfc_currents, model_b.state.sol_currents]),
            measured_radii=measured_radii_b,
            boundary_found=boundary_found_b,
            center=(float(env.cfg.physics.R0), float(env.cfg.physics.Z0)),
            measure_angles=_angles_np(env),
            ref_radii=ref_radii,
            Ip_ref=ref_ip,
            scenario=scenario,
            max_episode_steps=int(cfg.sim.max_episode_steps),
        )
        current_now_b = np.concatenate([model_b.state.pfc_currents, model_b.state.sol_currents])
        current_next_b = np.concatenate([control.pfc_currents_next, control.sol_currents_next])
        numpy_action = (current_next_b - current_now_b) / float(cfg.reference.t_step) / controller.derivative_scale
        action_diff = float(np.max(np.abs(numpy_action - torch_action)))
        max_action_diff = max(max_action_diff, action_diff)

        out_a = env.step(torch.as_tensor(torch_action.reshape(1, -1), dtype=torch.float32, device=device))
        result_b = sim_b.step_currents(current_next_b.reshape(1, -1))
        obs = out_a.obs

        ip_a = float(env._gpu_sim.Ip[0].detach().cpu().item())
        ip_b = float(result_b.state.Ip[0].detach().cpu().item())
        radii_a = _slice_from_obs(obs, env, "measured_boundary_radii")
        radii_b = result_b.boundary.radii[0].detach().cpu().numpy().astype(float)
        ip_diff = abs(ip_a - ip_b)
        radii_diff = float(np.nanmax(np.abs(radii_a - radii_b)))
        max_ip_diff = max(max_ip_diff, ip_diff)
        max_radii_diff = max(max_radii_diff, radii_diff)
        row = {
            "step": step,
            "obs_max_abs_diff": obs_diff,
            "obs_feature": feature_name,
            "action_max_abs_diff": action_diff,
            "ip_a": ip_a,
            "ip_b": ip_b,
            "ip_abs_diff": ip_diff,
            "boundary_radii_max_abs_diff_m": radii_diff,
            "ip_ref_a": scenario.Ip_ref((step + 1) * float(cfg.reference.t_step)),
            "shape_error_mean_m": float(np.mean(np.abs(radii_b - scenario.ref_radii(_angles_np(env), (step + 1) * float(cfg.reference.t_step))))),
            "action_rms": float(np.sqrt(np.mean(np.square(numpy_action)))),
        }
        rows.append(row)
        if first_mismatch is None and (obs_diff > float(args.obs_tolerance) or action_diff > float(args.action_tolerance) or ip_diff > float(args.state_tolerance) or radii_diff > float(args.radii_tolerance)):
            first_mismatch = dict(row)
        if bool(args.stop_on_first_mismatch) and first_mismatch is not None:
            break

    _write_csv(out_dir / "gpu_export_oracle_window_parity.csv", rows)
    summary = {
        "config": str(Path(args.config).resolve()),
        "export_dir": str(Path(args.export_dir).resolve()),
        "split": str(args.split),
        "window_index": int(args.window_index),
        "steps_requested": int(args.steps),
        "steps_run": len(rows),
        "max_obs_diff": max_obs_diff,
        "max_action_diff": max_action_diff,
        "max_ip_diff": max_ip_diff,
        "max_boundary_radii_diff_m": max_radii_diff,
        "first_mismatch": first_mismatch,
        "passed": first_mismatch is None,
    }
    (out_dir / "gpu_export_oracle_window_parity.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _plot(out_dir / "gpu_export_oracle_window_parity.png", rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if first_mismatch is None else 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "holdout", "all"))
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=386400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--obs-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--action-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--state-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--radii-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--stop-on-first-mismatch", action="store_true")
    return parser.parse_args(argv)


def _replace_split_and_gpu(cfg, *, split: str, device: str):
    from dataclasses import replace

    return replace(
        cfg,
        sim=replace(cfg.sim, csv_initial_state_split=str(split), compute_backend="gpu", gpu_device=str(device)),
        training=replace(cfg.training, production_mode=False),
    )


def _load_export_actor(export_dir: Path, *, obs_dim: int, action_dim: int, device: torch.device) -> FeedForwardGaussianActor:
    actor_path = export_dir / "actor.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"exact parity requires actor.pt in export bundle: {actor_path}")
    data = torch.load(actor_path, map_location=device, weights_only=False)
    state = data.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"export actor is missing actor_state_dict: {actor_path}")
    input_weight = state.get("input.weight")
    if not isinstance(input_weight, torch.Tensor) or input_weight.ndim != 2:
        raise ValueError(f"export actor has invalid input.weight in actor_state_dict: {actor_path}")
    hidden_dim = int(input_weight.shape[0])
    actor = FeedForwardGaussianActor(obs_dim, action_dim, hidden_dim).to(device)
    actor.load_state_dict(state)
    return actor


def _validate_contract(*, controller: LearnedMagneticController, env: TokamakMagneticControlEnv, cfg: object) -> None:
    """Reject exported-policy deployment when it no longer matches training."""
    if controller.action_contract != "absolute_jdot_command_v1":
        raise ValueError(f"expected absolute_jdot_command_v1 export, got {controller.action_contract!r}")
    if controller.obs_dim != env.obs_dim:
        raise ValueError(f"observation dimension mismatch: export={controller.obs_dim} env={env.obs_dim}")
    if controller.action_dim != env.action_dim:
        raise ValueError(f"action dimension mismatch: export={controller.action_dim} env={env.action_dim}")
    env_n_pfc = int(env.cfg.pfc.n_coils)
    env_n_sol = int(env.cfg.sol.n_coils)
    if controller.n_pfc != env_n_pfc or controller.n_sol != env_n_sol:
        raise ValueError(f"coil count mismatch: export=({controller.n_pfc}, {controller.n_sol}) env=({env_n_pfc}, {env_n_sol})")
    if not np.isclose(float(cfg.reference.t_step), 0.001):
        raise ValueError(f"expected local 1 ms timestep, got {cfg.reference.t_step}")
    if int(cfg.sim.max_episode_steps) != 100:
        raise ValueError(f"expected 100-step oracle-window episodes, got {cfg.sim.max_episode_steps}")
    export_angles = np.asarray(controller.schema.get("angles_rad", []), dtype=float)
    if export_angles.shape != _angles_np(env).shape or not np.allclose(export_angles, _angles_np(env), atol=1.0e-7, rtol=0.0):
        raise ValueError("boundary angle grid mismatch between export and environment")
    derivative_limits = _as_numpy(env.derivative_limits)
    current_limits = _as_numpy(env.current_limits)
    if controller.derivative_scale.shape != derivative_limits.shape:
        raise ValueError("derivative normalization shape mismatch")
    if not np.allclose(controller.derivative_scale, derivative_limits, rtol=1.0e-6, atol=1.0e-3):
        raise ValueError("derivative normalization mismatch between export and environment")
    if controller.current_scale.shape != current_limits.shape:
        raise ValueError("current normalization shape mismatch")
    if not np.allclose(controller.current_scale, current_limits, rtol=1.0e-6, atol=1.0e-3):
        raise ValueError("current normalization mismatch between export and environment")
    expected_reference_hash = _reference_hash(cfg.reference)
    export_reference_hash = str(controller.metadata.get("reference_hash", ""))
    if export_reference_hash and export_reference_hash != expected_reference_hash:
        raise ValueError(
            "reference hash mismatch between export and config: "
            f"export={export_reference_hash} config={expected_reference_hash}"
        )


def _reference_hash(reference_cfg: object) -> str:
    """Return the same reference-contract hash written into policy exports."""
    fragment = _config_fragment(reference_cfg)
    payload = json.dumps(fragment, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_fragment(obj: object) -> object:
    """Convert a dataclass-like config fragment into stable JSON data."""
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {str(key): _config_fragment(value) for key, value in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(key): _config_fragment(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_config_fragment(value) for value in obj]
    return obj


def _deployment_simulator_from_env(env: TokamakMagneticControlEnv, *, device: str) -> BatchedGpuTokamakSimulator:
    return BatchedGpuTokamakSimulator(
        grid=env.cfg.grid,
        pfc=env.cfg.pfc,
        sol=env.cfg.sol,
        settings=env.cfg.physics,
        batch_size=1,
        angles_rad=_angles_np(env),
        limiter_shape=env.cfg.limiter_shape,
        boundary_mode=env.cfg.boundary_mode,
        boundary_base_mode=env.cfg.boundary_base_mode,
        boundary_level_smoothing_alpha=env.cfg.boundary_level_smoothing_alpha,
        boundary_level_search_span_fraction=env.cfg.boundary_level_search_span_fraction,
        boundary_continuity_weight_radii=env.cfg.boundary_continuity_weight_radii,
        boundary_continuity_weight_mean_radius=env.cfg.boundary_continuity_weight_mean_radius,
        boundary_continuity_weight_center=env.cfg.boundary_continuity_weight_center,
        boundary_continuity_weight_area=env.cfg.boundary_continuity_weight_area,
        boundary_continuity_weight_level=env.cfg.boundary_continuity_weight_level,
        gpu_device=device,
    )


def _model_view(sim: BatchedGpuTokamakSimulator, *, lane: int) -> _ModelView:
    idx = int(lane)
    state = _StateView(
        Ip=float(sim.Ip[idx].detach().cpu().item()),
        pfc_currents=sim.pfc_currents[idx].detach().cpu().numpy().astype(float),
        sol_currents=sim.sol_currents[idx].detach().cpu().numpy().astype(float),
        pfc_current_derivs=sim.pfc_derivs[idx].detach().cpu().numpy().astype(float),
        sol_current_derivs=sim.sol_derivs[idx].detach().cpu().numpy().astype(float),
        psi=sim.psi[idx].detach().cpu().numpy().astype(float),
        t=float(sim.time_s[idx].detach().cpu().item()),
        step=int(sim.step_index[idx].detach().cpu().item()),
    )
    return _ModelView(pfc=sim.pfc, sol=sim.sol, t_step=float(sim.settings.t_step), state=state)


def _angles_np(env: TokamakMagneticControlEnv) -> np.ndarray:
    return _as_numpy(env.angles)


def _as_numpy(value: object) -> np.ndarray:
    """Return a float NumPy array from either tensor or array-like values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(float)
    return np.asarray(value, dtype=float)


def _slice_from_obs(obs: torch.Tensor, env: TokamakMagneticControlEnv, name: str) -> np.ndarray:
    start, stop = env._actor_feature_slices()[name]
    return obs[0, start:stop].detach().cpu().numpy().astype(float)


def _obs_difference(controller_obs: np.ndarray, env_obs: np.ndarray, schema: dict[str, object]) -> tuple[float, str]:
    delta = np.abs(np.asarray(controller_obs, dtype=float).reshape(-1) - np.asarray(env_obs, dtype=float).reshape(-1))
    max_diff = float(np.nanmax(delta)) if delta.size else 0.0
    feature_slices = schema.get("feature_slices", {})
    if not isinstance(feature_slices, dict):
        return max_diff, ""
    worst_name = ""
    worst_value = -1.0
    for name, bounds in feature_slices.items():
        try:
            start, stop = int(bounds[0]), int(bounds[1])
        except (TypeError, ValueError, IndexError):
            continue
        if stop <= start:
            continue
        value = float(np.nanmax(delta[start:stop])) if delta[start:stop].size else 0.0
        if value > worst_value:
            worst_value = value
            worst_name = str(name)
    return max_diff, worst_name


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    t = np.asarray([float(row["step"]) for row in rows], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(t, [float(row["obs_max_abs_diff"]) for row in rows], label="obs")
    axes[0].plot(t, [float(row["action_max_abs_diff"]) for row in rows], label="action")
    axes[0].set_ylabel("max diff")
    axes[0].legend()
    axes[1].plot(t, [float(row["ip_a"]) for row in rows], label="RL env")
    axes[1].plot(t, [float(row["ip_b"]) for row in rows], label="export GPU", linestyle="--")
    axes[1].plot(t, [float(row["ip_ref_a"]) for row in rows], label="ref", alpha=0.7)
    axes[1].set_ylabel("Ip [A]")
    axes[1].legend()
    axes[2].plot(t, [float(row["boundary_radii_max_abs_diff_m"]) for row in rows], label="radii diff")
    axes[2].set_ylabel("radii diff [m]")
    axes[2].legend()
    axes[3].plot(t, [float(row["shape_error_mean_m"]) for row in rows], label="shape mean")
    axes[3].plot(t, [float(row["action_rms"]) for row in rows], label="action rms")
    axes[3].set_xlabel("step")
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
