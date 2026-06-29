#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create a tokamak-sim reference for the 900-parent/500-cut hold-boundary eval task.")
    ap.add_argument("--config", required=True, help="RL experiment config used for the exported policy.")
    ap.add_argument("--reference-npz", required=True, help="Replay boundary NPZ for the selected shot, e.g. lqr_boundary_reference_3864.npz.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="holdout", choices=("train", "holdout", "all"))
    ap.add_argument("--library-index", type=int, default=0)
    ap.add_argument("--shot", default="3864")
    ap.add_argument("--seed", type=int, default=386400)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--parent-steps", type=int, default=900)
    ap.add_argument("--segment-min-steps", type=int, default=300)
    ap.add_argument("--angles", type=int, default=32)
    args = ap.parse_args(argv)

    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive")
    if int(args.parent_steps) < int(args.steps):
        raise ValueError("--parent-steps must be >= --steps")

    _ensure_paths()
    from tokamak_control.core.coils import CoilGroup
    from tokamak_control.io.config_io import dump_config, load_config
    from tokamak_rl_v2.config import load_experiment_config
    from tokamak_rl_v2.config.schema import IpReferenceConfig
    from tokamak_rl_v2.env.references import sample_hold_boundary_eval_cut_profile
    from tokamak_rl_v2.env.t15_csv_initial_states import CsvInitialStateLibrary

    cfg = load_experiment_config(args.config)
    sim_cfg = load_config(cfg.sim.config_path)
    library = CsvInitialStateLibrary(
        cfg.sim.csv_initial_state_library,
        n_pfc=sim_cfg.pfc.n_coils,
        n_sol=sim_cfg.sol.n_coils,
        split=str(args.split),
    )
    sample = library.take([int(args.library_index)])
    shot_id = str(sample.shot_ids[0])
    if shot_id != str(int(args.shot)):
        raise ValueError(f"selected library row belongs to shot {shot_id}, expected {args.shot}")
    source_index = int(sample.source_indices[0])
    ip0 = float(sample.ip0[0])
    pfc0 = np.asarray(sample.pfc0[0], dtype=float)
    sol0 = np.asarray(sample.sol0[0], dtype=float)

    reference_npz = Path(args.reference_npz)
    with np.load(reference_npz, allow_pickle=False) as data:
        files = set(data.files)
        required = {"t", "angles_rad", "radii_true"}
        missing = sorted(required - files)
        if missing:
            raise ValueError(f"{reference_npz} is missing required arrays: {', '.join(missing)}")
        source_t = np.asarray(data["t"], dtype=float).reshape(-1)
        source_angles = np.asarray(data["angles_rad"], dtype=float).reshape(-1)
        source_radii = np.asarray(data["radii_true"], dtype=float)
    if source_radii.ndim != 2 or source_radii.shape != (source_t.size, source_angles.size):
        raise ValueError(f"radii_true shape {source_radii.shape} does not match t/angles")
    if not (0 <= source_index < int(source_radii.shape[0])):
        raise IndexError(f"source_index={source_index} is outside {reference_npz}")
    if int(args.angles) != int(source_angles.size):
        raise ValueError(f"--angles={args.angles} does not match reference angles {source_angles.size}")

    limits_path = cfg.reference.ip.limits_path or (ROOT / "data/processed/t15_reference_limits.json").resolve()
    ip_cfg = IpReferenceConfig(
        kind="hold_boundary_eval_cut_profile",
        limits_path=limits_path,
        start_mode="reset_ip",
        parent_steps=int(args.parent_steps),
        segment_min_steps=int(args.segment_min_steps),
        segment_max_steps=int(args.parent_steps),
        segment_count_min=1,
        segment_count_max=3,
        hold_probability=0.45,
        ramp_rate_reference="robust_mean",
        ramp_up_rate_min_fraction=0.05,
        ramp_up_rate_fraction=0.20,
        ramp_down_rate_min_fraction=0.05,
        ramp_down_rate_fraction=0.20,
        hold_min_steps=int(args.segment_min_steps),
        hold_max_steps=int(args.parent_steps),
        final_hold_min_steps=0,
        smooth_ramps=False,
        max_delta_fraction=0.35,
    )
    cut = sample_hold_boundary_eval_cut_profile(
        ip_cfg,
        ip0,
        int(args.steps),
        np.random.default_rng(int(args.seed)),
        dt=float(cfg.reference.t_step),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hold_boundary_eval_cut900_seg300_shot{shot_id}_row{int(args.library_index):04d}_seed{int(args.seed)}"
    ref_out = out_dir / f"{stem}.npz"
    config_out = out_dir / f"T15MD_new_data_{stem}.toml"
    initial_out = out_dir / f"initial_state_{stem}.toml"
    summary_out = out_dir / f"{stem}.json"

    radii0 = np.asarray(source_radii[source_index], dtype=float).reshape(source_angles.size)
    t = np.arange(int(args.steps) + 1, dtype=float) * float(cfg.reference.t_step)
    radii = np.repeat(radii0.reshape(1, -1), int(args.steps) + 1, axis=0)
    center = (float(sim_cfg.physics.R0), float(sim_cfg.physics.Z0))
    boundary_poly = np.stack([_closed_poly(center, source_angles, row) for row in radii], axis=0)

    np.savez_compressed(
        ref_out,
        schema=np.asarray(["hold_boundary_eval_cut_profile_reference_v1"]),
        shot=np.asarray([int(shot_id)], dtype=np.int64),
        source_index=np.asarray([source_index], dtype=np.int64),
        source_time_s=np.asarray([float(sample.source_times_s[0])], dtype=float),
        library_index=np.asarray([int(args.library_index)], dtype=np.int64),
        seed=np.asarray([int(args.seed)], dtype=np.int64),
        parent_steps=np.asarray([int(args.parent_steps)], dtype=np.int64),
        cut_start_step=np.asarray([int(cut.cut_start_step)], dtype=np.int64),
        step=np.arange(1, int(args.steps) + 2, dtype=np.int64),
        t=t,
        angles_rad=source_angles,
        Ip_ref=np.asarray(cut.ip, dtype=float),
        Ip=np.asarray(cut.ip, dtype=float),
        parent_Ip_ref=np.asarray(cut.parent_ip, dtype=float),
        radii_true=radii,
        radii_ref=radii,
        initial_radii_true=radii0,
        boundary_found=np.ones((int(args.steps) + 1,), dtype=bool),
        boundary_poly_true=boundary_poly,
    )

    pfc = CoilGroup(name=sim_cfg.pfc.name, coils=list(sim_cfg.pfc.coils), currents=pfc0)
    sol = CoilGroup(name=sim_cfg.sol.name, coils=list(sim_cfg.sol.coils), currents=sol0)
    dump_config(
        config_out,
        grid=sim_cfg.grid,
        pfc=pfc,
        sol=sol,
        physics=sim_cfg.physics,
        compute=sim_cfg.compute,
        realism=sim_cfg.realism,
        limiter_name=sim_cfg.limiter_name,
        boundary_mode=sim_cfg.boundary_mode,
        boundary_base_mode=sim_cfg.boundary_base_mode,
        boundary_legacy_precision_index2=sim_cfg.boundary_legacy_precision_index2,
        boundary_track_level=sim_cfg.boundary_track_level,
        boundary_smooth_selected_level=sim_cfg.boundary_smooth_selected_level,
        boundary_soft_level_selection=sim_cfg.boundary_soft_level_selection,
        boundary_soft_level_candidates=sim_cfg.boundary_soft_level_candidates,
        boundary_soft_level_temperature=sim_cfg.boundary_soft_level_temperature,
        boundary_soft_level_radius_weight=sim_cfg.boundary_soft_level_radius_weight,
        boundary_soft_level_missing_penalty=sim_cfg.boundary_soft_level_missing_penalty,
        boundary_soft_level_roughness_penalty=sim_cfg.boundary_soft_level_roughness_penalty,
        boundary_level_smoothing_alpha=sim_cfg.boundary_level_smoothing_alpha,
        boundary_level_search_span_fraction=sim_cfg.boundary_level_search_span_fraction,
        boundary_continuity_weight_radii=sim_cfg.boundary_continuity_weight_radii,
        boundary_continuity_weight_mean_radius=sim_cfg.boundary_continuity_weight_mean_radius,
        boundary_continuity_weight_center=sim_cfg.boundary_continuity_weight_center,
        boundary_continuity_weight_area=sim_cfg.boundary_continuity_weight_area,
        boundary_continuity_weight_level=sim_cfg.boundary_continuity_weight_level,
    )
    _write_initial_state(initial_out, ip0=ip0, pfc0=pfc0, sol0=sol0)

    summary = {
        "reference_npz": str(ref_out),
        "config": str(config_out),
        "initial_state": str(initial_out),
        "shot": shot_id,
        "library_index": int(args.library_index),
        "source_index": source_index,
        "source_time_s": float(sample.source_times_s[0]),
        "ip0": ip0,
        "steps": int(args.steps),
        "parent_steps": int(args.parent_steps),
        "cut_start_step": int(cut.cut_start_step),
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _ensure_paths() -> None:
    sim_root = ROOT.parent / "tokamak-sim"
    for path in (ROOT, sim_root):
        p = str(path.resolve())
        if p not in sys.path:
            sys.path.insert(0, p)


def _closed_poly(center: tuple[float, float], angles: np.ndarray, radii: np.ndarray) -> np.ndarray:
    r0, z0 = center
    poly = np.column_stack((r0 + radii * np.cos(angles), z0 + radii * np.sin(angles)))
    return np.vstack((poly, poly[0]))


def _write_initial_state(path: Path, *, ip0: float, pfc0: np.ndarray, sol0: np.ndarray) -> None:
    def arr(values: np.ndarray) -> str:
        return "[\n" + "".join(f"    {float(v):.17g},\n" for v in np.asarray(values, dtype=float).reshape(-1)) + "]"

    text = (
        "version = 1\n\n"
        "[plasma]\n"
        f"Ip0 = {float(ip0):.17g}\n\n"
        "[coils.pfc]\n"
        f"currents = {arr(pfc0)}\n\n"
        "[coils.sol]\n"
        f"currents = {arr(sol0)}\n"
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
