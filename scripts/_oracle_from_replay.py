"""Загрузка oracle-окон из canonical endpoint-aligned T15 replay references."""

from __future__ import annotations

from pathlib import Path

import numpy as np


REPLAY_REFERENCE_SCHEMA = "t15md_replay_reference_v2"
PASSIVE_STATE_SIZE = 303


def load_oracle_from_replay(
    candidates: list[object],
    *,
    replay_root: Path,
    angles: int,
    n_pfc: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Собрать oracle-окна и exact reset states из fresh canonical replay."""
    accepted: list[dict[str, object]] = []
    cache: dict[str, dict[str, np.ndarray]] = {}

    for candidate in candidates:
        shot = str(candidate.shot_id)
        if shot not in cache:
            cache[shot] = _load_shot_replay(replay_root, shot)
        replay = cache[shot]

        start = int(candidate.source_index)
        endpoint_count = int(np.asarray(candidate.ip_target).shape[0])
        end = start + endpoint_count
        if start < 0 or end > int(replay["t"].shape[0]):
            raise RuntimeError(
                f"shot {shot}: oracle window [{start}:{end}] exceeds endpoint-aligned replay length "
                f"{int(replay['t'].shape[0])}"
            )
        if int(replay["source_index"][start]) != start:
            raise RuntimeError(
                f"shot {shot}: replay source_index row {start} is {int(replay['source_index'][start])}, expected {start}"
            )
        if not np.isclose(float(replay["t"][start]), float(candidate.time_s), rtol=0.0, atol=1.0e-9):
            raise RuntimeError(
                f"shot {shot}: source_index={start} time mismatch replay={float(replay['t'][start]):.9f}s "
                f"candidate={float(candidate.time_s):.9f}s"
            )

        window_ip = np.asarray(replay["Ip"][start:end], dtype=float)
        window_radii = np.asarray(replay["radii_true"][start:end], dtype=float)
        window_found = np.asarray(replay["boundary_found"][start:end], dtype=bool)
        window_projection_valid = np.asarray(
            replay["boundary_fixed_angle_valid"][start:end], dtype=bool
        )
        window_pfc = np.asarray(replay["pfc_currents"][start:end], dtype=float)
        window_sol = np.asarray(replay["sol_currents"][start:end], dtype=float)
        window_currents = np.concatenate((window_pfc, window_sol), axis=1)
        expected_currents = np.asarray(candidate.currents, dtype=float)
        expected_sol = int(expected_currents.shape[1]) - int(n_pfc) if expected_currents.ndim == 2 else -1

        if window_ip.shape != np.asarray(candidate.ip_target).shape or not np.all(np.isfinite(window_ip)):
            raise RuntimeError(f"shot {shot}: replay Ip window is invalid at source_index={start}")
        if window_radii.shape != (endpoint_count, int(angles)):
            raise RuntimeError(
                f"shot {shot}: expected replay boundary shape {(endpoint_count, int(angles))}, got {window_radii.shape}"
            )
        if not np.all(window_found):
            bad_steps = (start + np.flatnonzero(~window_found)).tolist()
            raise RuntimeError(
                f"shot {shot}: boundary extractor failed at replay source endpoints {bad_steps[:20]}"
            )
        if not np.all(window_projection_valid):
            bad_steps = (start + np.flatnonzero(~window_projection_valid)).tolist()
            raise RuntimeError(
                f"shot {shot}: fixed-angle projection invalid at replay source endpoints {bad_steps[:20]}"
            )
        bad_radii = ~np.isfinite(window_radii) | (window_radii <= 0.0)
        if bool(np.any(bad_radii)):
            preview = [
                (int(start + row), int(angle_index))
                for row, angle_index in np.argwhere(bad_radii)[:20]
            ]
            raise RuntimeError(
                f"shot {shot}: invalid fixed-angle boundary radii at (source_index, angle) {preview}"
            )
        if window_pfc.shape != (endpoint_count, int(n_pfc)) or expected_sol <= 0 or window_sol.shape != (endpoint_count, expected_sol):
            raise RuntimeError(
                f"shot {shot}: replay coil shapes do not match candidate contract "
                f"pfc={window_pfc.shape} sol={window_sol.shape} expected_pfc={int(n_pfc)} expected_sol={expected_sol}"
            )
        if window_currents.shape != expected_currents.shape or not np.allclose(
            window_currents,
            expected_currents,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise RuntimeError(
                f"shot {shot}: replay/source coil endpoint mismatch in window starting at source_index={start}"
            )

        hidden_state_a = float(replay["hidden_state_a"][start])
        passive_currents_a = np.asarray(replay["passive_currents_a"][start], dtype=float)
        if not np.isfinite(hidden_state_a) or not np.all(np.isfinite(passive_currents_a)):
            raise RuntimeError(f"shot {shot}: non-finite canonical reset state at source_index={start}")

        ip_error = np.abs(window_ip - np.asarray(candidate.ip_target, dtype=np.float64))
        mean_error = float(np.mean(ip_error))
        max_error = float(np.max(ip_error))
        accepted.append(
            {
                "shot_id": shot,
                "split": candidate.split,
                "source_index": start,
                "time_s": float(replay["t"][start]),
                "difficulty_bin": candidate.difficulty_bin,
                "ip0": float(window_ip[0]),
                "pfc0": np.asarray(window_pfc[0], dtype=np.float32),
                "sol0": np.asarray(window_sol[0], dtype=np.float32),
                "hidden_state_a": np.float64(hidden_state_a),
                "passive_currents_a": passive_currents_a.astype(np.float64, copy=True),
                "ip_target": np.asarray(candidate.ip_target, dtype=np.float32),
                "boundary_radii": window_radii.astype(np.float32, copy=True),
                "real_jdot_action": np.asarray(candidate.normalized_action, dtype=np.float32),
                "oracle_ip_mean_error_a": np.float32(mean_error),
                "oracle_ip_max_error_a": np.float32(max_error),
            }
        )

    return accepted, []


def _load_shot_replay(replay_root: Path, shot: str) -> dict[str, np.ndarray]:
    """Загрузить fresh endpoint-aligned replay reference одного разряда."""
    reference_path = replay_root / f"lqr_boundary_reference_{shot}.npz"
    if not reference_path.exists():
        raise RuntimeError(f"missing canonical replay reference for shot {shot}: {reference_path}")
    return _normalize_replay_arrays(reference_path)


def _normalize_replay_arrays(path: Path) -> dict[str, np.ndarray]:
    """Проверить schema и формы endpoint-aligned replay reference v2."""
    required = {
        "schema",
        "source_index",
        "t",
        "Ip",
        "radii_true",
        "boundary_found",
        "boundary_fixed_angle_valid",
        "pfc_currents",
        "sol_currents",
        "hidden_state_a",
        "passive_currents_a",
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"{path}: missing canonical replay reference fields {missing}")
        schema_values = np.asarray(payload["schema"]).astype(str).reshape(-1)
        if schema_values.shape != (1,) or str(schema_values[0]) != REPLAY_REFERENCE_SCHEMA:
            raise RuntimeError(
                f"{path}: expected replay schema {REPLAY_REFERENCE_SCHEMA!r}, got {schema_values.tolist()}"
            )
        arrays = {name: np.asarray(payload[name]) for name in required if name != "schema"}

    time = np.asarray(arrays["t"], dtype=float).reshape(-1)
    count = int(time.size)
    source_index = np.asarray(arrays["source_index"], dtype=np.int64).reshape(-1)
    ip = np.asarray(arrays["Ip"], dtype=float).reshape(-1)
    radii = np.asarray(arrays["radii_true"], dtype=float)
    found = np.asarray(arrays["boundary_found"], dtype=bool).reshape(-1)
    projection_valid = np.asarray(arrays["boundary_fixed_angle_valid"], dtype=bool).reshape(-1)
    pfc = np.asarray(arrays["pfc_currents"], dtype=float)
    sol = np.asarray(arrays["sol_currents"], dtype=float)
    hidden = np.asarray(arrays["hidden_state_a"], dtype=float).reshape(-1)
    passive = np.asarray(arrays["passive_currents_a"], dtype=float)

    if count <= 1 or np.any(np.diff(time) <= 0.0):
        raise RuntimeError(f"{path}: replay endpoint times must be strictly increasing")
    if not np.array_equal(source_index, np.arange(count, dtype=np.int64)):
        raise RuntimeError(f"{path}: source_index must equal row index for every replay endpoint")
    if ip.shape != (count,) or found.shape != (count,) or projection_valid.shape != (count,) or hidden.shape != (count,):
        raise RuntimeError(f"{path}: scalar replay arrays do not share endpoint length {count}")
    if radii.ndim != 2 or radii.shape[0] != count:
        raise RuntimeError(f"{path}: radii_true must have shape (N_endpoints, angles), got {radii.shape}")
    if pfc.ndim != 2 or pfc.shape[0] != count or sol.ndim != 2 or sol.shape[0] != count:
        raise RuntimeError(f"{path}: coil current arrays do not share endpoint length {count}")
    if passive.shape != (count, PASSIVE_STATE_SIZE):
        raise RuntimeError(
            f"{path}: passive_currents_a must have shape (N_endpoints, {PASSIVE_STATE_SIZE}), got {passive.shape}"
        )
    for name, array in (
        ("t", time),
        ("Ip", ip),
        ("radii_true", radii),
        ("pfc_currents", pfc),
        ("sol_currents", sol),
        ("hidden_state_a", hidden),
        ("passive_currents_a", passive),
    ):
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{path}: {name} contains non-finite values")

    return {
        "source_index": source_index,
        "t": time,
        "Ip": ip,
        "radii_true": radii,
        "boundary_found": found,
        "boundary_fixed_angle_valid": projection_valid,
        "pfc_currents": pfc,
        "sol_currents": sol,
        "hidden_state_a": hidden,
        "passive_currents_a": passive,
    }
