"""Загрузка oracle-окон из заранее построенных T15 replay-артефактов."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_oracle_from_replay(
    candidates: list[object],
    *,
    replay_root: Path,
    angles: int,
    mean_ip_limit: float,
    max_ip_limit: float,
    n_pfc: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Собрать oracle-окна из полных replay NPZ без повторной симуляции."""
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    cache: dict[str, dict[str, np.ndarray]] = {}

    for candidate in candidates:
        shot = str(candidate.shot_id)
        if shot not in cache:
            loaded = _load_shot_replay(replay_root, shot)
            if loaded is None:
                rejected.append(_reject(candidate, "no_replay_npz"))
                continue
            cache[shot] = loaded

        replay = cache[shot]
        start = int(candidate.source_index)
        steps = int(candidate.ip_target.shape[0])
        end = start + steps
        if end > int(replay["t"].shape[0]):
            rejected.append(_reject(candidate, f"window_out_of_bounds_{start}+{steps}"))
            continue

        window_ip = replay["Ip"][start:end]
        window_radii = replay["radii_true"][start:end]
        window_found = replay["boundary_found"][start:end]
        if not np.all(np.isfinite(window_ip)):
            rejected.append(_reject(candidate, "non_finite_ip"))
            continue
        if not bool(np.all(window_found)):
            rejected.append(_reject(candidate, "boundary_not_found"))
            continue
        if window_radii.ndim != 2 or int(window_radii.shape[1]) != int(angles):
            rejected.append(_reject(candidate, "wrong_boundary_angle_count"))
            continue
        if not np.all(np.isfinite(window_radii)):
            rejected.append(_reject(candidate, "non_finite_boundary_radii"))
            continue

        ip_error = np.abs(window_ip - np.asarray(candidate.ip_target, dtype=np.float64))
        mean_error = float(np.mean(ip_error))
        max_error = float(np.max(ip_error))
        if mean_error > float(mean_ip_limit):
            rejected.append(_reject(candidate, f"oracle_mean_ip_{mean_error:.0f}A"))
            continue
        if max_error > float(max_ip_limit):
            rejected.append(_reject(candidate, f"oracle_max_ip_{max_error:.0f}A"))
            continue

        accepted.append(
            {
                "shot_id": shot,
                "split": candidate.split,
                "source_index": start,
                "time_s": float(replay["t"][start]),
                "difficulty_bin": candidate.difficulty_bin,
                "ip0": float(candidate.ip_target[0]),
                "pfc0": np.asarray(candidate.currents[0, :n_pfc], dtype=np.float32),
                "sol0": np.asarray(candidate.currents[0, n_pfc:], dtype=np.float32),
                "ip_target": np.asarray(candidate.ip_target, dtype=np.float32),
                "boundary_radii": np.asarray(window_radii, dtype=np.float32),
                "real_jdot_action": np.asarray(candidate.normalized_action, dtype=np.float32),
                "oracle_ip_mean_error_a": np.float32(mean_error),
                "oracle_ip_max_error_a": np.float32(max_error),
            }
        )

    return accepted, rejected


def _load_shot_replay(replay_root: Path, shot: str) -> dict[str, np.ndarray] | None:
    """Загрузить канонический replay одного разряда."""
    reference_path = replay_root / f"lqr_boundary_reference_{shot}.npz"
    if reference_path.exists():
        return _normalize_replay_arrays(reference_path)

    replay_dir = replay_root / f"t15md_limited_replay_{shot}"
    candidates = sorted(replay_dir.glob("*/run*.npz")) if replay_dir.exists() else []
    if not candidates:
        return None
    return _normalize_replay_arrays(candidates[-1])


def _normalize_replay_arrays(path: Path) -> dict[str, np.ndarray]:
    """Привести поля replay NPZ к единому контракту oracle builder."""
    with np.load(path, allow_pickle=False) as payload:
        available = set(payload.files)
        ip_key = "Ip" if "Ip" in available else "Ip_ref"
        radii_key = "radii_true" if "radii_true" in available else "radii_ref"
        required = {"t", ip_key, radii_key}
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"Replay {path} is missing fields: {missing}")
        time = np.asarray(payload["t"], dtype=np.float64).reshape(-1)
        ip = np.asarray(payload[ip_key], dtype=np.float64).reshape(-1)
        radii = np.asarray(payload[radii_key], dtype=np.float64)
        found = (
            np.asarray(payload["boundary_found"], dtype=bool).reshape(-1)
            if "boundary_found" in available
            else np.all(np.isfinite(radii), axis=1)
        )
    if time.shape != ip.shape or time.shape != found.shape:
        raise ValueError(f"Replay {path} has inconsistent time-series lengths")
    if radii.ndim != 2 or int(radii.shape[0]) != int(time.shape[0]):
        raise ValueError(f"Replay {path} has invalid radii_true shape {radii.shape}")
    return {"t": time, "Ip": ip, "radii_true": radii, "boundary_found": found}


def _reject(candidate: object, reason: str) -> dict[str, object]:
    """Сформировать запись об отклонённом oracle-окне."""
    return {
        "shot_id": str(candidate.shot_id),
        "source_index": int(candidate.source_index),
        "time_s": float(candidate.time_s),
        "reason": str(reason),
    }
