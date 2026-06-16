from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np

from tokamak_rl_v2.env.references import PARAMETER_ORDER

AVAILABLE_REPLAY_START_SHOT_IDS: tuple[str, ...] = ("3854", "3855", "3856", "3857", "3858", "3859", "3862", "3863", "3864")
MAIN_7_REPLAY_START_SHOT_IDS: tuple[str, ...] = ("3856", "3857", "3858", "3859", "3862", "3863", "3864")


@dataclass(frozen=True, slots=True)
class ReplayStartStateRow:
    shot_id: str
    ip0: float
    pfc0: np.ndarray
    sol0: np.ndarray
    params0: np.ndarray


@dataclass(frozen=True, slots=True)
class ReplayStartStateSample:
    ip0: np.ndarray
    pfc0: np.ndarray
    sol0: np.ndarray
    params0: np.ndarray
    shot_ids: tuple[str, ...]


class ReplayStartStateLibrary:
    """Load coherent replay-start reset rows from the real T15 replay-start set."""

    def __init__(
        self,
        config_path: Path,
        *,
        n_pfc: int,
        n_sol: int,
        shot_ids: tuple[str, ...] | None = None,
    ) -> None:
        sim_root = Path(config_path).resolve().parents[1]
        initial_root = sim_root / "configs" / "initial_currents"
        boundary_root = sim_root / "output" / "t15_boundary_parameters"
        chosen_shot_ids = _validated_shot_ids(shot_ids)
        rows: list[ReplayStartStateRow] = []
        for shot_id in chosen_shot_ids:
            initial_path = initial_root / f"T15MD_new_data_{shot_id}.toml"
            boundary_path = boundary_root / f"simulated_replay_{shot_id}_boundary_params.csv"
            if not initial_path.exists():
                raise FileNotFoundError(f"missing replay-start currents file: {initial_path}")
            if not boundary_path.exists():
                raise FileNotFoundError(f"missing replay-start boundary file: {boundary_path}")
            pfc0, sol0 = _load_initial_currents(initial_path)
            if pfc0.shape != (int(n_pfc),):
                raise ValueError(f"replay-start shot {shot_id} has {pfc0.shape[0]} PFC currents, expected {int(n_pfc)}")
            if sol0.shape != (int(n_sol),):
                raise ValueError(f"replay-start shot {shot_id} has {sol0.shape[0]} SOL currents, expected {int(n_sol)}")
            ip0, params0 = _load_first_valid_boundary_row(boundary_path)
            rows.append(ReplayStartStateRow(shot_id=shot_id, ip0=ip0, pfc0=pfc0, sol0=sol0, params0=params0))
        self.rows = tuple(rows)

    def sample(self, rng: np.random.Generator, *, count: int) -> ReplayStartStateSample:
        count_i = int(count)
        if count_i <= 0:
            raise ValueError("count must be positive")
        idx = rng.integers(0, len(self.rows), size=(count_i,))
        chosen = [self.rows[int(i)] for i in idx]
        return ReplayStartStateSample(
            ip0=np.asarray([row.ip0 for row in chosen], dtype=float),
            pfc0=np.stack([row.pfc0 for row in chosen], axis=0).astype(float, copy=False),
            sol0=np.stack([row.sol0 for row in chosen], axis=0).astype(float, copy=False),
            params0=np.stack([row.params0 for row in chosen], axis=0).astype(float, copy=False),
            shot_ids=tuple(row.shot_id for row in chosen),
        )


def _load_initial_currents(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    coils = raw["coils"]
    pfc = np.asarray(coils["pfc"]["currents"], dtype=float).reshape(-1)
    sol = np.asarray(coils["sol"]["currents"], dtype=float).reshape(-1)
    if not np.all(np.isfinite(pfc)) or not np.all(np.isfinite(sol)):
        raise ValueError(f"non-finite replay-start currents in {path}")
    return pfc, sol


def _load_first_valid_boundary_row(path: Path) -> tuple[float, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("fit_status", "")).strip().lower() != "ok":
                continue
            values = [float(row["Ip"])] + [float(row[name]) for name in PARAMETER_ORDER]
            if not all(np.isfinite(values)):
                continue
            return float(values[0]), np.asarray(values[1:], dtype=float)
    raise ValueError(f"could not find a valid replay-start boundary row in {path}")


def _validated_shot_ids(shot_ids: tuple[str, ...] | None) -> tuple[str, ...]:
    chosen = AVAILABLE_REPLAY_START_SHOT_IDS if shot_ids is None else tuple(str(shot_id) for shot_id in shot_ids)
    if not chosen:
        raise ValueError("replay-start shot_ids must not be empty")
    unknown = sorted(set(chosen) - set(AVAILABLE_REPLAY_START_SHOT_IDS))
    if unknown:
        raise ValueError("unknown replay-start shot_ids: " + ", ".join(unknown))
    return chosen


__all__ = [
    "AVAILABLE_REPLAY_START_SHOT_IDS",
    "MAIN_7_REPLAY_START_SHOT_IDS",
    "ReplayStartStateLibrary",
    "ReplayStartStateRow",
    "ReplayStartStateSample",
]
