from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class CsvInitialStateSample:
    ip0: np.ndarray
    pfc0: np.ndarray
    sol0: np.ndarray
    params0: np.ndarray | None
    shot_ids: tuple[str, ...]
    source_indices: tuple[int, ...]
    source_times_s: tuple[float, ...]
    difficulty_bins: tuple[str, ...] = ()


class CsvInitialStateLibrary:
    """Processed coherent reset states extracted from real T15 CSV traces."""

    def __init__(self, path: str | Path, *, n_pfc: int, n_sol: int, split: str = "train") -> None:
        self.path = Path(path)
        split = str(split)
        if split not in {"train", "holdout", "all"}:
            raise ValueError("CSV initial-state split must be train, holdout, or all")
        if not self.path.exists():
            raise FileNotFoundError(f"CSV initial-state library does not exist: {self.path}")
        with np.load(self.path, allow_pickle=False) as data:
            required = {"shot_id", "source_index", "time_s", "ip0", "pfc0", "sol0"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"CSV initial-state library missing arrays: {', '.join(missing)}")
            self.shot_id = np.asarray(data["shot_id"]).astype(str)
            self.source_index = np.asarray(data["source_index"], dtype=np.int64).reshape(-1)
            self.time_s = np.asarray(data["time_s"], dtype=float).reshape(-1)
            self.ip0 = np.asarray(data["ip0"], dtype=float).reshape(-1)
            self.pfc0 = np.asarray(data["pfc0"], dtype=float)
            self.sol0 = np.asarray(data["sol0"], dtype=float)
            raw_params0 = np.asarray(data["params0"], dtype=float) if "params0" in data.files else None
            raw_split = np.asarray(data["split"]).astype(str) if "split" in data.files else None
            raw_difficulty = np.asarray(data["difficulty_bin"]).astype(str) if "difficulty_bin" in data.files else None
        if raw_split is None and split != "all":
            raise ValueError(f"CSV initial-state library missing split array required for split={split!r}: {self.path}")
        if raw_split is not None:
            raw_split = raw_split.reshape(-1)
            if raw_split.shape != self.ip0.shape:
                raise ValueError("CSV initial-state split array must have one value per row")
            unknown = sorted(set(raw_split.tolist()) - {"train", "holdout"})
            if unknown:
                raise ValueError("CSV initial-state split array contains unsupported values: " + ", ".join(unknown))
            if split != "all":
                keep = raw_split == split
                self.shot_id = self.shot_id[keep]
                self.source_index = self.source_index[keep]
                self.time_s = self.time_s[keep]
                self.ip0 = self.ip0[keep]
                self.pfc0 = self.pfc0[keep]
                self.sol0 = self.sol0[keep]
                if raw_params0 is not None:
                    raw_params0 = raw_params0[keep]
                if raw_difficulty is not None:
                    raw_difficulty = raw_difficulty.reshape(-1)[keep]
                raw_split = raw_split[keep]
            self.split = raw_split.astype(str)
        else:
            self.split = np.full(self.ip0.shape, "all", dtype="<U7")
        if raw_difficulty is not None:
            raw_difficulty = raw_difficulty.reshape(-1)
            if raw_difficulty.shape != self.ip0.shape:
                raise ValueError("CSV initial-state difficulty_bin array must have one value per row")
            if np.any(raw_difficulty == ""):
                raise ValueError("CSV initial-state difficulty_bin contains empty labels")
            self.difficulty_bin = raw_difficulty.astype(str)
        else:
            self.difficulty_bin = None
        count = int(self.ip0.shape[0])
        if count <= 0:
            raise ValueError(f"CSV initial-state library split={split!r} is empty: {self.path}")
        if self.shot_id.shape != (count,) or self.source_index.shape != (count,) or self.time_s.shape != (count,):
            raise ValueError("CSV initial-state metadata arrays must have one value per row")
        if self.pfc0.shape != (count, int(n_pfc)):
            raise ValueError(f"CSV initial-state PFC shape must be ({count}, {int(n_pfc)}), got {self.pfc0.shape}")
        if self.sol0.shape != (count, int(n_sol)):
            raise ValueError(f"CSV initial-state SOL shape must be ({count}, {int(n_sol)}), got {self.sol0.shape}")
        self.params0 = None
        if raw_params0 is not None:
            raw_params0 = np.asarray(raw_params0, dtype=float)
            if raw_params0.shape != (count, 5):
                raise ValueError(f"CSV initial-state params0 shape must be ({count}, 5), got {raw_params0.shape}")
            self.params0 = raw_params0
        for name, arr in (("time_s", self.time_s), ("ip0", self.ip0), ("pfc0", self.pfc0), ("sol0", self.sol0), ("params0", self.params0)):
            if arr is None:
                continue
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"CSV initial-state library contains non-finite {name}")
        self._bin_shot_indices: dict[str, dict[str, np.ndarray]] = {}
        if self.difficulty_bin is not None:
            for bin_name in sorted(set(self.difficulty_bin.tolist())):
                mask_bin = self.difficulty_bin == bin_name
                shots: dict[str, np.ndarray] = {}
                for shot in sorted(set(self.shot_id[mask_bin].tolist()), key=lambda value: int(value)):
                    idx = np.nonzero(mask_bin & (self.shot_id == shot))[0].astype(np.int64)
                    if idx.size:
                        shots[str(shot)] = idx
                if shots:
                    self._bin_shot_indices[str(bin_name)] = shots

    def __len__(self) -> int:
        return int(self.ip0.shape[0])

    def sample(self, rng: np.random.Generator, count: int) -> CsvInitialStateSample:
        if self.difficulty_bin is not None and self._bin_shot_indices:
            bins = np.asarray(sorted(self._bin_shot_indices), dtype=object)
            reps = int(np.ceil(int(count) / max(int(bins.size), 1)))
            chosen_bins = np.tile(bins, reps)[: int(count)]
            rng.shuffle(chosen_bins)
            idx = np.empty((int(count),), dtype=np.int64)
            for row, bin_name in enumerate(chosen_bins.tolist()):
                by_shot = self._bin_shot_indices[str(bin_name)]
                shots = sorted(by_shot)
                shot = str(shots[int(rng.integers(0, len(shots)))])
                candidates = by_shot[shot]
                idx[row] = int(candidates[int(rng.integers(0, int(candidates.size)))])
            return self.take(idx)
        idx = rng.integers(0, len(self), size=int(count))
        return self.take(idx)

    def take(self, indices: np.ndarray | list[int] | tuple[int, ...]) -> CsvInitialStateSample:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError("CSV initial-state take() requires at least one index")
        if np.any((idx < 0) | (idx >= len(self))):
            raise IndexError("CSV initial-state index out of range")
        return CsvInitialStateSample(
            ip0=self.ip0[idx].astype(float, copy=True),
            pfc0=self.pfc0[idx].astype(float, copy=True),
            sol0=self.sol0[idx].astype(float, copy=True),
            params0=None if self.params0 is None else self.params0[idx].astype(float, copy=True),
            shot_ids=tuple(str(v) for v in self.shot_id[idx].tolist()),
            source_indices=tuple(int(v) for v in self.source_index[idx].tolist()),
            source_times_s=tuple(float(v) for v in self.time_s[idx].tolist()),
            difficulty_bins=tuple(str(v) for v in self.difficulty_bin[idx].tolist()) if self.difficulty_bin is not None else (),
        )


def validate_split_nonoverlap(shot_id: np.ndarray, time_s: np.ndarray, split: np.ndarray, *, min_gap_s: float) -> None:
    gap = float(min_gap_s)
    if not np.isfinite(gap) or gap <= 0.0:
        raise ValueError("split non-overlap gap must be finite and positive")
    shot_arr = np.asarray(shot_id).astype(str).reshape(-1)
    time_arr = np.asarray(time_s, dtype=float).reshape(-1)
    split_arr = np.asarray(split).astype(str).reshape(-1)
    if shot_arr.shape != time_arr.shape or shot_arr.shape != split_arr.shape:
        raise ValueError("split non-overlap arrays must have matching one-dimensional shapes")
    for shot in sorted(set(shot_arr.tolist())):
        mask = shot_arr == str(shot)
        shot_times = time_arr[mask]
        shot_splits = split_arr[mask]
        train_times = np.sort(shot_times[shot_splits == "train"])
        holdout_times = np.sort(shot_times[shot_splits == "holdout"])
        if train_times.size == 0 or holdout_times.size == 0:
            continue
        for source, other, source_name, other_name in (
            (train_times, holdout_times, "train", "holdout"),
            (holdout_times, train_times, "holdout", "train"),
        ):
            indices = np.searchsorted(other, source, side="left")
            for pos, idx in enumerate(indices.tolist()):
                current = float(source[pos])
                for neighbor in (idx - 1, idx):
                    if 0 <= int(neighbor) < int(other.size):
                        distance = abs(current - float(other[int(neighbor)]))
                        if distance <= gap:
                            raise ValueError(
                                f"split rows overlap within one episode for shot {shot}: "
                                f"{source_name}@{current:.6f}s vs {other_name}@{float(other[int(neighbor)]):.6f}s "
                                f"(gap {distance:.6f}s <= {gap:.6f}s)"
                            )


__all__ = ["CsvInitialStateLibrary", "CsvInitialStateSample", "validate_split_nonoverlap"]
