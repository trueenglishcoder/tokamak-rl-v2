from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class T15ReferenceLimits:
    ip_min_a: float
    ip_max_a: float
    ip_p01_a: float
    ip_p99_a: float
    positive_dipdt_p95_a_per_s: float
    positive_dipdt_p99_a_per_s: float
    negative_dipdt_abs_p95_a_per_s: float
    negative_dipdt_abs_p99_a_per_s: float
    positive_ramp_mean_a_per_s: float | None
    negative_ramp_abs_mean_a_per_s: float | None
    sample_count: int
    shot_count: int

    @property
    def ip_width_a(self) -> float:
        return float(self.ip_p99_a) - float(self.ip_p01_a)


@lru_cache(maxsize=16)
def load_reference_limits(path: str | Path) -> T15ReferenceLimits:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    positive_key = "positive_dipdt_p95_a_per_s" if "positive_dipdt_p95_a_per_s" in raw else "positive_dip_dt_p95_a_per_s"
    negative_key = "negative_dipdt_abs_p95_a_per_s" if "negative_dipdt_abs_p95_a_per_s" in raw else "negative_dip_dt_abs_p95_a_per_s"
    positive_mean = raw.get("positive_ramp_mean_a_per_s", raw.get("positive_dipdt_ramp_mean_a_per_s"))
    negative_mean = raw.get("negative_ramp_abs_mean_a_per_s", raw.get("negative_dipdt_abs_ramp_mean_a_per_s"))
    limits = T15ReferenceLimits(
        ip_min_a=float(raw.get("ip_min_a", raw["ip_p01_a"])),
        ip_max_a=float(raw.get("ip_max_a", raw["ip_p99_a"])),
        ip_p01_a=float(raw["ip_p01_a"]),
        ip_p99_a=float(raw["ip_p99_a"]),
        positive_dipdt_p95_a_per_s=float(raw[positive_key]),
        positive_dipdt_p99_a_per_s=float(raw.get("positive_dipdt_p99_a_per_s", raw.get("positive_dip_dt_p99_a_per_s", raw[positive_key]))),
        negative_dipdt_abs_p95_a_per_s=float(raw[negative_key]),
        negative_dipdt_abs_p99_a_per_s=float(raw.get("negative_dipdt_abs_p99_a_per_s", raw.get("negative_dip_dt_abs_p99_a_per_s", raw[negative_key]))),
        positive_ramp_mean_a_per_s=None if positive_mean is None else float(positive_mean),
        negative_ramp_abs_mean_a_per_s=None if negative_mean is None else float(negative_mean),
        sample_count=int(raw.get("sample_count", 0)),
        shot_count=int(raw.get("shot_count", 0)),
    )
    if not (limits.ip_max_a >= limits.ip_p99_a > limits.ip_p01_a >= limits.ip_min_a > 0.0):
        raise ValueError(f"invalid Ip bounds in reference limits: {source}")
    if limits.positive_dipdt_p95_a_per_s <= 0.0 or limits.negative_dipdt_abs_p95_a_per_s <= 0.0:
        raise ValueError(f"invalid Ip ramp-rate limits: {source}")
    if limits.positive_ramp_mean_a_per_s is not None and limits.positive_ramp_mean_a_per_s <= 0.0:
        raise ValueError(f"invalid positive ramp mean in reference limits: {source}")
    if limits.negative_ramp_abs_mean_a_per_s is not None and limits.negative_ramp_abs_mean_a_per_s <= 0.0:
        raise ValueError(f"invalid negative ramp mean in reference limits: {source}")
    if limits.sample_count <= 0 or limits.shot_count <= 0:
        raise ValueError(f"reference limits contain no source data: {source}")
    return limits


__all__ = ["T15ReferenceLimits", "load_reference_limits"]
