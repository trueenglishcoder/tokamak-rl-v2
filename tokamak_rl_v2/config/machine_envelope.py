from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import math

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True, slots=True)
class ProxyTrainingEnvelope:
    ip_range_a: tuple[float, float]
    ip_rate_a_per_s: float
    boundary_margin_m: float
    boundary_rate_m_per_s: float
    coil_current_range_a: tuple[tuple[float, float], ...]
    coil_jdot_range_a_per_s: tuple[tuple[float, float], ...]

    def validate(self) -> None:
        _validate_range(self.ip_range_a, "proxy_training.ip_range_a")
        _validate_positive(self.ip_rate_a_per_s, "proxy_training.ip_rate_a_per_s")
        _validate_nonnegative(self.boundary_margin_m, "proxy_training.boundary_margin_m")
        _validate_positive(self.boundary_rate_m_per_s, "proxy_training.boundary_rate_m_per_s")
        if not self.coil_current_range_a:
            raise ValueError("proxy_training.coil_current_range_a must not be empty")
        if len(self.coil_jdot_range_a_per_s) != len(self.coil_current_range_a):
            raise ValueError("proxy_training.coil_jdot_range_a_per_s must match coil_current_range_a length")
        for i, value in enumerate(self.coil_current_range_a):
            _validate_range(value, f"proxy_training.coil_current_range_a[{i}]")
        for i, value in enumerate(self.coil_jdot_range_a_per_s):
            _validate_range(value, f"proxy_training.coil_jdot_range_a_per_s[{i}]")


@dataclass(frozen=True, slots=True)
class SoftPenaltyEnvelope:
    ip_soft_range_a: tuple[float, float]
    coil_current_soft_fraction: float
    coil_jdot_soft_fraction: float

    def validate(self) -> None:
        _validate_range(self.ip_soft_range_a, "soft_penalties.ip_soft_range_a")
        _validate_positive(self.coil_current_soft_fraction, "soft_penalties.coil_current_soft_fraction")
        _validate_positive(self.coil_jdot_soft_fraction, "soft_penalties.coil_jdot_soft_fraction")


@dataclass(frozen=True, slots=True)
class TerminationEnvelope:
    limiter_invalid_reference: bool = True
    boundary_loss_in_sim: bool = True
    invalid_state: bool = True
    extreme_proxy_actuator_violation: bool = True


@dataclass(frozen=True, slots=True)
class MachineEnvelope:
    name: str
    verified_geometry: Mapping[str, Any]
    observed_data: Mapping[str, Any]
    proxy_training: ProxyTrainingEnvelope
    soft_penalties: SoftPenaltyEnvelope
    termination: TerminationEnvelope

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ValueError("machine envelope name must not be empty")
        self.proxy_training.validate()
        self.soft_penalties.validate()
        if not bool(self.verified_geometry.get("limiter_surface", False)):
            raise ValueError("verified_geometry.limiter_surface must be true for T15 proxy target generation")
        if not bool(self.verified_geometry.get("coil_positions", False)):
            raise ValueError("verified_geometry.coil_positions must be true for T15 proxy target generation")


def load_machine_envelope(path: str | Path) -> MachineEnvelope:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    raw = json.loads(text) if source.suffix.lower() == ".json" or yaml is None else yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ValueError(f"machine envelope must be a mapping: {source}")
    machine_raw = raw.get("machine_envelope", raw)
    if not isinstance(machine_raw, Mapping):
        raise ValueError("machine_envelope must be a mapping")
    proxy_raw = _mapping(machine_raw.get("proxy_training"), "proxy_training")
    soft_raw = _mapping(machine_raw.get("soft_penalties"), "soft_penalties")
    termination_raw = _mapping(machine_raw.get("termination", {}), "termination")
    out = MachineEnvelope(
        name=str(machine_raw.get("name", source.stem)),
        verified_geometry=_mapping(machine_raw.get("verified_geometry"), "verified_geometry"),
        observed_data=_mapping(machine_raw.get("observed_data", {}), "observed_data"),
        proxy_training=ProxyTrainingEnvelope(
            ip_range_a=_number_pair(proxy_raw.get("ip_range_a"), "proxy_training.ip_range_a"),
            ip_rate_a_per_s=_positive_float(proxy_raw.get("ip_rate_a_per_s"), "proxy_training.ip_rate_a_per_s"),
            boundary_margin_m=_nonnegative_float(proxy_raw.get("boundary_margin_m", 0.0), "proxy_training.boundary_margin_m"),
            boundary_rate_m_per_s=_positive_float(proxy_raw.get("boundary_rate_m_per_s"), "proxy_training.boundary_rate_m_per_s"),
            coil_current_range_a=_tuple_ranges(proxy_raw.get("coil_current_range_a"), "proxy_training.coil_current_range_a"),
            coil_jdot_range_a_per_s=_tuple_ranges(proxy_raw.get("coil_jdot_range_a_per_s"), "proxy_training.coil_jdot_range_a_per_s"),
        ),
        soft_penalties=SoftPenaltyEnvelope(
            ip_soft_range_a=_number_pair(soft_raw.get("ip_soft_range_a", proxy_raw.get("ip_range_a")), "soft_penalties.ip_soft_range_a"),
            coil_current_soft_fraction=_positive_float(soft_raw.get("coil_current_soft_fraction", 1.0), "soft_penalties.coil_current_soft_fraction"),
            coil_jdot_soft_fraction=_positive_float(soft_raw.get("coil_jdot_soft_fraction", 1.0), "soft_penalties.coil_jdot_soft_fraction"),
        ),
        termination=TerminationEnvelope(
            limiter_invalid_reference=bool(termination_raw.get("limiter_invalid_reference", True)),
            boundary_loss_in_sim=bool(termination_raw.get("boundary_loss_in_sim", True)),
            invalid_state=bool(termination_raw.get("invalid_state", True)),
            extreme_proxy_actuator_violation=bool(termination_raw.get("extreme_proxy_actuator_violation", True)),
        ),
    )
    out.validate()
    return out


def _mapping(raw: object, name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return raw


def _number_pair(raw: object, name: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{name} must be a two-value list")
    out = (float(raw[0]), float(raw[1]))
    _validate_range(out, name)
    return out


def _tuple_ranges(raw: object, name: str) -> tuple[tuple[float, float], ...]:
    if isinstance(raw, Mapping):
        items = [raw[key] for key in sorted(raw)]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise ValueError(f"{name} must be a list or mapping")
    out = tuple(_number_pair(item, f"{name}[{i}]") for i, item in enumerate(items))
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def _positive_float(raw: object, name: str) -> float:
    value = float(raw)
    _validate_positive(value, name)
    return value


def _nonnegative_float(raw: object, name: str) -> float:
    value = float(raw)
    _validate_nonnegative(value, name)
    return value


def _validate_range(value: tuple[float, float], name: str) -> None:
    lo, hi = float(value[0]), float(value[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{name} bounds must be finite")
    if hi < lo:
        raise ValueError(f"{name} upper bound must be >= lower bound")


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
