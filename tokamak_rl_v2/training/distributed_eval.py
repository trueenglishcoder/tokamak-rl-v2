from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.distributed as dist


def distributed_is_initialized() -> bool:
    return bool(dist.is_available() and dist.is_initialized() and int(dist.get_world_size()) > 1)


def distributed_rank() -> int:
    if distributed_is_initialized():
        return int(dist.get_rank())
    return 0


def distributed_world_size() -> int:
    if distributed_is_initialized():
        return int(dist.get_world_size())
    return 1


def distributed_shard_count(total: int, *, rank: int | None = None, world_size: int | None = None) -> int:
    total_i = max(0, int(total))
    world_i = max(1, int(distributed_world_size() if world_size is None else world_size))
    rank_i = int(distributed_rank() if rank is None else rank)
    if rank_i < 0 or rank_i >= world_i:
        raise ValueError(f"rank must satisfy 0 <= rank < world_size, got rank={rank_i}, world_size={world_i}")
    base = total_i // world_i
    extra = total_i % world_i
    return base + (1 if rank_i < extra else 0)


def distributed_seed_offset(base_seed_offset: int, *, rank: int | None = None) -> int:
    rank_i = int(distributed_rank() if rank is None else rank)
    return int(base_seed_offset) + rank_i * 1_000_003


def _metric_reduce_mode(name: str) -> str:
    lower = str(name).lower()
    if lower.endswith("_count") or lower in {"count", "episodes", "episode_count"}:
        return "sum"
    if lower.endswith("_min") or lower in {
        "boundary_found_min",
        "boundary_found_late_min",
        "min_episode_steps",
        "min_episode_completion",
        "current_margin_fraction_min",
        "current_margin_fraction_late_min",
    }:
        return "min"
    if lower.endswith("_max") or lower in {
        "max_episode_steps",
        "current_over_limit_a_max",
        "current_over_limit_a_late_max",
        "shape_error_mean_m_max",
        "shape_error_max_m_max",
        "ip_error_a_max",
        "ip_error_a_late_max",
    }:
        return "max"
    return "mean"


def _as_finite_float(value: object) -> tuple[float, bool]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return 0.0, False
    return out, math.isfinite(out)


def distributed_reduce_metrics(
    metrics: Mapping[str, object],
    *,
    local_count: int,
    device: torch.device | str,
    enabled: bool | None = None,
) -> dict[str, float]:
    """Reduce scalar metric dictionaries across torch.distributed ranks.

    Mean-like metrics are weighted by local_count. Max/min metrics use the
    obvious extrema. Count-like metrics are summed. Non-numeric and non-finite
    values are ignored.
    """

    if enabled is None:
        enabled = distributed_is_initialized()
    if not enabled:
        out: dict[str, float] = {}
        for key, value in metrics.items():
            numeric, finite = _as_finite_float(value)
            if finite:
                out[str(key)] = numeric
        return out

    dev = torch.device(device)
    count = max(0, int(local_count))
    reduced: dict[str, float] = {}

    for raw_key in sorted(str(k) for k in metrics.keys()):
        value, finite = _as_finite_float(metrics.get(raw_key))
        mode = _metric_reduce_mode(raw_key)
        if mode == "mean":
            weight = float(count) if finite and count > 0 else 0.0
            pair = torch.tensor([value * weight if weight > 0.0 else 0.0, weight], dtype=torch.float64, device=dev)
            dist.all_reduce(pair, op=dist.ReduceOp.SUM)
            denom = float(pair[1].detach().cpu().item())
            reduced[raw_key] = float(pair[0].detach().cpu().item() / denom) if denom > 0.0 else float("nan")
        elif mode == "sum":
            tensor = torch.tensor([value if finite else 0.0], dtype=torch.float64, device=dev)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[raw_key] = float(tensor.detach().cpu().item())
        elif mode == "max":
            tensor = torch.tensor([value if finite else -float("inf")], dtype=torch.float64, device=dev)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            out = float(tensor.detach().cpu().item())
            reduced[raw_key] = out if math.isfinite(out) else float("nan")
        elif mode == "min":
            tensor = torch.tensor([value if finite else float("inf")], dtype=torch.float64, device=dev)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            out = float(tensor.detach().cpu().item())
            reduced[raw_key] = out if math.isfinite(out) else float("nan")
        else:  # pragma: no cover
            raise RuntimeError(f"unknown reduction mode {mode!r}")
    return reduced


def distributed_all_ok(ok: bool, *, device: torch.device | str) -> bool:
    if not distributed_is_initialized():
        return bool(ok)
    tensor = torch.tensor([1 if ok else 0], dtype=torch.int32, device=torch.device(device))
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(int(tensor.detach().cpu().item()) == 1)


def distributed_barrier_and_destroy() -> None:
    if not distributed_is_initialized():
        return
    try:
        dist.barrier()
    finally:
        dist.destroy_process_group()
