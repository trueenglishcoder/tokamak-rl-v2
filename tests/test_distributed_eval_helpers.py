from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tokamak_rl_v2" / "training" / "distributed_eval.py"
SPEC = importlib.util.spec_from_file_location("_distributed_eval_under_test", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
distributed_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distributed_eval)

distributed_reduce_metrics = distributed_eval.distributed_reduce_metrics
distributed_seed_offset = distributed_eval.distributed_seed_offset
distributed_shard_count = distributed_eval.distributed_shard_count


def test_distributed_shard_count_balances_remainder() -> None:
    assert [distributed_shard_count(10, rank=r, world_size=4) for r in range(4)] == [3, 3, 2, 2]
    assert [distributed_shard_count(3, rank=r, world_size=8) for r in range(8)] == [1, 1, 1, 0, 0, 0, 0, 0]


def test_distributed_seed_offset_is_rank_specific() -> None:
    assert distributed_seed_offset(200000, rank=0) == 200000
    assert distributed_seed_offset(200000, rank=2) == 2200006


def test_reduce_metrics_single_rank_filters_and_preserves_scalars() -> None:
    metrics = {
        "ip_error_a": 12000.0,
        "current_over_limit_a_max": 55.0,
        "boundary_found_late_min": 1.0,
        "bad": float("nan"),
        "text": "ignored",
    }
    out = distributed_reduce_metrics(metrics, local_count=5, device=torch.device("cpu"), enabled=False)
    assert out["ip_error_a"] == 12000.0
    assert out["current_over_limit_a_max"] == 55.0
    assert out["boundary_found_late_min"] == 1.0
    assert "text" not in out
    assert "bad" not in out
