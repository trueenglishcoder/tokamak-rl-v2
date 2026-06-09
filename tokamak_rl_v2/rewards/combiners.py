from __future__ import annotations

import torch
from torch import Tensor


def mean(values: Tensor, weights: Tensor | None = None, dim: int = -1) -> Tensor:
    mask = torch.isfinite(values)
    vals = torch.where(mask, values, torch.zeros_like(values))
    if weights is None:
        w = mask.to(dtype=values.dtype)
    else:
        w = weights.to(device=values.device, dtype=values.dtype) * mask.to(dtype=values.dtype)
    denom = torch.clamp(torch.sum(w, dim=dim), min=1.0e-12)
    return torch.sum(vals * w, dim=dim) / denom


def geometric_mean(values: Tensor, weights: Tensor | None = None, dim: int = -1) -> Tensor:
    vals = torch.clamp(values, min=1.0e-12, max=1.0)
    return torch.exp(mean(torch.log(vals), weights=weights, dim=dim))


def multiply(values: Tensor, weights: Tensor | None = None, dim: int = -1) -> Tensor:
    vals = torch.clamp(values, min=1.0e-12, max=1.0)
    if weights is None:
        return torch.prod(vals, dim=dim)
    w = weights.to(device=values.device, dtype=values.dtype)
    return torch.exp(torch.sum(torch.log(vals) * w, dim=dim))


def smooth_max(values: Tensor, *, alpha: float, weights: Tensor | None = None, dim: int = -1) -> Tensor:
    if abs(float(alpha)) < 1.0e-12:
        return mean(values, weights=weights, dim=dim)
    logits = values * float(alpha)
    if weights is not None:
        logits = logits + torch.log(torch.clamp(weights.to(device=values.device, dtype=values.dtype), min=1.0e-12))
    probs = torch.softmax(logits, dim=dim)
    return torch.sum(probs * values, dim=dim)
