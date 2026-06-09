from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def equal(error: Tensor, *, not_equal: float = 0.0) -> Tensor:
    return torch.where(error == 0, torch.ones_like(error), torch.full_like(error, float(not_equal)))


def clipped_linear(error: Tensor, *, good: float, bad: float) -> Tensor:
    good_f = float(good); bad_f = float(bad)
    if bad_f == good_f:
        return equal(error)
    q = 1.0 - (error - good_f) / (bad_f - good_f)
    return torch.clamp(q, 0.0, 1.0)


def sigmoid(error: Tensor, *, good: float, bad: float) -> Tensor:
    good_f = float(good); bad_f = float(bad)
    if bad_f == good_f:
        return equal(error)
    logit_good = math.log(0.95 / 0.05)
    logit_bad = math.log(0.05 / 0.95)
    k = (logit_bad - logit_good) / (bad_f - good_f)
    midpoint = good_f - logit_good / k
    return torch.sigmoid((error - midpoint) * k)


def _softplus_scale(good: float, bad: float) -> float:
    span = abs(float(bad) - float(good))
    if span <= 0.0:
        return 1.0
    target = 10.0 * math.log(2.0)
    lo, hi = 0.0, 1000.0 / span
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = math.log1p(math.exp(span * mid))
        if value < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def softplus(error: Tensor, *, good: float = 0.0, bad: float) -> Tensor:
    good_f = float(good); bad_f = float(bad)
    scale = _softplus_scale(good_f, bad_f)
    if bad_f >= good_f:
        numerator = F.softplus((bad_f - error) * scale)
        denominator = F.softplus(torch.as_tensor((bad_f - good_f) * scale, dtype=error.dtype, device=error.device))
    else:
        numerator = F.softplus((error - bad_f) * scale)
        denominator = F.softplus(torch.as_tensor((good_f - bad_f) * scale, dtype=error.dtype, device=error.device))
    return torch.clamp(numerator / torch.clamp(denominator, min=1.0e-12), 0.0, 1.0)


def neg_exp(error: Tensor, *, good: float = 0.0, bad: float) -> Tensor:
    span = abs(float(bad) - float(good))
    if span <= 0.0:
        return equal(error)
    alpha = math.log(10.0) / span
    distance = torch.clamp(error - float(good), min=0.0) if bad >= good else torch.clamp(float(good) - error, min=0.0)
    return torch.exp(-alpha * distance)
