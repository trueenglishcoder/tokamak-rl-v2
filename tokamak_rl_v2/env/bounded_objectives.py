from __future__ import annotations

import torch
from torch import Tensor


def bounded_margin_loss(value_fraction: Tensor, *, good: float, bad: float) -> Tensor:
    """Capped smooth loss for a normalized limit-usage fraction.

    ``value_fraction`` is expected to be a non-negative fraction of a known
    physical limit, for example abs(current) / current_limit.

    The returned loss is zero at and below ``good``, rises quadratically between
    ``good`` and ``bad``, and is capped at one at and above ``bad``. This is a
    TCV-style pre-failure shaping component: it applies before hard termination
    and does not encode any machine-specific coil-pair relation.
    """

    good_f = float(good)
    bad_f = float(bad)
    if not bad_f > good_f:
        raise ValueError("bad must be greater than good")
    width = max(bad_f - good_f, 1.0e-12)
    z = torch.clamp((torch.clamp(value_fraction, min=0.0) - good_f) / width, min=0.0, max=1.0)
    return z.pow(2)


def bounded_margin_quality(value_fraction: Tensor, *, good: float, bad: float) -> Tensor:
    """Quality value corresponding to :func:`bounded_margin_loss`.

    Returns one in the good region and zero in the bad region.
    """

    return torch.clamp(1.0 - bounded_margin_loss(value_fraction, good=good, bad=bad), min=0.0, max=1.0)


def fraction_margin(value_fraction: Tensor) -> Tensor:
    """Remaining normalized margin, ``1 - value_fraction``.

    Negative values mean the known limit has already been exceeded.
    """

    return 1.0 - value_fraction
