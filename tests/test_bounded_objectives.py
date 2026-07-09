from __future__ import annotations

import torch

from tokamak_rl_v2.rewards.bounded_objectives import bounded_margin_loss, bounded_margin_quality


def test_bounded_margin_loss_good_warning_bad_regions() -> None:
    x = torch.tensor([0.20, 0.70, 0.80, 0.90, 1.20])
    loss = bounded_margin_loss(x, good=0.70, bad=0.90)
    assert torch.allclose(loss[:2], torch.zeros(2))
    assert 0.0 < float(loss[2]) < 1.0
    assert torch.allclose(loss[3:], torch.ones(2))


def test_bounded_margin_quality_is_one_minus_loss() -> None:
    x = torch.tensor([0.70, 0.80, 0.90])
    loss = bounded_margin_loss(x, good=0.70, bad=0.90)
    quality = bounded_margin_quality(x, good=0.70, bad=0.90)
    assert torch.allclose(quality, 1.0 - loss)
    assert float(quality[0]) == 1.0
    assert float(quality[-1]) < 1.0e-6
