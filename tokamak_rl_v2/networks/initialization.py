from __future__ import annotations

import math

import torch
from torch import nn


def truncated_fanin_init(module: nn.Module, *, final_scale: float = 1.0) -> None:
    """Initialize Linear/LSTM weights with truncated normal scaled by fan-in."""
    if isinstance(module, nn.Linear):
        fan_in = max(int(module.weight.shape[1]), 1)
        std = final_scale / math.sqrt(float(fan_in))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:
                fan_in = max(int(param.shape[1]), 1)
                std = 1.0 / math.sqrt(float(fan_in))
                nn.init.trunc_normal_(param, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            elif "bias" in name:
                nn.init.zeros_(param)
