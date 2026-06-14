from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tokamak_rl_v2.networks.initialization import truncated_fanin_init


CRITIC_ACTION_INPUT_KIND = "normalized_action_v1"


@dataclass(frozen=True, slots=True)
class CriticState:
    h: Tensor
    c: Tensor


class RecurrentQCritic(nn.Module):
    """Published asymmetric recurrent Q critic."""

    def __init__(self, obs_dim: int, action_dim: int, lstm_hidden_dim: int = 256, mlp_hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.input_dim = self.obs_dim + self.action_dim
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.lstm = nn.LSTM(self.input_dim, self.lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(self.input_dim + self.lstm_hidden_dim, self.mlp_hidden_dim)
        self.fc2 = nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim)
        self.q_head = nn.Linear(self.mlp_hidden_dim, 1)
        self.apply(truncated_fanin_init)

    def forward(self, obs: Tensor, action: Tensor, state: CriticState | None = None, mask: Tensor | None = None) -> tuple[Tensor, CriticState]:
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
            action = action.unsqueeze(1)
        normalized_action = torch.clamp(action, -1.0, 1.0)
        x = torch.cat([obs, normalized_action], dim=-1)
        hx = None if state is None else (state.h.contiguous(), state.c.contiguous())
        y, (h, c) = self.lstm(x, hx)
        z = torch.cat([x, y], dim=-1)
        z = F.elu(self.fc1(z))
        z = F.elu(self.fc2(z))
        q = self.q_head(z).squeeze(-1)
        if mask is not None:
            q = q * mask.to(dtype=q.dtype)
        return q, CriticState(h=h, c=c)

    def zero_state(self, batch_size: int, device: torch.device | str) -> CriticState:
        dev = torch.device(device)
        h = torch.zeros((1, int(batch_size), self.lstm_hidden_dim), dtype=torch.float32, device=dev)
        c = torch.zeros((1, int(batch_size), self.lstm_hidden_dim), dtype=torch.float32, device=dev)
        return CriticState(h=h, c=c)
