from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal
import torch.nn.functional as F

from tokamak_rl_v2.networks.initialization import truncated_fanin_init


@dataclass(frozen=True, slots=True)
class ActorOutput:
    mean: Tensor
    std: Tensor

    @property
    def distribution(self) -> Normal:
        return Normal(self.mean, self.std)


class FeedForwardGaussianActor(nn.Module):
    """Published feedforward stochastic policy architecture."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, min_std: float = 1.0e-4) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_std = float(min_std)
        self.input = nn.Linear(self.obs_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.hidden1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hidden3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.std_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.apply(truncated_fanin_init)
        truncated_fanin_init(self.mean_head, final_scale=1.0e-4)
        truncated_fanin_init(self.std_head, final_scale=1.0e-4)

    def forward(self, obs: Tensor) -> ActorOutput:
        x = torch.tanh(self.input_norm(self.input(obs)))
        x = F.elu(self.hidden1(x))
        x = F.elu(self.hidden2(x))
        x = F.elu(self.hidden3(x))
        mean = torch.tanh(self.mean_head(x))
        std = F.softplus(self.std_head(x)) + self.min_std
        return ActorOutput(mean=mean, std=std)

    def sample(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self(obs)
        dist = out.distribution
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(torch.clamp(1.0 - action.pow(2), min=1.0e-6))).sum(dim=-1)
        return action, log_prob, out.mean

    def deterministic(self, obs: Tensor) -> Tensor:
        return self(obs).mean
