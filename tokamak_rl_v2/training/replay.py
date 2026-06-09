from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    obs: Tensor
    action: Tensor
    reward: Tensor
    discount: Tensor
    next_obs: Tensor
    done: Tensor
    mask: Tensor


class FIFOSequenceReplay:
    """Finite-capacity FIFO replay storing trajectory transitions."""

    def __init__(self, *, capacity_steps: int, obs_dim: int, action_dim: int, device: torch.device | str) -> None:
        self.capacity = int(capacity_steps)
        if self.capacity <= 0:
            raise ValueError("capacity_steps must be > 0")
        self.device = torch.device(device)
        self.obs = torch.zeros((self.capacity, int(obs_dim)), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((self.capacity, int(action_dim)), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.discount = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((self.capacity, int(obs_dim)), dtype=torch.float32, device=self.device)
        self.done = torch.ones((self.capacity,), dtype=torch.bool, device=self.device)
        self.write = 0
        self.size = 0

    def add_batch(self, obs: Tensor, action: Tensor, reward: Tensor, discount: Tensor, next_obs: Tensor, done: Tensor) -> None:
        B = int(obs.shape[0])
        for b in range(B):
            i = self.write
            self.obs[i] = obs[b].detach()
            self.action[i] = action[b].detach()
            self.reward[i] = reward[b].detach()
            self.discount[i] = discount[b].detach()
            self.next_obs[i] = next_obs[b].detach()
            self.done[i] = done[b].detach()
            self.write = (self.write + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def ready(self, sequence_length: int, batch_size: int) -> bool:
        return self.size >= int(sequence_length) + 1 and self.size >= int(batch_size)

    def sample(self, *, batch_size: int, sequence_length: int, generator: torch.Generator | None = None) -> SequenceBatch:
        if not self.ready(sequence_length, batch_size):
            raise RuntimeError("replay does not contain enough transitions")
        max_start = self.size - int(sequence_length)
        starts = torch.randint(0, max_start, (int(batch_size),), device=self.device, generator=generator)
        offsets = torch.arange(int(sequence_length), device=self.device)[None, :]
        # Use logical oldest-to-newest order despite ring storage.
        oldest = (self.write - self.size) % self.capacity
        idx = (oldest + starts[:, None] + offsets) % self.capacity
        mask = torch.ones((int(batch_size), int(sequence_length)), dtype=torch.float32, device=self.device)
        return SequenceBatch(
            obs=self.obs[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            discount=self.discount[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
            mask=mask,
        )
