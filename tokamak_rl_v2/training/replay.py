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
    """Episode-aware FIFO replay for recurrent sequence learning.

    Vectorized environments produce batches ordered by `[time, env]`. A flat
    ring buffer corrupts that into fake temporal sequences crossing environment
    lanes. This replay stores each env lane in its own episode slot and samples
    only contiguous windows from one physical episode.
    """

    def __init__(
        self,
        *,
        capacity_episodes: int | None = None,
        max_episode_steps: int | None = None,
        active_envs: int = 1,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str,
        capacity_steps: int | None = None,
    ) -> None:
        if capacity_episodes is None:
            if capacity_steps is None or max_episode_steps is None:
                raise ValueError("capacity_episodes and max_episode_steps are required")
            capacity_episodes = max(1, int(capacity_steps) // int(max_episode_steps))
        if max_episode_steps is None:
            raise ValueError("max_episode_steps is required")
        self.capacity_episodes = int(capacity_episodes)
        self.max_episode_steps = int(max_episode_steps)
        self.active_envs = int(active_envs)
        if self.capacity_episodes <= 0:
            raise ValueError("capacity_episodes must be > 0")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0")
        if self.active_envs <= 0:
            raise ValueError("active_envs must be > 0")
        if self.capacity_episodes < self.active_envs:
            raise ValueError("replay_capacity_episodes must be at least the active environment count")
        self.device = torch.device(device)

        shape = (self.capacity_episodes, self.max_episode_steps)
        self.obs = torch.zeros((*shape, int(obs_dim)), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((*shape, int(action_dim)), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.discount = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((*shape, int(obs_dim)), dtype=torch.float32, device=self.device)
        self.done = torch.ones(shape, dtype=torch.bool, device=self.device)
        self.valid = torch.zeros(shape, dtype=torch.bool, device=self.device)

        self.episode_lengths = torch.zeros((self.capacity_episodes,), dtype=torch.long, device=self.device)
        self.episode_closed = torch.zeros((self.capacity_episodes,), dtype=torch.bool, device=self.device)
        self.episode_generation = torch.zeros((self.capacity_episodes,), dtype=torch.long, device=self.device)
        self.active_slots = torch.full((self.active_envs,), -1, dtype=torch.long, device=self.device)
        self.active_generation = torch.zeros((self.active_envs,), dtype=torch.long, device=self.device)
        self.write_episode = 0
        self.size = 0
        self.completed_episodes = 0
        self.generation = 0
        self.start_episodes(self.active_envs)

    @property
    def capacity(self) -> int:
        return int(self.capacity_episodes * self.max_episode_steps)

    def start_episodes(self, active_envs: int | None = None) -> None:
        if active_envs is not None and int(active_envs) != self.active_envs:
            raise ValueError("active_envs cannot change after replay construction")
        for env_index in range(self.active_envs):
            self.active_slots[env_index] = self._allocate_episode_slot()
            self.active_generation[env_index] = self.generation

    def start_new_episodes(self) -> None:
        """Start fresh active episodes without appending across an external reset."""
        protected = {int(v) for v in self.active_slots.detach().cpu().tolist() if int(v) >= 0}
        for env_index in range(self.active_envs):
            slot = int(self.active_slots[env_index].item())
            if slot >= 0 and int(self.episode_lengths[slot].item()) > 0:
                self.episode_closed[slot] = True
                self.completed_episodes += 1
        for env_index in range(self.active_envs):
            new_slot = self._allocate_episode_slot(protected_slots=protected)
            self.active_slots[env_index] = new_slot
            self.active_generation[env_index] = self.generation
            protected.add(new_slot)

    def add_batch(self, obs: Tensor, action: Tensor, reward: Tensor, discount: Tensor, next_obs: Tensor, done: Tensor, lane_indices: Tensor | list[int] | None = None) -> None:
        B = int(obs.shape[0])
        if lane_indices is None:
            if B != self.active_envs:
                raise ValueError(f"add_batch expected {self.active_envs} env lanes, got {B}")
            lanes = torch.arange(B, dtype=torch.long, device=self.device)
        else:
            lanes = torch.as_tensor(lane_indices, dtype=torch.long, device=self.device).reshape(-1)
            if int(lanes.numel()) != B:
                raise ValueError("lane_indices length must match batch size")
            if torch.any((lanes < 0) | (lanes >= self.active_envs)):
                raise ValueError("lane_indices contain an out-of-range replay lane")
        for b in range(B):
            lane = int(lanes[b].item())
            slot = int(self.active_slots[lane].item())
            if slot < 0:
                slot = self._allocate_episode_slot()
                self.active_slots[lane] = slot
            pos = int(self.episode_lengths[slot].item())
            if pos >= self.max_episode_steps:
                self._close_active_episode(lane)
                slot = int(self.active_slots[lane].item())
                pos = 0

            self.obs[slot, pos] = obs[b].detach()
            self.action[slot, pos] = action[b].detach()
            self.reward[slot, pos] = reward[b].detach()
            self.discount[slot, pos] = discount[b].detach()
            self.next_obs[slot, pos] = next_obs[b].detach()
            self.done[slot, pos] = done[b].detach()
            self.valid[slot, pos] = True
            self.episode_lengths[slot] += 1

            if bool(done[b].item()) or int(self.episode_lengths[slot].item()) >= self.max_episode_steps:
                self._close_active_episode(lane)
        self.size = int(torch.sum(self.episode_lengths).item())

    def ready(self, sequence_length: int, batch_size: int, min_sequence_length: int | None = None) -> bool:
        min_len = self._effective_min_sequence_length(sequence_length, min_sequence_length)
        eligible = self._eligible_slots(min_len)
        return int(eligible.numel()) >= 1 and self.size >= min_len

    def sample(self, *, batch_size: int, sequence_length: int, min_sequence_length: int | None = None, generator: torch.Generator | None = None) -> SequenceBatch:
        if not self.ready(sequence_length, batch_size, min_sequence_length=min_sequence_length):
            raise RuntimeError("replay does not contain enough transitions")
        T = int(sequence_length)
        if T > self.max_episode_steps:
            raise ValueError("sequence_length cannot exceed max_episode_steps")
        min_len = self._effective_min_sequence_length(T, min_sequence_length)
        eligible = self._eligible_slots(min_len)
        choice = torch.randint(0, int(eligible.numel()), (int(batch_size),), device=self.device, generator=generator)
        slots = eligible[choice]
        lengths = self.episode_lengths[slots]
        max_starts = torch.clamp(lengths - T, min=0)
        rand = torch.rand((int(batch_size),), dtype=torch.float32, device=self.device, generator=generator)
        starts = torch.floor(rand * (max_starts.to(torch.float32) + 1.0)).to(torch.long)
        offsets = torch.arange(T, device=self.device)[None, :]
        idx = starts[:, None] + offsets
        mask = self.valid[slots[:, None], idx].to(torch.float32)
        return SequenceBatch(
            obs=self.obs[slots[:, None], idx],
            action=self.action[slots[:, None], idx],
            reward=self.reward[slots[:, None], idx],
            discount=self.discount[slots[:, None], idx],
            next_obs=self.next_obs[slots[:, None], idx],
            done=self.done[slots[:, None], idx],
            mask=mask,
        )

    def stats(self, *, sequence_length: int, min_sequence_length: int | None = None) -> dict[str, float]:
        lengths = self.episode_lengths.detach().to(dtype=torch.float32)
        nonzero = lengths[lengths > 0]
        min_len = self._effective_min_sequence_length(sequence_length, min_sequence_length)
        full = self._eligible_slots(int(sequence_length))
        short = self._eligible_slots(min_len)
        return {
            "replay_size": float(self.size),
            "replay_completed_episodes": float(self.completed_episodes),
            "replay_full_sequence_eligible_episodes": float(int(full.numel())),
            "replay_min_sequence_eligible_episodes": float(int(short.numel())),
            "replay_mean_episode_length": float(torch.mean(nonzero).item()) if int(nonzero.numel()) else 0.0,
            "replay_min_episode_length": float(torch.min(nonzero).item()) if int(nonzero.numel()) else 0.0,
            "replay_max_episode_length": float(torch.max(nonzero).item()) if int(nonzero.numel()) else 0.0,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity_episodes": self.capacity_episodes,
            "max_episode_steps": self.max_episode_steps,
            "active_envs": self.active_envs,
            "obs": self.obs.detach().cpu(),
            "action": self.action.detach().cpu(),
            "reward": self.reward.detach().cpu(),
            "discount": self.discount.detach().cpu(),
            "next_obs": self.next_obs.detach().cpu(),
            "done": self.done.detach().cpu(),
            "valid": self.valid.detach().cpu(),
            "episode_lengths": self.episode_lengths.detach().cpu(),
            "episode_closed": self.episode_closed.detach().cpu(),
            "episode_generation": self.episode_generation.detach().cpu(),
            "active_slots": self.active_slots.detach().cpu(),
            "active_generation": self.active_generation.detach().cpu(),
            "write_episode": int(self.write_episode),
            "size": int(self.size),
            "completed_episodes": int(self.completed_episodes),
            "generation": int(self.generation),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = (self.capacity_episodes, self.max_episode_steps, self.active_envs)
        got = (int(state["capacity_episodes"]), int(state["max_episode_steps"]), int(state["active_envs"]))
        if got != expected:
            raise ValueError(f"replay state shape mismatch: expected {expected}, got {got}")
        for name in ("obs", "action", "reward", "discount", "next_obs", "done", "valid", "episode_lengths", "episode_closed", "episode_generation", "active_slots", "active_generation"):
            getattr(self, name).copy_(torch.as_tensor(state[name], device=self.device))
        self.write_episode = int(state["write_episode"])
        self.size = int(state["size"])
        self.completed_episodes = int(state["completed_episodes"])
        self.generation = int(state["generation"])

    def _eligible_slots(self, sequence_length: int) -> Tensor:
        length_ok = self.episode_lengths >= int(sequence_length)
        return torch.nonzero(length_ok, as_tuple=False).reshape(-1)

    @staticmethod
    def _effective_min_sequence_length(sequence_length: int, min_sequence_length: int | None) -> int:
        seq = int(sequence_length)
        if seq <= 0:
            raise ValueError("sequence_length must be positive")
        if min_sequence_length is None:
            return seq
        min_len = int(min_sequence_length)
        if min_len <= 0:
            raise ValueError("min_sequence_length must be positive")
        return min(seq, min_len)

    def _allocate_episode_slot(self, *, protected_slots: set[int] | None = None) -> int:
        active = {int(v) for v in self.active_slots.detach().cpu().tolist() if int(v) >= 0}
        if protected_slots:
            active.update(int(v) for v in protected_slots if int(v) >= 0)
        slot = -1
        for offset in range(self.capacity_episodes):
            candidate = (int(self.write_episode) + offset) % self.capacity_episodes
            if candidate not in active:
                slot = candidate
                break
        if slot < 0:
            raise RuntimeError("replay has no free episode slot; increase replay_capacity_episodes above the active environment count")
        self.generation += 1
        self.obs[slot].zero_()
        self.action[slot].zero_()
        self.reward[slot].zero_()
        self.discount[slot].zero_()
        self.next_obs[slot].zero_()
        self.done[slot].fill_(True)
        self.valid[slot].zero_()
        self.episode_lengths[slot] = 0
        self.episode_closed[slot] = False
        self.episode_generation[slot] = self.generation
        self.write_episode = (slot + 1) % self.capacity_episodes
        return slot

    def _close_active_episode(self, env_index: int) -> None:
        slot = int(self.active_slots[int(env_index)].item())
        if slot >= 0:
            self.episode_closed[slot] = True
            self.completed_episodes += 1
        new_slot = self._allocate_episode_slot()
        self.active_slots[int(env_index)] = new_slot
        self.active_generation[int(env_index)] = self.generation
