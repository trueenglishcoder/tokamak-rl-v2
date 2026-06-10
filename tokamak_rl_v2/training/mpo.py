from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tokamak_rl_v2.config.schema import LearnerConfig
from tokamak_rl_v2.networks import FeedForwardGaussianActor, RecurrentQCritic
from tokamak_rl_v2.training.replay import SequenceBatch


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    critic_loss: float
    actor_loss: float
    mean_kl: float
    std_kl: float
    q_mean: float
    target_q_mean: float


class MaximumAPosterioriPolicyOptimiser:
    """Actor/critic learner implementing Maximum a Posteriori Policy Optimisation."""

    def __init__(
        self,
        *,
        actor: FeedForwardGaussianActor,
        critic: RecurrentQCritic,
        target_actor: FeedForwardGaussianActor,
        target_critic: RecurrentQCritic,
        config: LearnerConfig,
        device: torch.device | str,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.config = config
        self.device = torch.device(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(config.actor_lr))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(config.critic_lr))
        self.log_temperature = nn.Parameter(torch.tensor(float(config.temperature), dtype=torch.float32, device=self.device).log())
        self.temperature_optim = torch.optim.Adam([self.log_temperature], lr=float(config.kl_lr))
        self._hard_sync()

    def _hard_sync(self) -> None:
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(self, batch: SequenceBatch) -> UpdateMetrics:
        critic_loss, q_mean, target_mean = self._critic_update(batch)
        actor_loss, mean_kl, std_kl = self._actor_update(batch)
        self._soft_sync(float(self.config.target_update_tau))
        return UpdateMetrics(
            critic_loss=float(critic_loss), actor_loss=float(actor_loss), mean_kl=float(mean_kl), std_kl=float(std_kl), q_mean=float(q_mean), target_q_mean=float(target_mean)
        )

    def _critic_update(self, batch: SequenceBatch) -> tuple[float, float, float]:
        q, _ = self.critic(batch.obs, batch.action, mask=batch.mask)
        with torch.no_grad():
            next_action = self.target_actor.deterministic(batch.next_obs.reshape(-1, batch.next_obs.shape[-1])).reshape(batch.next_obs.shape[0], batch.next_obs.shape[1], -1)
            q_next, _ = self.target_critic(batch.next_obs, next_action, mask=batch.mask)
            target = batch.reward + batch.discount * (~batch.done).to(torch.float32) * q_next
        loss = torch.sum(((q - target).pow(2)) * batch.mask) / torch.clamp(torch.sum(batch.mask), min=1.0)
        self.critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()
        return float(loss.detach().cpu()), float(q.detach().mean().cpu()), float(target.detach().mean().cpu())

    def _actor_update(self, batch: SequenceBatch) -> tuple[float, float, float]:
        obs = batch.obs.reshape(-1, batch.obs.shape[-1]).detach()
        with torch.no_grad():
            old = self.target_actor(obs)
            dist = old.distribution
            K = int(self.config.action_samples)
            raw = dist.rsample((K,))
            sampled = torch.tanh(raw)
            q_values = self._sampled_q_values(obs, sampled)
            eta = torch.clamp(self.log_temperature.exp(), min=1.0e-6)
            weights = torch.softmax(q_values / eta, dim=0).detach()
        new = self.actor(obs)
        new_dist = new.distribution
        log_prob = new_dist.log_prob(raw).sum(dim=-1)
        mle_loss = -torch.mean(torch.sum(weights * log_prob, dim=0))
        mean_kl = torch.mean(((new.mean - old.mean).pow(2)) / (2.0 * old.std.pow(2).clamp_min(1.0e-6)))
        std_ratio = (new.std.pow(2) / old.std.pow(2).clamp_min(1.0e-6)).clamp_min(1.0e-6)
        std_kl = torch.mean(0.5 * (std_ratio - 1.0 - torch.log(std_ratio)))
        actor_loss = mle_loss + F.relu(mean_kl - float(self.config.mean_kl_epsilon)) + F.relu(std_kl - float(self.config.std_kl_epsilon))
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()
        temp_loss = self.log_temperature.exp() * (torch.mean(torch.sum(weights * torch.log(torch.clamp(weights, min=1.0e-12)), dim=0)) + float(self.config.mean_kl_epsilon)).detach()
        self.temperature_optim.zero_grad(set_to_none=True)
        temp_loss.backward()
        self.temperature_optim.step()
        return float(actor_loss.detach().cpu()), float(mean_kl.detach().cpu()), float(std_kl.detach().cpu())

    def _sampled_q_values(self, obs: Tensor, sampled_actions: Tensor) -> Tensor:
        """Evaluate sampled-action Q values without materializing K copies of full observations."""
        K = int(sampled_actions.shape[0])
        N = int(obs.shape[0])
        chunk_size = max(1, int(getattr(self.config, "actor_update_chunk_size", 2048)))
        out = torch.empty((K, N), dtype=obs.dtype, device=obs.device)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            obs_chunk = obs[start:end]
            for k in range(K):
                q, _ = self.critic(obs_chunk, sampled_actions[k, start:end])
                out[k, start:end] = q.reshape(-1)
        return out

    def _soft_sync(self, tau: float) -> None:
        with torch.no_grad():
            for target, source in zip(self.target_actor.parameters(), self.actor.parameters(), strict=True):
                target.mul_(1.0 - tau).add_(source, alpha=tau)
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                target.mul_(1.0 - tau).add_(source, alpha=tau)
