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
    actor_mle_loss: float
    actor_param_delta_norm: float
    action_mean_abs: float
    action_std_mean: float
    sampled_q_spread: float
    policy_weight_entropy: float
    policy_weight_max: float
    mpo_temperature: float


class MaximumAPosterioriPolicyOptimiser:
    """Maximum a Posteriori Policy Optimisation learner."""

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

    def state_dict(self) -> dict[str, object]:
        return {
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "temperature_optim": self.temperature_optim.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        self.temperature_optim.load_state_dict(state["temperature_optim"])
        self.log_temperature.data.copy_(torch.as_tensor(state["log_temperature"], dtype=torch.float32, device=self.device))

    def update(self, batch: SequenceBatch) -> UpdateMetrics:
        critic_loss, q_mean, target_mean = self._critic_update(batch)
        actor_metrics = self._actor_update(batch)
        self._soft_sync(float(self.config.target_update_tau))
        return UpdateMetrics(
            critic_loss=float(critic_loss),
            actor_loss=float(actor_metrics[0]),
            mean_kl=float(actor_metrics[1]),
            std_kl=float(actor_metrics[2]),
            q_mean=float(q_mean),
            target_q_mean=float(target_mean),
            actor_mle_loss=float(actor_metrics[3]),
            actor_param_delta_norm=float(actor_metrics[4]),
            action_mean_abs=float(actor_metrics[5]),
            action_std_mean=float(actor_metrics[6]),
            sampled_q_spread=float(actor_metrics[7]),
            policy_weight_entropy=float(actor_metrics[8]),
            policy_weight_max=float(actor_metrics[9]),
            mpo_temperature=float(actor_metrics[10]),
        )

    def _critic_update(self, batch: SequenceBatch) -> tuple[float, float, float]:
        mask = batch.mask.to(dtype=torch.float32)
        q, _ = self.critic(batch.obs, batch.action, mask=mask)
        with torch.no_grad():
            B, T, O = batch.next_obs.shape
            next_action = self.target_actor.deterministic(batch.next_obs.reshape(B * T, O)).reshape(B, T, -1)
            q_next, _ = self.target_critic(batch.next_obs, next_action, mask=mask)
            target = batch.reward + batch.discount * (~batch.done).to(torch.float32) * q_next
        denom = torch.clamp(mask.sum(), min=1.0)
        loss = torch.sum(((q - target).pow(2)) * mask) / denom
        self.critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()
        q_mean = torch.sum(q.detach() * mask) / denom
        target_mean = torch.sum(target.detach() * mask) / denom
        return float(loss.detach().cpu()), float(q_mean.cpu()), float(target_mean.cpu())

    def _actor_update(self, batch: SequenceBatch) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
        obs_seq = batch.obs.detach()
        mask = batch.mask.to(dtype=torch.float32)
        B, T, O = obs_seq.shape
        flat_obs = obs_seq.reshape(B * T, O)
        with torch.no_grad():
            old = self.target_actor(flat_obs)
            K = int(self.config.action_samples)
            raw = old.distribution.rsample((K,)).reshape(K, B, T, -1)
            sampled = torch.tanh(raw)
            q_values = self._sampled_q_values(obs_seq, sampled, mask=mask).detach()

        q_centered = q_values - torch.max(q_values, dim=0, keepdim=True).values
        eta = torch.clamp(self.log_temperature.exp(), min=1.0e-6, max=1.0e6)
        log_mean_exp = torch.logsumexp(q_centered / eta, dim=0) - torch.log(torch.as_tensor(float(K), dtype=q_values.dtype, device=q_values.device))
        dual = eta * (float(getattr(self.config, "mpo_epsilon", 0.1)) + log_mean_exp)
        denom = torch.clamp(mask.sum(), min=1.0)
        temp_loss = torch.sum(dual * mask) / denom
        self.temperature_optim.zero_grad(set_to_none=True)
        temp_loss.backward()
        self.temperature_optim.step()

        eta_for_weights = torch.clamp(self.log_temperature.exp().detach(), min=1.0e-6, max=1.0e6)
        weights = torch.softmax(q_centered / eta_for_weights, dim=0).detach()
        before = [p.detach().clone() for p in self.actor.parameters()]
        new = self.actor(flat_obs)
        new_dist = new.distribution
        log_prob = new_dist.log_prob(raw.reshape(K, B * T, -1)).sum(dim=-1).reshape(K, B, T)
        masked = mask[None, :, :]
        denom = torch.clamp(mask.sum(), min=1.0)
        mle_loss = -torch.sum(weights * log_prob * masked) / denom

        new_mean = new.mean.reshape(B, T, -1)
        new_std = new.std.reshape(B, T, -1)
        old_mean = old.mean.reshape(B, T, -1)
        old_std = old.std.reshape(B, T, -1)
        mean_kl_t = torch.mean(((new_mean - old_mean).pow(2)) / (2.0 * old_std.pow(2).clamp_min(1.0e-6)), dim=-1)
        std_ratio = (new_std.pow(2) / old_std.pow(2).clamp_min(1.0e-6)).clamp_min(1.0e-6)
        std_kl_t = torch.mean(0.5 * (std_ratio - 1.0 - torch.log(std_ratio)), dim=-1)
        mean_kl = torch.sum(mean_kl_t * mask) / denom
        std_kl = torch.sum(std_kl_t * mask) / denom
        actor_loss = mle_loss + F.relu(mean_kl - float(self.config.mean_kl_epsilon)) + F.relu(std_kl - float(self.config.std_kl_epsilon))
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()
        delta_sq = torch.zeros((), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for previous, current in zip(before, self.actor.parameters(), strict=True):
                delta_sq = delta_sq + torch.sum((current.detach() - previous).pow(2))
            weight_entropy = -torch.sum(weights * torch.log(torch.clamp(weights, min=1.0e-12)) * masked) / denom
            sampled_q_spread = torch.sum(torch.std(q_values, dim=0, unbiased=False) * mask) / denom
            action_mean_abs = torch.sum(torch.mean(torch.abs(new_mean.detach()), dim=-1) * mask) / denom
            action_std_mean = torch.sum(torch.mean(new_std.detach(), dim=-1) * mask) / denom

        weight_max = torch.sum(torch.max(weights, dim=0).values * mask) / denom
        return (
            float(actor_loss.detach().cpu()),
            float(mean_kl.detach().cpu()),
            float(std_kl.detach().cpu()),
            float(mle_loss.detach().cpu()),
            float(torch.sqrt(delta_sq).detach().cpu()),
            float(action_mean_abs.detach().cpu()),
            float(action_std_mean.detach().cpu()),
            float(sampled_q_spread.detach().cpu()),
            float(weight_entropy.detach().cpu()),
            float(weight_max.detach().cpu()),
            float(eta_for_weights.detach().cpu()),
        )

    def _sampled_q_values(self, obs: Tensor, sampled_actions: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Evaluate sampled-action Q values over full recurrent sequences."""
        if obs.ndim != 3 or sampled_actions.ndim != 4:
            raise ValueError("_sampled_q_values expects obs [B,T,O] and sampled_actions [K,B,T,A]")
        K = int(sampled_actions.shape[0])
        out = torch.empty((K, int(obs.shape[0]), int(obs.shape[1])), dtype=obs.dtype, device=obs.device)
        for k in range(K):
            q, _ = self.critic(obs, sampled_actions[k], mask=mask)
            out[k] = q
        return out

    def _soft_sync(self, tau: float) -> None:
        with torch.no_grad():
            for target, source in zip(self.target_actor.parameters(), self.actor.parameters(), strict=True):
                target.mul_(1.0 - tau).add_(source, alpha=tau)
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                target.mul_(1.0 - tau).add_(source, alpha=tau)
