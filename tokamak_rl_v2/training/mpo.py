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
    mean_kl_penalty: float
    std_kl_penalty: float


class MaximumAPosterioriPolicyOptimiser:
    """Maximum a Posteriori Policy Optimisation learner.

    The actor update follows the MPO coordinate-ascent structure:

    * E-step: sample actions from the target/old policy, evaluate them with the
      recurrent critic, and solve the per-batch temperature dual so the
      non-parametric improved action weights satisfy the sampled KL budget.
    * M-step: fit the parametric Gaussian actor to the weighted samples while
      enforcing separate mean and standard-deviation KL constraints with learned
      non-negative Lagrange multipliers.
    """

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
        self.log_mean_kl_penalty = nn.Parameter(torch.zeros((), dtype=torch.float32, device=self.device))
        self.log_std_kl_penalty = nn.Parameter(torch.zeros((), dtype=torch.float32, device=self.device))
        self.kl_optim = torch.optim.Adam([self.log_mean_kl_penalty, self.log_std_kl_penalty], lr=float(config.kl_lr))
        self.last_temperature = torch.tensor(float(config.temperature), dtype=torch.float32, device=self.device)
        self._hard_sync()

    def _hard_sync(self) -> None:
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def state_dict(self) -> dict[str, object]:
        return {
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "kl_optim": self.kl_optim.state_dict(),
            "log_mean_kl_penalty": self.log_mean_kl_penalty.detach().cpu(),
            "log_std_kl_penalty": self.log_std_kl_penalty.detach().cpu(),
            "last_temperature": self.last_temperature.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        if "kl_optim" in state:
            self.kl_optim.load_state_dict(state["kl_optim"])
            self.log_mean_kl_penalty.data.copy_(torch.as_tensor(state["log_mean_kl_penalty"], dtype=torch.float32, device=self.device))
            self.log_std_kl_penalty.data.copy_(torch.as_tensor(state["log_std_kl_penalty"], dtype=torch.float32, device=self.device))
            self.last_temperature.copy_(torch.as_tensor(state.get("last_temperature", float(self.config.temperature)), dtype=torch.float32, device=self.device))
        else:
            raise ValueError("checkpoint contains the obsolete MPO learner state and cannot be resumed exactly")

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
            mean_kl_penalty=float(actor_metrics[11]),
            std_kl_penalty=float(actor_metrics[12]),
        )

    def _critic_update(self, batch: SequenceBatch) -> tuple[float, float, float]:
        mask = batch.mask.to(dtype=torch.float32)
        q, _ = self.critic(batch.obs, batch.action, mask=mask)
        with torch.no_grad():
            B, T, O = batch.next_obs.shape
            next_action = self.target_actor.deterministic(batch.next_obs.reshape(B * T, O)).reshape(B, T, -1)
            q_next = self.target_critic.evaluate_query_actions_with_history(
                history_obs=batch.obs,
                history_action=batch.action,
                query_obs=batch.next_obs,
                query_action=next_action,
                mask=mask,
                include_current_history=True,
            )
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

    def _actor_update(self, batch: SequenceBatch) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float, float]:
        obs_seq = batch.obs.detach()
        mask = batch.mask.to(dtype=torch.float32)
        B, T, O = obs_seq.shape
        flat_obs = obs_seq.reshape(B * T, O)
        with torch.no_grad():
            old = self.target_actor(flat_obs)
            K = int(self.config.action_samples)
            raw = old.distribution.rsample((K,)).reshape(K, B, T, -1)
            sampled = torch.tanh(raw)
            q_values = self._sampled_q_values(obs_seq, batch.action.detach(), sampled, mask=mask).detach()
            q_centered = q_values - torch.max(q_values, dim=0, keepdim=True).values
            eta = self._solve_e_step_temperature(q_centered, mask=mask, epsilon=float(self.config.mpo_epsilon))
            weights = torch.softmax(q_centered / eta, dim=0).detach()

        before = [p.detach().clone() for p in self.actor.parameters()]
        new = self.actor(flat_obs)
        new_dist = new.distribution
        log_prob = new_dist.log_prob(raw.reshape(K, B * T, -1)).sum(dim=-1).reshape(K, B, T)
        masked = mask[None, :, :]
        denom = torch.clamp(mask.sum(), min=1.0)
        mle_loss = -torch.sum(weights * log_prob * masked) / denom

        new_mean = new.mean.reshape(B, T, -1)
        new_std = new.std.reshape(B, T, -1).clamp_min(1.0e-6)
        old_mean = old.mean.reshape(B, T, -1).detach()
        old_std = old.std.reshape(B, T, -1).detach().clamp_min(1.0e-6)
        old_var = old_std.pow(2)
        new_var = new_std.pow(2)

        mean_kl_t = torch.mean((new_mean - old_mean).pow(2) / (2.0 * old_var), dim=-1)
        std_kl_t = torch.mean(0.5 * (old_var / new_var - 1.0 + torch.log(new_var / old_var)), dim=-1)
        mean_kl = torch.sum(mean_kl_t * mask) / denom
        std_kl = torch.sum(std_kl_t * mask) / denom

        mean_penalty = F.softplus(self.log_mean_kl_penalty).clamp_min(1.0e-8)
        std_penalty = F.softplus(self.log_std_kl_penalty).clamp_min(1.0e-8)
        actor_loss = mle_loss + mean_penalty.detach() * mean_kl + std_penalty.detach() * std_kl
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()

        dual_loss = mean_penalty * (float(self.config.mean_kl_epsilon) - mean_kl.detach()) + std_penalty * (float(self.config.std_kl_epsilon) - std_kl.detach())
        self.kl_optim.zero_grad(set_to_none=True)
        dual_loss.backward()
        self.kl_optim.step()
        with torch.no_grad():
            self.log_mean_kl_penalty.clamp_(min=-20.0, max=20.0)
            self.log_std_kl_penalty.clamp_(min=-20.0, max=20.0)

        delta_sq = torch.zeros((), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for previous, current in zip(before, self.actor.parameters(), strict=True):
                delta_sq = delta_sq + torch.sum((current.detach() - previous).pow(2))
            weight_entropy = -torch.sum(weights * torch.log(torch.clamp(weights, min=1.0e-12)) * masked) / denom
            sampled_q_spread = torch.sum(torch.std(q_values, dim=0, unbiased=False) * mask) / denom
            action_mean_abs = torch.sum(torch.mean(torch.abs(torch.tanh(new_mean.detach())), dim=-1) * mask) / denom
            action_std_mean = torch.sum(torch.mean(new_std.detach(), dim=-1) * mask) / denom
            weight_max = torch.sum(torch.max(weights, dim=0).values * mask) / denom
            mean_penalty_after = F.softplus(self.log_mean_kl_penalty).clamp_min(1.0e-8)
            std_penalty_after = F.softplus(self.log_std_kl_penalty).clamp_min(1.0e-8)

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
            float(eta.detach().cpu()),
            float(mean_penalty_after.detach().cpu()),
            float(std_penalty_after.detach().cpu()),
        )

    def _solve_e_step_temperature(self, q_centered: Tensor, *, mask: Tensor, epsilon: float) -> Tensor:
        if q_centered.ndim != 3:
            raise ValueError("q_centered must have shape [K,B,T]")
        K = int(q_centered.shape[0])
        target_kl = float(max(epsilon, 0.0))
        max_kl = float(torch.log(torch.as_tensor(float(K), dtype=q_centered.dtype, device=q_centered.device)).detach().cpu())
        if target_kl <= 0.0:
            eta = torch.tensor(1.0e6, dtype=q_centered.dtype, device=q_centered.device)
            self.last_temperature.copy_(eta.detach().to(dtype=self.last_temperature.dtype))
            return eta
        target_kl = min(target_kl, max_kl - 1.0e-6)
        denom = torch.clamp(mask.to(dtype=q_centered.dtype).sum(), min=1.0)
        log_k = torch.log(torch.as_tensor(float(K), dtype=q_centered.dtype, device=q_centered.device))

        def average_kl(eta_value: Tensor) -> Tensor:
            weights = torch.softmax(q_centered / eta_value, dim=0)
            entropy = -torch.sum(weights * torch.log(torch.clamp(weights, min=1.0e-12)), dim=0)
            kl = log_k - entropy
            return torch.sum(kl * mask.to(dtype=q_centered.dtype)) / denom

        low = torch.tensor(1.0e-12, dtype=q_centered.dtype, device=q_centered.device)
        high = torch.tensor(max(float(self.last_temperature.detach().cpu()), 1.0), dtype=q_centered.dtype, device=q_centered.device)
        high_kl = average_kl(high)
        for _ in range(32):
            if float(high_kl.detach().cpu()) <= target_kl:
                break
            high = high * 2.0
            high_kl = average_kl(high)
        low_kl = average_kl(low)
        if float(low_kl.detach().cpu()) < target_kl:
            eta = high
            self.last_temperature.copy_(eta.detach().to(dtype=self.last_temperature.dtype))
            return eta
        for _ in range(48):
            mid = torch.sqrt(low * high)
            mid_kl = average_kl(mid)
            if float(mid_kl.detach().cpu()) > target_kl:
                low = mid
            else:
                high = mid
        eta = high.detach()
        self.last_temperature.copy_(eta.to(dtype=self.last_temperature.dtype))
        return eta

    def _sampled_q_values(self, obs: Tensor, history_action: Tensor, sampled_actions: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Evaluate sampled-action Q values using replay history hidden states."""
        if obs.ndim != 3 or sampled_actions.ndim != 4:
            raise ValueError("_sampled_q_values expects obs [B,T,O] and sampled_actions [K,B,T,A]")
        if history_action.shape[:2] != obs.shape[:2]:
            raise ValueError("history_action must have shape [B,T,A] matching obs")
        return self.critic.evaluate_query_actions_with_history(
            history_obs=obs,
            history_action=history_action,
            query_obs=obs,
            query_action=sampled_actions,
            mask=mask,
            include_current_history=False,
        )

    def _soft_sync(self, tau: float) -> None:
        with torch.no_grad():
            for target, source in zip(self.target_actor.parameters(), self.actor.parameters(), strict=True):
                target.mul_(1.0 - tau).add_(source, alpha=tau)
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
                target.mul_(1.0 - tau).add_(source, alpha=tau)
