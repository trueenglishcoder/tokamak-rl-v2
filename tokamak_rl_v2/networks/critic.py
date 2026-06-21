from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tokamak_rl_v2.networks.initialization import truncated_fanin_init


CRITIC_ACTION_INPUT_KIND = "requested_delta_jdot_v2"


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

    def evaluate_query_actions_with_history(
        self,
        *,
        history_obs: Tensor,
        history_action: Tensor,
        query_obs: Tensor,
        query_action: Tensor,
        mask: Tensor | None = None,
        include_current_history: bool = False,
    ) -> Tensor:
        """Evaluate query actions using hidden state from the real history.

        ``forward`` is correct for scoring an observed action sequence. MPO's
        actor update needs a different operation: rank many candidate actions
        at each replay state while keeping the recurrent state fixed to the
        replay history. Feeding an entire candidate-action sequence through the
        LSTM would give each candidate a fake candidate-action past.
        """

        if history_obs.ndim != 3 or history_action.ndim != 3 or query_obs.ndim != 3:
            raise ValueError("history_obs, history_action, and query_obs must have shape [B,T,*]")
        if query_action.ndim == 3:
            query_action = query_action.unsqueeze(0)
            squeeze_k = True
        elif query_action.ndim == 4:
            squeeze_k = False
        else:
            raise ValueError("query_action must have shape [B,T,A] or [K,B,T,A]")
        B, T, _ = history_obs.shape
        K = int(query_action.shape[0])
        if history_action.shape[:2] != (B, T) or query_obs.shape[:2] != (B, T) or query_action.shape[1:3] != (B, T):
            raise ValueError("query/history batch and time dimensions must match")

        h = torch.zeros((1, B, self.lstm_hidden_dim), dtype=history_obs.dtype, device=history_obs.device)
        c = torch.zeros((1, B, self.lstm_hidden_dim), dtype=history_obs.dtype, device=history_obs.device)
        mask_t = None if mask is None else mask.to(dtype=torch.bool, device=history_obs.device)
        values = []
        for t in range(T):
            active = None if mask_t is None else mask_t[:, t]
            hist_x = torch.cat([history_obs[:, t], torch.clamp(history_action[:, t], -1.0, 1.0)], dim=-1)
            if include_current_history:
                h, c = self._advance_history_state(hist_x, h, c, active)
            q_t = self._evaluate_query_step(query_obs[:, t], query_action[:, :, t], h, c)
            if active is not None:
                q_t = q_t * active.to(dtype=q_t.dtype)[None, :]
            values.append(q_t)
            if not include_current_history:
                h, c = self._advance_history_state(hist_x, h, c, active)
        out = torch.stack(values, dim=2)
        return out.squeeze(0) if squeeze_k else out

    def zero_state(self, batch_size: int, device: torch.device | str) -> CriticState:
        dev = torch.device(device)
        h = torch.zeros((1, int(batch_size), self.lstm_hidden_dim), dtype=torch.float32, device=dev)
        c = torch.zeros((1, int(batch_size), self.lstm_hidden_dim), dtype=torch.float32, device=dev)
        return CriticState(h=h, c=c)

    def _advance_history_state(self, x: Tensor, h: Tensor, c: Tensor, active: Tensor | None) -> tuple[Tensor, Tensor]:
        _y, (new_h, new_c) = self.lstm(x.unsqueeze(1), (h.contiguous(), c.contiguous()))
        if active is None:
            return new_h, new_c
        gate = active.to(dtype=torch.bool, device=x.device).reshape(1, -1, 1)
        return torch.where(gate, new_h, h), torch.where(gate, new_c, c)

    def _evaluate_query_step(self, obs: Tensor, action: Tensor, h: Tensor, c: Tensor) -> Tensor:
        K, B, _ = action.shape
        obs_k = obs.unsqueeze(0).expand(K, B, -1)
        x = torch.cat([obs_k, torch.clamp(action, -1.0, 1.0)], dim=-1)
        x_flat = x.reshape(K * B, -1)
        h0 = h.repeat(1, K, 1).contiguous()
        c0 = c.repeat(1, K, 1).contiguous()
        y, _state = self.lstm(x_flat.unsqueeze(1), (h0, c0))
        y = y.squeeze(1).reshape(K, B, self.lstm_hidden_dim)
        z = torch.cat([x, y], dim=-1)
        z = F.elu(self.fc1(z))
        z = F.elu(self.fc2(z))
        return self.q_head(z).squeeze(-1)
