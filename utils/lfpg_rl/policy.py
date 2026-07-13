from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation_cls: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation_cls())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class LFPGRLPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        hidden_dims: Sequence[int],
        bound_policy_mean: bool = False,
    ) -> None:
        super().__init__()
        if len(tuple(hidden_dims)) == 0:
            raise ValueError("hidden_dims must contain at least one layer size.")
        if action_high <= action_low:
            raise ValueError("action_high must be larger than action_low.")

        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        latent_dim = hidden_dims[-1]
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.bound_policy_mean = bool(bound_policy_mean)

        self.obs_encoder = build_mlp(
            input_dim=self.observation_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=latent_dim,
        )
        self.policy_mean = nn.Linear(latent_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        self.global_value_net = nn.Linear(latent_dim, 1)

    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        return self.obs_encoder(observations)

    def policy_stats(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_latent = self._encode_observations(observations)
        mean_actions = self.policy_mean(obs_latent)
        if self.bound_policy_mean:
            action_span = self.action_high - self.action_low
            mean_actions = self.action_low + action_span * torch.sigmoid(mean_actions)
        log_std = self.log_std.unsqueeze(0).expand_as(mean_actions)
        log_std = torch.clamp(log_std, min=-6.0, max=4.0)
        return mean_actions, log_std

    def get_distribution(
        self,
        observations: torch.Tensor,
    ) -> Normal:
        mean_actions, log_std = self.policy_stats(observations)
        return Normal(mean_actions, log_std.exp())

    def act(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.get_distribution(observations)
        raw_actions = distribution.mean if deterministic else distribution.sample()
        clipped_actions = torch.clamp(raw_actions, min=self.action_low, max=self.action_high)
        log_prob_dims = distribution.log_prob(raw_actions)
        entropy_dims = distribution.entropy()
        return raw_actions, clipped_actions, log_prob_dims, entropy_dims

    def evaluate(
        self,
        observations: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        obs_latent = self._encode_observations(observations)
        mean_actions, log_std = self.policy_stats(observations)
        distribution = Normal(mean_actions, log_std.exp())
        log_prob_dims = distribution.log_prob(raw_actions)
        entropy_dims = distribution.entropy()
        global_value = self.global_value_net(obs_latent).squeeze(-1)

        return {
            "log_prob_dims": log_prob_dims,
            "entropy_dims": entropy_dims,
            "global_value": global_value,
        }
