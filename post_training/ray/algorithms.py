"""RL algorithm implementations (pure PyTorch, no framework dependency).

Each algorithm computes advantages from rewards and a policy gradient loss
from log probabilities. Easy to test in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Algorithm(ABC):
    """Base class for RL algorithms."""

    @abstractmethod
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sequence advantages from rewards and log probs.

        Args:
            rewards: (batch,) scalar rewards per sequence.
            log_probs: (batch,) sum of log probs under current policy.
            ref_log_probs: (batch,) sum of log probs under reference policy.

        Returns:
            (batch,) advantage estimates.
        """

    @abstractmethod
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar policy gradient loss.

        Args:
            log_probs: (batch,) current policy log probs.
            old_log_probs: (batch,) log probs from rollout time.
            advantages: (batch,) advantage estimates.
            ref_log_probs: (batch,) reference policy log probs.

        Returns:
            Scalar loss tensor.
        """


class REINFORCE(Algorithm):
    """REINFORCE with KL penalty and exponential moving average baseline.

    advantage = reward - ema_baseline
    loss = -mean(advantage * log_prob) + kl_coef * mean(KL)

    where KL = log_prob - ref_log_prob (per-sequence KL divergence estimate).
    The EMA baseline avoids zero-advantage collapse when a full batch has
    identical (e.g. all-zero) rewards.
    """

    def __init__(self, kl_coef: float = 0.1, baseline_ema: float = 0.95):
        self.kl_coef = kl_coef
        self.baseline_ema = baseline_ema
        self._baseline = None

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        batch_mean = rewards.mean().item()
        if self._baseline is None:
            self._baseline = batch_mean
        else:
            self._baseline = (
                self.baseline_ema * self._baseline
                + (1 - self.baseline_ema) * batch_mean
            )
        return rewards - self._baseline

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        # Policy gradient: -E[A * log pi]
        pg_loss = -(advantages.detach() * log_probs).mean()

        # KL penalty: E[log pi - log pi_ref]
        kl = (log_probs - ref_log_probs).mean()

        return pg_loss + self.kl_coef * kl


class GRPO(Algorithm):
    """Group Relative Policy Optimization (DeepSeek-style).

    Generates G completions per prompt, computes group-relative advantages
    (zero-mean, unit-variance within each prompt's group), and uses a clipped
    policy gradient loss with KL penalty.

    advantage_i = (reward_i - mean(group)) / std(group)
    loss = -mean(min(ratio * A, clip(ratio) * A)) + beta * KL
    """

    def __init__(
        self,
        kl_coef: float = 0.05,
        cliprange: float = 0.2,
        num_generations: int = 4,
    ):
        self.kl_coef = kl_coef
        self.cliprange = cliprange
        self.num_generations = num_generations

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        G = self.num_generations
        grouped = rewards.view(-1, G)

        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = (grouped - mean) / std

        return advantages.view(-1)

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        # Clipped policy gradient (PPO-style)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)

        pg_loss1 = -advantages.detach() * ratio
        pg_loss2 = -advantages.detach() * clipped
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # KL penalty
        kl = (log_probs - ref_log_probs).mean()

        return pg_loss + self.kl_coef * kl


ALGORITHM_REGISTRY = {
    "reinforce": REINFORCE,
    "grpo": GRPO,
}


def build_algorithm(name: str, **kwargs) -> Algorithm:
    """Instantiate an algorithm by name."""
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {list(ALGORITHM_REGISTRY.keys())}"
        )
    return ALGORITHM_REGISTRY[name](**kwargs)
