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
    """REINFORCE with KL penalty.

    advantage = reward - batch_mean(reward)
    loss = -mean(advantage * log_prob) + kl_coef * mean(KL)

    where KL = log_prob - ref_log_prob (per-sequence KL divergence estimate).
    """

    def __init__(self, kl_coef: float = 0.1):
        self.kl_coef = kl_coef

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        return rewards - rewards.mean()

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


ALGORITHM_REGISTRY = {
    "reinforce": REINFORCE,
}


def build_algorithm(name: str, **kwargs) -> Algorithm:
    """Instantiate an algorithm by name."""
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {list(ALGORITHM_REGISTRY.keys())}"
        )
    return ALGORITHM_REGISTRY[name](**kwargs)
