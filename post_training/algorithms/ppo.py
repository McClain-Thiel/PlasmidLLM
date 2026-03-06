"""PPO algorithm — multiple epochs of clipped surrogate over one batch."""

from __future__ import annotations

import ray
import torch

from post_training.algorithms.base import Algorithm, Scorer


class PPOAlgorithm(Algorithm):
    """PPO with clipped surrogate and optional entropy bonus.

    Generates one batch of completions, collects old/ref log-probs,
    then runs multiple PPO epochs over the same data.
    """

    def __init__(
        self,
        cliprange: float = 0.2,
        entropy_coeff: float = 0.01,
        ppo_epochs: int = 4,
        micro_batch_size: int = 64,
    ):
        self.cliprange = cliprange
        self.entropy_coeff = entropy_coeff
        self.ppo_epochs = ppo_epochs
        self.micro_batch_size = micro_batch_size

    def compute_advantages(self, rewards: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simple z-score normalization."""
        if rewards.numel() > 1:
            return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return rewards

    def step(self, actors, prompts, scorer) -> dict[str, float]:
        actor = actors[0]

        gen = ray.get(actor.generate.remote(prompts))
        rewards = scorer.score(gen.prompts, gen.completion_texts)
        advantages = self.compute_advantages(rewards)

        old_lp = ray.get(actor.get_log_probs.remote(gen.full_ids, gen.prompt_len))
        ref_lp = ray.get(actor.get_log_probs.remote(
            gen.full_ids, gen.prompt_len, use_ref=True,
        ))

        loss_fn_kwargs = dict(
            cliprange=self.cliprange, entropy_coeff=self.entropy_coeff,
        )
        total_loss = 0.0

        for _epoch in range(self.ppo_epochs):
            loss = self.micro_batch_forward_backward(
                actor, gen.full_ids, gen.prompt_len,
                advantages, old_lp.per_token, ref_lp.per_token,
                old_lp.mask, "ppo", loss_fn_kwargs, self.micro_batch_size,
            )
            ray.get(actor.clip_and_step.remote())
            total_loss += loss

        if len(actors) > 1:
            self.sync_actors(actors)

        kl = ray.get(actor.get_kl.remote(gen.full_ids, gen.prompt_len))

        return {
            "loss": total_loss / self.ppo_epochs,
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "kl": kl["kl_per_seq"].mean().item(),
        }
