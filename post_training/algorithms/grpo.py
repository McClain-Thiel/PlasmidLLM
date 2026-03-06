"""GRPO algorithm — Group Relative Policy Optimization (DeepSeek-style).

For each prompt, generates N completions and normalizes rewards within
the group to produce advantages (no value function needed).
"""

from __future__ import annotations

import logging

import ray
import torch

from post_training.algorithms.base import Algorithm, Scorer

log = logging.getLogger(__name__)


class GRPOAlgorithm(Algorithm):
    """GRPO with multi-actor parallel generation and gradient averaging.

    Usage::

        algo = GRPOAlgorithm(kl_coef=0.3, cliprange=0.2, num_generations=8)
        metrics = algo.step(actors, prompts, scorer)
    """

    def __init__(
        self,
        kl_coef: float = 0.1,
        cliprange: float = 0.2,
        num_generations: int = 4,
        micro_batch_size: int = 64,
    ):
        self.kl_coef = kl_coef
        self.cliprange = cliprange
        self.num_generations = num_generations
        self.micro_batch_size = micro_batch_size

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        *,
        group_size: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Group-normalize rewards: (B*G,) → (B*G,) advantages.

        Rewards for the same prompt are contiguous — each group of G
        rewards is independently z-scored.
        """
        G = group_size or self.num_generations
        grouped = rewards.view(-1, G)

        means = grouped.mean(dim=1, keepdim=True)
        stds = grouped.std(dim=1, keepdim=True).clamp(min=1e-8)

        return ((grouped - means) / stds).view(-1)

    def step(self, actors, prompts, scorer) -> dict[str, float]:
        """Full GRPO step with group sampling and optional multi-actor sharding."""
        G = self.num_generations
        n_actors = len(actors)

        expanded = [p for p in prompts for _ in range(G)]
        shards = self._shard(expanded, n_actors)

        # ── Parallel generation across actors ─────────────────────────────
        gen_futures = [a.generate.remote(s) for a, s in zip(actors, shards)]
        generations = ray.get(gen_futures)

        # ── Score all completions ─────────────────────────────────────────
        all_prompts = [p for g in generations for p in g.prompts]
        all_completions = [t for g in generations for t in g.completion_texts]
        rewards = scorer.score(all_prompts, all_completions)
        advantages = self.compute_advantages(rewards, group_size=G)

        # ── Forward-backward on each actor with its shard ─────────────────
        loss_fn_kwargs = dict(cliprange=self.cliprange, kl_coef=self.kl_coef)

        idx = 0
        total_loss = 0.0
        for actor, gen in zip(actors, generations):
            n = len(gen.prompts)
            adv_slice = advantages[idx : idx + n]

            old_lp = ray.get(
                actor.get_log_probs.remote(gen.full_ids, gen.prompt_len),
            )
            ref_lp = ray.get(
                actor.get_log_probs.remote(
                    gen.full_ids, gen.prompt_len, use_ref=True,
                ),
            )

            loss = self.micro_batch_forward_backward(
                actor, gen.full_ids, gen.prompt_len,
                adv_slice, old_lp.per_token, ref_lp.per_token,
                old_lp.mask, "grpo", loss_fn_kwargs, self.micro_batch_size,
            )
            total_loss += loss
            idx += n

        # ── Sync gradients across actors → step ──────────────────────────
        if n_actors > 1:
            self.average_and_apply_gradients(actors)
        else:
            ray.get(actors[0].clip_and_step.remote())

        # ── Metrics ───────────────────────────────────────────────────────
        kl = ray.get(actors[0].get_kl.remote(
            generations[0].full_ids, generations[0].prompt_len,
        ))

        return {
            "loss": total_loss / n_actors,
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_best": (
                rewards.view(-1, G).max(dim=1).values.mean().item()
            ),
            "kl": kl["kl_per_seq"].mean().item(),
            "adv_mean": advantages.mean().item(),
            "adv_std": advantages.std().item(),
        }
