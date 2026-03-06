"""PPO algorithm — multiple epochs of clipped surrogate over one batch."""

from __future__ import annotations

import logging

import ray
import torch

from post_training.algorithms.base import Algorithm, Scorer
from post_training.common.utils import timer, wandb_log

log = logging.getLogger(__name__)


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
        self._global_step = 0

    def compute_advantages(self, rewards: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simple z-score normalization."""
        if rewards.numel() > 1:
            return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return rewards

    def step(self, actors, prompts, scorer) -> dict[str, float]:
        self._global_step += 1
        actor = actors[0]

        # ── Generate ──────────────────────────────────────────────────────
        log.info(
            "Step %d: generating completions for %d prompts",
            self._global_step, len(prompts),
        )
        with timer("generation") as t_gen:
            gen = ray.get(actor.generate.remote(prompts))

        comp_lengths = [len(t) for t in gen.completion_texts]
        log.info(
            "Generation done in %.1fs — %d completions, mean_len=%.0f chars",
            t_gen(), len(gen.completion_texts),
            sum(comp_lengths) / max(len(comp_lengths), 1),
        )

        # ── Score ─────────────────────────────────────────────────────────
        with timer("scoring") as t_score:
            rewards = scorer.score(gen.prompts, gen.completion_texts)
        advantages = self.compute_advantages(rewards)
        log.info(
            "Scoring done in %.1fs — mean_reward=%.3f std=%.3f",
            t_score(), rewards.mean().item(), rewards.std().item(),
        )

        # ── Collect old/ref log-probs once ────────────────────────────────
        old_lp = ray.get(actor.get_log_probs.remote(gen.full_ids, gen.prompt_len))
        ref_lp = ray.get(actor.get_log_probs.remote(
            gen.full_ids, gen.prompt_len, use_ref=True,
        ))

        # ── PPO epochs ────────────────────────────────────────────────────
        loss_fn_kwargs = dict(
            cliprange=self.cliprange, entropy_coeff=self.entropy_coeff,
        )
        total_loss = 0.0
        epoch_losses = []

        with timer("train") as t_train:
            for epoch in range(self.ppo_epochs):
                loss = self.micro_batch_forward_backward(
                    actor, gen.full_ids, gen.prompt_len,
                    advantages, old_lp.per_token, ref_lp.per_token,
                    old_lp.mask, "ppo", loss_fn_kwargs, self.micro_batch_size,
                )
                result = ray.get(actor.clip_and_step.remote())
                total_loss += loss
                epoch_losses.append(loss)
                log.debug(
                    "  PPO epoch %d/%d: loss=%.4f grad_norm=%.4f",
                    epoch + 1, self.ppo_epochs, loss, result.grad_norm,
                )

            if len(actors) > 1:
                self.sync_actors(actors)

        backward_result = result
        log.info(
            "Train done in %.1fs — %d PPO epochs, mean_loss=%.4f grad_norm=%.4f lr=%.2e",
            t_train(), self.ppo_epochs, total_loss / self.ppo_epochs,
            backward_result.grad_norm, backward_result.lr,
        )

        # ── Metrics ───────────────────────────────────────────────────────
        kl = ray.get(actor.get_kl.remote(gen.full_ids, gen.prompt_len))

        metrics = {
            "loss": total_loss / self.ppo_epochs,
            "loss_first_epoch": epoch_losses[0],
            "loss_last_epoch": epoch_losses[-1],
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_min": rewards.min().item(),
            "reward_max": rewards.max().item(),
            "kl": kl["kl_per_seq"].mean().item(),
            "grad_norm": backward_result.grad_norm,
            "lr": backward_result.lr,
            "mean_completion_len": sum(comp_lengths) / max(len(comp_lengths), 1),
            "time_generation": t_gen(),
            "time_scoring": t_score(),
            "time_train": t_train(),
            "time_total": t_gen() + t_score() + t_train(),
        }

        wandb_log(metrics, step=self._global_step, prefix="ppo/")

        return metrics
