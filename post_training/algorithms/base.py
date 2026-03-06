"""Base Algorithm class and shared helpers.

An Algorithm drives ModelActor(s) through their getter/setter API.
The actor is a GPU worker; the algorithm is the application logic.

Usage::

    actor = ModelActor.remote("McClain/PlasmidLM-kmer6-MoE")
    algo = GRPOAlgorithm(kl_coef=0.3, cliprange=0.2, num_generations=4)

    for prompts in dataloader:
        metrics = algo.step([actor], prompts, scorer)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Protocol

import ray
import torch

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Scorer protocol — reward computation is fully external
# ══════════════════════════════════════════════════════════════════════════


class Scorer(Protocol):
    """Anything that can score completions.  Ray tasks, LLM judges, etc."""

    def score(self, prompts: list[str], completions: list[str]) -> torch.Tensor:
        """Return (B,) reward tensor."""
        ...


# ══════════════════════════════════════════════════════════════════════════
# Base Algorithm
# ══════════════════════════════════════════════════════════════════════════


class Algorithm(ABC):
    """Drives ModelActor(s) through a single training step.

    Subclasses implement:
      - compute_advantages(): raw rewards → advantages
      - step(): full generate → score → train loop for one batch

    The base class provides helpers for common patterns (multi-actor
    gradient averaging, micro-batched forward-backward, etc.).
    """

    @abstractmethod
    def step(
        self,
        actors: list,
        prompts: list[str],
        scorer: Scorer,
    ) -> dict[str, float]:
        """One full training step: generate → score → update.

        Returns metrics dict.
        """

    @abstractmethod
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Raw rewards → advantages."""

    # ── Shared helpers ────────────────────────────────────────────────────

    @staticmethod
    def average_and_apply_gradients(actors: list) -> list:
        """Collect grads from all actors, average, broadcast, step.

        After this call all actors have identical weights.
        Returns list of BackwardResult from each actor.
        """
        all_grads = ray.get([a.get_gradients.remote() for a in actors])

        avg = {}
        for key in all_grads[0]:
            stacked = torch.stack([g[key] for g in all_grads if key in g])
            avg[key] = stacked.mean(dim=0)

        ray.get([a.set_gradients.remote(avg) for a in actors])
        return ray.get([a.clip_and_step.remote() for a in actors])

    @staticmethod
    def sync_actors(actors: list) -> None:
        """Broadcast weights from actors[0] to all others."""
        if len(actors) <= 1:
            return
        weights = ray.get(actors[0].get_weights.remote())
        ray.get([a.load_weights.remote(weights) for a in actors[1:]])

    @staticmethod
    def micro_batch_forward_backward(
        actor,
        full_ids: torch.Tensor,
        prompt_len: int,
        advantages: torch.Tensor,
        old_lp: torch.Tensor,
        ref_lp: torch.Tensor,
        mask: torch.Tensor,
        loss_fn_name: str,
        loss_fn_kwargs: dict | None = None,
        micro_batch_size: int = 64,
    ) -> float:
        """Split a batch into micro-batches, accumulate grads on actor.

        Calls zero_grad once, then forward_backward for each micro-batch
        with accumulation_scale set correctly.  Does NOT step the optimizer.
        Returns the mean loss across micro-batches.
        """
        B = full_ids.shape[0]
        n_mb = max(1, (B + micro_batch_size - 1) // micro_batch_size)

        ray.get(actor.zero_grad.remote())
        total_loss = 0.0

        for i in range(n_mb):
            s = i * micro_batch_size
            e = min(s + micro_batch_size, B)

            result = ray.get(actor.forward_backward.remote(
                full_ids=full_ids[s:e],
                prompt_len=prompt_len,
                advantages=advantages[s:e],
                old_log_probs=old_lp[s:e],
                ref_log_probs=ref_lp[s:e],
                mask=mask[s:e],
                loss_fn_name=loss_fn_name,
                loss_fn_kwargs=loss_fn_kwargs,
                accumulation_scale=n_mb,
            ))
            total_loss += result["loss"] / n_mb

        return total_loss

    @staticmethod
    def _shard(items: list, n: int) -> list[list]:
        """Split items into n roughly-equal shards."""
        k, r = divmod(len(items), n)
        shards, start = [], 0
        for i in range(n):
            end = start + k + (1 if i < r else 0)
            shards.append(items[start:end])
            start = end
        return shards
