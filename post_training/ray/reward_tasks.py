"""Distributed reward scoring via Ray tasks.

Provides a pluggable reward function registry and fan-out/fan-in scoring
over CPU workers. The motif registry DataFrame and category index are stored
in the Ray object store for zero-copy access by worker tasks.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Protocol

import ray

log = logging.getLogger(__name__)


# ── Reward function protocol & registry ───────────────────────────────────


class RewardFunction(Protocol):
    """Signature for pluggable reward functions."""

    def __call__(self, prompt: str, completion: str, context: dict) -> float: ...


REWARD_REGISTRY: Dict[str, RewardFunction] = {}


def register_reward(name: str, fn: RewardFunction) -> None:
    """Register a reward function by name."""
    REWARD_REGISTRY[name] = fn


def get_reward_fn(name: str) -> RewardFunction:
    """Look up a registered reward function."""
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward function '{name}'. "
            f"Available: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[name]


# ── Built-in: motif alignment reward ─────────────────────────────────────


def motif_alignment_reward(prompt: str, completion: str, context: dict) -> float:
    """Score a single sequence using Smith-Waterman motif alignment.

    Wraps ``compute_reward`` from ``post_training/reward.py`` plus EOS bonus
    and length penalty logic from ``plasmid_reward_fn``.

    Expected context keys:
        lookup_df: pd.DataFrame — loaded motif lookup
        category_index: dict — pre-built category index
        alpha: float — curriculum blending weight
        eos_bonus: float — bonus for proper EOS termination
        length_penalty_threshold: int — penalize sequences longer than this
    """
    from post_training.reward import compute_reward

    lookup_df = context["lookup_df"]
    category_index = context["category_index"]
    alpha = context.get("alpha", 1.0)
    eos_bonus = context.get("eos_bonus", 0.15)
    length_penalty_threshold = context.get("length_penalty_threshold", 3500)

    raw = completion.upper()

    # Check for proper termination before stripping tokens
    has_eos = "<EOS>" in completion or "</s>" in completion

    # Strip special tokens and non-DNA chars
    seq = re.sub(r"<[^>]+>", "", raw)
    seq = re.sub(r"[^ATGCN]", "", seq)

    if len(seq) < 100:
        return 0.0

    motif_reward = compute_reward(
        prompt, seq, lookup_df, alpha=alpha, category_index=category_index
    )

    reward = motif_reward + (eos_bonus if has_eos else 0.0)

    if len(seq) > length_penalty_threshold:
        excess = (len(seq) - length_penalty_threshold) / length_penalty_threshold
        reward *= max(0.5, 1.0 - 0.3 * excess)

    return reward


register_reward("motif_alignment", motif_alignment_reward)


# ── Ray remote scoring tasks ─────────────────────────────────────────────


@ray.remote(num_cpus=1)
def score_sequences_chunk(
    prompts: List[str],
    completions: List[str],
    context: dict,
    reward_fn_name: str,
) -> List[float]:
    """Score a chunk of sequences on a single CPU worker.

    Args:
        prompts: Batch of prompt strings.
        completions: Corresponding generated completions.
        context: Scoring context dict (resolved from ObjectRef by Ray).
        reward_fn_name: Name of registered reward function to use.

    Returns:
        List of scalar rewards.
    """
    fn = get_reward_fn(reward_fn_name)
    return [fn(p, c, context) for p, c in zip(prompts, completions)]


def score_batch_distributed(
    prompts: List[str],
    completions: List[str],
    context_ref: ray.ObjectRef,
    reward_fn_name: str = "motif_alignment",
    chunk_size: int = 16,
) -> List[float]:
    """Fan-out scoring across CPU workers, then fan-in results.

    Splits the batch into chunks of ``chunk_size``, submits each as a Ray
    task, and collects results in order.

    Args:
        prompts: All prompt strings in the batch.
        completions: All generated completions.
        context_ref: ObjectRef to scoring context in Ray object store.
        reward_fn_name: Registered reward function name.
        chunk_size: Sequences per CPU task.

    Returns:
        List of rewards in the same order as inputs.
    """
    n = len(prompts)
    if n == 0:
        return []

    futures = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        futures.append(
            score_sequences_chunk.remote(
                prompts[i:end],
                completions[i:end],
                context_ref,
                reward_fn_name,
            )
        )

    # Collect in order
    chunk_results = ray.get(futures)
    rewards = []
    for chunk in chunk_results:
        rewards.extend(chunk)
    return rewards
