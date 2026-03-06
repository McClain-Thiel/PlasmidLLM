"""Abstract base class for sequence scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Scorer(ABC):
    """Base class for scoring generated sequences against expected annotations.

    All scorers follow the same interface:
      1. __init__: load any reference data (motif registries, etc.)
      2. score_sequence: score a single (prompt, sequence) pair → float reward
      3. score_batch: vectorized scoring for RL training loops

    This makes scorers swappable for curriculum learning, ablations, etc.
    """

    @abstractmethod
    def score_sequence(
        self,
        prompt: str,
        sequence: str,
        **kwargs,
    ) -> float:
        """Score a single generated sequence against its prompt.

        Args:
            prompt: The conditioning prompt (with special tokens).
            sequence: The generated DNA sequence (raw, may contain special tokens).

        Returns:
            Scalar reward (higher = better).
        """

    def score_batch(
        self,
        prompts: list[str],
        sequences: list[str],
        **kwargs,
    ) -> list[float]:
        """Score a batch of (prompt, sequence) pairs.

        Default implementation loops over score_sequence. Subclasses may
        override for vectorized/parallel scoring.

        Args:
            prompts: List of prompt strings.
            sequences: List of generated sequences.

        Returns:
            List of scalar rewards.
        """
        return [
            self.score_sequence(p, s, **kwargs)
            for p, s in zip(prompts, sequences)
        ]

    def score_sequence_detailed(
        self,
        prompt: str,
        sequence: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Score with full details (for logging/analysis).

        Default returns just the scalar reward. Subclasses should override
        to return per-motif breakdowns, alignment details, etc.
        """
        return {"reward": self.score_sequence(prompt, sequence, **kwargs)}
