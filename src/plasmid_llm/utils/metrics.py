"""Evaluation metrics for plasmid language models."""

from __future__ import annotations

import math

import torch


def compute_perplexity(avg_loss: float) -> float:
    """Compute perplexity from average cross-entropy loss."""
    return math.exp(min(avg_loss, 100))  # clamp to avoid overflow


def gc_content(sequence: str) -> float:
    """Compute GC content of a DNA sequence."""
    seq = sequence.upper()
    gc = sum(1 for c in seq if c in "GC")
    total = sum(1 for c in seq if c in "ATCGN")
    return gc / max(total, 1)


def sequence_validity(sequence: str) -> float:
    """Fraction of characters that are valid DNA bases."""
    valid = set("ATCGNatcgn")
    total = len(sequence)
    if total == 0:
        return 0.0
    return sum(1 for c in sequence if c in valid) / total
