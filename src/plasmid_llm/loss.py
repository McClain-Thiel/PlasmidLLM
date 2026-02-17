"""Loss functions for PlasmidLLM training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal cross-entropy loss: -(1-p_t)^gamma * log(p_t).

    Down-weights easy (high-confidence) predictions so the model focuses
    on harder tokens — e.g. functional element motifs in bulk DNA.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-token CE (no reduction) — includes label smoothing
        ce = F.cross_entropy(
            logits, targets,
            reduction="none",
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        # p_t = probability assigned to the correct class
        p_t = torch.exp(-ce)

        # Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce

        # Mean over valid (non-ignored) tokens
        valid = targets != self.ignore_index
        if valid.any():
            return loss[valid].mean()
        return loss.mean()


def build_loss_fn(train_cfg: Any) -> nn.Module | None:
    """Factory: build loss function from TrainConfig.

    Returns None for standard cross-entropy (models use their existing
    F.cross_entropy default), or a FocalLoss module for focal loss.
    """
    loss_type = getattr(train_cfg, "loss_type", "cross_entropy")

    if loss_type == "cross_entropy":
        label_smoothing = getattr(train_cfg, "label_smoothing", 0.0)
        if label_smoothing > 0.0:
            # Wrap standard CE with label smoothing into a callable module
            return _CrossEntropyLoss(
                label_smoothing=label_smoothing,
                ignore_index=-100,
            )
        return None  # Use default F.cross_entropy in each model

    if loss_type == "focal":
        return FocalLoss(
            gamma=getattr(train_cfg, "focal_gamma", 2.0),
            label_smoothing=getattr(train_cfg, "label_smoothing", 0.0),
            ignore_index=-100,
        )

    raise ValueError(f"Unknown loss_type '{loss_type}'. Use 'cross_entropy' or 'focal'.")


class _CrossEntropyLoss(nn.Module):
    """Thin wrapper around F.cross_entropy with label smoothing."""

    def __init__(self, label_smoothing: float = 0.0, ignore_index: int = -100):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
