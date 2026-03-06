"""Tests for loss functions in LOSS_REGISTRY.

All tests are pure tensor math — no GPU, no Ray, no model loading.
"""

import pytest
import torch

from post_training.common.losses import (
    LOSS_REGISTRY,
    grpo_loss,
    ppo_loss,
    register_loss,
    reinforce_loss,
)


@pytest.fixture
def batch():
    """Fake per-token log-probs and mask for B=4, T=8."""
    torch.manual_seed(0)
    B, T = 4, 8
    log_probs = torch.randn(B, T) * 0.1
    old_log_probs = log_probs.detach().clone()
    ref_log_probs = torch.randn(B, T) * 0.1
    advantages = torch.randn(B)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -2:] = False
    return {
        "log_probs": log_probs,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "advantages": advantages,
        "mask": mask,
    }


class TestLossRegistry:
    def test_builtin_losses_registered(self):
        assert "reinforce" in LOSS_REGISTRY
        assert "ppo" in LOSS_REGISTRY
        assert "grpo" in LOSS_REGISTRY

    def test_register_custom_loss(self):
        @register_loss("_test_custom")
        def custom_loss(log_probs, old_log_probs, ref_log_probs,
                        advantages, mask, **kw):
            return torch.tensor(42.0)

        assert "_test_custom" in LOSS_REGISTRY
        result = LOSS_REGISTRY["_test_custom"](
            torch.zeros(1), torch.zeros(1), torch.zeros(1),
            torch.zeros(1), torch.ones(1, dtype=torch.bool),
        )
        assert result.item() == 42.0
        del LOSS_REGISTRY["_test_custom"]


class TestReinforceLoss:
    def test_returns_scalar(self, batch):
        loss = reinforce_loss(**batch)
        assert loss.dim() == 0

    def test_zero_advantage_gives_zero_loss(self, batch):
        batch["advantages"] = torch.zeros_like(batch["advantages"])
        loss = reinforce_loss(**batch)
        assert abs(loss.item()) < 1e-6

    def test_gradient_flows(self, batch):
        batch["log_probs"].requires_grad_(True)
        loss = reinforce_loss(**batch)
        loss.backward()
        assert batch["log_probs"].grad is not None


class TestPPOLoss:
    def test_returns_scalar(self, batch):
        loss = ppo_loss(**batch)
        assert loss.dim() == 0

    def test_clipping_matters(self, batch):
        batch["log_probs"] = batch["old_log_probs"] + 0.5
        loss_tight = ppo_loss(**batch, cliprange=0.1)
        loss_loose = ppo_loss(**batch, cliprange=0.9)
        assert loss_tight != loss_loose

    def test_identical_old_new_ratio_is_one(self, batch):
        batch["log_probs"] = batch["old_log_probs"].clone()
        loss = ppo_loss(**batch, cliprange=0.2)
        assert torch.isfinite(loss)


class TestGRPOLoss:
    def test_returns_scalar(self, batch):
        loss = grpo_loss(**batch)
        assert loss.dim() == 0

    def test_kl_coef_zero_removes_kl(self, batch):
        loss_no_kl = grpo_loss(**batch, kl_coef=0.0)
        loss_with_kl = grpo_loss(**batch, kl_coef=1.0)
        assert loss_no_kl != loss_with_kl

    def test_mask_respected(self, batch):
        loss_full = grpo_loss(**batch)
        batch["mask"][:, :4] = False
        loss_half = grpo_loss(**batch)
        assert loss_full != loss_half

    def test_gradient_flows(self, batch):
        batch["log_probs"].requires_grad_(True)
        loss = grpo_loss(**batch)
        loss.backward()
        assert batch["log_probs"].grad is not None
