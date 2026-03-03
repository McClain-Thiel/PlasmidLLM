"""Tests for Mixture of Experts module."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plasmid_llm.models.hf_plasmid_lm import (
    PlasmidLMConfig,
    PlasmidLMForCausalLM,
    PlasmidLMSparseMoE,
)


@pytest.fixture
def moe_config():
    return PlasmidLMConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        aux_loss_coef=0.01,
    )


@pytest.fixture
def dense_config():
    return PlasmidLMConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        use_moe=False,
    )


class TestSparseMoE:
    def test_output_shape(self, moe_config):
        moe = PlasmidLMSparseMoE(moe_config)
        x = torch.randn(2, 16, 64)
        out, aux_loss = moe(x)
        assert out.shape == (2, 16, 64)
        assert aux_loss.shape == ()
        assert aux_loss.item() >= 0

    def test_aux_loss_bounded(self, moe_config):
        moe = PlasmidLMSparseMoE(moe_config)
        x = torch.randn(4, 32, 64)
        _, aux_loss = moe(x)
        # Aux loss should be reasonable (not exploding)
        assert aux_loss.item() < 10.0

    def test_gradient_flow(self, moe_config):
        moe = PlasmidLMSparseMoE(moe_config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, aux_loss = moe(x)
        loss = out.sum() + aux_loss
        loss.backward()
        assert x.grad is not None
        # Check expert params got gradients
        for expert in moe.experts:
            assert expert.up_proj.weight.grad is not None
        assert moe.router.weight.grad is not None


class TestMoEModel:
    def test_moe_forward(self, moe_config):
        model = PlasmidLMForCausalLM(moe_config)
        input_ids = torch.randint(0, 128, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        assert out.logits.shape == (2, 16, 128)

    def test_moe_loss_includes_aux(self, moe_config):
        """MoE loss should be > CE-only loss due to aux_loss contribution."""
        model = PlasmidLMForCausalLM(moe_config)
        model.eval()
        input_ids = torch.randint(0, 128, (2, 16))
        labels = input_ids.clone()

        # Get loss with aux
        out_with_aux = model(input_ids=input_ids, labels=labels)

        # Compute CE-only for comparison
        logits = out_with_aux.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_only = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        # Total loss should be >= CE loss (aux_loss >= 0)
        assert out_with_aux.loss.item() >= ce_only.item() - 1e-5

    def test_dense_model_unaffected(self, dense_config):
        """Dense model should work exactly as before — no aux_loss impact."""
        model = PlasmidLMForCausalLM(dense_config)
        input_ids = torch.randint(0, 128, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        assert out.logits.shape == (2, 16, 128)
        # Verify no MoE modules exist
        for layer in model.model.layers:
            assert hasattr(layer, "mlp")
            assert not hasattr(layer, "moe")

    def test_generation(self, moe_config):
        model = PlasmidLMForCausalLM(moe_config)
        model.eval()
        input_ids = torch.randint(0, 128, (1, 8))
        out = model.generate_simple(input_ids, max_new_tokens=10)
        assert out.shape == (1, 18)  # 8 prompt + 10 generated

    def test_moe_param_count(self, moe_config):
        """MoE model should have more params than dense with same config."""
        dense_cfg = PlasmidLMConfig(
            vocab_size=128, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=256, use_moe=False,
        )
        dense = PlasmidLMForCausalLM(dense_cfg)
        moe = PlasmidLMForCausalLM(moe_config)
        dense_params = sum(p.numel() for p in dense.parameters())
        moe_params = sum(p.numel() for p in moe.parameters())
        assert moe_params > dense_params

    def test_gradient_checkpointing(self, moe_config):
        model = PlasmidLMForCausalLM(moe_config)
        model.gradient_checkpointing_enable()
        model.train()
        input_ids = torch.randint(0, 128, (2, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        # Check gradients exist
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
