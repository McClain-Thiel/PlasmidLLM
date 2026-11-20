"""Tests for Mixture of Experts module."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

    @pytest.mark.parametrize("config_name", ["dense_config", "moe_config"])
    def test_cached_generation_preserves_padding_mask(self, config_name, request):
        torch.manual_seed(42)
        model = PlasmidLMForCausalLM(request.getfixturevalue(config_name)).eval()
        short = torch.tensor([[11, 12, 13]])
        batched = torch.tensor([[0, 0, 11, 12, 13], [21, 22, 23, 24, 25]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        logits = []
        hook = model.lm_head.register_forward_hook(
            lambda module, inputs, output: logits.append(output.detach().clone())
        )
        try:
            single = model.generate_simple(short, max_new_tokens=4, top_k=1)
            single_logits = logits.copy()
            logits.clear()
            batch = model.generate_simple(
                batched, attention_mask=mask, max_new_tokens=4, top_k=1,
            )
        finally:
            hook.remove()
        assert len(logits) == len(single_logits)
        for actual, expected in zip(logits, single_logits):
            torch.testing.assert_close(actual[:1], expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(batch[:1, 5:], single[:, 3:])

    @pytest.mark.parametrize("config_name", ["dense_config", "moe_config"])
    def test_training_after_generation(self, config_name, request):
        model = PlasmidLMForCausalLM(request.getfixturevalue(config_name))
        input_ids = torch.randint(0, 128, (2, 8))
        for _ in range(2):
            model.eval()
            # Copy outside inference mode, as the GPU actor does when returning
            # generated IDs to the CPU; this isolates persistent model buffers.
            generated = model.generate_simple(input_ids, max_new_tokens=3).clone()
            model.train()
            model.zero_grad()
            loss = model(input_ids=generated, labels=generated).loss
            loss.backward()
            gradient = model.model.embed_tokens.weight.grad
            assert torch.isfinite(loss)
            assert gradient is not None and torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient) > 0

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
