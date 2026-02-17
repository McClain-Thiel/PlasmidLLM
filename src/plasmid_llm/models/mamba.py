"""Mamba SSM language model using native mamba-ssm CUDA kernels."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from plasmid_llm.models import register_model

# Use native mamba-ssm with CUDA kernels (required)
from mamba_ssm import MambaLMHeadModel as _MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig as _MambaConfig


@register_model("mamba")
class MambaLM(nn.Module):
    """Mamba-based language model using native mamba-ssm CUDA kernels.

    Wraps mamba_ssm.MambaLMHeadModel with our standard forward interface
    (input_ids, attention_mask, labels) and loss computation.
    """

    def __init__(self, config: Any, vocab_size: int, loss_fn=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_fn = loss_fn

        ssm_cfg = {
            "d_state": config.d_state,
            "d_conv": config.d_conv,
            "expand": config.expand,
        }

        mamba_config = _MambaConfig(
            d_model=config.d_model,
            n_layer=config.n_layers,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            pad_vocab_size_multiple=8,
            tie_embeddings=True,
        )
        self.mamba = _MambaLMHeadModel(mamba_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # mamba-ssm forward returns (logits,) — no built-in loss
        logits = self.mamba(input_ids).logits

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            if self.loss_fn is not None:
                loss = self.loss_fn(flat_logits, flat_labels)
            else:
                loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Stateful autoregressive generation using Mamba's recurrent state.

        Uses mamba-ssm's built-in generate which carries forward the SSM
        hidden state — O(1) per step instead of reprocessing the full sequence.
        """
        max_length = input_ids.shape[1] + max_new_tokens
        return self.mamba.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )
