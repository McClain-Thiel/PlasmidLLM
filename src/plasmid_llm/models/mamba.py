"""Mamba SSM language model using HuggingFace transformers implementation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MambaConfig, MambaForCausalLM

from plasmid_llm.models import register_model


@register_model("mamba")
class MambaLM(nn.Module):
    """Mamba-based language model wrapping HuggingFace's MambaForCausalLM.

    Uses the efficient SSM implementation from transformers which handles
    the selective scan properly without materializing huge intermediate tensors.
    """

    def __init__(self, config: Any, vocab_size: int):
        super().__init__()
        padded_vocab = ((vocab_size + 7) // 8) * 8
        self.vocab_size = vocab_size

        hf_config = MambaConfig(
            vocab_size=padded_vocab,
            hidden_size=config.d_model,
            state_size=config.d_state,
            num_hidden_layers=config.n_layers,
            expand=config.expand,
            conv_kernel=config.d_conv,
            use_mambapy=False,  # use the naive (but correct) implementation as fallback
            tie_word_embeddings=True,
            pad_token_id=0,
        )
        self.mamba = MambaForCausalLM(hf_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.mamba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        result = {"logits": outputs.logits, "loss": outputs.loss}
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        return self.mamba.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
        )
