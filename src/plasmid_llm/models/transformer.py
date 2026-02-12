"""GPT-2 style decoder-only transformer with Rotary Positional Embeddings (RoPE)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from plasmid_llm.models import register_model


def _rope_freqs(dim: int, max_len: int = 8192, base: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to query/key tensors."""
    # x: (B, n_heads, S, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[: x.shape[2]].unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, rope_freqs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, S, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to q, k
        q = _apply_rope(q, rope_freqs)
        k = _apply_rope(k, rope_freqs)

        # Scaled dot-product attention with causal mask
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=mask is None,
        )
        out = attn.transpose(1, 2).reshape(B, S, D)
        return self.dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope_freqs)
        x = x + self.mlp(self.ln2(x))
        return x


@register_model("transformer")
class TransformerLM(nn.Module):
    """Decoder-only transformer language model with RoPE."""

    def __init__(self, config: Any, vocab_size: int):
        super().__init__()
        d_model = config.d_model
        n_layers = config.n_layers
        n_heads = config.n_heads
        d_ff = config.d_ff
        dropout = config.dropout

        # Pad vocab to multiple of 8 for CUDA bf16 alignment
        padded_vocab = ((vocab_size + 7) // 8) * 8
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(padded_vocab, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, padded_vocab, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies
        head_dim = d_model // n_heads
        self.register_buffer("rope_freqs", _rope_freqs(head_dim), persistent=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.drop(self.tok_emb(input_ids))

        for block in self.blocks:
            x = block(x, self.rope_freqs)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)["logits"][:, -1, :]
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
