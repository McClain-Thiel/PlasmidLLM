"""GPT-2 style decoder-only transformer with Rotary Positional Embeddings (RoPE)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from plasmid_llm.models import register_model


def _rope_freqs(dim: int, max_len: int = 16384, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables (real-valued, bf16-safe)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)  # (max_len, dim/2)
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Apply rotary embeddings using real-valued rotation (bf16-safe)."""
    # x: (B, n_heads, S, head_dim)
    S = x.shape[2]
    cos = cos[offset:offset + S].unsqueeze(0).unsqueeze(0)  # (1, 1, S, dim/2)
    sin = sin[offset:offset + S].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


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
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, S, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to q, k with position offset
        q = _apply_rope(q, rope_cos, rope_sin, offset=pos_offset)
        k = _apply_rope(k, rope_cos, rope_sin, offset=pos_offset)

        # Append to KV cache if provided
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # Scaled dot-product attention with causal mask
        # When using KV cache for single-token decode, no causal mask needed
        # (the new token can attend to all cached positions)
        use_causal = mask is None and kv_cache is None
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=use_causal,
        )
        out = attn.transpose(1, 2).reshape(B, S, D)
        return self.dropout(self.out_proj(out)), new_kv_cache


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

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv_cache = self.attn(
            self.ln1(x), rope_cos, rope_sin, kv_cache=kv_cache, pos_offset=pos_offset,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv_cache


@register_model("transformer")
class TransformerLM(nn.Module):
    """Decoder-only transformer language model with RoPE."""

    def __init__(self, config: Any, vocab_size: int, loss_fn=None):
        super().__init__()
        d_model = config.d_model
        n_layers = config.n_layers
        n_heads = config.n_heads
        d_ff = config.d_ff
        dropout = config.dropout
        self.loss_fn = loss_fn

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

        # Precompute RoPE cos/sin tables
        head_dim = d_model // n_heads
        cos, sin = _rope_freqs(head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

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
            x, _ = block(x, self.rope_cos, self.rope_sin)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            # Shift: predict next token
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
        """Autoregressive generation with KV cache."""
        # Prefill: process the full prompt, build KV cache
        x = self.drop(self.tok_emb(input_ids))
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            x, kv = block(x, self.rope_cos, self.rope_sin, kv_cache=None, pos_offset=0)
            kv_caches.append(kv)
        x = self.ln_f(x)
        logits = self.head(x[:, -1:, :]).squeeze(1)  # (B, vocab)

        cur_len = input_ids.shape[1]

        for _ in range(max_new_tokens):
            # Sample next token
            scaled_logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(scaled_logits, top_k)
                scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode step: only process the new token with KV cache
            x = self.drop(self.tok_emb(next_token))  # (B, 1, d_model)
            new_kv_caches = []
            for i, block in enumerate(self.blocks):
                x, kv = block(
                    x, self.rope_cos, self.rope_sin,
                    kv_cache=kv_caches[i], pos_offset=cur_len,
                )
                new_kv_caches.append(kv)
            kv_caches = new_kv_caches
            x = self.ln_f(x)
            logits = self.head(x).squeeze(1)  # (B, vocab)
            cur_len += 1

        return input_ids
