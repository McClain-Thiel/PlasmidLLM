"""Transformer variants with FiLM conditioning and causal convolutions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from plasmid_llm.models import register_model
from plasmid_llm.models.transformer import (
    CausalSelfAttention,
    TransformerLM,
    _apply_rope,
    _rope_freqs,
)


# ---------------------------------------------------------------------------
# FiLM variant
# ---------------------------------------------------------------------------


class PromptEncoder(nn.Module):
    """Mean-pool prompt embeddings (positions before SEP) into a conditioning vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, embeddings: torch.Tensor, sep_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, S, D) token embeddings
            sep_positions: (B,) index of SEP token per batch element
        Returns:
            cond: (B, D) conditioning vector
        """
        B, S, D = embeddings.shape
        # Build mask: 1 for positions < sep_position, 0 otherwise
        positions = torch.arange(S, device=embeddings.device).unsqueeze(0)  # (1, S)
        mask = (positions < sep_positions.unsqueeze(1)).float()  # (B, S)
        # Mean pool (avoid div-by-zero with clamp)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        pooled = (embeddings * mask.unsqueeze(2)).sum(dim=1) / lengths  # (B, D)
        return self.proj(pooled)


class FiLMLayer(nn.Module):
    """FiLM modulation: gamma * x + beta, conditioned on a vector c."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gamma_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, d_model)
        # Zero-init weights so FiLM starts as identity (gamma=1, beta=0)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) layer output
            cond: (B, D) conditioning vector
        Returns:
            modulated: (B, S, D)
        """
        gamma = self.gamma_proj(cond).unsqueeze(1)  # (B, 1, D)
        beta = self.beta_proj(cond).unsqueeze(1)  # (B, 1, D)
        return gamma * x + beta


class FiLMTransformerBlock(nn.Module):
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
        self.film = FiLMLayer(d_model)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cond: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv_cache = self.attn(
            self.ln1(x), rope_cos, rope_sin, kv_cache=kv_cache, pos_offset=pos_offset,
        )
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.film(mlp_out, cond)
        return x, new_kv_cache


@register_model("transformer_film")
class TransformerFiLMLM(nn.Module):
    """Decoder-only transformer with FiLM conditioning from prompt tags."""

    def __init__(self, config: Any, vocab_size: int, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn
        d_model = config.d_model
        n_layers = config.n_layers
        n_heads = config.n_heads
        d_ff = config.d_ff
        dropout = config.dropout
        self.sep_token_id = config.sep_token_id

        padded_vocab = ((vocab_size + 7) // 8) * 8
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(padded_vocab, d_model)
        self.drop = nn.Dropout(dropout)
        self.prompt_encoder = PromptEncoder(d_model)
        self.blocks = nn.ModuleList(
            [FiLMTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, padded_vocab, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Precompute RoPE
        head_dim = d_model // n_heads
        cos, sin = _rope_freqs(head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            # Skip FiLM layers — they have their own init
            if "film" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_sep_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Find SEP token position per batch element."""
        # argmax returns first occurrence of True
        matches = input_ids == self.sep_token_id
        # If no SEP found, default to position 0
        sep_pos = matches.long().argmax(dim=1)
        return sep_pos

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        emb = self.drop(self.tok_emb(input_ids))

        sep_positions = self._get_sep_positions(input_ids)
        cond = self.prompt_encoder(emb, sep_positions)

        x = emb
        for block in self.blocks:
            x, _ = block(x, self.rope_cos, self.rope_sin, cond=cond)

        x = self.ln_f(x)
        logits = self.head(x)

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
        """Autoregressive generation with KV cache. Conditioning computed once at prefill."""
        emb = self.drop(self.tok_emb(input_ids))
        sep_positions = self._get_sep_positions(input_ids)
        cond = self.prompt_encoder(emb, sep_positions)

        x = emb
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            x, kv = block(x, self.rope_cos, self.rope_sin, cond=cond, kv_cache=None, pos_offset=0)
            kv_caches.append(kv)
        x = self.ln_f(x)
        logits = self.head(x[:, -1:, :]).squeeze(1)

        cur_len = input_ids.shape[1]

        for _ in range(max_new_tokens):
            scaled_logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(scaled_logits, top_k)
                scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            x = self.drop(self.tok_emb(next_token))
            new_kv_caches = []
            for i, block in enumerate(self.blocks):
                x, kv = block(
                    x, self.rope_cos, self.rope_sin, cond=cond,
                    kv_cache=kv_caches[i], pos_offset=cur_len,
                )
                new_kv_caches.append(kv)
            kv_caches = new_kv_caches
            x = self.ln_f(x)
            logits = self.head(x).squeeze(1)
            cur_len += 1

        return input_ids


# ---------------------------------------------------------------------------
# Conv variant
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """Depthwise causal Conv1d with state for autoregressive decode."""

    def __init__(self, d_model: int, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            groups=d_model, padding=0, bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, S, D)
            conv_state: (B, D, K-1) cached left context for decode
        Returns:
            out: (B, S, D)
            new_conv_state: (B, D, K-1)
        """
        # x -> (B, D, S) for Conv1d
        x_t = x.transpose(1, 2)

        if conv_state is not None:
            # Decode mode: prepend cached state
            x_padded = torch.cat([conv_state, x_t], dim=2)
        else:
            # Training/prefill: left-pad with zeros
            pad = torch.zeros(
                x_t.shape[0], x_t.shape[1], self.kernel_size - 1,
                device=x_t.device, dtype=x_t.dtype,
            )
            x_padded = torch.cat([pad, x_t], dim=2)

        # Save state: last K-1 columns of input (before conv)
        new_conv_state = x_padded[:, :, -(self.kernel_size - 1):].clone()

        out = self.conv(x_padded)  # (B, D, S)
        return out.transpose(1, 2), new_conv_state


class ConvTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.conv = CausalConv1d(d_model, kernel_size)
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
        conv_state: torch.Tensor | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        attn_out, new_kv_cache = self.attn(
            self.ln1(x), rope_cos, rope_sin, kv_cache=kv_cache, pos_offset=pos_offset,
        )
        x = x + attn_out
        conv_out, new_conv_state = self.conv(x, conv_state=conv_state)
        x = x + conv_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv_cache, new_conv_state


@register_model("transformer_conv")
class TransformerConvLM(nn.Module):
    """Decoder-only transformer with causal depthwise convolutions."""

    def __init__(self, config: Any, vocab_size: int, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn
        d_model = config.d_model
        n_layers = config.n_layers
        n_heads = config.n_heads
        d_ff = config.d_ff
        kernel_size = config.conv_kernel_size
        dropout = config.dropout

        padded_vocab = ((vocab_size + 7) // 8) * 8
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(padded_vocab, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ConvTransformerBlock(d_model, n_heads, d_ff, kernel_size, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, padded_vocab, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Precompute RoPE
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
            x, _, _ = block(x, self.rope_cos, self.rope_sin)

        x = self.ln_f(x)
        logits = self.head(x)

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
        """Autoregressive generation with KV cache and conv state."""
        x = self.drop(self.tok_emb(input_ids))
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        conv_states: list[torch.Tensor] = []
        for block in self.blocks:
            x, kv, cs = block(x, self.rope_cos, self.rope_sin, kv_cache=None, conv_state=None, pos_offset=0)
            kv_caches.append(kv)
            conv_states.append(cs)
        x = self.ln_f(x)
        logits = self.head(x[:, -1:, :]).squeeze(1)

        cur_len = input_ids.shape[1]

        for _ in range(max_new_tokens):
            scaled_logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(scaled_logits, top_k)
                scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            x = self.drop(self.tok_emb(next_token))
            new_kv_caches = []
            new_conv_states = []
            for i, block in enumerate(self.blocks):
                x, kv, cs = block(
                    x, self.rope_cos, self.rope_sin,
                    kv_cache=kv_caches[i], conv_state=conv_states[i], pos_offset=cur_len,
                )
                new_kv_caches.append(kv)
                new_conv_states.append(cs)
            kv_caches = new_kv_caches
            conv_states = new_conv_states
            x = self.ln_f(x)
            logits = self.head(x).squeeze(1)
            cur_len += 1

        return input_ids
