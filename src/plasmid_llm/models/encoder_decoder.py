"""Transformer encoder-decoder for plasmid generation.

Encoder processes prompt tokens bidirectionally.
Decoder generates DNA sequence autoregressively with cross-attention to encoder.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from plasmid_llm.models import register_model
from plasmid_llm.models.transformer import _apply_rope, _rope_freqs


class BidirectionalAttention(nn.Module):
    """Bidirectional self-attention for encoder."""

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
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, n_heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Convert key_padding_mask to attn_mask if provided
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) with True for pad tokens
            # Need to expand to (B, 1, 1, S) for broadcasting
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

        # Bidirectional attention (no causal mask)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = attn.transpose(1, 2).reshape(B, S, D)
        return self.dropout(self.out_proj(out))


class CrossAttention(nn.Module):
    """Cross-attention for decoder attending to encoder outputs."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        _, Src_S, _ = encoder_out.shape

        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(encoder_out).reshape(B, Src_S, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Convert encoder_mask to attn_mask if provided
        attn_mask = None
        if encoder_mask is not None:
            # encoder_mask: (B, Src_S) with True for pad tokens
            # Need to expand to (B, 1, 1, Src_S) for broadcasting
            attn_mask = encoder_mask.unsqueeze(1).unsqueeze(2)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = attn.transpose(1, 2).reshape(B, S, D)
        return self.dropout(self.out_proj(out))


class CausalSelfAttention(nn.Module):
    """Causal self-attention for decoder with RoPE."""

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
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q = _apply_rope(q, rope_cos, rope_sin, offset=pos_offset)
        k = _apply_rope(k, rope_cos, rope_sin, offset=pos_offset)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        use_causal = kv_cache is None
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=use_causal,
        )
        out = attn.transpose(1, 2).reshape(B, S, D)
        return self.dropout(self.out_proj(out)), new_kv_cache


class EncoderBlock(nn.Module):
    """Transformer encoder block with bidirectional attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = BidirectionalAttention(d_model, n_heads, dropout)
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
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block with causal self-attn, cross-attn, and MLP."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.ln3 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Self-attention
        attn_out, new_kv_cache = self.self_attn(
            self.ln1(x), rope_cos, rope_sin, kv_cache, pos_offset
        )
        x = x + attn_out

        # Cross-attention to encoder
        x = x + self.cross_attn(self.ln2(x), encoder_out, encoder_mask)

        # MLP
        x = x + self.mlp(self.ln3(x))

        return x, new_kv_cache


@register_model("encoder_decoder")
class EncoderDecoderLM(nn.Module):
    """Transformer encoder-decoder for plasmid generation.

    Encoder: Bidirectional processing of prompt tokens
    Decoder: Autoregressive generation of DNA with cross-attention to encoder
    """

    def __init__(self, config: Any, vocab_size: int, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn
        d_model = config.d_model
        n_enc_layers = config.n_enc_layers
        n_dec_layers = config.n_dec_layers
        n_heads = config.n_heads
        d_ff = config.d_ff
        dropout = config.dropout
        max_len = getattr(config, "max_seq_len", 8192)

        padded_vocab = ((vocab_size + 7) // 8) * 8
        self.vocab_size = padded_vocab  # Use padded vocab for consistency
        self.d_model = d_model
        self.sep_token_id = getattr(config, "sep_token_id", 3)

        # Shared embedding
        self.tok_emb = nn.Embedding(padded_vocab, d_model)
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.enc_pos_emb = nn.Embedding(max_len, d_model)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_enc_layers)
        ])
        self.enc_ln = nn.RMSNorm(d_model)

        # Decoder
        self.dec_pos_emb = nn.Embedding(max_len, d_model)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_dec_layers)
        ])
        self.dec_ln = nn.RMSNorm(d_model)

        # Output head (shared with embedding)
        self.head = nn.Linear(d_model, padded_vocab, bias=False)
        self.head.weight = self.tok_emb.weight

        # RoPE for decoder
        head_dim = d_model // n_heads
        cos, sin = _rope_freqs(head_dim, max_len=max_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode prompt tokens.

        Args:
            prompt_ids: (B, prompt_len) token IDs
            prompt_mask: (B, prompt_len) attention mask (1=real, 0=pad)
        Returns:
            encoder_out: (B, prompt_len, d_model)
        """
        B, S = prompt_ids.shape
        positions = torch.arange(S, device=prompt_ids.device).unsqueeze(0)

        x = self.tok_emb(prompt_ids) + self.enc_pos_emb(positions)
        x = self.dropout(x)

        # Invert mask for PyTorch (True = mask)
        key_padding_mask = ~prompt_mask.bool() if prompt_mask is not None else None

        for block in self.encoder_blocks:
            x = block(x, key_padding_mask)

        return self.enc_ln(x)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        kv_caches: list | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, list]:
        """Decode target sequence.

        Args:
            tgt_ids: (B, tgt_len) target token IDs
            encoder_out: (B, src_len, d_model) encoder outputs
            encoder_mask: (B, src_len) encoder padding mask
            kv_caches: list of cached KV tensors for each layer
            pos_offset: position offset for generation
        Returns:
            logits: (B, tgt_len, vocab_size)
            new_kv_caches: list of updated KV caches
        """
        B, S = tgt_ids.shape
        positions = torch.arange(S, device=tgt_ids.device).unsqueeze(0) + pos_offset

        x = self.tok_emb(tgt_ids) + self.dec_pos_emb(positions)
        x = self.dropout(x)

        new_kv_caches = []
        for i, block in enumerate(self.decoder_blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(
                x, encoder_out,
                self.rope_cos, self.rope_sin,
                encoder_mask, layer_cache, pos_offset
            )
            new_kv_caches.append(new_cache)

        x = self.dec_ln(x)
        logits = self.head(x)
        return logits, new_kv_caches

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training using teacher forcing.

        Args:
            input_ids: (B, S) full sequence (prompt + SEP + completion)
            attention_mask: (B, S) mask
            labels: (B, S) labels with -100 for prompt portion
        Returns:
            dict with logits and loss
        """
        B, S = input_ids.shape

        # Split into prompt and target at SEP token
        sep_mask = (input_ids == self.sep_token_id)
        sep_positions = sep_mask.long().argmax(dim=1)  # (B,)

        # Encode prompts
        max_prompt_len = sep_positions.max().item() + 1
        prompt_ids_list = []
        prompt_mask_list = []

        for b in range(B):
            sep_pos = sep_positions[b].item() + 1  # Include SEP
            prompt_ids_list.append(input_ids[b, :sep_pos])
            prompt_mask_list.append(attention_mask[b, :sep_pos] if attention_mask is not None else torch.ones(sep_pos, device=input_ids.device))

        # Pad prompts to same length
        prompt_ids = torch.full((B, max_prompt_len), 0, device=input_ids.device, dtype=torch.long)
        prompt_mask = torch.zeros((B, max_prompt_len), device=input_ids.device)

        for b in range(B):
            p_len = len(prompt_ids_list[b])
            prompt_ids[b, :p_len] = prompt_ids_list[b]
            prompt_mask[b, :p_len] = prompt_mask_list[b]

        # Encode
        encoder_out = self.encode(prompt_ids, prompt_mask)

        # Get target portion (after SEP)
        tgt_ids_list = []
        tgt_labels_list = []
        max_tgt_len = 0

        for b in range(B):
            sep_pos = sep_positions[b].item() + 1
            tgt_ids_list.append(input_ids[b, sep_pos:])
            if labels is not None:
                tgt_labels_list.append(labels[b, sep_pos:])
            max_tgt_len = max(max_tgt_len, S - sep_pos)

        # Pad targets and prepare decoder input (shift right, prepend BOS)
        # Decoder input: [BOS] + target[:-1]
        # Decoder should predict: target
        bos_id = 1  # <BOS> token id
        dec_input_ids = torch.full((B, max_tgt_len), bos_id, device=input_ids.device, dtype=torch.long)
        tgt_labels_padded = torch.full((B, max_tgt_len), -100, device=input_ids.device, dtype=torch.long)

        for b in range(B):
            t_len = len(tgt_ids_list[b])
            if t_len > 0:
                # Decoder input: [BOS] + target[:-1]
                dec_input_ids[b, 0] = bos_id
                if t_len > 1:
                    dec_input_ids[b, 1:t_len] = tgt_ids_list[b][:-1]
                # Labels: target (for loss computation)
                if labels is not None and len(tgt_labels_list[b]) > 0:
                    tgt_labels_padded[b, :len(tgt_labels_list[b])] = tgt_labels_list[b]

        # Decode with teacher forcing
        tgt_logits, _ = self.decode(dec_input_ids, encoder_out, prompt_mask)

        # Build full sequence logits aligned with input
        # Prompt portion: zeros (not used for loss)
        prompt_logits = torch.zeros(B, max_prompt_len, self.vocab_size, device=input_ids.device, dtype=tgt_logits.dtype)
        
        # Combine: prompt logits + target logits
        full_logits = torch.cat([prompt_logits, tgt_logits], dim=1)

        # Pad to input sequence length if needed
        if full_logits.shape[1] < S:
            pad_len = S - full_logits.shape[1]
            pad_logits = torch.zeros(B, pad_len, self.vocab_size, device=input_ids.device, dtype=full_logits.dtype)
            full_logits = torch.cat([full_logits, pad_logits], dim=1)
        elif full_logits.shape[1] > S:
            full_logits = full_logits[:, :S, :]

        # Compute loss
        loss = None
        if labels is not None:
            # Create proper labels for the target portion
            full_labels = torch.full((B, S), -100, device=input_ids.device, dtype=torch.long)
            for b in range(B):
                sep_pos = sep_positions[b].item() + 1
                t_len = len(tgt_labels_list[b]) if labels is not None else 0
                if t_len > 0 and sep_pos + t_len <= S:
                    full_labels[b, sep_pos:sep_pos+t_len] = tgt_labels_padded[b, :t_len]

            # Shift for next-token prediction
            shift_logits = full_logits[:, :-1, :].contiguous()
            shift_labels = full_labels[:, 1:].contiguous()
            
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            if self.loss_fn is not None:
                loss = self.loss_fn(flat_logits, flat_labels)
            else:
                loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

        return {"logits": full_logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with prompt encoding.

        Args:
            input_ids: (B, prompt_len) prompt tokens (including SEP)
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
        Returns:
            (B, prompt_len + max_new_tokens) generated sequence
        """
        B = input_ids.shape[0]

        # Create prompt mask (all 1s)
        prompt_mask = torch.ones_like(input_ids, dtype=torch.float)

        # Encode prompt once
        encoder_out = self.encode(input_ids, prompt_mask)

        # Start generation with BOS (or first pad)
        generated = torch.full((B, 1), 1, device=input_ids.device, dtype=torch.long)  # BOS token
        kv_caches = None
        cur_len = 1

        for _ in range(max_new_tokens):
            # Decode one step
            logits, kv_caches = self.decode(
                generated[:, -1:],  # Only last token
                encoder_out,
                prompt_mask,
                kv_caches,
                pos_offset=cur_len - 1,
            )
            logits = logits[:, -1, :]  # (B, vocab)

            # Sample
            scaled_logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(scaled_logits, top_k)
                scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            cur_len += 1

        return generated
