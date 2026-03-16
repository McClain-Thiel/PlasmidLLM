"""HuggingFace-compatible PlasmidLM model for use with AutoModelForCausalLM and vLLM."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_plasmid_lm import PlasmidLMConfig
from .moe import PlasmidLMSparseMoE


def _rope_freqs(dim: int, max_len: int, base: float) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    S = x.shape[2]
    cos = cos[offset:offset + S].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + S].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class PlasmidLMAttention(nn.Module):
    def __init__(self, config: PlasmidLMConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        dtype = q.dtype
        q = _apply_rope(q, rope_cos, rope_sin, offset=position_offset).to(dtype)
        k = _apply_rope(k, rope_cos, rope_sin, offset=position_offset).to(dtype)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_kv = (k, v)

        if attention_mask is not None:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            use_causal = past_key_value is None
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)
        out = attn.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out), new_kv


class PlasmidLMMLP(nn.Module):
    def __init__(self, config: PlasmidLMConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class PlasmidLMDecoderLayer(nn.Module):
    def __init__(self, config: PlasmidLMConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = PlasmidLMAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_moe = config.use_moe
        if self.use_moe:
            self.moe = PlasmidLMSparseMoE(config)
        else:
            self.mlp = PlasmidLMMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, new_kv = self.self_attn(hidden_states, rope_cos, rope_sin, past_key_value, position_offset, attention_mask)
        hidden_states = residual + attn_out

        residual = hidden_states
        if self.use_moe:
            moe_out, aux_loss = self.moe(self.post_attention_layernorm(hidden_states))
            hidden_states = residual + moe_out
        else:
            hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
        return hidden_states, new_kv, aux_loss


class PlasmidLMPreTrainedModel(PreTrainedModel):
    config_class = PlasmidLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PlasmidLMModel):
            module.gradient_checkpointing = value


class PlasmidLMModel(PlasmidLMPreTrainedModel):
    """Base model (backbone) — returned by AutoModel."""

    def __init__(self, config: PlasmidLMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([PlasmidLMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Lazy RoPE: computed on first forward call to ensure correct device
        # placement after from_pretrained (which uses meta device tensors).
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

        self.gradient_checkpointing = False
        self.post_init()

    def _init_rope(self, device: torch.device) -> None:
        """Compute and cache RoPE cos/sin on the given device."""
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cos, sin = _rope_freqs(head_dim, self.config.max_position_embeddings, self.config.rope_theta)
        self.register_buffer("rope_cos", cos.to(device), persistent=False)
        self.register_buffer("rope_sin", sin.to(device), persistent=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _build_4d_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        past_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Build a 4D causal+padding mask for SDPA.

        Returns (B, 1, S, S+past) float mask with 0 for attend and -inf for ignore,
        or None if no masking is needed (no padding, no past KV).
        """
        if attention_mask is None and past_seq_len == 0:
            # No padding and no cache — SDPA's is_causal=True handles this
            return None

        total_len = past_seq_len + seq_len
        # Causal mask: each query position can attend to itself and all prior positions
        causal = torch.triu(
            torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype),
            diagonal=past_seq_len + 1,
        )  # (S, S+past)
        mask_4d = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S+past)

        if attention_mask is not None:
            # attention_mask is (B, total_len) with 1=attend, 0=ignore
            # Use a large finite negative instead of -inf for padding mask.
            # With left-padding, the first padding positions can only attend
            # to other padding positions (causal blocks future). If we use
            # -inf, ALL keys are blocked → softmax([-inf,...]) = NaN.
            # Using min_dtype keeps at least the self-attention score finite,
            # so softmax produces a valid (though meaningless) output.
            min_dtype = torch.finfo(dtype).min
            pad_mask = torch.where(
                attention_mask[:, None, None, :].bool(),
                torch.zeros(1, device=device, dtype=dtype),
                torch.tensor(min_dtype, device=device, dtype=dtype),
            )  # (B, 1, 1, total_len)
            mask_4d = mask_4d + pad_mask

        return mask_4d

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        position_offset: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, list, torch.Tensor]:
        # Lazy RoPE init: compute on first forward for correct device placement
        if self.rope_cos is None:
            self._init_rope(input_ids.device)

        hidden_states = self.embed_tokens(input_ids)

        past_seq_len = past_key_values[0][0].shape[2] if past_key_values else 0
        mask_4d = self._build_4d_attention_mask(
            attention_mask, input_ids.shape[1], past_seq_len,
            input_ids.device, hidden_states.dtype,
        )

        new_kv_caches = []
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing recomputes activations on backward — no past_kv during training
                def make_ckpt_fn(l):
                    def fn(h, cos, sin):
                        out, kv, aux = l(h, cos, sin, None, 0)
                        return out, kv[0], kv[1], aux
                    return fn
                hidden_states, k, v, layer_aux = torch.utils.checkpoint.checkpoint(
                    make_ckpt_fn(layer), hidden_states, self.rope_cos, self.rope_sin,
                    use_reentrant=False,
                )
                new_kv = (k, v)
            else:
                hidden_states, new_kv, layer_aux = layer(
                    hidden_states, self.rope_cos, self.rope_sin, past_kv, position_offset, mask_4d
                )
            new_kv_caches.append(new_kv)
            total_aux_loss = total_aux_loss + layer_aux
        hidden_states = self.norm(hidden_states)
        return hidden_states, new_kv_caches, total_aux_loss


class PlasmidLMForCausalLM(PlasmidLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: PlasmidLMConfig):
        super().__init__(config)
        self.model = PlasmidLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        has_cache = False
        if past_key_values is not None:
            if isinstance(past_key_values, DynamicCache):
                has_cache = past_key_values.get_seq_length() > 0
            elif isinstance(past_key_values, list):
                has_cache = len(past_key_values) > 0 and past_key_values[0] is not None
        if has_cache:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    def _convert_cache_to_list(self, past_key_values) -> Optional[list]:
        """Convert DynamicCache to list of (k, v) tuples for our model."""
        if past_key_values is None:
            return None
        if isinstance(past_key_values, list):
            return past_key_values
        if isinstance(past_key_values, DynamicCache):
            if past_key_values.get_seq_length() == 0:
                return None
            return [(layer.keys, layer.values) for layer in past_key_values.layers]
        return None

    def _convert_list_to_cache(self, kv_list: list) -> DynamicCache:
        """Convert list of (k, v) tuples to DynamicCache."""
        cache = DynamicCache()
        for i, (k, v) in enumerate(kv_list):
            cache.update(k, v, layer_idx=i)
        return cache

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        kv_list = self._convert_cache_to_list(past_key_values)

        position_offset = 0
        if kv_list is not None:
            position_offset = kv_list[0][0].shape[2]

        hidden_states, new_kv_list, aux_loss = self.model(
            input_ids, kv_list, position_offset, attention_mask=attention_mask
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + self.config.aux_loss_coef * aux_loss

        new_cache = None
        if use_cache:
            new_cache = self._convert_list_to_cache(new_kv_list)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_cache,
        )

    @torch.no_grad()
    def generate_simple(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.0,
        attention_mask: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Fast autoregressive generation with KV cache, top-p/top-k, and EOS stopping.

        Bypasses HF GenerationMixin overhead for significantly faster decoding
        on small models where Python/kernel-launch overhead dominates.

        Args:
            input_ids: (B, S) token IDs, left-padded if batched.
            max_new_tokens: Maximum tokens to generate per sequence.
            temperature: Sampling temperature.
            top_k: If >0, keep only top-k logits before sampling.
            top_p: If >0, nucleus sampling — keep smallest set with cumulative prob >= top_p.
            attention_mask: (B, S) mask with 1=real, 0=pad. Required for left-padded inputs.
            eos_token_id: If set, stop each sequence when this token is generated.
            pad_token_id: Token to fill finished sequences with. Defaults to eos_token_id.
        """
        B = input_ids.shape[0]
        device = input_ids.device

        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else 0

        # Prefill — pass attention_mask so left-padding is handled correctly
        hidden_states, kv_caches, _ = self.model(
            input_ids, attention_mask=attention_mask,
        )
        logits = self.lm_head(hidden_states[:, -1:, :]).squeeze(1)  # (B, V)
        cur_len = input_ids.shape[1]

        # Track which sequences have finished
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        generated_tokens = []
        for _ in range(max_new_tokens):
            scaled = logits.float() / max(temperature, 1e-7)
            scaled = torch.nan_to_num(scaled, nan=0.0, posinf=1e4, neginf=-1e4)

            # Top-k filtering
            if top_k > 0:
                k = min(top_k, scaled.size(-1))
                topk_vals, _ = torch.topk(scaled, k)
                scaled[scaled < topk_vals[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                # Shift right so the first token above threshold is kept
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # Scatter back to original ordering
                scaled = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, 1)  # (B, 1)

            # Replace tokens for finished sequences with pad
            next_token = torch.where(
                finished.unsqueeze(1), pad_token_id, next_token,
            )
            generated_tokens.append(next_token)

            # Check for EOS
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(1) == eos_token_id)
                if finished.all():
                    break

            # Decode step — single token, no attention mask needed (KV cache has full context)
            hidden_states, kv_caches, _ = self.model(next_token, kv_caches, cur_len)
            logits = self.lm_head(hidden_states).squeeze(1)
            cur_len += 1

        if generated_tokens:
            return torch.cat([input_ids, torch.cat(generated_tokens, dim=1)], dim=1)
        return input_ids
