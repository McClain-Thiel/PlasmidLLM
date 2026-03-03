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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, new_kv = self.self_attn(hidden_states, rope_cos, rope_sin, past_key_value, position_offset)
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

        head_dim = config.hidden_size // config.num_attention_heads
        cos, sin = _rope_freqs(head_dim, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        position_offset: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, list, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
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
                    hidden_states, self.rope_cos, self.rope_sin, past_kv, position_offset
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

        hidden_states, new_kv_list, aux_loss = self.model(input_ids, kv_list, position_offset)
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
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple autoregressive generation with KV cache."""
        # Prefill (aux_loss ignored during generation)
        hidden_states, kv_caches, _ = self.model(input_ids)
        logits = self.lm_head(hidden_states[:, -1:, :]).squeeze(1)
        cur_len = input_ids.shape[1]

        for _ in range(max_new_tokens):
            scaled = logits.float() / temperature
            scaled = torch.nan_to_num(scaled, nan=0.0, posinf=1e4, neginf=-1e4)
            if top_k > 0:
                k = min(top_k, scaled.size(-1))
                v, _ = torch.topk(scaled, k)
                scaled[scaled < v[:, [-1]]] = float("-inf")
            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            hidden_states, kv_caches, _ = self.model(next_token, kv_caches, cur_len)
            logits = self.lm_head(hidden_states).squeeze(1)
            cur_len += 1

        return input_ids
