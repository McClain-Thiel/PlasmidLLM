"""Absorbing-state discrete diffusion model (MDLM / D3PM style)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from plasmid_llm.models import register_model


def _cosine_schedule(timesteps: int) -> torch.Tensor:
    """Cosine noise schedule — fraction of tokens masked at each timestep."""
    t = torch.linspace(0, 1, timesteps + 1)
    alphas = torch.cos(t * math.pi / 2) ** 2
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    return betas.clamp(0.0, 0.999)


def _linear_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(0.0001, 0.02, timesteps)


class DenoisingTransformerBlock(nn.Module):
    """Transformer block for the denoising backbone (bidirectional attention)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.RMSNorm(d_model)
        d_ff = d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class DenoisingTransformer(nn.Module):
    """Bidirectional transformer backbone for discrete diffusion."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DenoisingTransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(positions)

        # Add time conditioning
        t_emb = self.time_emb(t.float().unsqueeze(-1) / 1000.0)  # (B, d_model)
        h = h + t_emb.unsqueeze(1)

        h = self.drop(h)
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)

        h = self.ln_f(h)
        return self.head(h)


@register_model("diffusion")
class DiscreteDiffusionLM(nn.Module):
    """Absorbing-state discrete diffusion language model.

    Forward process: randomly replace tokens with PAD (absorbing state).
    Reverse process: predict original tokens from noised input.
    """

    def __init__(self, config: Any, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.timesteps = config.timesteps
        self.absorb_id = 0  # PAD token as absorbing state

        self.backbone = DenoisingTransformer(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Noise schedule
        if config.noise_schedule == "cosine":
            betas = _cosine_schedule(config.timesteps)
        else:
            betas = _linear_schedule(config.timesteps)

        # Cumulative mask probability: probability a token is masked at timestep t
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        # mask_prob[t] = 1 - alpha_cumprod[t] = probability of being masked
        self.register_buffer("mask_prob", 1 - alpha_cumprod)

    def _noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply forward diffusion: mask tokens to absorbing state."""
        mask_p = self.mask_prob[t]  # (B,)
        mask = torch.rand_like(x.float()) < mask_p.unsqueeze(1)
        noised = x.clone()
        noised[mask] = self.absorb_id
        return noised, mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, L = input_ids.shape

        # Use labels as clean targets if provided, else use input_ids
        clean = labels if labels is not None else input_ids
        # For computing loss, ignore -100 positions
        valid_mask = clean != -100
        clean_for_noise = clean.clamp(min=0)  # replace -100 with 0 for noise

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=input_ids.device)

        # Noise the sequence
        noised, noise_mask = self._noise(clean_for_noise, t)

        # Predict clean tokens
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        logits = self.backbone(noised, t, key_padding_mask=key_padding_mask)

        # Loss: only on masked positions that are valid (not -100 in labels)
        loss_mask = noise_mask & valid_mask
        if loss_mask.any():
            loss = F.cross_entropy(
                logits[loss_mask],
                clean_for_noise[loss_mask],
            )
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Iterative denoising: start from all-masked, progressively unmask."""
        B = input_ids.shape[0]
        # Start with prompt + masked completion
        x = torch.cat(
            [input_ids, torch.full((B, max_new_tokens), self.absorb_id, device=input_ids.device)],
            dim=1,
        )
        prompt_len = input_ids.shape[1]

        # Reverse diffusion from t=T-1 to t=0
        for t_val in reversed(range(self.timesteps)):
            t = torch.full((B,), t_val, device=x.device, dtype=torch.long)
            logits = self.backbone(x, t)

            # Only update masked positions in the completion region
            completion_logits = logits[:, prompt_len:, :]
            completion_logits = completion_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(completion_logits, top_k, dim=-1)
                completion_logits[completion_logits < v[..., -1:]] = float("-inf")
            probs = F.softmax(completion_logits, dim=-1)
            predicted = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, max_new_tokens)

            # Only unmask tokens that are still masked
            is_masked = x[:, prompt_len:] == self.absorb_id
            # Unmask with probability proportional to schedule
            if t_val > 0:
                unmask_prob = (self.mask_prob[t_val] - self.mask_prob[t_val - 1]) / self.mask_prob[t_val]
                unmask = torch.rand_like(is_masked.float()) < unmask_prob.unsqueeze(1)
                do_unmask = is_masked & unmask
            else:
                do_unmask = is_masked

            x[:, prompt_len:] = torch.where(do_unmask, predicted, x[:, prompt_len:])

        return x
