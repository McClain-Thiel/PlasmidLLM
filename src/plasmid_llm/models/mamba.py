"""Mamba SSM language model with fallback to pure-PyTorch S4-like implementation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from plasmid_llm.models import register_model

# Try to import mamba_ssm; fall back to pure PyTorch
try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class S4Block(nn.Module):
    """Simplified S4-like block as fallback when mamba_ssm is not installed.

    Uses a diagonal state-space model with discretization.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.A_log = nn.Parameter(torch.log(torch.randn(d_inner, d_state).abs() + 1e-4))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Conv
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        # SSM scan (simplified — parallel via cumsum approximation)
        proj = self.x_proj(x_ssm)
        B_ssm = proj[:, :, : self.d_state]
        C_ssm = proj[:, :, self.d_state : 2 * self.d_state]
        dt = F.softplus(self.dt_proj(proj[:, :, -1:]))  # (B, L, d_inner)

        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Discretize and scan (simplified linear recurrence via cumulative sum)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)  # (B, L, 1, d_state) * needs broadcast
        x_db = x_ssm.unsqueeze(-1) * dB  # (B, L, d_inner, d_state) — simplified

        # Simple recurrence (not parallelized — fine for moderate seq lengths)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + x_ssm[:, t, :, None] * dB[:, t]
            y_t = (h * C_ssm[:, t, None, :]).sum(-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        y = y + x_ssm * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Single Mamba block with residual and norm."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        if HAS_MAMBA:
            self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mixer = S4Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


@register_model("mamba")
class MambaLM(nn.Module):
    """Mamba-based language model."""

    def __init__(self, config: Any, vocab_size: int):
        super().__init__()
        d_model = config.d_model
        n_layers = config.n_layers
        d_state = config.d_state
        d_conv = config.d_conv
        expand = config.expand

        padded_vocab = ((vocab_size + 7) // 8) * 8
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(padded_vocab, d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)]
        )
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, padded_vocab, bias=False)
        self.head.weight = self.tok_emb.weight

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
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
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
