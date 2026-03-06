"""Mixture of Experts (Mixtral-style) for PlasmidLM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_plasmid_lm import PlasmidLMConfig


class PlasmidLMExpertMLP(nn.Module):
    """Single expert MLP — same architecture as PlasmidLMMLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class PlasmidLMSparseMoE(nn.Module):
    """Sparse Mixture of Experts with top-k routing and load balancing loss.

    Implements Mixtral-style routing: softmax over all experts, then select
    top-k. The output is a weighted sum of the selected experts' outputs.

    Two forward modes:
    - **Sparse** (default, training): Python loop dispatching tokens to selected
      experts only. Supports dynamic shapes but can't be compiled.
    - **Dense** (inference): All experts run on all tokens via batched matmul,
      then masked by routing weights. Fully static shapes — torch.compile
      friendly. ~3x more FLOPs but the experts are tiny so it's dominated by
      kernel launch overhead savings.

    Call ``prepare_dense()`` to switch to dense mode.
    """

    def __init__(self, config: PlasmidLMConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        intermediate = config.moe_intermediate_size

        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [PlasmidLMExpertMLP(config.hidden_size, intermediate) for _ in range(self.num_experts)]
        )
        self._dense_ready = False

    def prepare_dense(self) -> "PlasmidLMSparseMoE":
        """Stack and pre-transpose expert weights for batched matmul inference.

        After calling this, forward() uses the dense path automatically.
        The stacked buffers are non-persistent (not saved to state_dict).
        """
        # up_proj.weight: (I, H) → .T → (H, I);  stack → (E, H, I)
        up_w = torch.stack(
            [e.up_proj.weight.T for e in self.experts]
        ).contiguous()
        # down_proj.weight: (H, I) → .T → (I, H);  stack → (E, I, H)
        down_w = torch.stack(
            [e.down_proj.weight.T for e in self.experts]
        ).contiguous()
        self.register_buffer("_up_w", up_w, persistent=False)
        self.register_buffer("_down_w", down_w, persistent=False)
        self._dense_ready = True
        return self

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dense_ready:
            return self._forward_dense(hidden_states)
        return self._forward_sparse(hidden_states)

    def _forward_dense(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense dispatch: all experts on all tokens via batched matmul."""
        batch, seq_len, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)  # (N, H)
        num_tokens = flat.shape[0]

        router_logits = self.router(flat)  # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Build full (N, E) mask with only top-k entries nonzero
        expert_mask = torch.zeros(
            num_tokens, self.num_experts,
            device=flat.device, dtype=flat.dtype,
        )
        expert_mask.scatter_(1, top_indices, top_weights)

        # All experts, all tokens: single batched matmul pair
        # (E, N, H) @ (E, H, I) → (E, N, I)
        x = flat.unsqueeze(0).expand(self.num_experts, -1, -1)
        h = torch.bmm(x, self._up_w)
        h = F.gelu(h)
        # (E, N, I) @ (E, I, H) → (E, N, H)
        expert_out = torch.bmm(h, self._down_w)

        # Weighted sum over experts: (E, N, H), (N, E) → (N, H)
        output = torch.einsum("enh,ne->nh", expert_out, expert_mask)
        output = output.view(batch, seq_len, hidden)

        # Vectorized aux loss (no Python loop)
        with torch.no_grad():
            one_hot = torch.zeros(
                num_tokens, self.num_experts, device=flat.device,
            )
            one_hot.scatter_(1, top_indices, 1.0)
            f = one_hot.sum(dim=0) / (num_tokens * self.top_k)
        P = router_probs.mean(dim=0)
        aux_loss = self.num_experts * (f * P).sum()

        return output, aux_loss

    def _forward_sparse(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse dispatch: loop over experts, only compute on routed tokens."""
        batch, seq_len, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)
        num_tokens = flat.shape[0]

        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (top_indices == i).any(dim=-1)
            if not mask.any():
                continue
            expert_input = flat[mask]
            expert_output = expert(expert_input)
            match_positions = (top_indices[mask] == i)
            weights = (top_weights[mask] * match_positions.float()).sum(dim=-1, keepdim=True)
            output[mask] += weights * expert_output

        output = output.view(batch, seq_len, hidden)

        with torch.no_grad():
            expert_counts = torch.zeros(self.num_experts, device=flat.device)
            for k in range(self.top_k):
                expert_counts.scatter_add_(
                    0, top_indices[:, k],
                    torch.ones(num_tokens, device=flat.device),
                )
            f = expert_counts / (num_tokens * self.top_k)
        P = router_probs.mean(dim=0)
        aux_loss = self.num_experts * (f * P).sum()

        return output, aux_loss
