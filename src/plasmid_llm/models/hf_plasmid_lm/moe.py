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

    Load balancing auxiliary loss: L_aux = N * sum(f_i * P_i) where
    f_i = fraction of tokens routed to expert i, P_i = mean routing
    probability for expert i.
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

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            output: (batch, seq_len, hidden_size)
            aux_loss: scalar load balancing loss
        """
        batch, seq_len, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)  # (B*S, H)
        num_tokens = flat.shape[0]

        # Router: compute probabilities over experts
        router_logits = self.router(flat)  # (B*S, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        top_weights, top_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (B*S, top_k)
        # Normalize selected weights to sum to 1
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Dispatch: loop over experts, gather tokens, compute, scatter back
        output = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            # Mask for tokens where expert i is in the top-k
            mask = (top_indices == i).any(dim=-1)  # (B*S,)
            if not mask.any():
                continue
            expert_input = flat[mask]  # (n_tokens, H)
            expert_output = expert(expert_input)  # (n_tokens, H)
            # Weight for this expert for selected tokens
            # Find which top-k slot(s) matched and get corresponding weight
            match_positions = (top_indices[mask] == i)  # (n_tokens, top_k)
            weights = (top_weights[mask] * match_positions.float()).sum(dim=-1, keepdim=True)  # (n_tokens, 1)
            output[mask] += weights * expert_output

        output = output.view(batch, seq_len, hidden)

        # Load balancing auxiliary loss
        # f_i: fraction of tokens dispatched to expert i
        # P_i: mean routing probability assigned to expert i
        with torch.no_grad():
            # Count tokens per expert (based on top-k assignments)
            expert_counts = torch.zeros(self.num_experts, device=flat.device)
            for k in range(self.top_k):
                expert_counts.scatter_add_(0, top_indices[:, k], torch.ones(num_tokens, device=flat.device))
            f = expert_counts / (num_tokens * self.top_k)  # fraction

        P = router_probs.mean(dim=0)  # (num_experts,)
        aux_loss = self.num_experts * (f * P).sum()

        return output, aux_loss
