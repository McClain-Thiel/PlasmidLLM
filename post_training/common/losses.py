# ══════════════════════════════════════════════════════════════════════════
# Loss function registry — pure functions, no state
# ══════════════════════════════════════════════════════════════════════════
#
# Algorithms register loss functions here.  The actor calls them by name
# in forward_backward(), keeping the actor fully algorithm-agnostic while
# avoiding shipping arbitrary callables through Ray serialization.
#
# Every loss function has the same base signature + **kwargs for
# algorithm-specific hyperparams (cliprange, kl_coef, etc.).

from typing import Callable
import torch

LossFn = Callable[..., torch.Tensor]
LOSS_REGISTRY: dict[str, LossFn] = {}


def register_loss(name: str):
    """Decorator to register a loss function by name."""
    def wrapper(fn: LossFn) -> LossFn:
        LOSS_REGISTRY[name] = fn
        return fn
    return wrapper


@register_loss("reinforce")
def reinforce_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    mask_f = mask.float()
    counts = mask_f.sum(-1).clamp(min=1)
    seq_lp = (log_probs * mask_f).sum(-1) / counts
    return -(advantages.detach() * seq_lp).mean()


@register_loss("ppo")
def ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    cliprange: float = 0.2,
    entropy_coeff: float = 0.01,
    **kwargs,
) -> torch.Tensor:
    mask_f = mask.float()
    counts = mask_f.sum(-1).clamp(min=1)

    ratio = (log_probs - old_log_probs).exp()
    adv = advantages.detach()
    if adv.dim() == 1:
        adv = adv.unsqueeze(-1)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * adv
    policy_loss = -(torch.min(surr1, surr2) * mask_f).sum(-1) / counts

    entropy = -(log_probs * mask_f).sum(-1) / counts
    return (policy_loss - entropy_coeff * entropy).mean()


@register_loss("grpo")
def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    cliprange: float = 0.2,
    kl_coef: float = 0.1,
    **kwargs,
) -> torch.Tensor:
    mask_f = mask.float()
    counts = mask_f.sum(-1).clamp(min=1)

    ratio = (log_probs - old_log_probs).exp()
    adv = advantages.detach()
    if adv.dim() == 1:
        adv = adv.unsqueeze(-1)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * adv
    policy_loss = -(torch.min(surr1, surr2) * mask_f).sum(-1) / counts

    # KL(π ‖ π_ref) via Schulman unbiased estimator
    log_ratio = ref_log_probs - log_probs
    kl = (torch.exp(log_ratio) - log_ratio - 1.0) * mask_f
    kl_per_seq = kl.sum(-1) / counts

    return (policy_loss + kl_coef * kl_per_seq).mean()
