from dataclasses import dataclass
import torch


@dataclass
class GenerationResult:
    """Raw generation output."""
    prompts: list[str]
    completion_texts: list[str]
    completion_ids: torch.Tensor   # (B, T_comp) CPU
    full_ids: torch.Tensor         # (B, T_full) CPU
    prompt_len: int


@dataclass
class LogProbResult:
    """Per-token log-probs + mask."""
    per_token: torch.Tensor        # (B, T_comp) CPU, masked (0 at pads)
    mask: torch.Tensor             # (B, T_comp) CPU, bool
    mean_per_seq: torch.Tensor     # (B,) CPU


@dataclass
class EntropyResult:
    """Per-token entropy + mask."""
    per_token: torch.Tensor        # (B, T_comp) CPU
    mask: torch.Tensor             # (B, T_comp) CPU, bool
    mean_per_seq: torch.Tensor     # (B,) CPU


@dataclass
class LogitsResult:
    """Raw logits for completion tokens."""
    logits: torch.Tensor           # (B, T_comp, V) CPU
    mask: torch.Tensor             # (B, T_comp) CPU, bool


@dataclass
class BackwardResult:
    """What happened during clip + step."""
    grad_norm: float
    lr: float
    step: int
