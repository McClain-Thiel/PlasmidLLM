"""PPO model wrappers for PlasmidLM.

TRL's PPOTrainer (trl.experimental.ppo) requires nn.Module reward and value models
with a specific interface matching AutoModelForSequenceClassification:

  1. model.base_model_prefix — string name of the backbone attribute
  2. getattr(model, base_model_prefix) — backbone that accepts HF-style forward kwargs
     and returns an object with .hidden_states list
  3. model.score(hidden_states) — classification head returning (batch, seq_len, num_labels)

PlasmidRewardWrapper: Non-neural wrapper around plasmid_reward_fn. Caches input_ids in
  a fake backbone, then .score() decodes them and calls Smith-Waterman alignment.
  PPO doesn't backprop through the reward model, so this is safe.

PlasmidValueModel: Real neural model — PlasmidLM backbone + scalar value head.
  Predicts per-token expected reward for GAE advantage estimation.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import pandas as pd

from post_training.reward import build_category_index, plasmid_reward_fn


# ── HF-compatible backbone wrapper ──────────────────────────────────────────

class _HFBackboneWrapper(nn.Module):
    """Wraps PlasmidLMModel to accept HF-style kwargs and return HF-style output.

    PPOTrainer's get_reward() calls the backbone with:
        input_ids, attention_mask, position_ids, return_dict=True,
        output_hidden_states=True, use_cache=False
    and expects output.hidden_states[-1] to be the last-layer hidden states.

    PlasmidLMModel's forward only uses input_ids — extra kwargs are ignored.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                return_dict=True, output_hidden_states=False, use_cache=False, **kwargs):
        hidden_states, _ = self.backbone(input_ids)
        return SimpleNamespace(
            last_hidden_state=hidden_states,
            hidden_states=[hidden_states],
        )

    @property
    def embed_tokens(self):
        return self.backbone.embed_tokens


# ── Reward model wrapper ────────────────────────────────────────────────────

class _RewardBackbone(nn.Module):
    """Minimal backbone that caches input_ids for reward computation."""

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self._cached_input_ids = None

    def forward(self, input_ids, **kwargs):
        self._cached_input_ids = input_ids.detach()
        B, L = input_ids.shape
        dummy = torch.zeros(B, L, 1, device=input_ids.device, dtype=self.dummy_param.dtype)
        return SimpleNamespace(hidden_states=[dummy])


class PlasmidRewardWrapper(nn.Module):
    """Non-differentiable reward model wrapping plasmid_reward_fn for PPOTrainer.

    PPO doesn't backprop through the reward model — it only extracts scalar rewards.
    This wrapper decodes generated token IDs back to text, splits at <SEP>, and calls
    the Smith-Waterman alignment reward function.

    Interface:
        backbone.forward(input_ids, ...) → caches input_ids, returns dummy hidden_states
        score(hidden_states) → decodes cached input_ids, computes rewards, returns (B, L, 1)
    """

    base_model_prefix = "backbone"

    def __init__(self, tokenizer, lookup_df: pd.DataFrame, alpha: float = 0.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.lookup_df = lookup_df
        self.alpha = alpha
        self.category_index = build_category_index(lookup_df)
        self.backbone = _RewardBackbone()

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute rewards from cached input_ids.

        Returns (batch, seq_len, 1) — get_reward extracts scalar at last non-pad position.
        """
        input_ids = self.backbone._cached_input_ids
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        prompts, completions = [], []
        for text in texts:
            if "<SEP>" in text:
                p, c = text.split("<SEP>", 1)
                prompts.append(p + "<SEP>")
                completions.append(c)
            else:
                prompts.append(text)
                completions.append("")

        rewards = plasmid_reward_fn(
            prompts, completions, self.lookup_df,
            alpha=self.alpha, category_index=self.category_index,
        )
        scores = torch.tensor(rewards, device=hidden_states.device, dtype=torch.float32)

        # Expand scalar to every position: (batch, seq_len, 1)
        # get_reward() extracts the value at the last non-pad token position
        return scores.view(-1, 1, 1).expand(-1, hidden_states.shape[1], 1)


# ── Value model ─────────────────────────────────────────────────────────────

class PlasmidValueModel(nn.Module):
    """Value model: PlasmidLM backbone + scalar value head for PPO.

    Predicts per-token expected reward. PPO backprops through this model
    and uses its predictions as baseline for GAE advantage estimation.

    Args:
        backbone: PlasmidLMModel instance (will be wrapped for HF compatibility)
        hidden_size: Model hidden dimension (for value head)
    """

    base_model_prefix = "model"

    def __init__(self, backbone, hidden_size: int):
        super().__init__()
        self.model = _HFBackboneWrapper(backbone)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        # Initialize near zero — no prior on expected reward
        nn.init.normal_(self.score.weight, std=0.01)
