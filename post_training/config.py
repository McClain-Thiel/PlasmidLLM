"""Configuration dataclass for post-training runs.

Mirrors the PretrainingConfig pattern: a single dataclass that captures
the full (algorithm, model, scorer, training) configuration. Configs are
plain Python files exporting a ``config`` variable.

Usage::

    # In a config file (post_training/configs/grpo_motif.py):
    config = PostTrainingConfig(
        model="McClain/PlasmidLM-kmer6-MoE",
        algorithm="grpo",
        scorer="motif",
        ...
    )

    # In the runner:
    python -m post_training.runners.run post_training/configs/grpo_motif.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class PostTrainingConfig:
    """Full configuration for a post-training run.

    Groups: model, algorithm, scorer, training loop, logging.
    """

    # ── Model ─────────────────────────────────────────────────────────────
    model: str = "gpt2"
    bf16: bool = True
    seed: int = 42

    # ── Actor (ModelActor kwargs) ─────────────────────────────────────────
    num_actors: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # ── Algorithm ─────────────────────────────────────────────────────────
    algorithm: str = "grpo"

    # GRPO-specific
    kl_coef: float = 0.1
    cliprange: float = 0.2
    num_generations: int = 4

    # PPO-specific
    entropy_coeff: float = 0.01
    ppo_epochs: int = 4

    # Shared
    micro_batch_size: int = 64

    # ── Scorer ────────────────────────────────────────────────────────────
    scorer: str = "substring"
    scorer_kwargs: dict[str, Any] = field(default_factory=dict)

    # ── Generation ────────────────────────────────────────────────────────
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.95

    # ── Prompts ───────────────────────────────────────────────────────────
    prompts_parquet: Optional[str] = None
    prompts_list: Optional[list[str]] = None
    prompt_batch_size: int = 4
    filter_hard_tokens: bool = True

    # ── Training loop ─────────────────────────────────────────────────────
    steps: int = 100
    checkpoint_every: int = 50
    checkpoint_dir: str = "checkpoints/post_training"

    # ── Logging ───────────────────────────────────────────────────────────
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    @property
    def wandb_enabled(self) -> bool:
        return self.wandb_project is not None

    def to_wandb_config(self) -> dict[str, Any]:
        """Flat dict of all config values for wandb.config."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                for dk, dv in v.items():
                    d[f"scorer_{dk}"] = dv
            else:
                d[k] = v
        return d

    def algorithm_kwargs(self) -> dict[str, Any]:
        """Kwargs to pass to the algorithm constructor."""
        shared = dict(micro_batch_size=self.micro_batch_size)
        if self.algorithm == "grpo":
            return {
                **shared,
                "kl_coef": self.kl_coef,
                "cliprange": self.cliprange,
                "num_generations": self.num_generations,
            }
        elif self.algorithm == "ppo":
            return {
                **shared,
                "cliprange": self.cliprange,
                "entropy_coeff": self.entropy_coeff,
                "ppo_epochs": self.ppo_epochs,
            }
        return shared

    def actor_kwargs(self) -> dict[str, Any]:
        """Kwargs to pass to ModelActor.__init__ (minus model_id)."""
        return dict(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            max_steps=self.steps,
            max_grad_norm=self.max_grad_norm,
            bf16=self.bf16,
            seed=self.seed,
        )

    def generation_kwargs(self) -> dict[str, Any]:
        """Kwargs to pass to actor.generate()."""
        return dict(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
