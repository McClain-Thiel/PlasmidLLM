"""Configuration for Ray-based distributed post-training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from plasmid_llm.utils import _compute_file_hash


@dataclass
class RayPostTrainingConfig:
    """Configuration for Ray-based distributed post-training.

    Required inputs:
    - model_checkpoint: Path to pretrained PlasmidLM checkpoint
    - training_pairs: Parquet with prompt/token_prompt columns
    - motif_lookup: Parquet mapping tokens to sequences for reward scoring
    """

    # Data paths (required)
    model_checkpoint: Path
    training_pairs: Path
    motif_lookup: Path

    # Algorithm
    algorithm: str = "reinforce"  # "reinforce", "ppo", "grpo"

    # Ray resources
    num_cpu_workers: int = 4
    gpu_memory_fraction: float = 0.9

    # Generation
    generation_batch_size: int = 8
    max_completion_length: int = 1024
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.95

    # Training
    learning_rate: float = 1e-4
    max_steps: int = 5000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 100
    bf16: bool = True
    seed: int = 42

    # Algorithm-specific
    kl_coef: float = 0.1  # KL penalty coefficient
    cliprange: float = 0.2  # PPO clip range
    num_generations_per_prompt: int = 1  # GRPO: multiple completions per prompt
    beta: float = 0.05  # GRPO KL penalty

    # Curriculum
    curriculum_alpha_start: float = 0.0
    curriculum_alpha_end: float = 1.0
    curriculum_alpha_warmup_steps: int = 1000

    # Reward
    reward_fn_name: str = "motif_alignment"
    scoring_batch_size: int = 16  # sequences per CPU scoring chunk
    eos_bonus: float = 0.15
    length_penalty_threshold: int = 3500

    # Checkpointing & logging
    output_dir: Path = field(default_factory=lambda: Path("output/ray_post_training"))
    save_steps: int = 500
    logging_steps: int = 1

    # MLflow
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "plasmid_ray_post_training"

    # W&B
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Multi-GPU
    num_policy_gpus: int = 1  # Number of GPU actors for parallel generation

    def __post_init__(self):
        self.model_checkpoint = Path(self.model_checkpoint)
        self.training_pairs = Path(self.training_pairs)
        self.motif_lookup = Path(self.motif_lookup)
        self.output_dir = Path(self.output_dir)

        if not self.model_checkpoint.exists():
            raise FileNotFoundError(f"model_checkpoint not found: {self.model_checkpoint}")
        if not self.training_pairs.exists():
            raise FileNotFoundError(f"training_pairs not found: {self.training_pairs}")
        if not self.motif_lookup.exists():
            raise FileNotFoundError(f"motif_lookup not found: {self.motif_lookup}")

    def to_mlflow_params(self) -> dict:
        return {
            "model_checkpoint": str(self.model_checkpoint),
            "training_pairs": str(self.training_pairs),
            "motif_lookup": str(self.motif_lookup),
            "training_pairs_hash": _compute_file_hash(self.training_pairs),
            "motif_lookup_hash": _compute_file_hash(self.motif_lookup),
            "algorithm": self.algorithm,
            "learning_rate": self.learning_rate,
            "generation_batch_size": self.generation_batch_size,
            "max_completion_length": self.max_completion_length,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "kl_coef": self.kl_coef,
            "cliprange": self.cliprange,
            "curriculum_alpha_start": self.curriculum_alpha_start,
            "curriculum_alpha_end": self.curriculum_alpha_end,
            "curriculum_alpha_warmup_steps": self.curriculum_alpha_warmup_steps,
            "reward_fn_name": self.reward_fn_name,
            "eos_bonus": self.eos_bonus,
            "length_penalty_threshold": self.length_penalty_threshold,
            "bf16": self.bf16,
            "seed": self.seed,
        }
