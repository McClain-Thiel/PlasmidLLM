"""Configuration classes for PlasmidLM training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file (first 16 chars for logging)."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "unknown"


@dataclass
class PretrainingConfig:
    """Configuration for pretraining PlasmidLM.
    
    Required inputs:
    - training_pairs: Parquet with full_text column (<BOS><tokens><SEP>SEQUENCE<EOS>)
    - special_tokens: Text file with special tokens (one per line)
    """
    
    # Data paths (required)
    training_pairs: Path
    special_tokens: Path
    
    # Model architecture
    hidden_size: int = 384
    num_hidden_layers: int = 10
    num_attention_heads: int = 8
    intermediate_size: int = 1536
    max_seq_len: int = 4096
    
    # Training hyperparameters
    output_dir: Path = field(default_factory=lambda: Path("output/pretraining"))
    per_device_train_batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 100_000
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Evaluation
    val_split: float = 0.05
    eval_steps: int = 500
    save_steps: int = 5000
    logging_steps: int = 10
    early_stopping_patience: int = 10
    
    # System
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    seed: int = 42
    
    # MLflow tracking
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "plasmid_pretraining"
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate."""
        self.training_pairs = Path(self.training_pairs)
        self.special_tokens = Path(self.special_tokens)
        self.output_dir = Path(self.output_dir)
        
        if not self.training_pairs.exists():
            raise FileNotFoundError(f"training_pairs not found: {self.training_pairs}")
        if not self.special_tokens.exists():
            raise FileNotFoundError(f"special_tokens not found: {self.special_tokens}")
    
    def to_mlflow_params(self) -> dict:
        """Convert config to MLflow parameters for logging."""
        return {
            # Data
            "training_pairs": str(self.training_pairs),
            "special_tokens": str(self.special_tokens),
            "training_pairs_hash": _compute_file_hash(self.training_pairs),
            "special_tokens_hash": _compute_file_hash(self.special_tokens),
            "max_seq_len": self.max_seq_len,
            "val_split": self.val_split,
            
            # Architecture
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            
            # Training
            "batch_size": self.per_device_train_batch_size,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
        }


@dataclass
class PostTrainingConfig:
    """Configuration for post-training (RL) with PlasmidLM.
    
    Required inputs:
    - training_pairs: Same parquet used for pretraining (will filter has_hard_tokens=True)
    - motif_lookup: Parquet mapping tokens to sequences (from notebook)
    - model_checkpoint: Path to pretrained model
    """
    
    # Data paths (required)
    training_pairs: Path  # Same as pretraining - will filter has_hard_tokens=True
    motif_lookup: Path    # Motif lookup parquet (token -> sequences)
    model_checkpoint: Path
    
    # GRPO hyperparameters
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    max_steps: Optional[int] = None
    
    # Sampling parameters
    num_generations_per_prompt: int = 16  # How many completions to generate per prompt
    max_new_tokens: int = 8000
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    
    # GRPO-specific (from TRL)
    num_ppo_epochs: int = 4
    kl_penalty: str = "kl"  # "kl", "abs", or "mse"
    kl_coef: float = 0.05
    clip_range: float = 0.2
    vf_coef: float = 0.1
    
    # Training
    output_dir: Path = field(default_factory=lambda: Path("output/post_training"))
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 100
    seed: int = 42
    
    # System
    bf16: bool = False
    use_vllm: bool = True  # Use vLLM for fast sampling
    dataloader_num_workers: int = 4
    
    # MLflow tracking
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "plasmid_post_training"
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate."""
        self.training_pairs = Path(self.training_pairs)
        self.motif_lookup = Path(self.motif_lookup)
        self.model_checkpoint = Path(self.model_checkpoint)
        self.output_dir = Path(self.output_dir)
        
        if not self.training_pairs.exists():
            raise FileNotFoundError(f"training_pairs not found: {self.training_pairs}")
        if not self.motif_lookup.exists():
            raise FileNotFoundError(f"motif_lookup not found: {self.motif_lookup}")
        if not self.model_checkpoint.exists():
            raise FileNotFoundError(f"model_checkpoint not found: {self.model_checkpoint}")
    
    def to_mlflow_params(self) -> dict:
        """Convert config to MLflow parameters for logging."""
        return {
            # Data
            "training_pairs": str(self.training_pairs),
            "motif_lookup": str(self.motif_lookup),
            "model_checkpoint": str(self.model_checkpoint),
            "training_pairs_hash": _compute_file_hash(self.training_pairs),
            "motif_lookup_hash": _compute_file_hash(self.motif_lookup),
            
            # GRPO params
            "learning_rate": self.learning_rate,
            "batch_size": self.per_device_train_batch_size,
            "num_generations_per_prompt": self.num_generations_per_prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "num_ppo_epochs": self.num_ppo_epochs,
            "kl_coef": self.kl_coef,
            "kl_penalty": self.kl_penalty,
            
            # Sampling
            "top_k": self.top_k,
            "top_p": self.top_p,
            "use_vllm": self.use_vllm,
            "seed": self.seed,
        }
