"""Configuration classes for PlasmidLM training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from plasmid_llm.utils import _compute_file_hash


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

    # Tokenizer
    tokenizer_type: str = "char"  # "char" or "kmer"
    kmer_k: int = 6
    kmer_stride: int = 3

    # MoE
    use_moe: bool = False
    num_experts: int = 6
    num_experts_per_tok: int = 2
    moe_intermediate_size: Optional[int] = None  # defaults to intermediate_size
    aux_loss_coef: float = 0.01

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

    # W&B tracking
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

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
            # Tokenizer
            "tokenizer_type": self.tokenizer_type,
            "kmer_k": self.kmer_k,
            "kmer_stride": self.kmer_stride,
            # MoE
            "use_moe": self.use_moe,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size or self.intermediate_size,
            "aux_loss_coef": self.aux_loss_coef,

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
