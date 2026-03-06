"""Pretraining config: MoE with full-size experts (6 x 1536 intermediate).

~45M total params, ~22M active params per token.
Char-level tokenizer, 4K bp context.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plasmid_llm.config import PretrainingConfig

config = PretrainingConfig(
    # Data
    training_pairs=Path("data/training_pairs_v4.parquet"),
    special_tokens=Path("data/special_tokens.txt"),

    # Architecture
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,

    # MoE
    use_moe=True,
    num_experts=6,
    num_experts_per_tok=2,
    moe_intermediate_size=1536,  # full-size experts
    aux_loss_coef=0.01,

    # Training
    output_dir=Path("output/pretrain_moe_full"),
    per_device_train_batch_size=16,  # reduced for MoE memory
    learning_rate=3e-4,
    max_steps=100_000,
    warmup_steps=1000,
    weight_decay=0.1,
    max_grad_norm=1.0,

    # Evaluation
    val_split=0.05,
    eval_steps=500,
    save_steps=5000,
    logging_steps=10,
    early_stopping_patience=10,

    # System
    bf16=True,
    gradient_checkpointing=False,
    dataloader_num_workers=4,
    seed=42,

    # Tracking
    mlflow_experiment="plasmid_pretrain_moe_full",
)
