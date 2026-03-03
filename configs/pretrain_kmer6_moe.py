"""Pretraining config: 6-mer tokenizer + MoE (combined).

~46.5M total params, ~23.5M active. 6-mer stride=3 => ~12K bp context.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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

    # Tokenizer
    tokenizer_type="kmer",
    kmer_k=6,
    kmer_stride=3,

    # MoE
    use_moe=True,
    num_experts=6,
    num_experts_per_tok=2,
    moe_intermediate_size=1536,
    aux_loss_coef=0.01,

    # Training
    output_dir=Path("output/pretrain_kmer6_moe"),
    per_device_train_batch_size=16,
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
    mlflow_experiment="plasmid_pretrain_kmer6_moe",
)
