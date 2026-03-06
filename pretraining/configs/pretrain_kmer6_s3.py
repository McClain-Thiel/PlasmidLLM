"""Pretraining config: 6-mer tokenizer with stride=3 (overlap=3).

~18.6M params. Each token = 6 bases, stride=3 => ~12K bp context at max_seq_len=4096.
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

    # Tokenizer
    tokenizer_type="kmer",
    kmer_k=6,
    kmer_stride=3,

    # Training
    output_dir=Path("output/pretrain_kmer6_s3"),
    per_device_train_batch_size=32,
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
    mlflow_experiment="plasmid_pretrain_kmer6_s3",
)
