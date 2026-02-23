"""Pretraining config for g6-big (L4 GPU, 22GB VRAM)."""

from pathlib import Path
from plasmid_llm.config import PretrainingConfig

config = PretrainingConfig(
    # Data — training_pairs_v4 via S3 mount
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    special_tokens=Path("data/special_tokens.txt"),

    # Model architecture
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,

    # Training
    output_dir=Path("/opt/dlami/nvme/output/pretrain_v4"),
    per_device_train_batch_size=16,
    learning_rate=3e-4,
    max_steps=30_000,
    warmup_steps=1000,

    # System — L4 supports bf16
    bf16=True,
    gradient_checkpointing=False,
    dataloader_num_workers=4,

    # Eval & saving
    eval_steps=500,
    save_steps=500,
    logging_steps=10,
    early_stopping_patience=10,

    # MLflow — Databricks hosted tracking
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/plasmid_pretrain_v4",
)
