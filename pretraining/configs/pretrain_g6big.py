"""Pretraining config for g6-big (1x L4, 22GB VRAM).

Transformer (d=384, 10L, 8H) with batch_size=32.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plasmid_llm.config import PretrainingConfig

config = PretrainingConfig(
    # Data
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    special_tokens=Path("/opt/dlami/nvme/PlasmidLLM/data/special_tokens.txt"),

    # Architecture
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,

    # Training
    output_dir=Path("/opt/dlami/nvme/output/pretrain_v4"),
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

    # MLflow — Databricks hosted tracking
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/plasmid_pretrain_v4",
)
