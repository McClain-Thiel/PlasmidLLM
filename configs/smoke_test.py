"""Smoke test config — 10 steps on sample data."""

from pathlib import Path
from plasmid_llm.config import PretrainingConfig

config = PretrainingConfig(
    training_pairs=Path("data/training_pairs_sample.parquet"),
    special_tokens=Path("data/special_tokens.txt"),

    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,

    output_dir=Path("/tmp/plasmid_smoke_test"),
    per_device_train_batch_size=4,
    learning_rate=3e-4,
    max_steps=10,
    warmup_steps=2,
    bf16=True,
    gradient_checkpointing=False,
    dataloader_num_workers=2,

    eval_steps=5,
    save_steps=5,
    logging_steps=1,
    early_stopping_patience=100,

    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/plasmid_pretrain_v4",
)
