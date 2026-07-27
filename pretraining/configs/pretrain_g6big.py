"""Pretraining config for a single 22 GB-class GPU.

Transformer (d=384, 10L, 8H) with batch_size=32.

Set ``PLASMID_TRAINING_PAIRS`` and ``PLASMID_SPECIAL_TOKENS`` when the
inputs are not in the repository's gitignored ``data`` directory.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from plasmid_llm.config import PretrainingConfig  # noqa: E402

config = PretrainingConfig(
    # Data
    training_pairs=Path(
        os.environ.get(
            "PLASMID_TRAINING_PAIRS",
            REPO_ROOT / "data" / "training_pairs_v4.parquet",
        )
    ),
    special_tokens=Path(
        os.environ.get(
            "PLASMID_SPECIAL_TOKENS",
            REPO_ROOT / "data" / "special_tokens.txt",
        )
    ),
    # Architecture
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,
    # Training
    output_dir=Path(os.environ.get("PLASMID_PRETRAIN_OUTPUT", "output/pretrain_v4")),
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
    mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "databricks"),
    mlflow_experiment=os.environ.get(
        "MLFLOW_EXPERIMENT",
        "/plasmid_pretrain_v4",
    ),
)
