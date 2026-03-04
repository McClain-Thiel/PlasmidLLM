"""Myriad HPC pretraining config (1x A100, 32 CPU cores via SGE).

Paths: training_pairs staged to $PLASMID_DATA_DIR by job script,
special_tokens from repo data/ dir.
Output written under $HOME (persists after job ends).
MLflow tracks to Databricks — credentials loaded from ~/.databricks_env.
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plasmid_llm.config import PretrainingConfig

data_dir = Path(os.environ["PLASMID_DATA_DIR"])
home = Path(os.environ["HOME"])
repo_root = Path(__file__).resolve().parent.parent

config = PretrainingConfig(
    # Data — training pairs staged to $TMPDIR, special tokens from repo
    training_pairs=data_dir / "training_pairs_v4.parquet",
    special_tokens=repo_root / "data" / "special_tokens.txt",

    # Architecture — same as pretrain_v4
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,

    # Training — larger batch for A100 80GB
    output_dir=home / "PlasmidLLM" / "output" / "pretrain_myriad_v1",
    per_device_train_batch_size=64,
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

    # System — more workers for 32 CPUs
    bf16=True,
    gradient_checkpointing=False,
    dataloader_num_workers=16,
    seed=42,

    # MLflow — Databricks hosted tracking (credentials from ~/.databricks_env)
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/plasmid_pretrain_v4",
)
