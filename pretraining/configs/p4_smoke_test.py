"""Quick 100-step smoke test for p4 — verifies training pipeline end-to-end."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plasmid_llm.config import PretrainingConfig

DATA_DIR = Path("/opt/dlami/nvme/data")
OUTPUT_ROOT = Path("/opt/dlami/nvme/output")

config = PretrainingConfig(
    # Data
    training_pairs=DATA_DIR / "training_pairs_v4.parquet",
    special_tokens=Path(__file__).resolve().parent.parent / "data" / "special_tokens.txt",

    # Architecture — baseline
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,

    # Training — 100 steps only
    output_dir=OUTPUT_ROOT / "smoke_test",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    max_steps=100,
    warmup_steps=10,
    weight_decay=0.1,
    max_grad_norm=1.0,

    # Evaluation
    val_split=0.05,
    eval_steps=50,
    save_steps=50,
    logging_steps=10,
    early_stopping_patience=100,

    # System
    bf16=True,
    gradient_checkpointing=False,
    dataloader_num_workers=4,
    seed=42,

    # W&B
    wandb_project="PlasmidLLM",
    wandb_run_name="smoke_test_baseline_100steps",
)
