"""6-mer tokenizer, stride=3, dense transformer on p4 (8x A100-40GB).

~18.6M params. Each token = 6 bases with 3-base overlap.
At max_seq_len=4096, covers ~12K bp of DNA context (3x char baseline).
Tests whether multi-base commitment per step reduces repetitive generation.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plasmid_llm.config import PretrainingConfig

DATA_DIR = Path("/opt/dlami/nvme/data")
OUTPUT_ROOT = Path("/opt/dlami/nvme/output")

config = PretrainingConfig(
    # Data
    training_pairs=DATA_DIR / "training_pairs_v4.parquet",
    special_tokens=Path(__file__).resolve().parent.parent / "data" / "special_tokens.txt",

    # Architecture
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,

    # Tokenizer — 6-mer with stride 3
    tokenizer_type="kmer",
    kmer_k=6,
    kmer_stride=3,

    # Training
    output_dir=OUTPUT_ROOT / "kmer6_s3_dense",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
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

    # W&B
    wandb_project="PlasmidLLM",
    wandb_run_name="kmer6_s3_dense_d384_L10",
)
