"""6-mer tokenizer, stride=4, dense transformer on p4 (8x A100-40GB).

~18.6M params. Each token = 6 bases with 2-base overlap.
At max_seq_len=4096, covers ~16K bp of DNA context (4x char baseline).
Less overlap than s3 variant — tests impact of overlap amount on generation quality.
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

    # Tokenizer — 6-mer with stride 4
    tokenizer_type="kmer",
    kmer_k=6,
    kmer_stride=4,

    # Training
    output_dir=OUTPUT_ROOT / "kmer6_s4_dense",
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

    # System
    bf16=True,
    gradient_checkpointing=False,
    dataloader_num_workers=4,
    seed=42,

    # W&B
    wandb_project="PlasmidLLM",
    wandb_run_name="kmer6_s4_dense_d384_L10",
)
