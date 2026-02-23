"""GRPO post-training config for g6-big (L4 GPU, 22GB VRAM).

Smoke test used 2.4GB with batch=2, num_gen=2 at 8k completion.
Scaling to batch=4, num_gen=16 (64 seqs/step, ~16x more throughput).
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plasmid_llm.config import PostTrainingConfig

config = PostTrainingConfig(
    # Data — pretrained checkpoint + training pairs + motif registry
    model_checkpoint=Path("/opt/dlami/nvme/output/pretrain_v4/checkpoint-15000"),
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    motif_lookup=Path("/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet"),

    # GRPO hyperparameters
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch = 16 prompts, 16 gens each = 256 seqs
    num_train_epochs=1,
    max_steps=5000,

    # Sampling — 8k completions, 16 generations per prompt
    num_generations=16,
    max_completion_length=8192,
    temperature=0.8,
    top_k=50,
    top_p=0.95,

    # GRPO-specific
    num_iterations=1,
    beta=0.04,
    epsilon=0.2,
    loss_type="grpo",

    # Output & logging
    output_dir=Path("/opt/dlami/nvme/output/grpo_v1"),
    save_steps=500,
    logging_steps=1,
    eval_steps=500,
    seed=42,

    # System — L4 supports bf16
    bf16=True,
    use_vllm=False,
    dataloader_num_workers=4,

    # MLflow — Databricks hosted tracking
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/PlasmidLLM",
)
