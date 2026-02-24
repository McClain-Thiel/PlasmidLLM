"""RLOO (REINFORCE Leave-One-Out) post-training config for g6-big.

RLOO computes advantages as reward_i - mean(other_rewards_in_group),
without dividing by std. This gives honest signal even when generations
are near-identical (the scenario that killed GRPO for this DNA model).

Kept max_completion at 4096 (matches pretrain context) and high temp
for maximum diversity between generations.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plasmid_llm.config import PostTrainingConfig

config = PostTrainingConfig(
    # Data
    model_checkpoint=Path("/opt/dlami/nvme/output/pretrain_v4/checkpoint-15000"),
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    motif_lookup=Path("/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet"),

    # Training hyperparameters
    learning_rate=1e-3,              # high — RLOO advantages are unnormalized (~0.01), need strong lr
    per_device_train_batch_size=8,   # must be divisible by num_generations
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=5000,

    # Sampling — high temp + no top_k for diversity
    num_generations=8,
    max_completion_length=4096,
    temperature=1.2,
    top_k=0,
    top_p=0.95,

    # RLOO shares these params with GRPO
    num_iterations=1,
    beta=0.05,                       # light KL penalty — RLOO advantages are already small
    epsilon=0.2,
    loss_type="grpo",                # ignored by RLOOTrainer, but needed by PostTrainingConfig

    # Output & logging
    output_dir=Path("/opt/dlami/nvme/output/rloo_v1"),
    save_steps=500,
    logging_steps=1,
    eval_steps=500,
    seed=42,

    # System
    bf16=True,
    use_vllm=False,
    dataloader_num_workers=4,

    # MLflow
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
