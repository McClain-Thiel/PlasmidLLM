"""GRPO smoke test — 5 steps with short generations for local validation."""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plasmid_llm.config import PostTrainingConfig

config = PostTrainingConfig(
    # Data — update paths for your environment
    model_checkpoint=Path("/opt/dlami/nvme/output/pretrain_v4/checkpoint-15000"),
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    motif_lookup=Path("/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet"),

    # Minimal training
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=5,

    # Full-length generations to test memory fit
    num_generations=2,
    max_completion_length=8192,
    temperature=0.8,
    top_k=50,
    top_p=0.95,

    # GRPO-specific
    num_iterations=1,
    beta=0.04,
    epsilon=0.2,
    loss_type="grpo",

    # Output
    output_dir=Path("/tmp/plasmid_grpo_smoke"),
    save_steps=5,
    logging_steps=1,
    eval_steps=5,
    seed=42,

    # System
    bf16=True,
    use_vllm=False,
    dataloader_num_workers=0,

    # MLflow — Databricks
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/PlasmidLLM",
)
