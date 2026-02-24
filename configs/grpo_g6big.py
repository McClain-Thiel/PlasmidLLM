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
    learning_rate=1e-4,              # aggressive — 17M param model needs strong signal
    per_device_train_batch_size=8,   # must be divisible by num_generations
    gradient_accumulation_steps=1,   # update every step — RL benefits from fast updates
    num_train_epochs=1,
    max_steps=5000,

    # Sampling — short completions (1024) for more diversity between generations
    # At 4096 tokens, 8 generations are nearly identical (reward_std=0.004).
    # At 1024, per-token randomness compounds to structural differences → better GRPO signal.
    num_generations=8,
    max_completion_length=1024,
    temperature=1.0,                 # moderate temp — shorter seqs already give diversity
    top_k=0,                         # disabled — let temperature drive diversity
    top_p=0.95,

    # GRPO-specific
    num_iterations=1,
    beta=0.1,                        # moderate KL penalty — allow exploration
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
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
