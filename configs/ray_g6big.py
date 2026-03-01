"""Ray-based post-training config for g6-big (1x L4, ~8 vCPUs).

GRPO: 8 prompts × 4 generations = 32 sequences/step.
Sum-of-components reward (each capped at 1.0).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from post_training.ray.config import RayPostTrainingConfig

config = RayPostTrainingConfig(
    # Data — pretrained checkpoint + training pairs + motif registry
    model_checkpoint=Path("/opt/dlami/nvme/output/pretrain_v4/checkpoint-15000"),
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    motif_lookup=Path("/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet"),

    # Algorithm — GRPO with 4 rollouts per prompt
    algorithm="grpo",
    num_generations_per_prompt=4,

    # Ray resources — 1 GPU for policy, remaining CPUs for scoring
    num_cpu_workers=6,
    gpu_memory_fraction=0.9,

    # Generation — 8 unique prompts × 4 = 32 sequences/step
    generation_batch_size=8,
    max_completion_length=1024,
    temperature=1.0,
    top_k=0,
    top_p=0.95,

    # Training
    learning_rate=1e-5,
    max_steps=5000,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_steps=200,
    bf16=True,
    seed=42,

    # GRPO-specific
    kl_coef=0.05,
    cliprange=0.2,

    # Reward — sum of components, no curriculum
    reward_fn_name="motif_alignment",
    scoring_batch_size=4,
    eos_bonus=0.15,

    # Output & logging
    output_dir=Path("/opt/dlami/nvme/output/grpo_v1"),
    save_steps=500,
    logging_steps=1,

    # MLflow — Databricks hosted tracking
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
