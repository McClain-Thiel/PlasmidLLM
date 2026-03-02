"""Ray-based post-training config for g6-big (1x L4, ~8 vCPUs).

GRPO v2: restart from grpo_v1/checkpoint-1000 (peak learning).
Lower KL (0.01), more rollouts (G=8), cosine lr decay.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from post_training.ray.config import RayPostTrainingConfig

config = RayPostTrainingConfig(
    # Restart from grpo_v1 checkpoint-1000 (still actively learning)
    model_checkpoint=Path("/opt/dlami/nvme/output/grpo_v1/checkpoint-1000"),
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    motif_lookup=Path("/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet"),

    # Algorithm — GRPO with 8 rollouts per prompt for more diversity
    algorithm="grpo",
    num_generations_per_prompt=8,

    # Ray resources
    num_cpu_workers=6,
    gpu_memory_fraction=0.9,

    # Generation — 4 unique prompts × 8 = 32 sequences/step
    generation_batch_size=4,
    max_completion_length=1024,
    temperature=1.0,
    top_k=0,
    top_p=0.95,

    # Training — cosine decay after warmup
    learning_rate=1e-5,
    max_steps=5000,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_steps=100,
    bf16=True,
    seed=42,

    # GRPO — lower KL to let policy explore further
    kl_coef=0.01,
    cliprange=0.2,

    # Reward
    reward_fn_name="motif_alignment",
    scoring_batch_size=4,
    eos_bonus=0.15,

    # Output & logging
    output_dir=Path("/opt/dlami/nvme/output/grpo_v2"),
    save_steps=500,
    logging_steps=1,

    # MLflow
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
