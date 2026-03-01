"""Ray-based post-training config for g6-big (1x L4, ~8 vCPUs).

REINFORCE with KL penalty, short completions for diversity,
curriculum alpha ramp from presence -> exact scoring.

v4: Sum-of-components reward (not mean). Rewards placing more motifs.
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

    # Algorithm
    algorithm="reinforce",

    # Ray resources — 1 GPU for policy, remaining CPUs for scoring
    num_cpu_workers=6,
    gpu_memory_fraction=0.9,

    # Generation — short completions for more diversity
    generation_batch_size=8,
    max_completion_length=1024,
    temperature=1.0,
    top_k=0,
    top_p=0.95,

    # Training — conservative to prevent KL explosion
    learning_rate=1e-5,
    max_steps=5000,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_steps=500,
    bf16=True,
    seed=42,

    # REINFORCE-specific — strong KL constraint
    kl_coef=1.0,

    # Curriculum — ramp from presence to exact scoring
    curriculum_alpha_start=0.0,
    curriculum_alpha_end=1.0,
    curriculum_alpha_warmup_steps=2000,

    # Reward
    reward_fn_name="motif_alignment",
    scoring_batch_size=4,  # sequences per CPU task
    eos_bonus=0.15,
    length_penalty_threshold=3500,

    # Output & logging
    output_dir=Path("/opt/dlami/nvme/output/ray_reinforce_v4"),
    save_steps=500,
    logging_steps=1,

    # MLflow — Databricks hosted tracking
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
