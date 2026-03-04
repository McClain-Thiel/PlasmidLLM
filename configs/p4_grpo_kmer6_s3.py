"""GRPO post-training for kmer6 stride-3 dense model on p4 (1x A100-40GB, GPU 6).

Dense (non-MoE) model avoids stochastic routing issues that caused gradient
spikes in MoE GRPO training. Single GPU, 8 prompts x 8 rollouts = 64 seq/step.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from post_training.ray.config import RayPostTrainingConfig

DATA_DIR = Path("/opt/dlami/nvme/data")
OUTPUT_ROOT = Path("/opt/dlami/nvme/output")

config = RayPostTrainingConfig(
    # Model — kmer6 stride-3 dense pretrained checkpoint
    model_checkpoint=OUTPUT_ROOT / "kmer6_s3_dense" / "checkpoint-35000",
    training_pairs=DATA_DIR / "training_pairs_v4.parquet",
    motif_lookup=DATA_DIR / "motif_registry.parquet",

    # Algorithm — GRPO with 8 rollouts per prompt
    algorithm="grpo",
    num_generations_per_prompt=8,

    # Ray resources — 1 GPU for policy, ~15 CPUs for scoring
    num_cpu_workers=15,
    gpu_memory_fraction=0.9,

    # Generation — 8 unique prompts, 64 sequences/step
    generation_batch_size=8,
    max_completion_length=1024,
    temperature=1.0,
    top_k=0,
    top_p=0.95,

    # Training
    learning_rate=5e-6,
    max_steps=5000,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_steps=50,
    bf16=True,
    seed=42,

    # GRPO-specific
    kl_coef=0.3,
    cliprange=0.2,

    # Reward — sum of components
    reward_fn_name="motif_alignment",
    scoring_batch_size=8,
    eos_bonus=0.15,

    # Single GPU
    num_policy_gpus=1,

    # Output & logging
    output_dir=OUTPUT_ROOT / "grpo_kmer6_s3",
    save_steps=500,
    logging_steps=1,

    # W&B
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_kmer6_s3_lr5e6_kl03",

    # MLflow
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
