"""GRPO post-training for char-level dense baseline on p4 (1x A100-40GB, GPU 7).

Dense (non-MoE) char-level model. Single GPU, 8 prompts x 8 rollouts = 64 seq/step.
Comparison run against kmer6 to measure tokenization impact on RL training.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from post_training.ray.config import RayPostTrainingConfig

DATA_DIR = Path("/opt/dlami/nvme/data")
OUTPUT_ROOT = Path("/opt/dlami/nvme/output")

config = RayPostTrainingConfig(
    # Model — resume from GRPO checkpoint-1500 (reward improved 1.5 → 3.2)
    model_checkpoint=OUTPUT_ROOT / "grpo_baseline" / "checkpoint-1500",
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
    learning_rate=1e-5,
    max_steps=5000,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    weight_decay=0.01,
    warmup_steps=50,
    bf16=True,
    seed=42,

    # GRPO-specific
    kl_coef=0.1,
    cliprange=0.2,

    # Reward — sum of components
    reward_fn_name="motif_alignment",
    scoring_batch_size=8,
    eos_bonus=0.15,

    # Single GPU
    num_policy_gpus=1,

    # Output & logging
    output_dir=OUTPUT_ROOT / "grpo_baseline",
    save_steps=500,
    logging_steps=1,

    # W&B
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_baseline_char_resume1500",

    # MLflow
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
