"""GRPO post-training for kmer6+MoE model on p4 (2x A100-40GB, GPUs 6-7).

Sum-of-components reward: motif_alignment scores each genetic component
independently, model is rewarded for correctly placing all requested motifs.

2-GPU setup: each GPU generates 64 sequences (8 prompts x 8 rollouts),
128 total per step. Both train on the merged batch to stay in sync.

Starts from checkpoint-5000 of kmer6_moe_full pretraining run.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from post_training.ray.config import RayPostTrainingConfig

DATA_DIR = Path("/opt/dlami/nvme/data")
OUTPUT_ROOT = Path("/opt/dlami/nvme/output")

config = RayPostTrainingConfig(
    # Model — kmer6+MoE pretrained checkpoint
    model_checkpoint=OUTPUT_ROOT / "kmer6_moe_full" / "checkpoint-10000",
    training_pairs=DATA_DIR / "training_pairs_v4.parquet",
    motif_lookup=DATA_DIR / "motif_registry.parquet",

    # Algorithm — GRPO with 8 rollouts per prompt
    algorithm="grpo",
    num_generations_per_prompt=8,

    # Ray resources — 2 GPUs for policy, ~30 CPUs for scoring
    num_cpu_workers=30,
    gpu_memory_fraction=0.9,

    # Generation — 16 unique prompts split across 2 GPUs (8 each)
    # 8 prompts x 8 rollouts = 64 sequences per GPU, 128 total/step
    generation_batch_size=16,
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

    # Reward — sum of components, no curriculum
    reward_fn_name="motif_alignment",
    scoring_batch_size=8,
    eos_bonus=0.15,

    # Multi-GPU — parallel generation across 2 A100s
    num_policy_gpus=2,

    # Output & logging
    output_dir=OUTPUT_ROOT / "grpo_kmer6_moe",
    save_steps=500,
    logging_steps=1,

    # W&B
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_kmer6_moe_ckpt10k_pertoken",

    # MLflow — Databricks (disabled on p4, no creds)
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
