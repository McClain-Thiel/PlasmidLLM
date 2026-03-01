"""Myriad HPC GRPO config (1x A100, 32 CPU cores via SGE).

Paths read from $PLASMID_DATA_DIR (set by job script to $TMPDIR).
Output written under $HOME (persists after job ends).
MLflow tracks to Databricks — credentials loaded from ~/.databricks_env.
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from post_training.ray.config import RayPostTrainingConfig

data_dir = Path(os.environ["PLASMID_DATA_DIR"])
home = Path(os.environ["HOME"])

config = RayPostTrainingConfig(
    # Data — staged to $TMPDIR by job script
    model_checkpoint=data_dir / "checkpoint-15000",
    training_pairs=data_dir / "training_pairs_v4.parquet",
    motif_lookup=data_dir / "motif_registry.parquet",

    # Algorithm — GRPO with 4 rollouts per prompt
    algorithm="grpo",
    num_generations_per_prompt=4,

    # Ray resources — 1 GPU for policy, ~30 CPUs for scoring
    num_cpu_workers=30,
    gpu_memory_fraction=0.9,

    # Generation — 16 unique prompts × 4 = 64 sequences/step (A100 has 40-80GB)
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
    warmup_steps=200,
    bf16=True,
    seed=42,

    # GRPO-specific
    kl_coef=0.05,
    cliprange=0.2,

    # Reward — sum of components, no curriculum
    reward_fn_name="motif_alignment",
    scoring_batch_size=8,
    eos_bonus=0.15,

    # Output & logging — $HOME persists after job ends
    output_dir=home / "PlasmidLLM" / "output" / "grpo_myriad_v1",
    save_steps=500,
    logging_steps=1,

    # MLflow — Databricks hosted tracking (credentials from ~/.databricks_env)
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
