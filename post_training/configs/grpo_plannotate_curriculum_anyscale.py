"""GRPO + PlannotateScorer — resume from binary motif curriculum checkpoint.

The model has learned single-token motif placement via binary reward. Now
switch to plannotate scorer (composite, partial credit) on full multi-token
prompts to refine overall plasmid quality.

    CONFIG=grpo_plannotate_curriculum_anyscale python anyscale/run_anyscale.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

config = PostTrainingConfig(
    # Model — overridden by RESUME_FROM_S3 in entrypoint
    model="<org>/PlasmidLM-kmer6",
    bf16=True,
    num_actors=1,

    # Algorithm — GRPO, moderate KL
    algorithm="grpo",
    kl_coef=0.5,              # moderate -- allow adaptation to composite reward
    cliprange=0.2,
    num_generations=8,
    micro_batch_size=8,

    # Scorer — plannotate (composite partial credit)
    scorer="plannotate",
    scorer_kwargs={
        "plannotate_db_path": str(DATA_DIR / "plannotate_db.parquet"),
        "motif_registry_path": str(DATA_DIR / "motif_registry_combined.parquet"),
    },

    # Generation
    max_new_tokens=2500,
    temperature=0.3,
    top_p=0.95,

    # Prompts — full training data with multi-token prompts
    prompts_parquet=str(DATA_DIR / "training_pairs_v4.parquet"),
    prompt_batch_size=8,
    filter_hard_tokens=True,

    # Training
    steps=2000,
    learning_rate=3e-6,
    warmup_steps=50,
    max_grad_norm=1.0,
    checkpoint_every=200,
    checkpoint_dir="/tmp/checkpoints/grpo_plannotate_curriculum",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_plannotate_from_binary_motif",
)
