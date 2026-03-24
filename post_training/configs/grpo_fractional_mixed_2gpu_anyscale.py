"""GRPO + FractionalMotifScorer — mixed curriculum (1/2/3/5/8-tok + full), 2x L40S.

Trains on all difficulty levels simultaneously with fractional reward.
Includes full multi-token prompts in training data.

    CONFIG=grpo_fractional_mixed_2gpu_anyscale python anyscale/run_anyscale.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

config = PostTrainingConfig(
    # Model — overridden by RESUME_FROM_S3 in entrypoint
    model="McClain/PlasmidLM-kmer6",
    bf16=True,
    num_actors=2,

    # Algorithm — GRPO, moderate KL (allow exploration)
    algorithm="grpo",
    kl_coef=0.5,
    cliprange=0.2,
    num_generations=8,
    micro_batch_size=16,

    # Scorer — fractional motif (partial credit)
    scorer="fractional_motif",
    scorer_kwargs={
        "plannotate_db_path": str(DATA_DIR / "plannotate_db.parquet"),
        "motif_registry_path": str(DATA_DIR / "motif_registry_combined.parquet"),
        "min_pct_id": 95.0,
        "min_coverage": 80.0,
    },

    # Generation
    max_new_tokens=2500,
    temperature=0.2,
    top_p=0.90,

    # Prompts — mixed curriculum (1/2/3/5/8-tok + full)
    prompts_parquet=str(DATA_DIR / "training_pairs_mixed_curriculum.parquet"),
    prompt_batch_size=16,
    filter_hard_tokens=True,

    # Training — 3000 steps
    steps=3000,
    learning_rate=5e-6,
    warmup_steps=100,
    max_grad_norm=1.0,
    checkpoint_every=200,
    checkpoint_dir="/tmp/checkpoints/grpo_fractional_mixed",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_fractional_mixed_2gpu",
)
