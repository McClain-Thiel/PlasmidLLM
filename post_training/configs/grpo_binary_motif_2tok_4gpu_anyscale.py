"""GRPO + BinaryMotifScorer — 2-token curriculum stage, 4x L40S.

Resumes from 1-tok v2 final checkpoint. 4 actors for parallel generation.

    CONFIG=grpo_binary_motif_2tok_4gpu_anyscale python anyscale/run_anyscale.py
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
    num_actors=4,

    # Algorithm — GRPO, tight KL (stay close to 1-tok checkpoint)
    algorithm="grpo",
    kl_coef=1.0,
    cliprange=0.15,
    num_generations=8,
    micro_batch_size=16,

    # Scorer — binary motif (BLAST-based, hard 0/1)
    scorer="binary_motif",
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

    # Prompts — 2-token curriculum data
    prompts_parquet=str(DATA_DIR / "training_pairs_binary_motif_2tok.parquet"),
    prompt_batch_size=32,
    filter_hard_tokens=True,

    # Training — 600 steps, early stopping via checkpoints
    steps=600,
    learning_rate=3e-6,
    warmup_steps=30,
    max_grad_norm=1.0,
    checkpoint_every=100,
    checkpoint_dir="/tmp/checkpoints/grpo_binary_motif_2tok",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_binary_motif_2tok_4gpu_v3",
)
