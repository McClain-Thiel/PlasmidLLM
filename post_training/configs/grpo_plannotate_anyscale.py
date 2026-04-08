"""GRPO + PlannotateScorer on dense PlasmidLM-kmer6 — Anyscale.

Uses BLAST-based plannotate scoring instead of motif alignment.
Requires NCBI BLAST+ tools (installed via apt in entrypoint).

    CONFIG=grpo_plannotate_anyscale python infra/anyscale/run_anyscale.py
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

config = PostTrainingConfig(
    # Model — dense kmer6, start from v2 motif checkpoint
    model="McClain/PlasmidLM-kmer6",
    bf16=True,
    num_actors=1,

    # Algorithm — GRPO (same tuned hyperparams as v2)
    algorithm="grpo",
    kl_coef=0.5,
    cliprange=0.2,
    num_generations=8,
    micro_batch_size=8,

    # Scorer — plannotate BLAST
    scorer="plannotate",
    scorer_kwargs={
        "plannotate_db_path": str(DATA_DIR / "plannotate_db.parquet"),
        "motif_registry_path": str(DATA_DIR / "motif_registry_combined.parquet"),
    },

    # Generation
    max_new_tokens=2500,
    temperature=0.3,
    top_p=0.95,

    # Prompts — full training data
    prompts_parquet=str(DATA_DIR / "training_pairs_v4.parquet"),
    prompt_batch_size=8,
    filter_hard_tokens=True,

    # Training
    steps=1000,
    learning_rate=5e-6,
    warmup_steps=50,
    max_grad_norm=1.0,
    checkpoint_every=100,
    checkpoint_dir="/tmp/checkpoints/grpo_plannotate",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_plannotate_anyscale",
)
