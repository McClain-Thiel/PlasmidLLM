"""GRPO + BinaryMotifScorer on dense PlasmidLM-kmer6 — Anyscale.

Single-token curriculum: each prompt has one hard token, binary 0/1 reward.
Uses BLAST-based scoring (requires NCBI BLAST+ tools).

    CONFIG=grpo_binary_motif_anyscale python anyscale/run_anyscale.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

config = PostTrainingConfig(
    # Model — dense kmer6
    model="McClain/PlasmidLM-kmer6",
    bf16=True,
    num_actors=1,

    # Algorithm — GRPO
    algorithm="grpo",
    kl_coef=0.3,
    cliprange=0.2,
    num_generations=8,        # critical for binary reward variance
    micro_batch_size=8,

    # Scorer — binary motif (BLAST-based, hard 0/1)
    scorer="binary_motif",
    scorer_kwargs={
        "plannotate_db_path": str(DATA_DIR / "plannotate_db.parquet"),
        "motif_registry_path": str(DATA_DIR / "motif_registry_combined.parquet"),
        "min_pct_id": 95.0,
        "min_coverage": 80.0,
    },

    # Generation — low temperature for BLAST precision
    max_new_tokens=2500,
    temperature=0.3,
    top_p=0.95,

    # Prompts — single-token curriculum data
    prompts_parquet=str(DATA_DIR / "training_pairs_binary_motif_1tok.parquet"),
    prompt_batch_size=8,
    filter_hard_tokens=True,

    # Training
    steps=1000,
    learning_rate=5e-6,
    warmup_steps=50,
    max_grad_norm=1.0,
    checkpoint_every=100,
    checkpoint_dir="/tmp/checkpoints/grpo_binary_motif",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_binary_motif_1tok",
)
