"""GRPO + BinaryMotifScorer v2 — resume from 1tok checkpoint, tighter KL.

Continues single-token curriculum from v1 final checkpoint.
Lower LR, tighter KL clamp, lower temperature for exploitation.

    CONFIG=grpo_binary_motif_anyscale_v2 python anyscale/run_anyscale.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

config = PostTrainingConfig(
    # Model — will be overridden by RESUME_FROM_S3 in entrypoint
    model="McClain/PlasmidLM-kmer6",
    bf16=True,
    num_actors=1,

    # Algorithm — GRPO, tight KL
    algorithm="grpo",
    kl_coef=1.0,             # tight clamp — stay close to v1 checkpoint
    cliprange=0.15,           # tighter clipping for stability
    num_generations=8,        # proven stable on L40S
    micro_batch_size=8,

    # Scorer — binary motif (BLAST-based, hard 0/1)
    scorer="binary_motif",
    scorer_kwargs={
        "plannotate_db_path": str(DATA_DIR / "plannotate_db.parquet"),
        "motif_registry_path": str(DATA_DIR / "motif_registry_combined.parquet"),
        "min_pct_id": 95.0,
        "min_coverage": 80.0,
    },

    # Generation — low temperature for exploitation
    max_new_tokens=2500,
    temperature=0.2,
    top_p=0.90,

    # Prompts — same single-token curriculum data
    prompts_parquet=str(DATA_DIR / "training_pairs_binary_motif_1tok.parquet"),
    prompt_batch_size=8,      # match v1 (proven stable)
    filter_hard_tokens=True,

    # Training — 2000 steps, lower LR for refinement
    steps=2000,
    learning_rate=3e-6,
    warmup_steps=50,
    max_grad_norm=1.0,
    checkpoint_every=200,
    checkpoint_dir="/tmp/checkpoints/grpo_binary_motif_v2",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_binary_motif_1tok_v2",
)
