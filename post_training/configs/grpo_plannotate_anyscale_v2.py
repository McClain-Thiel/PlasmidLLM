"""GRPO + PlannotateScorer — fast iteration config.

Changes from v1:
- temperature 0.3→0.7 for more exploration
- num_actors 1→4 for parallel generation
- num_generations 8→4 (higher temp provides variance)
- kl_coef 0.5→0.3 (looser to accommodate higher temp)
- steps 1000→200 (signal check, not full run)
- checkpoint_every 50

    CONFIG=grpo_plannotate_anyscale_v2 python anyscale/run_anyscale.py
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

config = PostTrainingConfig(
    # Model — start from v1 plannotate checkpoint
    model="McClain/PlasmidLM-kmer6",
    bf16=True,
    num_actors=2,

    # Algorithm — GRPO (tuned for fast iteration)
    algorithm="grpo",
    kl_coef=0.3,
    cliprange=0.2,
    num_generations=4,
    micro_batch_size=8,

    # Scorer — plannotate BLAST
    scorer="plannotate",
    scorer_kwargs={
        "plannotate_db_path": str(DATA_DIR / "plannotate_db.parquet"),
        "motif_registry_path": str(DATA_DIR / "motif_registry_combined.parquet"),
    },

    # Generation — higher temp for exploration
    max_new_tokens=2500,
    temperature=0.7,
    top_p=0.95,

    # Prompts
    prompts_parquet=str(DATA_DIR / "training_pairs_v4.parquet"),
    prompt_batch_size=8,
    filter_hard_tokens=True,

    # Training — short run for signal check
    steps=200,
    learning_rate=5e-6,
    warmup_steps=20,
    max_grad_norm=1.0,
    checkpoint_every=50,
    checkpoint_dir="/tmp/checkpoints/grpo_plannotate_v2",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_plannotate_fast_v2",
)
