"""GRPO + MotifScorer on dense PlasmidLM-kmer6 — Anyscale v2.

Tuned from v1 results:
- num_generations 4→8 for better advantage estimation
- kl_coef 0.3→0.5 to prevent KL drift
- Resume from v1 checkpoint (downloaded from S3 by entrypoint)

    anyscale job submit -f anyscale/job_config.yaml
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

config = PostTrainingConfig(
    # Model — dense kmer6 from HuggingFace
    model="McClain/PlasmidLM-kmer6",
    bf16=True,
    num_actors=1,

    # Algorithm — GRPO (tuned)
    algorithm="grpo",
    kl_coef=0.5,           # tighter KL (was 0.3)
    cliprange=0.2,
    num_generations=8,      # doubled for better advantage estimates (was 4)
    micro_batch_size=8,

    # Scorer — motif alignment (combined registry)
    scorer="motif",
    scorer_kwargs={
        "motif_db_path": str(DATA_DIR / "motif_registry_combined.parquet"),
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
    checkpoint_dir="/tmp/checkpoints/grpo_dense_motif_v2",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_dense_motif_anyscale_v2",
)
