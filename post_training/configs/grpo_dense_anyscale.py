"""GRPO + MotifScorer on dense PlasmidLM-kmer6 — Anyscale job.

Run via Anyscale:
    anyscale job submit -f anyscale/job_config.yaml

Or locally:
    python -m post_training.runners.run post_training/configs/grpo_dense_anyscale.py
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

S3_PREFIX = os.environ.get(
    "S3_CHECKPOINT_PREFIX",
    "s3://anyscale-production-data-vm-us-east-1-f7164253"
    "/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm/checkpoints",
)

config = PostTrainingConfig(
    # Model — dense kmer6 from HuggingFace
    model="McClain/PlasmidLM-kmer6",
    bf16=True,
    num_actors=1,

    # Algorithm — GRPO
    algorithm="grpo",
    kl_coef=0.3,           # higher KL to prevent drift (learned from prior runs)
    cliprange=0.2,
    num_generations=4,
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
    learning_rate=5e-6,     # conservative LR (learned from prior runs)
    warmup_steps=50,
    max_grad_norm=1.0,
    checkpoint_every=100,
    checkpoint_dir="/tmp/checkpoints/grpo_dense_motif",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_dense_motif_anyscale",
)
