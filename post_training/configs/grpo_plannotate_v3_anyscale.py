"""GRPO + PlannotateScorer v3 — 2000 steps with length penalty, 1x L40S.

Improvements over v1 (1000-step run):
- 2000 steps (v1 reward was still climbing when LR hit 0)
- Length penalty to prevent sequence length drift (v1 drifted 8k→9.5k)
- Longer warmup (100 steps vs 50)
- m5.4xlarge head node for faster BLAST scoring

    CONFIG=grpo_plannotate_v3_anyscale python anyscale/run_anyscale.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

config = PostTrainingConfig(
    # Model — overridden by RESUME_FROM_S3 in entrypoint
    model="McClain/PlasmidLM-kmer6",
    bf16=True,
    num_actors=1,

    # Algorithm — GRPO
    algorithm="grpo",
    kl_coef=0.5,
    cliprange=0.2,
    num_generations=8,
    micro_batch_size=8,

    # Scorer — plannotate BLAST (composite partial credit)
    scorer="plannotate",
    scorer_kwargs={
        "plannotate_db_path": str(DATA_DIR / "plannotate_db.parquet"),
        "motif_registry_path": str(DATA_DIR / "motif_registry_combined.parquet"),
        "norm_score_cap": 1.5,
    },

    # Length penalty — starts at 6k (very gentle), ramps up for bloated seqs
    # At 6k: 0, 7k: 0.02, 8k: 0.04, 9k: 0.06, 10k: 0.08, cap at 0.15
    length_penalty_per_kb=0.02,
    length_soft_max=6000.0,
    length_penalty_cap=0.15,

    # Generation
    max_new_tokens=2500,
    temperature=0.3,
    top_p=0.95,

    # Prompts — full training data with multi-token prompts
    prompts_parquet=str(DATA_DIR / "training_pairs_v4.parquet"),
    prompt_batch_size=8,
    filter_hard_tokens=True,

    # Eval — holdout prompts every 100 steps
    eval_parquet=str(DATA_DIR / "holdout_full_prompts.parquet"),
    eval_every=100,
    eval_batch_size=8,

    # Training — 2000 steps (v1 was 1000, reward still climbing)
    steps=2000,
    learning_rate=5e-6,
    warmup_steps=100,
    max_grad_norm=1.0,
    checkpoint_every=200,
    checkpoint_dir="/tmp/checkpoints/grpo_plannotate_v3",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_plannotate_v3",
)
