"""GRPO training with AlignmentScorer on PlasmidLM.

Uses Smith-Waterman score-ratio reward. Prompts from training data.

    python -m post_training.runners.run post_training/configs/grpo_alignment.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

config = PostTrainingConfig(
    # Model
    model="McClain/PlasmidLM-kmer6-MoE",
    bf16=True,
    num_actors=1,

    # Algorithm
    algorithm="grpo",
    kl_coef=0.1,
    cliprange=0.2,
    num_generations=8,
    micro_batch_size=32,

    # Scorer — Smith-Waterman alignment
    scorer="alignment",
    scorer_kwargs={
        "motif_lookup_path": str(DATA_DIR / "motif_registry.parquet"),
        "eos_bonus": 0.15,
    },

    # Generation
    max_new_tokens=1024,
    temperature=1.0,
    top_p=0.95,

    # Prompts
    prompts_parquet=str(DATA_DIR / "training_pairs_v4.parquet"),
    prompt_batch_size=8,
    filter_hard_tokens=True,

    # Training
    steps=500,
    learning_rate=1e-5,
    warmup_steps=50,
    checkpoint_every=100,
    checkpoint_dir="checkpoints/grpo_alignment",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_alignment_kmer6moe",
)
