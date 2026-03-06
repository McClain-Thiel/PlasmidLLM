"""PPO training with MotifScorer on PlasmidLM.

PPO alternative: clipped surrogate + entropy bonus, multiple epochs per batch.

    python -m post_training.runners.run post_training/configs/ppo_motif.py
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
    algorithm="ppo",
    cliprange=0.2,
    entropy_coeff=0.01,
    ppo_epochs=4,
    micro_batch_size=32,

    # Scorer
    scorer="motif",
    scorer_kwargs={
        "motif_db_path": str(DATA_DIR / "motif_registry.parquet"),
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
    checkpoint_dir="checkpoints/ppo_motif",

    # Logging
    wandb_project="PlasmidLLM",
    wandb_run_name="ppo_motif_kmer6moe",
)
