"""Quick 20-step smoke test — verifies the post-training pipeline end-to-end.

Uses gpt2 + substring scorer + GRPO. No GPU required (runs on CPU).

    python -m post_training.runners.run post_training/configs/smoke_test.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

config = PostTrainingConfig(
    # Model — small, CPU-friendly
    model="gpt2",
    bf16=False,

    # Algorithm
    algorithm="grpo",
    kl_coef=0.1,
    cliprange=0.2,
    num_generations=4,
    micro_batch_size=32,

    # Scorer — toy substring scorer (no data dependencies)
    scorer="substring",
    scorer_kwargs={"targets": ["the", "is", "and"]},

    # Generation
    max_new_tokens=32,

    # Prompts — inline list
    prompts_list=[
        "Once upon a time",
        "The experiment showed",
        "In a world where",
        "The data revealed",
    ],

    # Training — minimal
    steps=20,
    learning_rate=1e-5,
    checkpoint_every=10,
    checkpoint_dir="checkpoints/smoke_test",
)
