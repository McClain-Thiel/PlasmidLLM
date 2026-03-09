"""Quick smoke test: GRPO + MotifScorer + dense PlasmidLM-kmer6.

5 steps, 2 generations per prompt, small batch — just verify the loop runs.

    python -m post_training.runners.run post_training/configs/grpo_dense_smoke.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from post_training.config import PostTrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

config = PostTrainingConfig(
    model="McClain/PlasmidLM-kmer6",
    bf16=True,
    num_actors=1,

    algorithm="grpo",
    kl_coef=0.1,
    cliprange=0.2,
    num_generations=2,
    micro_batch_size=4,

    scorer="motif",
    scorer_kwargs={
        "motif_db_path": str(DATA_DIR / "motif_registry.parquet"),
    },

    max_new_tokens=2500,
    temperature=0.3,
    top_p=0.95,

    prompts_list=[
        "<BOS><AMR_KANAMYCIN><ORI_COLE1><SEP>",
        "<BOS><AMR_AMPICILLIN><ORI_COLE1><SEP>",
        "<BOS><AMR_KANAMYCIN><PROM_T7><SEP>",
        "<BOS><AMR_AMPICILLIN><PROM_CMV><SEP>",
    ],
    prompt_batch_size=4,

    steps=5,
    learning_rate=1e-5,
    warmup_steps=2,
    checkpoint_every=0,
    checkpoint_dir="/opt/dlami/nvme/checkpoints/grpo_dense_smoke",
)
