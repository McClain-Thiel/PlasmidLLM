"""Matched v1/v2 ablation. Change only PLASMID_REWARD_VERSION within each seed pair.

All runs start from the same pinned base, use the same training/validation prompts,
and use 4,096 generated k-mer tokens (~12 kb at stride 3), not 4,096 base pairs.
Set PLASMID_RUN_ID to a new experiment ID; repeated attempts need a new ID.
"""

import os
from pathlib import Path

from post_training.config import PostTrainingConfig

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
version = os.environ["PLASMID_REWARD_VERSION"]
if version not in {"v1", "v2", "f1", "f1_orf"}:
    raise ValueError("Reward version must be v1, f1, f1_orf, or v2")
seed = int(os.environ.get("PLASMID_SEED", "42"))
run_name = f"{os.environ['PLASMID_RUN_ID']}_{version}_s{seed}"
model = os.environ["PLASMID_BASE_MODEL"]
revision = os.environ.get("PLASMID_BASE_REVISION")
if not (model.startswith("s3://") or Path(model).is_dir()) and not revision:
    raise ValueError("Pin PLASMID_BASE_REVISION to a Hugging Face commit SHA")

scorer_kwargs = {"motif_registry_path": str(DATA / "motif_registry_combined.parquet")}
if version == "v1":
    scorer_kwargs.update(plannotate_db_path=str(DATA / "plannotate_db.parquet"), norm_score_cap=5.0)
else:
    scorer_kwargs.update(f1_weight=0.6, orf_weight=0.0 if version == "f1" else 0.3,
                         tag_weight=0.1 if version == "v2" else 0.0)

config = PostTrainingConfig(
    model=model, model_revision=revision, seed=seed, bf16=True, num_actors=1,
    algorithm="grpo", kl_coef=0.5, cliprange=0.2, num_generations=8, micro_batch_size=4,
    scorer="plannotate" if version == "v1" else "plannotate_v2", scorer_kwargs=scorer_kwargs,
    max_new_tokens=4096, temperature=0.3, top_p=0.95,
    prompts_parquet=os.environ["PLASMID_TRAIN_PROMPTS"],
    eval_parquet=os.environ["PLASMID_VALIDATION_PROMPTS"],
    prompt_batch_size=8, eval_batch_size=8, eval_every=100,
    steps=int(os.environ.get("PLASMID_STEPS", "1000")), learning_rate=5e-6,
    warmup_steps=50, max_grad_norm=1.0,
    checkpoint_every=int(os.environ.get("PLASMID_CHECKPOINT_EVERY", "100")),
    checkpoint_dir=str(Path("checkpoints") / run_name),
    s3_checkpoint_prefix=f"{os.environ['S3_CHECKPOINT_PREFIX'].rstrip('/')}/{run_name}",
    wandb_project="PlasmidLLM", wandb_run_name=run_name,
    wandb_tags=["plasmidrl", "ablation", "reward-v2", version, f"seed-{seed}"],
)
