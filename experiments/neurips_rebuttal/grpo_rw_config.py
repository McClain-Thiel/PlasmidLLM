"""Continue GRPO from the released post-GRPO checkpoint on novel-combination-augmented
rollouts, to test the F16 fix: does training on novel component combinations teach the model
to produce the requested substitute instead of reverting to the training co-occurrence?

Only change vs v3: the rollout prompt distribution (50% novel combinations), the starting
checkpoint (the released post-GRPO model, not the base), a shorter schedule, and the eval set
(the novel benchmark, so the F16 metric is watched during training). Reward is unchanged.

    CONFIG rebuttal/grpo_data  python -m post_training.runners.run grpo_novel_aug.py
"""

from pathlib import Path

from post_training.config import PostTrainingConfig

D = Path("/opt/dlami/nvme/rebuttal")
GD = D / "grpo_data"

config = PostTrainingConfig(
    model="McClain/PlasmidLM-kmer6-GRPO-plannotate",  # continue from post-GRPO
    bf16=True,
    num_actors=1,

    algorithm="grpo",
    kl_coef=0.5,
    cliprange=0.2,
    num_generations=8,
    micro_batch_size=8,

    scorer="plannotate",
    scorer_kwargs={
        "plannotate_db_path": str(GD / "plannotate_db.parquet"),
        "motif_registry_path": str(GD / "motif_registry_combined.parquet"),
        "token_weights_path": str(GD / "token_weights.json"),
    },

    length_penalty_per_kb=0.02,
    length_soft_max=6000.0,
    length_penalty_cap=0.15,

    max_new_tokens=2500,
    temperature=0.3,
    top_p=0.95,

    # rollout on the novel-combination-augmented set
    prompts_parquet=str(D / "rollout_producible.parquet"),
    prompt_batch_size=8,
    filter_hard_tokens=True,

    # watch the F16 metric during training
    eval_parquet=str(D / "novel_producible_benchmark.parquet"),
    eval_every=50,
    eval_batch_size=4,

    steps=250,
    learning_rate=5e-6,
    warmup_steps=30,
    max_grad_norm=1.0,
    checkpoint_every=100,
    checkpoint_dir=str(D / "grpo_rw_ckpt"),
    s3_checkpoint_prefix="s3://phd-research-storage-1758274488/neurips_rebuttal/grpo_rw_ckpt",

    wandb_project="PlasmidLLM",
    wandb_run_name="grpo_rw_weighted",
)
