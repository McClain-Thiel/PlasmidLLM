"""Experiment B: Pure Presence Reward (α = 0 throughout).

Hypothesis: Before teaching specificity, we need the model to reliably generate
functional DNA components of ANY type. This experiment tests how fast the model
improves when rewarded for structural correctness alone.

At α=0, the reward for "<ORI_COLE1>" is:
    1.0 × any_ori_score  (ColE1 specificity doesn't matter)

Baseline data shows:
  AMR/ELEM/REPORTER: 1.0 presence (already solved)
  PROM/TAG: 0.3-0.4 presence (room to improve)
  ORI: 0.075 presence (major gap)

This experiment measures the learning rate for structural components.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_ppo import PPORunConfig

config = PPORunConfig(
    # Data
    model_checkpoint=Path("/opt/dlami/nvme/output/pretrain_v4/checkpoint-15000"),
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    motif_lookup=Path("/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet"),

    # Training
    learning_rate=3e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=400,

    # Generation
    response_length=1024,
    temperature=1.0,
    local_rollout_forward_batch_size=8,

    # PPO algorithm
    num_ppo_epochs=2,
    num_mini_batches=1,
    kl_coef=0.5,
    cliprange=0.1,
    vf_coef=0.1,
    cliprange_value=0.2,
    gamma=1.0,
    lam=0.95,
    whiten_rewards=False,

    # KEY CHANGE: alpha stays at 0.0 (pure presence reward)
    alpha_start=0.0,
    alpha_end=0.0,
    alpha_warmup_steps=1,  # Irrelevant since start==end

    # Output
    output_dir=Path("/opt/dlami/nvme/output/exp_b_presence_only"),
    save_steps=200,
    logging_steps=1,
    seed=42,
    bf16=True,

    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
