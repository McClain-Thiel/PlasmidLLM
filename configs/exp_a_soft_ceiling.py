"""Experiment A: Soft Alpha Ceiling (α max = 0.5).

Hypothesis: α=1.0 (exact match only) is too hard — most tokens score near 0,
giving no RL signal. By capping α at 0.5, we maintain partial credit from
category presence (any valid component of the right type) while still pushing
toward specificity.

Comparison: Current curriculum ramps α 0→1 over 1000 steps then stays at 1.0.
At α=0.5, the reward for "<ORI_COLE1>" is:
    0.5 × exact_cole1_score + 0.5 × any_ori_score

This should provide denser signal than α=1.0 where most scores are 0.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_ppo import PPORunConfig

config = PPORunConfig(
    # Data — same as main PPO config
    model_checkpoint=Path("/opt/dlami/nvme/output/pretrain_v4/checkpoint-15000"),
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    motif_lookup=Path("/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet"),

    # Training — same conservative settings
    learning_rate=3e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=400,  # Short experiment

    # Generation
    response_length=1024,
    temperature=1.0,
    local_rollout_forward_batch_size=8,

    # PPO algorithm — same as main
    num_ppo_epochs=2,
    num_mini_batches=1,
    kl_coef=0.5,
    cliprange=0.1,
    vf_coef=0.1,
    cliprange_value=0.2,
    gamma=1.0,
    lam=0.95,
    whiten_rewards=False,

    # KEY CHANGE: alpha caps at 0.5 (never goes to full exact match)
    alpha_start=0.0,
    alpha_end=0.5,
    alpha_warmup_steps=400,  # Linear ramp over entire experiment

    # Output
    output_dir=Path("/opt/dlami/nvme/output/exp_a_soft_ceiling"),
    save_steps=200,
    logging_steps=1,
    seed=42,
    bf16=True,

    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
