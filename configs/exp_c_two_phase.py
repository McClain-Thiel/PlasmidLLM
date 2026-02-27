"""Experiment C: Two-Phase Curriculum (presence plateau → gradual ramp).

Hypothesis: The current curriculum's problem is that it transitions too fast
from easy (α=0) to hard (α=1). A better approach:
  Phase 1 (steps 0-200): α=0.0 constant — learn structural correctness
  Phase 2 (steps 200-400): α ramps 0.0→0.7 — gradually introduce specificity

This gives the model 200 dedicated steps to consolidate structure learning
before any specificity pressure, then a gentle ramp to moderate specificity
(caps at 0.7, not 1.0, maintaining some presence reward).

Requires a modified CurriculumAlphaCallback — see TwoPhaseAlphaCallback below.
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
    total_episodes=6400,  # 400 steps × batch_size(4) × grad_accum(4)

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

    # Two-phase schedule:
    #   Phase 1 (steps 0-200): α = 0.0 constant (presence plateau)
    #   Phase 2 (steps 200-400): α ramps 0.0 → 0.7 over 200 steps
    alpha_start=0.0,
    alpha_end=0.7,
    alpha_plateau_steps=200,   # Hold α=0 for 200 steps
    alpha_warmup_steps=200,    # Then ramp to 0.7 over 200 steps

    # Output
    output_dir=Path("/opt/dlami/nvme/output/exp_c_two_phase"),
    save_steps=200,
    logging_steps=1,
    seed=42,
    bf16=True,

    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
