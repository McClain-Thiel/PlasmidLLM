"""PPO post-training config for g6-big (L4 GPU, 22GB VRAM).

PPO uses a learned value function as baseline — doesn't need intra-group diversity.
This avoids the problem that killed GRPO/RLOO: near-identical DNA generations
from the same prompt (reward_std ≈ 0.004).

Memory budget (22GB):
  Policy model: ~68MB (17M params)
  Reference model: ~68MB (frozen copy)
  Value model: ~68MB (backbone) + tiny value head
  Generation KV cache + optimizer states: ~2-4GB
  Total: ~5-6GB → comfortable margin
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_ppo import PPORunConfig

config = PPORunConfig(
    # Data — same paths as GRPO/RLOO/RFT configs
    model_checkpoint=Path("/opt/dlami/nvme/output/pretrain_v4/checkpoint-15000"),
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    motif_lookup=Path("/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet"),

    # Training — conservative lr to avoid policy collapse
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch = 16
    max_steps=5000,

    # Generation — 1024 tokens, same as RFT (model never produces EOS)
    response_length=1024,
    temperature=1.0,
    local_rollout_forward_batch_size=8,  # generation batch size (no grad)

    # PPO algorithm
    num_ppo_epochs=4,         # inner optimization epochs per batch
    num_mini_batches=1,       # single minibatch (small model, fits in memory)
    kl_coef=0.05,             # light KL penalty — allow exploration
    cliprange=0.2,            # standard PPO clipping
    vf_coef=0.1,              # value loss weight (low — reward learning is secondary)
    cliprange_value=0.2,
    gamma=1.0,                # no discounting (we care about total reward, not temporal)
    lam=0.95,                 # GAE lambda
    whiten_rewards=False,     # don't normalize — our rewards are already bounded [0, ~0.5]

    # Output
    output_dir=Path("/opt/dlami/nvme/output/ppo_v1"),
    save_steps=500,
    logging_steps=1,
    seed=42,
    bf16=True,

    # MLflow — Databricks hosted tracking
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
