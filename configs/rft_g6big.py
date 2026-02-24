"""RFT (Rejection Sampling Fine-Tuning) config for g6-big (L4 GPU, 22GB VRAM).

Expert Iteration approach:
  1. Generate completions → score with motif alignment → keep top completions → SFT
  2. Repeat for N iterations

GRPO failed because the DNA model generates near-identical sequences within a group
(reward_std < 0.01), giving GRPO zero advantage signal. RFT avoids this by filtering
across prompts — some prompts naturally produce high-reward completions, others don't.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_rft import RFTConfig

config = RFTConfig(
    # Data — same paths as GRPO config
    model_checkpoint=Path("/opt/dlami/nvme/output/pretrain_v4/checkpoint-15000"),
    training_pairs=Path("/mnt/s3/phd-research-storage-1758274488/databricks_export/training_pairs_v4.parquet"),
    motif_lookup=Path("/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/motif_registry.parquet"),

    # RFT iterations
    num_rft_iterations=5,
    reward_threshold=0.15,        # keep completions above this (~top 50% based on GRPO data)

    # Generation — one completion per prompt, 4096 tokens to match pretrain context
    gen_batch_size=16,            # larger batches for generation efficiency
    max_completion_length=4096,
    temperature=1.0,
    top_p=0.95,
    num_samples_per_prompt=1,     # 1 = fast first pass; bump to 2-4 for better coverage
    max_prompts_per_iter=5000,    # 5000 prompts × 4096 tokens ≈ manageable generation time

    # SFT training — conservative to avoid forgetting
    learning_rate=5e-5,
    sft_epochs=1,                 # single pass through filtered data
    sft_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch = 16
    max_seq_length=4096,

    # Output
    output_dir=Path("/opt/dlami/nvme/output/rft_v1"),
    save_steps=200,
    logging_steps=10,
    seed=42,
    bf16=True,

    # MLflow — Databricks hosted tracking
    mlflow_tracking_uri="databricks",
    mlflow_experiment="/Users/mcclain.thiel@gmail.com/tracking/PlasmidLLM",
)
