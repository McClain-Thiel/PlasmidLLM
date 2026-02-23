"""Example GRPO post-training configuration for PlasmidLM."""

from pathlib import Path
from plasmid_llm.config import PostTrainingConfig

# Create GRPO configuration
config = PostTrainingConfig(
    # Data (update these paths to your actual data)
    training_pairs=Path("data/training_pairs.parquet"),  # Same as pretraining
    motif_lookup=Path("data/motif_lookup.parquet"),      # From build_motif_registry.py
    model_checkpoint=Path("output/pretraining/final"),    # Pretrained model

    # GRPO hyperparameters
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,

    # Sampling (how many completions to generate per prompt)
    num_generations=16,          # 4-16 recommended
    max_completion_length=8000,
    temperature=0.8,
    top_k=50,
    top_p=0.95,

    # GRPO-specific tuning (TRL 0.16+ naming)
    num_iterations=1,            # inner optimization epochs per batch
    beta=0.05,                   # KL divergence penalty (prevent mode collapse)
    epsilon=0.2,                 # PPO clipping range
    loss_type="grpo",

    # Output
    output_dir=Path("output/grpo_run1"),
    save_steps=500,
    logging_steps=10,
    eval_steps=100,

    # System
    bf16=True,
    use_vllm=False,
    seed=42,

    # MLflow
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment="plasmid_grpo",
)
