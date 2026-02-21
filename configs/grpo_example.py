"""Example GRPO post-training configuration for PlasmidLM."""

from pathlib import Path
from plasmid_llm.config import PostTrainingConfig

# Create GRPO configuration
config = PostTrainingConfig(
    # Data (update these paths to your actual data)
    training_pairs=Path("data/training_pairs.parquet"),  # Same as pretraining
    motif_lookup=Path("data/motif_lookup.parquet"),      # From your notebook
    model_checkpoint=Path("output/pretraining/final"),    # Pretrained model
    
    # GRPO hyperparameters
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    
    # Sampling (how many completions to generate per prompt)
    num_generations_per_prompt=16,  # 16-32 recommended
    max_new_tokens=8000,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    
    # GRPO-specific tuning
    num_ppo_epochs=4,     # How many times to update on same batch
    kl_coef=0.05,         # KL divergence penalty (prevent mode collapse)
    clip_range=0.2,       # PPO clipping range
    vf_coef=0.1,          # Value function coefficient
    
    # Output
    output_dir=Path("output/grpo_run1"),
    save_steps=500,
    logging_steps=10,
    eval_steps=100,
    
    # System
    bf16=True,
    use_vllm=True,  # Use vLLM for fast sampling
    seed=42,
    
    # MLflow
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment="plasmid_grpo",
)
