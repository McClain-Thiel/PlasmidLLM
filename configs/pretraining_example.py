"""Example pretraining configuration for PlasmidLM."""

from pathlib import Path
from plasmid_llm.config import PretrainingConfig

# Create configuration
config = PretrainingConfig(
    # Data (update these paths to your actual data)
    training_pairs=Path("data/training_pairs.parquet"),
    special_tokens=Path("data/special_tokens.txt"),
    
    # Model architecture
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,
    
    # Training
    output_dir=Path("output/pretraining_run1"),
    per_device_train_batch_size=32,
    learning_rate=3e-4,
    max_steps=100_000,
    warmup_steps=1000,
    
    # System
    bf16=True,
    gradient_checkpointing=False,
    
    # MLflow
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment="plasmid_pretraining",
)
