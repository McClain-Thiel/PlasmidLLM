# Config-Based Training System

Simple, config-driven training for PlasmidLM with automatic data lineage tracking.

## Quick Start

### 1. Prepare Your Data

You need just 2-3 files:

**For Pretraining:**
- `training_pairs.parquet` - Column: `full_text` with `<BOS><tokens><SEP>SEQUENCE<EOS>`
- `special_tokens.txt` - List of special tokens (one per line)

**For Post-Training (RL):**
- `prompts.parquet` - Column: `prompt`, filtered to `has_hard_tokens=True`
- `motif_lookup.json` - Motif registry for reward calculation
- Reference to pretraining data (for lineage tracking)

### 2. Create a Config File

```python
# configs/my_run.py
from pathlib import Path
from plasmid_llm.config import PretrainingConfig

config = PretrainingConfig(
    # Data
    training_pairs=Path("data/training_pairs.parquet"),
    special_tokens=Path("data/special_tokens.txt"),
    
    # Model
    hidden_size=384,
    num_hidden_layers=10,
    
    # Training
    output_dir=Path("output/my_run"),
    max_steps=100_000,
    
    # MLflow
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment="plasmid_pretraining",
)
```

### 3. Train

```bash
python scripts/train_with_config.py configs/my_run.py
```

That's it! The tokenizer is built dynamically from `special_tokens.txt`, and everything is logged to MLflow automatically.

## What Gets Logged

Every training run automatically logs to MLflow:

- **Data Lineage**:
  - File paths
  - SHA256 hashes (first 16 chars)
  - Git commit
  
- **Config Parameters**: All hyperparameters from your config
  
- **Training Metrics**: Loss, accuracy, perplexity

- **Model Artifacts**: Final model + tokenizer

## Config Options

### PretrainingConfig

```python
@dataclass
class PretrainingConfig:
    # Data (required)
    training_pairs: Path
    special_tokens: Path
    
    # Model architecture (defaults shown)
    hidden_size: int = 384
    num_hidden_layers: int = 10
    num_attention_heads: int = 8
    intermediate_size: int = 1536
    max_seq_len: int = 4096
    
    # Training
    per_device_train_batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 100_000
    warmup_steps: int = 1000
    
    # MLflow
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "plasmid_pretraining"
```

### PostTrainingConfig

```python
@dataclass
class PostTrainingConfig:
    # Data (required)
    prompts: Path
    motif_lookup: Path
    pretraining_data_ref: Path  # Link back to pretraining
    model_checkpoint: Path
    
    # RL parameters (TBD)
    reward_weight: float = 1.0
    kl_penalty: float = 0.1
```

## Post-Training Lineage

Post-training configs automatically link back to pretraining data:

```python
# configs/post_training_run.py
from plasmid_llm.config import PostTrainingConfig

config = PostTrainingConfig(
    prompts=Path("data/prompts.parquet"),
    motif_lookup=Path("data/motif_lookup.json"),
    
    # Link back to pretraining data for lineage
    pretraining_data_ref=Path("data/training_pairs.parquet"),
    
    model_checkpoint=Path("output/pretraining/final"),
)
```

When you run post-training, MLflow logs:
- Hash of `pretraining_data_ref` → verify it's the same dataset
- Hash of `prompts` and `motif_lookup`
- All RL parameters

This ensures complete data lineage from pretraining → post-training.

## Test Data

For development/testing, add small test files to `data/test/`:

```
data/test/
├── special_tokens.txt         # ~20 tokens
├── training_pairs.parquet     # ~100 examples
├── prompts.parquet           # ~50 examples
└── motif_lookup.json         # Minimal registry
```

Tests automatically use these files. See `tests/test_tokenizer_integration.py`.

## Benefits

- ✅ **Simple**: One config file per run
- ✅ **Reproducible**: Automatic hash logging
- ✅ **Type-safe**: Python dataclasses with validation
- ✅ **Traceable**: Post-training links to pretraining data
- ✅ **Clean CLI**: `python train_with_config.py config.py`

## Migration from Old Script

Old way:
```bash
python scripts/train_hf.py \
  --data_path data.parquet \
  --vocab_path vocab.json \
  --hidden_size 384 \
  --num_hidden_layers 10 \
  # ... 20 more flags
```

New way:
```bash
# Put everything in a config file
python scripts/train_with_config.py configs/my_run.py
```

The old `train_hf.py` still works but is no longer maintained. Use `train_with_config.py` for new runs.
