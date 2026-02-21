# PlasmidLM

Language models for synthetic plasmid DNA sequence generation with reinforcement learning.

## Overview

PlasmidLM is a transformer-based language model that learns to generate plasmid DNA sequences containing specific functional elements (promoters, origins of replication, antibiotic resistance genes, etc.). The system supports:

- **Pretraining**: Causal language modeling on plasmid sequences with categorical tokens
- **Post-training**: Reinforcement learning (GRPO) using sequence alignment rewards
- **Generation**: Fast inference with vLLM for producing novel plasmid designs

All training uses Python configuration files and automatic MLflow experiment tracking.

## Quick Start

```bash
# Install
pip install -e .

# For RL training
pip install -e ".[rl]"

# Pretrain a model
python scripts/train_with_config.py configs/pretraining_example.py

# Fine-tune with GRPO
python scripts/train_grpo.py configs/grpo_example.py

# Generate sequences
python scripts/generate.py --hf-model output/final --prompt "<BOS><AMR_KANAMYCIN><SEP>"
```

## Architecture

### Model

Transformer decoder with:
- RoPE positional embeddings
- RMSNorm
- Character-level tokenization (DNA bases + categorical tokens)
- Compatible with HuggingFace `transformers` and `vLLM`

### Tokenization

Dynamic tokenizer built from a list of special tokens:
- Categorical tokens: `<AMR_KANAMYCIN>`, `<ORI_COLE1>`, `<PROM_T7>`, etc.
- DNA bases: `A`, `T`, `C`, `G`, `N` (case-insensitive)
- Control tokens: `<BOS>`, `<EOS>`, `<SEP>`, `<PAD>`

Format: `<BOS><token1><token2>...<SEP>ATCGATCG...<EOS>`

### Reward Function (Post-Training)

Uses Smith-Waterman sequence alignment to verify generated sequences contain expected motifs:
- DNA alignment (forward + reverse complement)
- Protein alignment (6-frame translation for CDS)
- Score per motif averaged across all prompted elements
- Returns rewards ∈ [0, 1]

## Training Pipeline

### Stage 1: Pretraining

Causal language modeling on plasmid sequences.

**Input data**:
- `training_pairs.parquet`: Full sequences with format `<BOS><tokens><SEP>SEQUENCE<EOS>`
- `special_tokens.txt`: List of categorical tokens (one per line)

**Config**:
```python
from plasmid_llm.config import PretrainingConfig

config = PretrainingConfig(
    training_pairs="data/training_pairs.parquet",
    special_tokens="data/special_tokens.txt",
    hidden_size=384,
    num_hidden_layers=10,
    max_steps=100_000,
)
```

**Run**: `python scripts/train_with_config.py configs/my_pretrain.py`

### Stage 2: Post-Training (GRPO)

Reinforcement learning to improve motif placement and sequence quality.

**Additional data**:
- `motif_lookup.parquet`: Maps tokens to canonical sequences (for reward calculation)

**Config**:
```python
from plasmid_llm.config import PostTrainingConfig

config = PostTrainingConfig(
    training_pairs="data/training_pairs.parquet",  # Same as pretraining
    motif_lookup="data/motif_lookup.parquet",
    model_checkpoint="output/pretraining/final",
    num_generations_per_prompt=16,
    learning_rate=1e-5,
)
```

**Run**: `python scripts/train_grpo.py configs/my_grpo.py`

## Project Structure

```
PlasmidLLM/
├── scripts/
│   ├── build_motif_registry.py      # Create motif lookup from database
│   ├── train_with_config.py         # Pretraining script
│   ├── train_grpo.py                # GRPO post-training script
│   ├── train_hf.py                  # Legacy training (still works)
│   ├── generate.py                  # vLLM generation
│   └── inference_sample.py          # Batch generation + eval
├── src/plasmid_llm/
│   ├── config.py                    # Config dataclasses
│   ├── data.py                      # PyTorch datasets
│   ├── models/hf_plasmid_lm/        # HuggingFace model implementation
│   └── utils/                       # Helpers
├── post_training/
│   ├── reward.py                    # Sequence alignment reward function
│   └── README.md                    # GRPO guide + tuning tips
├── configs/                         # Example configurations
│   ├── pretraining_example.py
│   └── grpo_example.py
├── tests/                           # Unit and integration tests
└── data/                            # Data directory
    └── test/                        # Test data (user-provided)
```

## Key Features

**Config-driven training**: Python dataclasses instead of command-line flags
```python
config = PretrainingConfig(...)  # Type-safe, validated, version-controlled
```

**Automatic lineage tracking**: Every run logs to MLflow:
- Input file paths + SHA256 hashes
- Model hyperparameters
- Training metrics
- Git commit
- For post-training: links to pretraining data

**HuggingFace compatible**: Works with:
- `transformers.AutoModelForCausalLM`
- `transformers.Trainer`
- `vLLM` (fast inference)
- `trl.GRPOTrainer` (RL)

**Fast RL training**: Uses vLLM for rapid sequence generation during GRPO

## Data Format

### Data Location

**Sample data** (included in repo):
- `data/training_pairs_sample.parquet` - 1000 plasmids for testing/development
- `data/motif_registry.parquet` - 360 motifs for reward computation

**Full dataset** (production):
- Location: `s3://phd-research-storage-1758274488/databricks_export/`
- Files:
  - `training_pairs.parquet` - Full training dataset
  - `motif_lookup.parquet` - Complete motif registry

For production training, download from S3:
```bash
# Example (requires AWS credentials)
aws s3 cp s3://phd-research-storage-1758274488/databricks_export/training_pairs.parquet data/
aws s3 cp s3://phd-research-storage-1758274488/databricks_export/motif_lookup.parquet data/
```

### Training Pairs

Parquet file with column:
- `full_text`: `<BOS><AMR_KANAMYCIN><ORI_COLE1><SEP>ATCGATCG...TACG<EOS>`

Optional columns:
- `has_hard_tokens`: Boolean (filtered automatically in post-training)
- `token_prompt`: Pre-split prompts (alternative format)

### Motif Lookup

Parquet file mapping tokens to sequences:

| token | dna_seq | is_cds | seq_type | sseqid |
|-------|---------|--------|----------|--------|
| `<AMR_KANAMYCIN>` | ATGACC... | True | dna | NC_001234.1 |
| `<ORI_COLE1>` | TTGACA... | False | dna | pBR322 |

See `scripts/build_motif_registry.py` (queries plannotate database).

### Special Tokens

Text file, one token per line:
```
<PAD>
<BOS>
<EOS>
<SEP>
<UNK>
<AMR_KANAMYCIN>
<AMR_AMPICILLIN>
<ORI_COLE1>
...
```

## Project Structure

```
PlasmidLLM/
├── README.md                    # Main documentation
├── pyproject.toml              # Dependencies & package config
│
├── configs/                    # Training configurations
│   ├── pretraining_example.py # Example pretraining config
│   └── grpo_example.py         # Example GRPO config
│
├── docs/                       # Documentation
│   ├── CONFIG_TRAINING.md      # Pretraining guide
│   ├── GRPO_GUIDE.md           # GRPO post-training guide
│   ├── MIGRATION.md            # Recent changes
│   └── PROJECT_STRUCTURE.md    # Detailed structure
│
├── scripts/                    # Executable scripts
│   ├── train_with_config.py   # ✨ Pretraining (recommended)
│   ├── train_grpo.py           # ✨ GRPO post-training
│   ├── generate.py             # vLLM generation
│   ├── inference_sample.py     # Batch generation
│   └── build_motif_registry.py # Build motif database
│
├── src/plasmid_llm/           # Core library
│   ├── config.py              # Config dataclasses
│   ├── data.py                # PyTorch datasets
│   └── models/hf_plasmid_lm/  # HF-compatible model
│
├── post_training/             # RL components
│   └── reward.py              # Alignment reward function
│
├── tests/                     # Tests
│   ├── test_tokenizer_integration.py
│   └── test_reward.py
│
└── data/                      # Data directory
    ├── training_pairs_sample.parquet  # 1000 sample plasmids for testing
    ├── motif_registry.parquet         # Motif lookup (360 motifs)
    ├── motif_registry.json            # JSON version of motif registry
    ├── token_vocabulary_v3.json       # ⭐ Primary token vocabulary (250 tokens)
    ├── token_vocabulary.json          # Legacy vocabulary (101 tokens)
    ├── special_tokens.txt             # Token list for dynamic tokenizer (250 tokens from v3)
    └── README.md                      # Data documentation
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed descriptions.

## MLflow Integration

All experiments automatically track:

**Pretraining**:
- Data hashes (training_pairs, special_tokens)
- Model architecture (hidden_size, num_layers, etc.)
- Training metrics (loss, accuracy, perplexity)
- Git commit

**Post-training**:
- Links to pretraining (data hash verification)
- Motif lookup hash
- GRPO hyperparameters (kl_coef, num_generations, etc.)
- Reward metrics (mean, std, per-motif scores)

View: `mlflow ui --backend-store-uri <uri>`

## Testing

```bash
# All tests
pytest

# Tokenizer tests (requires test data in data/test/)
pytest tests/test_tokenizer_integration.py

# Reward function tests
pytest tests/test_reward.py

# Integration tests
pytest tests/ -m integration
```

## Documentation

- **[docs/CONFIG_TRAINING.md](docs/CONFIG_TRAINING.md)** - Pretraining guide
- **[docs/GRPO_GUIDE.md](docs/GRPO_GUIDE.md)** - GRPO post-training guide with hyperparameter tuning
- **[docs/MIGRATION.md](docs/MIGRATION.md)** - Recent changes and migration guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation notes

## Dependencies

**Core**:
- PyTorch 2.1+
- Transformers 4.30+
- PyArrow (parquet)
- MLflow 2.10+

**RL (optional)**:
- TRL 0.8+ (GRPO)
- vLLM 0.4+ (fast sampling)
- parasail 1.3+ (sequence alignment)
- Biopython 1.80+ (DNA tools)

See `pyproject.toml` for full list.

## Citation

If you use this code, please cite:

```bibtex
[Citation to be added]
```
