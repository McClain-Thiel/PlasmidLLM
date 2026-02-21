# PlasmidLLM Project Structure

## Directory Layout

```
PlasmidLLM/
├── README.md                    # Main documentation
├── pyproject.toml               # Python package config & dependencies
│
├── configs/                     # Training configurations
│   ├── pretraining_example.py  # Example pretraining config
│   └── grpo_example.py          # Example GRPO config
│
├── docs/                        # Documentation
│   ├── CONFIG_TRAINING.md       # Pretraining guide
│   ├── GRPO_GUIDE.md            # GRPO post-training guide
│   ├── MIGRATION.md             # Recent changes
│   └── IMPLEMENTATION_SUMMARY.md # Technical notes
│
├── scripts/                     # Executable scripts
│   ├── train_with_config.py    # ✨ Main pretraining script (config-based)
│   ├── train_grpo.py            # ✨ GRPO post-training script
│   ├── train_hf.py              # Legacy training (still works)
│   ├── generate.py              # vLLM generation
│   ├── inference_sample.py      # Batch generation + eval
│   └── build_motif_registry.py  # Build motif lookup database
│
├── src/plasmid_llm/             # Core library
│   ├── __init__.py
│   ├── config.py                # Config dataclasses (PretrainingConfig, PostTrainingConfig)
│   ├── data.py                  # PyTorch datasets (PlasmidDataset, PlasmidPromptsDataset)
│   ├── models/                  # Model implementations
│   │   └── hf_plasmid_lm/       # HuggingFace-compatible model
│   │       ├── __init__.py
│   │       ├── configuration_plasmid_lm.py
│   │       ├── modeling_plasmid_lm.py     # Transformer architecture
│   │       └── tokenization_plasmid_lm.py # Character-level tokenizer
│   └── utils/                   # Helper utilities
│
├── post_training/               # RL components
│   └── reward.py                # Sequence alignment reward function
│
├── tests/                       # Tests
│   ├── __init__.py
│   ├── test_tokenizer_integration.py
│   └── test_reward.py
│
└── data/                        # Data directory (user-provided)
    └── test/                    # Test data
        └── README.md            # Test data specification
```

## Key Files

### Configuration

- **`pyproject.toml`**: Package dependencies and metadata
- **`configs/*.py`**: Training configurations (Python dataclasses)

### Training Scripts

- **`scripts/train_with_config.py`**: Main pretraining script (recommended)
- **`scripts/train_grpo.py`**: GRPO post-training with RL
- **`scripts/train_hf.py`**: Legacy training (command-line args)

### Core Library

- **`src/plasmid_llm/config.py`**: Type-safe config classes
- **`src/plasmid_llm/data.py`**: Dataset loaders
- **`src/plasmid_llm/models/hf_plasmid_lm/`**: HF-compatible model

### Post-Training

- **`post_training/reward.py`**: Smith-Waterman alignment reward function

### Documentation

- **`docs/CONFIG_TRAINING.md`**: How to pretrain
- **`docs/GRPO_GUIDE.md`**: How to fine-tune with GRPO
- **`docs/MIGRATION.md`**: Recent changes

## Typical Workflow

1. **Prepare data** → `data/training_pairs.parquet`, `data/special_tokens.txt`, `data/motif_lookup.parquet`
2. **Create config** → `configs/my_pretrain.py`
3. **Pretrain** → `python scripts/train_with_config.py configs/my_pretrain.py`
4. **Create GRPO config** → `configs/my_grpo.py`
5. **Post-train** → `python scripts/train_grpo.py configs/my_grpo.py`
6. **Generate** → `python scripts/generate.py --hf-model output/final --prompt "..."`

## Notes

- **Legacy code**: `train_hf.py` still works but uses old command-line approach
- **Test data**: Add files to `data/test/` for integration tests
- **Model format**: HuggingFace-compatible (works with `transformers`, `vLLM`, `trl`)
