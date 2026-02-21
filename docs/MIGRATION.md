# Data-Centric Training Migration

## Summary of Changes

This migration accomplishes two main goals:

1. **Unified Tokenizer**: Removed redundant `PlasmidTokenizer`, kept only HF-compatible `PlasmidLMTokenizer`
2. **Config-Based Training**: New simple system using Python dataclasses

---

## Part 1: Tokenizer Consolidation

### What Was Removed

- `src/plasmid_llm/tokenizer.py` - Custom tokenizer (~150 lines)

### What Was Kept

- `src/plasmid_llm/models/hf_plasmid_lm/tokenization_plasmid_lm.py` - HF-compatible tokenizer

### Migration Guide

```python
# Before:
from plasmid_llm.tokenizer import PlasmidTokenizer
tokenizer = PlasmidTokenizer("vocab.json")

# After:
from plasmid_llm.models.hf_plasmid_lm import PlasmidLMTokenizer
tokenizer = PlasmidLMTokenizer("vocab.json")
```

API is identical:
- `tokenizer.encode(text)` → `list[int]`
- `tokenizer.decode(ids)` → `str`
- `tokenizer.vocab_size`, `tokenizer.pad_token_id`, etc.

Plus new HF methods:
- `tokenizer.save_pretrained(path)`
- `tokenizer.from_pretrained(path)`

---

## Part 2: Config-Based Training (NEW)

### The Problem

Training with 20+ command-line flags is messy and hard to reproduce.

### The Solution

**Config files as Python dataclasses** - type-safe, validated, version-controlled.

### Quick Start

**1. Create config:**

```python
# configs/my_run.py
from pathlib import Path
from plasmid_llm.config import PretrainingConfig

config = PretrainingConfig(
    training_pairs=Path("data/training_pairs.parquet"),
    special_tokens=Path("data/special_tokens.txt"),
    hidden_size=384,
    max_steps=100_000,
    mlflow_tracking_uri="http://localhost:5000",
)
```

**2. Train:**

```bash
python scripts/train_with_config.py configs/my_run.py
```

That's it! Tokenizer is built automatically from `special_tokens.txt`.

### What You Need

**Pretraining (2 files):**
1. `training_pairs.parquet` - Column: `full_text` with `<BOS><tokens><SEP>SEQUENCE<EOS>`
2. `special_tokens.txt` - Special tokens list (one per line)

**Post-Training (3 files):**
1. `prompts.parquet` - Filtered to `has_hard_tokens=True`
2. `motif_lookup.json` - Motif registry for rewards
3. Reference to pretraining data (for lineage)

### Automatic Lineage Tracking

Every run logs to MLflow:
- File paths + SHA256 hashes
- Git commit
- All config parameters
- Training metrics
- Model artifacts

Post-training automatically links back to pretraining data via hash comparison.

### Benefits

- ✅ One config file (not 20 flags)
- ✅ Type-safe with validation
- ✅ Easy to version control
- ✅ Automatic lineage tracking
- ✅ Dynamic tokenizer generation

---

## Files Changed/Created

### Removed
- `src/plasmid_llm/tokenizer.py`
- `scripts/create_data_manifest.py`
- `src/plasmid_llm/data_manifest.py`

### Created
- `src/plasmid_llm/config.py` - Config dataclasses
- `scripts/train_with_config.py` - New training script
- `configs/pretraining_example.py` - Example config
- `tests/test_tokenizer_integration.py` - Integration tests
- `data/test/` - Test data directory
- `CONFIG_TRAINING.md` - Config system docs

### Updated
- `src/plasmid_llm/data.py` - Generic tokenizer support
- `scripts/generate.py` - Use PlasmidLMTokenizer
- `scripts/inference_sample.py` - Use PlasmidLMTokenizer
- `scripts/train_hf.py` - Simplified MLflow integration (still works, but prefer config-based)
- `tests/test_tokenizer.py` - Test PlasmidLMTokenizer
- `tests/test_data.py` - Use PlasmidLMTokenizer

---

## Next Steps

1. **Add test data**: Place small files in `data/test/` (see `data/test/README.md`)
2. **Create your config**: Copy `configs/pretraining_example.py`
3. **Run training**: `python scripts/train_with_config.py configs/your_config.py`

See `CONFIG_TRAINING.md` for full documentation.
