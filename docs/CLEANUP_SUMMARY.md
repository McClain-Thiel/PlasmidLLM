# Repository Cleanup Summary

## Changes Made

### Documentation Reorganization

**Moved to `docs/`**:
- `CONFIG_TRAINING.md` → `docs/CONFIG_TRAINING.md`
- `MIGRATION.md` → `docs/MIGRATION.md`  
- `IMPLEMENTATION_SUMMARY.md` → `docs/IMPLEMENTATION_SUMMARY.md`
- `post_training/README.md` → `docs/GRPO_GUIDE.md`

**Created**:
- `docs/PROJECT_STRUCTURE.md` - Detailed project structure guide

**Result**: Root directory is now cleaner with only `README.md` and `pyproject.toml`

### Scripts Cleanup

**Removed**:
- `scripts/export_databricks_to_s3.py` - Utility script not part of core workflow

**Marked as legacy**:
- `scripts/train_hf.py` - Still works but marked as legacy (use `train_with_config.py` instead)

**Active scripts** (7 total):
- `train_with_config.py` - Main pretraining (config-based)
- `train_grpo.py` - GRPO post-training
- `generate.py` - vLLM generation
- `inference_sample.py` - Batch generation + evaluation
- `build_motif_registry.py` - Build motif lookup database
- `train_hf.py` - Legacy training (command-line args)

### Directory Structure

**Before**:
```
PlasmidLLM/
├── README.md
├── CONFIG_TRAINING.md
├── MIGRATION.md
├── IMPLEMENTATION_SUMMARY.md
├── pyproject.toml
├── scripts/ (8 files)
├── post_training/
│   ├── reward.py
│   └── README.md
└── ...
```

**After**:
```
PlasmidLLM/
├── README.md
├── pyproject.toml
├── docs/                       # ← All docs here
│   ├── CONFIG_TRAINING.md
│   ├── GRPO_GUIDE.md
│   ├── MIGRATION.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   └── PROJECT_STRUCTURE.md
├── scripts/ (7 files)         # ← Cleaned up
├── post_training/
│   └── reward.py              # ← Just code now
└── ...
```

## Benefits

1. **Cleaner root**: Only essential files at top level
2. **Centralized docs**: All documentation in `docs/`
3. **Clear separation**: Code vs documentation
4. **Easier navigation**: Logical grouping
5. **Simpler onboarding**: README → docs/ for details

## What Wasn't Changed

- Core library (`src/plasmid_llm/`) - No changes
- Tests - No changes
- Configs - No changes
- Model implementations - No changes
- All functionality preserved

## Updated References

All internal links updated:
- `README.md` now points to `docs/*`
- Legacy markers added where appropriate
- PROJECT_STRUCTURE.md provides detailed layout

## Next Steps for Users

If you had local bookmarks or scripts referencing old paths:
- Update doc links: `CONFIG_TRAINING.md` → `docs/CONFIG_TRAINING.md`
- Update doc links: `post_training/README.md` → `docs/GRPO_GUIDE.md`
- No code changes needed - all imports/scripts work the same
