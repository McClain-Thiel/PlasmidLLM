# GRPO Post-Training for PlasmidLM

Reinforcement learning fine-tuning using Group Relative Policy Optimization (GRPO) with sequence alignment rewards.

## Overview

**Goal**: Improve a pretrained PlasmidLM model to generate sequences that contain expected motifs in the correct locations.

**Method**: GRPO (simplified PPO) with Smith-Waterman alignment rewards

**Reward Function**: For each hard token in the prompt (e.g., `<AMR_KANAMYCIN>`), check if the generated sequence contains the expected motif using DNA and protein alignment. Reward = mean alignment score across all motifs.

## Quick Start

### 1. Prerequisites

```bash
# Install RL dependencies
pip install -e ".[rl]"

# This installs:
# - trl (HuggingFace TRL for GRPO)
# - vllm (fast inference)
# - parasail (sequence alignment)
# - biopython (DNA translation)
```

### 2. Prepare Data

You need:
- **training_pairs.parquet** - Same file used for pretraining (will auto-filter to `has_hard_tokens=True`)
- **motif_lookup.parquet** - Motif registry with columns:
  - `token`: e.g., `<AMR_KANAMYCIN>`
  - `dna_seq`: DNA sequence
  - `is_cds`: Boolean (True for coding sequences)
  - `seq_type`: "dna" or "protein"
- **Pretrained model checkpoint** - From pretraining step

### 3. Create Config

```python
# configs/my_grpo_run.py
from pathlib import Path
from plasmid_llm.config import PostTrainingConfig

config = PostTrainingConfig(
    # Data
    training_pairs=Path("data/training_pairs.parquet"),
    motif_lookup=Path("data/motif_lookup.parquet"),
    model_checkpoint=Path("output/pretraining/final"),
    
    # GRPO settings
    num_generations_per_prompt=16,  # Sample 16 completions per prompt
    learning_rate=1e-5,
    num_train_epochs=1,
    
    # Sampling
    max_new_tokens=8000,
    temperature=0.8,
    
    # MLflow
    mlflow_tracking_uri="http://localhost:5000",
)
```

### 4. Train

```bash
python scripts/train_grpo.py configs/my_grpo_run.py
```

## How It Works

### Reward Function

For a prompt like `<BOS><AMR_KANAMYCIN><ORI_COLE1><SEP>`:

1. **Parse hard tokens**: `[<AMR_KANAMYCIN>, <ORI_COLE1>]`
2. **For each token**:
   - Load expected sequences from motif lookup
   - Align generated sequence to expected sequence (both strands)
   - If CDS: also check protein alignment (6-frame translation)
   - Compute score_ratio = alignment_score / max_possible_score
3. **Final reward**: mean(score_ratios)

**Result**: Reward ∈ [0, 1]
- ~1.0 = all motifs found with high similarity
- ~0.5 = some motifs found or partial matches
- ~0.0 = no motifs found

### GRPO Algorithm

1. **Sample**: Generate N completions per prompt using current model
2. **Score**: Compute rewards using sequence alignment
3. **Group**: For each prompt, compute advantages relative to group mean
4. **Update**: Increase probability of high-reward completions
5. **Repeat**: Run for multiple epochs

Benefits over standard PPO:
- No separate critic model needed
- Uses group statistics for advantage estimation
- More stable and simpler

## Config Parameters

### Sampling

```python
num_generations_per_prompt: int = 16  # How many completions per prompt
max_new_tokens: int = 8000            # Max sequence length
temperature: float = 0.8              # Sampling temperature
top_k: int = 50                       # Top-k sampling
top_p: float = 0.95                   # Nucleus sampling
```

### GRPO Hyperparameters

```python
learning_rate: float = 1e-5       # Lower than pretraining
num_ppo_epochs: int = 4           # Update iterations per batch
kl_coef: float = 0.05             # KL penalty (prevent mode collapse)
clip_range: float = 0.2           # PPO clipping
vf_coef: float = 0.1              # Value function weight
```

### Tuning Guide

- **KL coefficient** (`kl_coef`): Start at 0.05
  - Too low → model collapses to single output
  - Too high → model doesn't improve
  
- **Generations per prompt**: 16-32 recommended
  - More = better advantage estimates but slower
  
- **Learning rate**: 1e-5 typical
  - Lower than pretraining to prevent forgetting
  
- **Temperature**: 0.8-1.0
  - Higher = more exploration
  - Lower = more exploitation of known good sequences

## Expected Behavior

### During Training

Monitor these metrics in MLflow:

- **mean_reward**: Should increase over time
  - Start: ~0.2-0.4 (random baseline)
  - Goal: ~0.7+ (most motifs found)

- **kl_divergence**: Should stay bounded
  - Too high → increase `kl_coef`
  - Zero → decrease `kl_coef`

- **loss**: PPO loss, should decrease

### Signs of Success

✅ Reward increases steadily
✅ KL divergence stays < 0.5
✅ Generated sequences contain expected motifs
✅ Model doesn't collapse (still generates diverse sequences)

### Warning Signs

⚠️ Reward plateaus quickly → increase `learning_rate` or `num_generations`
⚠️ KL divergence explodes → increase `kl_coef`
⚠️ All generations identical → model collapsed, decrease `kl_coef` or increase `temperature`

## Data Lineage

MLflow automatically logs:
- Hash of `training_pairs.parquet` (links to pretraining)
- Hash of `motif_lookup.parquet`
- Model checkpoint path
- All hyperparameters
- Git commit

This ensures full traceability: pretraining → post-training

## Testing

```bash
# Test reward function
pytest tests/test_reward.py

# Integration test with real data (if available)
pytest tests/test_reward.py -m integration
```

## Example Output

After training, you'll have:

```
output/grpo_run1/
├── final/                    # Improved model
│   ├── config.json
│   ├── model.safetensors
│   └── vocab.json
├── checkpoint-500/           # Intermediate checkpoints
├── checkpoint-1000/
└── runs/                     # TensorBoard logs
```

## Troubleshooting

### Out of Memory

- Reduce `per_device_train_batch_size`
- Reduce `num_generations_per_prompt`
- Reduce `max_new_tokens`
- Enable `gradient_checkpointing`

### Slow Training

- Use `use_vllm=True` for faster sampling
- Reduce `num_generations_per_prompt`
- Increase `per_device_train_batch_size` if memory allows

### Poor Rewards

- Check motif lookup contains expected tokens
- Verify sequences in lookup are correct
- Try lower `temperature` (less randomness)
- Increase `num_ppo_epochs` (more updates per batch)

## Advanced: Custom Reward Functions

The reward function can be customized in `post_training/reward.py`:

```python
def compute_reward(prompt, candidate_seq, lookup_df):
    # Your custom logic here
    # Return float in [0, 1]
    return reward
```

Options:
- Add GC content penalty/bonus
- Add length constraints
- Add spacing constraints between motifs
- Combine with other metrics (secondary structure, etc.)

## References

- GRPO: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- TRL: [Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- Smith-Waterman: [parasail library](https://github.com/jeffdaily/parasail)
