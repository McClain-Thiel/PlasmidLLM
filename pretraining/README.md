# pretraining

Causal language modeling on plasmid DNA sequences. Trains a transformer decoder from scratch to generate plasmid sequences conditioned on categorical tokens describing the desired functional elements.

## Quick start

```bash
# Smoke test (100 steps, verifies pipeline end-to-end)
python -m pretraining.train pretraining/configs/p4_smoke_test.py

# Full training run
python -m pretraining.train pretraining/configs/p4_kmer6_moe.py
```

## How it works

Each training example is a single string:

```
<BOS><AMR_KANAMYCIN><ORI_COLE1><PROM_T7><SEP>ATCGATCG...TACG<EOS>
```

The categorical tokens before `<SEP>` describe what the plasmid should contain (resistance genes, origins of replication, promoters, etc.). The DNA sequence after `<SEP>` is the actual plasmid. The model learns to generate the sequence conditioned on the token prompt via standard next-token prediction, with prompt tokens masked from the loss.

## Model

Transformer decoder (`plasmid_llm/models/hf_plasmid_lm/`) with:

- RoPE positional embeddings
- RMSNorm
- Optional Mixture-of-Experts (MoE) feed-forward layers
- HuggingFace-compatible — works with `transformers`, `vLLM`, `trl`

Typical configurations range from ~7M params (baseline dense) to ~46M total / ~23M active (k-mer + MoE).

## Tokenization

Two tokenizer modes, selected via `tokenizer_type` in the config:

| Mode | Description | Effective context |
|------|-------------|-------------------|
| `char` | One token per DNA base (A/T/C/G/N) + special tokens | 4096 bp at `max_seq_len=4096` |
| `kmer` | Non-overlapping or strided k-mers (e.g. 6-mer stride 3) + special tokens | ~12K bp at `max_seq_len=4096` with k=6, stride=3 |

Both modes share the same special token vocabulary. The tokenizer is built dynamically from `special_tokens.txt` at training time, and saved into the output directory as `vocab.json` for reproducibility.

## Configuration

Training is fully config-driven using Python dataclasses (`plasmid_llm.config.PretrainingConfig`). No command-line flags — everything lives in a config file:

```python
from plasmid_llm.config import PretrainingConfig
from pathlib import Path

config = PretrainingConfig(
    # Data
    training_pairs=Path("data/training_pairs_v4.parquet"),
    special_tokens=Path("data/special_tokens.txt"),

    # Architecture
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,

    # Tokenizer
    tokenizer_type="kmer",
    kmer_k=6,
    kmer_stride=3,

    # MoE (optional)
    use_moe=True,
    num_experts=6,
    num_experts_per_tok=2,
    moe_intermediate_size=1536,
    aux_loss_coef=0.01,

    # Training
    output_dir=Path("output/my_run"),
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    max_steps=100_000,
    warmup_steps=1000,
    weight_decay=0.1,

    # Tracking
    wandb_project="PlasmidLLM",
)
```

See `pretraining/configs/` for ready-to-use examples covering baseline, k-mer, MoE, and combined configurations.

### Key config fields

| Field | Default | Description |
|-------|---------|-------------|
| `training_pairs` | — | Parquet file with `full_text` column |
| `special_tokens` | — | Text file listing all special tokens, one per line |
| `tokenizer_type` | `"char"` | `"char"` or `"kmer"` |
| `kmer_k` / `kmer_stride` | 6 / 3 | K-mer size and stride (only used when `tokenizer_type="kmer"`) |
| `use_moe` | `False` | Enable Mixture-of-Experts |
| `num_experts` / `num_experts_per_tok` | 6 / 2 | MoE routing parameters |
| `bf16` | `False` | BFloat16 mixed precision |
| `gradient_checkpointing` | `False` | Trade compute for memory |
| `early_stopping_patience` | 10 | Eval steps without improvement before stopping |
| `wandb_project` | `None` | Set to enable W&B logging |
| `mlflow_tracking_uri` | `None` | Set to enable MLflow logging |

## Input data

**Required files:**

1. **`training_pairs.parquet`** — Parquet file with a `full_text` column containing formatted sequences (`<BOS><tokens><SEP>SEQUENCE<EOS>`). Alternative column names `prompt`/`sequence` or `token_prompt`/`token_completion` are also supported (the dataset class handles both).

2. **`special_tokens.txt`** — One token per line, defining the full vocabulary of categorical tokens plus control tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<SEP>`, `<UNK>`).

**Sample data** is included in `data/` for testing. Full production data lives in S3 — see `data/README.md`.

## Experiment tracking

Supports both MLflow and W&B. Each run automatically logs:

- Input file paths + SHA256 hashes (for data lineage)
- Full model architecture and hyperparameters
- Training metrics (loss, token accuracy, perplexity)
- Git commit hash
- Checkpoints as artifacts

## Output

The training script saves:

- Checkpoints at `save_steps` intervals (keeps last 5)
- Final model to `{output_dir}/final/` with model weights, config, tokenizer, and vocab
- The final checkpoint is fully self-contained and loadable with `AutoModelForCausalLM.from_pretrained()`

## Configs

| Config | Description |
|--------|-------------|
| `p4_smoke_test.py` | 100-step pipeline test |
| `p4_baseline.py` | Dense char-level baseline |
| `p4_kmer6_s3.py` | 6-mer tokenizer, stride 3, dense FFN |
| `p4_kmer6_s4.py` | 6-mer tokenizer, stride 4, dense FFN |
| `p4_moe_half.py` | Char-level + MoE (half-size experts) |
| `p4_moe_full.py` | Char-level + MoE (full-size experts) |
| `p4_kmer6_moe.py` | 6-mer + MoE (best-of-both-worlds) |
