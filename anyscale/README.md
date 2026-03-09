# Anyscale Post-Training

Run GRPO post-training of PlasmidLM on Anyscale's managed Ray infrastructure with GPU autoscaling.

## Prerequisites

- Anyscale CLI installed (`pip install anyscale` — requires Python 3.9 for the CLI)
- Anyscale credentials at `~/.anyscale/credentials.json`
- wandb API key and HF token in `../.env` (see below)

## Setup

### 1. Secrets in `.env`

The project `.env` (gitignored) must contain:

```
WANDB_API_KEY=<your-wandb-api-key>
HF_TOKEN=<your-huggingface-token>
```

These are injected into the Anyscale job as environment variables by the submit script.

### 2. Data in S3

Training data lives in the Anyscale-managed S3 bucket. The job entrypoint downloads it automatically at startup.

```
s3://anyscale-production-data-vm-us-east-1-f7164253/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm/
├── data/
│   ├── motif_registry_combined.parquet   # Motif DB for MotifScorer
│   └── training_pairs_v4.parquet         # Prompts (108K plasmid prompts)
└── checkpoints/                          # Training checkpoints synced here
```

To re-upload data:
```bash
aws s3 cp data/motif_registry_combined.parquet \
  s3://anyscale-production-data-vm-us-east-1-f7164253/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm/data/
aws s3 cp data/training_pairs_v4.parquet \
  s3://anyscale-production-data-vm-us-east-1-f7164253/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm/data/
```

## Running

### Quick start

```bash
bash anyscale/submit.sh
```

This sources `.env`, uploads the working directory (excluding large data files, .git, .venv, etc.), and submits the job with `--wait` to stream logs.

### Manual submission

```bash
source .env
pyenv shell 3.9.8  # anyscale CLI needs this python version

anyscale job submit \
    --name grpo-dense-motif \
    --cloud helix-anyscale-cloud \
    --image-uri "anyscale/ray:2.54.0-py312-cu128" \
    --requirements anyscale/requirements.txt \
    --working-dir . \
    --exclude "*.pyc" --exclude "__pycache__" --exclude ".venv" \
    --exclude ".git" --exclude "notebooks" --exclude "docs" \
    --exclude "data/*.parquet" --exclude "data/test" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "PYTHONPATH=." \
    -- python anyscale/run_anyscale.py
```

Add `--wait` to block and stream logs in real time.

### Monitoring

```bash
# Check job status
pyenv shell 3.9.8
anyscale job status --id <JOB_ID>

# Stream logs
anyscale job logs --id <JOB_ID>

# List all jobs
anyscale job list
```

Training metrics are logged to wandb at https://wandb.ai/mcclain/PlasmidLLM.

## Architecture

```
submit.sh                  # Shell wrapper: loads .env, calls anyscale job submit
run_anyscale.py            # Job entrypoint: S3 data download → training → S3 checkpoint sync
requirements.txt           # Python deps installed on top of base Ray image
job_config.yaml            # Alternative YAML-based job config (not used by submit.sh)
```

### Flow

1. Anyscale provisions a `g5.2xlarge` head node (A10G GPU, 24GB VRAM)
2. Ray autoscaler spins up GPU worker nodes as needed
3. `run_anyscale.py` downloads data from S3 → local disk
4. Loads config from `post_training/configs/grpo_dense_anyscale.py`
5. Runs GRPO training loop with MotifScorer
6. Background thread syncs checkpoints to S3 every 5 minutes
7. Final checkpoint synced on completion

### Training config

See `post_training/configs/grpo_dense_anyscale.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `McClain/PlasmidLM-kmer6` | 19.3M param dense transformer |
| Algorithm | GRPO | Group Relative Policy Optimization |
| Scorer | MotifScorer (combined) | CIGAR-based motif alignment |
| Steps | 1000 | ~100 checkpoints at every=100 |
| LR | 5e-6 | Conservative to prevent drift |
| KL coef | 0.3 | Higher KL penalty (learned from prior runs) |
| Generations | 4 per prompt | GRPO group size |
| Batch size | 8 prompts | × 4 gen = 32 completions/step |

### Checkpoints

Checkpoints are saved locally to `/tmp/checkpoints/grpo_dense_motif/` and synced to:
```
s3://anyscale-production-data-vm-us-east-1-f7164253/.../plasmid_llm/checkpoints/grpo_dense_motif/
├── step_100/
├── step_200/
├── ...
└── final/
```

To download a checkpoint:
```bash
aws s3 sync \
  s3://anyscale-production-data-vm-us-east-1-f7164253/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm/checkpoints/grpo_dense_motif/final/ \
  ./checkpoints/grpo_dense_motif_anyscale/
```

## Scaling up

To use multiple GPU actors for parallel generation + gradient averaging, change `num_actors` in the config and adjust the compute config to add worker nodes:

```yaml
# In job submit or compute_config.yaml
compute_config:
  head_node:
    instance_type: g5.2xlarge
  worker_nodes:
    - instance_type: g5.xlarge
      min_nodes: 3
      max_nodes: 3
```

Then set `num_actors=4` in `grpo_dense_anyscale.py`. Each actor gets 1 GPU.

## Files

| File | Purpose |
|------|---------|
| `submit.sh` | Submit job with secrets from .env |
| `run_anyscale.py` | Job entrypoint (S3 download → train → S3 upload) |
| `requirements.txt` | Python dependencies |
| `job_config.yaml` | YAML job config (alternative to CLI flags) |
| `compute_config.yaml` | Compute config template |
