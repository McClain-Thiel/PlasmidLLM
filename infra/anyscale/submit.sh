#!/usr/bin/env bash
# Submit the GRPO post-training job to Anyscale.
# Sources deployment settings from the repository .env file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Load secrets from .env
set -a
source "$PROJECT_DIR/.env"
set +a

: "${ANYSCALE_CLOUD:?Set ANYSCALE_CLOUD in .env}"
: "${ANYSCALE_COMPUTE_CONFIG:?Set ANYSCALE_COMPUTE_CONFIG in .env}"
: "${S3_DATA_PREFIX:?Set S3_DATA_PREFIX in .env}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env}"

cd "$PROJECT_DIR"

anyscale job submit \
    --name "${ANYSCALE_JOB_NAME:-grpo-plannotate}" \
    --cloud "$ANYSCALE_CLOUD" \
    --compute-config "$ANYSCALE_COMPUTE_CONFIG" \
    --image-uri "anyscale/ray:2.54.0-py312-cu128" \
    --requirements "$SCRIPT_DIR/requirements.txt" \
    --working-dir "$PROJECT_DIR" \
    --exclude "*.pyc" \
    --exclude "__pycache__" \
    --exclude ".venv" \
    --exclude ".git" \
    --exclude "notebooks" \
    --exclude "docs" \
    --exclude "*.egg-info" \
    --exclude "data/*.parquet" \
    --exclude "data/test" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "S3_DATA_PREFIX=$S3_DATA_PREFIX" \
    --env "S3_CHECKPOINT_PREFIX=${S3_CHECKPOINT_PREFIX:-$S3_DATA_PREFIX/checkpoints}" \
    --env "CONFIG=${CONFIG:-grpo_plannotate_anyscale}" \
    --env "PYTHONPATH=." \
    --wait \
    -- python infra/anyscale/run_anyscale.py
