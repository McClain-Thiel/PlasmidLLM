#!/usr/bin/env bash
# Submit the GRPO post-training job to Anyscale.
# Sources secrets from ../.env and passes them as env vars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load secrets from .env
set -a
source "$PROJECT_DIR/.env"
set +a

cd "$PROJECT_DIR"

pyenv shell 3.9.8

anyscale job submit \
    --name grpo-dense-motif \
    --cloud helix-anyscale-cloud \
    --compute-config plasmid-grpo-gpu \
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
    --env "PYTHONPATH=." \
    --wait \
    -- python anyscale/run_anyscale.py
