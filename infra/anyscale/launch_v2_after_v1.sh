#!/usr/bin/env bash
# Poll v1 job until done, then launch v2 with tuned hyperparams.
set -euo pipefail

V1_JOB_ID="prodjob_2vvh8l6hlby6b8m35wrc3exwsp"
V1_S3_CKPT="s3://anyscale-production-data-vm-us-east-1-f7164253/org_wlxw8le5gjzi3dwtlu8qik7ngu/plasmid_llm/checkpoints/grpo_dense_motif/final/"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load secrets
set -a
source "$PROJECT_DIR/.env"
set +a

cd "$PROJECT_DIR"

echo "Waiting for v1 job $V1_JOB_ID to finish..."

while true; do
    STATE=$(PYENV_VERSION=3.9.8 anyscale job status --id "$V1_JOB_ID" 2>&1 | grep "^state:" | awk '{print $2}')
    echo "$(date): v1 state=$STATE"
    if [ "$STATE" = "SUCCEEDED" ] || [ "$STATE" = "FAILED" ] || [ "$STATE" = "STOPPED" ]; then
        break
    fi
    sleep 300
done

echo "v1 finished with state=$STATE. Launching v2..."

PYENV_VERSION=3.9.8 anyscale job submit \
    --name grpo-dense-motif-v2 \
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
    --exclude "wandb" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "PYTHONPATH=." \
    --env "CONFIG=grpo_dense_anyscale_v2" \
    --env "RESUME_FROM_S3=$V1_S3_CKPT" \
    -- python infra/anyscale/run_anyscale.py

echo "v2 submitted!"
