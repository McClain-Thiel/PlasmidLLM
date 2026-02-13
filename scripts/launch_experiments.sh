#!/bin/bash
# Launch experiments sequentially on single GPU.

set -e
cd /home/ubuntu/PlasmidLLM

# Load Databricks config from .databricks folder if available
if [ -f .databricks/connect_config ]; then
    export DATABRICKS_CONFIG_FILE=".databricks/connect_config"
fi

if [ -z "${DATABRICKS_TOKEN}" ]; then
    echo "ERROR: DATABRICKS_TOKEN env var is not set" >&2
    exit 1
fi

S3_PARQUET="/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/training_pairs.parquet"
S3_VOCAB="/mnt/s3/phd-research-storage-1758274488/addgene_clean/tokenization/token_vocabulary.json"
EXPERIMENT="/Users/mcclain.thiel@gmail.com/PlasmidLLM"

echo "============================================================"
echo "Experiment 1: Transformer 15M (d384 x10) @ 8k, lr=1e-4"
echo "============================================================"
.venv/bin/python scripts/train.py \
    model=transformer_15m \
    data.max_seq_len=8192 \
    data.parquet_path="$S3_PARQUET" \
    data.vocab_path="$S3_VOCAB" \
    train.batch_size=8 \
    train.lr=2e-4 \
    train.max_steps=50000 \
    train.checkpoint_dir=checkpoints/transformer_15m_8k \
    mlflow.experiment_name="$EXPERIMENT" \
    mlflow.run_name=transformer_15m_d384_seq8k_lr2e4

echo ""
echo "============================================================"
echo "Experiment 2: Mamba 15M (d320 x20) @ 8k, lr=1e-4"
echo "============================================================"
.venv/bin/python scripts/train.py \
    model=mamba_15m \
    data.max_seq_len=8192 \
    data.parquet_path="$S3_PARQUET" \
    data.vocab_path="$S3_VOCAB" \
    train.batch_size=8 \
    train.lr=2e-4 \
    train.max_steps=50000 \
    train.checkpoint_dir=checkpoints/mamba_15m_8k \
    mlflow.experiment_name="$EXPERIMENT" \
    mlflow.run_name=mamba_15m_d320_seq8k_lr2e4

echo ""
echo "============================================================"
echo "Experiment 3: Mamba Large 40M (d512 x24) @ 8k, lr=1e-4"
echo "============================================================"
.venv/bin/python scripts/train.py \
    model=mamba_large \
    data.max_seq_len=8192 \
    data.parquet_path="$S3_PARQUET" \
    data.vocab_path="$S3_VOCAB" \
    train.batch_size=4 \
    train.lr=2e-4 \
    train.max_steps=50000 \
    train.checkpoint_dir=checkpoints/mamba_large_8k \
    mlflow.experiment_name="$EXPERIMENT" \
    mlflow.run_name=mamba_large_d512_seq8k_lr2e4

echo ""
echo "All experiments complete!"
